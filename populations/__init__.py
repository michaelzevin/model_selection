import sys
import os
import pickle
import itertools
import copy
import pdb

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

# Need to ensure all parameters are normalized over the same range
param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1)}

"""
Set of classes used to construct statistical models of populations.
"""

class Model(object):
    """
    Base model class. Mostly used to root the inheritance tree.
    """
    def __init__(self):
        pass

    def __call__(self, data):
        return None

class KDEModel(Model):
    @staticmethod
    def from_samples(label, samples, params, weighting=False, Nsamps=None, **kwargs):
        """
        Generate a KDE model instance from samples, where params are series in the samples dataframe. Additional kwargs are passed to nothing at the moment. If weighting is true, will weight the samples used to generate the KDE according to the weights provided in samples.
        """
        if weighting:
            if "weight" not in samples.columns:
                raise ValueError("No weights were specified for your samples!")
            # only use samples that have weights greater than 0
            samples = samples.loc[samples["weight"] > 0]

        # downsample so the KDE construction doesn't take forever
        if Nsamps:
            samples = samples.sample(int(Nsamps))

        kde_samples = samples[params]
        weights = samples["weight"] if weighting else None

        return KDEModel(label, kde_samples, weights)


    def __init__(self, label, samples, weights=None):
        super()
        self.label = label
        self._samples = samples
        self._weights = weights

        # Normalize data s.t. they all are on the unit cube
        self._norm_params = [param_bounds[param] for param in samples.columns]
        samples = normalize_samples(np.asarray(samples), self._norm_params)

        _kde = KernelDensity(kernel='gaussian', bandwidth=0.01, rtol=1e-8)
        _kde.fit(samples, sample_weight=weights)
        self._kde = _kde

        # account for unnormalized inputs (sklearn returns logp)
        # also need to scale pdf by parameter range
        pdf_scale = scale_to_unity(self._norm_params)
        self._logpdf = lambda x: _kde.score_samples(normalize_samples(x, self._norm_params)) - np.log(pdf_scale)
        self._pdf = lambda x: np.exp(_kde.score_samples(normalize_samples(x, self._norm_params)))/pdf_scale

        # keep bounds of the samples
        self._bin_edges = []
        for dmin, dmax in zip(samples.min(axis=0), samples.max(axis=0)):
            self._bin_edges.append(np.linspace(dmin, dmax, 100))
        self._bin_edges = np.asarray(self._bin_edges)

        self._cached_values = None

    def sample(self, N=1, random_state=None):
        """
        Samples KDE and denormalizes sampled data
        """
        samps_norm = self._kde.sample(n_samples=N, random_state=random_state)
        samps =  denormalize_samples(samps_norm, self._norm_params)
        return samps

    def bin_centers(self):
        """
        Return the center points of the bins. Note that this returns the Cartesian product of the bin_edges, which are the linear "axes" of the parameter dimensions. This allows for N-D plotting without holding N^{param} sets of points to evaluate.
        """
        edges = [(be[1:] + be[:-1]) / 2 for be in self._bin_edges]
        return np.asarray(list(itertools.product(*edges)))

    def freeze(self, data, data_pdf=None):
        """
        Caches the values of the model PDF at the data points provided. This is useful to construct the hierarchal model likelihood since p_hyperparam(data) is evaluated many times, but only needs to be once because it's a fixed value, dependent only on the observations
        """
        self._cached_values = None
        self._cached_values = self(data, data_pdf)

    def __call__(self, data, data_pdf=None):
        """
        The expectation is that "data" is a n_obs x n_sample x n_dim array. If data_pdf is None, each observation is expected to have equal posterior probability. Otherwise, the posterior values should be provided as the same dimensions of the samples.
        """
        if self._cached_values is not None:
            return self._cached_values

        prob = np.ones(data.shape[0]) * 1e-20
        for i, obs in enumerate(np.atleast_3d(data)):
            # FIXME: Incorrect normalization on "ones"?
            # THIS IS WHERE THE POSTERIOR SAMPLES ARE TURNED INTO PROBABILITY
            d_pdf = data_pdf[i] if data_pdf is not None else 1
            prob[i] += np.sum(self._pdf(obs) / d_pdf)
        return prob

    def extent(self):
        bc = np.squeeze(self.bin_centers())
        bmin = bc.min(axis=0)
        bmax = bc.max(axis=0)
        return np.asarray([bmin, bmax])

    def plot(self, ax=None, **kwargs):
        """
        If a matplotlib.axes.Axes object is provided, plot a graph of the pdf on the axes. Otherwise, return the information needed to do so.
        """
        dim = kwargs.pop("dim") if "dim" in kwargs else None

        if dim is not None:
            newkde = self.marginalize_to(dim)
            return newkde.plot(ax, **kwargs)

        _eval_pts = self.bin_centers()
        self._bin_heights = self._pdf(_eval_pts)
        if ax is not None:
            ax.plot(np.squeeze(_eval_pts), self._bin_heights, **kwargs)
        return _eval_pts, self._bin_heights

    def marginalize(self, params):
        """
        Generate a new, lower dimensional, KDEModel from the parameters in [params]
        """
        return KDEModel(self._samples[params], self._weights)


class AdditiveModel(Model):

    @staticmethod
    def from_additivemodel(other_model, new_fracs):
        """
        Create a new AdditiveModel from old one, but with new relative fractions.
        """
        new_model = copy.copy(other_model)
        new_model.rel_fracs = {}

        assert len(other_model) == len(new_fracs)
        for i, b in zip(other_model, new_fracs):
            new_model.rel_fracs[i] = b

        new_model.freeze()
        return new_model

    def __init__(self):
        super(AdditiveModel, self).__init__()
        self.rel_fracs = {}

    def __len__(self):
        return len(self.rel_fracs)

    def __iter__(self):
        for submdl in sorted(self.rel_fracs.keys(), key=lambda mdl: mdl.label):
            yield submdl

    def marginalize_to(self, dim):
        """
        Generate a new, lower dimensional, KDEModel from the dimension indexed by 'dim'.
        """
        new_model = copy.copy(self)
        new_model.rel_fracs = {}

        for i, b in self.rel_fracs.items():
            new_model.rel_fracs[i.marginalize_to(dim)] = b

        # Invalidate the caches, we have a "new" model
        for mdl in new_model:
            mdl._cached_values = None

        new_model.freeze()
        return new_model

    def append_model(self, mdl, rel):
        """
        Add a model with given relative abundance
        """
        self.rel_fracs[mdl] = rel

    def freeze(self):
        """
        Normalize relative fractions to unity.
        """
        norm = float(sum(self.rel_fracs.values()))
        for mdl in self.rel_fracs:
            self.rel_fracs[mdl] /= norm

    def freeze_dist(self, data, data_pdf=None):
        """
        Call `Model.freeze` with data on all constituent distributions. Note that this does *not* freeze the relative fractions (see `AdditiveModel.freeze`).
        """
        for mdl in self.rel_fracs:
            mdl.freeze(data, data_pdf)

    def __call__(self, data):
        self.freeze()
        pdf = 0.
        for mdl in self:
            pdf += mdl(data) * self.rel_fracs[mdl]
        return pdf

    def __repr__(self):
        #return " ".join(["%s %.3f" % (k.label, v) for k, v in self.rel_fracs.iteritems()])
        return " ".join(["%s %.3f" % (mdl.label, self.rel_fracs[mdl]) for mdl in self])

    def plot(self, ax=None, **kwargs):
        """
        If a matplotlib.axes.Axes object is provided, plot a graph of the pdf on the axes. Otherwise, return the information needed to do so. Additional kwargs are passed to plotting function.
        """
        dim = kwargs.pop("dim") if "dim" in kwargs else None
        self.freeze()

        # Get a selection of points from the various models
        # If marginalization is requested, then produce the proper 1d models
        # first
        if dim is not None:
            mdls = [mdl.marginalize_to(dim) for mdl in self.rel_fracs]
        else:
            mdls = self.rel_fracs
        pts = np.asarray([mdl.bin_centers() for mdl in mdls])

        # FIXME: We may want to decimate to a certain number of points for
        # evaluation time performance
        # Collapse the points from the different models into a single array
        pts = pts.reshape(-1, pts.shape[-1])
        if pts.ndim == 2:
            pts = pts[:,np.newaxis,:]

        pdf = np.zeros(pts.shape[0])
        for mdl, frac in zip(mdls, self.rel_fracs.values()):
            # FIXME: Need a way to unfreeze the distributions
            tmp, mdl._cached_values = mdl._cached_values, None
            pdf += mdl(pts) * frac
            mdl._cached_values = tmp

        if ax is not None:
            pts = np.squeeze(pts)
            srt = pts.argsort()
            ax.plot(pts[srt], pdf[srt], **kwargs)
        return pts, pdf

    def generate_observations(self, n_obs, phigh=None, smeared=False):
        obsdata = []
        # This is an attempt to find the bounding volume
        bins, height = self.plot()
        ### This is the time-consuming part, goes up by ^N for N parameters... ###
        height = phigh or height.max()

        bins = np.squeeze(bins)
        ### number of input parameter dimensions... ###
        # FIXME try setting a lower limit of 5 M...this should suffice for the bound since spread is so low here
        bmin, bmax = bins.min(axis=0), bins.max(axis=0)
        ### Actual rejection sampling ###
        # This is to ensure we generate the same 'true' KDE over multiple realizations
        np.random.seed()   #FIXME we should store this seed for reproducibility purposes
        while len(obsdata) < n_obs:
            if bmin.shape:
                rx = np.random.uniform(bmin, bmax, (n_obs, bmin.shape[0]))
            else:
                rx = np.random.uniform(bmin, bmax, (n_obs, 1))

            ry = np.random.uniform(0., height, n_obs)
            # One dimensional output
            if rx.ndim == 2:
                rx = rx[:,np.newaxis,:]
            # FIXME: since the height is only an estimate, if we calculate a
            # value here which is larger than height, we should reset with the
            # new height
            px = self(rx)
            obsdata.extend(rx[px>ry])
        obsdata = np.asarray(obsdata[:n_obs])

        if not smeared:
            return obsdata

        #
        # Smear out data
        #
        # FIXME: make "smeared" kwarg into a function that takes in an
        # observation point and returns a distribution so we can do do FM or PE
        n_pos = 100
        # set lower and upper limits for posterior samples
        lower, upper = 0.0, 100.0
        smeared = np.zeros((obsdata.shape[0], n_pos, obsdata.shape[-1]))
        poster = np.zeros((obsdata.shape[0], n_pos, obsdata.shape[-1]))
        # FIXME: this is too specified for multiple parameters, for now we can estimate by just having a gaussian width associated with each event
        def FM(mc,snr):
            lnM = 1.2*10**(-5.) * (10. / snr) * (mc)**(5./3)
            return mc * np.sqrt(lnM)
        for o, obs in enumerate(obsdata):
            snr = 10.   # Define the snr to be used in the FM calculation
            width = FM(obs[0,0],snr)
            #srx = [sp.stats.norm.rvs(loc=prm, scale=width, size=n_pos) for prm in obs.T]
            srx = [sp.stats.truncnorm.rvs((lower-prm)/width,(upper-prm)/width,loc=prm,scale=width, size=n_pos) for prm in obs.T]
            prx = sp.stats.norm.pdf(srx, loc=obs, scale=width)
            # hand back each sample evaluated at the pdf...not the pdf itself
            smeared[o,:,:] = np.transpose(srx)
            poster[o,:,:] = np.transpose(prx)
        # FIXME We will have another line here that provides the lalinference prior here
        return smeared   #, poster


def generate_observations(n_obs,fcn, phigh=1.0, smeared=False):
    """
    Generate a set of observations from a given model. Generic function, but could be supplanted by more specific model generators.
    """
    obsdata = []

    while len(obsdata) < n_obs:
        rx = np.random.uniform(3, 80, n_obs)
        ry = np.random.uniform(0., phigh, n_obs)
        px = fcn(np.atleast_2d(rx).T)
        obsdata.extend(rx[px>ry])
    obsdata = np.asarray(obsdata)[:n_obs]
    if not smeared:
        return obsdata

    #
    # Smear out data
    #
    smeared = []
    for datapoint in obsdata:
        srx = np.random.normal(datapoint, 1, 100)
        smeared.append(srx)
    smeared = np.asarray(smeared)
    return smeared


def normalize_samples(samples, bounds):
    """
    Normalizes samples to range [0,1] for the purposes of KDE construction
    """
    norm_samples = np.transpose([((x-b[0])/(b[1]-b[0])) for x, b in zip(samples.T, bounds)])
    return norm_samples


def denormalize_samples(norm_samples, bounds):
    """
    Denormalizes samples that are drawn from the normalzed KDE
    """
    samples = np.transpose([(x*(b[1]-b[0]) + b[0]) for x, b in zip(norm_samples.T, bounds)])
    return samples


def scale_to_unity(bounds):
    """
    Provides scale factor to renormalize pdf evaluation on the original bounds of the data
    """
    ranges = [b[1]-b[0] for b in bounds]
    scale_factor = np.product(ranges)
    return scale_factor

