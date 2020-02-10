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
from scipy.stats import norm, truncnorm
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import kalepy as kale

# Need to ensure all parameters are normalized over the same range
_param_bounds = {"mchirp": [0,100], "q": [0,1], "chieff": [-1,1]}
_smear_sigmas = {"mchirp": 1.1731, "q": 0.1837, "chieff": 0.1043}

_kde_method = 'scipy'

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


    def __init__(self, label, samples, weights=None, kde_method=_kde_method):
        super()
        self.label = label
        self._samples = samples
        self._weights = weights

        # Normalize data s.t. they all are on the unit cube
        self._param_bounds = [_param_bounds[param] for param in samples.keys()]
        self._smear_sigmas = [_smear_sigmas[param] for param in samples.columns]
        samples = normalize_samples(np.asarray(samples), self._param_bounds)

        # add a little bit of scatter to samples that have the exact same values, as this will freak out the KDE generator
        for idx, param in enumerate(samples.T):
            if len(np.unique(param))==1:
                samples[:,idx] += np.random.normal(loc=0.0, scale=1e-5, size=samples.shape[0])

        # also need to scale pdf by parameter range, so save this
        pdf_scale = scale_to_unity(self._param_bounds)

        # get the KDE objects, specify functions for pdf and logpdf
        if kde_method=='scipy':
            _kde = gaussian_kde(samples.T, weights=weights, bw_method=0.01)
            self._logpdf = lambda x: _kde.logpdf(normalize_samples(x, self._param_bounds).T) - np.log(pdf_scale)
            self._pdf = lambda x: _kde.pdf(normalize_samples(x, self._param_bounds).T)/pdf_scale

        elif kde_method=='scikit':
            bw_scott = samples.shape[0]**(-1./(samples.shape[1]+4))
            _kde = KernelDensity(kernel='gaussian', bandwidth=0.01, rtol=1e-8)
            _kde.fit(samples, sample_weight=weights)
            self._logpdf = lambda x: _kde.score_samples(normalize_samples(x, self._param_bounds)) - np.log(pdf_scale)
            self._pdf = lambda x: np.exp(_kde.score_samples(normalize_samples(x, self._param_bounds)))/pdf_scale

        elif kde_method=='statsmodels':
            # this does not support weighted samples
            _kde = KDEMultivariate(samples, var_type='uuu')
            self._logpdf = lambda x: np.log(_kde.pdf(normalize_samples(x, self._param_bounds))) - np.log(pdf_scale)
            self._pdf = lambda x: _kde.pdf(normalize_samples(x, self._param_bounds))/pdf_scale

        elif kde_method=='kalepy':
            # this is Luke Kelley's KDE generator that supports weighting and boundaries
            # takes in data in shape (D,N)
            _kde = kale.KDE(samples.T, weights=weights, kernel='parabola')
            self._logpdf = lambda x: np.log(_kde.pdf(normalize_samples(x, self._param_bounds).T, reflect=[0.0,1.0]*np.ones_like(self._param_bounds))) - np.log(pdf_scale)
            self._pdf = lambda x: _kde.pdf(normalize_samples(x, self._param_bounds).T, reflect=[0.0,1.0]*np.ones_like(self._param_bounds))/pdf_scale

        self._kde = _kde


        # keep bounds of the samples
        self._bin_edges = []
        for dmin, dmax in zip(samples.min(axis=0), samples.max(axis=0)):
            self._bin_edges.append(np.linspace(dmin, dmax, 100))
        self._bin_edges = np.asarray(self._bin_edges)

        self._cached_values = None

    def sample(self, N=1, kde_method=_kde_method):
        """
        Samples KDE and denormalizes sampled data
        """
        if kde_method=='scipy':
            samps_norm = self._kde.resample(N).T

        elif kde_method=='scikit':
            samps_norm = self._kde.sample(n_samples=N)

        elif kde_method=='statsmodels':
            raise NameError("Sampling for statsmodels not implemented yet...")

        elif kde_method=='kalepy':
            samps_norm = np.atleast_2d(self._kde.resample(N, reflect=[0.0,1.0]*np.ones_like(self._param_bounds))).T

        samps =  denormalize_samples(samps_norm, self._param_bounds)
        return samps

    def rel_frac(self, beta):
        """
        Stores the relative fraction of samples that are drawn from this KDE model
        """
        self._rel_frac = beta

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
        The expectation is that "data" is a n_obs x n_sample x n_params array. If data_pdf is None, each observation is expected to have equal posterior probability. Otherwise, the posterior values should be provided as the same dimensions of the samples.
        """
        if self._cached_values is not None:
            return self._cached_values

        prob = np.ones(data.shape[0]) * 1e-20
        for idx, obs in enumerate(np.atleast_3d(data)):
            # Evaluate the KDE at the samples
            d_pdf = data_pdf[i] if data_pdf is not None else 1
            # FIXME: does it matter that this the average rather than the sum?
            prob[idx] += np.sum(self._pdf(obs) / d_pdf) / len(obs)
        return prob

    def extent(self):
        bc = np.squeeze(self.bin_centers())
        bmin = bc.min(axis=0)
        bmax = bc.max(axis=0)
        return np.asarray([bmin, bmax])

    def marginalize(self, params):
        """
        Generate a new, lower dimensional, KDEModel from the parameters in [params]
        """
        label = self.label
        for p in params:
            label += '_'+p
        label += '_marginal'

        return KDEModel(label, self._samples[params], self._weights)

    def generate_observations(self, Nobs, Nsamps=1, smeared=None):
        """
        Generates samples from KDE model. This will generated Nobs samples. If smeared is not None, will return Nsamps posterior samples according to the available methods.
        """

        observations = self.sample(Nobs)

        if not smeared:
            # assume a delta function measurement
            obsdata = np.expand_dims(observations, 1)
            return obsdata

        # smear out observations
        if smeared not in ["gaussian"]:
            raise ValueError("Unspecified smearing procedure: {0:s}".format(smeared))

        # set up obsdata as [obs, samps, params]
        obsdata = np.zeros((Nobs, Nsamps, observations.shape[-1]))

        if smeared == "gaussian":
            for idx, obs in enumerate(observations):
                for pidx in np.arange(observations.shape[-1]):
                    mu = obs[pidx]
                    sigma = self._smear_sigmas[pidx]
                    low_lim = self._param_bounds[pidx][0]
                    high_lim = self._param_bounds[pidx][1]

                    # construnct gaussian and drawn samples
                    dist = norm(loc=mu, scale=sigma)
                    samps = dist.rvs(Nsamps)

                    # reflect samples if drawn past the parameters bounds
                    above_idxs = np.argwhere(samps>high_lim)
                    samps[above_idxs] = high_lim - (samps[above_idxs]-high_lim)
                    below_idxs = np.argwhere(samps<low_lim)
                    samps[below_idxs] = low_lim + (low_lim - samps[below_idxs])

                    obsdata[idx, :, pidx] = samps

            return obsdata





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






