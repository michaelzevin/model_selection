import sys
import os
import pickle
import itertools
import copy
from tqdm import tqdm

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from .utils import selection_effects
from .utils.bounded_Nd_kde import Bounded_Nd_kde

from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck15

# Need to ensure all parameters are normalized over the same range
_param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,5)}
_posterior_sigmas = {"mchirp": 1.1731, "q": 0.1837, "chieff": 0.1043, "z": 0.0463}
_snrscale_sigmas = {"mchirp": 0.08, "eta": 0.21, "chieff": 0.1, "Theta": 0.21}
_KDE_maxsamps = int(1e4)

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
    def from_samples(label, samples, params, weighting=None, **kwargs):
        """
        Generate a KDE model instance from :samples:, where :params: are \
        series in the :samples: dataframe. Additional :kwargs: are passed to \
        nothing at the moment. If :weighting: is provided, will weight the \
        samples used to generate the KDE according to the column with name \
        specified by :weighting: in the :samples: dataframe.
        """
        if weighting is not None:
            if weighting not in samples.columns:
                raise ValueError("{0:s} was specified for your weights, \
but cannot find this column in the samples datafarme!")
            # only use samples in KDE that have weights greater than 0
            samples = samples.loc[samples[weighting] > 0]

        # downsample so the KDE construction doesn't take forever
        if len(samples) > _KDE_maxsamps:
            samples = samples.sample(_KDE_maxsamps)

        kde_samples = samples[params]
        weights = samples[weighting] if weighting else None

        return KDEModel(label, kde_samples, weights)


    def __init__(self, label, samples, weights=None):
        super()
        self.label = label
        self._samples = samples
        self._weights = weights

        # Normalize data s.t. they all are on the unit cube
        self._param_bounds = [_param_bounds[param] for param in samples.keys()]
        self._posterior_sigmas = [_posterior_sigmas[param] for param in samples.columns]
        samples = normalize_samples(np.asarray(samples), self._param_bounds)

        # add a little bit of scatter to samples that have the exact same 
        # values, as this will freak out the KDE generator
        for idx, param in enumerate(samples.T):
            if len(np.unique(param))==1:
                samples[:,idx] += np.random.normal(loc=0.0, scale=1e-5, \
                                                size=samples.shape[0])

        # also need to scale pdf by parameter range, so save this
        pdf_scale = scale_to_unity(self._param_bounds)

        # Get the KDE objects, specify functions for pdf and logpdf. 
        # This custom KDE handles multiple dimensions, bounds, and weights
        # and takes in samples (Ndim x Nsamps)
        _kde = Bounded_Nd_kde(samples.T, weights=weights, \
                                        bounds=self._param_bounds)
        self._pdf = lambda x: _kde(normalize_samples(x, self._param_bounds).T)\
                                        / pdf_scale
        self._kde = _kde


        # keep bounds of the samples
        self._bin_edges = []
        for dmin, dmax in zip(samples.min(axis=0), samples.max(axis=0)):
            self._bin_edges.append(np.linspace(dmin, dmax, 100))
        self._bin_edges = np.asarray(self._bin_edges)

        self._cached_values = None

    def sample(self, N=1):
        """
        Samples KDE and denormalizes sampled data
        """
        samps_norm = self._kde.bounded_resample(N).T
        samps =  denormalize_samples(samps_norm, self._param_bounds)
        return samps

    def rel_frac(self, beta):
        """
        Stores the relative fraction of samples that are drawn from this KDE model
        """
        self._rel_frac = beta

    def bin_centers(self):
        """
        Return the center points of the bins. Note that this returns the \
        Cartesian product of the bin_edges, which are the linear "axes" of \
        the parameter dimensions. This allows for N-D plotting without \
        holding N^{param} sets of points to evaluate.
        """
        edges = [(be[1:] + be[:-1]) / 2 for be in self._bin_edges]
        return np.asarray(list(itertools.product(*edges)))

    def freeze(self, data, data_pdf=None):
        """
        Caches the values of the model PDF at the data points provided. This \
        is useful to construct the hierarchal model likelihood since \
        p_hyperparam(data) is evaluated many times, but only needs to be once \
        because it's a fixed value, dependent only on the observations
        """
        self._cached_values = None
        self._cached_values = self(data, data_pdf)

    def __call__(self, data, data_pdf=None):
        """
        The expectation is that "data" is a [Nobs x Nsample x Nparams] array. \
        If data_pdf is None, each observation is expected to have equal \
        posterior probability. Otherwise, the posterior values should be \
        provided as the same dimensions of the samples.
        """
        if self._cached_values is not None:
            return self._cached_values

        prob = np.ones(data.shape[0]) * 1e-20
        for idx, obs in enumerate(np.atleast_3d(data)):
            # Evaluate the KDE at the samples
            d_pdf = data_pdf[idx] if data_pdf is not None else 1
            # FIXME: does it matter that we average rather than sum?
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

    def generate_observations(self, Nobs, Nsamps=1, measurement_uncertainty=None, rho_thresh=8):
        """
        Generates samples from KDE model. This will generated Nobs samples. \
        If measurement_uncertainty is not None, will return Nsamps posterior samples \
        according to the available methods.
        """

        observations = self.sample(Nobs)

        if not measurement_uncertainty:
            # assume a delta function measurement
            obsdata = np.expand_dims(observations, 1)
            return obsdata

        # --- smear out observations
        # set up obsdata as [obs, samps, params]
        obsdata = np.zeros((Nobs, Nsamps, observations.shape[-1]))

        if measurement_uncertainty == "gwevents":
            for idx, obs in enumerate(observations):
                for pidx in np.arange(observations.shape[-1]):
                    mu = obs[pidx]
                    sigma = self._posterior_sigmas[pidx]
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

        elif measurement_uncertainty == "snr":
            # Follow procedures from Fishbach et al. 2018, Farr et al. 2019
            # Need both chirp mass and redshift to use this method!!!
            # FIXME we want this to work when we aren't using these parameters for inference!
            params = list(self._samples.keys())
            for p in ['mchirp','q','z']:
                if p not in params:
                    raise KeyError("Need 'mchirp', 'q', and 'z' to use this measurement uncertainty method!")
            for idx, obs in enumerate(observations):
                z = obs[params.index('z')]
                mc = obs[params.index('mchirp')]
                mcdet = mc*(1+z)
                q = obs[params.index('q')]
                eta = q * (1+q)**(-2)
                dL = cosmo.luminosity_distance(z).to(u.Gpc).value
                # Get approximate optimal SNR
                rho_opt = selection_effects.snr_opt_approx(mcdet, dL)
                # Projection factor (ifo just needed for antenna pattern)
                ifo = selection_effects.get_detector("H1")
                Theta  = selection_effects.sample_extrinsic(ifo)
                rho = rho_opt * Theta
                # Apply Gaussian noise to SNR
                rho_obs = rho + np.random.normal(loc=0, scale=1)
                # FIXME: do we also need to choose a ML value??
                rho_obs = 25
                
                # Now, convert from "true" values to "observed" values
                mcdet_sig = _snrscale_sigmas['mchirp']*rho_thresh / rho_obs
                mcdet_obs = 10**(np.log10(mcdet) + norm.rvs(loc=0, \
                    scale=mcdet_sig, size=Nsamps))
                eta_sig = _snrscale_sigmas['eta']*rho_thresh / rho_obs
                eta_obs = truncnorm.rvs(a=(0-eta)/eta_sig, b=(0.25-eta)/eta_sig, loc=eta, \
                    scale=eta_sig, size=Nsamps)
                Theta_sig = _snrscale_sigmas['Theta']*rho_thresh / rho_obs
                Theta_obs = truncnorm.rvs(a=(0-Theta)/Theta_sig, b=(1-Theta)/Theta_sig, loc=Theta, \
                    scale=Theta_sig, size=Nsamps)

                # get m1 and m2 detector-frame observations
                Mvar = mcdet_obs / (eta_obs**(3./5))
                m1det_obs = Mvar + np.sqrt(Mvar**2 - 4*eta_obs*Mvar**2) / 2
                m2det_obs = Mvar - np.sqrt(Mvar**2 - 4*eta_obs*Mvar**2) / 2
                # get dL and redshift observations
                dL_obs = dL*selection_effects.snr_opt_approx(mcdet_obs, dL)*Theta_obs / rho_obs
                z_obs = np.asarray([z_at_value(cosmo.luminosity_distance, d) for d in dL_obs*u.Gpc])
                # get source-frame chirp mass
                m1_obs = m1det_obs / (1+z_obs)
                m2_obs = m2det_obs / (1+z_obs)
                mc_obs = (m1_obs * m2_obs)**(3./5) / (m1_obs + m2_obs)**(1./5)

                for pidx, param in enumerate(params):
                    if param=='mchirp':
                        obsdata[idx, :, pidx] = mc_obs
                    if param=='q':
                        q_obs = (1-np.sqrt(1-4*eta_obs)-2*eta_obs)/(2*eta_obs)
                        obsdata[idx, :, pidx] = mc_obs
                    if param=='chieff':
                        chieff = obs[params.index('chieff')]
                        chieff_sig = _snrscale_sigmas['chieff']*rho_thresh / rho_obs
                        chieff_obs = truncnorm.rvs(a=(-1-chieff)/chieff_sig, b=(1-chieff)/chieff_sig, \
                            loc=chieff, scale=chieff_sig, size=Nsamps)
                        obsdata[idx, :, pidx] = chieff_obs
                    if param=='z':
                        obsdata[idx, :, pidx] = z_obs

            import pdb; pdb.set_trace()
            return obsdata





def normalize_samples(samples, bounds):
    """
    Normalizes samples to range [0,1] for the purposes of KDE construction
    """
    norm_samples = np.transpose([((x-b[0])/(b[1]-b[0])) for x, b in \
                                        zip(samples.T, bounds)])
    return norm_samples


def denormalize_samples(norm_samples, bounds):
    """
    Denormalizes samples that are drawn from the normalzed KDE
    """
    samples = np.transpose([(x*(b[1]-b[0]) + b[0]) for x, b in \
                                        zip(norm_samples.T, bounds)])
    return samples


def scale_to_unity(bounds):
    """
    Provides scale factor to renormalize pdf evaluation on the original 
    bounds of the data
    """
    ranges = [b[1]-b[0] for b in bounds]
    scale_factor = np.product(ranges)
    return scale_factor


