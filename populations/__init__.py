import sys
import os
import pickle
import itertools
import copy
from tqdm import tqdm
import warnings
import pdb

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from .utils.selection_effects import detection_probability, _PSD_defaults
from .utils.bounded_Nd_kde import Bounded_Nd_kde
from .utils.transform import mchirpq_to_m1m2, mtotq_to_m1m2, mtoteta_to_m1m2, chieff_to_s1s2, mtotq_to_mc, mtoteta_to_mchirpq, eta_to_q

from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck15

# Need to ensure all parameters are normalized over the same range
_param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,2)}
_posterior_sigmas = {"mchirp": 1.1731, "q": 0.1837, "chieff": 0.1043, "z": 0.0463}
_snrscale_sigmas = {"mchirp": 0.08, "eta": 0.022, "chieff": 0.14, "Theta": 0.21}
_maxsamps = int(1e5)
_kde_bandwidth = 0.01

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
    def from_samples(label, samples, params, sensitivity=None, normalize=False, **kwargs):
        """
        Generate a KDE model instance from :samples:, where :params: are \
        series in the :samples: dataframe. Additional :kwargs: are passed to \
        nothing at the moment. If 'weight' is a column in your population \
        model, will assume this is the cosmological weight of each sample, and \
        will include this in the construction of all your KDEs. If :sensitivity: \
        is provided, samples used to generate the detection-weighted KDE will be \
        weighted according to the key in the argument :sensitivity:.
        """
        # check that the provdided sensitivity is in the dataframe
        if sensitivity is not None:
            if sensitivity not in samples.columns:
                raise ValueError("{0:s} was specified for your detection weights, but cannot find this column in the samples datafarme!")
                
        # get the conversion factor between the underlying and detectable populatin
        if sensitivity is not None:
            # if cosmological weights are provided, do mock draws from the pop
            if 'weight' in samples.keys():
                mock_samp = samples.sample(int(1e6), weights=samples['weight'], replace=True)
            else:
                mock_samp = samples.sample(int(1e6), replace=True)
            detectable_convfac = np.sum(mock_samp[sensitivity]) / len(mock_samp)
        else:
            detectable_convfac = 1.0

        # downsample population 
        if len(samples) > _maxsamps:
            samples = samples.sample(_maxsamps)

        ### GET WEIGHTS ###
        # if cosmological weights are provided...
        if 'weight' in samples.keys():
            cosmo_weights = samples['weight'] / np.sum(samples['weight'])
        else:
            cosmo_weights = np.ones(len(samples)) / len(samples)
        # if detection weights are provided...
        if sensitivity is not None:
            det_weights = samples[sensitivity] / np.sum(samples[sensitivity])
        else:
            det_weights = np.ones(len(samples)) / len(samples)

        # get samples for the parameters in question
        kde_samples = samples[params]

        return KDEModel(label, kde_samples, cosmo_weights, det_weights, detectable_convfac, normalize=normalize)


    def __init__(self, label, samples, cosmo_weights=None, det_weights=None, detectable_convfac=1, normalize=False):
        super()
        self.label = label
        self.samples = samples
        self.cosmo_weights = cosmo_weights
        self.det_weights = det_weights
        self.detectable_convfac = detectable_convfac
        self.normalize = normalize

        # Save range of each parameter
        self.sample_range = {}
        for param in samples.keys():
            self.sample_range[param] = (samples[param].min(), samples[param].max())

        # Combine the cosmological and detection weights
        if (cosmo_weights is not None) and (det_weights is not None):
            combined_weights = cosmo_weights * det_weights
        elif det_weights is not None:
            combined_weights = det_weights
        else:
            combined_weights = np.ones(len(samples)) / len(samples)
        combined_weights /= np.sum(combined_weights)
        self.combined_weights = combined_weights

        # Normalize data s.t. they all are on the unit cube
        self.param_bounds = [_param_bounds[param] for param in samples.keys()]
        self.posterior_sigmas = [_posterior_sigmas[param] for param in samples.columns]
        if self.normalize==True:
            samples = normalize_samples(np.asarray(samples), self.param_bounds)
            # also need to scale pdf by parameter range, so save this
            pdf_scale = scale_to_unity(self.param_bounds)
        else:
            samples = np.asarray(samples)

        # add a little bit of scatter to samples that have the exact same values, as this will freak out the KDE generator
        for idx, param in enumerate(samples.T):
            if len(np.unique(param))==1:
                samples[:,idx] += np.random.normal(loc=0.0, scale=1e-5, size=samples.shape[0])

        # Get the KDE objects, specify function for pdf
        # This custom KDE handles multiple dimensions, bounds, and weights
        # and takes in samples (Ndim x Nsamps)
        # We save both the detection-weighted and underlying KDEs, as we'll need both
        kde = Bounded_Nd_kde(samples.T, weights=combined_weights, bw_method=_kde_bandwidth, bounds=self.param_bounds)
        if cosmo_weights is not None:
            kde_underlying = Bounded_Nd_kde(samples.T, weights=cosmo_weights, bw_method=_kde_bandwidth, bounds=self.param_bounds)
        else:
            kde_underlying = Bounded_Nd_kde(samples.T, weights=None, bw_method=_kde_bandwidth, bounds=self.param_bounds)
        self.kde = kde
        self.kde_underlying = kde_underlying

        if self.normalize==True:
            self.pdf = lambda x: kde(normalize_samples(x, self.param_bounds).T) / pdf_scale
            self.pdf_underlying = lambda x: kde_underlying(normalize_samples(x, self.param_bounds).T) / pdf_scale
        else:
            self.pdf =  lambda x: kde(x.T)
            self.pdf_underlying =  lambda x: kde_underlying(x.T)

        self.cached_values = None

    def sample(self, N=1, weighted_kde=False):
        """
        Samples KDE and denormalizes sampled data
        """
        # FIXME this needs to be expanded to draw from the underlying KDE and calculate SNRs
        kde = self.kde if weighted_kde==True else self.kde_underlying
        if self.normalize==True:
            samps = denormalize_samples(kde.bounded_resample(N).T, self.param_bounds)
        else:
            samps = kde.bounded_resample(N).T
        return samps

    def rel_frac(self, beta):
        """
        Stores the relative fraction of samples that are drawn from this KDE model
        """
        self.rel_frac = beta

    def freeze(self, data, data_pdf=None):
        """
        Caches the values of the model PDF at the data points provided. This \
        is useful to construct the hierarchal model likelihood since \
        p_hyperparam(data) is evaluated many times, but only needs to be once \
        because it's a fixed value, dependent only on the observations
        """
        self.cached_values = None
        self.cached_values = self(data, data_pdf)

    def __call__(self, data, data_pdf=None):
        """
        The expectation is that "data" is a [Nobs x Nsample x Nparams] array. \
        If data_pdf is None, each observation is expected to have equal \
        posterior probability. Otherwise, the posterior values should be \
        provided as the same dimensions of the samples.
        """
        if self.cached_values is not None:
            return self.cached_values

        prob = np.ones(data.shape[0]) * 1e-20
        for idx, obs in enumerate(np.atleast_3d(data)):
            # Evaluate the KDE at the samples
            d_pdf = data_pdf[idx] if data_pdf is not None else 1
            # FIXME: does it matter that we average rather than sum?
            prob[idx] += np.sum(self.pdf(obs) / d_pdf) / len(obs)
        return prob

    def marginalize(self, params):
        """
        Generate a new, lower dimensional, KDEModel from the parameters in [params]
        """
        label = self.label
        for p in params:
            label += '_'+p
        label += '_marginal'

        return KDEModel(label, self.samples[params], self.cosmo_weights, self.det_weights, self.detectable_convfac, self.normalize)

    def generate_observations(self, Nobs, detector='design_network', psd_path=None, from_detectable=False):
        """
        Generates samples from KDE model. This will generated Nobs samples, storing the attribute 'self.observations' with dimensions [Nobs x Nparam]. 
        """
        # FIXME I'll need to change this up to work for single parameters...

        if from_detectable==True:
            observations = self.sample(Nobs, weighted_kde=True)
            self.observations = observations
            return observations

        params = list(self.samples.keys())

        if detector not in _PSD_defaults.keys():
            # fall back on drawing from detection-weighted KDE
            warnings.warn('The detector ({}) you specified is not in PSD defaults, falling back to generating observations using the detection-weighted KDEs and measurement uncertainties tuned to GW events'.format(detector))
            self.detector = None
            self.snr_thresh = None
            snrs = np.nan * np.ones(Nobs)
            observations = self.sample(Nobs, weighted_kde=True)
        elif not (set(['mchirp','q','z']).issubset(set(params)) \
                | set(['mtot','q','z']).issubset(set(params)) \
                | set(['mtot','eta','z']).issubset(set(params))):
            # fall back on drawing from detection-weighted KDE
            warnings.warn('The parameters you specified for inference ({}) do not have enough information to draw detectable sources from the underlying population, falling back to generating observations using the detection-weighted KDEs and measurement uncertainties tuned to GW events'.format(','.join(params)))
            self.detector = None
            self.snr_thresh = None
            snrs = np.nan * np.ones(Nobs)
            observations = self.sample(Nobs, weighted_kde=True)

        else:
            # draw observations from underlying distributions and calculate SNRs
            self.detector = detector
            self.snr_thresh = _PSD_defaults['snr_network'] if 'network' in detector else _PSD_defaults['snr_single']

            # first check if spin info is provided
            if not (set(['mchirp','q','z','chieff']).issubset(set(params)) \
              | set(['mtot','q','z','chieff']).issubset(set(params)) \
              | set(['mtot','eta','z','chieff']).issubset(set(params))):
                spin_info = False
                warnings.warn('The parameters you specified for inference ({}) do not have spin information, assuming non-spinning BHs in the SNR calculations.'.format(','.join(params)))
            else:
                spin_info = True
            
            print('   generating observations from underlying distribution for {}'.format(self.label))
            observations = np.zeros((Nobs, self.samples.shape[-1]))
            snrs = np.zeros(Nobs)
            Thetas = np.zeros(Nobs)
            for n in tqdm(np.arange(Nobs)):
                detected=False
                while detected==False:
                    obs = self.sample(1, weighted_kde=False)[0]
                    # convert to component masses
                    if set(['mchirp','q']).issubset(set(params)):
                        m1, m2 = mchirpq_to_m1m2(obs[params.index('mchirp')],obs[params.index('q')])
                    elif set(['mtot','q']).issubset(set(params)):
                        m1, m2 = mtotq_to_m1m2(obs[params.index('mtot')],obs[params.index('q')])
                    elif set(['mtot','eta']).issubset(set(params)):
                        m1, m2 = mtoteta_to_m1m2(obs[params.index('mtot')],obs[params.index('eta')])
                    # convert to component spins
                    if spin_info == True:
                        s1, s2 = chieff_to_s1s2(obs[params.index('chieff')])
                    else:
                        s1, s2 = (0,0,0), (0,0,0)
                    # get redshift
                    z = obs[params.index('z')]

                    # see whether the system is detected, this will either be 1 or 0 for a single Ntrial
                    system = [m1,m2,z,s1,s2]
                    pdet, snr, Theta = detection_probability(system, ifos=_PSD_defaults[self.detector], rho_thresh=self.snr_thresh, Ntrials=1, return_snr=True, psd_path=psd_path)
                    if pdet>0:
                        observations[n,:] = obs
                        snrs[n] = np.float(snr)
                        Thetas[n] = np.float(Theta)
                        detected=True

        self.observations = observations
        self.snrs = snrs
        self.Thetas = Thetas
        return observations


    def measurement_uncertainty(self, Nsamps, method='delta', observation_noise=False):
        """
        Mocks up measurement uncertainty from observations using specified method
        """

        params = list(self.samples.keys())

        # If systems were not able to be drawn from the underlying distribution and method='snr' was specified, fall back to using 'gwevents'
        if method=='snr' and any(np.isnan(self.snrs)):
            warnings.warn("You specified SNR-dependent measurement uncertainties, but your method for generating observations does not allow for SNR calculations. Falling back to using the method 'gwevents'.")
            method = 'gwevents'

        if method=='delta':
            # assume a delta function measurement
            obsdata = np.expand_dims(self.observations, 1)
            return obsdata

        # set up obsdata as [obs, samps, params]
        obsdata = np.zeros((self.observations.shape[0], Nsamps, self.observations.shape[-1]))
        
        # for 'gwevents', assume snr-independent measurement uncertainty based on the typical values for events in the catalog
        if method == "gwevents":
            for idx, obs in enumerate(self.observations):
                for pidx in np.arange(self.observations.shape[-1]):
                    mu = obs[pidx]
                    sigma = self.posterior_sigmas[pidx]
                    low_lim = self.param_bounds[pidx][0]
                    high_lim = self.param_bounds[pidx][1]

                    # construnct gaussian and drawn samples
                    dist = norm(loc=mu, scale=sigma)

                    # if observation_noise is specified, wiggle around the observed value
                    if observation_noise==True:
                        mu_obs = dist.rvs()
                        dist = norm(loc=mu_obs, scale=sigma)

                    samps = dist.rvs(Nsamps)

                    # reflect samples if drawn past the parameters bounds
                    above_idxs = np.argwhere(samps>high_lim)
                    samps[above_idxs] = high_lim - (samps[above_idxs]-high_lim)
                    below_idxs = np.argwhere(samps<low_lim)
                    samps[below_idxs] = low_lim + (low_lim - samps[below_idxs])

                    obsdata[idx, :, pidx] = samps


        # for 'snr', use SNR-dependent measurement uncertainty following procedures from Fishbach et al. 2018
        if method == "snr":

            for idx, (obs,snr,Theta) in enumerate(zip(self.observations, self.snrs, self.Thetas)):
                # convert to mchirp, q
                if set(['mchirp','q']).issubset(set(params)):
                    mc_true = obs[params.index('mchirp')]
                    q_true = obs[params.index('q')]
                elif set(['mtot','q']).issubset(set(params)):
                    mc_true = mtotq_to_mc(obs[params.index('mtot')], obs[params.index('q')])
                    q_true = obs[params.index('q')]
                elif set(['mtot','eta']).issubset(set(params)):
                    mc_true, q_true = mtoteta_to_mchirpq(obs[params].index('mtot'), obs[params].index('q'))

                z_true = obs[params.index('z')]
                mcdet_true = mc_true*(1+z_true)
                eta_true = q_true * (1+q_true)**(-2)
                Theta_true = Theta
                dL_true = cosmo.luminosity_distance(z_true).to(u.Gpc).value

                # apply Gaussian noise to SNR
                snr_obs = snr + np.random.normal(loc=0, scale=1)

                # get the snr-weighted sigma for the detector-frame chirp mass, and draw samples
                mc_sigma = _snrscale_sigmas['mchirp']*self.snr_thresh / snr_obs
                if observation_noise==True:
                    mcdet_obs = float(10**(np.log10(mcdet_true) + norm.rvs(loc=0, scale=mc_sigma, size=1)))
                else:
                    mcdet_obs = mcdet_true
                mcdet_samps = 10**(np.log10(mcdet_obs) + norm.rvs(loc=0, scale=mc_sigma, size=Nsamps))

                # get the snr-weighted sigma for eta, and draw samples
                eta_sigma = _snrscale_sigmas['eta']*self.snr_thresh / snr_obs
                if observation_noise==True:
                    eta_obs = float(truncnorm.rvs(a=(0-eta_true)/eta_sigma, b=(0.25-eta_true)/eta_sigma, loc=eta_true, scale=eta_sigma, size=1))
                else:
                    eta_obs = eta_true
                eta_samps = truncnorm.rvs(a=(0-eta_obs)/eta_sigma, b=(0.25-eta_obs)/eta_sigma, loc=eta_obs, scale=eta_sigma, size=Nsamps)

                # get samples for projection factor (use the true value as the observed value)
                snr_opt = snr/Theta
                Theta_sigma = 0.3 / (1.0 + snr_opt/self.snr_thresh)
                Theta_samps = truncnorm.rvs(a=(0-Theta)/Theta_sigma, b=(1-Theta)/Theta_sigma, loc=Theta, scale=Theta_sigma, size=Nsamps)

                # get luminosity distance and redshift observed samples
                dL_samps = dL_true * (Theta_samps/Theta)
                z_samps = np.asarray([z_at_value(cosmo.luminosity_distance, d) for d in dL_samps*u.Gpc])

                # get source-frame chirp mass and other mass parameters
                mc_samps = mcdet_samps / (1+z_samps)
                q_samps = eta_to_q(eta_samps)
                m1_samps, m2_samps = mchirpq_to_m1m2(mc_samps,q_samps)
                m1_samps, m2_samps = mchirpq_to_m1m2(mc_samps,q_samps)
                mtot_samps = (m1_samps + m2_samps)

                for pidx, param in enumerate(params):
                    if param=='mchirp':
                        obsdata[idx, :, pidx] = mc_samps
                    elif param=='mtot':
                        obsdata[idx, :, pidx] = mtot_samps
                    elif param=='q':
                        obsdata[idx, :, pidx] = q_samps
                    elif param=='eta':
                        obsdata[idx, :, pidx] = eta_samps
                    elif param=='chieff':
                        chieff_true = obs[params.index('chieff')]
                        chieff_sigma = _snrscale_sigmas['chieff']*self.snr_thresh / snr_obs
                        if observation_noise==True:
                            chieff_obs = float(truncnorm.rvs(a=(-1-chieff_true)/chieff_sigma, b=(1-chieff_true)/chieff_sigma, loc=chieff_true, scale=chieff_sigma, size=1))
                        else:
                            chieff_obs = chieff_true
                        chieff_samps = truncnorm.rvs(a=(-1-chieff_obs)/chieff_sigma, b=(1-chieff_obs)/chieff_sigma, loc=chieff_obs, scale=chieff_sigma, size=Nsamps)
                        obsdata[idx, :, pidx] = chieff_samps
                    elif param=='z':
                        obsdata[idx, :, pidx] = z_samps

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


