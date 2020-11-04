import sys
import os
import pickle
import itertools
import copy
from tqdm import tqdm
import multiprocessing
from functools import partial
import warnings
import pdb

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from .utils.selection_effects import projection_factor_Dominik2015_interp, _PSD_defaults
from .utils.bounded_Nd_kde import Bounded_Nd_kde
from .utils.transform import mchirpq_to_m1m2, mtotq_to_m1m2, mtoteta_to_m1m2, chieff_to_s1s2, mtotq_to_mc, mtoteta_to_mchirpq, eta_to_q

from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck15

# Need to ensure all parameters are normalized over the same range
_param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,2)}
_posterior_sigmas = {"mchirp": 1.512, "q": 0.166, "chieff": 0.1043, "z": 0.0463}
_snrscale_sigmas = {"mchirp": 0.04, "eta": 0.03, "chieff": 0.14}
_maxsamps = int(1e5)
_kde_bandwidth = 0.01

# Get the interpolation function for the projection factor in Dominik+2015
# which takes in a random number and spits out a projection factor 'w'
projection_factor_interp = projection_factor_Dominik2015_interp()

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
        Generate a KDE model instance from `samples`, where `params` are \
        series in the `samples` dataframe. Additional *kwargs* can be passed \
        specifying KDE bandwidth. If `weight` is a column in your population \
        model, will assume this is the cosmological weight of each sample, and \
        will include this in the construction of all your KDEs. If `sensitivity` \
        is provided, samples used to generate the detection-weighted KDE will be \
        weighted according to the key in the argument `pdet_*sensitivity*`.
        """
        # check that the provdided sensitivity series is in the dataframe
        if sensitivity is not None:
            if 'pdet_'+sensitivity not in samples.columns:
                raise ValueError("{0:s} was specified for your detection weights, but cannot find this column in the samples datafarme!")
                
        # get the conversion factor between the underlying and detectable population
        if sensitivity is not None:
            # if cosmological weights are provided, do mock draws from the pop
            if 'weight' in samples.keys():
                mock_samp = samples.sample(int(1e6), weights=(samples['weight']/len(samples)), replace=True)
            else:
                mock_samp = samples.sample(int(1e6), replace=True)
            detectable_convfac = np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)
        else:
            detectable_convfac = 1.0

        # downsample population
        if len(samples) > _maxsamps:
            samples = samples.sample(_maxsamps)

        ### GET WEIGHTS ###
        # if cosmological weights are provided...
        if 'weight' in samples.keys():
            cosmo_weights = np.asarray(samples['weight'])
        else:
            cosmo_weights = np.ones(len(samples))
        # if detection weights are provided...
        if sensitivity is not None:
            pdets = np.asarray(samples['pdet_'+sensitivity])
        else:
            pdets = np.ones(len(samples))

        # get samples for the parameters in question
        kde_samples = samples[params]

        # get optimal SNRs for this sensitivity
        if sensitivity is not None:
            optimal_snrs = np.asarray(samples['snropt_'+sensitivity])
        else:
            optimal_snrs = np.nan*np.ones(len(samples))

        # get KDE bandwidth, if specified in kwargs
        bandwidth = kwargs['bandwidth'] if 'bandwidth' in kwargs.keys() else _kde_bandwidth

        return KDEModel(label, kde_samples, params, bandwidth, cosmo_weights, sensitivity, pdets, optimal_snrs, detectable_convfac, normalize=normalize)


    def __init__(self, label, samples, params, bandwidth=_kde_bandwidth, cosmo_weights=None, sensitivity=None, pdets=None, optimal_snrs=None, detectable_convfac=1, normalize=False):
        super()
        self.label = label
        self.samples = samples
        self.params = params
        self.bandwidth = bandwidth
        self.cosmo_weights = cosmo_weights
        self.sensitivity = sensitivity
        self.pdets = pdets
        self.optimal_snrs = optimal_snrs
        self.detectable_convfac = detectable_convfac
        self.normalize = normalize

        # Save range of each parameter
        self.sample_range = {}
        for param in samples.keys():
            self.sample_range[param] = (samples[param].min(), samples[param].max())

        # Combine the cosmological and detection weights
        if (cosmo_weights is not None) and (pdets is not None):
            combined_weights = (cosmo_weights / np.sum(cosmo_weights)) * (pdets / np.sum(pdets))
        elif pdets is not None:
            combined_weights = (pdets / np.sum(pdets))
        else:
            combined_weights = np.ones(len(samples))
        combined_weights /= np.sum(combined_weights)
        self.combined_weights = combined_weights

        # Normalize data s.t. they all are on the unit cube
        self.param_bounds = [_param_bounds[param] for param in samples.keys()]
        if self.normalize==True:
            samples = normalize_samples(np.asarray(samples), self.param_bounds)
            # also need to scale pdf by parameter range, so save this
            pdf_scale = scale_to_unity(self.param_bounds)
        else:
            samples = np.asarray(samples)
            pdf_scale = None
        self.pdf_scale = pdf_scale
        

        # add a little bit of scatter to samples that have the exact same values, as this will freak out the KDE generator
        for idx, param in enumerate(samples.T):
            if len(np.unique(param))==1:
                samples[:,idx] += np.random.normal(loc=0.0, scale=1e-5, size=samples.shape[0])

        # Get the KDE objects, specify function for pdf
        # This custom KDE handles multiple dimensions, bounds, and weights
        # and takes in samples (Ndim x Nsamps)
        if self.normalize==True:
            kde = Bounded_Nd_kde(samples.T, weights=combined_weights, bw_method=bandwidth, bounds=[(0,1)]*len(self.params))
            self.pdf = lambda x: kde(normalize_samples(x, self.param_bounds).T) / pdf_scale
        else:
            kde = Bounded_Nd_kde(samples.T, weights=combined_weights, bw_method=bandwidth, bounds=self.param_bounds)
            self.pdf =  lambda x: kde(x.T)
        self.kde = kde

        self.cached_values = None

    def sample(self, N=1):
        """
        Samples KDE and denormalizes sampled data
        """
        kde = self.kde
        if self.normalize==True:
            samps = denormalize_samples(kde.bounded_resample(N).T, self.param_bounds)
        else:
            samps = kde.bounded_resample(N).T
        return samps

    def rel_frac(self, beta):
        """
        Stores the relative fraction of samples that are drawn from this KDE model
        This is the 'detectable' branching fraction
        """
        self.rel_frac = beta

    def underlying_frac(self, beta):
        """
        Stores the branching fraction of the underlying population
        """
        self.underlying_frac = beta

    def Nobs_from_beta(self, Nobs):
        """
        Stores the branching fraction of the underlying population
        """
        self.Nobs_from_beta = Nobs

    def freeze(self, data, data_pdf=None, multiproc=True):
        """
        Caches the values of the model PDF at the data points provided. This \
        is useful to construct the hierarchal model likelihood since \
        p_hyperparam(data) is evaluated many times, but only needs to be once \
        because it's a fixed value, dependent only on the observations
        """
        self.cached_values = None
        d_pdf = data_pdf if data_pdf is not None else np.ones((data.shape[0],data.shape[1]))
        pdf_vals = []

        if multiproc==True:

            processes = []
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            for idx, (d,dp) in tqdm(enumerate(zip(data,d_pdf)), total=len(data)):
                d = d.reshape((1, d.shape[0], d.shape[1]))
                dp = [dp]
                p = multiprocessing.Process(target=self, args=(d,dp,idx,return_dict,))
                processes.append(p)
                p.start()
            for process in processes:
                process.join()

            for i in sorted(list(return_dict.keys())):
                pdf_vals.append(return_dict[i])
        else:
            for idx, (d,dp) in tqdm(enumerate(zip(data,d_pdf)), total=len(data)):
                d = d.reshape((1, d.shape[0], d.shape[1]))
                dp = dp.reshape((1, dp.shape[0]))
                pdf_vals.append(self(d, dp))

        pdf_vals = np.asarray(pdf_vals).flatten()
        self.cached_values = pdf_vals

    def __call__(self, data, data_pdf=None, proc_idx=None, return_dict=None):
        """
        The expectation is that "data" is a [Nobs x Nsample x Nparams] array. \
        If data_pdf is None, each observation is expected to have equal \
        posterior probability. Otherwise, the posterior values should be \
        provided as the same dimensions of the samples.
        """
        if self.cached_values is not None:
            return self.cached_values

        prob = np.ones(data.shape[0]) * 1e-20
        d_pdf = data_pdf if data_pdf is not None else np.ones((data.shape[0],data.shape[1]))
        for idx, (obs, dp) in enumerate(zip(np.atleast_3d(data),d_pdf)):
            # Evaluate the KDE at the samples
            prob[idx] += np.sum(self.pdf(obs) / dp) / len(obs)
            # FIXME: this is where we should divide out the prior on theta? If so, data_pdf should have the dimensions [Nobs x Nsamples]. But wouldn't this be double-counting the p(\theta) term?
        # store value for multiprocessing
        if return_dict is not None:
            return_dict[proc_idx] = prob
        return prob

    def marginalize(self, params, bandwidth=_kde_bandwidth):
        """
        Generate a new, lower dimensional, KDEModel from the parameters in [params]
        """
        label = self.label
        for p in params:
            label += '_'+p
        label += '_marginal'

        return KDEModel(label, self.samples[params], params, bandwidth, self.cosmo_weights, self.sensitivity, self.pdets, self.optimal_snrs, self.detectable_convfac, self.normalize)


    def generate_observations(self, Nobs, uncertainty, sample_from_kde=False, sensitivity='design_network', psd_path=None, multiproc=True, verbose=False):
        """
        Generates samples from KDE model. This will generated Nobs samples, storing the attribute 'self.observations' with dimensions [Nobs x Nparam]. 
        """
        if verbose:
            print("   drawing {} observations from channel {}...".format(Nobs, self.label))

        ### If sample_from_KDE is specified... ###
        # draw samples from the detection-weighted KDE, which is quicker,
        # but not compatible with SNR-dependent uncertainty
        if sample_from_kde==True:
            if uncertainty=='snr':
                raise ValueError("You cannot sample from the detection-weighted KDE with an SNR-dependent measurement uncertainty, since we need the detection probabilities and optimal SNRs of individual systems! If you wish to use SNR-weighted uncertainties, please do not use the argument 'sample-from-kde'.")
            observations = self.sample(Nobs)
            self.observations = observations
            return observations

        ### Otherwise, draw samples from the population used to construct the KDEs ###
        self.snr_thresh = _PSD_defaults['snr_network'] if 'network' in sensitivity else _PSD_defaults['snr_single']

        # allocate empty arrays
        observations = np.zeros((Nobs, self.samples.shape[-1]))
        snrs = np.zeros(Nobs)
        Thetas = np.zeros(Nobs)

        # find indices for systems that can potentially be detected
        # loop until we have enough systems with SNRs greater than the SNR threshold
        recovered_idxs = []
        for idx in tqdm(np.arange(Nobs), total=Nobs):
            detected = False
            while detected==False:
                sys_idx = np.random.choice(np.arange(len(self.pdets)), p=(self.cosmo_weights/np.sum(self.cosmo_weights)))
                pdet = self.pdets[sys_idx]
                snr_opt = self.optimal_snrs[sys_idx]
                Theta = float(projection_factor_interp(np.random.random()))

                # if the SNR is greater than the threshold, the system is "observed"
                if snr_opt*Theta >= self.snr_thresh:
                    if sys_idx in recovered_idxs:
                        continue
                    detected = True
                    observations[idx,:] = np.asarray(self.samples.iloc[sys_idx])
                    snrs[idx] = snr_opt*Theta
                    Thetas[idx] = Theta
                    recovered_idxs.append(sys_idx)

        self.observations = observations
        self.snrs = snrs
        self.Thetas = Thetas
        return observations


    def measurement_uncertainty(self, Nsamps, method='delta', observation_noise=False, verbose=False):
        """
        Mocks up measurement uncertainty from observations using specified method
        """
        if verbose:
            print("   mocking up observation uncertainties for the {} channel using the '{}' method...".format(self.label, method))

        params = self.params

        if method=='delta':
            # assume a delta function measurement
            obsdata = np.expand_dims(self.observations, 1)
            return obsdata

        # set up obsdata as [obs, samps, params]
        obsdata = np.zeros((self.observations.shape[0], Nsamps, self.observations.shape[-1]))
        
        # for 'gwevents', assume snr-independent measurement uncertainty based on the typical values for events in the catalog
        if method == "gwevents":
            for idx, obs in tqdm(enumerate(self.observations), total=len(self.observations)):
                for pidx in np.arange(self.observations.shape[-1]):
                    mu = obs[pidx]
                    sigma = [_posterior_sigmas[param] for param in self.samples.columns][pidx]
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


        # for 'snr', use SNR-dependent measurement uncertainty following procedures from Fishbach, Holz, & Farr 2018 (2018ApJ...863L..41F)
        if method == "snr":

            # to use SNR-dependent uncertainty, we need to make sure that the correct parameters are supplied

            for idx, (obs,snr,Theta) in tqdm(enumerate(zip(self.observations, self.snrs, self.Thetas)), total=len(self.observations)):
                # convert to mchirp, q
                if set(['mchirp','q']).issubset(set(params)):
                    mc_true = obs[params.index('mchirp')]
                    q_true = obs[params.index('q')]
                elif set(['mtot','q']).issubset(set(params)):
                    mc_true = mtotq_to_mc(obs[params.index('mtot')], obs[params.index('q')])
                    q_true = obs[params.index('q')]
                elif set(['mtot','eta']).issubset(set(params)):
                    mc_true, q_true = mtoteta_to_mchirpq(obs[params].index('mtot'), obs[params].index('q'))
                else:
                    raise ValueError("You need to have a mass and mass ratio parameter to to SNR-weighted uncertainty!")

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
                # Note that our Theta is the projection factor (between 0 and 1), rather than the Theta from Finn & Chernoff 1993
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

