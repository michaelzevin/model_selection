import sys
import numpy as np
import scipy as sp
from scipy.stats import dirichlet
import pandas as pd

import emcee
from emcee import PTSampler

VERBOSE=True
_valid_samplers = {'PT_emcee': PTSampler}

_sampler = 'PT_emcee'
_prior = 'emcee_lnp'
_likelihood = 'emcee_lnlike'
_nwalkers = 16
_ntemps = 4
_nsteps = 500
_fburnin = 0.2

"""
Class for initializing and running the sampler.
"""

class Sampler(object):
    """
    Sampler class.
    """
    def __init__(self, model_names, channels, **kwargs):

        # construct dict that relates submodels to their index number
        self.model_names = model_names
        self.channels = channels

        submodels_dict = {}
        for idx, model in enumerate(model_names):
            submodels_dict[idx] = model
        self.submodels_dict = submodels_dict

        # note that ndim is (Nchannels-1) + 1 for the model index
        self.ndim = (len(channels)-1) + 1
        # kwargs
        self.sampler_name = kwargs['sampler'] if 'sampler' in kwargs else _sampler
        if self.sampler_name not in _valid_samplers.keys():
            raise NameError("Sampler {0:s} is unknown, check valid samplers!".format(self.sampler_name))
        self.sampler = _valid_samplers[self.sampler_name]

        self.prior_name = kwargs['prior'] if 'prior' in kwargs else _prior
        if self.prior_name not in _valid_priors.keys():
            raise NameError("Prior function {0:s} is unknown, check valid priors!".format(self.prior_name))
        self.prior = _valid_priors[self.prior_name]

        self.likelihood_name = kwargs['likelihood'] if 'likelihood' in kwargs else _likelihood
        if self.likelihood_name not in _valid_likelihoods.keys():
            raise NameError("Likelihood function {0:s} is unknown, check valid likelihoods!".format(self.likelihood_name))
        self.likelihood = _valid_likelihoods[self.likelihood_name]

        self.nwalkers = kwargs['nwalkers'] if 'nwalkers' in kwargs else _nwalkers
        self.ntemps = kwargs['ntemps'] if 'ntemps' in kwargs else _ntemps
        self.nsteps = kwargs['nsteps'] if 'nsteps' in kwargs else _nsteps
        self.fburnin = kwargs['fburnin'] if 'fburnin' in kwargs else _fburnin

        # if we want to sample in a single submodel, save that submodel index
        self.smdl_name = kwargs['smdl'] if 'smdl' in kwargs else None


    def sample(self, kde_models, obsdata):
        """
        Initialize and run the sampler
        """

        # Set up initial point for the walkers
        p0PT = np.empty(shape=(self.ntemps, self.nwalkers, self.ndim))
        _concentration = np.ones(len(self.channels))
        p0PT[:,:,:] = dirichlet.rvs(_concentration, (p0PT.shape[0], p0PT.shape[1]))

        # we overwrite one of the betas with the model index; we only use Nchannel-1 betas in the inference because of the implicit constraint that Sum(betas) = 1
        p0PT[:,:,0] = np.random.uniform(0, len(self.model_names), size=(self.ntemps, self.nwalkers))

        # do the sampling
        likelihood_args = [obsdata, kde_models, self.submodels_dict, self.model_names, self.channels, _concentration, self.smdl_name]
        prior_args = [self.model_names, _concentration]

        if VERBOSE:
            print("Sampling...")
        sampler = self.sampler(self.ntemps, self.nwalkers, self.ndim, self.likelihood, self.prior, loglargs=likelihood_args, logpargs=prior_args)
        for idx, result in enumerate(sampler.sample(p0PT, iterations=self.nsteps)):
            if VERBOSE:
                if (idx+1) % 50 == 0:
                    sys.stderr.write("\r  {0}% (N={1})".format(float(idx+1)*100. / self.nsteps, idx+1))
        if VERBOSE:
            print("\nSampling complete!\n")

        # remove the burnin
        burnin_steps = int(self.nsteps * self.fburnin)
        self.Nsteps_final = self.nsteps - burnin_steps
        samples = sampler.chain[:,:,burnin_steps:,:]
        lnprb = sampler.lnprobability[:,:,burnin_steps:]

        # synthesize last betas, since they sum to unity
        last_betas = (1.0-np.sum(samples[...,1:], axis=3))
        last_betas = np.expand_dims(last_betas, axis=3)
        samples = np.concatenate((samples, last_betas), axis=3)

        self.samples = samples
        self.lnprb = lnprb
        evidence = sampler.thermodynamic_integration_log_evidence(fburnin=self.fburnin)
        self.lnZ, self.dlnZ = evidence[0], evidence[1]




# --- Define the likelihood and prior

def lnp(x, model_names, _concentration):
    """
    Log of the prior. Returns logL of -inf for points outside, uniform within. Is conditional on the sum of the betas being one.
    """
    model = x[0]
    betas_tmp = np.asarray(x[1:])
    betas_tmp = np.append(betas_tmp, 1-np.sum(betas_tmp))

    if np.floor(model) not in range(0, len(model_names)):
        return -np.inf

    if np.any(betas_tmp < 0.0):
        return -np.inf

    if np.sum(betas_tmp) > 1.0:
        return -np.inf

    # Dirchlet distribution prior for betas
    return dirichlet.logpdf(betas_tmp, _concentration)


def lnlike(x, data, kde_models, submodels_dict, model_names, channels, _concentration, smdl_name):
    """
    Log of the likelihood. Selects on model, then tests beta.
    """
    # if sampling a specific submodel only, override the sampler
    if smdl_name:
        for idx, mdl in submodels_dict.items():
            if mdl==smdl_name:
                model_idx = idx
    else:
        model_idx = int(np.floor(x[0]))
    model = submodels_dict[model_idx]
    betas_tmp = np.asarray(x[1:])
    betas_tmp = np.append(betas_tmp, 1-np.sum(betas_tmp))

    # Prior
    lp = lnp(x, model_names, _concentration)
    if not np.isfinite(lp):
        return lp

    # Likelihood
    prob = np.zeros(data.shape[0])

    # Iterate over channels in this submodel, return cached values
    for channel, beta in zip(channels, betas_tmp):
        smdl = kde_models[model][channel]
        prob += beta * smdl(data)

    return np.log(prob).sum() + lp



_valid_priors = {'emcee_lnp': lnp}
_valid_likelihoods = {'emcee_lnlike': lnlike}
