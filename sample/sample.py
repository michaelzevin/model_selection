import sys
import numpy as np
import scipy as sp
from scipy.stats import dirichlet
import pandas as pd
from functools import reduce
import operator
import pdb

import emcee
from emcee import EnsembleSampler

_valid_samplers = {'emcee': EnsembleSampler}

_sampler = 'emcee'
_prior = 'emcee_lnp'
_likelihood = 'emcee_lnlike'
_posterior = 'emcee_lnpost'

_nwalkers = 16
_nsteps = 10000
_fburnin = 0.2

"""
Class for initializing and running the sampler.
"""

class Sampler(object):
    """
    Sampler class.
    """
    def __init__(self, model_names, **kwargs):

        # Store the number of population hyperparameters and formation channels
        hyperparams = list(set([x.split('/', 1)[1] for x in model_names]))
        Nhyper = np.max([len(x.split('/')) for x in hyperparams])
        channels = sorted(list(set([x.split('/')[0] for x in model_names])))

        # construct dict that relates submodels to their index number
        submodels_dict = {}
        ctr=0
        while ctr < Nhyper:
            submodels_dict[ctr] = {}
            hyper_set = sorted(list(set([x.split('/')[ctr] for x in hyperparams])))
            for idx, model in enumerate(hyper_set):
                submodels_dict[ctr][idx] = model
            ctr += 1

        # note that ndim is (Nchannels-1) + Nhyper for the model indices
        ndim = (len(channels)-1) + Nhyper

        # store as attributes
        self.Nhyper = Nhyper
        self.model_names = model_names
        self.channels = channels
        self.ndim = ndim
        self.submodels_dict = submodels_dict


        # kwargs
        self.sampler_name = kwargs['sampler'] if 'sampler' in kwargs \
                                                            else _sampler
        if self.sampler_name not in _valid_samplers.keys():
            raise NameError("Sampler {0:s} is unknown, check valid \
samplers!".format(self.sampler_name))
        self.sampler = _valid_samplers[self.sampler_name]

        self.prior_name = kwargs['prior'] if 'prior' in kwargs else _prior
        if self.prior_name not in _valid_priors.keys():
            raise NameError("Prior function {0:s} is unknown, check valid \
priors!".format(self.prior_name))
        self.prior = _valid_priors[self.prior_name]

        self.likelihood_name = kwargs['likelihood'] if 'likelihood' in kwargs \
                                                            else _likelihood
        if self.likelihood_name not in _valid_likelihoods.keys():
            raise NameError("Likelihood function {0:s} is unknown, check \
valid likelihoods!".format(self.likelihood_name))
        self.likelihood = _valid_likelihoods[self.likelihood_name]

        self.posterior_name = kwargs['posterior'] if 'posterior' in kwargs \
                                                            else _posterior
        if self.posterior_name not in _valid_posteriors.keys():
            raise NameError("Posterior function {0:s} is unknown, check valid \
posteriors!".format(self.posterior_name))
        self.posterior = _valid_posteriors[self.posterior_name]

        self.nwalkers = kwargs['nwalkers'] if 'nwalkers' in kwargs \
                                                            else _nwalkers
        self.nsteps = kwargs['nsteps'] if 'nsteps' in kwargs else _nsteps
        self.fburnin = kwargs['fburnin'] if 'fburnin' in kwargs else _fburnin


    def sample(self, kde_models, obsdata, verbose=False):
        """
        Initialize and run the sampler
        """

        # --- Set up initial point for the walkers
        p0 = np.empty(shape=(self.nwalkers, self.ndim))

        # first, for the population hyperparameters
        for idx in np.arange(self.Nhyper):
            p0[:,idx] = np.random.uniform(0, len(self.submodels_dict[idx]), size=self.nwalkers)
        # second, for the branching fractions (we have Nchannel-1 betasin the inference because of the implicit constraint that Sum(betas) = 1
        _concentration = np.ones(len(self.channels))
        beta_p0 =  dirichlet.rvs(_concentration, p0.shape[0])
        p0[:,self.Nhyper:] = beta_p0[:,:-1]

        # --- Do the sampling
        posterior_args = [obsdata, kde_models, self.submodels_dict, self.channels, _concentration]
        if verbose:
            print("Sampling...")
        sampler = self.sampler(self.nwalkers, self.ndim, self.posterior, args=posterior_args)
        for idx, result in enumerate(sampler.sample(p0, iterations=self.nsteps)):
            if verbose:
                if (idx+1) % 50 == 0:
                    sys.stderr.write("\r  {0}% (N={1})".\
                                format(float(idx+1)*100. / self.nsteps, idx+1))
        if verbose:
            print("\nSampling complete!\n")

        # remove the burnin
        burnin_steps = int(self.nsteps * self.fburnin)
        self.Nsteps_final = self.nsteps - burnin_steps
        samples = sampler.chain[:,burnin_steps:,:]
        lnprb = sampler.lnprobability[:,burnin_steps:]

        # synthesize last betas, since they sum to unity
        last_betas = (1.0-np.sum(samples[...,self.Nhyper:], axis=2))
        last_betas = np.expand_dims(last_betas, axis=2)
        samples = np.concatenate((samples, last_betas), axis=2)

        self.samples = samples
        self.lnprb = lnprb



# --- Define the likelihood and prior

def lnp(x, submodels_dict, _concentration):
    """
    Log of the prior. 
    Returns logL of -inf for points outside, uniform within. 
    Is conditional on the sum of the betas being one.
    """
    # first get prior on the hyperparameters, flat between the model indices
    for hyper_idx in list(submodels_dict.keys()):
        hyperparam = x[hyper_idx]
        if ((hyperparam < 0) | (hyperparam > len(submodels_dict[hyper_idx]))):
            return -np.inf

    # second, get the prior on the betas as a Dirichlet prior
    betas_tmp = np.asarray(x[len(submodels_dict):])
    betas_tmp = np.append(betas_tmp, 1-np.sum(betas_tmp)) #synthesize last beta
    if np.any(betas_tmp < 0.0):
        return -np.inf
    if np.sum(betas_tmp) != 1.0:
        return -np.inf

    # Dirchlet distribution prior for betas
    return dirichlet.logpdf(betas_tmp, _concentration)


def lnlike(x, data, kde_models, submodels_dict, channels):
    """
    Log of the prior x likelihood. 
    Selects on model, then tests beta.
    """
    model_list = []
    for hyper_idx in list(submodels_dict.keys()):
        model_list.append(submodels_dict[hyper_idx][int(np.floor(x[hyper_idx]))])

    # get detectable betas
    betas_tmp = np.asarray(x[len(submodels_dict):])
    betas_tmp = np.append(betas_tmp, 1-np.sum(betas_tmp))

    # Likelihood
    prob = np.zeros(data.shape[0])

    # Iterate over channels in this submodel, return cached values
    for channel, beta in zip(channels, betas_tmp):
        model_list_tmp = model_list.copy()
        model_list_tmp.insert(0,channel)
        smdl = reduce(operator.getitem, model_list_tmp, kde_models)
        prob += beta * smdl(data)

    return np.log(prob).sum()


def lnpost(x, data, kde_models, submodels_dict, channels, _concentration):
    """
    Combines the prior and likelihood to give a log posterior probability 
    at a given point
    """
    # Prior
    log_prior = lnp(x, submodels_dict, _concentration)
    if not np.isfinite(log_prior):
        return log_prior

    # Likelihood
    log_like = lnlike(x, data, kde_models, submodels_dict, channels)

    return log_like + log_prior




_valid_priors = {'emcee_lnp': lnp}
_valid_likelihoods = {'emcee_lnlike': lnlike}
_valid_posteriors = {'emcee_lnpost': lnpost}
