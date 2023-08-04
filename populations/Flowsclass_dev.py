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
import time

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.special import logit
from scipy.special import logsumexp
from scipy.special import expit
from .utils.selection_effects import projection_factor_Dominik2015_interp, _PSD_defaults
from .utils.flow import NFlow
from .utils.transform import mchirpq_to_m1m2, mtotq_to_m1m2, mtoteta_to_m1m2, chieff_to_s1s2, mtotq_to_mc, mtoteta_to_mchirpq, eta_to_q

from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck18

# Need to ensure all parameters are normalized over the same range
_param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,10)}
_posterior_sigmas = {"mchirp": 1.512, "q": 0.166, "chieff": 0.1043, "z": 0.0463}
_snrscale_sigmas = {"mchirp": 0.04, "eta": 0.03, "chieff": 0.14}
_maxsamps = int(1e5)

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

    #Needs functions such as:
    #setting branching fractions/chi_b/alpha
    #

class FlowModel(Model):
    @staticmethod
    def from_samples(channel, samples, params, sensitivity=None, normalize=False, detectable=False):
        """
        Generate a Flow model instance from `samples`, where `params` are series in the `samples` dataframe. 
        
        If `weight` is a column in your population model, will assume this is the cosmological weight of each sample,
        and will include this in the construction of all your KDEs. If `sensitivity` 
        is provided, samples used to generate the detection-weighted KDE will be 
        weighted according to the key in the argument `pdet_*sensitivity*`.

        Inputs
        ----------
        channel : str
            channel label of form 'CE'
        samples : Dataframe
            binary samples from population synthesis.
            for all params (KDEs)
        params : list of str
            subset of mchirp, q, chieff, z
        """
        # check that the provdided sensitivity series is in the dataframe
        if sensitivity is not None:
            if 'pdet_'+sensitivity not in samples.columns:
                raise ValueError("{0:s} was specified for your detection weights, but cannot find this column in the samples datafarme!")

        #TO CHECK - will be alpha for each channel not each sub-pop?      
        # get *\alpha* for each model, defined as \int p(\theta|\lambda) Pdet(\theta) d\theta
        if sensitivity is not None:
            # if cosmological weights are provided, do mock draws from the pop
            if 'weight' in samples.keys():
                mock_samp = samples.sample(int(1e6), weights=(samples['weight']/len(samples)), replace=True)
            else:
                mock_samp = samples.sample(int(1e6), replace=True)
            alpha = np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)
        else:
            alpha = 1.0

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

        # get optimal SNRs for this sensitivity
        if sensitivity is not None:
            optimal_snrs = np.asarray(samples['snropt_'+sensitivity])
        else:
            optimal_snrs = np.nan*np.ones(len(samples))

        return FlowModel(channel, samples, params, cosmo_weights, sensitivity, pdets, optimal_snrs, alpha,
                         normalize=normalize, detectable=detectable)


    def __init__(self, label, samples, params, cosmo_weights=None, sensitivity=None, pdets=None, optimal_snrs=None,
                 alpha=1, normalize=False, detectable=False):
        """
        Will be passed in all data.

        Needs to:
        (1) map samples with logistic mapping etc, separate training and validation data
        (2) instintiate Nflow class with following methods:
            (2) train and validate flow, and other methods in current flow class
            (3) load in flow as option - flag for this
        """
        
        super()
        self.channel_label = label
        self.samples = samples
        self.params = params
        self.cosmo_weights = cosmo_weights
        self.sensitivity = sensitivity
        self.pdets = pdets
        self.optimal_snrs = optimal_snrs
        self.alpha = alpha
        self.normalize = normalize
        self.detectable = detectable

        #population hyperparams
        self.hps = [[0.,0.1,0.2,0.5]]
        if label=='CE':
            self.hps.append([0.2,0.5,1.,2.,5.])
        else:
            self.hps.append([1])

        self.no_params = np.shape(params)[0]
        self.conditionals = 2 if self.channel_label =='CE' else 1


        # Combine the cosmological and detection weights
        # detectable only used for plotting
        if self.detectable == True:
            if (cosmo_weights is not None) and (pdets is not None):
                combined_weights = (cosmo_weights / np.sum(cosmo_weights)) * (pdets / np.sum(pdets))
            elif pdets is not None:
                combined_weights = (pdets / np.sum(pdets))
            else:
                combined_weights = np.ones(len(samples))
            combined_weights /= np.sum(combined_weights)
            self.combined_weights = combined_weights
        else:
            if (cosmo_weights is not None):
                combined_weights = (cosmo_weights / np.sum(cosmo_weights))
            else:
                combined_weights = np.ones(len(samples))
            combined_weights /= np.sum(combined_weights)
            self.combined_weights = combined_weights

        # Gets the KDE objects, specify function for pdf
        # This custom KDE handles multiple dimensions, bounds, and weights, and takes in samples (Ndim x Nsamps)
        # By default, the detection-weighted KDE and underlying KDE (for samples that have Pdet>0)  are saved
        
        #flow parameters
        self.no_trans = 6
        self.no_neurons = 128
        batch_size=10000
        total_hps = np.shape(self.hps[0])[0]*np.shape(self.hps[1])[0]

        channel_ids = {'CE':0, 'CHE':1,'GC':2,'NSC':3, 'SMT':4}
        channel_id = channel_ids[self.channel_label] #will be 0, 1, 2, 3, or 4
        #number of data points (total) for each channel
        #no_binaries is total number of samples across sub-populations for non-CE channels, and no samples in each sub-population for CE channel
        channel_samples = [1e6,864124,896611,582961, 4e6]
        no_binaries = int(channel_samples[channel_id])

        flow = NFlow(self.no_trans, self.no_neurons, self.no_params, self.conditionals, no_binaries, batch_size, total_hps, RNVP=False, num_bins=4)
        self.flow = flow


    def map_samples(self, samples, params, filepath, channel):
        """
        Maps samples with logistic mapping (mchirp, q, z samples) and tanh (chieff).
        Stacks data by [mchirp,q,chieff,z,weight,chi_b,(alpha)].
        Handles any channel.

        Parameters
        ----------
        samples : dict
            dictionary of data in form 
            ['mchirp', 'q', 'chieff', 'z', 'm1' 'm2' 's1x' 's1y' 's1z' 's2x' 's2y' 's2z'
            'weight' 'pdet_midhighlatelow_network' 'snropt_midhighlatelow_network'
            'pdet_midhighlatelow' 'snropt_midhighlatelow']
        channel_lable : str
            corresponds to {'CE', 'CHE','GC','NSC', 'SMT'}
        params : list of str
            list of parameters to be used for inference, typically ['mchirp', 'q', 'chieff', 'z']
        
        Returns
        -------
        training_data : array
            data samples to be used for training the normalising flow.
            [mchirp, q, chieff, z, weights, chi_b,(alpha)]
        val_data : array
            data samples to be used for validating the normalising flow.
            for the non-CE channels this is the same as the training data.
            for the CE channel this is set to 2 of the 20 sub-populations
        mappings : array
            constants used to map the mchirp, q, and z distributions.
        """
        channel_ids = {'CE':0, 'CHE':1,'GC':2,'NSC':3, 'SMT':4}
        channel_id = channel_ids[self.channel_label] #will be 0, 1, 2, 3, or 4
        #number of data points (total) for each channel
        
        channel_samples = [1e6,864124,896611,582961, 4e6]
        no_binaries = int(channel_samples[channel_id])

        params = params + ['weight'] #read in weights as well

        if self.channel_label != 'CE':
            #Channels with 1D hyperparameters: SMT, GC, NSC, CHE

            #put data from required parameters for all alphas and chi_bs into model_stack
            models = np.zeros((no_binaries, self.no_params+1))
            model_size = np.zeros(self.no_params)
            cumulsize = np.zeros(self.no_params)

            #stack data
            for chib_id, xb in enumerate(self.hps[0]):
                model_size[chib_id] = np.shape(samples[(channel_id,chib_id)][params])[0]
                cumulsize[chib_id] = np.sum(model_size)
                models[int(cumulsize[chib_id-1]):int(np.sum(model_size))]=np.asarray(samples[(channel_id,chib_id)][params])

                models_stack = np.copy(models) #np.concatenate(models, axis=0)

            #logit and renormalise distributions pre-batching
            models_stack[:,0], max_logit_mchirp, max_mchirp = self.logistic(models_stack[:,0], True)
            if channel_id == 2: #add extra tiny amount to GC mass ratios as q=1 samples exist
                models_stack[:,1], max_q, extra_scale = self.logistic(models_stack[:,1], True)
            else:
                models_stack[:,1], max_q, _ = self.logistic(models_stack[:,1])
            models_stack[:,2] = np.arctanh(models_stack[:,2])
            models_stack[:,3],max_logit_z, max_z = self.logistic(models_stack[:,3], True)

            training_hps_stack = np.repeat(self.hps[0], (model_size).astype(int), axis=0)
            training_hps_stack = np.reshape(training_hps_stack,(-1,1))
            validation_hps_stack = np.reshape(training_hps_stack,(-1,1))
            train_models_stack = models_stack
            validation_models_stack = models_stack

        else:
            #CE channel with alpha parameter treatment

            #put data from required parameters for all alphas and chi_bs into model_stack
            models = np.zeros((4,5,no_binaries, self.no_params+1))
            removed_model_id =[7,11]
            val_hps = [[0.1,1],[0.2,.5]]

            #format which chi_bs and alphas match which parameter values being read in
            chi_b_alpha_pairs= np.zeros((20,2))
            chi_b_alpha_pairs[:,0] = np.repeat(self.hps[0],np.shape(self.hps[1])[0])
            chi_b_alpha_pairs[:,1] = np.tile(self.hps[1], np.shape(self.hps[0])[0])

            training_hp_pairs = np.delete(chi_b_alpha_pairs, removed_model_id, 0) #removes [0.1,1] and [0.2,0.5] point
            training_hps_stack = np.repeat(training_hp_pairs, no_binaries, axis=0) #repeats to cover all samples in each population
            validation_hps_stack = np.repeat(val_hps, no_binaries, axis=0)
            all_chi_b_alphas = np.repeat(chi_b_alpha_pairs, no_binaries, axis=0)

            #stack data
            for chib_id in range(4):
                for alpha_id in range(5):
                    models[chib_id, alpha_id]=np.asarray(samples[(chib_id, alpha_id)][params])

            #removing the sepeartion of chi_b and alpha into axes and just stringing them all together instead
            joined_chib_samples = np.concatenate(models, axis=0)
            models_stack = np.concatenate(joined_chib_samples, axis=0) #all models if needed

            #logit and renormalise distributions pre-batching

            #TO CHANGE - needs to account for different sets of parameters
            #chirp mass original range 0 to inf
            joined_chib_samples[:,:,0], max_logit_mchirp, max_mchirp = self.logistic(joined_chib_samples[:,:,0], True)

            #mass ratio - original range 0 to 1
            joined_chib_samples[:,:,1], max_q, _ = self.logistic(joined_chib_samples[:,:,1])

            #chieff - original range -0.5 to +1
            joined_chib_samples[:,:,2] = np.arctanh(joined_chib_samples[:,:,2])

            #redshift - original range 0 to inf
            joined_chib_samples[:,:,3], max_logit_z, max_z = self.logistic(joined_chib_samples[:,:,3], True)

            #keep samples seperated by model id (combined chi_b and alpha id) until validation samples are removed, then concatenate
            train_models = np.delete(joined_chib_samples, removed_model_id, 0) #removes samples from validation models
            train_models_stack = np.concatenate(train_models, axis=0)

            validation_model = joined_chib_samples[removed_model_id,:,:]
            validation_models_stack = np.concatenate(validation_model, axis=0)

        #concatenate data plus weights with hyperparams
        training_data = np.concatenate((train_models_stack, training_hps_stack), axis=1)
        val_data = np.concatenate((validation_models_stack, validation_hps_stack), axis=1)
        mappings = np.asarray([max_logit_mchirp, max_mchirp, max_q, None, max_logit_z, max_z])
        np.save(f'{filepath}{channel}_mappings.npy',mappings)
        
        return(training_data, val_data, mappings)

    #TO CHANGE - for fake observations. 
    def sample(self, N=1):
        """
        Samples KDE and denormalizes sampled data
        """
        kde = self.kde
        if self.normalize==True:
            #need samples from flow instead of kde??? at what conditionals?
            samps = denormalize_samples(kde.bounded_resample(N).T, self.param_bounds)
        else:
            samps = kde.bounded_resample(N).T
        return samps

    #dummy freeze class?


    def __call__(self, data, conditional_hp_idxs, prior_pdf=None, proc_idx=None, return_dict=None):
        """
        Calculate the likelihood of the observations give a particular hypermodel (given by conditional_hps).
        (this is the hyperlikelihood). \
        The expectation is that "data" is a [Nobs x Nsample x Nparams] array. \
        If prior_pdf is None, each observation is expected to have equal \
        posterior probability. Otherwise, the prior weights should be \
        provided as the dimemsions [samples(Nobs), samples(Nsamps)].

        Returns: likelihood in shape ?
        """
        
        likelihood = np.ones(data.shape[0]) * -np.inf
        prior_pdf = prior_pdf if prior_pdf is not None else np.ones((data.shape[0],data.shape[1]))
        prior_pdf[prior_pdf==0] = 1e-50

        conditional_hps = []

        #self.conditionals is number of hyperparameters, self.hps is list of hyperparameters [[chi_b],[alpha]]
        for i in range(self.conditionals):
            conditional_hps.append(self.hps[i][conditional_hp_idxs[i]])
        conditional_hps = np.asarray(conditional_hps)

        mapped_obs = self.map_obs(data)

        #conditionals tiled into shape Nobs x Nsamples x Nconditionals
        conditionals = np.repeat([conditional_hps],np.shape(mapped_obs)[1], axis=0)
        conditionals = np.repeat([conditionals],np.shape(mapped_obs)[0], axis=0)

        #calculates likelihoods for all events and all samples
        likelihoods_per_samp = self.flow.get_logprob(mapped_obs, conditionals) -np.log(prior_pdf)
        if np.any(np.isnan(likelihood_per_samp)):
            raise Exception('Obs data is outside of range of samples for channel - cannot logistic map.')

        #adds likelihoods from samples together and then sums over events, normalise by number of samples
        likelihood = logsumexp(likelihood, logsumexp(likelihoods_per_samp, axis=1) - np.log(data.shape[1]))
        
        # store value for multiprocessing
        if return_dict is not None:
            return_dict[proc_idx] = likelihood
        
        return likelihood

    def map_obs(self,data):
        """
        data : array
            Nobs x Nsamples x Nparams

        TO CHANGE - account for different subsets of parameters.
        mappings in form [max_logit_mchirp, max_mchirp, max_q, None, max_logit_z, max_z]
        """
        mapped_data = np.zeros((np.shape(data)[0],np.shape(data)[1]))

        mapped_data[:,0],_,_= self.logistic(data[:,0], True, False, self.mappings[0], self.mappings[1])
        mapped_data[:,1],_,_= self.logistic(data[:,1], True, False, self.mappings[2])
        mapped_data[:,2]= np.tanh(data[:,2])
        mapped_data[:,3],_,_= self.logistic(data[:,3], True, False, self.mappings[4], self.mappings[5])

        return mapped_data


    def logistic(self, data,rescaling=False, wholedataset=True, max =1, rescale_max=1):
        if rescaling:
            if wholedataset:
                rescale_max = np.max(data) + 0.01
            else:
                rescale_max = rescale_max
            d = data/rescale_max
        else:
            rescale_max = None
        #if data <0 or data >1:
            #raise Exception('Data out of bounds for logistic mapping')
        d = logit(d)
        if wholedataset:
            max = np.max(d)
        else:
            max = max
        d /= max
        return([d, max, rescale_max])

    def expistic(self, data, max, rescale_max=None):
        data*=max
        data = expit(data)
        if rescale_max != None:
            data *=rescale_max
        return(data)

    def train(self, lr, epochs, batch_no, filepath, channel):
        training_data, val_data, self.mappings = self.map_samples(self.samples, self.params, filepath, channel)
        save_filename = f'{filepath}{channel}.pt'
        self.flow.trainval(lr, epochs, batch_no, save_filename, training_data, val_data)

    def load_model(self, filepath, channel):
        self.flow.load_model(f'{filepath}{channel}.pt')
        self.mappings = np.load(f'{filepath}{channel}_mappings.npy', allow_pickle=True)


    ######CURRENTLY don't worry about functions below here - theyre used for plotting or simulated events
    """
    def marginalize(self, params, alpha, bandwidth=_kde_bandwidth):

        #Generate a new, lower dimensional, KDEModel from the parameters in [params]

        label = self.label
        for p in params:
            label += '_'+p
        label += '_marginal'

        return KDEModel(label, self.samples[params], params, bandwidth, self.cosmo_weights, self.sensitivity, self.pdets, self.optimal_snrs, alpha, self.normalize, self.detectable)


    def generate_observations(self, Nobs, uncertainty, sample_from_kde=False, sensitivity='design_network', multiproc=True, verbose=False):

        #Generates samples from KDE model. This will generated Nobs samples, storing the attribute 'self.observations' with dimensions [Nobs x Nparam]. 

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

        #Mocks up measurement uncertainty from observations using specified method

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
    """