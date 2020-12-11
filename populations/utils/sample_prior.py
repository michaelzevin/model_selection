import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from functools import reduce
import operator
from copy import deepcopy
import emcee
from emcee import EnsembleSampler

_repo_path = "/Users/michaelzevin/research/github/model_selection/model_selection/"
sys.path.insert(0, _repo_path)
from populations import bbh_models

_filepath = '/Users/michaelzevin/research/model_selection/model_selection/data/spin_models/models_reduced.hdf5'
_basepath = '/Users/michaelzevin/research/model_selection/model_selection/data/beta_prior/'
_sensitivity = 'midhighlatelow_network'

# set number of channels, params, and number of runs so normalization is the same
channels = ['CE','CHE','GC','NSC','SMT']
#channels = ['CE','GC']
params = ['mchirp','q','chieff','z']
Nruns=100
Nchannels=len(channels)

# set chain info
Nwalkers=16
Nsteps=Nruns*10000
fburnin=0.2

# set random seed
np.random.seed(11)

# --- Useful functions for accessing items in dictionary
def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)
def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

# get KDE models, which have the detection efficiencies as an attribute
model_names, kde_models = bbh_models.get_models(_filepath, channels, params, sensitivity=_sensitivity)

# set hyperparam dict
hyperparams = list(set([x.split('/', 1)[1] for x in model_names]))
Nhyper = np.max([len(x.split('/')) for x in hyperparams])
hyperparam_dict  = {}
hyperidx=0
while hyperidx < Nhyper:
    hyperidx_with_Nhyper = np.argwhere(np.asarray([len(x.split('/')) for x in hyperparams])>hyperidx).flatten()
    hyperparams_at_level = sorted(set([x.split('/')[hyperidx] for x in np.asarray(hyperparams)[hyperidx_with_Nhyper]]))
    hyperparam_dict[hyperidx] = hyperparams_at_level
    hyperidx += 1

# ---  Copy kde_models so that they all have the same levels of hyperparameters
all_models_at_deepest = all([len(x.split('/')[1:])==Nhyper for x in model_names])
while all_models_at_deepest==False:
    # loop until all models have the same length
    for model in model_names:
        # See number of hyperparameters in model, subtract one for channel
        Nhyper_in_model = len(model.split('/'))-1
        kde_hold = getFromDict(kde_models, model.split('/'))
        # loop until this model has all the hyperparam levels as well
        while Nhyper_in_model < Nhyper:
            # remove kde model from old level
            setInDict(kde_models, model.split('/'), {})
            model_names.remove(model)
            for new_hyperparam in hyperparam_dict[Nhyper_in_model]:
                # copy the same kde model for the higher hyperparam level
                new_kde = deepcopy(kde_hold)
                new_level = model.split('/') + [new_hyperparam]
                setInDict(kde_models, new_level, new_kde)
                # add new model name
                model_names.append(model+'/'+new_hyperparam)
            Nhyper_in_model += 1
        model_names.sort()
    # see if all models are at deepest level else repeat
    all_models_at_deepest = all([len(x.split('/')[1:])==Nhyper for x in model_names])
hyperparams = list(set([x.split('/', 1)[1] for x in model_names]))

# create submodels dict
submodels_dict = {}
ctr=0
while ctr < Nhyper:
    submodels_dict[ctr] = {}
    hyper_set = sorted(list(set([x.split('/')[ctr] for x in hyperparams])))
    for idx, model in enumerate(hyper_set):
        submodels_dict[ctr][idx] = model
    ctr += 1

# get starting points p0
Ndim = (len(channels)-1) + Nhyper
p0 = np.empty(shape=(Nwalkers,Ndim))
# first, for the population hyperparameters
for idx in np.arange(Nhyper):
    p0[:,idx] = np.random.uniform(0, len(submodels_dict[idx]), size=Nwalkers)
# second, for the branching fractions (we have Nchannel-1 betasin the inference because of the implicit constraint that Sum(betas) = 1
_concentration = np.ones(len(channels))
beta_p0 =  dirichlet.rvs(_concentration, p0.shape[0])
p0[:,Nhyper:] = beta_p0[:,:-1]


# prior function
def lnp(x, submodels_dict, concentration):
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
    return dirichlet.logpdf(betas_tmp, concentration)

# do sampling
sampler = EnsembleSampler(Nwalkers, Ndim, lnp, args=[submodels_dict, _concentration])
for idx, result in enumerate(sampler.sample(p0, iterations=Nsteps)):
    if (idx+1) % (Nsteps/200) == 0:
        sys.stderr.write("\r  {0}% (N={1})".format(float(idx+1)*100. / Nsteps, idx+1))

# remove burnin
burnin_steps = int(Nsteps * fburnin)
samples = sampler.chain[:, burnin_steps:, :]
lnprb = sampler.lnprobability[:,burnin_steps:]

# synthesize last beta
last_betas = (1.0-np.sum(samples[...,Nhyper:], axis=2))
last_betas = np.expand_dims(last_betas, axis=2)
samples = np.concatenate((samples, last_betas), axis=2)

# take the floor of the hyperparameters to get the model indices
for hyper_idx in list(submodels_dict.keys()):
    samples[:,:,hyper_idx] = np.floor(samples[:,:,hyper_idx]).astype(int)

# reshape prior samples
samples = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))
lnprb = lnprb.reshape((lnprb.shape[0]*lnprb.shape[1]))

# Get prior samples for both *detectable* branching fractions and *underlying* branching fractions
detectable_samples = samples.copy()
smdls = list(set([x.split('/',1)[1] for x in model_names]))
# get the conversion factors between the detectable and underlying distributions
for smdl in sorted(smdls):
    detectable_convfacs = []
    for channel in channels:
        detectable_convfacs.append(getFromDict(kde_models, [channel]+smdl.split('/')).alpha)
    detectable_convfacs = np.asarray(detectable_convfacs)
    # loop over hyperparams to get samples in this submodel
    hyperparams = smdl.split('/')
    for idx, param in enumerate(hyperparams):
        hyper_idx = list(submodels_dict[idx].keys())[list(submodels_dict[idx].values()).index(param)]
        if idx==0:
            matching_idxs = np.where(samples[:,idx] == hyper_idx)[0]
            matching_samps = samples[matching_idxs]
        else:
            matching_idxs = matching_idxs[np.where(matching_samps[:,idx] == hyper_idx)[0]]
            matching_samps = samples[matching_idxs]
    # if no samples are in this model, continue
    if len(matching_idxs)==0:
        continue
    # convert hyperparams of these samples accordingly to get the underlying betas
    converted_betas = detectable_samples[matching_idxs,len(hyperparams):] * detectable_convfacs
    converted_betas /= converted_betas.sum(axis=1, keepdims=True)
    detectable_samples[matching_idxs,len(hyperparams):] = converted_betas


# save as hdf file
file_name = 'prior_samples_'+str(Nchannels)+'channel.hdf5'
p0_cols = ['p'+str(n) for n in np.arange(Nhyper)]
beta_cols = ['beta_'+c for c in channels]
columns = p0_cols + beta_cols
df = pd.DataFrame(samples, columns=columns)
df.to_hdf(os.path.join(_basepath,file_name), key='samples')
df = pd.DataFrame(detectable_samples, columns=columns)
df.to_hdf(os.path.join(_basepath,file_name), key='detectable_samples')
columns = ['lnprb']
df = pd.DataFrame(lnprb, columns=columns)
df.to_hdf(os.path.join(_basepath,file_name), key='lnprb')
