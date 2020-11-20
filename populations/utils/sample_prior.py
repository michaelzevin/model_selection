import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
import emcee
from emcee import EnsembleSampler

_basepath = '/Users/michaelzevin/research/model_selection/model_selection/data/beta_prior/'

# set number of channels
Nchannels=5
Ndim=(Nchannels-1)

# set chain info
Nwalkers=16
Nsteps=100*10000
fburnin=0.2

# get starting points p0
p0 = np.empty(shape=(Nwalkers,Ndim))
_concentration = np.ones(Nchannels)
beta_p0 =  dirichlet.rvs(_concentration, p0.shape[0])
p0[:, :] = beta_p0[:,:-1]

# prior function
def lnp(x, concentration):
    """
    Log of the prior.
    Returns logL of -inf for points outside, uniform within.
    Is conditional on the sum of the betas being one.
    """

    betas_tmp = np.asarray(x)
    betas_tmp = np.append(betas_tmp, 1-np.sum(betas_tmp)) #synthesize last beta
    if np.any(betas_tmp < 0.0):
        return -np.inf
    if np.sum(betas_tmp) != 1.0:
        return -np.inf

    # Dirchlet distribution prior for betas
    return dirichlet.logpdf(betas_tmp, concentration)

# do sampling
sampler = EnsembleSampler(Nwalkers, Ndim, lnp, args=[_concentration])
for idx, result in enumerate(sampler.sample(p0, iterations=Nsteps)):
    if (idx+1) % (Nsteps/200) == 0:
        sys.stderr.write("\r  {0}% (N={1})".format(float(idx+1)*100. / Nsteps, idx+1))

# remove burnin
burnin_steps = int(Nsteps * fburnin)
samples = sampler.chain[:, burnin_steps:, :]
lnprb = sampler.lnprobability[:,burnin_steps:]

# synthesize last beta
last_betas = (1.0-np.sum(samples, axis=2))
last_betas = np.expand_dims(last_betas, axis=2)
samples = np.concatenate((samples, last_betas), axis=2)

# reshape prior samples
samples = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))
lnprb = lnprb.reshape((lnprb.shape[0]*lnprb.shape[1]))

# save as hdf file
file_name = 'prior_samples_'+str(Nchannels)+'channel.hdf5'
columns = ['beta'+str(n+1) for n in np.arange(Nchannels)]
df = pd.DataFrame(samples, columns=columns)
df.to_hdf(os.path.join(_basepath,file_name), key='samples')
columns = ['lnprb']
df = pd.DataFrame(lnprb, columns=columns)
df.to_hdf(os.path.join(_basepath,file_name), key='lnprb')
