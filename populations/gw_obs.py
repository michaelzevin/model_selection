import os
import numpy as np
import pandas as pd

"""
Function for using GW observations for generating the observations in model selection. Events should be stored as dataframes in hdf5 files ('GWXXXXXX*.hdf5') with the parameters being series in these dataframes. The key containing the posterior samples should be names 'posterior_samples'. 
"""

# gw events to be used
_path = "/Users/michaelzevin/research/model_selection/second_generation/data/GW_observations/"
_events = ["GW150914","GW151012","GW151226","GW170104","GW170608","GW170729","GW170809","GW170814","GW170818","GW170823"]
_Nsamps=100
_files = []

# Check to see if the files are in the obspath, else raise error
for gw in _events:
    ctr=0
    for f in os.listdir(_path):
        if gw in f:
            _files.append(f)
            ctr+=1
    if ctr==0:
        raise ValueError("Posterior samples for {0:s} not in the directory '{1:s}'!".format(gw,_path))
    if ctr>1:
        raise ValueError("More than one posterior sample file for {0:s} is present in the directory '{1:s}'!".format(gw,_path))



def generate_observations(params, smeared=None):        
    
    # Just take the median values
    if not smeared:
        samples_shape = (len(_events), 1, len(params))
        samples=np.zeros(samples_shape)

        for idx, f in enumerate(_files):
            df = pd.read_hdf(_path+f, key='posterior_samples')
            for pidx, p in enumerate(params):
                samples[idx, :, pidx] = np.median(df[p])
                
        return samples, _events

    # Smear out data using a Gaussian
    if smeared=='gaussian':
        samples_shape = (len(_events), _Nsamps, len(params))
        samples=np.zeros(samples_shape)

        for idx, f in enumerate(_files):
            df = pd.read_hdf(_path+f, key='posterior_samples')
            for pidx, p in enumerate(params):
                mean = np.mean(df[p])
                low = np.percentile(df[p], 16)
                high = np.percentile(df[p], 84)
                sigma = ((high-mean) + (mean-low))/2.0

                samples[idx, :, pidx] = np.random.normal(loc=mean, scale=sigma, size=_Nsamps)

        return samples, _events

    # Smear out data using posteriors samples
    if smeared=='posteriors':
        samples_shape = (len(_events), _Nsamps, len(params))
        samples=np.zeros(samples_shape)

        for idx, f in enumerate(_files):
            df = pd.read_hdf(_path+f, key='posterior_samples')
            for pidx, p in enumerate(params):
                if len(df) >= _Nsamps:
                    samples[idx, :, pidx] = df[p].sample(_Nsamps, replace=False)
                else:
                    samples[idx, :, pidx] = df[p].sample(_Nsamps, replace=True)

        return samples, _events

