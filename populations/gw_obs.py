import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import astropy.units as u
from astropy import cosmology
from astropy.cosmology import z_at_value
cosmo = cosmology.Planck18

from . import *

"""
Function for using GW observations for generating the observations in model 
selection. Events should be stored as dataframes in hdf5 files 
('GWXXXXXX*.hdf5') with the parameters being series in these dataframes. The 
key containing the posterior samples should be consistent with the naming 
scheme of GWTC-1.
"""

# can specify only a subset of GW events to use by uncommenting the line below
_events_to_use = None
#_events_to_use = ["GW150914","GW151012","GW151226","GW170104","GW170608","GW170729","GW170809","GW170814","GW170818","GW170823"]

# specify the hdf5 key for the approximant being used
_posterior_key = "combined"

# conversion function
def _gwtc_to_mchirp(gw):
    m1 = gw['m1_detector_frame_Msun']
    m2 = gw['m2_detector_frame_Msun']
    return (m1*m2)**(3./5) / (m1+m2)**(1./5)
def _gwtc_to_q(gw):
    m1 = np.asarray(gw['m1_detector_frame_Msun'])
    m2 = np.asarray(gw['m2_detector_frame_Msun'])
    q = m2/m1
    pos_idxs = np.argwhere(q > 1)
    q[pos_idxs] = m1[pos_idxs]/m2[pos_idxs]
    return q
def _gwtc_to_chieff(gw):
    m1 = np.asarray(gw['m1_detector_frame_Msun'])
    m2 = np.asarray(gw['m2_detector_frame_Msun'])
    a1 = np.asarray(gw['spin1'])
    a2 = np.asarray(gw['spin2'])
    cost1 = np.asarray(gw['costilt1'])
    cost2 = np.asarray(gw['costilt2'])
    return (m1*a1*cost1 + m2*a2*cost2) / (m1+m2)
def _gwtc_to_redshift(gw):
    # This takes time and should be done in preprocessing!
    print('converting luminosity distances to redshift...')
    redz = []
    dL = np.asarray(gw['luminosity_distance_Mpc'])
    for val in tqdm(dL):
        redz.append(z_at_value(cosmo.luminosity_distance, val*u.Mpc))
    return np.asarray(redz)
    
gwtc_dict = {'z': 'redshift'}
gwtc_transforms = {'mchirp': _gwtc_to_mchirp, 'q': _gwtc_to_q, \
                   'chieff': _gwtc_to_chieff, 'z': _gwtc_to_redshift}


def generate_observations(params, gwpath, Nsamps, mesaurement_uncertainty='delta', prior=None):

    if _events_to_use:
        gw_names = _events_to_use
        gw_files = [gw+'.hdf5' for gw in gw_names]
    else:
        gw_files = []
        for f in os.listdir(gwpath):
            if 'prior' not in f:
                gw_files.append(f)
        gw_names = [gw.split('.')[0] for gw in gw_files]

    # Check to see if the files are in the obspath, else raise error
    if _events_to_use:
        for gw_file, gw_name in zip(gw_files,gw_names):
            ctr=0
            tmp = []
            for f in os.listdir(gwpath):
                if gw_file in f:
                    ctr+=1
            if ctr==0:
                raise ValueError("Posterior samples for {0:s} not in the \
    directory '{1:s}'!".format(gw,gwpath))
            if ctr>1:
                raise ValueError("More than one posterior sample file for {0:s} \
    is present in the directory '{1:s}'!".format(gw,gwpath))


    # Set up samples for the specified smearing, as well as observations
    observations = np.zeros((len(gw_files), len(params)))
    if mesaurement_uncertainty=='delta':
        samples_shape = (len(gw_files), 1, len(params))
        samples=np.zeros(samples_shape)
    elif mesaurement_uncertainty in ['gaussian', 'posteriors']:
        samples_shape = (len(gw_files), Nsamps, len(params))
        samples=np.zeros(samples_shape)
    else:
        raise ValueError("{0:s} is not an available options for smearing GW observations!".format(mesaurement_uncertainty))

    # If prior key is set, set up empty array for prior weights p(theta)
    if prior is not None:
        p_theta = np.zeros((samples.shape[0],samples.shape[1]))
    else:
        p_theta = np.ones((samples.shape[0],samples.shape[1]))

    # Now, get the samples for each event
    for idx, f in enumerate(gw_files):
        df = pd.read_hdf(os.path.join(gwpath,f), key=_posterior_key)
        # Check to see if the necessary parameters are in the files or the 
        # transformations provided, else raise error
        for pidx, p in enumerate(params):
            # first, see if parameter is in the keys already
            if p in df.columns:
                continue
            elif p in gwtc_dict.keys():
                df = df.rename({gwtc_dict[p]:p}, axis='columns')
            elif p in gwtc_transforms.keys():
                df[p] = gwtc_transforms[p](df)
            else:
                raise KeyError("Parameter {0:s} not found in the GW data, and no transformations exist to generate it from the GW data!".format(p))

        # see if the specified prior key is in the data
        if prior is not None:
            if prior not in df.columns:
                raise KeyError("Prior key {0:s} is not in the GW data for file {1:s}!".format(prior,f))

        # get the median observations (need to return for later)
        observations[idx, :] = np.median(df[params], axis=0)

        # delta function observations
        if mesaurement_uncertainty == 'delta':
            samples[idx, :, :] = np.median(df[params], axis=0)
            if prior is not None:
                p_theta[idx, :] = np.median(df[prior])
        # gaussian smearing
        if mesaurement_uncertainty == 'gaussian':
            for pidx, p in enumerate(params):
                mean = np.mean(df[p])
                low = np.percentile(df[p], 16)
                high = np.percentile(df[p], 84)
                sigma = ((high-mean) + (mean-low))/2.0
                samples[idx, :, pidx] = np.random.normal(loc=mean, \
                                                scale=sigma, size=Nsamps)
        if mesaurement_uncertainty == 'posteriors':
            if len(df) >= Nsamps:
                sample_idxs = np.random.choice(np.arange(len(df)), size=Nsamps, replace=False)
            else:
                sample_idxs = np.random.choice(np.arange(len(df)), size=Nsamps, replace=True)

            samples[idx, :, :] = df[params].iloc[sample_idxs]
            if prior is not None:
                p_theta[idx, :] = df[prior].iloc[sample_idxs]

    return observations, samples, p_theta, gw_names

