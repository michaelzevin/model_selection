#!/usr/bin/env python

"""
Function for incorporating selection effects on a population of binaries.
"""

import os
import numpy as np
import pandas as pd
import h5py
import argparse

from scipy.interpolate import interp1d
from scipy.interpolate import interpn
from scipy.interpolate import griddata

import itertools
import multiprocessing
from functools import partial
from tqdm import tqdm

import astropy.units as u
from astropy import cosmology
from astropy.cosmology import z_at_value

# import important lal stuff
from pycbc.detector import Detector

from utils import selection_effects

cosmo = cosmology.Planck15

### Specify PSD information
_PSD_defaults = selection_effects._PSD_defaults
_PSDs_for_pdet = ['midhighlatelow_network']
#_PSDs_for_pdet = ['midhighlatelow','midhighlatelow_network','design','design_network']


### Argument handling
argp = argparse.ArgumentParser()
argp.add_argument("--input-path", type=str, required=True, help="Path to population models that need detection weights. These should be stored in a hdf5 file with the following hierarchical structure: channel/param0/param1...")
argp.add_argument("--output-path", type=str, required=True, help="Path to output population models that include detection weights.")
argp.add_argument("--psd-path", type=str, required=True, help="Path to directory with PSD files, saved in same format as Observing Scenarios data release.")
argp.add_argument("--model", type=str, default=None, help="HDF key for the model you wish to run selection effects on. If None, will run for all models the file.")
argp.add_argument("--Ntrials", type=int, default=1000, help="Define the number of monte carlo trails used for calculating the average SNR. Default=1000")
argp.add_argument("--multiproc", type=int, default=1, help="Number of cores you want to use. Default=1")

args = argp.parse_args()


### MAIN FUNCTION ###

# read hdf5 file, save all submodels in a list as their hierarchical structure
models = []
def find_submodels(name, obj):
    if isinstance(obj, h5py.Dataset):
        models.append(name.rsplit('/', 1)[0])

if args.model is not None:
    models.append(args.model)
else:
    f = h5py.File(args.input_path, 'r')
    f.visititems(find_submodels)
    # get all unique models
    models = sorted(list(set(models)))
    f.close()


for model in models:
    channel, smdl = model.split('/', 1)
    print("Calculating weights for model: {0:s}, channel: {1:s}".format(channel, smdl))
    pop = pd.read_hdf(args.input_path, key=model)

    # --- detector weights
    # loop over ifo configurations and sensitivities for calculating different pdet
    print("  Calculating detector weights...")
    for name in _PSDs_for_pdet:
        ifos = _PSD_defaults[name]
        print("    configuration {0:s}...".format(name))
        if "network" in name:
            snr_min = _PSD_defaults['snr_network']
        else:
            snr_min = _PSD_defaults['snr_single']

        # set up partial functions and organize data for multiprocessing
        systems_info = []
        func = partial(selection_effects.detection_probability, ifos=ifos, rho_thresh=snr_min, Ntrials=args.Ntrials, psd_path=args.psd_path)

        # set up systems for multiprocessing
        for idx, system in pop.iterrows():
            systems_info.append([system['m1'], system['m2'], system['z'], (system['s1x'],system['s1y'],system['s1z']), (system['s2x'],system['s2y'],system['s2z'])])


        if args.multiproc > 1:
            results = []
            mp = int(args.multiproc)
            pool = multiprocessing.Pool(mp)
            results = pool.imap(func, systems_info)
            results = np.asarray(list(results))
            pool.close()
            pool.join()
        else:
            results = []
            for system in tqdm(systems_info):
                results.append(func(system))

        results = np.reshape(results, (len(results),2))
        pop['pdet_'+name] = results[:,0]
        pop['snropt_'+name] = results[:,1]

    # save pop model
    pop.to_hdf(args.output_path, key=model, mode='a')


