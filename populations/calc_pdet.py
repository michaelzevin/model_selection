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
_ligo_psd = "LIGO_P1200087.dat"
_virgo_psd = "Virgo_P1200087.dat"

ifos_O3_single = {"H1":"midhighlatelow"}
ifos_O3_network = {"H1":"midhighlatelow",
            "L1":"midhighlatelow",
            "V1":"midhighlatelow"}
ifos_design_single = {"H1":"design"}
ifos_design_network = {"H1":"design",
            "L1":"design",
            "V1":"design"}

_configs = [ifos_O3_single,ifos_O3_network,ifos_design_single,ifos_design_network]
_names = ['midhighlatelow','midhighlatelow_network','design','design_network']
#_configs = [ifos_design_network]
#_names = ['pdet_designnetwork']


### Argument handling
argp = argparse.ArgumentParser()
argp.add_argument("--input-file", type=str, required=True, help="Path to infinite-sensitivity population models.")
argp.add_argument("--output-file", type=str, required=True, help="Path to save models with detection weights.")
argp.add_argument("--psd-path", type=str, required=True, help="Path to directory  with PSD files, saved in same format as Observing Scenarios data release.")
argp.add_argument("--Ntrials", type=int, default=1000, help="Define the number of monte carlo trails used for calculating the average SNR. Default=1000")
argp.add_argument("--multiproc", type=int, default=1, help="Number of cores you want to use. Default=1")
argp.add_argument("--z-max", type=str, default=3.0, help="Maximum redshift value for sampling. Default=3.0")
argp.add_argument("--snr-min-single", type=float, default=8, help="Define the SNR threshold for detectability in single-detector configuration. Default=8")
argp.add_argument("--snr-min-network", type=float, default=12, help="Define the SNR threshold for detectability in multiple-network configuration. Default=12")

args = argp.parse_args()


### MAIN FUNCTION ###

# read hdf5 file, save all submodels in a list as their hierarchical structure
models = []
def find_submodels(name, obj):
    if isinstance(obj, h5py.Dataset):
        models.append(name.rsplit('/', 1)[0])

f = h5py.File(args.input_file, 'r')
f.visititems(find_submodels)
# get all unique models
models = sorted(list(set(models)))
f.close()


for model in models:
    channel, smdl = tmp.split('/', 1)
    print("Calculating weights for model: {0:s}, channel: {1:s}".format(channel, smdl))
    pop = pd.read_hdf(args.input_file, key=model)

    # --- detector weights
    # loop over ifo configurations and sensitivities for calculating different pdet
    print("  Calculating detector weights...")
    for ifos, name in zip(_configs,_names):
        print("    configuration {0:s}...".format(name))
        if "network" in name:
            snr_min = args.snr_min_network
        else:
            snr_min = args.snr_min_single

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

        pop[name] = results

    # save pop model
    pop.to_hdf(args.output_file, key=model, mode='a')


