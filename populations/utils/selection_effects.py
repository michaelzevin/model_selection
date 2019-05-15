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

import lalsimulation
import lal

import astropy.units as u
from astropy import cosmology
from astropy.cosmology import z_at_value

# import important lal stuff
import lal, lalsimulation
from pycbc.detector import Detector

cosmo = cosmology.Planck15

_dirpath = '/projects/b1095/michaelzevin/model_selection/second_generation/data/processed_noredz/'
_outdir = '/projects/b1095/michaelzevin/model_selection/second_generation/data/processed/'

argp = argparse.ArgumentParser()
argp.add_argument("-m", "--model", type=str, required=True, help="Name of the model you wish to calculate SNR weights for.")
argp.add_argument("-n", "--Ntrials", type=int, default=100, help="Define the number of monte carlo trails used for calculating the average SNR. Default=100")
argp.add_argument("-s", "--snr-min", type=float, default=8, help="Define the SNR threshold for detectability. Default=8")
argp.add_argument("-mp", "--multiproc", type=int, default=1, help="Number of cores you want to use. Default=1")
argp.add_argument("-zm", "--z-max", type=str, default=2.0, help="Maximum redshift value for sampling. Default=2.0")
args = argp.parse_args()


# --- Functions --- #

# Spin vector
def spin_vector(a, phi):
    """
    Generates a random spin vector using spin magnitude and tilt.
    """
    theta = np.random.uniform(0, 2*np.pi, size=len(a))
    ax = a*np.sin(theta)*np.cos(phi)
    ay = a*np.sin(theta)*np.sin(phi)
    az = a*np.cos(theta)
    return ax, ay, az

# Comoving volume
def Vc(z):
    Dl = cosmo.luminosity_distance(z)
    return 4./3*np.pi*Dl**3 / (1+z)**3

# Waveform generator
def get_waveform(m1, m2, s1, s2, z, vary_params=False, **kwargs):

    # read masses
    m1 = m1*u.Msun.to(u.kg)
    m2 = m2*u.Msun.to(u.kg)

    # assume non-spinning
    s1x, s1y, s1z = s1[0],s1[1],s1[2]
    s2x, s2y, s2z = s2[0],s2[1],s2[2]

    # read f_low, or assume 10 Hz if not in kwargs
    f_low = kwargs["f_low"] if "f_low" in kwargs else 10. # Stock f_low: 10 Hz

    # read approximant, or assume IMRPhenomPv2
    approx = lalsimulation.GetApproximantFromString(kwargs["approx"] if "approx" in kwargs else "IMRPhenomPv2")

    # read distance
    distance = cosmology.Planck15.comoving_distance(z).value * u.Mpc.to(u.m)

    if vary_params:
        ra, dec = np.random.uniform(0, 2*np.pi), np.arcsin(np.random.uniform(-1, 1))
        incl = np.arccos(np.random.uniform(-1, 1))
        psi = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        fp, fx = det.antenna_pattern(ra, dec, psi, 1e9)
        gp, gx = (1 + np.cos(incl)**2)/2, np.cos(incl)
    else:
        ra, dec, incl = np.pi, 0.0, 0.0
        psi, phi = 0., 0.
        fp, fx = 1, 1
        gp, gx = (1 + np.cos(incl)**2)/2, np.cos(incl)


    hpf, hxf = lalsimulation.SimInspiralFD(
                                           # masses
                                           m1, m2,
                                           # spins
                                           s1x, s1y, s1z,
                                           s2x, s2y, s2z,
                                           # distance
                                           distance,
                                           # inclination
                                           incl,
                                           # phiRef, longAscNodes,
                                           phi, 0,
                                           # eccentricity
                                           0,
                                           # meanPerAno
                                           0,
                                           # deltaF, f_min, f_max, f_ref
                                           df, f_low, 2048., f_low,
                                           # LALparams
                                           lal.CreateDict(),
                                           # approximant
                                           approx)

    hpf.data.data *= fp * gp
    hxf.data.data *= fx * gx

    return hpf, hxf

# SNR
def snr(hpf, hxf, psd, freqs, f_low=10):
    """
    Calculates:
    \rho^2 = 4 \int_{f_0}^{f_h} \frac{\tilde{h}^{\conj}(f)\tilde{h}(f)}{S(f)} df
    """
    hf = hpf.data.data + hxf.data.data
    idx = np.where(freqs >= f_low)
    hf_cut = hf[idx]

    return np.sqrt(4*np.real(np.sum(hf_cut*np.conj(hf_cut)*df/psd)))

# MC sample over waveforms
def monte_carlo_waveforms(hpf,hxf):

    # vary extrinsic parametrers
    ra, dec = np.random.uniform(0, 2*np.pi), np.arcsin(np.random.uniform(-1, 1))
    incl = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, np.pi)

    # Use pycbc for detector response
    fp, fx = det.antenna_pattern(ra, dec, psi, 1e9)
    gp, gx = (1 + np.cos(incl)**2)/2, np.cos(incl)

    p = fp * gp
    x = fx * gx

    return np.sqrt(p**2 + x**2)


# Parallelizing function
def calc_weights(system, calc_redz, snr_min, z_max, Ntrials):

    # --- sample redshifts flat in comoving volume, if it isn't already provided
    if calc_redz:
        Vc_max = Vc(z_max).value
        randVc = np.random.uniform(0,Vc_max) * u.Mpc**3
        randDc = (3./(4*np.pi)*randVc)**(1./3)
        z = z_at_value(cosmo.comoving_distance, randDc)

    # --- randomly generate waveforms and do SNR calculations
    m1, m2 = system[0], system[1]
    s1, s2 = system[2], system[3]
    if not calc_redz:
        z = system[4]

    # calculate optimal SN
    hpf_opt, hxf_opt = get_waveform(m1, m2, s1, s2, z, vary_params=False)
    # if optimal SNR < snr_min, just append a weight of 0
    hxf_opt.data.data *= 0
    snr_opt = snr(hpf_opt, hxf_opt, psd, freqs, f_low)
    if snr_opt < float(snr_min):
        weight = 0.0
    else:
        snrs = []
        for i in range(Ntrials):
            snr_factor = monte_carlo_waveforms(hpf_opt,hxf_opt)
            snrs.append(snr_opt*snr_factor)
        # now, we see what percentage of SNRs passed our threshold
        passed = sum(1 for i in snrs if i>=float(snr_min))
        weight = float(passed) / len(snrs)

    return z, weight


def normalize_weights(x, low=0):
    # normalizes between 0 and 1
    if all(t==0 for t in x):
        return np.zeros_like(x)
    else:
        return((x-low)/(x.max()-low))


### MAIN FUNCTION ###

# --- read in the population
channels = list(h5py.File(_dirpath+args.model+'.hdf', 'r').keys())

for channel in channels:
    print(channel)

    pop = pd.read_hdf(_dirpath+args.model+'.hdf', key=channel)

    # --- generate spin vectors with randomized azimuthal angle
    pop['s1x'], pop['s1y'], pop['s1z'] = spin_vector(pop['a1'], np.arccos(pop['cos_t1']))
    pop['s2x'], pop['s2y'], pop['s2z'] = spin_vector(pop['a2'], np.arccos(pop['cos_t2']))

    # --- get the PSDs
    df = 1./32
    f_low = 10.
    freqs = np.arange(f_low, 2048 + df, df)
    psd = np.asarray(list(map(lalsimulation.SimNoisePSDaLIGOZeroDetHighPower, freqs)))

    # --- get the response function for IFO of choice
    det = Detector("H1")

    # --- set up partial functions and organize data for multiprocessing
    systems_info = []
    if 'z' in pop.keys():
        func = partial(calc_weights, calc_redz=False, snr_min=args.snr_min, z_max=args.z_max, Ntrials=args.Ntrials)
        for idx, system in pop.iterrows():
            systems_info.append([system['m1'], system['m2'], (system['s1x'],system['s1y'],system['s1z']), (system['s2x'],system['s2y'],system['s2z']), system['z']])
    else:
        func = partial(calc_weights, calc_redz=True, snr_min=args.snr_min, z_max=args.z_max, Ntrials=args.Ntrials)
        for idx, system in pop.iterrows():
            systems_info.append([system['m1'], system['m2'], (system['s1x'],system['s1y'],system['s1z']), (system['s2x'],system['s2y'],system['s2z'])])

    if args.multiproc > 1:
        mp = int(args.multiproc)
        pool = multiprocessing.Pool(mp)
        results = pool.map(func, systems_info)
        pool.close()
        pool.join()
    else:
        results = []
        for system in systems_info:
            results.append(func(system))

    results = np.transpose(results)
    z = results[0]
    weight = results[1]

    # record redshifts
    if 'z' not in pop.keys():
        pop['z'] = z

    # record weights
    if 'clus_weight' in pop.keys():
        combined_weight = pop['clus_weight'] * weight
        pop['weight'] = normalize_weights(combined_weight)
    else:
        pop['weight'] = normalize_weights(weight)

    # get rid of systems with 0 weight
    pop = pop.loc[pop['weight'] > 0]

    # save pop model
    pop.to_hdf(_outdir+args.model+'.hdf', key=channel)


