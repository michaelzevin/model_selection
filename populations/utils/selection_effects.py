#!/usr/bin/env python

"""
Function for incorporating selection effects on a population of binaries.
"""

import os
import numpy as np
import pandas as pd

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
from lalinference.rapid_pe import lalsimutils
from pylal.antenna import response

cosmo = cosmology.Planck15



argp = argparse.ArgumentParser()
argp.add_argument("-f", "--file-path", type=str, help="Path the hdf data file.")
argp.add_argument("-k", "--key", type=str, help="Channel key in population file.")
argp.add_argument("-n", "--Ntrials", type=int, default=100, help="Define the number of monte carlo trails used for calculating the average SNR. Default=100")
argp.add_argument("-s", "--snr-min", type=float, default=8, help="Define the SNR threshold for detectability. Default=8")
argp.add_argument("-mp", "--multiproc", type=int, default=1, help="Number of cores you want to use. Default=1")
argp.add_argument("-z", "--redz", action='store_true', help="Specified whether redshift information is already contained in the dataframe. Default=False")
argp.add_argument("-zm", "--z-max", type=str, default=2.0, help="Maximum redshift value for sampling. Default=2.0")
args = argp.parse_args()


# --- read in the population
pop = pd.read_hdf(args.file_path, key=args.key)

# --- generate spin vectors with randomized azimuthal angle
pop['a1x'], pop['a1y'], pop['a1z'] = spin_vector(pop['a1'], np.arccos(pop['cos_t1']))
pop['a2x'], pop['a2y'], pop['a2z'] = spin_vector(pop['a2'], np.arccos(pop['cos_t2']))

# --- sample redshifts flat in comoving volume, if it isn't already provided
Vc_max = Vc(args.zMax).value
randVc = np.random.uniform(0,Vc_max,size=len(pop)) * u.Mpc**3
randDc = (3./(4*np.pi)*randVc)**(1./3)
zs = [z_at_value(cosmo.comoving_distance, d) for d in randDc]
pop['z'] = np.asarray(zs)

# --- get the PSDs
df = 1./32
f_low = 10.
f = np.arange(0, 2048 + df, df)
psd = map(lalsimulation.SimNoisePSDaLIGOZeroDetHighPower, f)

# --- randomly generate waveforms and do SNR calculations
for idx, system in pop.iteritems():
    m1, m2 = system['m1'], system['m2']
    s1, s2 = (system['s1x'],system['s1y'],system['s1z']), (system['s2x'],system['s2y'],system['s2z'])
    z = system['z']

    # calculate optimal SN
    hpf_opt, hxf_opt = get_waveforms(m1, m2, s1, s2, z, vary_params=False)
    # if optimal SNR < snr_min, just append a weight of 0
    hxf.data.data *= 0
    snr_opt = snr(hpf, hxf, psd, f_low)
    if snr_opt < float(args.snr_min):
        pop.loc[idx, 'weight'] = 0.0
    else:
        snrs = []
        for i in range(args.Nsamps):
            mc_snr = monte_carlo_waveforms(hpf,hxf)
            snrs.append(snr_opt*mc_snr)
        # now, we see what percentage of SNRs passed our threshold
        passed = sum(1 for i in snrs if i>=float(args.snr_min))
        pop.loc[idx, 'weight'] = float(passed) / len(snrs)



# --- Fucntions --- #

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
    flow = kwargs["flow"] if "flow" in kwargs else 10. # Stock f_low: 10 Hz

    # read approximant, or assume IMRPhenomPv2
    approx = lalsimulation.GetApproximantFromString(kwargs["approx"] if "approx" in kwargs else "IMRPhenomPv2")

    # read distance
    distance = cosmology.Planck15.comoving_distance(z).value * u.Mpc.to(u.m)

    if vary_params:
        ra, dec = np.random.uniform(0, 2*np.pi), np.arcsin(np.random.uniform(-1, 1))
        incl = np.arccos(np.random.uniform(-1, 1))
        psi = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
    else:
        ra, dec, incl = np.pi, 0.0, 0.0
        psi, phi = 0., 0.

    # Use pylal for detector response
    #fp, fx, _, _ = response(1e9, ra, dec, incl, psi, 'radians', 'L1')
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
                                           0, 0,
                                           # eccentricity
                                           0,
                                           # meanPerAno
                                           0,
                                           # deltaF, f_min, f_max, f_ref
                                           df, flow, 2048., f_low,
                                           # LALparams
                                           lal.CreateDict(),
                                           # approximant
                                           approx)

    hpf.data.data *= fp * gp
    hxf.data.data *= fx * gx

    return hpf, hxf

# SNR
def snr(hpf, hxf, psd, f, f_low=10):
    """
    Calculates:
    \rho^2 = 4 \int_{f_0}^{f_h} \frac{\tilde{h}^{\conj}(f)\tilde{h}(f)}{S(f)} df
    """
    hf = hpf.data.data + hxf.data.data
    idx = np.where(f > f_low)
    hf_cut = hf[idx]
    psd_cut = np.array(psd)[idx]

    return np.sqrt(4*np.real(np.sum(hf_cut*np.conj(hf_cut)*df/psd_cut)))

# MC sample over waveforms
def monte_carlo_waveforms(hpf,hxf):

    # vary the other ers
    ra, dec = np.random.uniform(0, 2*np.pi), np.arcsin(np.random.uniform(-1, 1))
    incl = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)

    # Use pylal for detector response
    fp, fx, _, _ = response(1e9, ra, dec, incl, psi, 'radians', 'L1')
    gp, gx = (1 + np.cos(incl)**2)/2, np.cos(incl)

    p = fp * gp
    x = fx * gx

    return np.sqrt(p**2 + x**2)
