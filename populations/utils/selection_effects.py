#!/usr/bin/env python

"""
Functions for incorporating selection effects on a population of binaries.
"""

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

import astropy.units as u
from astropy import cosmology
from astropy.cosmology import z_at_value

import lal, lalsimulation
from pycbc.detector import Detector

cosmo = cosmology.Planck18


# --- Useful functions --- #

# Waveform generator
def get_waveform(m1, m2, z, s1=None, s2=None, vary_params=False, **kwargs):
    """
    Generates waveform. If vary_params=False, will calculate optimal SNR.
    """

    # read (source frame) masses
    m1 = m1*u.Msun.to(u.kg) * (1+z)
    m2 = m2*u.Msun.to(u.kg) * (1+z)

    # assume non-spinning unless spins provided
    if s1:
        s1x, s1y, s1z = s1[0],s1[1],s1[2]
    else:
        s1x, s1y, s1z = 0,0,0
    if s2:
        s2x, s2y, s2z = s2[0],s2[1],s2[2]
    else:
        s2x, s2y, s2z = 0,0,0

    # read f_low, df or assume 10 Hz if not in kwargs
    f_low = kwargs["f_low"] if "f_low" in kwargs else 10.
    f_high = kwargs["f_high"] if "f_high" in kwargs else 2048.
    df = kwargs["df"] if "df" in kwargs else 1./32

    # read approximant, or assume IMRPhenomPv2
    approx = lalsimulation.GetApproximantFromString(kwargs["approx"] if "approx" in kwargs else "IMRPhenomPv2")

    # read distance
    distance = cosmo.luminosity_distance(z).value * u.Mpc.to(u.m)

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
        fp, fx = 1., 0.
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
                                           df, f_low, f_high, f_low,
                                           # LALparams
                                           lal.CreateDict(),
                                           # approximant
                                           approx)

    hpf.data.data *= fp * gp
    hxf.data.data *= fx * gx

    freqs = np.arange(0., f_high+df, df)

    return hpf, hxf, freqs


# PSD call function
def get_psd(psd, ifo=None, **kwargs):
    """
    Gets the PSD from either observing scenarios table or lalsimulation
    """
    # read f_low, f_high, and df or assume default values if not in kwargs
    f_low = kwargs["f_low"] if "f_low" in kwargs else 10.
    f_high = kwargs["f_high"] if "f_high" in kwargs else 2048.

    psd_path = kwargs["psd_path"] if "psd_path" in kwargs else None

    # try to read tables if strings are provided
    if type(psd)==str:
        if ifo in ["H1", "L1"]:
            psd_data = pd.read_csv(psd_path+'/LIGO_P1200087.dat', sep=' ', index_col=False)
        elif ifo in ["V1"]:
            psd_data = pd.read_csv(psd_path+'/Virgo_P1200087.dat', sep=' ', index_col=False)
        freqs = np.asarray(psd_data["freq"])
        # observeing scenarios table provides ASDs
        psd_vals = np.asarray(psd_data[psd])**2
        psd_interp = interp1d(freqs, psd_vals)

    # otherwise, assume lalsimulation psd was provided
    else:
        freqs = np.arange(f_low, f_high+df, df)
        psd_vals = np.asarray(list(map(psd, freqs)))
        psd_interp = interp1d(freqs, psd_vals)

    return psd_interp


# Detector call function
def get_detector(ifo):
    """
    Gets the detector for the response function
    """
    if ifo not in ["H1","L1","V1"]:
        raise NameError("Detector '{0:s}' not recognized!".format(ifo))
    det = Detector(ifo)
    return det


# SNR
def snr(hpf, hxf, freqs, psd, **kwargs):
    """
    Calculates:
    \rho^2 = 4 \int_{f_0}^{f_h} \frac{\tilde{h}^{\conj}(f)\tilde{h}(f)}{S(f)} df
    """
    # read f_low, df or assume 10 Hz if not in kwargs
    f_low = kwargs["f_low"] if "f_low" in kwargs else 10.
    df = kwargs["df"] if "df" in kwargs else 1./32

    hf = hpf.data.data + hxf.data.data
    idx = np.where(freqs >= f_low)
    hf_cut = hf[idx]
    psd_vals = psd(freqs[idx])

    return np.sqrt(4*np.real(np.sum(hf_cut*np.conj(hf_cut)*df/psd_vals)))

def snr_opt_approx(mc, dL, N=11.15):
    """
    Gets optimal SNR from first-order approxmation. 
    The normalization factor is calculated using an equal-mass Mc_det=10 Msun binary at 1 Gpc.
    """
    return N*(mc/10)**(5./6) * (dL / 1)**(-1)

# Sampling of extrinsic parameters
def sample_extrinsic():
    """
    Varies extrinsic parameters of waveform for calculating detection probability
    """

    # vary extrinsic parametrers
    ra, dec = np.random.uniform(0, 2*np.pi), np.arcsin(np.random.uniform(-1, 1))
    incl = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, np.pi)

    return ra, dec, incl, psi

# Calculate projection factor
def projection_factor(det, ra, dec, incl, psi):
    """
    Calculates projection factor Theta for a given detector antenna pattern and 
    extrinsic parameters
    """

    # inclination
    gp, gx = (1 + np.cos(incl)**2)/2, np.cos(incl)

    # Use pycbc for detector response
    fp, fx = det.antenna_pattern(ra, dec, psi, 1e9)

    p = fp * gp
    x = fx * gx

    return np.sqrt(p**2 + x**2)

def projection_factor_Dominik2015_interp(alpha=1.0, a2=0.374222, a4=2.04216, a8=-2.63948):
    # Interpolation of the projection factor 'w' from Dominik+2015
    def w_cdf(w, alpha=alpha, a2=a2, a4=a4, a8=a8):
        # Note that their PDF is P(w>w'), what we really want is P(w<w') so we do 1-P(w) at the end
        term1 = a2*((1-w/alpha)**2)
        term2 = a4*((1-w/alpha)**4)
        term3 = a8*((1-w/alpha)**8)
        term4 = (1-a2-a4-a8)*((1-w/alpha)**10)
        return 1-(term1+term2+term3+term4)

    w_pts = np.linspace(0,1,10000)
    interp_func = interp1d(w_cdf(w_pts), w_pts)
    return interp_func


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
    """
    Comoving volume from redshift
    """
    Dl = cosmo.luminosity_distance(z)
    return 4./3*np.pi*Dl**3 / (1+z)**3


# Generate redshifts uniform in comoving volume
def gen_redshifts_unicomvol(n=1, z_max=3):
    """
    Generates n redshifts uniformly in comoving volume, up to a z_max
    """
    zs =  []
    for i in np.arange(n):
        Vc_max = Vc(z_max).value
        randVc = np.random.uniform(0,Vc_max) * u.Mpc**3
        randDc = (3./(4*np.pi)*randVc)**(1./3)
        zs.append(z_at_value(cosmo.comoving_distance, randDc))
    return np.asarray(zs)



# Detection probability function
def detection_probability(system, ifos={"H1":"midhighlatelow"}, rho_thresh=8.0, Ntrials=1000, return_snrs=False, **kwargs):
    """
    Calls other functions in this file to calculate a detection probability
    For multiprocessing purposes, takes in array 'system' of form:
    [m1, m2, z, (s1x,s1y,s1z), (s2x,s2y,s2z)]
    """
    # get system parameters
    m1 = system[0]
    m2 = system[1]
    z = system[2]
    s1 = system[3]
    s2 = system[4]

    # read f_low, df or assume 10 Hz if not in kwargs
    f_low = kwargs["f_low"] if "f_low" in kwargs else 10.
    f_high = kwargs["f_high"] if "f_high" in kwargs else 2048.
    df = kwargs["df"] if "df" in kwargs else 1./32
    psd_path = kwargs["psd_path"] if "psd_path" in kwargs else None

    # get the detectors of choice for the response function
    detectors = {}
    for ifo in ifos.keys():
        detectors[ifo] = get_detector(ifo)

    # get the psds
    psds={}
    for ifo, psd in ifos.items():
        psd_interp = get_psd(psd, ifo, f_low=f_low, f_high=f_high, psd_path=psd_path)
        psds[ifo] = psd_interp

    # calculate optimal SNR (sometimes hits runtime error for some reason)
    try:
        hpf_opt, hxf_opt, freqs = get_waveform(m1, m2, z, s1, s2, vary_params=False, f_low=f_low, f_high=f_high, df=df)
        rho_opts={}
        for ifo in ifos.keys():
            rho_opts[ifo] = snr(hpf_opt, hxf_opt, freqs, psds[ifo], f_low=f_low, df=df)

        snr_opt = np.linalg.norm(list(rho_opts.values()))
    except RuntimeError:
        snr_opt = 0.0

    # if the combined SNR is less than the detection threshold, give weight of 0
    if snr_opt < float(rho_thresh):
        weight = 0.0
        snrs = np.asarray(snr_opt)
        Thetas = np.nan

    else:
        snrs = []
        for i in range(Ntrials):
            network_snr = []
            ra, dec, incl, psi = sample_extrinsic()
            for ifo, det in detectors.items():
                rho_factor = projection_factor(det, ra, dec, incl, psi)
                network_snr.append(rho_opts[ifo]*rho_factor)
            snrs.append(np.linalg.norm(network_snr))
        # now, we see what percentage of SNRs passed our threshold
        snrs = np.asarray(snrs)
        passed = sum(1 for i in snrs if i>=float(rho_thresh))
        weight = float(passed) / len(snrs)

        # calculate projection factor
        Thetas = snrs/snr_opt

    if return_snrs==True:
        return weight, snr_opt, snrs, Thetas
    else:
        return weight, snr_opt



_PSD_defaults = {
    "ligo_psd": "LIGO_P1200087.dat",
    "virgo_psd": "Virgo_P1200087.dat",
    "midhighlatelow": {"H1":"midhighlatelow"},
    "midhighlatelow_network": {"H1":"midhighlatelow",
            "L1":"midhighlatelow",
            "V1":"midhighlatelow"},
    "design": {"H1":"design"},
    "design_network": {"H1":"design",
            "L1":"design",
            "V1":"design"},
    "snr_single": 8,
    "snr_network": 10}

