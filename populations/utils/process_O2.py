import os
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15, z_at_value
import astropy.units as u


# gw events to process
_path = "/Users/michaelzevin/research/ligo/O2/PE/GWTC-1_sample_release/"
_events = ["GW150914","GW151012","GW151226","GW170104","GW170608","GW170729","GW170809","GW170814","GW170818","GW170823"]

# initialize cosmology
cosmo = Planck15

def calc_redshift(D):
    return z_at_value(cosmo.luminosity_distance, D*u.Mpc)

def to_source_frame(z, m1, m2):
    m1_src = m1/(1+z)
    m2_src = m2/(1+z)
    return m1_src, m2_src

def to_mchirp(m1, m2):
    return ((m1*m2)**(3./5) / (m1+m2)**(1./5))

def to_q(m1, m2):
    q1 = m1/m2
    q2 = m2/m1
    return np.minimum(q1,q2)

def to_chieff(m1, m2, a1, a2, cos_t1, cos_t2):
    return ((m1*a1*cos_t1 + m2*a2*cos_t2) / (m1+m2))


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


for gw, f in zip(_events, _files):
    print(f)
    df = pd.read_hdf(_path+f, key='Overall_posterior')

    # calculate redshifts
    zs = []
    print(len(df))
    for idx, row in df.iterrows():
        zs.append(calc_redshift(row['luminosity_distance_Mpc']))
    df['z'] = zs

    df['m1_src'], df['m2_src'] = to_source_frame(df['z'], df['m1_detector_frame_Msun'], df['m2_detector_frame_Msun'])

    df['mchirp'] = to_mchirp(df['m1_src'], df['m2_src'])

    df['q'] = to_q(df['m1_src'], df['m2_src'])

    df['chieff'] = to_chieff(df['m1_src'], df['m2_src'], df['spin1'], df['spin2'], df['costilt1'], df['costilt2'])



    df = df[['mchirp','q','chieff','z']]

    df.to_hdf('/Users/michaelzevin/research/model_selection/spins/data/gw_events/O2.hdf', key=gw)

