import numpy as np
import pandas as pd
import random

def isotropic_spin(N=1):
    return np.arccos(np.random.uniform(-1,1, size=N))

def spin_vector(a, phi):
    """
    Generates a random spin vector using spin magnitude and tilt.
    """
    theta = np.random.uniform(0, 2*np.pi, size=len(a))
    ax = a*np.sin(theta)*np.cos(phi)
    ay = a*np.sin(theta)*np.sin(phi)
    az = a*np.cos(theta)
    return ax, ay, az

_basepath = '/projects/b1095/michaelzevin/model_selection/second_generation/data/unprocessed/'
_outdir = '/projects/b1095/michaelzevin/model_selection/second_generation/data/unprocessed/spin_models/'
_popnames = ['chi00','chi01','chi02','chi05']

# read in unprocessed data
cluster = pd.read_csv(_basepath+'cluster_bbh_models/cluster.dat', sep=' ', index_col=None)
field = pd.read_csv(_basepath+'field_bbh_models/field.dat', sep=' ', index_col=None)
field = field[['mass_1','mass_2','sep','porb','ecc','Vsys_1','Vsys_2','SNkick_1','SNkick_2','tilt_1','tilt_2','Z']]
chi_vals = cluster['chi_b'].unique()

for name, chi_b in zip(_popnames, chi_vals):
    print(name)
    clus = cluster.loc[cluster['chi_b']==chi_b]
    clus['cos_t1'] = np.cos(isotropic_spin(N=len(clus)))
    clus['cos_t2'] = np.cos(isotropic_spin(N=len(clus)))
    clus['s1x'], clus['s1y'], clus['s1z'] = spin_vector(clus['a1'], np.arccos(clus['cos_t1']))
    clus['s2x'], clus['s2y'], clus['s2z'] = spin_vector(clus['a2'], np.arccos(clus['cos_t2']))

    f = field.sample(len(clus))
    f = f.rename(columns={'mass_1':'m1', 'mass_2':'m2'})
    f['a1'] = chi_b
    f['a2'] = chi_b
    f['cos_t1'] = np.cos(f['tilt_1'])
    f['cos_t2'] = np.cos(f['tilt_2'])
    f['s1x'], f['s1y'], f['s1z'] = spin_vector(f['a1'], np.arccos(f['cos_t1']))
    f['s2x'], f['s2y'], f['s2z'] = spin_vector(f['a2'], np.arccos(f['cos_t2']))


    clus.to_hdf(_outdir + name+'.hdf', key='cluster')
    f.to_hdf(_outdir + name+'.hdf', key='field')
