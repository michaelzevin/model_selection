import os
import numpy as np
import pandas as pd
import h5py

import pdb

from . import *
from .transform import _DEFAULT_TRANSFORMS, _to_chi_eff, _uniform_spinmag, _isotropic_spinmag


VERBOSE=True


_VALID_SPIN_DISTR = {
    # Uniform - |a| ~ uniform distribution in 0, 1
    "uniform": _uniform_spinmag,
    # Isotropic - |a| ~ a^2 distribution in 0, 1
    "isotropic": _isotropic_spinmag
}


### Initialization dictionaries ###
models = {}
kde_models = {}

def get_models(dirpath, params, spin_distr=None, weighting=False, **kwargs):
    """
    Call this to get all the models and submodels living in dir, as well as KDEs of these models, packed inside of dictionaries labelled in the dict structure models[model][channel].

    Will first look for these parameters as series in the dataframe. If they are not present, it will try to construct these parameters if the valid transformations are present in transforms.py. 

    If chieff is one of the params for inference and spin magnitudes are not provided, this function will first check if spin_distr is provided and if so, will generate spin magnitudes and calculate chieff using these spins and the m1/m2 specified in the dataframes.
    """

    # --- Read in the models, parse the inference parameters
    if VERBOSE:
        print("\nReading models and applying transformations...\n")
    model_files = [dirpath + f for f in os.listdir(dirpath)]
    channels = list(h5py.File(model_files[0]).keys())

    # all models should be saved as hdf files living at dirpath with the channels as keys
    for mdl_file in os.listdir(dirpath):
        mdl_name = mdl_file.split('.')[0]
        models[mdl_name] = {}
        for channel in channels:
            inference_params = pd.DataFrame()
            df = pd.read_hdf(dirpath + mdl_file, key=channel)

            # check if params in the dataframe, otherwise perform transformations
            for param in params:
                if param not in df.columns:
                    # default transformations
                    if param in _DEFAULT_TRANSFORMS.keys():
                        df[param] = _DEFAULT_TRANSFORMS[param](df)
                    # chieff transformations
                    elif param=='chieff':
                        df['theta1'] = _DEFAULT_TRANSFORMS['theta1'](df)
                        df['theta2'] = _DEFAULT_TRANSFORMS['theta2'](df)
                        # check if spin magnitudes have been provided
                        if not {'a1','a2'}.issubset(df.columns):
                            if spin_distr in _VALID_SPIN_DISTR:
                                df['a1'],df['a2'] = _VALID_SPIN_DISTR[spin_distr](df)
                            else:
                                raise NameError("Spin magnitudes not provided and valid spin distribution was not specified, so can't generate effective spins!")

                        df['chieff'] = _to_chi_eff(df)
                    # otherwise, raise an error
                    else:
                        raise NameError("You specified the parameter {0:s} for inference, but it is not in your population data and you haven't written a transformation to calculate it!".format(param))

            # store model in models dict
            models[mdl_name][channel] = df


    # --- Now, construct KDE models
    if VERBOSE:
        print("Constructing model KDEs...\n")
    for mdl_name in models.keys():
        kde_models[mdl_name] = {}
        for channel in channels:
            df = models[mdl_name][channel]

            label = mdl_name + '_' + channel
            mdl = KDEModel.from_samples(label, df, params, weighting=weighting, Nsamps=1e4)
            kde_models[mdl_name][channel] = mdl

    return models, kde_models


