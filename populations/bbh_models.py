import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from . import *
from .utils.transform import _DEFAULT_TRANSFORMS, _to_chieff, \
_uniform_spinmag, _isotropic_spinmag



_VALID_SPIN_DISTR = {
    # Uniform - |a| ~ uniform distribution in 0, 1
    "uniform": _uniform_spinmag,
    # Isotropic - |a| ~ a^2 distribution in 0, 1
    "isotropic": _isotropic_spinmag
}


def get_params(df, params):
    inference_params = pd.DataFrame()

    # check if :params: in the dataframe, otherwise perform transformations
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

    return df


def get_models(file_path, specific_channels, params, spin_distr=None, sensitivity=None, normalize=False, verbose=False, **kwargs):
    """
    Call this to get all the models and submodels, as well
    as KDEs of these models, packed inside of dictionaries labelled in the
    dict structure models[channel][smdl]. Will first look for :params: as
    series in the dataframe. If they are not present, it will try to construct
    these parameters if the valid transformations are present in transforms.py.

    If chieff is one of the :params: for inference and spin magnitudes are not
    provided, this function will first check if :spin_distr: is provided and
    if so, will generate spin magnitudes and calculate chieff using these
    spins and the m1/m2 specified in the dataframes.
    """

    # --- Read in the models, parse the inference parameters
    if verbose:
        print("\nReading models and applying transformations...\n")

    # all models should be saved in 'file_path' in a hierarchical structure, with the channel being the top group
    f = h5py.File(file_path, "r")
    # find all the deepest models to set up dictionary
    deepest_models = []
    def find_submodels(name, obj):
        if isinstance(obj, h5py.Dataset):
            deepest_models.append(name.rsplit('/', 1)[0])
    f.visititems(find_submodels)
    f.close()
    deepest_models = sorted(list(set(deepest_models)))
    # if only using specific formation channels, remove other models
    if specific_channels:
        deepest_models_cut = []
        for chnl in specific_channels:
            for mdl in deepest_models:
                if chnl+'/' in mdl:
                    deepest_models_cut.append(mdl)
        deepest_models = deepest_models_cut

    # save all models as pandas dataframes in dict structure
    models = {}
    for smdl in tqdm(deepest_models):
        smdl_list = smdl.split('/')
        current_level = models
        for part in smdl_list:
            if part not in current_level:
                if part == smdl_list[-1]:
                    # if we are on the last level, save dataframe
                    df = pd.read_hdf(file_path, key=smdl)
                    df = get_params(df, params)
                    current_level[part] = df
                else:
                    current_level[part] = {}

            current_level = current_level[part]

    # --- Now, construct KDE models
    if verbose:
        print("\nConstructing KDEs for populations...\n")
    kde_models = {}
    for smdl in tqdm(deepest_models):
        smdl_list = smdl.split('/')
        current_level = models
        current_level_kde = kde_models
        for part in smdl_list:
            if part not in current_level_kde:
                if part == smdl_list[-1]:
                    # if we are on the last level, save kdes
                    df = current_level[part]
                    label = '/'.join(smdl_list)
                    mdl = KDEModel.from_samples(label, df, params, sensitivity=sensitivity, normalize=normalize)
                    current_level_kde[part] = mdl
                else:
                    current_level_kde[part] = {}

            current_level = current_level[part]
            current_level_kde = current_level_kde[part]

    return deepest_models, models, kde_models

