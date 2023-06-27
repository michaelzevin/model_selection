import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from . import *
from .utils.transform import _DEFAULT_TRANSFORMS, _to_chieff, \
_uniform_spinmag, _isotropic_spinmag
from .Flowsclass_dev import FlowModel



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

def get_model_keys(path):
    alpha_val = '10'
    all_models = []
    models = []
    def find_submodels(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_models.append(name.rsplit('/', 1)[0])
            
    f = h5py.File(path, 'r')
    f.visititems(find_submodels)
    # get all unique models
    all_models = sorted(list(set(all_models)))
    f.close()

    # use only models with given alpha value
    for model in all_models:
        if 'alpha' in model:
            if 'alpha'+alpha_val in model:
                models.append('/'+model)
        else:
            models.append('/' + model)
    return(np.split(np.array(models), 5))

def get_model_keys_CE(path):
    all_models = []
    models = []
    def find_submodels(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_models.append(name.rsplit('/', 1)[0])
            
    f = h5py.File(path, 'r')
    f.visititems(find_submodels)
    # get all unique models
    all_models = sorted(list(set(all_models)))
    f.close()

    # use only models with given alpha value
    for model in all_models:
        if 'CE' in model:
            models.append('/'+model)
    return(np.split(np.array(models), 4))

def read_hdf5(path, channel):
    if channel=='CE':
        popsynth_outputs = {}
        models = np.asarray(get_model_keys_CE(path))
        for i in range(models.shape[0]):
            for j in range(models.shape[1]):
                popsynth_outputs[i,j]=pd.read_hdf(path, key=models[i,j])
    else:
        popsynth_outputs = {}
        models = np.asarray(get_model_keys(path))
        for i in range(models.shape[0]):
            for j in range(models.shape[1]):
                popsynth_outputs[i,j]=pd.read_hdf(path, key=models[i,j])
    return(popsynth_outputs)


def get_models(file_path, channels, params, spin_distr=None, sensitivity=None, normalize=False, detectable=False, useKDE=False, **kwargs):
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

    Parameters
    ----------
    file_path : str
        filepath to models_reduced.hdf5
    channels : list of str
        which channels to load models of, from CE, CHE, SMT, GC and NSC
    params : list of str
        which binary parameters to read from file, from mchirp, q, chieff, and z.
        fed to likelihood model
    useKDE : bool
        flag for whether to use KDEs or flows in inference

    Returns
    ----------
    deepest_models : list of str
        list of submodels to get likelihood models from, in format 'CE/chi00/alpha02'
    [kde_]models : dictionary? of KDEs
        dictionary of KDE models for each submodel
    """

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
    if channels:
        deepest_models_cut = []
        for chnl in channels:
            for mdl in deepest_models:
                if chnl+'/' in mdl:
                    deepest_models_cut.append(mdl)
        deepest_models = deepest_models_cut

    # Save all KDE models as pandas dataframes in dict structure

    # TO CHANGE -- get likelihood model. instead of KDEmodel call a parent class which has KDEs or FLows
    # TO CHANGE -- each flow model needs all submodels at once
        # while KDE model is instantiated for each submodel seperately
        # read all data, pass all to likelihood model?
        # then within this model either instantiate flows or each KDE model seperately

    if useKDE == True:
        kde_models = {}
        #tqdm shows progress meter
        for smdl in tqdm(deepest_models):
            smdl_list = smdl.split('/')
            current_level = kde_models
            for part in smdl_list:
                if part not in current_level:
                    if part == smdl_list[-1]:
                        # if we are on the last level, read in data and store kdes
                        df = pd.read_hdf(file_path, key=smdl)
                        label = '/'.join(smdl_list)
                        print('label')
                        print(label)
                        mdl = KDEModel.from_samples(label, df, params, sensitivity=sensitivity, normalize=normalize, detectable=detectable)
                        current_level[part] = mdl
                    else:
                        current_level[part] = {}

                current_level = current_level[part]
        return deepest_models, kde_models
    else:
        flow_models = {}
        
        for chnl in tqdm(channels):
            popsynth_outputs = read_hdf5(file_path, chnl)
            flow_models[chnl] = FlowModel.from_samples(chnl, popsynth_outputs, params)
        return flow_models
            

