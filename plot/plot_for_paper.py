import numpy as np
import pandas as pd
import os
import pdb
from functools import reduce
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from matplotlib import gridspec

from populations import *

cp = sns.color_palette("colorblind", 6)
_basepath, _ = os.path.split(os.path.realpath(__file__))
plt.style.use(_basepath+"/.MATPLOTLIB_RCPARAMS.sty")

_param_bounds = {"mchirp": (0,75), "q": (0,1), "chieff": (-1,1), "z": (0,2)}
_param_ticks = {"mchirp": [0,25,50,75], "q": [0,0.25,0.5,0.75,1], "chieff": [-1,-0.5,0,0.5,1], "z": [0,0.5,1.0,1.5,2.0]}
_pdf_bounds = {"mchirp": (0,0.09), "q": (0,32), "chieff": (0,13), "z": (0,4)}
_pdf_ticks = {"mchirp": [0.0,0.025,0.050,0.075], "q": [0,10,20,30], "chieff": [0,3,6,9,12], "z": (0,1,2,3,4)}
_labels_dict = {"mchirp": r"$\mathcal{M}_{\rm c}$ [$M_{\odot}$]", "q": r"$q$", \
"chieff": r"$\chi_{\rm eff}$", "z": r"$z$", "chi00": r"$\chi_\mathrm{b}=0.0$", \
"chi01": r"$\chi_\mathrm{b}=0.1$", "chi02": r"$\chi_\mathrm{b}=0.2$", \
"chi05": r"$\chi_\mathrm{b}=0.5$", "alpha02": r"$\alpha_\mathrm{CE}=0.2$", \
"alpha05": r"$\alpha_\mathrm{CE}=0.5$", "alpha10": r"$\alpha_\mathrm{CE}=1.0$", \
"alpha20": r"$\alpha_\mathrm{CE}=2.0$", "alpha50": r"$\alpha_\mathrm{CE}=5.0$", \
"CE": r"$\texttt{CE}$", "CHE": r"$\texttt{CHE}$", "GC": r"$\texttt{GC}$", \
"NSC": r"$\texttt{NSC}$", "SMT": r"$\texttt{SMT}$"}
_Nsamps = 100000
_marg_kde_bandwidth = 0.02

# --- Useful functions for accessing items in dictionary
def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)
def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def plot_1D_kdemodels(model_names, kde_models, params, observations, obsdata, model0, name=None, fixed_vals=[], plot_obs=False, plot_obs_samples=False):
    """
    Plots all the KDEs for each channel in each model, as well as the *true* model described by the input branching fraction.
    If more than one population parameter is being considered, specify in list 'fixed_vals' (e.g., ['alpha10']).
    """
    # filter to only get models for one population parameter
    models_to_plot = []
    if fixed_vals:
        for model in model_names:
            ctr = len(model.split('/'))-2
            for fval in fixed_vals:
                if fval in model.split('/'):
                    ctr -= 1
            if ctr==0:
                models_to_plot.append(model)
    models_to_plot.sort()

    channels = list(kde_models.keys())
    Nchannels = int(len(kde_models))
    Nparams = int(len(params))
    Nsbmdls = int(len(models_to_plot)/len(kde_models))

    fig, axs = plt.subplots(Nsbmdls, Nparams, figsize=(6*Nparams, 5*Nsbmdls))

    # loop over all models...
    print('   plotting population models...')
    for cidx, channel in enumerate(channels):
        channel_smdls = [x for x in models_to_plot if channel+'/' in x]
        for idx, model in enumerate(channel_smdls):
            kde = getFromDict(kde_models, model.split('/'))

            # if this kde is in model0, allocate array for samples
            if model0 and (kde.label == model0[channel].label):
                channel_model0_samples = np.zeros(shape=(int(kde.rel_frac*_Nsamps),Nparams))

            # loop over all parameters...
            for pidx, param in enumerate(params):
                if axs.ndim == 1:
                    ax = axs[idx]
                else:
                    ax = axs[idx,pidx]

                # marginalize the kde (this redoes the KDE in 1D)
                marg_kde = kde.marginalize([param], alpha=1, bandwidth=_marg_kde_bandwidth)

                # evaluate the marginalized kde over the param range
                eval_pts = np.linspace(*_param_bounds[param], 100)
                eval_pts = eval_pts.reshape(100,1,1)
                pdf = marg_kde(eval_pts)

                # if this model is in model0, sample the marginalized KDE
                if model0 and (kde.label == model0[channel].label):
                    channel_model0_samples[:,pidx] = marg_kde.sample(int(kde.rel_frac*_Nsamps)).flatten()

                # labels and legend
                I_am_legend = False
                if model0:
                    if (kde.label == model0[channel].label) and (pidx==(len(params)-1)):
                        label = _labels_dict[channel]
                        I_am_legend = True
                    else:
                        label=None
                else:
                    if idx==0 and pidx==(len(params)-1):
                        label = channel
                        I_am_legend = True
                    else:
                        label=None

                # plot the kde
                ax.plot(eval_pts.flatten(), pdf, color=cp[cidx], label=label)

                # Format plot
                if I_am_legend==True:
                    ax.legend(prop={'size':40}, loc='center', ncol=5, bbox_to_anchor=(-1.28,1.4))
                if cidx==Nchannels-1:
                    ax.set_xlim(*_param_bounds[param])
                    ax.set_xticks(_param_ticks[param])
                    ax.set_ylim(*_pdf_bounds[param])
                    ax.set_yticks(_pdf_ticks[param])
                    if idx==Nsbmdls-1:
                        ax.set_xlabel(_labels_dict[param], fontsize=35)
                        ax.set_xticklabels(_param_ticks[param])
                    else:
                        ax.set_xticklabels([])
                    if pidx==0:
                        ax.set_ylabel(_labels_dict[model.split('/')[1]]+"\n"+r"$p(\theta)$", fontsize=35, labelpad=8)
                    if idx==0:
                        ax.set_title(_labels_dict[param], fontsize=35)



        # append the draws from model0 from all channels
        if model0:
            if cidx==0:
                model0_samples = channel_model0_samples
            else:
                model0_samples = np.concatenate((model0_samples, channel_model0_samples))


    # Plot model0, obsdata, and formatting
    print('   plotting model0 and observations...')
    for idx in np.arange(Nsbmdls):
        for pidx, param in enumerate(params):
            if axs.ndim == 1:
                ax = axs[idx]
            else:
                ax = axs[idx,pidx]
            
            # construct combined KDE model and plot
            if model0:
                combined_samples = pd.DataFrame(model0_samples[:,pidx].flatten(), columns=[param])
                combined_kde = KDEModel.from_samples('combined_kde', combined_samples, [param], weighting=None, bandwidth=_marg_kde_bandwidth)
                eval_pts = np.linspace(*_param_bounds[param], 100)
                eval_pts = eval_pts.reshape(100,1,1)
                pdf = combined_kde(eval_pts)

                ax.plot(eval_pts.flatten(), pdf, color='k', linestyle='--')

            # plot the observations, if specified
            y_max = ax.get_ylim()[1]
            if plot_obs:
                for obs in observations:
                    # delta function observations
                    ax.axvline(obs[pidx], ymax=0.1, color='k', \
                                                alpha=0.4, zorder=-10)

            if plot_obs_samples:
                for obs in obsdata:
                    ax.axvline(np.median(obs[:,pidx]), ymax=0.1, \
                                        color='k', alpha=0.4, zorder=-20)
                    # construct KDE from observations
                    obs_samps = pd.DataFrame(obs[:,pidx], columns=[param])
                    obs_kde = KDEModel.from_samples('obs_kde', obs_samps, \
                                                   [param], weighting=None)
                    eval_pts = np.linspace(obs_samps.min(), \
                                                    obs_samps.max(), 100)
                    eval_pts = eval_pts.reshape(100,1,1)
                    pdf = obs_kde(eval_pts)

                    # scale down the pdf
                    pdf = 0.2 * pdf/(pdf.max()/pdf_max)

                    ax.fill_between(eval_pts.flatten(), \
                      y1=np.zeros_like(pdf), y2=pdf, color='k', alpha=0.05)

    plt.subplots_adjust(right=0.97, bottom=0.08)
    fname = '/Users/michaelzevin/research/model_selection/model_selection/paper/figures/pop_models.png'
    plt.savefig(fname)
    plt.close()
