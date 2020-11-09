"""
Plotting functions so we don't bog down the executable
"""

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

_param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,2)}
_labels_dict = {"mchirp": r"$\mathcal{M}_{\rm c}$ [M$_{\odot}$]", "q": r"q", \
"chieff": r"$\chi_{\rm eff}$", "z": r"$z$", "chi00": r"$\chi_\mathrm{b}=0.0$", \
"chi01": r"$\chi_\mathrm{b}=0.1$", "chi02": r"$\chi_\mathrm{b}=0.2$", \
"chi05": r"$\chi_\mathrm{b}=0.5$", "alpha02": r"$\alpha_\mathrm{CE}=0.2$", \
"alpha05": r"$\alpha_\mathrm{CE}=0.5$", "alpha10": r"$\alpha_\mathrm{CE}=1.0$", \
"alpha20": r"$\alpha_\mathrm{CE}=2.0$", "alpha50": r"$\alpha_\mathrm{CE}=5.0$"}
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
                marg_kde = kde.marginalize([param], bandwidth=_marg_kde_bandwidth)

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
                        label = channel+r" ($\beta$={0:0.2f})".format(kde.rel_frac)
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
                    ax.legend(prop={'size':30}, loc='center', bbox_to_anchor=(1.0,0.5))
                if cidx==Nchannels-1:
                    ax.set_xlim(*_param_bounds[param])
                    ax.set_ylim(bottom=0)#, top=pdf_max)
                    if idx==Nsbmdls-1:
                        ax.set_xlabel(_labels_dict[param], fontsize=40)
                    if pidx==0:
                        ax.set_ylabel(_labels_dict[model.split('/')[1]], fontsize=50)
                    if idx==0:
                        ax.set_title(_labels_dict[param], fontsize=40)



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
                    ax.axvline(obs[pidx], ymax=0.1, color='b', \
                                                alpha=0.4, zorder=-10)

            if plot_obs_samples:
                for obs in obsdata:
                    ax.axvline(np.median(obs[:,pidx]), ymax=0.1, \
                                        color='b', alpha=0.4, zorder=-20)
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
                      y1=np.zeros_like(pdf), y2=pdf, color='b', alpha=0.05)

    # Titles and saving
    if model0:
        model0_name = model0[channels[0]].label.split('/', 1)[1]
    else:
        model0_name='GW observations'
    if fixed_vals:
        plt.suptitle("Sampled model: {0:s}".format(model0_name)+'\n (fixed values: '+\
                            ','.join('{}'.format(x) for x in fixed_vals)+')', fontsize=55)
    else:
        plt.suptitle("Sampled model: {0:s}".format(model0_name), fontsize=55)
    if name:
        fname = 'marginalized_kdes_'+name+'.png'
    else:
        fname = 'marginalized_kdes.png'
    plt.savefig(fname)
    plt.close()




def plot_samples(samples, submodels_dict, model_names, channels, model0, name=None, hyper_idx=0, detectable_beta=False):
    """
    Plots the models that the chains are exploring, and histograms of the 
    branching fraction recovered for each model.

    :hyper_marg_idx: defines the index of the hyperparaeter in submodels_dict
    you wish to plot, marginalizing over the other parameters
    """

    Nhyper = len(submodels_dict)

    # setup the plots
    fig = plt.figure(figsize=(12,7))
    gs = gridspec.GridSpec(len(channels), 3, wspace=0.2, hspace=0.2)
    ax_chains, ax_margs = [], []
    for cidx, channel in enumerate(channels):
        ax_chains.append(fig.add_subplot(gs[cidx, :2]))
        ax_margs.append(fig.add_subplot(gs[cidx, -1]))

    # plot the chains moving in beta space, colored by their model
    for chain in samples:
        for midx, model in submodels_dict[hyper_idx].items():
            smdl_locs = np.argwhere(chain[:,hyper_idx]==midx)[:,0]
            steps = np.arange(chain.shape[0])
            for cidx, channel in enumerate(channels):
                ax_chains[cidx].scatter(steps[smdl_locs], \
                    chain[smdl_locs,cidx+Nhyper], color=cp[midx], s=0.5, alpha=0.2)

    # plot the histograms on beta for each model
    # compactify all the chains in samples
    samples_allchains = np.reshape(samples, (samples.shape[0]*samples.shape[1], samples.shape[2]))
    basemdl_samps = len(np.argwhere(samples_allchains[:,hyper_idx]==0).flatten())
    h_max = 0
    for midx, model in submodels_dict[hyper_idx].items():
        smdl_locs = np.argwhere(samples_allchains[:,hyper_idx]==midx).flatten()
        mdl_samps = len(smdl_locs)
        if basemdl_samps > 0:
            BF = float(mdl_samps)/basemdl_samps
        else:
            BF = float(mdl_samps)
        for cidx, channel in enumerate(channels):
            h, bins, _ = ax_margs[cidx].hist(samples_allchains[smdl_locs, cidx+Nhyper], \
                orientation='horizontal', histtype='step', color=cp[midx], bins=50, \
                alpha=0.7, label=model+', BF={0:0.1e}'.format(BF))
            h_max = h.max() if h.max() > h_max else h_max


    # format plot
    for cidx, (channel, ax_chain, ax_marg) in enumerate(zip(channels, \
                                                ax_chains, ax_margs)):

        # plot the injected value
        if model0:
            ax_chain.axhline(model0[channel].rel_frac, color='k', \
                    linestyle='--', alpha=0.7)
            ax_marg.axhline(model0[channel].rel_frac, color='k', \
                    linestyle='--', alpha=0.7)

        # tick labels
        if cidx != len(channels)-1:
            ax_chain.set_xticklabels([])
            ax_marg.set_xticklabels([])
        ax_chain.set_yticks([0,0.5,1.0])
        ax_marg.set_yticklabels([])
        ax_chain.tick_params(axis='both', labelsize=20)
        ax_marg.tick_params(axis='both', labelsize=20)

        # legend
        if cidx == 0:
            ax_marg.legend(loc='center', bbox_to_anchor=[1.0,1.0], prop={'size':10})

        if cidx == len(channels)-1:
            ax_chain.set_xlabel('Step', fontsize=30)
            ax_marg.set_xlabel(r"p($\beta$)", fontsize=30)

        ax_chain.set_ylabel(r"$\beta_{%s}$" % format(channel), fontsize=30)
        ax_chain.set_xlim(0,samples.shape[1])
        ax_chain.set_ylim(0,1)
        ax_marg.set_xlim(0,h_max+10)
        ax_marg.set_ylim(0,1)


    # title
    if model0:
        # find the deepest model
        channel_depth = 0
        for channel in channels:
            if len(model0[channel].label.split('/')) > channel_depth:
                channel_depth = len(model0[channel].label.split('/'))
                deepest_channel = channel
        model0_name = model0[deepest_channel].label.split('/', 1)[1]
    else:
        model0_name='GW observations'
    plt.suptitle("True model: {0:s}".format(model0_name), fontsize=40)
    fname = 'samples'
    if detectable_beta==True:
        fname = 'samples_detectable'
    elif detectable_beta==False:
        fname = 'samples_underlying'
    if name:
        fname = fname+'_hyperidx'+str(hyper_idx)+'_'+name+'.png'
    else:
        fname = fname+'_hyperidx'+str(hyper_idx)+'.png'
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(fname)
    plt.close()




