"""
Plotting functions so we don't bog down the executable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from matplotlib import gridspec

from . import *

colors = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00']

params = {
        # latex
        'text.usetex': False,

        # fonts
        'font.family': 'serif',

        # figure and axes
        'figure.figsize': (12, 8),
        'figure.titlesize': 35,
        'axes.grid': True,
        'axes.titlesize':35,
        #'axes.labelweight': 'bold',
        'axes.labelsize': 25,

        # tick markers
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'xtick.major.size': 10.0,
        'ytick.major.size': 10.0,
        'xtick.minor.size': 3.0,
        'ytick.minor.size': 3.0,

        # legend
        'legend.fontsize': 20,
        'legend.frameon': True,
        'legend.framealpha': 0.5,

        # colors
        'image.cmap': 'viridis',

        # saving figures
        'savefig.dpi': 150
        }


_param_bounds = {"mchirp": (0,70), "q": (0,1), "chieff": (-1,1)}
_labels_dict = {"mchirp": r"$\mathcal{M}_{\rm c}$ [M$_{\odot}$]", "q": r"q", "chieff": r"$\chi_{\rm eff}$"}
_Nsamps = 10000

def plot_1D_kdemodels(kde_models, params, obsdata, model0, name=None, plot_obs=False, plot_obs_samples=False):
    """
    Plots all the KDEs for each channel in each model, as well as the *true* model described by the input branching fraction.
    """
    fig, axs = plt.subplots(len(kde_models), len(params), figsize=(5*len(kde_models), 6*len(params)))

    model_names = list(kde_models.keys())
    model_names.sort()

    # loop over all models...
    for idx, model in enumerate(model_names):

        # loop over all parameters...
        for pidx, param in enumerate(params):
            ax = axs[idx,pidx]

            pdf_max = 0
            # loop over all channels...
            for cidx, channel in enumerate(kde_models[model]):
                kde = kde_models[model][channel]

                # marginalize the kde
                marg_kde = kde.marginalize([param])

                # evaluate the marginalized kde over the param range
                eval_pts = np.linspace(*_param_bounds[param], 100)
                eval_pts = eval_pts.reshape(100,1,1)
                pdf = marg_kde(eval_pts)
                pdf_max = pdf.max() if pdf.max() > pdf_max else pdf_max

                # sample the marginalized KDE
                if model0:
                    if cidx==0:
                        combined_samples = marg_kde.sample(int(kde._rel_frac*_Nsamps))
                    else:
                        combined_samples = np.concatenate((combined_samples, marg_kde.sample(int(kde._rel_frac*_Nsamps))))

                # legend label
                if not model0:
                    label = channel
                else:
                    label = channel+r" ($\beta$={0:0.1f})".format(kde._rel_frac)

                # plot the kde
                ax.plot(eval_pts.flatten(), pdf, color=colors[cidx], label=label)

                # format plot
                ax.set_xlim(*_param_bounds[param])
                ax.set_ylim(0, pdf_max)

                if idx==len(kde_models)-1:
                    ax.set_xlabel(_labels_dict[param], fontsize=20)
                if pidx==0:
                    ax.set_ylabel(model, fontsize=20)
                if idx==0 and pidx==0:
                    ax.legend(prop={'size':20})
                if idx==0:
                    ax.set_title(_labels_dict[param], fontsize=25)

            # construct combined KDE model and plot
            if model0:
                combined_samples = pd.DataFrame(combined_samples.flatten(), columns=[param])
                combined_kde = KDEModel.from_samples('combined_kde', combined_samples, [param], weighting=None)
                eval_pts = np.linspace(*_param_bounds[param], 100)
                eval_pts = eval_pts.reshape(100,1,1)
                pdf = combined_kde(eval_pts)

                ax.plot(eval_pts.flatten(), pdf, color='k', linestyle='--')

            # plot the observations, if specified
            if plot_obs:
                y_max = ax.get_ylim()[1]
                for obs in obsdata:
                    if obs.shape[0] == 1:
                        # delta function observations
                        ax.axvline(obs[:,pidx], ymax=0.2, color='b', alpha=0.5, zorder=10)
                    elif plot_obs_samples:
                        ax.axvline(np.median(obs[:,pidx]), ymax=0.2, color='b', alpha=0.5, zorder=10)
                        # construct KDE from observations
                        obs_samps = pd.DataFrame(obs[:,pidx], columns=[param])
                        obs_kde = KDEModel.from_samples('obs_kde', obs_samps, [param], weighting=None)
                        eval_pts = np.linspace(obs_samps.min(), obs_samps.max(), 100)
                        eval_pts = eval_pts.reshape(100,1,1)
                        pdf = obs_kde(eval_pts)

                        # scale down the pdf
                        pdf = 0.2 * pdf/(pdf.max()/pdf_max)

                        ax.fill_between(eval_pts.flatten(), y1=np.zeros_like(pdf), y2=pdf, color='b', alpha=0.05)
                    else:
                        ax.axvline(np.median(obs[:,pidx]), ymax=0.2, color='b', alpha=0.5, zorder=10)

    if model0:
        model0_name = model0[list(model0.keys())[0]].label.split('_')[0]
    else:
        model0_name='GW observations'
    plt.suptitle("Sampled model: {0:s}".format(model0_name), fontsize=40)
    if name:
        fname = 'marginalized_kdes_'+name+'.png'
    else:
        fname = 'marginalized_kdes.png'
    plt.savefig(fname)
    plt.close()




def plot_samples(samples, model_names, channels, model0, name=None):
    """
    Plots the models that the chains are exploring, and histograms of the branching fraction recovered for each model.
    """
    # take the low-T chain
    samples = samples[0]

    # take the floor of the model samples
    samples[:,:,0] = np.floor(samples[:,:,0])

    # setup the plots
    fig = plt.figure(figsize=(12,7))
    gs = gridspec.GridSpec(len(channels),3, wspace=0.1, hspace=0.1)
    ax_chains, ax_margs = [], []
    for cidx, channel in enumerate(channels):
        ax_chains.append(fig.add_subplot(gs[cidx, :2]))
        ax_margs.append(fig.add_subplot(gs[cidx, -1]))

    # plot the chains moving in beta space, colored by their model
    for chain in samples:
        for midx, model in enumerate(model_names):
            smdl_locs = np.argwhere(chain[:,0]==midx)
            steps = np.arange(len(chain))
            for cidx, channel in enumerate(channels):
                ax_chains[cidx].scatter(steps[smdl_locs], chain[smdl_locs,cidx+1], color=colors[midx], s=0.5, alpha=0.2)

    # plot the histograms on beta for each model
    basemdl_samps = len(np.argwhere(samples[:,:,0]==0))
    for midx, model in enumerate(model_names):
        smdl_locs = np.argwhere(samples[:,:,0]==midx)
        mdl_samps = len(smdl_locs)
        if basemdl_samps > 0:
            BF = float(mdl_samps)/basemdl_samps
        else:
            BF = float(mdl_samps)
        for cidx, channel in enumerate(channels):
            ax_margs[cidx].hist(samples[smdl_locs[:,0], smdl_locs[:,1], cidx+1], orientation='horizontal', histtype='step', color=colors[midx], bins=50, alpha=0.7, label=model+', BF={0:0.1e}'.format(BF))


    # format plot
    for cidx, (channel, ax_chain, ax_marg) in enumerate(zip(channels, ax_chains, ax_margs)):

        # plot the injected value
        if model0:
            ax_chain.axhline(model0[channel]._rel_frac, color='k', linestyle='--', alpha=0.7)
            ax_marg.axhline(model0[channel]._rel_frac, color='k', linestyle='--', alpha=0.7)

        # legend
        if cidx == 0:
            ax_marg.legend(loc='upper right', prop={'size':12})

        if cidx == len(channels)-1:
            ax_chain.set_xlabel('Step', fontsize=20)
            ax_marg.set_xlabel(r"$\beta$", fontsize=20)

        ax_chain.set_ylabel(r"$\beta_{%s}$" % format(channel), fontsize=20)
        ax_chain.set_xlim(0,samples.shape[1])
        ax_chain.set_ylim(0,1)
        ax_marg.set_ylim(0,1)
        ax_marg.set_yticks([])


    # title
    if model0:
        model0_name = model0[list(model0.keys())[0]].label.split('_')[0]
    else:
        model0_name='GW observations'
    plt.suptitle("True model: {0:s}".format(model0_name), fontsize=30)
    if name:
        fname = 'samples_'+name+'.png'
    else:
        fname = 'samples.png'
    plt.savefig(fname)
    plt.close()




