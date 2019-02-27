#!/usr/bin/env python3
__author__ = 'Feng Zhu'
__email__ = 'fengzhu@usc.edu'
__version__ = '0.2.2'


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import numpy as np
import pandas as pd
import os
import glob
from scipy.stats.mstats import mquantiles

def plot_gmt_ts(exp_dir, savefig_path=None, plot_vars=['gmt_ensemble', 'nhmt_ensemble', 'shmt_ensemble'],
        pannel_size=[10, 4], font_scale=1.5, hspace=0.5, ylim=[-1, 1]):
    ''' Plot timeseries

    Args:
        exp_dir (str): the path of the results directory that contains subdirs r0, r1, ...

    Returns:
        fig (figure): the output figure
    '''
    # load data
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    paths = sorted(glob.glob(os.path.join(exp_dir, 'r*')))
    filename = 'gmt_ensemble.npz'
    data = np.load(os.path.join(paths[0], filename))
    gmt_tmp = data['gmt_ensemble']
    nt = np.shape(gmt_tmp)[0]
    nEN = np.shape(gmt_tmp)[-1]
    nMC = len(paths)

    nvar = len(plot_vars)
    sns.set(style="darkgrid", font_scale=font_scale)
    fig = plt.figure(figsize=[pannel_size[0], pannel_size[1]*nvar])

    ax_title = {
            'gmt_ensemble': 'Global mean temperature',
            'shmt_ensemble': 'SH mean temperature',
            'nhmt_ensemble': 'NH mean temperature',
            }

    for plot_i, var in enumerate(plot_vars):
        gmt = np.ndarray((nt, nEN*nMC))
        for i, path in enumerate(paths):
            data = np.load(os.path.join(path, filename))
            gmt[:, nEN*i:nEN+nEN*i] = data[var]

        gmt_qs = mquantiles(gmt, [0.025, 0.25, 0.5, 0.75, 0.975], axis=-1)

        # plot
        gs = gridspec.GridSpec(nvar, 1)
        gs.update(wspace=0, hspace=hspace)

        to = np.arange(nt)
        ax_gmt = plt.subplot(gs[plot_i, 0])
        ax_gmt.plot(to, gmt_qs[:,2], '-', color=sns.xkcd_rgb['pale red'], alpha=1, label='median')
        ax_gmt.fill_between(to, gmt_qs[:,3], gmt_qs[:,1], color=sns.xkcd_rgb['pale red'], alpha=0.5, label='central 95%')
        ax_gmt.fill_between(to, gmt_qs[:,-1], gmt_qs[:,0], color=sns.xkcd_rgb['pale red'], alpha=0.1, label='from 2.5% to 97.5%')
        ax_gmt.set_title(ax_title[var])
        ax_gmt.set_ylabel('T anom. (K)')
        ax_gmt.set_xlabel('Year (AD)')
        ax_gmt.legend(loc='upper center', ncol=3, frameon=False)
        ax_gmt.set_ylim(ylim)

    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.close(fig)

    return fig
