import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import glob
from scipy.stats.mstats import mquantiles
import pickle

from .. import LMRt


def load_gmt_from_jobs(exp_dir, qs=[0.025, 0.25, 0.5, 0.75, 0.975], var='gmt_ensemble'):
    # load data
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    paths = sorted(glob.glob(os.path.join(exp_dir, 'job_r*')))
    with open(paths[0], 'rb') as f:
        job_cfg, job_da = pickle.load(f)

    gmt_tmp = job_da.gmt_ens_save
    nt = np.shape(gmt_tmp)[0]
    nEN = np.shape(gmt_tmp)[-1]
    nMC = len(paths)

    gmt = np.ndarray((nt, nEN*nMC))
    for i, path in enumerate(paths):
        with open(path, 'rb') as f:
            job_cfg, job_da = pickle.load(f)

        job_gmt = {
            'gmt_ensemble': job_da.gmt_ens_save,
            'nhmt_ensemble': job_da.nhmt_ens_save,
            'shmt_ensemble': job_da.shmt_ens_save,
        }
        gmt[:, nEN*i:nEN+nEN*i] = job_gmt[var]

    gmt_qs = mquantiles(gmt, qs, axis=-1)
    return gmt_qs


def plot_gmt_vs_inst(exp_dir, inst_filesdict, qs=[0.025, 0.25, 0.5, 0.75, 0.975], var='gmt_ensemble'):
    gmt_qs = load_jobs(exp_dir, qs=qs, var=var)




def plot_gmt_ts(exp_dir, savefig_path=None, plot_vars=['gmt_ensemble', 'nhmt_ensemble', 'shmt_ensemble'],
        qs=[0.025, 0.25, 0.5, 0.75, 0.975], pannel_size=[10, 4], font_scale=1.5, hspace=0.5, ylim=[-1, 1],
        plot_prior=False, prior_var_name='tas_sfc_Amon'):
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

        gmt_qs = mquantiles(gmt, qs, axis=-1)

        # plot
        gs = gridspec.GridSpec(nvar, 1)
        gs.update(wspace=0, hspace=hspace)

        to = np.arange(nt)
        ax = plt.subplot(gs[plot_i, 0])
        if qs[2] == 0.5:
            label='median'
        else:
            label='{}%'.format(qs[2]*100)

        ax.plot(to, gmt_qs[:,2], '-', color=sns.xkcd_rgb['pale red'], alpha=1, label='{}'.format(label))
        ax.fill_between(to, gmt_qs[:,-2], gmt_qs[:,1], color=sns.xkcd_rgb['pale red'], alpha=0.5,
                label='{}% to {}%'.format(qs[1]*100, qs[-2]*100))
        ax.fill_between(to, gmt_qs[:,-1], gmt_qs[:,0], color=sns.xkcd_rgb['pale red'], alpha=0.1,
                label='{}% to {}%'.format(qs[0]*100, qs[-1]*100))
        ax.set_title(ax_title[var])
        ax.set_ylabel('T anom. (K)')
        ax.set_xlabel('Year (AD)')
        ax.legend(loc='upper center', ncol=3, frameon=False)
        ax.set_ylim(ylim)

        if plot_prior:
            prior_gmt = np.zeros([nMC,nEN,nt])
            prior_nhmt = np.zeros([nMC,nEN,nt])
            prior_shmt = np.zeros([nMC,nEN,nt])
            for citer, path in enumerate(paths):
                data = np.load(os.path.join(path, 'Xb_one.npz'))
                Xb_one = data['Xb_one']
                Xb_one_coords = data['Xb_one_coords']
                state_info = data['state_info'].item()
                posbeg = state_info[prior_var_name]['pos'][0]
                posend = state_info[prior_var_name]['pos'][1]
                tas_prior = Xb_one[posbeg:posend+1, :]
                tas_coords = Xb_one_coords[posbeg:posend+1, :]
                nlat, nlon = state_info[prior_var_name]['spacedims']
                lat_lalo = tas_coords[:, 0].reshape(nlat, nlon)
                nstate, nens = tas_prior.shape
                tas_lalo = tas_prior.transpose().reshape(nens, nlat, nlon)
                [gmt,nhmt,shmt] = LMRt.utils.global_hemispheric_means(tas_lalo, lat_lalo[:, 0])

                prior_gmt[citer,:,:]  = np.repeat(gmt[:,np.newaxis],nt,1)
                prior_nhmt[citer,:,:] = np.repeat(nhmt[:,np.newaxis],nt,1)
                prior_shmt[citer,:,:] = np.repeat(shmt[:,np.newaxis],nt,1)

            if var == 'gmt_ensemble':
                gmtp = prior_gmt.transpose(2,0,1).reshape(nt,-1)
            elif var == 'nhmt_ensemble':
                gmtp = prior_nhmt.transpose(2,0,1).reshape(nt,-1)
            elif var == 'shmt_ensemble':
                gmtp = prior_shmt.transpose(2,0,1).reshape(nt,-1)

            gmtp_qs = mquantiles(gmtp, qs, axis=-1)

            ax.plot(to, gmtp_qs[:,2], '-', color=sns.xkcd_rgb['grey'], alpha=1)
            ax.fill_between(to, gmtp_qs[:,3], gmtp_qs[:,1], color=sns.xkcd_rgb['grey'], alpha=0.5)
            ax.fill_between(to, gmtp_qs[:,-1], gmtp_qs[:,0], color=sns.xkcd_rgb['grey'], alpha=0.1)

    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_gmt_ts_from_jobs(exp_dir, savefig_path=None, plot_vars=['gmt_ensemble', 'nhmt_ensemble', 'shmt_ensemble'],
                          qs=[0.025, 0.25, 0.5, 0.75, 0.975], pannel_size=[10, 4],
                          font_scale=1.5, hspace=0.5, ylim=[-1, 1],
                          plot_prior=False, prior_var_name='tas_sfc_Amon'):
    ''' Plot timeseries

    Args:
        exp_dir (str): the path of the results directory that contains subdirs r0, r1, ...

    Returns:
        fig (figure): the output figure
    '''
    # load data
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    paths = sorted(glob.glob(os.path.join(exp_dir, 'job_r*')))
    with open(paths[0], 'rb') as f:
        job_cfg, job_da = pickle.load(f)

    gmt_tmp = job_da.gmt_ens_save
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
            with open(path, 'rb') as f:
                job_cfg, job_da = pickle.load(f)

            job_gmt = {
                'gmt_ensemble': job_da.gmt_ens_save,
                'nhmt_ensemble': job_da.nhmt_ens_save,
                'shmt_ensemble': job_da.shmt_ens_save,
            }

            gmt[:, nEN*i:nEN+nEN*i] = job_gmt[var]

        gmt_qs = mquantiles(gmt, qs, axis=-1)

        # plot
        gs = gridspec.GridSpec(nvar, 1)
        gs.update(wspace=0, hspace=hspace)

        to = np.arange(nt)
        ax = plt.subplot(gs[plot_i, 0])
        if qs[2] == 0.5:
            label='median'
        else:
            label='{}%'.format(qs[2]*100)

        ax.plot(to, gmt_qs[:,2], '-', color=sns.xkcd_rgb['pale red'], alpha=1, label='{}'.format(label))
        ax.fill_between(to, gmt_qs[:,-2], gmt_qs[:,1], color=sns.xkcd_rgb['pale red'], alpha=0.5,
                label='{}% to {}%'.format(qs[1]*100, qs[-2]*100))
        ax.fill_between(to, gmt_qs[:,-1], gmt_qs[:,0], color=sns.xkcd_rgb['pale red'], alpha=0.1,
                label='{}% to {}%'.format(qs[0]*100, qs[-1]*100))
        ax.set_title(ax_title[var])
        ax.set_ylabel('T anom. (K)')
        ax.set_xlabel('Year (AD)')
        ax.legend(loc='upper center', ncol=3, frameon=False)
        ax.set_ylim(ylim)

    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_vslite_params(lat_obs, lon_obs, T1, T2, M1, M2,
                       T1_ticks=None, T2_ticks=None, M1_ticks=None, M2_ticks=None, save_path=None):
    sns.set(style='ticks', font_scale=2)
    fig = plt.figure(figsize=[18, 12])
    gs = mpl.gridspec.GridSpec(2, 2)
    gs.update(wspace=0.1, hspace=0.1)

    map_proj = ccrs.Robinson()
    # map_proj = ccrs.PlateCarree()
    # map_proj = ccrs.Mercator()
    # map_proj = ccrs.Miller()
    # map_proj = ccrs.Mollweide()
    # map_proj = ccrs.EqualEarth()

    pad = 0.05
    fraction = 0.05

    # T1
    ax1 = plt.subplot(gs[0, 0], projection=map_proj)
    ax1.set_title('T1')
    ax1.set_global()
    ax1.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax1.gridlines(edgecolor='gray', linestyle=':')
    z = T1
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    im = ax1.scatter(
        lon_obs, lat_obs, marker='o', norm=norm,
        c=z, cmap='Reds', s=20, transform=ccrs.Geodetic()
    )
    cbar1 = fig.colorbar(im, ax=ax1, orientation='horizontal', pad=pad, fraction=fraction)
    if T1_ticks:
        cbar1.set_ticks(T1_ticks)

    # T2
    ax2 = plt.subplot(gs[0, 1], projection=map_proj)
    ax2.set_title('T2')
    ax2.set_global()
    ax2.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax2.gridlines(edgecolor='gray', linestyle=':')
    z = T2
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    im = ax2.scatter(
        lon_obs, lat_obs, marker='o', norm=norm,
        c=z, cmap='Reds', s=20, transform=ccrs.Geodetic()
    )
    cbar2 = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=pad, fraction=fraction)
    if T2_ticks:
        cbar2.set_ticks(T2_ticks)

    # M1
    ax3 = plt.subplot(gs[1, 0], projection=map_proj)
    ax3.set_title('M1')
    ax3.set_global()
    ax3.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax3.gridlines(edgecolor='gray', linestyle=':')
    z = M1
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    im = ax3.scatter(
        lon_obs, lat_obs, marker='o', norm=norm,
        c=z, cmap='Blues', s=20, transform=ccrs.Geodetic()
    )
    cbar3 = fig.colorbar(im, ax=ax3, orientation='horizontal', pad=pad, fraction=fraction)
    if M1_ticks:
        cbar3.set_ticks(M1_ticks)

    # M2
    ax4 = plt.subplot(gs[1, 1], projection=map_proj)
    ax4.set_title('M2')
    ax4.set_global()
    ax4.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax4.gridlines(edgecolor='gray', linestyle=':')
    z = M2
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    im = ax4.scatter(
        lon_obs, lat_obs, marker='o', norm=norm,
        c=z, cmap='Blues', s=20, transform=ccrs.Geodetic()
    )
    cbar4 = fig.colorbar(im, ax=ax4, orientation='horizontal', pad=pad, fraction=fraction)
    if M2_ticks:
        cbar4.set_ticks(M2_ticks)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig
