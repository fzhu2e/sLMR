import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import numpy as np
import pandas as pd
import os
import glob
from scipy.stats.mstats import mquantiles
import pickle


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
                [gmt,nhmt,shmt] = global_hemispheric_means(tas_lalo, lat_lalo[:, 0])

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


def global_hemispheric_means(field,lat):

    """
    Adapted from LMR_utils.py by Greg Hakim & Robert Tardif | U. of Washington

     compute global and hemispheric mean valuee for all times in the input (i.e. field) array
     input:  field[ntime,nlat,nlon] or field[nlat,nlon]
             lat[nlat,nlon] in degrees

     output: gm : global mean of "field"
            nhm : northern hemispheric mean of "field"
            shm : southern hemispheric mean of "field"
    """

    # Originator: Greg Hakim
    #             University of Washington
    #             August 2015
    #
    # Modifications:
    #           - Modified to handle presence of missing values (nan) in arrays
    #             in calculation of spatial averages [ R. Tardif, November 2015 ]
    #           - Enhanced flexibility in the handling of missing values
    #             [ R. Tardif, Aug. 2017 ]

    # set number of times, lats, lons; array indices for lat and lon
    if len(np.shape(field)) == 3: # time is a dimension
        ntime,nlat,nlon = np.shape(field)
        lati = 1
        loni = 2
    else: # only spatial dims
        ntime = 1
        nlat,nlon = np.shape(field)
        field = field[None,:] # add time dim of size 1 for consistent array dims
        lati = 1
        loni = 2

    # latitude weighting
    lat_weight = np.cos(np.deg2rad(lat))
    tmp = np.ones([nlon,nlat])
    W = np.multiply(lat_weight,tmp).T

    # define hemispheres
    eqind = nlat//2

    if lat[0] > 0:
        # data has NH -> SH format
        W_NH = W[0:eqind+1]
        field_NH = field[:,0:eqind+1,:]
        W_SH = W[eqind+1:]
        field_SH = field[:,eqind+1:,:]
    else:
        # data has SH -> NH format
        W_NH = W[eqind:]
        field_NH = field[:,eqind:,:]
        W_SH = W[0:eqind]
        field_SH = field[:,0:eqind,:]

    gm  = np.zeros(ntime)
    nhm = np.zeros(ntime)
    shm = np.zeros(ntime)

    # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation)
    # Get arrays indices of valid values
    indok    = np.isfinite(field)
    indok_nh = np.isfinite(field_NH)
    indok_sh = np.isfinite(field_SH)
    for t in range(ntime):
        if lati == 0:
            # Global
            gm[t]  = np.average(field[indok],weights=W[indok])
            # NH
            nhm[t] = np.average(field_NH[indok_nh],weights=W_NH[indok_nh])
            # SH
            shm[t] = np.average(field_SH[indok_sh],weights=W_SH[indok_sh])
        else:
            # Global
            indok_2d    = indok[t,:,:]
            if indok_2d.any():
                field_2d    = np.squeeze(field[t,:,:])
                gm[t]       = np.average(field_2d[indok_2d],weights=W[indok_2d])
            else:
                gm[t] = np.nan
            # NH
            indok_nh_2d = indok_nh[t,:,:]
            if indok_nh_2d.any():
                field_nh_2d = np.squeeze(field_NH[t,:,:])
                nhm[t]      = np.average(field_nh_2d[indok_nh_2d],weights=W_NH[indok_nh_2d])
            else:
                nhm[t] = np.nan
            # SH
            indok_sh_2d = indok_sh[t,:,:]
            if indok_sh_2d.any():
                field_sh_2d = np.squeeze(field_SH[t,:,:])
                shm[t]      = np.average(field_sh_2d[indok_sh_2d],weights=W_SH[indok_sh_2d])
            else:
                shm[t] = np.nan

# original code (keep for now...)
#    for t in xrange(ntime):
#        if lati == 0:
#            gm[t]  = np.sum(np.multiply(W,field))/(np.sum(np.sum(W)))
#            nhm[t] = np.sum(np.multiply(W_NH,field_NH))/(np.sum(np.sum(W_NH)))
#            shm[t] = np.sum(np.multiply(W_SH,field_SH))/(np.sum(np.sum(W_SH)))
#        else:
#            gm[t]  = np.sum(np.multiply(W,field[t,:,:]))/(np.sum(np.sum(W)))
#            nhm[t] = np.sum(np.multiply(W_NH,field_NH[t,:,:]))/(np.sum(np.sum(W_NH)))
#            shm[t] = np.sum(np.multiply(W_SH,field_SH[t,:,:]))/(np.sum(np.sum(W_SH)))


    return gm,nhm,shm
