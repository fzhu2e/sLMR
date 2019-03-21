''' Some useful functions
'''
import numpy as np
import collections
from copy import deepcopy
from time import time
import functools
import xarray as xr
from datetime import datetime
import random
import pandas as pd


def timeit(func):
    ''' Decorator: print the lapse time running a function
    '''
    @functools.wraps(func)
    def decorated_func(*args, **kwargs):
        ts = time()
        res = func(*args, **kwargs)
        te = time()
        print(f'{func.__name__}: {(te-ts)*1e3:2.2f} ms')
        return res
    return decorated_func


def update_nested_dict(d, other):
    ''' Ref: https://code.i-harness.com/en/q/3154af
    '''
    for k, v in other.items():
        d_v = d.get(k)
        if isinstance(v, collections.Mapping) and isinstance(d_v, collections.Mapping):
            update_nested_dict(d_v, v)
        else:
            d[k] = deepcopy(v)
        return d


def load_netcdf(filepath, verbose=False):
    ''' Load the model output in .nc
        Timeseries will be annualized and anomaly will be calculated.
    '''
    def determine_vartype(spacecoords):
        vartypes = {
            0: '0D:time series',
            1: '2D:horizontal',
            2: '2D:meridional_vertical',
        }
        n_spacecoords = len(spacecoords)
        if n_spacecoords == 0:
            type_ind = 0
        elif n_spacecoords in [2, 3]:
            if 'lat' in spacecoords and 'lon' in spacecoords:
                type_ind = 1
            elif 'lat' in spacecoords and 'lev' in spacecoords:
                type_ind = 2
        else:
            raise SystemExit('ERROR: Fail to handle dimensions.')

        return vartypes[type_ind]

    def make2D(lat, lon):
        nd = len(np.shape(lat))

        if nd == 1:
            nlat = np.size(lat)
            nlon = np.size(lon)
            lat2D = np.repeat(lat, nlon).reshape(nlat, nlon)
            lon2D = np.repeat(lon, nlat).reshape(nlon, nlat).T
            return lat2D, lon2D

        elif nd == 2:
            print('Input lat/lon already 2-D!')
            return lat, lon

        else:
            raise SystemExit('ERROR: Cannot handle the dimensions for lat/lon!')

        pass

    datadict = {}
    if verbose:
        print(f'Reading file: {filepath}')
    ds = xr.open_dataset(filepath)
    ds_ann = ds.groupby('time.year').mean('time')
    time_yrs = np.asarray(
        [datetime(y, 1, 1, 0, 0) for y in np.asarray(ds_ann['year'])]
    )
    lat = np.asarray(ds_ann['lat'])
    lon = np.asarray(ds_ann['lon'])
    lat2D, lon2D = make2D(lat, lon)

    dim_set = set(ds_ann.dims)
    var_set = set(ds_ann.variables)
    var = var_set.difference(dim_set)

    for v in var:
        d = {}

        dims = set(ds_ann[v].dims)
        spacecoords = dims.difference(['year'])
        vartype = determine_vartype(spacecoords)

        climo_xr = ds_ann[v].mean(dim='year')
        climo = np.asarray(climo_xr)
        value = np.asarray(ds_ann[v] - climo_xr)
        if verbose:
            print('Anomalies provided as the prior: Removing the temporal mean (for every gridpoint)...')
            print(f'{v}: Global(monthly): mean={np.nanmean(value)}, std-dev={np.nanstd(value)}')

        for dim in dims:
            if dim == 'time':
                d[dim] = time_yrs
            elif dim == 'lat':
                d[dim] = lat2D
            elif dim == 'lon':
                d[dim] = lon2D

        d['vartype'] = vartype
        d['spacecoords'] = spacecoords
        d['years'] = time_yrs
        d['climo'] = climo
        d['value'] = value

        datadict[f'{v}_sfc_Amon'] = d

    return datadict


def populate_ensemble(datadict, cfg, seed, verbose=False):
    ''' Populate the prior ensemble from gridded model/analysis data
    '''
    state_vect_info = {}
    Nx = 0
    timedim = []
    for var in datadict.keys():
        vartype = datadict[var]['vartype']
        dct = {}
        timedim.append(len(datadict[var]['years']))
        spacecoords = datadict[var]['spacecoords']
        dim1, dim2 = spacecoords
        ndim1, ndim2 = datadict[var][dim1].shape
        ndimtot = ndim1*ndim2
        dct['pos'] = (Nx, Nx+ndimtot-1)
        dct['spacecoords'] = spacecoords
        dct['spacedims'] = (ndim1, ndim2)
        dct['vartype'] = vartype
        state_vect_info[var] = dct
        Nx += ndimtot

    if verbose:
        print('State vector information:')
        print(state_vect_info)

    if all(x == timedim[0] for x in timedim):
        ntime = timedim[0]
    else:
        raise SystemExit('ERROR im populate_ensemble: time dimension not consistent across all state variables. Exiting!')

    Xb = np.zeros((Nx, cfg.core.nens))

    random.seed(seed)
    ind_ens = random.sample(list(range(ntime)), cfg.core.nens)

    if verbose:
        print(f'shape of Xb: ({Nx} x {cfg.core.nens})')
        print('seed=', seed)
        print('sampled inds=', ind_ens)

    Xb_coords = np.empty((Nx, 2))
    Xb_coords[:, :] = np.nan

    for var in datadict.keys():
        vartype = datadict[var]['vartype']
        indstart = state_vect_info[var]['pos'][0]
        indend = state_vect_info[var]['pos'][1]

        for i in range(cfg.core.nens):
            Xb[indstart:indend+1, i] = datadict[var]['value'][ind_ens[i], :, :].flatten()

        coordname1, coordname2 = state_vect_info[var]['spacecoords']
        coord1, coord2 = datadict[var][coordname1], datadict[var][coordname1]

        if len(coord1.shape) == 1 and len(coord2.shape) == 1:
            ndim1 = coord1.shape[0]
            ndim2 = coord2.shape[0]
            X_coord1 = np.array([coord1, ]*ndim2).transpose()
            X_coord2 = np.array([coord2, ]*ndim1)
        elif len(coord1.shape) == 2 and len(coord2.shape) == 2:
            ndim1, ndim2 = coord1.shape
            X_coord1 = coord1
            X_coord2 = coord2

        Xb_coords[indstart:indend+1, 0] = X_coord1.flatten()
        Xb_coords[indstart:indend+1, 1] = X_coord2.flatten()

        if np.any(np.isnan(Xb)):
            # Returning state vector Xb as masked array
            Xb_res = np.ma.masked_invalid(Xb)
            # Set fill_value to np.nan
            np.ma.set_fill_value(Xb_res, np.nan)
        else:
            Xb_res = Xb

    return Xb_res, ind_ens, Xb_coords


def get_proxy(cfg, proxies_df_filepath, metadata_df_filepath,
              linear_precalib_filepath=None, bilinear_precalib_filepath=None,
              verbose=False):
    db_proxies = pd.read_pickle(proxies_df_filepath).to_dense()
    db_metadata = pd.read_pickle(metadata_df_filepath)

    proxy_db_cfg = {
        'LMRdb': cfg.proxies.LMRdb,
    }
    db_name = cfg.proxies.use_from[0]

    all_proxy_ids = []
    for proxy_order in proxy_db_cfg[db_name].proxy_order:
        archive = proxy_order.split('_', 1)[0]

        for measure in proxy_db_cfg[db_name].proxy_assim2[proxy_order]:
            archive_mask = db_metadata['Archive type'] == archive
            measure_mask = db_metadata['Proxy measurement'] == measure

            for proxy_resolution in proxy_db_cfg[db_name].proxy_resolution:
                resolution_mask = db_metadata['Resolution (yr)'] == proxy_resolution

                proxies = db_metadata['Proxy ID'][archive_mask & measure_mask & resolution_mask]
                all_proxy_ids += proxies.tolist()

    Proxy = collections.namedtuple(
        'Proxy',
        ['id', 'start_yr', 'end_yr', 'lat', 'lon', 'elev', 'seasonality', 'values', 'time', 'psm_key']
    )

    all_proxies = []
    picked_proxy_ids = []
    start, finish = cfg.core.recon_period
    check_precalib_file = linear_precalib_filepath or bilinear_precalib_filepath

    for site in all_proxy_ids:
        site_meta = db_metadata[db_metadata['Proxy ID'] == site]
        start_yr = site_meta['Youngest (C.E.)'].iloc[0]
        end_yr = site_meta['Oldest (C.E.)'].iloc[0]
        lat = site_meta['Lat (N)'].iloc[0]
        lon = site_meta['Lon (E)'].iloc[0]
        elev = site_meta['Elev'].iloc[0]
        seasonality = site_meta['Seasonality'].iloc[0]
        site_data = db_proxies[site]
        values = site_data[(site_data.index >= start) & (site_data.index <= finish)]
        values = values[values.notnull()]
        time = values.index.values

        if len(values) == 0:
            raise ValueError('ERROR: No obs in specified time range!')
        if proxy_db_cfg[db_name].proxy_timeseries_kind == 'anom':
            values = values - np.mean(values)

        try:
            pmeasure = site_meta['Proxy measurement'].iloc[0]
            db_type = site_meta['Archive type'].iloc[0]
            proxy_type = proxy_db_cfg[db_name].proxy_type_mapping[(db_type, pmeasure)]
        except (KeyError, ValueError) as e:
            print(f'Proxy type/measurement not found in mapping: {e}')
            raise ValueError(e)

        psm_key = proxy_db_cfg[db_name].proxy_psm_type[proxy_type]

        if check_precalib_file:
            precalib_filepath = {
                'linear': linear_precalib_filepath,
                'bilinear': bilinear_precalib_filepath,
            }

            psm_data = pd.read_pickle(precalib_filepath[psm_key])

            try:
                psm_site_data = psm_data[(proxy_type, site)]
                pobj = Proxy(site, start_yr, end_yr, lat, lon, elev, seasonality, values, time, psm_key)
                all_proxies.append(pobj)
                picked_proxy_ids.append(site)
            except KeyError:
                err_msg = f'Proxy in database but not found in pre-calibration file {precalib_filepath[psm_key]}...\nSkipping: {site}'
                if verbose:
                    print(err_msg)
        else:
            pobj = Proxy(site, start_yr, end_yr, lat, lon, elev, seasonality, values, time, psm_key)
            all_proxies.append(pobj)
            picked_proxy_ids.append(site)

    return picked_proxy_ids, all_proxies


def generate_proxy_ind(cfg, all_proxy_ids, seed):
    nsites = len(all_proxy_ids)
    nsites_assim = int(nsites * cfg.proxies.proxy_frac)

    random.seed(seed)

    ind_assim = random.sample(range(nsites), nsites_assim)
    ind_assim.sort()

    ind_eval = list(set(range(nsites)) - set(ind_assim))
    ind_eval.sort()

    return ind_assim, ind_eval


def get_ye(proxy_manager, prior_sample_idxs, ye_filesdict, proxy_set, verbose=False):
    num_samples = len(prior_sample_idxs)

    sites_proxy_objs = {
        'assim': proxy_manager.sites_assim_proxy_objs(),
        'eval': proxy_manager.sites_eval_proxy_objs(),
    }

    num_proxies = {
        'assim': len(proxy_manager.ind_assim),
        'eval': len(proxy_manager.ind_eval),
    }

    psm_keys = {
        'assim': list(set([pobj.psm_key for pobj in proxy_manager.sites_assim_proxy_objs()])),
        'eval': list(set([pobj.psm_key for pobj in proxy_manager.sites_eval_proxy_objs()])),
    }

    ye_all = np.zeros((num_proxies[proxy_set], num_samples))
    ye_all_coords = np.zeros((num_proxies[proxy_set], 2))

    precalc_files = {}
    for psm_key in psm_keys[proxy_set]:
        if verbose:
            print(f'Loading precalculated Ye from:\n {ye_filesdict[psm_key]}')

        precalc_files[psm_key] = np.load(ye_filesdict[psm_key])

    if verbose:
        print('Now extracting proxy type-dependent Ye values...')

    for i, pobj in enumerate(sites_proxy_objs[proxy_set]):
        psm_key = pobj.psm_key
        pid_idx_map = precalc_files[psm_key]['pid_index_map'][()]
        precalc_vals = precalc_files[psm_key]['ye_vals']

        pidx = pid_idx_map[pobj.id]
        ye_all[i] = precalc_vals[pidx, prior_sample_idxs]
        ye_all_coords[i] = np.asarray([pobj.lat, pobj.lon], dtype=np.float64)

    return ye_all, ye_all_coords


def get_valid_proxies(cfg, proxy_manager, target_year, Ye_assim, Ye_assim_coords, proxy_inds=None, verbose=False):
    recon_timescale = cfg.core.recon_timescale
    if verbose:
        print(f'finding proxy records for year: {target_year}')
        print(f'recon_timescale = {recon_timescale}')

    start_yr = int(target_year-recon_timescale//2)
    end_yr = int(target_year+recon_timescale//2)

    vY = []
    vR = []
    vP = []
    vT = []
    for proxy_idx, Y in enumerate(proxy_manager.sites_assim_proxy_objs()):
        # Check if we have proxy ob for current time interval
        if recon_timescale > 1:
            # exclude lower bound to not include same obs in adjacent time intervals
            Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
        else:
            # use all available proxies from config.yml
            if proxy_inds is None:
                #Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                Yvals = Y.values[(Y.time == target_year)]
                # use only the selected proxies (e.g., randomly filtered post-config)
            else:
                if proxy_idx in proxy_inds:
                    #Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                    Yvals = Y.values[(Y.time == target_year)]
                else:
                    Yvals = pd.DataFrame()

        if Yvals.empty:
            if verbose: print('no obs for this year')
            pass
        else:
            nYobs = len(Yvals)
            Yobs =  Yvals.mean()
            ob_err = Y.psm_obj.R/nYobs
            #           if (target_year >=start_yr) & (target_year <= end_yr):
            vY.append(Yobs)
            vR.append(ob_err)
            vP.append(proxy_idx)
            vT.append(Y.type)
            vYe = Ye_assim[vP, :]
            vYe_coords = Ye_assim_coords[vP, :]
