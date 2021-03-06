''' Some useful functions
'''
import numpy as np
import pandas as pd
from collections import namedtuple
from time import time
import functools
import xarray as xr
import netCDF4
from datetime import datetime
import random
from spharm import Spharmt, regrid
import prysm
import os
from tqdm import tqdm
import pickle

from . import load_gridded_data  # original file from LMR

Proxy = namedtuple(
    'Proxy',
    ['id', 'type', 'start_yr', 'end_yr', 'lat', 'lon', 'elev', 'seasonality', 'values', 'time', 'psm_obj']
)

PSM = namedtuple('PSM', ['psm_key', 'R'])

Grid = namedtuple('Grid', ['lat', 'lon', 'nlat', 'nlon', 'nens'])

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


def setup_cfg(cfg):
    proxy_db_cfg = {
        'LMRdb': cfg.proxies.LMRdb,
    }

    for db_name, db_cfg in proxy_db_cfg.items():
        db_cfg.proxy_type_mapping = {}
        for ptype, measurements in db_cfg.proxy_assim2.items():
            # Fetch proxy type name that occurs before underscore
            type_name = ptype.split('_', 1)[0]
            for measure in measurements:
                db_cfg.proxy_type_mapping[(type_name, measure)] = ptype

    return cfg


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
        #  spacecoords = dims.difference(['year'])
        spacecoords = ('lat', 'lon')
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


def get_prior(filepath, datatype, cfg, anom_reference_period=(1951, 1980), verbose=False):
    read_func = {
        'CMIP5': load_gridded_data.read_gridded_data_CMIP5_model,
    }

    prior_datadir = os.path.dirname(filepath)
    prior_datafile = os.path.basename(filepath)
    statevars = dict(cfg.prior.state_variables)
    statevars_info = dict(cfg.prior.state_variables_info)

    if cfg.core.recon_timescale == 1:
        avgInterval = {'annual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
    elif cfg.core.recon_timescale > 1:
        avgInterval = {'multiyaer': [cfg.core.recon_timescale]}
    else:
        print('ERROR in config.: unrecognized job.cfg.core.recon_timescale!')
        raise SystemExit()

    detrend = cfg.prior.detrend

    datadict = read_func[datatype](
        prior_datadir, prior_datafile, statevars, avgInterval,
        detrend=detrend, anom_ref=anom_reference_period,
        var_info=statevars_info,
    )

    return datadict


def get_nc_vars(filepath, varnames, useLib='netCDF4', annualize=False):
    ''' Get variables from given ncfile
    '''
    var_list = []

    if type(varnames) is str:
        varnames = [varnames]

    def load_with_xarray(annualize=False):
        with xr.open_dataset(filepath) as ds:

            if annualize:
                ds = ds.groupby('time.year').mean('time')

            for varname in varnames:
                if varname == 'year_float':
                    year = ds['time.year'].values
                    month = ds['time.month'].values
                    day = ds['time.day'].values

                    year_float = []
                    for y, m, d in zip(year, month, day):
                        date = datetime(year=y, month=m, day=d)
                        fst_day = datetime(year=y, month=1, day=1)
                        lst_day = datetime(year=y+1, month=1, day=1)
                        year_part = date - fst_day
                        year_length = lst_day - fst_day
                        year_float.append(y + year_part/year_length)

                    var_list.append(np.asarray(year_float))

                else:
                    var_tmp = ds[varname].values
                    if varname == 'lon':
                        if np.min(var_tmp) < 0:
                            var_tmp = np.mod(var_tmp, 360)  # convert from (-180, 180) to (0, 360)
                    var_list.append(var_tmp)

        return var_list

    def load_with_netCDF4(annualize=False):
        # TODO: annualize
        with netCDF4.Dataset(filepath, 'r') as ds:
            for varname in varnames:
                if varname == 'year_float':
                    time = ds.variables['time']
                    time_convert = netCDF4.num2date(time[:], time.units, time.calendar)

                    year_float = []
                    for t in time_convert:
                        y, m, d = t.year, t.month, t.day
                        date = datetime(year=y, month=m, day=d)
                        fst_day = datetime(year=y, month=1, day=1)
                        lst_day = datetime(year=y+1, month=1, day=1)
                        year_part = date - fst_day
                        year_length = lst_day - fst_day
                        year_float.append(y + year_part/year_length)

                    var_list.append(np.asarray(year_float))

                else:
                    var_tmp = ds.variables[varname][:]
                    if varname == 'lon':
                        if np.min(var_tmp) < 0:
                            var_tmp = np.mod(var_tmp, 360)  # convert from (-180, 180) to (0, 360)
                    var_list.append(var_tmp)

        return var_list

    load_nc = {
        'xarray': load_with_xarray,
        'netCDF4': load_with_netCDF4,
    }

    var_list = load_nc[useLib](annualize=annualize)

    if len(var_list) == 1:
        var_list = var_list[0]

    return var_list


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
        coord1, coord2 = datadict[var][coordname1], datadict[var][coordname2]

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

    return Xb_res, ind_ens, Xb_coords, state_vect_info


def regrid_prior(cfg, X, verbose=False):
    nens = cfg.core.nens
    regrid_method = cfg.prior.regrid_method
    regrid_resolution = cfg.prior.regrid_resolution

    regrid_func = {
        'spherical_harmonics': regrid_sphere,
    }

    new_state_info = {}
    Nx = 0
    for var in list(X.full_state_info.keys()):
        dct = {}

        dct['vartype'] = X.full_state_info[var]['vartype']

        # variable indices in full state vector
        ibeg_full = X.full_state_info[var]['pos'][0]
        iend_full = X.full_state_info[var]['pos'][1]
        # extract array corresponding to state variable "var"
        var_array_full = X.ens[ibeg_full:iend_full+1, :]
        # corresponding spatial coordinates
        coords_array_full = X.coords[ibeg_full:iend_full+1, :]

        nlat = X.full_state_info[var]['spacedims'][0]
        nlon = X.full_state_info[var]['spacedims'][1]

        var_array_new, lat_new, lon_new = regrid_func[regrid_method](nlat, nlon, nens, var_array_full, regrid_resolution)

        nlat_new = np.shape(lat_new)[0]
        nlon_new = np.shape(lat_new)[1]

        if verbose:
            print(('=> Full array:      ' + str(np.min(var_array_full)) + ' ' +
                   str(np.max(var_array_full)) + ' ' + str(np.mean(var_array_full)) +
                   ' ' + str(np.std(var_array_full))))
            print(('=> Truncated array: ' + str(np.min(var_array_new)) + ' ' +
                   str(np.max(var_array_new)) + ' ' + str(np.mean(var_array_new)) +
                   ' ' + str(np.std(var_array_new))))

        # corresponding indices in truncated state vector
        ibeg_new = Nx
        iend_new = Nx+(nlat_new*nlon_new)-1
        # for new state info dictionary
        dct['pos'] = (ibeg_new, iend_new)
        dct['spacecoords'] = X.full_state_info[var]['spacecoords']
        dct['spacedims'] = (nlat_new, nlon_new)
        # updated dimension
        new_dims = (nlat_new*nlon_new)

        # array with new spatial coords
        coords_array_new = np.zeros(shape=[new_dims, 2])
        coords_array_new[:, 0] = lat_new.flatten()
        coords_array_new[:, 1] = lon_new.flatten()

        # fill in new state info dictionary
        new_state_info[var] = dct

        # if 1st time in loop over state variables, create Xb_one array as copy
        # of var_array_new
        if Nx == 0:
            Xb_one = np.copy(var_array_new)
            Xb_one_coords = np.copy(coords_array_new)
        else:  # if not 1st time, append to existing array
            Xb_one = np.append(Xb_one, var_array_new, axis=0)
            Xb_one_coords = np.append(Xb_one_coords, coords_array_new, axis=0)

        # making sure Xb_one has proper mask, if it contains
        # at least one invalid value
        if np.isnan(Xb_one).any():
            Xb_one = np.ma.masked_invalid(Xb_one)
            np.ma.set_fill_value(Xb_one, np.nan)

        # updating dimension of new state vector
        Nx = Nx + new_dims

        return Xb_one, Xb_one_coords, new_state_info


def get_proxy(cfg, proxies_df_filepath, metadata_df_filepath, precalib_filesdict=None, verbose=False):

    db_proxies = pd.read_pickle(proxies_df_filepath).to_dense()
    db_metadata = pd.read_pickle(metadata_df_filepath)

    proxy_db_cfg = {
        'LMRdb': cfg.proxies.LMRdb,
    }
    db_name = cfg.proxies.use_from[0]

    all_proxy_ids = []
    for name in proxy_db_cfg[db_name].proxy_order:
        archive = name.split('_', 1)[0]
        archive_mask = db_metadata['Archive type'] == archive

        measure_mask = db_metadata['Proxy measurement'] == 0
        measure_mask &= False

        for measure in proxy_db_cfg[db_name].proxy_assim2[name]:
            measure_mask |= db_metadata['Proxy measurement'] == measure

        resolution_mask = db_metadata['Resolution (yr)'] == 0
        resolution_mask &= False
        for proxy_resolution in proxy_db_cfg[db_name].proxy_resolution:
            resolution_mask |= db_metadata['Resolution (yr)'] == proxy_resolution

        proxies = db_metadata['Proxy ID'][archive_mask & measure_mask & resolution_mask]
        proxies_list = proxies.tolist()

        all_proxy_ids += proxies_list

    all_proxies = []
    picked_proxy_ids = []
    start, finish = cfg.core.recon_period

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

        # check if pre-calibration files are provided

        if precalib_filesdict and psm_key in precalib_filesdict.keys():
            psm_data = pd.read_pickle(precalib_filesdict[psm_key])
            try:
                if verbose:
                    print(site, psm_key)
                psm_site_data = psm_data[(proxy_type, site)]
                psm_obj = PSM(psm_key, psm_site_data['PSMmse'])
                pobj = Proxy(site, proxy_type, start_yr, end_yr, lat, lon, elev, seasonality, values, time, psm_obj)
                all_proxies.append(pobj)
                picked_proxy_ids.append(site)
            except KeyError:
                #  err_msg = f'Proxy in database but not found in pre-calibration file {precalib_filesdict[psm_key]}...\nSkipping: {site}'
                err_msg = f'Proxy in database but not found in pre-calibration files...\nSkipping: {site}'
                if verbose:
                    print(err_msg)
        else:
            # TODO: calibrate
            psm_obj = PSM(psm_key, None)
            pobj = Proxy(site, proxy_type, start_yr, end_yr, lat, lon, elev, seasonality, values, time, psm_obj)
            all_proxies.append(pobj)
            picked_proxy_ids.append(site)

    return picked_proxy_ids, all_proxies


def get_env_vars(prior_filesdict, rename_vars={'d18O': 'd18Opr', 'tos': 'sst', 'sos': 'sss'},
                 useLib='netCDF4', verbose=False):
    prior_vars = {}

    first_item = True
    for prior_varname, prior_filepath in prior_filesdict.items():
        if verbose:
            print(f'Loading [{prior_varname}] from {prior_filepath} ...')
        if first_item:
            lat_model, lon_model, time_model, prior_vars[prior_varname] = get_nc_vars(
                prior_filepath, ['lat', 'lon', 'year_float', prior_varname], useLib=useLib,
            )
            first_item = False
        else:
            prior_vars[prior_varname] = get_nc_vars(prior_filepath, prior_varname)

    if rename_vars:
        for old_name, new_name in rename_vars.items():
            if old_name in prior_vars:
                prior_vars[new_name] = prior_vars.pop(old_name)

    return lat_model, lon_model, time_model, prior_vars


def calc_ye(proxy_manager, ptype, psm_name,
            lat_model, lon_model, time_model,
            prior_vars, verbose=False, **psm_params):
    pid_map = {}
    ye_out = []
    count = 0

    if 'vslite_params_path' in psm_params:
        # load parameters for VS-Lite
        with open(psm_params['vslite_params_path'], 'rb') as f:
            res = pickle.load(f)
            pid_obs = res['pid_obs']
            T1 = res['T1']
            T2 = res['T2']
            M1 = res['M1']
            M2 = res['M2']

    for idx, pobj in enumerate(proxy_manager.all_proxies):
        if pobj.type == ptype:
            count += 1
            if verbose:
                print(f'\nProcessing #{count} - {pobj.id} ...')

            if 'vslite_params_path' in psm_params and pobj.id in pid_obs:
                # load parameters for VS-Lite
                ind = pid_obs.index(pobj.id)
                psm_params['T1'] = T1[ind]
                psm_params['T2'] = T2[ind]
                psm_params['M1'] = M1[ind]
                psm_params['M2'] = M2[ind]

            ye_tmp, _ = prysm.forward(
                psm_name, pobj.lat, pobj.lon,
                lat_model, lon_model, time_model,
                prior_vars, verbose=verbose, **psm_params,
            )
            ye_out.append(ye_tmp)
            pid_map[pobj.id] = idx
        else:
            # PSM not available; skip
            continue

    ye_out = np.asarray(ye_out)

    return pid_map, ye_out


def est_vslite_params(proxy_manager, tas_filepath, pr_filepath,
                      matlab_path=None, func_path=None, restart_matlab_period=100,
                      lat_lon_idx_path=None, seed=0, verbose=False):
    from pymatbridge import Matlab

    pid_obs = []
    lat_obs = []
    lon_obs = []
    values_obs = []
    for idx, pobj in enumerate(proxy_manager.all_proxies):
        if pobj.psm_obj.psm_key == 'prysm.vslite':
            lat_obs.append(pobj.lat)
            lon_obs.append(pobj.lon)
            values_obs.append(pobj.values)
            pid_obs.append(pobj.id)

    lat_grid, lon_grid, time_grid, tas = get_nc_vars(tas_filepath, ['lat', 'lon', 'year_float', 'tmp'])
    pr = get_nc_vars(pr_filepath, ['pre'])

    if lat_lon_idx_path is None:
        lat_ind, lon_ind = find_closest_loc(lat_grid, lon_grid, lat_obs, lon_obs, mode='latlon')
    else:
        with open(lat_lon_idx_path, 'rb') as f:
            lat_ind, lon_ind = pickle.load(f)

    T = tas[:, lat_ind, lon_ind]
    P = pr[:, lat_ind, lon_ind]

    if matlab_path is None:
        raise ValueError('ERROR: matlab_path must be set!')
    if func_path is None:
        root_path = os.path.dirname(__file__)
        func_path = os.path.join(root_path, 'estimate_vslite_params_v2_3.m')
        if verbose:
            print(func_path)

    mlab = Matlab(matlab_path)
    mlab.start()

    T1 = []
    T2 = []
    M1 = []
    M2 = []

    for i, trw_data in enumerate(tqdm(values_obs)):
        if verbose:
            print(f'#{i+1} - Target: ({lat_obs[i]}, {lon_obs[i]}); Found: ({lat_grid[lat_ind[i]]:.2f}, {lon_grid[lon_ind[i]]:.2f});', end=' ')
        trw_year = np.asarray(trw_data.index)
        trw_value = np.asarray(trw_data.values)
        trw_year, trw_value = pick_range(trw_year, trw_value, 1901, 2001)
        grid_year, grid_tas = pick_years(trw_year, time_grid, T[:, i])
        grid_year, grid_pr = pick_years(trw_year, time_grid, P[:, i])
        nyr = int(len(grid_year)/12)

        if verbose:
            print(f'{nyr} available years')
            print(f'Running estimate_vslite_params_v2_3.m ...', end=' ')

        # restart Matlab kernel
        if np.mod(i+1, restart_matlab_period) == 0:
            mlab.stop()
            mlab.start()

        start_time = time()
        res = mlab.run_func(
            func_path,
            grid_tas.reshape(nyr, 12).T, grid_pr.reshape(nyr, 12).T, lat_obs[i], trw_value,
            'seed', seed,
            nargout=4,
        )

        used_time = time() - start_time
        if verbose:
            print(res)
            print(f'{used_time:.2f} sec')

        T1_tmp = res['result'][0]
        T2_tmp = res['result'][1]
        M1_tmp = res['result'][2]
        M2_tmp = res['result'][3]

        if verbose:
            print(f'T1={T1_tmp}, T2={T2_tmp}, M1={M1_tmp}, M2={M2_tmp}\n')

        T1.append(T1_tmp)
        T2.append(T2_tmp)
        M1.append(M1_tmp)
        M2.append(M2_tmp)

    res_dict = {
        'pid_obs': pid_obs,
        'lat_obs': lat_obs,
        'lon_obs': lon_obs,
        'values_obs': values_obs,
        'T1': T1,
        'T2': T2,
        'M1': M1,
        'M2': M2,
    }

    return res_dict


def pick_range(ts, ys, lb, ub):
    ''' Pick range [lb, ub) from a timeseries pair (ts, ys)
    '''
    range_mask = (ts >= lb) & (ts < ub)
    return ts[range_mask], ys[range_mask]


def pick_years(year_int, time_grid, var_grid):
    ''' Pick years from a timeseries pair (time_grid, var_grid) based on year_int
    '''
    year_int = [int(y) for y in year_int]
    mask = []
    for year_float in time_grid:
        if int(year_float) in year_int:
            mask.append(True)
        else:
            mask.append(False)
    return time_grid[mask], var_grid[mask]


def overlap_ts(time1, value1, time2, value2):
    mask1 = []
    for t1 in time1:
        if t1 in time2:
            mask1.append(True)
        else:
            mask1.append(False)

    time1_overlap = time1[mask1]
    value1_overlap = value1[mask1]

    mask2 = []
    for t2 in time2:
        if t2 in time1_overlap:
            mask2.append(True)
        else:
            mask2.append(False)

    time2_overlap = time2[mask2]
    value2_overlap = value2[mask2]

    return time1_overlap, value1_overlap, time2_overlap, value2_overlap


def calibrate_psm(proxy_manager, ptype, psm_name,
                  lat_model, lon_model, time_model,
                  inst_vars, verbose=False, **psm_params):
    precalib_dict = {}
    count = 0

    start_yr = int(time_model[0])
    end_yr = int(time_model[-1])

    for idx, pobj in enumerate(proxy_manager.all_proxies):
        if pobj.type == ptype:
            count += 1
            if verbose:
                print(f'\nProcessing #{count} - {pobj.id} ...')
            ye_tmp, _ = forward(
                psm_name, pobj.lat, pobj.lon,
                lat_model, lon_model, time_model,
                inst_vars, verbose=verbose, **psm_params,
            )

            #  Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
            proxy_value = pobj.values[(pobj.values.index >= start_yr) & (pobj.values.index <= end_yr)]

            resid = ye_tmp - proxy_value
            PSMmse = np.mean((resid) ** 2)

            precalib_dict[(pobj.type, pobj.id)] = {
                'PSMmse': PSMmse,
            }
        else:
            # PSM not available; skip
            continue

    return precalib_dict


def generate_proxy_ind(cfg, nsites, seed):
    nsites_assim = int(nsites * cfg.proxies.proxy_frac)

    random.seed(seed)

    ind_assim = random.sample(range(nsites), nsites_assim)
    ind_assim.sort()

    ind_eval = list(set(range(nsites)) - set(ind_assim))
    ind_eval.sort()

    return ind_assim, ind_eval


def get_ye(proxy_manager, prior_sample_idxs, ye_filesdict, proxy_set, verbose=False):

    if verbose:
        print(f'-------------------------------------------')
        print(f'Loading Ye files for proxy set: {proxy_set}')
        print(f'-------------------------------------------')

    num_samples = len(prior_sample_idxs)

    sites_proxy_objs = {
        'assim': proxy_manager.sites_assim_proxy_objs,
        'eval': proxy_manager.sites_eval_proxy_objs,
    }

    num_proxies = {
        'assim': len(proxy_manager.ind_assim),
        'eval': len(proxy_manager.ind_eval),
    }

    psm_keys = {
        'assim': list(set([pobj.psm_obj.psm_key for pobj in proxy_manager.sites_assim_proxy_objs])),
        'eval': list(set([pobj.psm_obj.psm_key for pobj in proxy_manager.sites_eval_proxy_objs])),
    }

    ye_all = np.zeros((num_proxies[proxy_set], num_samples))
    ye_all_coords = np.zeros((num_proxies[proxy_set], 2))

    precalc_files = {}
    for psm_key in psm_keys[proxy_set]:
        if verbose:
            print(f'Loading precalculated Ye from:\n {ye_filesdict[psm_key]}\n')

        precalc_files[psm_key] = np.load(ye_filesdict[psm_key])

    if verbose:
        print('Now extracting proxy type-dependent Ye values...\n')

    for i, pobj in enumerate(sites_proxy_objs[proxy_set]):
        psm_key = pobj.psm_obj.psm_key
        pid_idx_map = precalc_files[psm_key]['pid_index_map'][()]
        precalc_vals = precalc_files[psm_key]['ye_vals']

        pidx = pid_idx_map[pobj.id]
        ye_all[i] = precalc_vals[pidx, prior_sample_idxs]
        ye_all_coords[i] = np.asarray([pobj.lat, pobj.lon], dtype=np.float64)

    # delete nans
    delete_site_rows = []
    for i, ye in enumerate(ye_all):
        if any(np.isnan(ye)):
            delete_site_rows.append(i)

    ye_all = np.delete(ye_all, delete_site_rows, 0)
    ye_all_coords = np.delete(ye_all_coords, delete_site_rows, 0)

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
    #  for proxy_idx, Y in enumerate(proxy_manager.sites_assim_proxy_objs()):
    for proxy_idx, Y in enumerate(proxy_manager.sites_assim_proxy_objs):
        # Check if we have proxy ob for current time interval
        if recon_timescale > 1:
            # exclude lower bound to not include same obs in adjacent time intervals
            Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
        else:
            # use all available proxies from config.yml
            if proxy_inds is None:
                Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                #  Yvals = Y.values[(Y.time == target_year)]
                # use only the selected proxies (e.g., randomly filtered post-config)
            else:
                if proxy_idx in proxy_inds:
                    #Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                    Yvals = Y.values[(Y.time == target_year)]
                else:
                    Yvals = pd.DataFrame()

        if Yvals.empty:
            #  print('empty:', proxy_idx)
            if verbose: print('no obs for this year')
            pass
        else:
            nYobs = len(Yvals)
            Yobs = Yvals.mean()
            ob_err = Y.psm_obj.R/nYobs
            #           if (target_year >=start_yr) & (target_year <= end_yr):
            vY.append(Yobs)
            vR.append(ob_err)
            vP.append(proxy_idx)
            vT.append(Y.type)

    vYe = Ye_assim[vP, :]
    vYe_coords = Ye_assim_coords[vP, :]

    return vY, vR, vP, vYe, vT, vYe_coords


def make_grid(prior):
    lat, lon = sorted(list(set(prior.coords[:, 0]))), sorted(list(set(prior.coords[:, 1])))
    nlat, nlon = np.size(lat), np.size(lon)
    nens = np.shape(prior.ens)[-1]

    g = Grid(lat, lon, nlat, nlon, nens)
    return g


def update_year_lite(target_year, cfg, Xb_one, grid, proxy_manager, ye_all, ye_all_coords,
                     Xb_one_aug, Xb_one_coords, X,
                     ibeg_tas, iend_tas,
                     proxy_inds=None, da_solver='ESRF',
                     verbose=False):

    vY, vR, vP, vYe, vT, vYe_coords = get_valid_proxies(
        cfg, proxy_manager, target_year, ye_all, ye_all_coords, proxy_inds=proxy_inds, verbose=verbose)

    if da_solver == 'ESRF':
        xam, Xap, Xa = Kalman_ESRF(
            cfg, vY, vR, vYe, Xb_one,
            proxy_manager, X, Xb_one_aug, Xb_one_coords, verbose=verbose
        )
    elif da_solver == 'optimal':
        xam, Xap, _ = Kalman_optimal(vY, vR, vYe, Xb_one, verbose=verbose)
    else:
        raise ValueError('ERROR: Wrong da_solver!!!')

    nens = grid.nens
    gmt_ens = np.zeros(nens)
    nhmt_ens = np.zeros(nens)
    shmt_ens = np.zeros(nens)

    for k in range(nens):
        xam_lalo = np.reshape(Xa[ibeg_tas:iend_tas+1, k], [grid.nlat, grid.nlon])
        gmt_ens[k], nhmt_ens[k], shmt_ens[k] = global_hemispheric_means(xam_lalo, grid.lat)

    return gmt_ens, nhmt_ens, shmt_ens


def update_year(yr_idx, target_year,
                cfg, Xb_one_aug, Xb_one_coords, X, sites_assim_proxy_objs,
                assim_proxy_count, eval_proxy_count, grid,
                ibeg_tas, iend_tas,
                verbose=False):

    recon_timescale = cfg.core.recon_timescale
    start_yr = int(target_year-recon_timescale//2)
    end_yr = int(target_year+recon_timescale//2)

    Xb = Xb_one_aug.copy()

    inflate = cfg.core.inflation_fact

    for proxy_idx, Y in enumerate(sites_assim_proxy_objs):
        try:
            if recon_timescale > 1:
                # exclude lower bound to not include same obs in adjacent time intervals
                Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
            else:
                Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                if Yvals.empty:
                    raise KeyError()
                nYobs = len(Yvals)
                Yobs = Yvals.mean()

        except KeyError:
            continue  # skip to next loop iteration (proxy record)

        loc = cov_localization(cfg.core.loc_rad, Y, X, Xb_one_coords)

        Ye = Xb[proxy_idx - (assim_proxy_count+eval_proxy_count)]

        ob_err = Y.psm_obj.R/float(nYobs)

        Xa = enkf_update_array(Xb, Yobs, Ye, ob_err, loc=loc, inflate=inflate)

        xbvar = Xb.var(axis=1, ddof=1)
        xavar = Xa.var(axis=1, ddof=1)
        vardiff = xavar - xbvar
        if (not np.isfinite(np.min(vardiff))) or (not np.isfinite(np.max(vardiff))):
            print('ERROR: Reconstruction has blown-up. Exiting!')
            print(f'Y.id={Y.id}')
            print(f'np.max(Xb)={np.max(Xb)}')
            print(f'Y.psm_obj.psm_key={Y.psm_obj.psm_key}')
            print(f'Y.psm_obj.R={Y.psm_obj.R}')
            print(f'np.min(xbvar)={np.min(xbvar)}, np.max(xbvar)={np.max(xbvar)}')
            print(f'np.min(xavar)={np.min(xavar)}, np.max(xavar)={np.max(xavar)}')
            #  print(f'np.min(vardiff)={np.min(vardiff)}')
            #  print(f'np.max(vardiff)={np.max(vardiff)}')
            #  print(f'ob_err={ob_err}')
            #  print(f'nYobs={nYobs}')
            raise SystemExit(1)

        if verbose:
            print('min/max change in variance: ('+str(np.min(vardiff))+','+str(np.max(vardiff))+')')

        Xb = Xa

    xam_lalo = Xa[ibeg_tas:iend_tas+1, :].T.reshape(grid.nens, grid.nlat, grid.nlon)

    gmt_ens, nhmt_ens, shmt_ens = global_hemispheric_means(xam_lalo, grid.lat)

    return gmt_ens, nhmt_ens, shmt_ens


def regrid_sphere(nlat, nlon, Nens, X, ntrunc):
    """ Truncate lat,lon grid to another resolution in spherical harmonic space. Triangular truncation

    Inputs:
    nlat            : number of latitudes
    nlon            : number of longitudes
    Nens            : number of ensemble members
    X               : data array of shape (nlat*nlon,Nens)
    ntrunc          : triangular truncation (e.g., use 42 for T42)

    Outputs :
    lat_new : 2D latitude array on the new grid (nlat_new,nlon_new)
    lon_new : 2D longitude array on the new grid (nlat_new,nlon_new)
    X_new   : truncated data array of shape (nlat_new*nlon_new, Nens)

    Originator: Greg Hakim
                University of Washington
                May 2015
    """
    # create the spectral object on the original grid
    specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')

    # truncate to a lower resolution grid (triangular truncation)
    ifix = np.remainder(ntrunc,2.0).astype(int)
    nlat_new = ntrunc + ifix
    nlon_new = int(nlat_new*1.5)

    # create the spectral object on the new grid
    specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')

    # create new lat,lon grid arrays
    # Note: AP - According to github.com/jswhit/pyspharm documentation the
    #  latitudes will not include the equator or poles when nlats is even.
    if nlat_new % 2 == 0:
        include_poles = False
    else:
        include_poles = True

    lat_new, lon_new, _, _ = generate_latlon(nlat_new, nlon_new,
                                             include_endpts=include_poles)

    # transform each ensemble member, one at a time
    X_new = np.zeros([nlat_new*nlon_new,Nens])
    for k in range(Nens):
        X_lalo = np.reshape(X[:,k],(nlat,nlon))
        Xbtrunc = regrid(specob_lmr, specob_new, X_lalo, ntrunc=nlat_new-1, smooth=None)
        vectmp = Xbtrunc.flatten()
        X_new[:,k] = vectmp

    return X_new,lat_new,lon_new


def find_closest_loc(lat, lon, target_lat, target_lon, mode='latlon', verbose=False):
    ''' Find the closet model sites (lat, lon) based on the given target (lat, lon) list

    Args:
        lat, lon (array): the model latitude and longitude arrays
        target_lat, target_lon (array): the target latitude and longitude arrays
        mode (str):
        + latlon: the model lat/lon is a 1-D array
        + mesh: the model lat/lon is a 2-D array

    Returns:
        lat_ind, lon_ind (array): the indices of the found closest model sites

    '''

    if mode is 'latlon':
        # model locations
        mesh = np.meshgrid(lon, lat)

        list_of_grids = list(zip(*(grid.flat for grid in mesh)))
        model_lon, model_lat = zip(*list_of_grids)

    elif mode is 'mesh':
        model_lat = lat.flatten()
        model_lon = lon.flatten()

    elif mode is 'list':
        model_lat = lat
        model_lon = lon

    model_locations = []

    for m_lat, m_lon in zip(model_lat, model_lon):
        model_locations.append((m_lat, m_lon))

    # target locations
    if np.size(target_lat) > 1:
        #  target_locations_dup = list(zip(target_lat, target_lon))
        #  target_locations = list(set(target_locations_dup))  # remove duplicated locations
        target_locations = list(zip(target_lat, target_lon))
        n_loc = np.shape(target_locations)[0]
    else:
        target_locations = [(target_lat, target_lon)]
        n_loc = 1

    lat_ind = np.zeros(n_loc, dtype=int)
    lon_ind = np.zeros(n_loc, dtype=int)

    # get the closest grid
    for i, target_loc in (enumerate(tqdm(target_locations)) if verbose else enumerate(target_locations)):
        X = target_loc
        Y = model_locations
        distance, index = spatial.KDTree(Y).query(X)
        closest = Y[index]
        nlon = np.shape(lon)[-1]

        lat_ind[i] = index // nlon
        lon_ind[i] = index % nlon

        #  if np.size(target_lat) > 1:
            #  df_ind[i] = target_locations_dup.index(target_loc)

    if np.size(target_lat) > 1:
        #  return lat_ind, lon_ind, df_ind
        return lat_ind, lon_ind
    else:
        return lat_ind[0], lon_ind[0]


def coefficient_efficiency(ref,test,valid=None):
    """
    Compute the coefficient of efficiency for a test time series, with respect to a reference time series.

    Inputs:
    test:  test array
    ref:   reference array, of same size as test
    valid: fraction of valid data required to calculate the statistic

    Note: Assumes that the first dimension in test and ref arrays is time!!!

    Outputs:
    CE: CE statistic calculated following Nash & Sutcliffe (1970)
    """

    # check array dimensions
    dims_test = test.shape
    dims_ref  = ref.shape
    #print('dims_test: ', dims_test, ' dims_ref: ', dims_ref)

    if len(dims_ref) == 3:   # 3D: time + 2D spatial
        dims = dims_ref[1:3]
    elif len(dims_ref) == 2: # 2D: time + 1D spatial
        dims = dims_ref[1:2]
    elif len(dims_ref) == 1: # 0D: time series
        dims = 1
    else:
        print('In coefficient_efficiency(): Problem with input array dimension! Exiting...')
        SystemExit(1)

    CE = np.zeros(dims)

    # error
    error = test - ref

    # CE
    numer = np.nansum(np.power(error,2),axis=0)
    denom = np.nansum(np.power(ref-np.nanmean(ref,axis=0),2),axis=0)
    CE    = 1. - np.divide(numer,denom)

    if valid:
        nbok  = np.sum(np.isfinite(ref),axis=0)
        nball = float(dims_ref[0])
        ratio = np.divide(nbok,nball)
        indok  = np.where(ratio >= valid)
        indbad = np.where(ratio < valid)
        dim_indbad = len(indbad)
        testlist = [indbad[k].size for k in range(dim_indbad)]
        if not all(v == 0 for v in testlist):
            if isinstance(dims,(tuple,list)):
                CE[indbad] = np.nan
            else:
                CE = np.nan

    return CE

def generate_latlon(nlats, nlons, include_endpts=False,
                    lat_bnd=(-90,90), lon_bnd=(0, 360)):
    """
    Generate regularly spaced latitude and longitude arrays where each point
    is the center of the respective grid cell.

    Parameters
    ----------
    nlats: int
        Number of latitude points
    nlons: int
        Number of longitude points
    lat_bnd: tuple(float), optional
        Bounding latitudes for gridcell edges (not centers).  Accepts values
        in range of [-90, 90].
    lon_bnd: tuple(float), optional
        Bounding longitudes for gridcell edges (not centers).  Accepts values
        in range of [-180, 360].
    include_endpts: bool
        Include the poles in the latitude array.

    Returns
    -------
    lat_center_2d:
        Array of central latitide points (nlat x nlon)
    lon_center_2d:
        Array of central longitude points (nlat x nlon)
    lat_corner:
        Array of latitude boundaries for all grid cells (nlat+1)
    lon_corner:
        Array of longitude boundaries for all grid cells (nlon+1)
    """

    if len(lat_bnd) != 2 or len(lon_bnd) != 2:
        raise ValueError('Bound tuples must be of length 2')
    if np.any(np.diff(lat_bnd) < 0) or np.any(np.diff(lon_bnd) < 0):
        raise ValueError('Lower bounds must be less than upper bounds.')
    if np.any(abs(np.array(lat_bnd)) > 90):
        raise ValueError('Latitude bounds must be between -90 and 90')
    if np.any(abs(np.diff(lon_bnd)) > 360):
        raise ValueError('Longitude bound difference must not exceed 360')
    if np.any(np.array(lon_bnd) < -180) or np.any(np.array(lon_bnd) > 360):
        raise ValueError('Longitude bounds must be between -180 and 360')

    lon_center = np.linspace(lon_bnd[0], lon_bnd[1], nlons, endpoint=False)

    if include_endpts:
        lat_center = np.linspace(lat_bnd[0], lat_bnd[1], nlats)
    else:
        tmp = np.linspace(lat_bnd[0], lat_bnd[1], nlats+1)
        lat_center = (tmp[:-1] + tmp[1:]) / 2.

    lon_center_2d, lat_center_2d = np.meshgrid(lon_center, lat_center)
    lat_corner, lon_corner = calculate_latlon_bnds(lat_center, lon_center)

    return lat_center_2d, lon_center_2d, lat_corner, lon_corner


def calculate_latlon_bnds(lats, lons):
    """
    Calculate the bounds for regularly gridded lats and lons.

    Parameters
    ----------
    lats: ndarray
        Regularly spaced latitudes.  Must be 1-dimensional and monotonically
        increase with index.
    lons:  ndarray
        Regularly spaced longitudes.  Must be 1-dimensional and monotonically
        increase with index.

    Returns
    -------
    lat_bnds:
        Array of latitude boundaries for each input latitude of length
        len(lats)+1.
    lon_bnds:
        Array of longitude boundaries for each input longitude of length
        len(lons)+1.

    """
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError('Expected 1D-array for input lats and lons.')
    if np.any(np.diff(lats) < 0) or np.any(np.diff(lons) < 0):
        raise ValueError('Expected monotonic value increase with index for '
                         'input latitudes and longitudes')

    # Note: assumes lats are monotonic increase with index
    dlat = abs(lats[1] - lats[0]) / 2.
    dlon = abs(lons[1] - lons[0]) / 2.

    # Check that inputs are regularly spaced
    lat_space = np.diff(lats)
    lon_space = np.diff(lons)

    lat_bnds = np.zeros(len(lats)+1)
    lon_bnds = np.zeros(len(lons)+1)

    lat_bnds[1:-1] = lats[:-1] + lat_space/2.
    lat_bnds[0] = lats[0] - lat_space[0]/2.
    lat_bnds[-1] = lats[-1] + lat_space[-1]/2.

    if lat_bnds[0] < -90:
        lat_bnds[0] = -90.
    if lat_bnds[-1] > 90:
        lat_bnds[-1] = 90.

    lon_bnds[1:-1] = lons[:-1] + lon_space/2.
    lon_bnds[0] = lons[0] - lon_space[0]/2.
    lon_bnds[-1] = lons[-1] + lon_space[-1]/2.

    return lat_bnds, lon_bnds


def cov_localization(locRad, Y, X, X_coords):
    """
    Originator: R. Tardif,
                Dept. Atmos. Sciences, Univ. of Washington
    -----------------------------------------------------------------
     Inputs:
        locRad : Localization radius (distance in km beyond which cov are forced to zero)
             Y : Proxy object, needed to get ob site lat/lon (to calculate distances w.r.t. grid pts
             X : Prior object, needed to get state vector info.
      X_coords : Array containing geographic location information of state vector elements

     Output:
        covLoc : Localization vector (weights) applied to ensemble covariance estimates.
                 Dims = (Nx x 1), with Nx the dimension of the state vector.

     Note: Uses the Gaspari-Cohn localization function.

    """

    # declare the localization array, filled with ones to start with (as in no localization)
    stateVectDim, nbdimcoord = X_coords.shape
    covLoc = np.ones(shape=[stateVectDim],dtype=np.float64)

    # Mask to identify elements of state vector that are "localizeable"
    # i.e. fields with (lat,lon)
    localizeable = covLoc == 1. # Initialize as True

    for var in X.trunc_state_info.keys():
        [var_state_pos_begin,var_state_pos_end] =  X.trunc_state_info[var]['pos']
        # if variable is not a field with lats & lons, tag localizeable as False
        if X.trunc_state_info[var]['spacecoords'] != ('lat', 'lon'):
            localizeable[var_state_pos_begin:var_state_pos_end+1] = False

    # array of distances between state vector elements & proxy site
    # initialized as zeros: this is important!
    dists = np.zeros(shape=[stateVectDim])

    # geographic location of proxy site
    site_lat = Y.lat
    site_lon = Y.lon
    # geographic locations of elements of state vector
    X_lon = X_coords[:,1]
    X_lat = X_coords[:,0]

    # calculate distances for elements tagged as "localizeable".
    dists[localizeable] = np.array(haversine(site_lon, site_lat,
                                                       X_lon[localizeable],
                                                       X_lat[localizeable]),dtype=np.float64)

    # those not "localizeable" are assigned with a disdtance of "nan"
    # so these elements will not be included in the indexing
    # according to distances (see below)
    dists[~localizeable] = np.nan

    # Some transformation to variables used in calculating localization weights
    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    # values for distances very near the localization radius
    # TODO: revisit calculations to minimize round-off errors
    covLoc[covLoc < 0.0] = 0.0

    return covLoc


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km


def enkf_update_array(Xb, obvalue, Ye, ob_err, loc=None, inflate=None):
    """
    Function to do the ensemble square-root filter (EnSRF) update
    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator: G. J. Hakim, with code borrowed from L. Madaus
                Dept. Atmos. Sciences, Univ. of Washington

    Revisions:

    1 September 2017:
                    - changed varye = np.var(Ye) to varye = np.var(Ye,ddof=1)
                    for an unbiased calculation of the variance.
                    (G. Hakim - U. Washington)

    -----------------------------------------------------------------
     Inputs:
          Xb: background ensemble estimates of state (Nx x Nens)
     obvalue: proxy value
          Ye: background ensemble estimate of the proxy (Nens x 1)
      ob_err: proxy error variance
         loc: localization vector (Nx x 1) [optional]
     inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array: Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = np.mean(Xb,axis=1)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy
    mye   = np.mean(Ye)
    varye = np.var(Ye,ddof=1)

    # lowercase ye has ensemble-mean removed
    ye = np.subtract(Ye, mye)

    # innovation
    try:
        innov = obvalue - mye
    except:
        print('innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye))
        print('returning Xb unchanged...')
        return Xb

    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp,np.transpose(ye)) / (Nens-1)

    # Option to inflate the covariances by a certain factor
    #if inflate is not None:
    #    kcov = inflate * kcov # This implementation is not correct. To be revised later.

    # Option to localize the gain
    if loc is not None:
        kcov = np.multiply(kcov,loc)

    # Kalman gain
    kmat = np.divide(kcov, kdenom)

    # update ensemble mean
    xam = xbm + np.multiply(kmat,innov)

    # update the ensemble members using the square-root approach
    beta = 1./(1. + np.sqrt(ob_err/(varye+ob_err)))
    kmat = np.multiply(beta,kmat)
    ye   = np.array(ye)[np.newaxis]
    kmat = np.array(kmat)[np.newaxis]
    Xap  = Xbp - np.dot(kmat.T, ye)

    # full state
    Xa = np.add(xam[:,None], Xap)

    # if masked array, making sure that fill_value = nan in the new array
    if np.ma.isMaskedArray(Xa): np.ma.set_fill_value(Xa, np.nan)

    # Return the full state
    return Xa


def global_hemispheric_means(field, lat):
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

    return gm, nhm, shm


def Kalman_optimal(Y, vR, Ye, Xb, loc_rad=None, nsvs=None, transform_only=False, verbose=False):
    ''' Kalman Filter

    Y: observation vector (p x 1)
    vR: observation error variance vector (p x 1)
    Ye: prior-estimated observation vector (p x n)
    Xbp: prior ensemble perturbation matrix (m x n)

    Originator:

    Greg Hakim
    University of Washington
    26 February 2018

    Modifications:
    11 April 2018: Fixed bug in handling singular value matrix (rectangular, not square)
    '''

    if verbose:
        print('\n all-at-once solve...\n')

    begin_time = time()

    nobs = Ye.shape[0]
    nens = Ye.shape[1]
    ndof = np.min([nobs, nens])

    if verbose:
        print('number of obs: '+str(nobs))
        print('number of ensemble members: '+str(nens))

    # ensemble prior mean and perturbations
    xbm = Xb.mean(axis=1)
    #Xbp = Xb - Xb.mean(axis=1,keepdims=True)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    #  R = np.diag(vR)
    Risr = np.diag(1./np.sqrt(vR))
    # (suffix key: m=ensemble mean, p=perturbation from ensemble mean; f=full value)
    # keepdims=True needed for broadcasting to work; (p,1) shape rather than (p,)
    Yem = Ye.mean(axis=1, keepdims=True)
    Yep = Ye - Yem
    Htp = np.dot(Risr, Yep)/np.sqrt(nens-1)
    Htm = np.dot(Risr, Yem)
    Yt = np.dot(Risr, Y)
    # numpy svd quirk: V is actually V^T!
    U, s, V = np.linalg.svd(Htp, full_matrices=True)
    if not nsvs:
        nsvs = len(s) - 1
    if verbose:
        print('ndof :'+str(ndof))
        print('U :'+str(U.shape))
        print('s :'+str(s.shape))
        print('V :'+str(V.shape))
        print('recontructing using ' + str(nsvs) + ' singular values')

    innov = np.dot(U.T, Yt-np.squeeze(Htm))
    # Kalman gain
    Kpre = s[0:nsvs]/(s[0:nsvs]*s[0:nsvs] + 1)
    K = np.zeros([nens, nobs])
    np.fill_diagonal(K, Kpre)
    # ensemble-mean analysis increment in transformed space
    xhatinc = np.dot(K, innov)
    # ensemble-mean analysis increment in the transformed ensemble space
    xtinc = np.dot(V.T, xhatinc)/np.sqrt(nens-1)
    if transform_only:
        xam = []
        Xap = []
    else:
        # ensemble-mean analysis increment in the original space
        xinc = np.dot(Xbp, xtinc)
        # ensemble mean analysis in the original space
        xam = xbm + xinc

        # transform the ensemble perturbations
        lam = np.zeros([nobs, nens])
        np.fill_diagonal(lam, s[0:nsvs])
        tmp = np.linalg.inv(np.dot(lam, lam.T) + np.identity(nobs))
        sigsq = np.identity(nens) - np.dot(np.dot(lam.T, tmp), lam)
        sig = np.sqrt(sigsq)
        T = np.dot(V.T, sig)
        Xap = np.dot(Xbp, T)
        # perturbations must have zero mean
        #Xap = Xap - Xap.mean(axis=1,keepdims=True)
        if verbose:
            print('min s:', np.min(s))
    elapsed_time = time() - begin_time
    if verbose:
        print('shape of U: ' + str(U.shape))
        print('shape of s: ' + str(s.shape))
        print('shape of V: ' + str(V.shape))
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    readme = '''
    The SVD dictionary contains the SVD matrices U,s,V where V
    is the transpose of what numpy returns. xtinc is the ensemble-mean
    analysis increment in the intermediate space; *any* state variable
    can be reconstructed from this matrix.
    '''
    SVD = {
        'U': U,
        's': s,
        'V': np.transpose(V),
        'xtinc': xtinc,
        'readme': readme,
    }
    return xam, Xap, SVD


def Kalman_ESRF(cfg, vY, vR, vYe, Xb_in,
                proxy_manager, X, Xb_one_aug, Xb_one_coords, verbose=False):

    if verbose:
        print('Ensemble square root filter...')

    begin_time = time()

    # number of state variables
    nx = Xb_in.shape[0]

    # augmented state vector with Ye appended
    # Xb = np.append(Xb_in, vYe, axis=0)
    Xb = Xb_one_aug

    inflate = cfg.core.inflation_fact

    # need to add code block to compute localization factor
    nobs = len(vY)
    if verbose: print('appended state...')
    for k in range(nobs):
        #if np.mod(k,100)==0: print k
        obvalue = vY[k]
        ob_err = vR[k]
        Ye = Xb[nx+k,:]
        Y = proxy_manager.sites_assim_proxy_objs[k]
        loc = cov_localization(cfg.core.loc_rad, Y, X, Xb_one_coords)
        Xa = enkf_update_array(Xb, obvalue, Ye, ob_err, loc=loc, inflate=inflate)
        Xb = Xa

    # ensemble mean and perturbations
    Xap = Xa[0:nx,:] - Xa[0:nx,:].mean(axis=1,keepdims=True)
    xam = Xa[0:nx,:].mean(axis=1)

    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    return xam, Xap, Xa

