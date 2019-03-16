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


def load_netcdf(filepath):
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


def populate_ensemble(datadict, cfg, iter_ind=0, verbose=False):
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

    random.seed(cfg.wrapper.multi_seed[iter_ind])
    ind_ens = random.sample(list(range(ntime)), cfg.core.nens)

    if verbose:
        print(f'shape of Xb: ({Nx} x {cfg.core.nens})')
        print('seed=', cfg.wrapper.multi_seed[iter_ind])
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

    return Xb_res
