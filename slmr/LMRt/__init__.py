''' LMRt: LMR Turbo

This submodule aims to provide a package version of the LMR framework.
It is inspired by LMR_lite.py originated by Greg Hakim (Univ. of Washington),
with several additional features:
    - it supports multiprocessing
    - it is designed to be imported and used in Jupyter notebooks (or scripts)
    - it supports easer setup for different priors, proxies, and PSMs

Originator: Feng Zhu (fengzhu@usc.edu)

'''
#  from pathos.multiprocessing import ProcessingPool as Pool
import yaml
import os
from dotmap import DotMap
from collections import namedtuple

from . import utils


class ReconJob:
    ''' A reconstruction job
    '''

    def __init__(self):
        pwd = os.path.dirname(__file__)
        with open(os.path.join(pwd, './cfg/cfg_template.yml'), 'r') as f:
            cfg_dict = yaml.load(f)
            self.cfg = DotMap(cfg_dict)
            print('>>> self.cfg created')

        proxy_db_cfg = {
            'LMRdb': self.cfg.proxies.LMRdb,
        }

        for db_name, db_cfg in proxy_db_cfg.items():
            db_cfg.proxy_type_mapping = {}
            for ptype, measurements in db_cfg.proxy_assim2.items():
                # Fetch proxy type name that occurs before underscore
                type_name = ptype.split('_', 1)[0]
                for measure in measurements:
                    db_cfg.proxy_type_mapping[(type_name, measure)] = ptype

    def load_cfg(self, cfg_filepath):
        with open(cfg_filepath, 'r') as f:
            cfg_new = yaml.load(f)

        cfg_dict = utils.update_nested_dict(self.cfg, cfg_new)
        self.cfg = DotMap(cfg_dict)
        print('>>> self.cfg updated')

    def load_proxies(self, proxies_df_filepath, metadata_df_filepath, seed=0):
        all_proxy_ids, all_proxies = utils.get_proxy(self.cfg, proxies_df_filepath, metadata_df_filepath)
        ind_assim, ind_eval = utils.generate_proxy_ind(self.cfg, all_proxy_ids, seed=seed)

        ProxyManager = namedtuple('ProxyManager', ['ind_assim', 'ind_eval', 'all_proxies'])
        self.proxy_manager = ProxyManager(ind_assim, ind_eval, all_proxies)
        print('>>> self.proxy_manager created')

    def load_prior(self, prior_filepath, seed=0, verbose=False):
        prior_dict = utils.load_netcdf(prior_filepath, verbose=verbose)
        ens, prior_sample_indices, coords = utils.populate_ensemble(
            prior_dict, self.cfg, seed=seed, verbose=verbose)

        Prior = namedtuple('Prior', ['ens', 'prior_sample_indices', 'coords'])
        self.prior = Prior(ens, prior_sample_indices, coords)
        print('>>> self.prior created')

    def build_ye_file(self, prior_filepath, verbose=False):
        if verbose:
            print(f'Starting Ye precalculation using prior data from {prior_filepath}')
        pass

    def load_ye_file(self, ye_filepath, verbose=False):
        if verbose:
            print(f'Loading precalculated Ye from {ye_filepath}')
        pass
