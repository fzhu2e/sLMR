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
import numpy as np
import pickle
from tqdm import tqdm

from . import utils

ProxyManager = namedtuple(
    'ProxyManager',
    ['all_proxies', 'ind_assim', 'ind_eval', 'sites_assim_proxy_objs', 'sites_eval_proxy_objs']
)

Prior = namedtuple('Prior', ['prior_dict', 'ens', 'prior_sample_indices', 'coords', 'full_state_info', 'trunc_state_info'])

Y = namedtuple('Y', ['Ye_assim', 'Ye_assim_coords', 'Ye_eval', 'Ye_eval_coords'])

DA = namedtuple('DA', ['gmt_ens_save', 'nhmt_ens_save', 'shmt_ens_save'])


class ReconJob:
    ''' A reconstruction job
    '''

    def __init__(self):
        pwd = os.path.dirname(__file__)
        with open(os.path.join(pwd, './cfg/cfg_template.yml'), 'r') as f:
            cfg_dict = yaml.load(f)
            self.cfg = DotMap(cfg_dict)
            print(f'pid={os.getpid()} >>> job.cfg created')

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
        print(f'pid={os.getpid()} >>> job.cfg updated')

    def load_proxies(self, proxies_df_filepath, metadata_df_filepath, precalib_filesdict=None,
                     seed=0, verbose=False, print_proxy_count=True):

        all_proxy_ids, all_proxies = utils.get_proxy(self.cfg, proxies_df_filepath, metadata_df_filepath,
                                                     precalib_filesdict=precalib_filesdict,
                                                     verbose=verbose)

        ind_assim, ind_eval = utils.generate_proxy_ind(self.cfg, len(all_proxy_ids), seed=seed)

        sites_assim_proxy_objs = []
        for i in ind_assim:
            sites_assim_proxy_objs.append(all_proxies[i])

        sites_eval_proxy_objs = []
        for i in ind_eval:
            sites_eval_proxy_objs.append(all_proxies[i])

        #  def pobj_generator(inds):
            #  for ind in inds:
                #  yield all_proxies[ind]

        #  def sites_assim_proxy_objs():
            #  return pobj_generator(ind_assim)

        #  def sites_eval_proxy_objs():
            #  return pobj_generator(ind_eval)

        self.proxy_manager = ProxyManager(all_proxies, ind_assim, ind_eval, sites_assim_proxy_objs, sites_eval_proxy_objs)
        print(f'pid={os.getpid()} >>> job.proxy_manager created')

        if print_proxy_count:
            assim_sites_types = {}
            #  for pobj in self.proxy_manager.sites_assim_proxy_objs():
            for pobj in self.proxy_manager.sites_assim_proxy_objs:
                if pobj.type not in assim_sites_types:
                    assim_sites_types[pobj.type] = 1
                else:
                    assim_sites_types[pobj.type] += 1

            assim_proxy_count = 0
            for pkey, pnum in sorted(assim_sites_types.items()):
                print(f'{pkey:>45s}:{pnum:5d}')
                assim_proxy_count += pnum

            print(f'{"TOTAL":>45s}:{assim_proxy_count:5d}')

    def load_prior(self, prior_filepath, seed=0, verbose=False):
        prior_dict = utils.load_netcdf(prior_filepath, verbose=verbose)
        ens, prior_sample_indices, coords, full_state_info = utils.populate_ensemble(
            prior_dict, self.cfg, seed=seed, verbose=verbose)

        self.prior = Prior(prior_dict, ens, prior_sample_indices, coords, full_state_info, full_state_info)
        print(f'pid={os.getpid()} >>> job.prior created')
        if self.cfg.prior.regrid_method:
            ens, coords, trunc_state_info = utils.regrid_prior(self.cfg, self.prior)
            self.prior = Prior(prior_dict, ens, prior_sample_indices, coords, full_state_info, trunc_state_info)
            print(f'pid={os.getpid()} >>> job.prior regridded')

    def build_ye_files(self, prior_filepath, verbose=False):
        if verbose:
            print(f'Starting Ye precalculation using prior data from {prior_filepath}')
        pass

    def load_ye_files(self, ye_filesdict, verbose=False):
        ''' Load precalculated Ye files

        Args:
            ye_filesdict (dict): e.g. {'linear': linear_filepath, 'blinear': bilinear_filepath}
            proxy_set (str): 'assim' or 'eval'
        '''
        Ye_assim, Ye_assim_coords = utils.get_ye(self.proxy_manager,
                                             self.prior.prior_sample_indices,
                                             ye_filesdict=ye_filesdict,
                                             proxy_set='assim',
                                             verbose=verbose)

        Ye_eval, Ye_eval_coords = utils.get_ye(self.proxy_manager,
                                             self.prior.prior_sample_indices,
                                             ye_filesdict=ye_filesdict,
                                             proxy_set='eval',
                                             verbose=verbose)

        self.ye = Y(Ye_assim, Ye_assim_coords, Ye_eval, Ye_eval_coords)
        print(f'pid={os.getpid()} >>> job.ye created')

    def run_da_lite(self, recon_years=None, proxy_inds=None, verbose=False):
        cfg = self.cfg
        prior = self.prior
        proxy_manager = self.proxy_manager
        Ye_assim = self.ye.Ye_assim
        Ye_assim_coords = self.ye.Ye_assim_coords

        if recon_years is None:
            yr_start = cfg.core.recon_period[0]
            yr_end = cfg.core.recon_period[1]
            recon_years = list(range(yr_start, yr_end))
        else:
            yr_start, yr_end = recon_years[0], recon_years[-1]
        print(f'pid={os.getpid()} >>> Recon. period: {yr_start}...{yr_end}')

        Xb_one = prior.ens
        grid = utils.make_grid(prior)
        nyr = len(recon_years)

        gmt_ens_save = np.zeros((nyr, grid.nens))
        nhmt_ens_save = np.zeros((nyr, grid.nens))
        shmt_ens_save = np.zeros((nyr, grid.nens))

        for yk, target_year in enumerate(tqdm(recon_years, desc='KF updating')):
            gmt_ens_save[yk], nhmt_ens_save[yk], shmt_ens_save[yk] = utils.update_year_lite(
                target_year, cfg, Xb_one, grid, proxy_manager, Ye_assim, Ye_assim_coords, verbose=verbose)

        self.da = DA(gmt_ens_save, nhmt_ens_save, shmt_ens_save)
        print(f'pid={os.getpid()} >>> job.da created')

    def run_da(self, recon_years=None, proxy_inds=None, verbose=False):
        cfg = self.cfg
        prior = self.prior
        proxy_manager = self.proxy_manager

        Ye_assim = self.ye.Ye_assim
        Ye_assim_coords = self.ye.Ye_assim_coords
        assim_proxy_count = np.shape(Ye_assim)[0]

        Ye_eval = self.ye.Ye_eval
        Ye_eval_coords = self.ye.Ye_eval_coords
        eval_proxy_count = np.shape(Ye_eval)[0]

        ibeg_tas = prior.trunc_state_info['tas_sfc_Amon']['pos'][0]
        iend_tas = prior.trunc_state_info['tas_sfc_Amon']['pos'][1]

        if recon_years is None:
            yr_start = cfg.core.recon_period[0]
            yr_end = cfg.core.recon_period[1]
            recon_years = list(range(yr_start, yr_end))
        else:
            yr_start, yr_end = recon_years[0], recon_years[-1]
        print(f'pid={os.getpid()} >>> Recon. period: {yr_start}...{yr_end}')

        Xb_one = prior.ens
        Xb_one_aug = np.append(Xb_one, Ye_assim, axis=0)
        Xb_one_aug = np.append(Xb_one_aug, Ye_eval, axis=0)
        Xb_one_coords = np.append(prior.coords, Ye_assim_coords, axis=0)
        Xb_one_coords = np.append(Xb_one_coords, Ye_eval_coords, axis=0)

        grid = utils.make_grid(prior)
        nyr = len(recon_years)

        gmt_ens_save = np.zeros((nyr, grid.nens))
        nhmt_ens_save = np.zeros((nyr, grid.nens))
        shmt_ens_save = np.zeros((nyr, grid.nens))

        for yr_idx, target_year in enumerate(tqdm(recon_years, desc='KF updating')):
            gmt_ens_save[yr_idx], nhmt_ens_save[yr_idx], shmt_ens_save[yr_idx] = utils.update_year(
                yr_idx, target_year,
                cfg, Xb_one_aug, Xb_one_coords, prior, proxy_manager.sites_assim_proxy_objs,
                assim_proxy_count, eval_proxy_count, grid,
                ibeg_tas, iend_tas,
                verbose=verbose
            )

        self.da = DA(gmt_ens_save, nhmt_ens_save, shmt_ens_save)
        print(f'pid={os.getpid()} >>> job.da created')

    def run(self, prior_filepath, db_proxies_filepath, db_metadata_filepath,
            recon_years=None, seed=0, precalib_filesdict=None, ye_filesdict=None,
            verbose=False, print_proxy_count=False, save_dirpath=None, mode='normal'):

        self.load_prior(prior_filepath, verbose=verbose, seed=seed)
        self.load_proxies(db_proxies_filepath, db_metadata_filepath, precalib_filesdict=precalib_filesdict,
                          verbose=verbose, seed=seed, print_proxy_count=print_proxy_count)
        self.load_ye_files(ye_filesdict=ye_filesdict, verbose=verbose)

        if mode == 'lite':
            self.run_da_lite(recon_years=recon_years, verbose=verbose)
        else:
            self.run_da(recon_years=recon_years, verbose=verbose)

        if save_dirpath:
            os.makedirs(save_dirpath, exist_ok=True)
            save_path = os.path.join(save_dirpath, f'job_r{seed:02d}.pkl')
            print(f'pid={os.getpid()} >>> Saving job.da to: {save_path}')
            with open(save_path, 'wb') as f:
                pickle.dump([self.cfg, self.da], f)
