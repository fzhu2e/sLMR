''' LMRt: LMR Turbo

This submodule aims to provide a package version of the LMR framework.
It is inspired by LMR_lite.py originated by Greg Hakim (Univ. of Washington),
with several additional features:
    - it supports multiprocessing
    - it is designed to be imported and used in Jupyter notebooks (or scripts)
    - it supports easer setup for different priors, proxies, and PSMs

Originator: Feng Zhu (fengzhu@usc.edu)

'''
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import yaml
import os
from dotmap import DotMap
from . import utils


class ReconJob:
    ''' A reconstruction job
    '''

    def __init__(self):
        pwd = os.path.dirname(__file__)
        with open(os.path.join(pwd, './cfg/cfg_template.yml'), 'r') as f:
            cfg_dict = yaml.load(f)
            self.cfg = DotMap(cfg_dict)

        self.proxy_manager = None
        self.X = None
        self.Xb_one = None
        self.Ye_assim = None
        self.Ye_assim_coords = None

    def load_cfg(self, cfg_filepath):
        with open(cfg_filepath, 'r') as f:
            cfg_new = yaml.load(f)

        cfg_dict = utils.update_nested_dict(self.cfg, cfg_new)
        self.cfg = DotMap(cfg_dict)

    def load_proxies(self):
        self.proxy_manager = utils.load_proxies(self.cfg)

    def load_prior(self):
        self.X, self.Xb_one = utils.load_prior(self.cfg)
