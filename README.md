[![PyPI](https://img.shields.io/pypi/v/slmr.svg)]()

# sLMR

This is a scripting system for the Last Millennium Reanalysis project,
or [LMR](https://atmos.washington.edu/~hakim/lmr/).

## Package dependencies
+ [p2k](https://github.com/fzhu2e/p2k): A collection of functions used for PAGES2k dataset analysis (`pip install p2k`)
+ [tqdm](https://github.com/tqdm/tqdm): A fast, extensible progress bar for Python and CLI (`pip install tqdm`)
+ [prysm-api](https://github.com/fzhu2e/prysm-api): The API for PRoxY System Modeling (PRYSM) (`pip install prysm-api`)
+ [dotmap](https://github.com/drgrib/dotmap): Dot access dictionary with dynamic hierarchy creation and ordered iteration (`pip install dotmap`)
+ [xarray](https://github.com/pydata/xarray): N-D labeled arrays and datasets in Python (`pip install xarray`)
+ [netCDF4](https://github.com/Unidata/netcdf4-python): the python interface for netCDF4 format (`conda install netCDF4`)
+ [pyspharm](https://code.google.com/archive/p/pyspharm/): an  object-oriented python interface to the NCAR SPHEREPACK library (`conda install -c conda-forge pyspharm`)

## Features

+ [v0.1] Running LMR with [Slurm](https://slurm.schedmd.com/) on a cluster with just one command line
+ [v0.2] Post-processing: pick files with the same filename from different directories
+ [v0.3] LMR Turbo (LMRt): the packaged version of the LMR framework with multiprocessing support 
    + [a quickstart notebook](https://nbviewer.jupyter.org/github/fzhu2e/sLMR/blob/master/notebooks/01.lmrt_quickstart.ipynb)
    + [a notebook on building Ye files](https://nbviewer.jupyter.org/github/fzhu2e/sLMR/blob/master/notebooks/02.build_Ye.ipynb)

## How to install
Simply
```bash
pip install slmr
```
and there will be an executable command `slmr` in your `PATH`.

## Usage examples
Below are some usage examples of `slmr`.
For more details, please check
 ```bash
 slmr -h
 ```

### Running LMR with Slurm
We need to prepare all the data and configurations required for LMR first,
then we are able to run LMR with [Slurm](https://slurm.schedmd.com/) on a cluster
with just one command line:

```bash
slmr job -c config.yml -n 4 -nn hungus -rp 0 2000 -em slmr@gmail.com -x test_ccsm4

# slmr job: use the mode of submitting a slurm job
# -c config.yml: use "config.yml" as a configuration template
# -n 4 -nn hungus: run LRM with 4 threads on the node "hungus"
# -rp 0 2000: reconstruction period to be from 0 to 2000 C.E.
# -em slmr@gmail.com: notification will be sent to "slmr@gmail.com"
# -x test_ccsm4: the experiment is named as "test_ccsm4"
 ```

### Post-processing: pick files
```bash
slmr pp pick_files -f gmt.npz -d dir1 dir2 -s ./gmt_files

# slmr pp: use the mode of post-processing
# pick_files: the post-processing task to be pick_files
# -f gmt.npz: pick files named as "gmt.npz"
# -d dir1 dir2: pick files from the specified directories dir1 and dir2; more dirs can be followed
# -s ./gmt_files: save the found files to the directory "./gmt_files"
```

## License
MIT License

Copyright (c) 2018 Feng Zhu
