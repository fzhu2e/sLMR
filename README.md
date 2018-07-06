# sLMR

This is a scripting system for the Last Millennium Reanalysis project,
or [LMR](https://atmos.washington.edu/~hakim/lmr/).

## Requirement
Python 3

## Features

+ Run LMR with [Slurm](https://slurm.schedmd.com/) on a cluster with just one command line

## How to install
Simply
```bash
pip install slmr
```
and there will be an executable command `slmr` in your `PATH`.

## Usage example of the executable command `slmr`
We need to prepare all the data and configurations required for LMR first,
then we are able to run LMR with [Slurm](https://slurm.schedmd.com/) on a cluster
with just one command line:

```bash
slrm -c config.yml -n 4 -nn hungus -rp 0 2000 -em slmr@gmail.com -x test_ccsm4

# -c config.yml: use "config.yml" as a configuration template
# -n 4 -nn hungus: run LRM with 4 threads on the node "hungus"
# -rp 0 2000: reconstruction period to be from 0 to 2000 C.E.
# -em slmr@gmail.com: notification will be sent to "slmr@gmail.com"
# -x test_ccsm4: the experiment is named as "test_ccsm4"
 ```

 For more options, please check
 ```bash
 slmr -h
 ```

## License
MIT License

Copyright (c) 2018 Feng Zhu
