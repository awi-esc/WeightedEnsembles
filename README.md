# WeightedEnsembles

This repository contains all code for our paper **On combining climate moodels into weighted ensembles**.

## Data

All data used in the paper is stored in **data**. We load and process it with the script src/compute_data.jl, using our Julia package [ModelWeights](https://awi-esc.github.io/ModelWeights.jl/dev/). 

The original data that we load is (all regridded to 5x5 degrees):
- CMIP6 near-surface air temperature (tas), annual timeseries from 1950-2014
- CMIP6 sea-level pressure (psl), annual timeseries from 1950-2014
- CMIP6 near-surface air temperature (tas), scenario SSP5.85, annual timeseries from 2015-2100
- ERA5 near-surface air temperature (tas) and sea-level pressure (psl), annual timeseries from 1950-2014

The data directory further contains .csv files listing all models and model runs used for the respective variable and experiment.

