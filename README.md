# WeightedEnsembles

This repository contains all code for our paper **On combining climate models into weighted ensembles**.

## Data

We load and process the original data with the script `src/compute-data.jl`, using our Julia package [ModelWeights](https://awi-esc.github.io/ModelWeights.jl/dev/). 

The original data that we load is (all regridded to 5x5 degrees):
- CMIP6 near-surface air temperature (tas), annual timeseries from 1950-2014
- CMIP6 sea-level pressure (psl), annual timeseries from 1950-2014
- CMIP6 near-surface air temperature (tas), scenario SSP5.85, annual timeseries from 2015-2100
- ERA5 near-surface air temperature (tas) and sea-level pressure (psl), annual timeseries from 1950-2014

All processed data used in the paper is stored in `data`, which further contains .csv files listing all models and model runs used for the respective variable and experiment.

## Plots

To reproduce the figures from our paper, run the script for the respective section from the top level directory, e.g.:

``julia --project=. src/section3-priors.jl``

The plots are stored in `output/plots/`. 


