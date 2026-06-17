# WeightedEnsembles

This repository contains all code for our paper **On combining climate models into weighted ensembles** ([see manuscript](manuscript/submission-on-combining-climate-models-into-weighted-ensembles.pdf)).

## Data

We load and process the original data with the script `src/compute-data.jl`, using our Julia package [ModelWeights](https://awi-esc.github.io/ModelWeights.jl/dev/). 

The original data that we load is (all regridded to 5x5 degrees):
- CMIP6 near-surface air temperature (tas), annual timeseries from 1950-2014
- CMIP6 sea-level pressure (psl), annual timeseries from 1950-2014
- CMIP6 near-surface air temperature (tas), scenario SSP5.85, annual timeseries from 2015-2100
- ERA5 near-surface air temperature (tas) and sea-level pressure (psl), annual timeseries from 1950-2014

All processed data used in the paper is stored in `data`, which further contains .csv files listing all models and model runs used for the respective variable and experiment.


## Reproduce figures

Activate and instantiate the julia project (WeightedEnsembles.jl) by running the following from the top-level directory (or use --project=path/to/WeightedEnsembles) where `-e` evaluates the following expression as julia code:

``julia --project=. -e "using Pkg; Pkg.instantiate()"``

Then run the following commands to create the figures from the paper for the respective sections:

- `julia --project=. section3-priors.jl`
- `julia --project=. section4-combined-diagnostics.jl`
- `julia --project=. section6-ecs.jl`

All figures are stored in a subdirectory `plots` of the `plot_dir` specified in `src/config.jl`. The default name for the created directory is `output`.

## Preprocess data 

To recompute the data that we use, you can use the script `src/compute-data.jl`. Update the paths where mentioned to point to your raw data. The generated data will be stored in the `data_dir` specified in `src/config.jl`. So, to not overwrite our provided preprocessed data, this path must be changed.

## Other

The Manifest.toml file is published for reproducibility, as it contains the exact status of the julia environment that can be instantiated by running `using Pkg; Pkg.instantiate()`.
