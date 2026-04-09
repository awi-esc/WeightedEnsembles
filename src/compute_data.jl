import ModelWeights as mw
import ModelWeights.Data as mwd
import ModelWeights.Timeseries as mwt
import ModelWeights.Weights as mww
import ModelWeights.Plots as mwp

using CSV
using DataFrames
using DimensionalData
using Statistics
using YAXArrays

# --------------------- Load model data --------------------- #
data_dir = "./data/"
base_dir = "/albedo/work/projects/p_forclima/preproc_data_esmvaltool"
paths_data = joinpath.(
    base_dir,
    ["historical/recipe_cmip6_historical_tas_timeseries_20250228_081213/preproc/historical/tas_CLIM-ann",
     "ssp585/recipe_cmip6_ssp585_tas_timeseries_20250226_074213/preproc/ssp585/tas_CLIM-ann",
     "historical/recipe_cmip6_historical_psl_timeseries_20250313_130922/preproc/historical/psl_CLIM-ann"
    ]
)
data_hist_proj_members = mwd.defineDataMap(
    paths_data, 
    ["tas_annual_historical", "tas_annual_ssp585", "psl_annual_historical"]; 
    filename_format = :esmvaltool_cmip6,
    constraint = Dict("level_shared" => "model")
)
model_data = mwd.summarizeMembers(data_hist_proj_members)

ecs_data_csv = DataFrame(CSV.File(joinpath(data_dir, "ecs", "ecs-unique.csv")))
# Just cmip6 models
ecs_data_csv = filter(row -> row.mip == "CMIP6", ecs_data_csv)
ecs_data = YAXArray(
    (Dim{:model}(String.(ecs_data_csv[!,:model])),),
    ecs_data_csv[!, :ECS]
)
ecs_models = ecs_data.model

# --------------------- Process model data --------------------- #
mwd.apply!(
    model_data, mwt.filterTimeseries, 1950, 2014; 
    ids = ["tas_annual_historical", "psl_annual_historical"]
)
mwd.apply!(model_data, mwt.filterTimeseries, 2015, 2100; ids = ["tas_annual_ssp585"])


mwd.apply!(model_data, mwt.filterTimeseries, 1980, 2014; 
    ids = ["tas_annual_historical", "psl_annual_historical"], 
    ids_new = ["tas_annual_diagnostic", "psl_annual_diagnostic"]
)
mwd.apply!(model_data, mwt.filterTimeseries, 1995, 2014; 
    ids = ["tas_annual_historical"], ids_new =  ["tas_annual_reference"]
)
# make sure that only models available for all time periods are included:
mwd.subsetModelData!(model_data, mwd.MODEL_LEVEL)
mwd.apply!(model_data, yax ->  yax[model = Where(x -> x in ecs_models)])

# save timeseries data, missing values need to be replaced by NaNs to save data
df = mwd.apply(model_data, x -> coalesce.(x, NaN))
savecube(df["tas_annual_historical"], joinpath(data_dir, "timeseries", "models_tas_annual_historical.nc"); driver=:netcdf, layername="tas_annual_historical")
savecube(df["psl_annual_historical"], joinpath(data_dir, "timeseries", "models_psl_annual_historical.nc"); driver=:netcdf, layername="psl_annual_historical")
savecube(df["tas_annual_ssp585"], joinpath(data_dir, "timeseries", "models_tas_annual_ssp585.nc"); driver=:netcdf, layername="tas_annual_ssp585")

# Write used models to CSV files
# also add used model runs per model
models = Array(model_data["tas_annual_historical"].model)
CSV.write(joinpath(data_dir, "models.csv"), DataFrame(model=models))

data = mwd.apply(data_hist_proj_members, mwd.subsetModelData, models)

df1 = mwd.listModels(data["psl_annual_historical"])
df2 = mwd.listModels(data["tas_annual_historical"])
df3 = mwd.listModels(data["tas_annual_ssp585"])

CSV.write(joinpath(data_dir, "members-psl-historical.csv"), df1)
CSV.write(joinpath(data_dir, "members-tas-historical.csv"), df2)
CSV.write(joinpath(data_dir, "members-tas-ssp585.csv"), df3)

# --------------------- Load observational data (ERA5) --------------------- #
base_dir = "/albedo/work/projects/p_forclima/preproc_data_esmvaltool/obs/recipe_ERA5_20250718_180812/preproc/historical"
data_dirs = ["psl_CLIM-ann", "tas_CLIM-ann"]
obs_era5 = mw.defineDataMap(
    joinpath.(base_dir, data_dirs),
    data_dirs;
    dtype = "observations",
    filename_format = :esmvaltool,
    constraint = Dict(
        "variables" => ["tas", "psl"]
    )
)
obs_data = mwd.apply(obs_era5, mwd.setDim, :model, nothing, ["ERA5"])
mwd.renameDict!(obs_data, ["psl_CLIM-ann", "tas_CLIM-ann"], ["psl_annual", "tas_annual"])

mwd.apply!(obs_data, mwt.filterTimeseries, 1950, 2014; ids = ["tas_annual", "psl_annual"])

# save timeseries data:
df_obs = mwd.apply(obs_data, x -> coalesce.(x, NaN))
savecube(df_obs["tas_annual"], joinpath(data_dir, "timeseries", "obs_tas_annual_historical.nc"); driver=:netcdf, layername="tas_annual")
savecube(df_obs["psl_annual"], joinpath(data_dir, "timeseries", "obs_psl_annual_historical.nc"); driver=:netcdf, layername="psl_annual")
        
# --------------------- Process observational data (ERA5) --------------------- #
mwd.apply!(obs_data, mwt.filterTimeseries, 1980, 2014; 
    ids = ["tas_annual", "psl_annual"], 
    ids_new = ["tas_annual_diagnostic", "psl_annual_diagnostic"]
)
mwd.apply!(obs_data, mwt.filterTimeseries, 1995, 2014; 
    ids = ["tas_annual"], ids_new = ["tas_annual_reference"]
)

# --------------------- Compute diagnostics --------------------- #
for (_, dm) in enumerate([model_data, obs_data])
    mwd.apply!(
        dm, mwd.climatology;
        ids = ["tas_annual_diagnostic", "psl_annual_diagnostic"],
        ids_new = ["tas_CLIM", "psl_CLIM"]
    )
    mwd.apply!(
        dm, mwd.anomaliesGM;
        ids = ["tas_CLIM", "psl_CLIM"],
        ids_new = ["tas_ANOM-GM", "psl_ANOM-GM"]
    )
end
diagnostic_ids = ["tas_ANOM-GM", "psl_ANOM-GM"];
obs_diagnostics = mwd.apply(mwd.subsetDataMap(obs_data, diagnostic_ids), x -> coalesce.(x, NaN))
model_diagnostics = mwd.apply(mwd.subsetDataMap(model_data, diagnostic_ids), x -> coalesce.(x, NaN))
# save observational diagnostic data
savecube(obs_diagnostics["tas_ANOM-GM"], joinpath(data_dir, "diagnostics", "obs_tas_ANOM-GM_1980-2014.nc"); driver=:netcdf, layername="tas_ANOM-GM")
savecube(obs_diagnostics["psl_ANOM-GM"], joinpath(data_dir, "diagnostics", "obs_psl_ANOM-GM_1980-2014.nc"); driver=:netcdf, layername="psl_ANOM-GM")

# save model diagnostic data
savecube(model_diagnostics["tas_ANOM-GM"], joinpath(data_dir, "diagnostics", "models_tas_ANOM-GM_1980-2014.nc"); driver=:netcdf, layername="tas_ANOM-GM")
savecube(model_diagnostics["psl_ANOM-GM"], joinpath(data_dir, "diagnostics", "models_psl_ANOM-GM_1980-2014.nc"); driver=:netcdf, layername="tas_ANOM-GM")

# --------------------- Process projection data --------------------- #
ids_ts = ["tas_annual_historical", "tas_annual_ssp585", "tas_annual_reference"]
projections = mwd.subsetDataMap(model_data, ids_ts)

# timeseries global means of tas data
ids_annual_gms = map(id -> replace(id, "annual" => "annual-GM"), ids_ts)
mwd.apply!(projections, mwd.globalMeans; ids = ids_ts, ids_new = ids_annual_gms)

# timeseries anomalies of global means of tas data with respect to reference period 
tas_gm_ref = mean(projections["tas_annual-GM_reference"], dims=:time)[time = 1]

for id_gm in ids_annual_gms
    mwd.apply!(
        projections, mwd.anomalies, tas_gm_ref;
        ids = [id_gm],
        ids_new = [replace(id_gm, "GM" => "GM-ANOM")]
    )
end

savecube(projections["tas_annual-GM-ANOM_historical"], joinpath(data_dir, "timeseries-projection-plot", "model_tas_gms-anomalies-ref_historical.nc"); driver=:netcdf, layername="tas_annual-GM-ANOM_historical")
savecube(projections["tas_annual-GM-ANOM_ssp585"], joinpath(data_dir, "timeseries-projection-plot", "model_tas_gms-anomalies-ref_ssp585.nc"); driver=:netcdf, layername="tas_annual-GM-ANOM_ssp585")

# for observational data also save the anomalies of the annual climatologies wrt the 
# reference time period from 1995-2014:
mean_ref = dropdims(mean(obs_data["tas_annual_reference"], dims=:time); dims=:time)
obs_data["tas_ANOM-ann"] = mwd.anomalies(obs_data["tas_annual"], mean_ref)
obs_data["tas_ANOM-ann-GM"] = mwd.globalMeans(obs_data["tas_ANOM-ann"])

savecube(obs_data["tas_ANOM-ann-GM"], joinpath(data_dir, "timeseries-projection-plot", "obs_tas_gms-anomalies-ref.nc"); driver=:netcdf, layername="tas_ANOM-ann-GM")


