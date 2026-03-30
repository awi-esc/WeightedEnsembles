import ModelWeights as mw
import ModelWeights.Data as mwd
import ModelWeights.Timeseries as mwt
import ModelWeights.Weights as mww
import ModelWeights.Plots as mwp

using CSV
using DataFrames
# using DimensionalData
# using Statistics
# using YAXArrays

data_dir = "./data/"
target_data_dir = "./output/data";
plot_dir = "./output/plots";

# get model data
model_ids = mwd.loadModelsFromCSV(
    joinpath(data_dir, "cmip6-models-brunner-et-al.csv"), "Model"
);
member_ids = mwd.loadModelsFromCSV(
    joinpath(data_dir, "cmip6-models-brunner-et-al.csv"), "Model"; col_variants = "Variants"
);
# apparently, CESM2 r1i1p1f1 and r2i1pf1f1 were replaced by newer runs because of a bug in the first version, 
# they are now called: (see https://bb.cgd.ucar.edu/cesm/threads/query-about-cmip6-cesm2-model-under-ssp585.5718/)
push!(member_ids, "CESM2#r10i1p1f1")
push!(member_ids, "CESM2#r11i1p1f1")

# to use same models for ECS based weights and historical based,  filter out "FIO-ESM-2-0"
# since ECS value missing for it
#filter!(x -> !startswith(x, "FIO-ESM-2-0" * mwd.MODEL_MEMBER_DELIM), member_ids)

# TODO: check original data
model_data =  mw.defineDataMap(
    #"./work/configs/climwip.yml"; 
    "/Users/brgrus001/ModelWeightsPaper/work/configs/climwip.yml",
    dtype = "cmip", 
    constraint = Dict("models" => member_ids, "level_shared" => "member")
)

# all members found
members_found = vcat(map(x -> sort(lookup(model_data[x], :member)), collect(keys(model_data)))...);
members_found = unique(map(x -> split(x, "_")[1], members_found));
# all members found across all datasets
members_found = mwd.sharedLevelMembers(model_data);
members_found = map(x -> split(x, "_")[1], members_found);

models_found = mwd.modelsFromMemberIDs(members_found; uniq=true);
# members/models used in Brunner et al. that weren't found
filter(x -> !(x in members_found), member_ids)
filter(x -> !(x in models_found), model_ids)

filter(x -> startswith(x, "CESM2#"), members_found)

mwd.apply!(model_data, mwt.filterTimeseries, 2014, 2100; 
    ids = ["tas_CLIM-ann_ssp126", "tas_CLIM-ann_ssp585"]
)
mwd.apply!(model_data, mwt.filterTimeseries, 1980, 2014; 
    ids = ["tas_CLIM-ann_historical", "psl_CLIM-ann_historical"], 
    ids_new = ["tas_CLIM-ann_diagnostic", "psl_CLIM-ann_diagnostic"]
)
mwd.apply!(model_data, mwt.filterTimeseries, 1995, 2014; 
    ids = ["tas_CLIM-ann_historical"], 
    ids_new = ["tas_CLIM-ann_reference"]
)

# get observational data (ERA5)
base_dir = "/albedo/work/projects/p_forclima/preproc_data_esmvaltool/obs/recipe_ERA5_20250718_180812/preproc/historical"
data_dirs = ["psl_CLIM-ann", "psl_CLIM", "tas_CLIM-ann", "tas_CLIM"]
obs_era5 = mw.defineDataMap(
    joinpath.(base_dir, data_dirs),
    data_dirs;
    dtype = "observations",
    filename_format = :esmvaltool,
    constraint = Dict(
        "variables" => ["tas", "psl"]
    )
)
obs_era5 = mwd.apply(obs_era5, mwd.setDim, :model, nothing, ["ERA5"])
# MERRA2 data
base_dir = "/albedo/work/projects/p_forclima/preproc_data_esmvaltool/obs/recipe_MERRA2_20250918_114516/preproc/historical"
obs_merra2 = mw.defineDataMap(
    joinpath.(base_dir, data_dirs),
    data_dirs;
    dtype = "observations",
    filename_format = :esmvaltool,
    constraint = Dict(
        "variables" => ["tas", "psl"]
    )
)
obs_merra2 = mwd.apply(obs_merra2, mwd.setDim, :model, nothing, ["MERRA2"])
# TODO: join 2 observational dataests, get ERA5 data up to 2022
obs_data = obs_era5

mwd.apply!(obs_data, mwt.filterTimeseries, 1980, 2014; 
    ids = ["tas_CLIM-ann", "psl_CLIM-ann"], 
    ids_new = ["tas_CLIM-ann_diagnostic", "psl_CLIM-ann_diagnostic"]
)
mwd.apply!(obs_data, mwt.filterTimeseries, 1995, 2014; 
    ids = ["tas_CLIM-ann"], 
    ids_new = ["tas_CLIM-ann_reference"]
)
# for observational data also save the anomalies of the annual climatologies wrt the 
# reference time period from 1995-2014:
mean_ref = dropdims(mean(obs_data["tas_CLIM-ann_reference"], dims=:time); dims=:time)
obs_data["tas_ANOM-ann"] = mwd.anomalies(obs_data["tas_CLIM-ann"], mean_ref)
obs_data["tas_GM-ANOM-ann"] = mwd.globalMeansNoMissing(obs_data["tas_ANOM-ann"])
obs_ids = filter(x -> !(endswith(x, "_diagnostic")), collect(keys(obs_data)))
df = mwd.subsetDataMap(obs_data, obs_ids)
mwd.writeDataToDisk(df, joinpath(target_data_dir, "obs-data.jld2"))

# Compute diagnostics for computing model weights
for (_, dm) in enumerate([model_data, obs_data])
    mwd.apply!(
        dm, mwd.climatology;
        ids = ["tas_CLIM-ann_diagnostic", "psl_CLIM-ann_diagnostic"],
        ids_new = ["tas_CLIM_diagnostic", "psl_CLIM_diagnostic"]
    )
    mwd.apply!(
        dm, mwd.anomaliesGM;
        ids = ["tas_CLIM_diagnostic", "psl_CLIM_diagnostic"],
        ids_new = ["tas_ANOM_diagnostic", "psl_ANOM_diagnostic"]
    )
    mwd.apply!(
        dm, mwt.linearTrend;
        ids = ["tas_CLIM-ann_diagnostic", "psl_CLIM-ann_diagnostic"], 
        ids_new = ["tas_TREND_diagnostic", "psl_TREND_diagnostic"]
    )
    mwd.apply!(
        dm, mwt.detrend;
        ids = ["tas_CLIM-ann_diagnostic", "psl_CLIM-ann_diagnostic"],
        ids_new = ["tas_CLIM-ann-detrended_diagnostic", "psl_CLIM-ann-detrended_diagnostic"]
    )

    for v in ["tas", "psl"]
        dm[v * "_STD_diagnostic"] = mapslices(
            x -> Statistics.std(x), dm[v * "_CLIM-ann-detrended_diagnostic"], dims=(:time,)
        )
    end
end

diagnostic_ids_perform = [
    "tas_TREND_diagnostic", "tas_ANOM_diagnostic", "psl_ANOM_diagnostic",
    "tas_STD_diagnostic", "psl_STD_diagnostic"
];
diagnostic_ids_indep = ["tas_CLIM_diagnostic", "psl_CLIM_diagnostic"];
model_diagnostics = mwd.subsetDataMap(model_data, vcat(diagnostic_ids_indep, diagnostic_ids_perform))
obs_diagnostics = mwd.subsetDataMap(obs_data, diagnostic_ids_perform)

# shorten ids:
ids_perform = map(x -> String(split(x, "_diagnostic")[1]), diagnostic_ids_perform)
ids_indep = map(x -> String(split(x, "_diagnostic")[1]), diagnostic_ids_indep)
mwd.renameDict!(obs_diagnostics, diagnostic_ids_perform, ids_perform)
mwd.renameDict!(model_diagnostics, vcat(diagnostic_ids_indep, diagnostic_ids_perform), vcat(ids_indep, ids_perform))

# Get projection data
ids_clims = ["tas_CLIM-ann_historical", "tas_CLIM-ann_ssp126", "tas_CLIM-ann_ssp585", "tas_CLIM-ann_reference"]
ids_gms = map(id -> replace(id, "CLIM" => "GM"), ids_clims)
projections = mwd.subsetDataMap(model_data, ids_clims)
mwd.apply!(projections, mwd.globalMeansNoMissing; ids = ids_clims, ids_new = ids_gms)
mwd.summarizeMembers!(projections)
# add anomalies to projection data for Global means and spatial field
tas_gm_ref = mean(projections["tas_GM-ann_reference"], dims=:time)[time = 1]
tas_clim_ref = mean(projections["tas_CLIM-ann_reference"], dims=:time)[time = 1]
for (id_gm, id_clim) in zip(ids_gms, ids_clims)
    mwd.apply!(
        projections, mwd.anomalies, tas_gm_ref;
        ids = [id_gm],
        ids_new = [replace(id_gm, "GM" => "GM-ANOM")]
    )
    mwd.apply!(
        projections, mwd.anomalies, tas_clim_ref;
        ids = [id_clim],
        ids_new = [replace(id_clim, "CLIM" => "ANOM")]
    )
end
data = Dict(
    "diagnostic_data" => Dict("models" => model_diagnostics, "observations" => obs_diagnostics),
    "projections" => projections
)
mwd.writeDataToDisk(data, joinpath(target_data_dir, "data.jld2"))











# diagnostics = collect(keys(climwip_weights_orig.config.performance))

# data = mwd.readDataFromDisk(joinpath(data_dir, "example-brunner_data.jld2"))
# model_data_orig = mwd.convertToYAX(data["diagnostic_data"]["models"]);

# model_data = data["diagnostic_data"]["models"]
# mwd.summarizeMembers!(model_data)
# model_data = mwd.convertToYAX(model_data)[diagnostic = Where(x -> x in diagnostics)]
# models = collect(lookup(model_data, :model))
# N_models = length(models)


# obs_data_orig = mwd.convertToYAX(data["diagnostic_data"]["observations"])
# obs_data = mwd.convertToYAX(data["diagnostic_data"]["observations"])[model=1, diagnostic = Where(x -> x in diagnostics)]

# # standardize data by observational standard deviation
# std_obs = std(obs_data, dims=(:lon, :lat))
# std_obs = dropdims(std_obs; dims=(:lon, :lat))
# model_data_standardized = model_data ./ std_obs
# obs_data_standardized = obs_data ./ std_obs

# w_diagnostics = YAXArray(
#     (Dim{:diagnostic}(diagnostics),),
#     Float64.(collect(values(climwip_weights_orig.config.performance)))
# )

# # z-standardize data 
# mean_models = dropdims(mean(model_data, dims=(:lon, :lat)); dims=(:lon, :lat))
# std_models = dropdims(std(model_data, dims=(:lon, :lat)); dims = (:lon, :lat))
# model_data_zstd = (model_data .- mean_models) ./ std_models

# mean_obs = dropdims(mean(obs_data, dims=(:lon, :lat)); dims=(:lon, :lat))
# std_obs = dropdims(std(obs_data, dims=(:lon, :lat)); dims=(:lon, :lat))
# obs_data_zstd = (obs_data .- mean_obs) ./ std_obs


# s = size(model_data)
# other_idxs = collect(dimnum(model_data, otherdims(model_data, (:model, :diagnostic))))
# n_grid_points = reduce(*, s[other_idxs])
# latitudes = collect(lookup(model_data, :lat));

# members_data = model_data_orig[diagnostic = Where(x -> x in diagnostics)]
# N_members = size(members_data, :member)