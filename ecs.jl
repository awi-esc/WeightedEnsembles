import ModelWeights as mw
import ModelWeights.Data as mwd 
import ModelWeights.Plots as mwp
import ModelWeights.Timeseries as mwt
import ModelWeights.Weights as mww

using CairoMakie
using CSV
using DataFrames
using Dates
using Dierckx
using Random
using Statistics
using Turing
using YAXArrays

include("./config.jl")

# ----------------------------- Data ----------------------------------------- #
# Get ECS values
begin
    base_dir = "/albedo/work/projects/p_forclima/preproc_data_esmvaltool"
    ecs_data_csv = DataFrame(CSV.File(joinpath(data_dir, "ecs-unique.csv")))
    # Just cmip6 models
    ecs_data_csv = filter(row -> row.mip == "CMIP6", ecs_data_csv)
    ecs_data = YAXArray(
        (Dim{:model}(String.(ecs_data_csv[!,:model])),),
        ecs_data_csv[!, :ECS]
    )
    ecs_models = ecs_data.model

    # Target ECS-distribution (based on Sherwood et al):
    data_pdf_ecs_wd = mwd.readDataFromDisk(joinpath(target_data_dir, "ecs_data-lh-fn-wd.jld2"))
    xs = data_pdf_ecs_wd["xs"]
    ys = data_pdf_ecs_wd["ys"]
    pdf_ecs = Dierckx.Spline1D(xs, ys, k=data_pdf_ecs_wd["k"], s=data_pdf_ecs_wd["s"])
    lh_fn_ecs(x) = pdf_ecs(x)
end

# Get tas data historical and projection
begin
    paths_data = joinpath.(
        base_dir,
        ["historical/recipe_cmip6_historical_tas_timeseries_20250228_081213/preproc/historical/tas_CLIM-ann",
        "ssp585/recipe_cmip6_ssp585_tas_timeseries_20250226_074213/preproc/ssp585/tas_CLIM-ann"]
    )
    data_hist_proj_members = mwd.defineDataMap(
        paths_data, 
        ["historical-tas", "ssp585-tas"]; 
        filename_format = :esmvaltool_cmip6,
        constraint = Dict("level_shared" => "model")
    )
    data_hist_proj = mwd.summarizeMembers(data_hist_proj_members)

    # Data reference period
    data_ref = mwt.filterTimeseries(data_hist_proj["historical-tas"], 1995, 2014)
    # Data projections
    data_proj = data_hist_proj["ssp585-tas"]
    data_proj = mwt.filterTimeseries(data_proj, 2015, 2100)
    # Data historical
    data_hist = mwt.filterTimeseries(data_hist_proj["historical-tas"], 1950, 2014)

    models = intersect(ecs_models, data_ref.model, data_proj.model, data_hist.model)
    data_proj = data_proj[model = Where(x -> x in models)]
    data_ref = data_ref[model = Where(x -> x in models)]
    data_hist = data_hist[model = Where(x -> x in models)]
    # compute anomalies
    proj_anom = data_proj .- dropdims(mean(data_ref; dims=:time), dims=:time)
    hist_anom = data_hist .- dropdims(mean(data_ref; dims=:time), dims=:time)
    ecs_data = ecs_data[model = Where(x -> x in models)]

    n_models = length(models)

    # Observational tas-data
    path_obs_data = joinpath(base_dir, "obs/recipe_ERA5_20250718_180812/preproc/historical")
    data_dirs = ["tas_CLIM-ann"]
    obs_data = mw.defineDataMap(
        joinpath.(path_obs_data, data_dirs),
        data_dirs;
        dtype = "observations",
        filename_format = :esmvaltool,
        constraint = Dict(
            "variables" => ["tas"]
        )
    )
    obs_tas = mwt.filterTimeseries(obs_data["tas_CLIM-ann"], 1950, 2014)[model = 1]
    # Get observational anomalies
    times = Dates.year.(obs_tas.time)
    indices_ref = findall(x -> x >= 1995 && x <= 2014, times)
    obs_anom = obs_tas .- dropdims(mean(obs_tas[time=indices_ref]; dims=:time); dims=:time)
    obs_anom_gms = mwd.globalMeansNoMissing(obs_anom)
end
# ---------------------------------------------------------------------------------------- #

# Expected ECS (prior + posterior) for different Dirichlet priors
n_iter = 10000; n_chains = 10;
params = [fill(1, n_models), fill(1/n_models, n_models)]
weights_priors = Vector(undef, 2)
weights_posteriors = Vector(undef, 2)
for (i, alphas) in enumerate(params)
    Random.seed!(1712)
    # 1. Prior samples Dirichlet
    begin
        model_ecs_dirichlet = mww.weightedAvgModelECS(
            ecs_data.data, alphas, lh_fn_ecs, false
        )
        samples_prior_dirichlet = Turing.sample(model_ecs_dirichlet, Prior(), MCMCThreads(), n_iter, n_chains)
        ws_prior_dirichlet, _ = mww.drawFromSamples(samples_prior_dirichlet, n_iter, n_chains, n_models)
        weights_priors[i] = ws_prior_dirichlet
    end
    # 2. Posterior samples with Dirichlet Prior
    begin
        samples_ecs_dirichlet = Turing.sample(model_ecs_dirichlet, MH(), MCMCThreads(), n_iter, n_chains)
        ws_posterior_dirichlet, _ = mww.drawFromSamples(samples_ecs_dirichlet, n_iter, n_chains, n_models)
        weights_posteriors[i] = ws_posterior_dirichlet
    end
end
mwd.writeDataToDisk(weights_posteriors, joinpath(target_data_dir, "ecs-weights-posteriors-different-dirichlet-priors.jld2"))
mwd.writeDataToDisk(weights_priors, joinpath(target_data_dir, "ecs-weights-priors-different-dirichlet-priors.jld2"))


chain = 1;
titles = ["Prior: Dirichlet([1,..., 1])", "Prior: Dirichlet([1/N, ..., 1/N])"]
f_expected_ecs = mwp.plotExpectedECS(
    ecs_data.data, weights_priors[2], weights_posteriors[2];
    ws_prior2 = weights_priors[1], 
    ws_posterior2 = weights_posteriors[1],
    target_distr_x = xs, 
    target_distr_y = pdf_ecs.(xs),
    chain = chain, 
    titles = reverse(titles),
    frame_visible = false,
    fig_size = (560, 280), 
    add_legend = true,
    xlims = (1, 6),
    ylims = (-0.2, 2.5),
    pos_legend = :lt,
    ps_legend = (15, 15),
    marker_size_models = 25,
    color_prior = COLORS_ECS[1],
    color_posterior = COLORS_ECS[2],
    alpha_densities = 0.8
)
mwp.savePlot(f_expected_ecs, joinpath(plot_dir, "fig8.pdf"); overwrite=true)


# Bring weights together
ws_posterior_dirichlet = weights_posteriors[2] # using Dirichlet(1/N) prior
begin
    mean_weights_dirichlet = mean(ws_posterior_dirichlet; dims=1)
    mw_dirichlet_yax = YAXArray(
        (DimensionalData.dims(ecs_data, :model), Dim{:chain}(map(x->"chain$x", 1:n_chains))), 
        mean_weights_dirichlet[1, :, :]
    )
    mw_dirichlet_yax = mwd.setDim(mw_dirichlet_yax, :chain, :weight, nothing)
    f_weights = mwp.plotWeights(mw_dirichlet_yax; one_plot=true, nbanks=2, fig_size = (650,400), ls = 10, fs=10)

    # Add individual performance weighting: compute weights based on each model's ECS value
    iw_weights = mww.likelihoodWeights(
    Array(ecs_data.data), Array(ecs_data.model), lh_fn_ecs, "individual performance weighting"
    )
    # Add equal weights 
    eq_weights = YAXArray(
        (dims(iw_weights, :model), Dim{:weight}(["equal"])),
        reshape(fill(1/n_models, n_models), :, 1)
    )
    all_weights = mwd.mergeYAX(
        [mw_dirichlet_yax, iw_weights, eq_weights], 
        :weight,
        [Array(mw_dirichlet_yax.weight)..., Array(iw_weights.weight)..., Array(eq_weights.weight)...]
    )
end
mwd.writeDataToDisk(all_weights, joinpath(target_data_dir, "all-weights-ecs.jld2"))
# mwp.plotWeights(all_weights; one_plot=true, nbanks=3)

#----------------------------- Make projection plots -----------------------------# 
df = mwd.DataMap();
df["ssp585"] = proj_anom
df["historical"] = hist_anom
mwd.apply!(df, mwd.globalMeansNoMissing)

n_models = size(df["ssp585"], :model)
# add multi-model mean 
ws = deepcopy(all_weights[weight=[11, 1, 2]])
names = Array(ws.weight)
names[2] = "Ensemble performance weighting"
names[3] = "Multi-Model-Mean"
ws = mwd.setDim(ws, :weight, nothing, names)
ws[weight = At("Multi-Model-Mean")] = fill(1/n_models, n_models)

years_hist = Array(Dates.year.(df["historical"].time))
years_proj = Array(Dates.year.(df["ssp585"].time))
years_all = vcat(years_hist, years_proj)

iw_weights = all_weights[weight = Where(x -> x == "individual performance weighting")]

chain = 1;
alpha = 0.5

idx = 1 # Prior Dirichlet(1) (for appendix)
idx = 2 # Prior Dirichlet(1/N)
ws_posterior_dirichlet = weights_posteriors[idx] 
ws_prior_dirichlet = weights_priors[idx]



add_prior = true;
begin
    plots_legend = []
    labels_legend = []
    f = Figure(size=(560, 280))
        ax = Axis(
            f[1,1], 
            ylabel = "Increase in global mean tas in °C\n reference period: 1995-2014", 
            xticks = (years_all[1:10:end], string.(years_all[1:10:end]))
        )
    Makie.xlims!(ax, years_all[1], years_all[end])
    years = Dict(
        "ssp585" => years_proj,
        "historical" => years_hist
    )
    for (i, experiment) in enumerate(collect(keys(df)))
        data = df[experiment]
        n_timesteps = size(data, :time)
        # Multi Model Mean with quantiles
        begin
            quantiles_data = zeros(5, n_timesteps)
            for t in 1:n_timesteps
                quantiles = map(p -> mwd.quantile(vec(data[time=t]), p), [0.05, 0.25, 0.5, 0.75, 0.95])
                quantiles_data[:, t] .= quantiles
            end
            mmm_band = Makie.band!(ax, years[experiment],  quantiles_data[1,:], quantiles_data[end,:], color=COLORS_PROJ[2], alpha=alpha)
            mmm = Makie.lines!(ax, years[experiment], vec(mean(data; dims=:model)), color=COLORS_PROJ[2], label = "Unweighted (multi-model mean)", linewidth=3)            
            if i==1
                push!(plots_legend, [mmm_band, mmm])
                push!(labels_legend, "Unweighted (multi-model mean)")
            end
        end
        # Ensemble weighting posterior
        begin
            weighted_data = zeros(n_iter, n_timesteps)
            for i in 1:n_iter
                wavg = mww.weightedAvg(data, ws_posterior_dirichlet[i,:,chain])
                weighted_data[i, :] .= wavg
            end

            quantiles_data = zeros(5,n_timesteps)
            for t in 1:n_timesteps
                quantiles = map(p -> mwd.quantile(weighted_data[:,t], p), [0.05, 0.25, 0.5, 0.75, 0.95])
                quantiles_data[:, t] .= quantiles
            end

            ew_band = Makie.band!(ax, years[experiment],  quantiles_data[1,:], quantiles_data[end,:], color=COLORS_PROJ[1], alpha=alpha)
            ew_mean = Makie.lines!(ax, years[experiment], vec(mean(weighted_data; dims=1)), color=COLORS_PROJ[1], label = "Ensemble performance weighting (posterior)", linewidth=3)
            
            if i==1
                push!(plots_legend, [ew_band, ew_mean])
                push!(labels_legend, "Ensemble performance weighting (posterior)")
            end
        end
        # Prior
        if add_prior
            weighted_data = zeros(n_iter, n_timesteps)
            for i in 1:n_iter
                wavg = mww.weightedAvg(data, ws_prior_dirichlet[i,:,chain])
                weighted_data[i, :] .= wavg
            end
            quantiles_data = zeros(5,n_timesteps)
            for t in 1:n_timesteps
                quantiles = map(p -> mwd.quantile(weighted_data[:,t], p), [0.05, 0.25, 0.5, 0.75, 0.95])
                quantiles_data[:, t] .= quantiles
            end
            prior_mean = Makie.lines!(
                ax, years[experiment], vec(mean(weighted_data; dims=1)), 
                color = COLORS_PROJ[3], 
                label = "Ensemble performance weighting (prior)", 
                linewidth = 3,
                linestyle = :dash
            )
            Makie.lines!(
                ax, years[experiment],  quantiles_data[1,:],  
                color=COLORS_PROJ[3], 
                linestyle = :dash
            )
            Makie.lines!(ax, years[experiment], quantiles_data[end,:], 
                color = COLORS_PROJ[3],
                linestyle = :dash
            )
            if i==1
                push!(plots_legend, prior_mean)
                push!(labels_legend, "Ensemble performance weighting (prior)")
            end
        end
    end
    # add observational data
    Makie.lines!(ax, years_hist, obs_anom_gms.data, color=:black, label = "Observational data")
    axislegend(ax, position = :lt, merge=true)
    f
end
mwp.savePlot(f, joinpath(plot_dir, "fig9.pdf"); overwrite=true)
#mwp.savePlot(f, joinpath(plot_dir, "figA1.pdf"); overwrite=true) # for idx=1 (Prior: Dir(1))

