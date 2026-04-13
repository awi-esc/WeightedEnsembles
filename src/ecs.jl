import ModelWeights as mw
import ModelWeights.Data as mwd 
import ModelWeights.Plots as mwp
import ModelWeights.Timeseries as mwt
import ModelWeights.Weights as mww

using CairoMakie
using CSV
using DataFrames
using Dates
using DimensionalData
using Dierckx
using Random
using Statistics
using Turing
using YAXArrays

include("./config.jl")

# ----------------------------- Data ----------------------------------------- #
# Get tas data for historical and projections
proj_anom = readcubedata(open_dataset(joinpath(data_dir, "timeseries-projection-plot", "model_tas_gms-anomalies-ref_ssp585.nc"))["tas_annual-GM-ANOM_ssp585"])
hist_anom = readcubedata(open_dataset(joinpath(data_dir, "timeseries-projection-plot", "model_tas_gms-anomalies-ref_historical.nc"))["tas_annual-GM-ANOM_historical"])

models = Array(proj_anom.model)
n_models = length(models)

obs_anom = readcubedata(open_dataset(joinpath(data_dir, "timeseries-projection-plot", "obs_tas_gms-anomalies-ref.nc"))["tas_ANOM-ann-GM"])[model = At("ERA5")]

# Get ECS values
begin
    ecs_data_csv = DataFrame(CSV.File(joinpath(data_dir, "ecs", "ecs-unique.csv")))
    # Just cmip6 models
    ecs_data_csv = filter(row -> row.mip == "CMIP6", ecs_data_csv)
    ecs_data_csv = filter(row -> row.model in models, ecs_data_csv)

    ecs_data = YAXArray(
        (Dim{:model}(String.(ecs_data_csv[!,:model])),),
        ecs_data_csv[!, :ECS]
    )
    # Target ECS-distribution (based on Sherwood et al):
    data_pdf_ecs_wd = mwd.readDataFromDisk(joinpath(data_dir, "ecs", "ecs_data-lh-fn-wd.jld2"))
    xs = data_pdf_ecs_wd["xs"]
    ys = data_pdf_ecs_wd["ys"]
    pdf_ecs = Dierckx.Spline1D(xs, ys, k=data_pdf_ecs_wd["k"], s=data_pdf_ecs_wd["s"])
    lh_fn_ecs(x) = pdf_ecs(x)
end

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
mwp.savePlot(f_expected_ecs, joinpath(plot_dir, "fig8.pdf"))


# Bring mean weights together
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
mwd.writeDataToDisk(all_weights, joinpath(target_data_dir, "all-weights-ecs.jld2"); add_hour = false)
# mwp.plotWeights(all_weights; one_plot=true, nbanks=3)

#----------------------------- Make projection plots -----------------------------# 
df = mwd.DataMap();
df["ssp585"] = proj_anom
df["historical"] = hist_anom

years_hist = Array(Dates.year.(df["historical"].time))
years_proj = Array(Dates.year.(df["ssp585"].time))
years_all = vcat(years_hist, years_proj)

chain = 1;
alpha = 0.5
idx = 1 # Prior Dirichlet(1) (for appendix)
idx = 2 # Prior Dirichlet(1/N) (main text)
ws_posterior_dirichlet = weights_posteriors[idx] 
ws_prior_dirichlet = weights_priors[idx]

# choose which weightings to show
begin
    add_prior = true;
    add_unweighted = true;
    add_ipw = false;
    add_epw = true;
    add_mean_posterior = false;
end
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
        if add_unweighted
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

        # - Add individual performance weighting and mean posterior weight vector (not in paper)-- #
        # weighted averages for each timestep with mean posterior weight vector 
        if add_mean_posterior
            w_mean = mean(ws_posterior_dirichlet[:,:,chain]; dims=1)
            wavg = mww.weightedAvg(data, vec(w_mean))
            quantiles_data = zeros(5, n_timesteps)
            for t in 1:n_timesteps
                quantiles = map(p -> mwd.quantile(vec(data[time=t]), p; w=w_mean), [0.05, 0.25, 0.5, 0.75, 0.95])
                quantiles_data[:, t] .= quantiles
            end
            mmm_band = Makie.band!(ax, years[experiment],  quantiles_data[1,:], quantiles_data[end,:], color=COLORS[5], alpha=alpha)
            mmm = Makie.lines!(ax, years[experiment], vec(wavg), color=COLORS[5], label = "Weighted (mean posterior)", linewidth=3)            
            if i==1
                push!(plots_legend, [mmm_band, mmm])
                push!(labels_legend, "Weighted (mean posterior)")
            end
        end
        # weighted averages for each timestep with individual performance weight vector 
        if add_ipw
            w = all_weights[weight=At("individual performance weighting")]
            wavg = mww.weightedAvg(data, vec(w))
            quantiles_data = zeros(5, n_timesteps)
            for t in 1:n_timesteps
                quantiles = map(p -> mwd.quantile(vec(data[time=t]), p; w=w), [0.05, 0.25, 0.5, 0.75, 0.95])
                quantiles_data[:, t] .= quantiles
            end
            mmm_band = Makie.band!(ax, years[experiment],  quantiles_data[1,:], quantiles_data[end,:], color=COLORS[4], alpha=alpha)
            mmm = Makie.lines!(ax, years[experiment], vec(wavg), color=COLORS[4], label = "Weighted (individual)", linewidth=3)            
            if i==1
                push!(plots_legend, [mmm_band, mmm])
                push!(labels_legend, "Weighted (individual)")
            end
        end
        # -------------------------------------------------------------------------------- #
        # #Ensemble performance weighting posterior
        if add_epw
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
    Makie.lines!(ax, years_hist, obs_anom.data, color=:black, label = "Observational data")
    axislegend(ax, position = :lt, merge=true)
    f
end
mwp.savePlot(f, joinpath(plot_dir, "fig9.pdf"); overwrite=true)
mwp.savePlot(f, joinpath(plot_dir, "figA1.pdf"); overwrite=true) # for idx=1 (Prior: Dir(1))

