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
    ecs_data_csv = DataFrame(CSV.File(joinpath("data", "ecs", "ecs-unique.csv")))
    # Just cmip6 models
    ecs_data_csv = filter(row -> row.mip == "CMIP6", ecs_data_csv)
    ecs_data_csv = filter(row -> row.model in models, ecs_data_csv)

    ecs_data = YAXArray(
        (Dim{:model}(String.(ecs_data_csv[!,:model])),),
        ecs_data_csv[!, :ECS]
    )
    # Target ECS-distribution (based on Sherwood et al):
    data_pdf_ecs = open_dataset(joinpath("data", "ecs", "ecs-pdf.nc"))
    ecs_density = data_pdf_ecs["density"]
    xs = lookup(ecs_density, :x)
    ys = Array(ecs_density)
    pdf_ecs = Dierckx.Spline1D(xs, ys, k=ecs_density.properties["k"], s=ecs_density.properties["s"])
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

# ---------------------------- save weights to publish ----------------------------------- #
function saveWeights(data, dimensions, target_path)
    yax = YAXArray(dimensions, data)
    savecube_safe(yax, target_path; driver=:netcdf, layername="weight")
end

dimensions = (Dim{:iteration}(1:n_iter), Dim{:model}(models), Dim{:chain}(1:n_chains))
saveWeights(weights_posteriors[1], dimensions, joinpath(target_data_dir, "weights-ecs", "posterior-weights-dirichlet-1.nc"))
saveWeights(weights_posteriors[2], dimensions, joinpath(target_data_dir, "weights-ecs", "posterior-weights-dirichlet-1-over-N.nc"))

saveWeights(weights_priors[1], dimensions, joinpath(target_data_dir, "weights-ecs", "prior-weights-dirichlet-1.nc"))
saveWeights(weights_priors[2], dimensions, joinpath(target_data_dir, "weights-ecs", "prior-weights-dirichlet-1-over-N.nc"))
# ---------------------------------------------------------------------------------------- #

# Make plot
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

# Individual performance weighting: compute weights based on each model's ECS value (not in paper)
iw_weights = mww.likelihoodWeights(
    Array(ecs_data.data), Array(ecs_data.model), lh_fn_ecs, "individual performance weighting"
)[weight = 1]

#----------------------------- Make projection plots -----------------------------# 
df = mwd.DataMap();
df["ssp585"] = proj_anom
df["historical"] = hist_anom

years_hist = Array(Dates.year.(df["historical"].time))
years_proj = Array(Dates.year.(df["ssp585"].time))
years_all = vcat(years_hist, years_proj)

chain = 1;
alpha = 0.25;
linewidth = 3;

# choose weights
#idx = 1 # Prior Dirichlet(1) (for appendix)
#idx = 2 # Prior Dirichlet(1/N) (main text)

for idx in 1:2
    begin
        ws_posterior_dirichlet = weights_posteriors[idx] 
        ws_prior_dirichlet = weights_priors[idx]
    end

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
                if i==1
                    push!(plots_legend, [mmm_band])#, mmm])
                    push!(labels_legend, "Unweighted (multi-model mean)")
                end
            end
            # Prior
            weighted_data_prior = zeros(n_iter, n_timesteps)
            if add_prior
                for i in 1:n_iter
                    wavg = mww.weightedAvg(data, ws_prior_dirichlet[i,:,chain])
                    weighted_data_prior[i, :] .= wavg
                end
                quantiles_data = zeros(5,n_timesteps)
                for t in 1:n_timesteps
                    quantiles = map(p -> mwd.quantile(weighted_data_prior[:,t], p), [0.05, 0.25, 0.5, 0.75, 0.95])
                    quantiles_data[:, t] .= quantiles
                end
                prior_band = Makie.band!(ax, years[experiment],  quantiles_data[1,:], quantiles_data[end,:], color=COLORS_PROJ[3], alpha=alpha)
                if i==1
                    push!(plots_legend, [prior_band])
                    push!(labels_legend, "Ensemble performance weighting (prior)")
                end
            end
            # Ensemble performance weighting posterior
            weighted_data_posterior = zeros(n_iter, n_timesteps)
            if add_epw
                for i in 1:n_iter
                    wavg = mww.weightedAvg(data, ws_posterior_dirichlet[i,:,chain])
                    weighted_data_posterior[i, :] .= wavg
                end

                quantiles_data = zeros(5,n_timesteps)
                for t in 1:n_timesteps
                    quantiles = map(p -> mwd.quantile(weighted_data_posterior[:,t], p), [0.05, 0.25, 0.5, 0.75, 0.95])
                    quantiles_data[:, t] .= quantiles
                end

                ew_band = Makie.band!(ax, years[experiment],  quantiles_data[1,:], quantiles_data[end,:], color=COLORS_PROJ[1], alpha=alpha)            
                if i==1
                    push!(plots_legend, [ew_band])
                    push!(labels_legend, "Ensemble performance weighting (posterior)")
                end
            end

            mmm = Makie.lines!(
                ax, 
                years[experiment], 
                vec(mean(data; dims=:model)), 
                color=COLORS_PROJ[2], 
                label = "Unweighted (multi-model mean)", 
                linewidth=linewidth
            )            
            if add_prior
                prior_mean = Makie.lines!(
                    ax, years[experiment], vec(mean(weighted_data_prior; dims=1)), 
                    color = COLORS_PROJ[3], 
                    label = "Ensemble performance weighting (prior)", 
                    linewidth = 2,
                    linestyle = :dash
                )
            end
            if add_epw
                ew_mean = Makie.lines!(
                    ax, 
                    years[experiment], 
                    vec(mean(weighted_data_posterior; dims=1)), 
                    color=COLORS_PROJ[1], 
                    label = "Ensemble performance weighting (posterior)", 
                    linewidth = linewidth
                )
            end

            # -------------------------------------------------------------------------------- #
            # Add individual performance weighting and mean posterior weight vector (not in paper)#
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
                wavg = mww.weightedAvg(data, vec(iw_weights))
                quantiles_data = zeros(5, n_timesteps)
                for t in 1:n_timesteps
                    quantiles = map(p -> mwd.quantile(vec(data[time=t]), p; w=iw_weights), [0.05, 0.25, 0.5, 0.75, 0.95])
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
        end
        # add observational data
        Makie.lines!(ax, years_hist, obs_anom.data, color=:black, label = "Observational data")
        axislegend(ax, position = :lt, merge=true)
        f
    end
    if idx == 1
        mwp.savePlot(f, joinpath(plot_dir, "figA1.pdf")) # for idx=1 (Prior: Dir(1))
    elseif idx == 2
        mwp.savePlot(f, joinpath(plot_dir, "fig9.pdf")) # for idx=2 (Prior: Dir(1/N))
    end
end



