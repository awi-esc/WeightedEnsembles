import ModelWeights.Data as mwd
import ModelWeights.Plots as mwp
import ModelWeights.Weights as mww

using DimensionalData
using Random

include("config.jl")
include("fns_independence.jl")

# ---------------------------- Load data ------------------------------------------------- #
diagnostics = ["tas_ANOM", "psl_ANOM"]

model_data = open_dataset(joinpath(data_dir, "diagnostics", "models_tas_ANOM-GM_1980-2014.nc"))["tas_ANOM-GM"]
models = collect(lookup(model_data, :model))
N_models = length(models)

obs_data = open_dataset(joinpath(data_dir, "diagnostics", "obs_tas_ANOM-GM_1980-2014.nc"))["tas_ANOM-GM"][model=1]

latitudes = Array(obs_data.lat)
s = size(obs_data)[1:2]

# -----------------------  Create toy data ------------------------------------------------#
# Ensemble of two real models
real_models = ["AWI-CM-1-1-MR", "CESM2-WACCM"]
# get respective data for single diagnostic 
data_real = copy(model_data[model=Where(x -> x in real_models)])
obs = copy(obs_data)

# Constructed data based on observations
Random.seed!(1712)
data_pattern, data_rnd = makeToyData(data_real, obs; fixed_sigma=true, sigma=2)

# sanity check mean squared errors:
mses_pattern = round.(mwd.distancesData(data_pattern, obs; metric=:mse), digits=2).data
mses_rnd = round.(mwd.distancesData(data_rnd, obs; metric=:mse), digits=2).data
mses_real = round.(mwd.distancesData(data_real, obs).data, digits=2)

# Plot without largest and smallest latitudes (to see differrences better):
bounds = (-65, 65);
obs_mid_lats = mwd.limitLat(obs, bounds);
xticks = -180:60:180; yticks = -60:20:60;
color_range = (-20, 20);

fig_size = (700,160);
legend_title = "Temperature anomaly (°C)";
f1 = plotToyData(
    mwd.limitLat(data_pattern, bounds), obs_mid_lats, color_range;
    legend_label = legend_title,
    xticks = xticks, 
    yticks = yticks,
    fig_size = fig_size
)
f2 = plotToyData(
    mwd.limitLat(data_rnd, bounds), obs_mid_lats, color_range;
    legend_label = legend_title,
    xticks = xticks,
    yticks = yticks,
    fig_size = fig_size
)
f3 = plotToyData(
    mwd.limitLat(data_real, bounds), obs_mid_lats, color_range;
    names = real_models,
    legend_label = legend_title,
    xticks = xticks,
    yticks = yticks,
    fig_size = fig_size
)
mwp.savePlot(f1, joinpath(plot_dir, "fig3.pdf"))
mwp.savePlot(f2, joinpath(plot_dir, "fig4.pdf"))
mwp.savePlot(f3, joinpath(plot_dir, "fig5.pdf"))

# ----------------------------- Sample the data -------------------------------------------#
n_reps_model2 = [1, 2, 4, 6, 8, 10];
model_epw = epwGaussian

aw_mat = mwd.areaWeightMatrix(latitudes, Bool.(fill(false, s[1], s[2])))
# To use prior 1/x*n with different x, just change the x= part in the argument!)
vargs = ["prior 1/(x*n), x=1", aw_mat]
# To use Dirichlet(1), skip the prior argument:
# vargs = [aw_mat]

likelihood_fn = likelihoodSSE
begin
    observations = vec(Array(obs))
    # error pattern
    mean_weights_pattern, posterior_samples = weightsRepModelMCMC(
        data_pattern, observations, n_reps_model2, model_epw, vargs...
    )
    w_epw_pattern = getWeightsForPlot(mean_weights_pattern, 2, n_reps_model2, ["m1", "m2"])
    # random errors
    mean_weights_rnd, _ = weightsRepModelMCMC(
        data_rnd, observations, n_reps_model2, model_epw, vargs...
    )
    w_epw_rnd = getWeightsForPlot(mean_weights_rnd, 2, n_reps_model2, ["m1", "m2"])
    # real model data
    mean_weights_real, _ = weightsRepModelMCMC(
        data_real, observations, n_reps_model2, model_epw, vargs...
    )
    w_epw_real = getWeightsForPlot(mean_weights_real, 2, n_reps_model2, real_models)

    weights_epw = [w_epw_pattern, w_epw_rnd, w_epw_real]
end
begin
    # Add comparison to weights based relative to sum of squared errors   
    w_pattern = getWeightsForPlot(
        weightsRepModel(data_pattern, obs, n_reps_model2, likelihood_fn, latitudes), 2, n_reps_model2, ["m1", "m2"]
    )
    w_rnd = getWeightsForPlot(
        weightsRepModel(data_rnd, obs, n_reps_model2, likelihood_fn, latitudes), 2, n_reps_model2, ["m1", "m2"]
    )
    w_real = getWeightsForPlot(
        weightsRepModel(data_real, obs, n_reps_model2, likelihood_fn, latitudes), 2, n_reps_model2, real_models
    )
    weights_ipw = [w_pattern, w_rnd, w_real]
end

mwd.writeDataToDisk(weights_ipw, joinpath(target_data_dir, "independence-weights-individual.jld2"))
mwd.writeDataToDisk(weights_epw, joinpath(target_data_dir, "independence-weights-ensemble.jld2"))

# ------------------------------ Figure 6 ------------------------------------------- #
n_reps = length(n_reps_model2);
y_label = "Model weights"
begin
    f = Figure(size=(600, 360));
    names = [
        "Error pattern  $(Array(mses_pattern))",
        "Random errors  $(Array(mses_rnd))", 
        "Real data      $(mses_real)"
    ]
    alpha = 0.8
    letters1 = ["a","b","c"]
    letters2 = ["d","e","f"]
    # Individual performance weighting
    for (i, (df, tit)) in enumerate(zip(weights_ipw, names))
        ylab = i == 1 ? y_label : ""
        ax = Axis(
            f[1, i],
            ylabel = ylab,
            xlabel = L"$n_2$",
            title = tit,
            xticks=(1:n_reps, string.(n_reps_model2)), 
            yticks=(0:0.25:1, string.(0:0.25:1)),
            titlefont = :regular
        )
        Makie.ylims!(ax, (0, 1))
        m1 = Makie.scatterlines!(ax, 1:n_reps, Array(df[model=1]), marker = '*', markersize = 30, color=(COLORS_MODELS[1], alpha))
        m3 = Makie.scatterlines!(ax, 1:n_reps, Array(df[model=2]), color=(COLORS_MODELS[2], alpha)) # summed copies
        m2 = Makie.scatterlines!(ax, 1:n_reps, Array(df[:, end]), color=(COLORS_MODELS[2], alpha), linestyle=:dash) # single copy
        Label(f[1, i, TopLeft()], letters1[i]; fontsize = 12, font = :bold, padding = (0,0,10,0))
        if i == length(weights_ipw)
            Legend(
                f[3,:], 
                [m1, m3, m2], 
                ["Model 1", "Sum of all copies of Model 2", "Single copy of model 2"],
                orientation = :horizontal,
                framevisible = false
            )
        end
    end
    # Ensemble performance weighting
    for (i, (df, tit)) in enumerate(zip(weights_epw, names))
        ylab = i == 1 ? y_label : ""
        ax = Axis(
            f[2,i], 
            ylabel = ylab,
            xlabel = L"$n_2$",
            title = tit,
            xticks=(1:n_reps, string.(n_reps_model2)), 
            yticks=(0:0.25:1, string.(0:0.25:1)),
            titlefont = :regular
        )
        Makie.ylims!(ax, (0, 1))
        m1 = Makie.scatterlines!(ax, 1:n_reps, Array(df[:,1]), marker = '*', markersize=30, color=COLORS_MODELS[1])
        m2 = Makie.scatterlines!(ax, 1:n_reps, Array(df[:,end]), color=COLORS_MODELS[2], linestyle=:dash)
        m3 = Makie.scatterlines!(ax, 1:n_reps, Array(df[:,2]), color=COLORS_MODELS[2])
        Label(f[2, i, TopLeft()], letters2[i]; fontsize = 12, font = :bold, padding = (0,0,10,0))
    end
    rowgap!(f.layout, 5)
    rowsize!(f.layout, 3, Relative(0.05))
    f
end
mwp.savePlot(f, joinpath(plot_dir, "fig6.pdf"))

# -------------------- Figure 7: 2 copies of model2 -------------------------------------  #
data_rep = mwd.repeatModel(data_real, 2; index=2)
data_rep_models = Array(data_rep.model)
n_models = size(data_rep, 3)

prior = fill(1/n_models, n_models)
aw_mat = mwd.areaWeightMatrix(latitudes, Bool.(fill(false, s[1], s[2])))

model = epwGaussian(collect(data_rep), vec(collect(obs)), prior, aw_mat)

n_iter=15000; n_chains = 5;
Random.seed!(2310)
samples, posterior_mat, posterior_list = mww.drawFromModel(
    model, n_iter, n_chains, n_models;
)
# boxplot of weights
f = mwp.boxplotMCMCWeights(
    posterior_list, data_rep_models, chain=1; xticks=0:0.2:1, xlims=(0,1), fig_size = (300, 225)
);
Label(f[1, 1, TopLeft()], "b"; fontsize = 12, font = :bold, padding = (0,80,0,0))
f
mwp.savePlot(f, joinpath(plot_dir, "fig7b-boxplot.pdf"))

# plot correlation between weights
f = mwp.plotCorrWeights(
    data_rep_models, posterior_mat[:,:,1], [(1,2), (2,3)]; 
    fig_size = (300, 225),
    xlims = (0, 0.8),
    ylims = (0, 0.6),
    xticks = 0:0.2:1
);
Label(f[1, 1, TopLeft()], "a"; fontsize = 12, font = :bold, padding = (0,30,0,0))
f
mwp.savePlot(f, joinpath(plot_dir, "fig7a-scatterplot.pdf"))
