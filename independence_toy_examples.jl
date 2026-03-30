import ModelWeights.Data as mwd
import ModelWeights.Plots as mwp
import ModelWeights.Weights as mww

using DimensionalData
using Random

diagnostics = ["tas_ANOM", "psl_ANOM"]
data = mwd.readDataFromDisk("/Users/brgrus001/ModelWeightsPaper/work/output/data/example-brunner_data.jld2")
model_data = data["diagnostic_data"]["models"]
mwd.summarizeMembers!(model_data)
model_data = mwd.convertToYAX(model_data)[diagnostic = Where(x -> x in diagnostics)]
models = collect(lookup(model_data, :model))
N_models = length(models)

obs_data = mwd.convertToYAX(data["diagnostic_data"]["observations"])[model=1, diagnostic = Where(x -> x in diagnostics)]
include("fns_independence.jl")

# ---------------------------- Load data ------------------------------------------------- #
data = mwd.readDataFromDisk("./output/data/data.jld2")
# model_data = data[]
# obs_data = data[]

latitudes = Array(obs_data.lat)
s = size(obs_data)[1:2]

# -----------------------  Create toy data ------------------------------------------------#
# Ensemble of two real models
real_models = ["AWI-CM-1-1-MR", "CESM2-WACCM"]
# get respective data for single diagnostic 
data_real = copy(model_data[diagnostic=At("tas_ANOM"), model=Where(x -> x in real_models)])
obs = copy(obs_data[diagnostic=At("tas_ANOM")])

# Constructed data based on observations
Random.seed!(1712)
data_pattern, data_rnd = makeToyData(data_real, obs; fixed_sigma=true)
# sanity check mean squared errors:
mses_pattern = round.(mwd.distancesData(data_pattern, obs; metric=:mse), digits=2).data
mses_rnd = round.(mwd.distancesData(data_rnd, obs; metric=:mse), digits=2).data
mses_real = round.(mwd.distancesData(data_real, obs).data, digits=2)

# Plot without largest and smallest latitudes (to see differrences better):
bounds = (-65, 65);
obs_mid_lats = mwd.limitLat(obs, bounds);
xticks = -180:60:180; yticks = -60:20:60;
color_range = (-20, 20);
f1 = plotToyData(
    mwd.limitLat(data_pattern, bounds), obs_mid_lats, color_range;
    xticks = xticks,
    yticks = yticks,
    legend_label = "Temperature anomaly (°C)",
    fig_size = (700, 200)
)
f2 = plotToyData(
    mwd.limitLat(data_rnd, bounds), obs_mid_lats, color_range;
    xticks = xticks,
    yticks = yticks,
    legend_label = "Temperature anomaly (°C)",
    fig_size = (700, 200)
)
f3 = plotToyData(
    mwd.limitLat(data_real, bounds), obs_mid_lats, color_range;
    names = real_models,
    xticks = xticks,
    yticks = yticks,
    legend_label = "Temperature anomaly (°C)",
    fig_size = (700, 200)
)
mwp.savePlot(f1, joinpath(plot_dir, "data-error-pattern.pdf"))
mwp.savePlot(f2, joinpath(plot_dir, "data-random-err.pdf"))
mwp.savePlot(f3, joinpath(plot_dir, "data-real-models.pdf"))

# ----------------------------- Sample the data -------------------------------------------#
n_reps_model2 = [1, 2, 4, 6, 8, 10];
model_epw = epwGaussian

aw_mat = mwd.areaWeightMatrix(latitudes, Bool.(fill(false, s[1], s[2])))
# To use prior 1/x*n use,just change the x= part in the argument!)
vargs = ["prior 1/(x*n), x=1", aw_mat]
# OR to use dirichlet(1) for bmaGaussianwA:
# vargs = [aw_mat]

likelihood_fn = likelihoodSSE
begin
    observations = vec(Array(obs))
    # error pattern
    mean_weights_pattern, _ = weightsRepModelMCMC(
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
    
    # Add comparison to weights based relative to mse    
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
        Label(f[1, i, TopLeft()], letters1[i]; fontsize = 20, padding = (0,0,10,0))
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
        Label(f[2, i, TopLeft()], letters2[i]; fontsize = 20, padding = (0,0,10,0))
    end
    rowgap!(f.layout, 5)
    rowsize!(f.layout, 3, Relative(0.05))
    f
end
mwp.savePlot(f, joinpath(plot_dir, "results-independence-toy-examples.pdf"))

# -------------------- Figure 7: 2 copies of model2 -------------------------------------  #
data_rep = mwd.repeatModel(data_real, 2; index=2)
data_rep_models = Array(data_rep.model)
n_models = size(data_rep, 3)

prior = fill(1/n_models, n_models)
aw_mat = mwd.areaWeightMatrix(latitudes, Bool.(fill(false, s[1], s[2])))

model = epwGaussian(collect(data_rep), vec(collect(obs)), prior, aw_mat)

n_iter=15000; n_chains = 5;
samples, posterior_mat, posterior_list = mww.drawFromModel(
    model, n_iter, n_chains, n_models;
)
# boxplot of weights
f = mwp.boxplotMCMCWeights(
    posterior_list, data_rep_models, chain=1; xticks=0:0.2:1, xlims=(0,1), fig_size = (300, 225)
)
Label(f[1, 1, TopLeft()], "a"; fontsize = 12, font = :bold, padding = (0,80,0,0))
f
mwp.savePlot(f, joinpath(plot_dir, "fig7a.pdf"); overwrite=true)

# plot correlation between weights
f = mwp.plotCorrWeights(
    data_rep_models, posterior_mat[:,:,1], [(1,2), (2,3)]; 
    fig_size = (300, 225),
    xlims = (0, 0.8),
    ylims = (0, 0.6),
    xticks = 0:0.2:1
)
Label(f[1, 1, TopLeft()], "b"; fontsize = 12, font = :bold, padding = (0,10,10,0))
f
mwp.savePlot(f, joinpath(plot_dir, "fig7b.pdf"); overwrite=true)


function getMSEs(df, obs, samples_w, latitudes; chain=1)
    n_iter = size(samples_w, 1)
    n_models = size(samples_w, 2)
    mses = Vector(undef, n_iter)
    for i in 1:n_iter
        w = samples_w[i,:, chain]
        weighted_avg =  sum(df .* reshape(w, 1,1,n_models); dims=:model)
        mses[i] = mwd.distancesData(weighted_avg, obs, latitudes; metric=:mse)
    end
    return mses
end

# for every prior sample plot MSE, 
function plotWeightMetric(df, samples_w; chain=1, title="", ylab="")
    w1 = samples_w[:,1,chain]
    indices = sortperm(w1)
    w_sorted = samples_w[indices, :, chain]
    
    f = Figure()
    xlab = "Weight Model 1"
    ax = Axis(f[1,1], xlabel=xlab, ylabel=ylab)
    Makie.scatter!(ax, w_sorted[:,1], df[indices])
    
    ax2 = Axis(f[1,2], xlabel=xlab)
    Makie.density!(ax2, w_sorted[:, 1], color=(:grey, 0.3))

    Label(f[0,:], title)
    return f
end
mses = getMSEs(df, obs, prior_mat, latitudes)
f_prior = plotWeightMetric(mses, prior_mat; title="Prior", ylab="MSE")
# same for posterior
posterior_samples, posterior_mat, posterior_list = mww.drawFromModel(
    model, n_iter, n_chains, n_models
)

mses = getMSEs(df, obs, posterior_mat, latitudes)
f_posterior = plotWeightMetric(mses, posterior_mat; title="Posterior", ylab="MSE")

f_posterior = plotWeightMetric(
    posterior_samples.value[:,:logprior,1], posterior_mat;
    title="Posterior", ylab="Log prior"
)
f_posterior = plotWeightMetric(
    posterior_samples.value[:,:loglikelihood,1], posterior_mat;
    ylab="Log likelihood Posterior samples"
)
############################################################################################


w_bma = mean_weights_real[1,:][model = Where(x -> x in [m1, m2])]
mwd.distancesData(sum(data_real .* reshape(w_bma.data, 1,1,2), dims=3), obs).data
mwd.distancesData(sum(data_real .* reshape([0.5, 0.5], 1,1,2), dims=3), obs).data


# Plot correlation of weights
df = mwd.repeatModel(data_real, 2; index=2)
err = df .- obs
f = Figure()
m1=Makie.density!(Axis(f[1,1], title="$(real_models[1])"), vec(err[:,:,1]),)
m2=Makie.density!(Axis(f[1,2], title="$(real_models[2])"), vec(err[:,:,2]))
f


model = bmaGaussianWA(collect(df), vec(collect(obs)), vec(aw_mat))
n_iter = 15000; n_chains = 3; n_models = size(df, 3);
samples, posterior_mat, posterior_list = mww.drawFromModel(
    model, n_iter, n_chains, n_models
)
chain = 1;
f_posterior = mwp.boxplotMCMCWeights(
    posterior_list[chain], 
    Array(dims(df, :model)); 
    title = ""
)
f_corr = mwp.plotCorrWeights(
        lookup(df, :model), 
        posterior_mat[:, :, chain], 
        [(1,2), (2, 3)]
)
mwp.savePlot(f_posterior, joinpath(plot_dir, "independence-posterior.png"); overwrite=true)
mwp.savePlot(f_corr, joinpath(plot_dir, "independence-corr.png"); overwrite=true)

# Plot posterior over variances
vars = samples.value[:, :sigma_sq, :]
f_var = Figure();
ax = Axis(f_var[1,1], xlabel = L"posterior samples $\sigma$")
for c in 1:size(vars, 2)
    Makie.density!(ax, vars[:,c], label="Chain $c")
end
axislegend(framevisible=false)
f_var

mean_var = mean(vars[:,chain])

likelihood_fn = Normal(0, sqrt(mean_var))
w_bma = mean(posterior_mat, dims=1)[1,:,chain]
wa_bma = mww.weightedAvg(df, w_bma)[model=1]

w = [0.5, 0.25, 0.25]
wa = mww.weightedAvg(df, w)[model = 1]

sum(map(x -> Distributions.logpdf(likelihood_fn, x), vec(wa .- obs)))
sum(map(x -> Distributions.logpdf(likelihood_fn, x), vec(wa_bma .- obs)))


# Toy example for uncertainties
n = 20;
samples = rand(Normal(10, sqrt(3)), n);
quantiles = [mwd.quantile(samples, 0.05), mwd.quantile(samples, 0.95)]
quantiles_w = [
    mwd.quantile(samples, 0.05; w=fill(1/n, n)), 
    mwd.quantile(samples, 0.95; w=fill(1/n, n))
]

f = Figure();
ax = Axis(f[1,1], xlabel="samples")
Makie.density!(ax, samples)
Makie.scatter!(ax, samples, fill(0, n), markersize=15)
Makie.lines!(ax, quantiles, [0.05, 0.05], color=:red)
Makie.lines!(ax, quantiles_w, [0.075, 0.075], color=:yellow)

f


# Dependency of weights with increasing mse
mses_all = mwd.distancesData(model_data[diagnostic=1], obs)

indices = sortperm(mses_all).data
n = length(indices)
models_sorted = Array(mses_all.model[indices])
mses_sorted = YAXArray(
    (Dim{:model}(models_sorted),),
    mses_all.data[indices]
)
w_climwip = mwd.readDataFromDisk("work/output/data/example-brunner/tas_ANOM.jld2").w[weight = At("wP-historical")]
w_inverse_mse = (1 ./ mses_all) ./ sum((1 ./ mses_all))

sortperm(w_climwip).data == sortperm(w_inverse_mse)
reverse(indices) == sortperm(w_climwip).data

f = Figure();
ax = Axis(f[1,1], xlabel="Mean squared error", ylabel="weight")
Makie.scatterlines!(ax, mses_sorted.data, w_climwip[indices].data, label="ClimWIP performance")
Makie.scatterlines!(ax, mses_sorted.data, w_inverse_mse[indices].data, label="Inverse MSE")
Makie.hlines!(ax, 1/n; linestyle=:dash, color=:grey, label="Equal weighting")
axislegend()
f
mwp.savePlot(f, joinpath(plot_dir, "relation-mse-weights.png"); overwrite=true)
