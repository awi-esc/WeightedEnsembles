include("./config.jl")
import ModelWeights.Data as mwd
import ModelWeights.Weights as mww
import ModelWeights.Plots as mwp

using DimensionalData
using Distributions
using MCMCChains
using Random
using Statistics
using YAXArrays
# --------------------------  Generate toy data -------------------------- #
diagnostics = ["tas_ANOM", "psl_ANOM"];
obs_tas = readcubedata(open_dataset(joinpath(data_dir, "diagnostics", "obs_tas_ANOM-GM_1980-2014.nc"))["tas_ANOM-GM"])[model=1]
obs_psl = readcubedata(open_dataset(joinpath(data_dir, "diagnostics", "obs_psl_ANOM-GM_1980-2014.nc"))["psl_ANOM-GM"])[model=1]
obs = mwd.mergeYAX([obs_tas, obs_psl], :diagnostic, diagnostics)

so = size(obs)
dims_obs = dims(obs)
std_obs_d1 = std(obs[diagnostic = 1])
std_obs_d2 = std(obs[diagnostic = 2])

s_d1 = 2; s_d2 = 90; # sigma for errors
toy_model = YAXArray(
    (dims_obs[1], dims_obs[2], Dim{:model}(["m1", "m2"]), dims_obs[3]),
    zeros(so[1], so[2], 2, so[3])
)

Random.seed!(1712);
# First model very good for diagnostic 1, second model very good for diagnostic 2
epsilon_d1 = rand(Normal(0, s_d1), so[1], so[2]);
epsilon_d2 = rand(Normal(0, s_d2), so[1], so[2]);
toy_model[diagnostic=1, model=1] .= obs[diagnostic=1] .+ epsilon_d1
toy_model[diagnostic=2, model=2] .= obs[diagnostic=2] .+ epsilon_d2

# model 1, diagnostic 2: pretty good
toy_model[model=1, diagnostic=2] .= obs[diagnostic=2] .+ rand(Normal(0, 1.5 * s_d2), so[1], so[2])
# model 2, diagnostic 1: okay
toy_model[model=2, diagnostic=1] .= obs[diagnostic=1] .+ rand(Normal(0, 3 * s_d1), so[1], so[2])

# standardize by observational std to bring diagnostics on same scale
toy_model[diagnostic = 1] = toy_model[diagnostic = 1] ./ std_obs_d1
toy_model[diagnostic = 2] = toy_model[diagnostic = 2] ./ std_obs_d2
obs[diagnostic = 1] = obs[diagnostic = 1] ./ std_obs_d1
obs[diagnostic = 2] = obs[diagnostic = 2] ./ std_obs_d2

# check RMSEs
rmse1 = mwd.distancesData(toy_model[diagnostic=1], obs[diagnostic=1]; metric=:rmse).data
rmse2 = mwd.distancesData(toy_model[diagnostic=2], obs[diagnostic=2]; metric=:rmse).data

# Make scatterplots
alpha = 0.8
begin
    f1 = Figure(size = (600, 300))
    ax1 = Axis(
        f1[1,1], 
        title = "Diagnostic variable 1 \n (normalized tas anomalies)",
        xlabel = "observed", 
        ylabel = "predicted",
        yticks = -4:1:1,
        titlefont = :regular
    )
    Label(f1[1, 1, TopLeft()], "a"; fontsize = 12, font=:bold, padding = (10,0,0,0))
    obs_d1 = vec(Array(obs[diagnostic = 1]))
    m2_d1 = vec(Array(toy_model[model = 2, diagnostic = 1]))
    m1_d1 = vec(Array(toy_model[model = 1, diagnostic = 1]))

    min_d1 = minimum(vcat(m1_d1, m2_d1, obs_d1))
    max_d1 = maximum(vcat(m1_d1, m2_d1, obs_d1))

    Makie.scatter!(
        ax1, obs_d1, m2_d1, 
        color = (COLORS_MODELS[2], alpha), label = "Model2", markersize = 5
    )
    Makie.scatter!(
        ax1, obs_d1, m1_d1, 
        color = (COLORS_MODELS[1], alpha), label = "Model1", markersize = 5
    )
    lines!(ax1, [min_d1, max_d1], [min_d1, max_d1], color = :grey)
    ax2 = Axis(
        f1[1,2], 
        title = "Diagnostic variable 2 \n (normalized psl anomalies)",
        xlabel = "observed",
        titlefont = :regular,
        yticks = -4:1:1
    )
    Label(f1[1, 2, TopLeft()], "b"; fontsize = 12, font=:bold, padding = (0,0,0,0))

    obs_d2 = vec(Array(obs[diagnostic = 2]))
    m1_d2 = vec(Array(toy_model[model = 1, diagnostic = 2]))
    plot_model1 = Makie.scatter!(
        ax2, obs_d2, m1_d2, 
        color = (COLORS_MODELS[1], alpha), label = "Model1", markersize = 5
    )
    m2_d2 = vec(Array(toy_model[model = 2, diagnostic = 2]))
    plot_model2 = Makie.scatter!(
        ax2, obs_d2, m2_d2, 
        color = (COLORS_MODELS[2], alpha), label = "Model2", markersize = 5
    )

    min_d2 = minimum(vcat(m1_d2, m2_d2, obs_d2))
    max_d2 = maximum(vcat(m1_d2, m2_d2, obs_d2))
    lines!(ax2, [min_d2, max_d2], [min_d2, max_d2], color = :grey)

    Legend(f1[2,:], [plot_model1, plot_model2], ["Model 1", "Model 2"], 
        framevisible = false, 
        patchlabelgap = 0.1,
        orientation = :horizontal
    )
    rowsize!(f1.layout, 2, Relative(0.05))
    f1
end
mwp.savePlot(f1, joinpath(plot_dir, "fig2.pdf"))

latitudes = collect(lookup(obs.lat))
aw_mat = mwd.areaWeightMatrix(latitudes, Bool.(fill(false, so[1], so[2])))
n_models = 2;

# 1. Compute weights based on joint diagnostics
n_iter = 10000; n_chains = 10;
model_joint = mww.weightedAvgModel(
    collect(toy_model), 
    collect(obs),
    [1, 1], # weights for diagnostics
    fill(1/n_models, n_models), # prior weights
    aw_mat
)
Random.seed!(710)
samples_joint_prior, prior_mat, prior_list = mww.drawFromModel(
    model_joint, n_iter, n_chains, n_models; from_prior = true
)
Random.seed!(710)
samples_joint, posterior_mat, posterior_list = mww.drawFromModel(
    model_joint, n_iter, n_chains, n_models;
)

MCMCChains.ess_rhat(samples_joint)

chain = 1;
mean_w_joint = mean(posterior_mat, dims=1)
round.(mean_w_joint[1,:,chain], digits=2)
mwp.boxplotMCMCWeights(
    posterior_list, Array(toy_model.model); chain=chain, chains_prior=prior_list
)

# 2. Compute weights for each diagnostic separately
# Diagnostic 1
model_d1 = mww.weightedAvgModel(
    collect(toy_model[diagnostic=1]), 
    collect(obs[diagnostic=1]),
    [1], # weights diagnostics
    fill(1/n_models, n_models), # parameters Dirichlet prior
    aw_mat
)
Random.seed!(710)
samples_d1, posterior_mat, posterior_list = mww.drawFromModel(
    model_d1, n_iter, n_chains, n_models;
)
mean1 = mean(posterior_mat, dims=1)
round.(mean1[1,:,chain], digits=2)

# Diagnostic 2
model_d2 = mww.weightedAvgModel(
    collect(toy_model[diagnostic=2]), 
    collect(obs[diagnostic=2]),
    [1], # weights diagnostics
    fill(1/n_models, n_models), # parameters Dirichlet prior
    aw_mat
)
Random.seed!(710)
samples_d2, posterior_mat, posterior_list = mww.drawFromModel(
    model_d2, n_iter, n_chains, 2;
)
mean2 = mean(posterior_mat, dims=1)
round.(mean2[1,:,chain], digits=2)

# 3. Compute weighted average models + corresponding RMSEs
w_sep = YAXArray(
    (Dim{:diagnostic}(diagnostics), Dim{:model}(["m1", "m2"])),
    vcat(mean1[:,:, chain], mean2[:,:, chain])
)
mean_w_sep = mean(w_sep, dims=:diagnostic)[diagnostic=1] 
round.(mean_w_sep.data, digits=2)
    
# Compute RMSE
chain = 1;
# averaged weights
mstar_sep = mww.weightedAvg(toy_model, mean_w_sep, 3)
#mse_mstar_sep = mwd.distancesData(mwd.insertSingletonDim(mstar_sep, 3, :model, "m*_sep"), obs)
mse_sep_d1 = mwd.distancesData(
    mwd.insertSingletonDim(mstar_sep[diagnostic=1], 3, :model, "m*_sep"),
    obs[diagnostic=1]
)
mse_sep_d2 = mwd.distancesData(
    mwd.insertSingletonDim(mstar_sep[diagnostic=2], 3, :model, "m*_sep"),
    obs[diagnostic=2]
)
mse_sep = cat(mse_sep_d1, mse_sep_d2; dims=Dim{:diagnostic}(diagnostics))
round.(mse_sep.data, digits=2)

mstar_joint = mww.weightedAvg(toy_model, YAXArray((dims(toy_model, :model),), mean_w_joint[1,:,chain]), 3)
mse_joint_d1 = mwd.distancesData(
    mwd.insertSingletonDim(mstar_joint[diagnostic=1], 3, :model, "m*_joint"),
    obs[diagnostic=1]
)
mse_joint_d2 = mwd.distancesData(
    mwd.insertSingletonDim(mstar_joint[diagnostic=2], 3, :model, "m*_joint"),
    obs[diagnostic=2]
)
mse_joint = cat(mse_joint_d1, mse_joint_d2; dims=Dim{:diagnostic}(diagnostics))
round.(mse_joint.data, digits=2)

# based on single diagnostic
mstar_d1 = mww.weightedAvg(toy_model, mean1[1, :, chain], 3)
mstar_d2 = mww.weightedAvg(toy_model, mean2[1, :, chain], 3)
mses_single = []
for mstar in [mstar_d1, mstar_d2]
    mse_d1 = mwd.distancesData(
        mwd.insertSingletonDim(mstar[diagnostic=1], 3, :model, "m*_d1"),
        obs[diagnostic=1]
    )
    mse_d2 = mwd.distancesData(
        mwd.insertSingletonDim(mstar[diagnostic=2], 3, :model, "m*_d2"),
        obs[diagnostic=2]
    )
    mses = cat(mse_d1, mse_d2; dims=:model)
    push!(mses_single, mses)
end
mses_single

# For comparison Multi-Model-Mean (MMM)
mmm = mww.weightedAvg(toy_model, YAXArray((dims(toy_model, :model),), [0.5, 0.5]), 3)
rmse_mmm_d1 = mwd.distancesData(
    mwd.insertSingletonDim(mmm[diagnostic=1], 3, :model, "mmm"),
    obs[diagnostic=1]
)
rmse_mmm_d2 = mwd.distancesData(
    mwd.insertSingletonDim(mmm[diagnostic=2], 3, :model, "mmm"),
    obs[diagnostic=2]
)
rmse_mmm = cat(rmse_mmm_d1, rmse_mmm_d2; dims=Dim{:diagnostic}(diagnostics))
round.(rmse_mmm.data, digits=2)


# for comparison: just use single model
rmse_m1_d1 = round.(mwd.distancesData(toy_model[model=1:1, diagnostic=1], obs[diagnostic=1]).data, digits=2)
rmse_m1_d2 = round.(mwd.distancesData(toy_model[model=1:1, diagnostic=2], obs[diagnostic=2]).data, digits=2)

rmse_m2_d1 = round.(mwd.distancesData(toy_model[model=2:2, diagnostic=1], obs[diagnostic=1]).data, digits=2)
rmse_m2_d2 = round.(mwd.distancesData(toy_model[model=2:2, diagnostic=2], obs[diagnostic=2]).data, digits=2)

