using CairoMakie
using LinearAlgebra
using Turing
using YAXArrays

"""
# Arguments:
- `data`: lon x lat x 2 with two models
- `obs`: lon x lat
- `fixed_sigma::Bool`: default false
- `sigma::T`: default: 2; used if `fixed_sigma` is true
"""
function makeToyData(data, obs; fixed_sigma::Bool=false, sigma::T=2) where T<:Number
    data_pattern = similar(data)
    data_rnd = similar(data)
    s1, s2, _ = size(data)
    if !fixed_sigma
        sigma_y = std(obs)
        sigma = sigma_y / 5
    end
    epsilon = rand(Normal(0, sigma), (Int(s1/2), s2))
    # 1. Error pattern: one half perfect, other same error added
    data_pattern = mwd.setDim(data_pattern, :model, nothing, ["m1", "m2"])
    data_pattern[lon=1:s2, lat=:, model=1] .= obs[lon=1:s2, lat=:];
    data_pattern[lon=s2+1:end, lat=:, model=1] .= (obs[lon=s2+1:end, lat=:] .+ epsilon);

    data_pattern[lon=1:s2, lat=:, model=2] .= (obs[lon=1:s2, lat=:] .+ epsilon);
    data_pattern[lon=s2+1:end, lat=:, model=2] .= obs[lon=s2+1:end, lat=:];

    # 2. Random errors: each model everywhere obs + random noise (different, but same distr.)
    data_rnd = mwd.setDim(data_rnd, :model, nothing, ["m1", "m2"])
    data_rnd[model=1] .= obs .+ rand(Normal(0, sigma), (s1, s2))
    data_rnd[model=2] .= obs .+ rand(Normal(0, sigma), (s1, s2))

    return (data_pattern, data_rnd)
end


function plotToyData(
    data_rep, obs, color_range::Tuple; 
    names::AbstractArray = [], 
    xticks::Union{Nothing, AbstractArray} = nothing,
    yticks::Union{Nothing, AbstractArray} = nothing,
    legend_label::String = "",
    title::String = "",
    font_size::Number = 10,
    fig_size::Tuple = (630, 200)
)
    name_m1 = isempty(names) ? "Model 1" : names[1]
    name_m2 = isempty(names) ? "Model 2" : names[2]
    f = Figure(size = fig_size)
    mwp.plotValsOnMap!(
        f, obs, "Observations"; 
        pos = (x=1, y=1), 
        color_range = color_range,
        fontsize = font_size,
        xlabel = "",
        ylabel = "",
        xticks = xticks,
        yticks = yticks
    )
    Label(f[1, 1, TopLeft()], "a"; fontsize = 12, font = :bold, padding = (20,0,10,0))
    mwp.plotValsOnMap!(
        f, data_rep[:,:,1], name_m1; 
        pos = (x=1, y=2), 
        color_range = color_range,
        fontsize = font_size,
        xlabel = "",
        ylabel = "",
        xticks = xticks,
        yticks = yticks
    )
    Label(f[1, 2, TopLeft()], "b"; fontsize = 12, font = :bold, padding = (10,0,10,0))
    mwp.plotValsOnMap!(
        f, data_rep[:,:,2], name_m2; 
        pos = (x=1, y=3), 
        pos_legend = (x=1, y=4),
        orient_legend = :vertical,
        legend_label = legend_label,
        color_range = color_range, 
        fontsize = font_size,
        xlabel = "",
        ylabel = "",
        xticks = xticks,
        yticks = yticks
    )
    Label(f[1, 3, TopLeft()], "c"; fontsize = 12, font = :bold, padding = (20,0,10,0))
    if !isempty(title)
        Label(f[0,:], title, fontsize=font_size)
    end
    return f
end

function likelihoodInvMSE(data, obs, latitudes)
    return 1 ./ mwd.distancesData(data, obs, latitudes; metric=:mse)
end

function likelihoodSSE(data, obs, latitudes)
    s1, s2, _ = size(data)
    return s1 * s2 * likelihoodInvMSE(data, obs, latitudes)
end

# function likelihoodGaussian(data, obs, latitudes)
#     distances = mwd.distancesData(data, obs, latitudes; metric = :rmse)
#     normed_dists = distances ./ median(distances)
#     return exp.(-1 .* (normed_dists ./ 0.2).^2)
# end

"""
# Arguments:
- `data`: last dimension are models
- `obs`: observational data
"""
@model function epwGaussian(
    data::AbstractArray, obs::AbstractArray, prior_params::AbstractArray, area_weights::AbstractArray
)
    w ~ Dirichlet(prior_params)
    weighted_avg = mww.weightedAvg(data, w)
    sigma_sq ~ InverseGamma(2, 3)
    covariance = Diagonal(vec((1 ./ area_weights) .* sigma_sq))
    obs ~ Distributions.MvNormal(vec(weighted_avg), covariance)
end



"""
# Arguments:
- `data`: last dimension are models
- `obs`: observational data
"""
@model function epwGaussian(
    data::AbstractArray, 
    obs::AbstractArray, 
    prior_params::AbstractArray,
    hyperprior_sigma::AbstractArray,
    area_weights::AbstractArray
)
    w ~ Dirichlet(prior_params)
    a ~ Uniform(hyperprior_sigma...)
    b ~ Uniform(hyperprior_sigma...)
    weighted_avg = mww.weightedAvg(data, w)
    sigma_sq ~ InverseGamma(a, b)
    covariance = Diagonal(vec((1 ./ area_weights) .* sigma_sq))
    obs ~ Distributions.MvNormal(vec(weighted_avg), covariance)
end


"""
    weightsRepModelMCMC(data, obs n_reps, likelihood_fn, args...; n_iter=15000)

Repeat second model in `data` for respectively `n_reps` times and compute BMA weights. 
    
Use one MCMC chain. Return mean weights for model 1, first copy of model 2 and for the sum 
of all copies of model2.
"""
function weightsRepModelMCMC(data, obs, n_reps, likelihood_fn, args...; n_iter=15000)
    mean_weights = Vector(undef, length(n_reps))
    posterior_samples = Vector(undef, length(n_reps))
    use_prior_arg = !isnothing(args) && isa(args[1], String) && startswith(args[1], "prior 1/(x*n), x=")
    x = use_prior_arg ? parse(Float32, split(args[1], "=")[2]) : nothing
    arguments = !isnothing(args) ? map(x -> args[x], 1:length(args)) : nothing
    @info "Prior: 1/($x * n)"
    n_chains = 1; chain = 1
    for (i, i_rep) in enumerate(n_reps)
        data_rep = mwd.repeatModel(data, i_rep; index=2)
        n_models = size(data_rep, 3)
        if use_prior_arg
            n_models = i_rep + 1
            arguments[1] = fill(1 / (x * n_models), n_models)
            args = arguments
        end
        model = likelihood_fn(collect(data_rep), collect(obs), args...)
        samples, posterior_mat, posterior_list = mww.drawFromModel(
            model, n_iter, n_chains, n_models
        )
        mean_weights[i] = mean(posterior_mat, dims=1)[1,:,chain]
        posterior_samples[i] = posterior_mat[:,:, chain]
    end
    return (mean_weights, posterior_samples)
end

"""
    weightsRepModel(data, obs, n_reps, likelihood_fn, latitudes)

Repeat second model in `data` for respectively `n_reps` times and compute model weights 
proportional to the models' MSEs. 
"""
function weightsRepModel(data, obs, n_reps, likelihood_fn, latitudes)
    weights = Vector(undef, length(n_reps))
    for (i, i_rep) in enumerate(n_reps)
        data_rep = mwd.repeatModel(data, i_rep; index=2)
        lh = likelihood_fn(Array(data_rep.data), obs, latitudes)
        weights[i] = lh ./ sum(lh)
    end
    return weights
end

function sumWeightsRep(weights::AbstractArray, i_start::Int, n::Int, names::AbstractArray)
    summed_w = sumWeightsRep(weights, i_start, i_start + n - 1)
    new_names = names[1:i_start]
    i_end = i_start + n - 1
    if i_end < length(weights) 
        new_names = [new_names..., names[i_end+1:end]]
    end
    return YAXArray(
        (Dim{:model}(), new_names),
        summed_w
    )
end

function sumWeightsRep(weights::AbstractArray, i_start::Int, n_rep::Int)
    n = length(weights)
    summed_w = zeros(n - n_rep + 1)
    if i_start > 1
        summed_w[1:i_start-1] .= weights[1:i_start-1]
    end
    i_end = i_start + n_rep - 1
    summed_w[i_start] = sum(weights[i_start: i_end])
    if i_end < n
        summed_w[i_start+1:end] .= weights[i_end + 1 : end]
    end
    return summed_w    
end

function getWeightsForPlot(weights, i_start, n_reps, new_names)
    ws = zeros(length(n_reps), length(new_names) + 1)
    for (i, i_rep) in enumerate(n_reps)
        ws[i, :] = [sumWeightsRep(weights[i], i_start, i_rep)..., weights[i][i_start]]
    end
    return YAXArray(
        (Dim{:nb_rep}(n_reps), Dim{:model}([new_names..., new_names[i_start] * ".1"])),
        ws
    )
end
