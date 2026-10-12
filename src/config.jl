using CairoMakie
using GeoMakie, Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."));

data_dir = joinpath(ROOT, "data");
target_data_dir = joinpath(ROOT, "output", "data");
plot_dir = joinpath(ROOT, "output", "plots");

Makie.set_theme!(fontsize = 10);
COLORS_MODELS = [
    RGBf(([195, 82, 58] ./ 255)...), # Model 1
    RGBf(([58, 175, 195] ./ 255)...), # Model 2
    RGBf(([80, 40, 110] ./ 255)...) # Model 3
];

# for Dirichlet-Weighted averages plot and for projection plot
COLORS = [
    RGBf(56/255, 180/255, 145/255),  # teal
    RGBf(237/255, 120/255, 46/255),  # orange
    RGBf(100/255, 130/255, 210/255),  # blue
    RGBf(200/255, 50/255, 160/255),  # magenta
    RGBf(180/255, 180/255, 30/255),  # yellow
];

# for ECS plot
COLORS_ECS = [
    RGBf(([186, 69, 129] ./ 255)...),
    RGBf(([69, 186, 126] ./ 255)...)
];

COLORS_PROJ = [
    RGBf(([ 45, 115, 190] ./ 255)...),  # blue    (lum 0.15)
    RGBf(([215, 125,  45] ./ 255)...),  # orange  (lum 0.27)
    RGBf(([155,  65, 195] ./ 255)...),  # purple  (lum 0.13)
];

function savecube_safe(cube, path; kwargs...)
    mkpath(dirname(path))
    savecube(cube, path; kwargs...)
end


# Combine Fig 7a and 7b into a single figure 
# In a later version of ModelWeights than used here, these two functions are adapted to not create their own figures.
function myPlotCorrWeights(
    gp::GridPosition,
    models_labels::AbstractVector,
    weights::AbstractArray, 
    pairs::AbstractVector;
    color::Symbol = :grey,
    color_means::Union{Symbol, ColorTypes.RGB, Nothing} = CairoMakie.RGBf(206/255, 250/255, 220/255),
    xlims::Union{Nothing, Tuple} = nothing,
    ylims::Union{Nothing, Tuple} = nothing, 
    fs::Int = 11,
    marker_size::Int = 30,
    xticks::AbstractArray = 0:0.1:1,
    yticks::AbstractArray = 0:0.1:1
)
    axes = []
    for (i, (m1, m2)) in enumerate(pairs)
        ax = Axis(
            gp[1, i], 
            xlabel = models_labels[m1], ylabel = models_labels[m2],
            xticklabelsize = fs, yticklabelsize = fs,
            xticks = (xticks, string.(xticks)),
            yticks = (yticks, string.(yticks))
        );
        push!(axes, ax)
        Makie.scatter!(ax, weights[:, m1], weights[:, m2], color=color)
        if !isnothing(color_means)
            Makie.scatter!(ax, mean(weights[:, m1]), mean(weights[:, m2]),
                 color = color_means, marker = '*', markersize = marker_size
            )
        end
        if !isnothing(xlims)
            Makie.xlims!(ax, xlims...)
        end
        if !isnothing(ylims)
            Makie.ylims!(ax, ylims...)
        end
    end
    return axes
end


function myBoxplotMCMCWeights(
    gp::GridPosition,
    chains::AbstractArray, 
    models::AbstractArray{String};
    chain::Int = 1,
    chains_prior::AbstractArray = [],
    xticks::AbstractArray = 0:0.1:1,
    xlims::Union{Tuple, Nothing} = nothing,
    show_outliers::Bool = true,
    show_median::Bool = true,
    title::String = "",
    marker_size::Int = 30
)
    N_models = length(models)
    samples = chains[chain]
    prior = !isempty(chains_prior)
    N_iter = size(samples, 1)

    offset = prior ? 2 : 1
    ys = [(x-1) * offset for x in 1:N_models]
    ax = Axis(gp[1,1], xticks = (xticks, string.(xticks)), yticks = (ys, models), title=title, xlabel="Weight")
    if prior
        Makie.ylims!(ax, -offset, maximum(ys) + offset/2)
    else
        Makie.ylims!(ax, -offset/2, maximum(ys) + offset/2)
    end
    if !isnothing(xlims)
        Makie.xlims!(ax, xlims...)
    end
    for m in 1:length(models)
        Makie.boxplot!(
            ax, 
            fill(ys[m], N_iter), samples[:, m], 
            orientation = :horizontal,
            color=:grey, 
            show_outliers = show_outliers,
            show_median = show_median
        )
        Makie.scatter!(
            ax, 
            mean(samples[:, m]),
            ys[m],
            color=RGBf(206/255, 250/255, 220/255),
            marker = '*',
            markersize = marker_size,
            label = "BMA-mean"
        )
        if prior
            y = ys[m] - offset / 3
            Makie.boxplot!(
                ax, 
                fill(y, N_iter), chains_prior[chain][:, m], 
                orientation = :horizontal,
                color=:grey, 
                alpha = 0.5
            )
            Makie.scatter!(
                ax, 
                mean(chains_prior[chain][:, m]),
                y,
                color=RGBf(250/255, 206/255, 236/255),
                marker = '*',
                markersize = marker_size,
                label = "Mean prior"
            )
        end
    end
    Makie.vlines!(ax, 1/N_models, color=:grey, linestyle=:dash)
    return ax
end