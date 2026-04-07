import ModelWeights.Plots as mwp
using CairoMakie
using Distributions
using Statistics

include("config.jl")

# -------------------------- Dirichlet Prior -------------------------- #
function meanDirichlet(alphas)
    alpha0 = sum(alphas)
    return alphas ./ alpha0
end

function varDirichlet(alphas)
    alpha0 = sum(alphas)
    numerator = alphas .* (alpha0 .- alphas)
    denominator = alpha0^2 * (alpha0 + 1)
    return numerator ./ denominator
end

function stdDirichlet(alphas)
    return sqrt.(varDirichlet(alphas))
end


begin
    w = [0.1, 0.4, 0.5]
    ws = [
        fill(1.0, 3),
        round.(fill(1/3, 3), digits=2),
        round.(w .* 0.1, digits=2),
        round.(w .* 10, digits=2), 
        round.(w .* 100, digits=2)
    ]
    xlabels = string.(ws)
    ys = [1, 5:5:(length(ws)-1) * 5...]
    f = Figure(size = (325, 350));
    ax = Axis(f[1,1],     
        yticks = (ys,  string.(ws)), yticklabelsize = 10,
        xticks = (0:0.2:1, string.(0:0.2:1)), xticklabelsize = 10,
        ylabel="Dirichlet Parameters",
        xlabel = "Weights"
    )
    Makie.ylims!(ax, 0, (length(ws)+2) * 5)
    N = 10000;
    for (i, w) in enumerate(ws)
        dirichlet = Distributions.Dirichlet(w)

        mus = meanDirichlet(w)
        sigmas_sq = varDirichlet(w)
        sigmas = sqrt.(sigmas_sq)

        weight_vectors = rand(dirichlet, N)
        n_models = length(w)
        for m in 1:n_models
            y = i==1 ? 1 : 5 * (i-1)
            Makie.density!(
                ax,
                weight_vectors[m, :], 
                offset = y,
                direction = :x,
                alpha=0.7,
                color = COLORS_MODELS[m],
                label = "Model $m"
            )
            Makie.scatter!(ax2, mus[m], y, marker='*', markersize=30, color = COLORS_MODELS[m])
            # Makie.scatter!(
            #     ax, mean(weight_vectors[m,:]), y, marker='+', markersize=30, color = COLORS[m]
            # )
            # xs = mus[m] - sigmas[m] : 0.01: mus[m] + sigmas[m]
            # Makie.lines!(ax, xs, fill(y, length(xs)), color=COLORS[m], label="Model $m", linewidth=5)
        end
    end
    axislegend(
        merge = true, 
        orientation = :vertical,
        position = :rt, 
        framevisible = false, 
        patchsize = (10, 10)
    )
    f
end
mwp.savePlot(f, joinpath(plot_dir, "fig1a.pdf"); overwrite=true)


# -------------------- Some more plots (not in paper) ----------------------#
# Distribution of maximum weight values for different Dirichlet distributions
dirAlpha(N, x) = fill(1/(x*N), N)

f1 = Figure(size=(800,300))
N = 38
titles = [
    latexstring("\\alpha = 1/(N); \\quad N=$N"), 
    latexstring("\\alpha = \\frac{1}{10N}; \\quad N=$N")
]
for (i,x) in enumerate([1, 10])
    alphas = dirAlpha(N, x)
    prior = Distributions.Dirichlet(alphas)
    prior_samples = rand(prior, 1000)
    max_weights = vec(maximum(prior_samples; dims=1))

    alpha_round = round(alphas[1], digits=3)
    Makie.hist!(
        Axis(f1[1,i], xlabel="Maximum weight", title = titles[i]),
        max_weights
    )
end
f1

# Inverse Gamma distriubtions (not in paper)
f2 = Figure()
ax2 = Axis(f2[1,1], title = "InverseGamma distributions")
xs = 0:0.01:5
for (a,b) in [(1,1), (2,1), (3,3), (2,3)]
    d = InverseGamma(a,b)
    ys = Distributions.pdf.(d, xs)
    Makie.lines!(ax2, xs, ys, label = "($a,$b)")
end
axislegend(ax2)
f2