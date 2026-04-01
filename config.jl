using CairoMakie

target_data_dir = "./output/data"
plot_dir = "./output/plots"

Makie.set_theme!(fontsize = 10)
COLORS_MODELS = [
    RGBf(([191, 86, 64] ./ 255)...), # Model 1
    RGBf(([64, 169, 191] ./ 255)...), # Model 2
    RGBf(([80, 181, 74] ./ 255)...) # Model 3
]

    # COLORS = [
    #     RGBf(102/255, 194/255, 165/255),
    #     RGBf(252/255, 141/255, 98/255),
    #     RGBf(141/255, 160/255, 203/255),
    #     RGBf(231/255, 41/255, 138/255),
    #     RGBf(102/155, 166/155, 30/155)
    # ]

    # COLORS2 = [
    #     RGBf(([167, 88, 105]./255)...),
    #     RGBf(([88, 167, 150]./255)...)
    # ]

    # COLORS_PROJ = [
    #     RGBf(([122, 182, 73] ./ 255)...),
    #     RGBf(([133, 73, 182] ./ 255)...),
    #     RGBf(([73, 160, 182] ./ 255)...)
    # ]

