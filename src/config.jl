using CairoMakie

data_dir = "./data"
target_data_dir = "./output/data"
plot_dir = "./output/plots"

Makie.set_theme!(fontsize = 10)
# COLORS_MODELS = [
#     RGBf(([191, 86, 64] ./ 255)...), # Model 1
#     RGBf(([64, 169, 191] ./ 255)...), # Model 2
#     RGBf(([80, 181, 74] ./ 255)...) # Model 3
# ]
COLORS_MODELS = [
    RGBf(([195, 82, 58] ./ 255)...), # Model 1
    RGBf(([58, 175, 195] ./ 255)...), # Model 2
    RGBf(([80, 40, 110] ./ 255)...) # Model 3
];

# for Dirichlet-Weighted averages plot and for projection plot
# COLORS = [
#     RGBf(102/255, 194/255, 165/255),
#     RGBf(252/255, 141/255, 98/255),
#     RGBf(141/255, 160/255, 203/255),
#     RGBf(231/255, 41/255, 138/255),
#     RGBf(102/155, 166/155, 30/155)
# ]
COLORS = [
    RGBf(56/255, 180/255, 145/255),  # teal
    RGBf(237/255, 120/255, 46/255),  # orange
    RGBf(100/255, 130/255, 210/255),  # blue
    RGBf(200/255, 50/255, 160/255),  # magenta
    RGBf(180/255, 180/255, 30/255),  # yellow
]

# for ECS plot
COLORS_ECS = [
    RGBf(([186, 69, 129] ./ 255)...),
    RGBf(([69, 186, 126] ./ 255)...)
]
# COLORS_ECS = [
#     RGBf(([140,  55, 185] ./ 255)...),  # violet
#     RGBf(([ 55, 155, 220] ./ 255)...),  # sky-blue
# ]

# COLORS_PROJ = [
#     RGBf(([55, 132, 200] ./ 255)...),
#     RGBf(([200,123, 55] ./ 255)...),
#     RGBf(([80, 170, 120] ./ 255)...)
# ]
COLORS_PROJ = [
    RGBf(([ 45, 115, 190] ./ 255)...),  # blue    (lum 0.15)
    RGBf(([215, 125,  45] ./ 255)...),  # orange  (lum 0.27)
    RGBf(([155,  65, 195] ./ 255)...),  # purple  (lum 0.13)
]

