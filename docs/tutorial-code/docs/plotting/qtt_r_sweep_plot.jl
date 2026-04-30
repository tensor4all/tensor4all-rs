using CairoMakie
using DelimitedFiles

# Rust exports CSV files; Julia turns them into figures.
const DOCS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(DOCS_DIR, "data")
const PLOTS_DIR = joinpath(DOCS_DIR, "plots")

function ensure_dirs()
    isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)
end

function load_samples()
    path = joinpath(DATA_DIR, "qtt_r_sweep_samples.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        r = parse.(Int, rows[:, 1]),
        npoints = parse.(Int, rows[:, 2]),
        index = parse.(Int, rows[:, 3]),
        x = parse.(Float64, rows[:, 4]),
        exact = parse.(Float64, rows[:, 5]),
        qtt = parse.(Float64, rows[:, 6]),
        abs_error = parse.(Float64, rows[:, 7]),
    )
end

function load_stats()
    path = joinpath(DATA_DIR, "qtt_r_sweep_stats.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        r = parse.(Int, rows[:, 1]),
        npoints = parse.(Int, rows[:, 2]),
        build_time_sec = parse.(Float64, rows[:, 3]),
        mean_abs_error = parse.(Float64, rows[:, 4]),
        max_abs_error = parse.(Float64, rows[:, 5]),
        rank = parse.(Int, rows[:, 6]),
    )
end

function plot_samples(samples)
    rs = sort(unique(samples.r))
    selected_rs = [2, 4, 8]
    # Colorblind-friendly palette based on the Okabe-Ito colors.
    palette = [
        RGBf(0.000, 0.447, 0.698),
        RGBf(0.902, 0.624, 0.000),
        RGBf(0.000, 0.620, 0.451),
        RGBf(0.835, 0.369, 0.000),
        RGBf(0.800, 0.475, 0.655),
        RGBf(0.337, 0.706, 0.914),
        RGBf(0.941, 0.894, 0.259),
        RGBf(0.000, 0.000, 0.000),
    ]
    markers = [:circle, :rect, :diamond, :utriangle, :xcross, :star5, :pentagon, :dtriangle,
               :ltriangle, :rtriangle, :cross, :hexagon, :star4, :star8]

    # Small R have few samples → show them prominently; large R are denser → smaller
    markersize_map = Dict(2 => 32, 4 => 24, 8 => 10)

    fig = Figure(size = (1400, 750), fontsize = 28)
    ax = Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "value",
        title = "QTT samples for sin(30x)",
    )

    xs = range(0.0, 1.0, length = 1000)
    lines!(
        ax,
        xs,
        sin.(30 .* xs),
        color = (:gray55, 0.65),
        linewidth = 2.5,
        label = "analytic sin(30x)",
    )

    for (i, r) in enumerate(selected_rs)
        rows = samples.r .== r
        markersize = get(markersize_map, r, 10)
        scatter!(
            ax,
            samples.x[rows],
            samples.qtt[rows],
            color = palette[(i - 1) % length(palette) + 1],
            marker = markers[(i - 1) % length(markers) + 1],
            markersize = markersize,
            strokewidth = 0,
            label = "R = $r",
        )
    end

    axislegend(ax, position = :rb, nbanks = 2)
    return fig
end

function plot_error(stats)
    fig = Figure(size = (1100, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "R",
        ylabel = "mean absolute error",
        title = "Mean absolute QTT error per R",
        yscale = log10,
    )

    values = max.(stats.mean_abs_error, eps(Float64))
    lines!(ax, stats.r, values, color = :dodgerblue3, linewidth = 3)
    scatter!(ax, stats.r, values, color = :dodgerblue3, markersize = 12)
    return fig
end

function plot_runtime(stats)
    fig = Figure(size = (1100, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "R",
        ylabel = "build time [s]",
        title = "QTT construction runtime per R",
    )

    lines!(ax, stats.r, stats.build_time_sec, color = :forestgreen, linewidth = 3)
    scatter!(ax, stats.r, stats.build_time_sec, color = :forestgreen, markersize = 12)
    return fig
end

function main()
    ensure_dirs()
    samples = load_samples()
    stats = load_stats()

    fig1 = plot_samples(samples)
    fig2 = plot_error(stats)
    fig3 = plot_runtime(stats)

    save_plot(fig1, "qtt_r_sweep_samples")
    save_plot(fig2, "qtt_r_sweep_error")
    save_plot(fig3, "qtt_r_sweep_runtime")

    println("wrote plots to $(PLOTS_DIR)")
end

function save_plot(fig, basename)
    save(joinpath(PLOTS_DIR, "$(basename).png"), fig)
    save(joinpath(PLOTS_DIR, "$(basename).pdf"), fig)
end

main()
