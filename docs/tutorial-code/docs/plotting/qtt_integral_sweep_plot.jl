using CairoMakie
using DelimitedFiles

# Rust writes one CSV row per R value; Julia turns that table into a small plot.
const DOCS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(DOCS_DIR, "data")
const PLOTS_DIR = joinpath(DOCS_DIR, "plots")

function ensure_dirs()
    isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)
end

function load_sweep()
    path = joinpath(DATA_DIR, "qtt_integral_sweep.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        r = parse.(Int, rows[:, 1]),
        npoints = parse.(Int, rows[:, 2]),
        integral = parse.(Float64, rows[:, 3]),
        exact_integral = parse.(Float64, rows[:, 4]),
        abs_error = parse.(Float64, rows[:, 5]),
        rank = parse.(Int, rows[:, 6]),
    )
end

function plot_error(sweep)
    fig = Figure(size = (1000, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "R",
        ylabel = "absolute integral error",
        title = "Integral error for x^2 on [-1, 2]",
        yscale = log10,
    )

    values = max.(sweep.abs_error, eps(Float64))
    lines!(ax, sweep.r, values, color = :dodgerblue3, linewidth = 3)
    scatter!(ax, sweep.r, values, color = :dodgerblue3, markersize = 13)
    return fig
end

function save_plot(fig, basename)
    save(joinpath(PLOTS_DIR, "$(basename).png"), fig)
    save(joinpath(PLOTS_DIR, "$(basename).pdf"), fig)
end

function main()
    ensure_dirs()
    sweep = load_sweep()
    fig = plot_error(sweep)
    save_plot(fig, "qtt_integral_sweep")
    println("wrote plots to $(PLOTS_DIR)")
end

main()
