using CairoMakie
using DelimitedFiles

# Rust exports CSV files; Julia turns them into figures.
const DOCS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(DOCS_DIR, "data")
const PLOTS_DIR = joinpath(DOCS_DIR, "plots")
include("bond_envelopes.jl")

function ensure_dirs()
    isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)
end

function load_samples()
    path = joinpath(DATA_DIR, "qtt_interval_samples.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        index = parse.(Int, rows[:, 1]),
        x = parse.(Float64, rows[:, 2]),
        exact = parse.(Float64, rows[:, 3]),
        qtt = parse.(Float64, rows[:, 4]),
        abs_error = parse.(Float64, rows[:, 5]),
    )
end

function load_bonds()
    path = joinpath(DATA_DIR, "qtt_interval_bond_dims.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        bond_index = parse.(Int, rows[:, 1]),
        bond_dim = parse.(Int, rows[:, 2]),
    )
end

function save_plot(fig, basename)
    save(joinpath(PLOTS_DIR, "$(basename).png"), fig)
    save(joinpath(PLOTS_DIR, "$(basename).pdf"), fig)
end

function plot_function_vs_qtt(samples)
    fig = Figure(size = (1200, 700), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "value",
        title = "QTT approximation of x² on [-1, 2]",
    )

    x_min = minimum(samples.x)
    x_max = maximum(samples.x)
    xs = range(x_min, x_max, length = 1000)

    # Draw the analytic curve first and keep it visually in the background.
    lines!(ax, xs, xs .^ 2, color = (:gray55, 0.65), linewidth = 2.5, label = "analytic x²")
    scatter!(ax, samples.x, samples.qtt, color = :dodgerblue3, markersize = 12, label = "QTT samples")
    axislegend(ax, position = :rb)
    text!(
        ax,
        x_min + 0.03 * (x_max - x_min),
        maximum(samples.exact) * 0.98,
        text = "max abs error = $(round(maximum(samples.abs_error), sigdigits = 3))",
        align = (:left, :top),
        fontsize = 16,
        color = :gray30,
    )

    return fig
end

function plot_bond_dims(bonds)
    fig = Figure(size = (1200, 600), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Bond dimensions of the interval QTT",
        yscale = log2,
    )

    lines!(ax, bonds.bond_index, bonds.bond_dim, color = :rebeccapurple, linewidth = 3, label = "bond dimension")
    scatter!(ax, bonds.bond_index, bonds.bond_dim, color = :rebeccapurple, markersize = 10)
    add_worst_case_envelope!(ax, bonds.bond_index; base = 2, label = "worst case")
    axislegend(ax, position = :rb)
    return fig
end

function main()
    ensure_dirs()
    samples = load_samples()
    bonds = load_bonds()

    fig1 = plot_function_vs_qtt(samples)
    fig2 = plot_bond_dims(bonds)

    save_plot(fig1, "qtt_interval_function_vs_qtt")
    save_plot(fig2, "qtt_interval_bond_dims")

    println("wrote plots to $(PLOTS_DIR)")
end

main()
