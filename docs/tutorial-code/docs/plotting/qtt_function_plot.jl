using CairoMakie
using DelimitedFiles

# The Rust binary writes CSV files into docs/data and this script renders plots
# into docs/plots. Keeping the visualization here avoids mixing it with the
# Rust implementation.
const DOCS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(DOCS_DIR, "data")
const PLOTS_DIR = joinpath(DOCS_DIR, "plots")
include("bond_envelopes.jl")

function ensure_dirs()
    isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)
end

function load_samples()
    path = joinpath(DATA_DIR, "qtt_function_samples.csv")
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
    path = joinpath(DATA_DIR, "qtt_function_bond_dims.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        bond_index = parse.(Int, rows[:, 1]),
        bond_dim = parse.(Int, rows[:, 2]),
    )
end

function plot_function_vs_qtt(samples)
    fig = Figure(size = (1200, 700), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "value",
        title = "QTT approximation of the target function",
    )

    # The notebook mostly compares curves through line plots, so we use the same
    # style here and overlay the QTT values as points.
    lines!(ax, samples.x, samples.exact, color = :black, linewidth = 3, label = "analytic target")
    scatter!(ax, samples.x, samples.qtt, color = :dodgerblue3, markersize = 12, label = "QTT samples")
    axislegend(ax, position = :rb)
    text!(
        ax,
        0.03,
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
        title = "Bond dimensions of the QTT",
        yscale = log2,
    )

    # This matches the notebook style more closely: a line plot and a log2 axis.
    lines!(ax, bonds.bond_index, bonds.bond_dim, color = :rebeccapurple, linewidth = 3, label = "bond dimension")
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

    save(joinpath(PLOTS_DIR, "qtt_function_vs_qtt.png"), fig1)
    save(joinpath(PLOTS_DIR, "qtt_function_vs_qtt.pdf"), fig1)
    save(joinpath(PLOTS_DIR, "qtt_function_bond_dims.png"), fig2)
    save(joinpath(PLOTS_DIR, "qtt_function_bond_dims.pdf"), fig2)

    println("wrote plots to $(PLOTS_DIR)")
end

main()
