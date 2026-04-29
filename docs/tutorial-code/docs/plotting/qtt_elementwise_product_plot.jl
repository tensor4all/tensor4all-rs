using CairoMakie
using DelimitedFiles

# This plotting script stays separate from Rust on purpose:
# Rust exports CSV data, Julia turns that data into publication-style figures.
const DOCS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(DOCS_DIR, "data")
const PLOTS_DIR = joinpath(DOCS_DIR, "plots")
include("bond_envelopes.jl")

function ensure_dirs()
    isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)
end

function load_samples()
    path = joinpath(DATA_DIR, "qtt_elementwise_product_samples.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        index = parse.(Int, rows[:, 1]),
        x = parse.(Float64, rows[:, 2]),
        cosh_exact = parse.(Float64, rows[:, 3]),
        cosh_qtt = parse.(Float64, rows[:, 4]),
        factor_b_exact = parse.(Float64, rows[:, 5]),
        factor_b_qtt = parse.(Float64, rows[:, 6]),
        product_exact = parse.(Float64, rows[:, 7]),
        product_raw = parse.(Float64, rows[:, 8]),
        product_compressed = parse.(Float64, rows[:, 9]),
        abs_error_raw = parse.(Float64, rows[:, 10]),
        abs_error_compressed = parse.(Float64, rows[:, 11]),
    )
end

function load_bonds()
    path = joinpath(DATA_DIR, "qtt_elementwise_product_bond_dims.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        bond_index = parse.(Int, rows[:, 1]),
        cosh = parse.(Int, rows[:, 2]),
        factor_b = parse.(Int, rows[:, 3]),
        product_raw = parse.(Int, rows[:, 4]),
        product_compressed = parse.(Int, rows[:, 5]),
    )
end

function plot_factors(samples)
    fig = Figure(size = (1300, 650), fontsize = 22)

    ax1 = Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "value",
        title = "Factor QTT 1: cosh(x)",
    )
    lines!(ax1, samples.x, samples.cosh_exact, color = :black, linewidth = 3, label = "analytic cosh")
    scatter!(ax1, samples.x, samples.cosh_qtt, color = :dodgerblue3, markersize = 10, label = "QTT samples")
    axislegend(ax1, position = :rb)

    ax2 = Axis(
        fig[1, 2],
        xlabel = "x",
        ylabel = "value",
        title = "Factor QTT B (default: sin(10x))",
    )
    lines!(ax2, samples.x, samples.factor_b_exact, color = :black, linewidth = 3, label = "analytic factor B")
    scatter!(ax2, samples.x, samples.factor_b_qtt, color = :darkorange3, markersize = 10, label = "QTT samples")
    axislegend(ax2, position = :rb)

    return fig
end

function plot_product(samples)
    fig = Figure(size = (1300, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "value",
        title = "Pointwise product: cosh(x) .* factor B",
    )

    # The raw TreeTN product is exact at the tensor-network level, and the
    # compressed version shows what the library can recover after bond reduction.
    lines!(ax, samples.x, samples.product_exact, color = :black, linewidth = 3, label = "analytic product")
    scatter!(ax, samples.x, samples.product_raw, color = :forestgreen, markersize = 10, label = "raw TreeTN product")
    scatter!(ax, samples.x, samples.product_compressed, color = :crimson, markersize = 10, label = "compressed TreeTN product")
    axislegend(ax, position = :rb)
    text!(
        ax,
        0.03,
        maximum(samples.product_exact) * 0.98,
        text = "max raw err = $(round(maximum(samples.abs_error_raw), sigdigits = 3))\nmax compressed err = $(round(maximum(samples.abs_error_compressed), sigdigits = 3))",
        align = (:left, :top),
        fontsize = 16,
        color = :gray30,
    )

    return fig
end

function plot_bond_dims(bonds)
    fig = Figure(size = (1300, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Bond dimensions before and after compression",
        yscale = log2,
    )

    lines!(ax, bonds.bond_index, bonds.cosh, color = :dodgerblue3, linewidth = 3, label = "cosh(x)")
    lines!(ax, bonds.bond_index, bonds.factor_b, color = :darkorange3, linewidth = 3, label = "factor B")
    lines!(ax, bonds.bond_index, bonds.product_raw, color = :forestgreen, linewidth = 3, label = "raw product")
    lines!(ax, bonds.bond_index, bonds.product_compressed, color = :crimson, linewidth = 3, label = "compressed product")
    add_worst_case_envelope!(ax, bonds.bond_index; base = 2, label = "worst case")
    axislegend(ax, position = :rb)

    return fig
end

function main()
    ensure_dirs()
    samples = load_samples()
    bonds = load_bonds()

    fig1 = plot_factors(samples)
    fig2 = plot_product(samples)
    fig3 = plot_bond_dims(bonds)

    save_plot(fig1, "qtt_elementwise_product_factors")
    save_plot(fig2, "qtt_elementwise_product_product")
    save_plot(fig3, "qtt_elementwise_product_bond_dims")

    println("wrote plots to $(PLOTS_DIR)")
end

function save_plot(fig, basename)
    save(joinpath(PLOTS_DIR, "$(basename).png"), fig)
    save(joinpath(PLOTS_DIR, "$(basename).pdf"), fig)
end

main()
