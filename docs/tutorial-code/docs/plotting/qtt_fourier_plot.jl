using CairoMakie
using DelimitedFiles

const DOCS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(DOCS_DIR, "data")
const PLOTS_DIR = joinpath(DOCS_DIR, "plots")
include("bond_envelopes.jl")

function ensure_dirs()
    isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)
end

function load_samples()
    path = joinpath(DATA_DIR, "qtt_fourier_samples.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        index = parse.(Int, rows[:, 1]),
        x = parse.(Float64, rows[:, 2]),
        k = parse.(Float64, rows[:, 3]),
        analytic_re = parse.(Float64, rows[:, 4]),
        analytic_im = parse.(Float64, rows[:, 5]),
        qtt_re = parse.(Float64, rows[:, 6]),
        qtt_im = parse.(Float64, rows[:, 7]),
        abs_error = parse.(Float64, rows[:, 8]),
    )
end

function load_bond_dims()
    path = joinpath(DATA_DIR, "qtt_fourier_bond_dims.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        bond_index = parse.(Int, rows[:, 1]),
        input_bond_dim = parse.(Int, rows[:, 2]),
        transformed_bond_dim = parse.(Int, rows[:, 3]),
    )
end

function load_operator_bond_dims()
    path = joinpath(DATA_DIR, "qtt_fourier_operator_bond_dims.csv")
    raw = readdlm(path, ',', String)
    rows = raw[2:end, :]

    return (
        bond_index = parse.(Int, rows[:, 1]),
        bond_dim = parse.(Int, rows[:, 2]),
    )
end

function plot_transform(samples)
    fig = Figure(size = (1300, 750), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "frequency k",
        ylabel = "real part",
        title = "Gaussian Fourier transform: analytic reference vs QTT",
    )

    lines!(
        ax,
        samples.k,
        samples.analytic_re,
        color = (:gray40, 0.7),
        linewidth = 3,
        label = "analytic reference",
    )
    scatter!(
        ax,
        samples.k,
        samples.qtt_re,
        color = :dodgerblue3,
        markersize = 8,
        label = "QTT approximation",
    )
    axislegend(ax, position = :rb)
    return fig
end

function plot_bond_dims(bonds)
    fig = Figure(size = (1100, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Input and transformed bond dimensions",
        yscale = log2,
    )

    lines!(
        ax,
        bonds.bond_index,
        bonds.input_bond_dim,
        color = :forestgreen,
        linewidth = 3,
        label = "input",
    )
    lines!(
        ax,
        bonds.bond_index,
        bonds.transformed_bond_dim,
        color = :purple4,
        linewidth = 3,
        label = "transformed",
    )
    add_worst_case_envelope!(ax, bonds.bond_index; base = 2, label = "worst case")
    axislegend(ax, position = :rb)
    return fig
end

function plot_operator_bond_dims(bonds)
    fig = Figure(size = (1100, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Fourier MPO bond dimensions",
        yscale = log2,
    )

    lines!(
        ax,
        bonds.bond_index,
        bonds.bond_dim,
        color = :darkorange3,
        linewidth = 3,
        label = "Fourier MPO",
    )
    scatter!(
        ax,
        bonds.bond_index,
        bonds.bond_dim,
        color = :darkorange3,
        markersize = 11,
    )
    add_worst_case_envelope!(ax, bonds.bond_index; base = 4, label = "worst case")
    axislegend(ax, position = :rb)
    return fig
end

function save_plot(fig, basename)
    save(joinpath(PLOTS_DIR, "$(basename).png"), fig)
    save(joinpath(PLOTS_DIR, "$(basename).pdf"), fig)
end

function main()
    ensure_dirs()
    samples = load_samples()
    bonds = load_bond_dims()
    operator_bonds = load_operator_bond_dims()

    fig1 = plot_transform(samples)
    fig2 = plot_bond_dims(bonds)
    fig3 = plot_operator_bond_dims(operator_bonds)

    save_plot(fig1, "qtt_fourier_transform")
    save_plot(fig2, "qtt_fourier_bond_dims")
    save_plot(fig3, "qtt_fourier_operator_bond_dims")

    println("wrote plots to $(PLOTS_DIR)")
end

main()
