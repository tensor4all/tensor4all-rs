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
    raw = readdlm(joinpath(DATA_DIR, "qtt_partial_fourier2d_samples.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        k_index = parse.(Int, rows[:, 1]),
        t_index = parse.(Int, rows[:, 2]),
        k = parse.(Float64, rows[:, 4]),
        t = parse.(Float64, rows[:, 5]),
        analytic_re = parse.(Float64, rows[:, 6]),
        qtt_re = parse.(Float64, rows[:, 8]),
        abs_error = parse.(Float64, rows[:, 10]),
    )
end

function load_bond_dims()
    raw = readdlm(joinpath(DATA_DIR, "qtt_partial_fourier2d_bond_dims.csv"), ',', String)
    rows = raw[2:end, :]
    parse_optional_int(value) = isempty(strip(value)) ? missing : parse(Int, value)
    return (
        bond_index = parse.(Int, rows[:, 1]),
        input_bond_dim = parse_optional_int.(rows[:, 2]),
        transformed_bond_dim = parse_optional_int.(rows[:, 3]),
    )
end

function load_operator_bond_dims()
    raw = readdlm(joinpath(DATA_DIR, "qtt_partial_fourier2d_operator_bond_dims.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        bond_index = parse.(Int, rows[:, 1]),
        bond_dim = parse.(Int, rows[:, 2]),
    )
end

function to_matrix(samples, values)
    n_k = maximum(samples.k_index)
    n_t = maximum(samples.t_index)
    matrix = Array{Float64}(undef, n_k, n_t)
    for row in eachindex(values)
        matrix[samples.k_index[row], samples.t_index[row]] = values[row]
    end
    return matrix
end

function save_plot(fig, basename)
    save(joinpath(PLOTS_DIR, string(basename, ".png")), fig)
    save(joinpath(PLOTS_DIR, string(basename, ".pdf")), fig)
end

function plot_values(samples)
    ks = sort(unique(samples.k))
    ts = sort(unique(samples.t))
    analytic = to_matrix(samples, samples.analytic_re)
    qtt = to_matrix(samples, samples.qtt_re)

    fig = Figure(size = (1350, 600), fontsize = 20)
    ax1 = Axis(
        fig[1, 1],
        xlabel = "spatial frequency k",
        ylabel = "time t",
        title = "Analytic Re F(k,t)",
    )
    hm1 = heatmap!(ax1, ks, ts, analytic'; colormap = :lipari)
    Colorbar(fig[1, 2], hm1, vertical = true, label = "Re F(k,t)", ticks = WilkinsonTicks(6))

    ax2 = Axis(
        fig[1, 3],
        xlabel = "spatial frequency k",
        ylabel = "time t",
        title = "QTT Re F(k,t)",
    )
    hm2 = heatmap!(ax2, ks, ts, qtt'; colormap = :lipari)
    Colorbar(fig[1, 4], hm2, vertical = true, label = "Re F(k,t)", ticks = WilkinsonTicks(6))

    colgap!(fig.layout, 10)
    return fig
end

function plot_error(samples)
    ks = sort(unique(samples.k))
    ts = sort(unique(samples.t))
    errors = to_matrix(samples, samples.abs_error)

    fig = Figure(size = (900, 600), fontsize = 20)
    ax = Axis(
        fig[1, 1],
        xlabel = "spatial frequency k",
        ylabel = "time t",
        title = "Absolute error",
    )
    hm = heatmap!(ax, ks, ts, errors'; colormap = :lipari)
    Colorbar(fig[1, 2], hm, vertical = true, label = "|QTT - analytic|", ticks = WilkinsonTicks(6))
    return fig
end

function plot_bond_dims(bonds, operator_bonds)
    fig = Figure(size = (1200, 550), fontsize = 22)
    ax1 = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "State bond dimensions",
        yscale = log2,
    )

    input_rows = .!ismissing.(bonds.input_bond_dim)
    transformed_rows = .!ismissing.(bonds.transformed_bond_dim)

    if any(input_rows)
        lines!(
            ax1,
            bonds.bond_index[input_rows],
            collect(skipmissing(bonds.input_bond_dim)),
            color = :dodgerblue3,
            linewidth = 3,
            label = "input",
        )
    end
    if any(transformed_rows)
        lines!(
            ax1,
            bonds.bond_index[transformed_rows],
            collect(skipmissing(bonds.transformed_bond_dim)),
            color = :purple4,
            linewidth = 3,
            label = "transformed",
        )
    end

    add_worst_case_envelope!(ax1, bonds.bond_index; base = 2, label = "worst case")
    axislegend(ax1, position = :rb)

    ax2 = Axis(
        fig[1, 2],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Partial Fourier MPO",
        yscale = log2,
    )

    lines!(
        ax2,
        operator_bonds.bond_index,
        operator_bonds.bond_dim,
        color = :darkorange3,
        linewidth = 3,
        label = "Fourier MPO",
    )
    scatter!(
        ax2,
        operator_bonds.bond_index,
        operator_bonds.bond_dim,
        color = :darkorange3,
        markersize = 11,
    )
    add_worst_case_envelope!(ax2, operator_bonds.bond_index; base = 4, label = "worst case")
    axislegend(ax2, position = :rb)

    colgap!(fig.layout, 45)

    return fig
end

function main()
    ensure_dirs()
    samples = load_samples()
    bonds = load_bond_dims()
    operator_bonds = load_operator_bond_dims()

    save_plot(plot_values(samples), "qtt_partial_fourier2d_values")
    save_plot(plot_error(samples), "qtt_partial_fourier2d_error")
    save_plot(plot_bond_dims(bonds, operator_bonds), "qtt_partial_fourier2d_bond_dims")

    println("wrote plots to ", PLOTS_DIR)
end

main()
