using CairoMakie
using DelimitedFiles

const DOCS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(DOCS_DIR, "data")
const PLOTS_DIR = joinpath(DOCS_DIR, "plots")
include("bond_envelopes.jl")

function ensure_dirs()
    isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)
end

function parse_optional_int(value)
    return isempty(value) ? missing : parse(Int, value)
end

function load_samples()
    raw = readdlm(joinpath(DATA_DIR, "qtt_difference_kernel_samples.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        x_index = parse.(Int, rows[:, 1]),
        xprime_index = parse.(Int, rows[:, 2]),
        x = parse.(Int, rows[:, 3]),
        xprime = parse.(Int, rows[:, 4]),
        difference = parse.(Int, rows[:, 5]),
        source_exact = parse.(Float64, rows[:, 6]),
        source_qtt = parse.(Float64, rows[:, 7]),
        source_abs_error = parse.(Float64, rows[:, 8]),
        kernel_exact = parse.(Float64, rows[:, 9]),
        kernel_mpo = parse.(Float64, rows[:, 10]),
        abs_error = parse.(Float64, rows[:, 11]),
    )
end

function load_bond_dims()
    raw = readdlm(joinpath(DATA_DIR, "qtt_difference_kernel_bond_dims.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        bond_index = parse.(Int, rows[:, 1]),
        kernel_qtt_bond_dim = parse_optional_int.(rows[:, 2]),
        difference_kernel_mpo_bond_dim = parse_optional_int.(rows[:, 3]),
    )
end

function to_matrix(samples, values)
    nx = maximum(samples.x_index)
    nxprime = maximum(samples.xprime_index)
    matrix = Array{Float64}(undef, nx, nxprime)
    for row in eachindex(values)
        matrix[samples.x_index[row], samples.xprime_index[row]] = values[row]
    end
    return matrix
end

function source_profile(samples)
    first_row = findall(samples.xprime .== 0)
    order = sortperm(samples.difference[first_row])
    rows = first_row[order]
    return (
        difference = samples.difference[rows],
        exact = samples.source_exact[rows],
        qtt = samples.source_qtt[rows],
    )
end

function save_plot(fig, basename)
    save(joinpath(PLOTS_DIR, "$(basename).png"), fig)
    save(joinpath(PLOTS_DIR, "$(basename).pdf"), fig)
end

function plot_values(samples)
    profile = source_profile(samples)
    xs = sort(unique(samples.x))
    xprimes = sort(unique(samples.xprime))
    matrix = to_matrix(samples, samples.kernel_mpo)

    fig = Figure(size = (1500, 650), fontsize = 22)

    ax1 = Axis(
        fig[1, 1],
        xlabel = "z",
        ylabel = "f(z)",
        title = "Periodic source kernel",
    )
    lines!(ax1, profile.difference, profile.exact, color = :black, linewidth = 3, label = "analytic")
    scatter!(ax1, profile.difference, profile.qtt, color = :dodgerblue3, markersize = 10, label = "QTT")
    axislegend(ax1, position = :rt)

    ax2 = Axis(
        fig[1, 2],
        xlabel = "x",
        ylabel = "x'",
        title = "MPO matrix A[x, x']",
    )
    hm = heatmap!(ax2, xs, xprimes, matrix; colormap = :lipari)
    Colorbar(fig[1, 3], hm, vertical = true, ticks = WilkinsonTicks(6))

    colgap!(fig.layout, 20)
    return fig
end

function plot_error(samples)
    xs = sort(unique(samples.x))
    xprimes = sort(unique(samples.xprime))
    matrix = to_matrix(samples, samples.abs_error)

    fig = Figure(size = (900, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "x'",
        title = "Difference-kernel MPO absolute error",
    )
    hm = heatmap!(ax, xs, xprimes, matrix; colormap = :lipari)
    Colorbar(fig[1, 2], hm, vertical = true, ticks = WilkinsonTicks(6))
    return fig
end

function plot_bond_dims(bonds)
    fig = Figure(size = (1200, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Difference-kernel MPO bond dimensions",
        yscale = log2,
    )

    qtt_rows = .!ismissing.(bonds.kernel_qtt_bond_dim)
    mpo_rows = .!ismissing.(bonds.difference_kernel_mpo_bond_dim)

    lines!(
        ax,
        bonds.bond_index[qtt_rows],
        collect(skipmissing(bonds.kernel_qtt_bond_dim)),
        color = :dodgerblue3,
        linewidth = 3,
        label = "kernel QTT",
    )
    scatter!(
        ax,
        bonds.bond_index[qtt_rows],
        collect(skipmissing(bonds.kernel_qtt_bond_dim)),
        color = :dodgerblue3,
        markersize = 10,
    )
    lines!(
        ax,
        bonds.bond_index[mpo_rows],
        collect(skipmissing(bonds.difference_kernel_mpo_bond_dim)),
        color = :crimson,
        linewidth = 3,
        label = "difference-kernel MPO",
    )
    scatter!(
        ax,
        bonds.bond_index[mpo_rows],
        collect(skipmissing(bonds.difference_kernel_mpo_bond_dim)),
        color = :crimson,
        markersize = 10,
    )

    add_worst_case_envelope!(ax, bonds.bond_index; base = 4, label = "worst case MPO")
    axislegend(ax, position = :rb)
    return fig
end

function main()
    ensure_dirs()
    samples = load_samples()
    bonds = load_bond_dims()

    save_plot(plot_values(samples), "qtt_difference_kernel_values")
    save_plot(plot_error(samples), "qtt_difference_kernel_error")
    save_plot(plot_bond_dims(bonds), "qtt_difference_kernel_bond_dims")

    println("wrote plots to $(PLOTS_DIR)")
end

main()
