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
    raw = readdlm(joinpath(DATA_DIR, "qtt_multivariate_samples.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        x_index = parse.(Int, rows[:, 1]),
        y_index = parse.(Int, rows[:, 2]),
        x = parse.(Float64, rows[:, 3]),
        y = parse.(Float64, rows[:, 4]),
        exact = parse.(Float64, rows[:, 5]),
        interleaved_qtt = parse.(Float64, rows[:, 6]),
        grouped_qtt = parse.(Float64, rows[:, 7]),
        interleaved_abs_error = parse.(Float64, rows[:, 8]),
        grouped_abs_error = parse.(Float64, rows[:, 9]),
    )
end

function load_bonds()
    raw = readdlm(joinpath(DATA_DIR, "qtt_multivariate_bond_dims.csv"), ',', String)
    rows = raw[2:end, :]
    parse_optional_int(value) = isempty(value) ? missing : parse(Int, value)
    return (
        bond_index = parse.(Int, rows[:, 1]),
        interleaved_bond_dim = parse_optional_int.(rows[:, 2]),
        grouped_bond_dim = parse_optional_int.(rows[:, 3]),
    )
end

function to_matrix(samples, values)
    nx = maximum(samples.x_index)
    ny = maximum(samples.y_index)
    matrix = Array{Float64}(undef, nx, ny)
    for row in eachindex(values)
        matrix[samples.x_index[row], samples.y_index[row]] = values[row]
    end
    return matrix
end

function save_plot(fig, basename)
    save(joinpath(PLOTS_DIR, "$(basename).png"), fig)
    save(joinpath(PLOTS_DIR, "$(basename).pdf"), fig)
end

function plot_values(samples)
    xs = sort(unique(samples.x))
    ys = sort(unique(samples.y))
    exact = to_matrix(samples, samples.exact)
    interleaved = to_matrix(samples, samples.interleaved_qtt)
    grouped = to_matrix(samples, samples.grouped_qtt)

    fig = Figure(size = (1500, 520), fontsize = 20)
    titles = ["exact f(x, y)", "interleaved QTT", "grouped QTT"]
    matrices = [exact, interleaved, grouped]
    for i in 1:3
        ax = Axis(fig[1, i], xlabel = "x", ylabel = "y", title = titles[i])
        hm = heatmap!(ax, xs, ys, matrices[i])
        Colorbar(fig[2, i], hm, vertical = false)
    end
    return fig
end

function plot_errors(samples)
    xs = sort(unique(samples.x))
    ys = sort(unique(samples.y))
    interleaved_error = to_matrix(samples, samples.interleaved_abs_error)
    grouped_error = to_matrix(samples, samples.grouped_abs_error)

    fig = Figure(size = (1100, 520), fontsize = 20)
    titles = ["interleaved abs error", "grouped abs error"]
    matrices = [interleaved_error, grouped_error]
    for i in 1:2
        ax = Axis(fig[1, i], xlabel = "x", ylabel = "y", title = titles[i])
        hm = heatmap!(ax, xs, ys, matrices[i])
        Colorbar(fig[2, i], hm, vertical = false)
    end
    return fig
end

function plot_bond_dims(bonds)
    fig = Figure(size = (1200, 600), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Bond dimensions for 2D QTT layouts",
        yscale = log2,
    )

    interleaved_rows = .!ismissing.(bonds.interleaved_bond_dim)
    grouped_rows = .!ismissing.(bonds.grouped_bond_dim)

    lines!(
        ax,
        bonds.bond_index[interleaved_rows],
        collect(skipmissing(bonds.interleaved_bond_dim)),
        color = :dodgerblue3,
        linewidth = 3,
        label = "interleaved",
    )
    scatter!(
        ax,
        bonds.bond_index[interleaved_rows],
        collect(skipmissing(bonds.interleaved_bond_dim)),
        color = :dodgerblue3,
        markersize = 10,
    )
    lines!(
        ax,
        bonds.bond_index[grouped_rows],
        collect(skipmissing(bonds.grouped_bond_dim)),
        color = :firebrick3,
        linewidth = 3,
        label = "grouped",
    )
    scatter!(
        ax,
        bonds.bond_index[grouped_rows],
        collect(skipmissing(bonds.grouped_bond_dim)),
        color = :firebrick3,
        markersize = 10,
    )
    add_worst_case_envelope!(ax, bonds.bond_index; base = 2, label = "worst case")
    axislegend(ax, position = :rb)
    return fig
end

function main()
    ensure_dirs()
    samples = load_samples()
    bonds = load_bonds()

    save_plot(plot_values(samples), "qtt_multivariate_values")
    save_plot(plot_errors(samples), "qtt_multivariate_error")
    save_plot(plot_bond_dims(bonds), "qtt_multivariate_bond_dims")

    println("wrote plots to $(PLOTS_DIR)")
end

main()
