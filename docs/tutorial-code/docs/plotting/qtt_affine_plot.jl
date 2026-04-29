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
    raw = readdlm(joinpath(DATA_DIR, "qtt_affine_samples.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        x_index = parse.(Int, rows[:, 1]),
        y_index = parse.(Int, rows[:, 2]),
        x = parse.(Int, rows[:, 3]),
        y = parse.(Int, rows[:, 4]),
        source_u_periodic = parse.(Int, rows[:, 5]),
        source_v = parse.(Int, rows[:, 6]),
        source_exact = parse.(Float64, rows[:, 7]),
        periodic_exact = parse.(Float64, rows[:, 8]),
        periodic_qtt = parse.(Float64, rows[:, 9]),
        periodic_abs_error = parse.(Float64, rows[:, 10]),
        open_exact = parse.(Float64, rows[:, 11]),
        open_qtt = parse.(Float64, rows[:, 12]),
        open_abs_error = parse.(Float64, rows[:, 13]),
    )
end

function load_bond_dims()
    raw = readdlm(joinpath(DATA_DIR, "qtt_affine_bond_dims.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        bond_index = parse.(Int, rows[:, 1]),
        input_bond_dim = parse_optional_int.(rows[:, 2]),
        periodic_transformed_bond_dim = parse_optional_int.(rows[:, 3]),
        open_transformed_bond_dim = parse_optional_int.(rows[:, 4]),
    )
end

function load_operator_bond_dims()
    raw = readdlm(joinpath(DATA_DIR, "qtt_affine_operator_bond_dims.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        bond_index = parse.(Int, rows[:, 1]),
        periodic_operator_bond_dim = parse_optional_int.(rows[:, 2]),
        open_operator_bond_dim = parse_optional_int.(rows[:, 3]),
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
    source = to_matrix(samples, samples.source_exact)
    periodic = to_matrix(samples, samples.periodic_qtt)
    open = to_matrix(samples, samples.open_qtt)
    colorrange = extrema(vcat(vec(source), vec(periodic), vec(open)))

    fig = Figure(size = (1800, 600), fontsize = 20)
    titles = ["source g(x, y)", "periodic pullback", "open pullback"]
    matrices = [source, periodic, open]
    for i in 1:3
        ax = Axis(fig[1, 2i-1], xlabel = "x", ylabel = "y", title = titles[i])
        hm = heatmap!(ax, xs, ys, matrices[i]; colorrange, colormap = :lipari)
        Colorbar(fig[1, 2i], hm, vertical = true, ticks = WilkinsonTicks(6))
    end
    colgap!(fig.layout, 10)
    return fig
end

function plot_errors(samples)
    xs = sort(unique(samples.x))
    ys = sort(unique(samples.y))
    periodic_error = to_matrix(samples, samples.periodic_abs_error)
    open_error = to_matrix(samples, samples.open_abs_error)

    fig = Figure(size = (1400, 600), fontsize = 20)
    titles = ["periodic abs error", "open abs error"]
    matrices = [periodic_error, open_error]
    for i in 1:2
        ax = Axis(fig[1, 2i-1], xlabel = "x", ylabel = "y", title = titles[i])
        hm = heatmap!(ax, xs, ys, matrices[i]; colormap = :lipari)
        Colorbar(fig[1, 2i], hm, vertical = true, ticks = WilkinsonTicks(6))
    end
    colgap!(fig.layout, 10)
    return fig
end

function plot_bond_dims(bonds)
    fig = Figure(size = (1200, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Affine pullback bond dimensions",
        yscale = log2,
    )

    rows_input = .!ismissing.(bonds.input_bond_dim)
    rows_periodic = .!ismissing.(bonds.periodic_transformed_bond_dim)
    rows_open = .!ismissing.(bonds.open_transformed_bond_dim)


    lines!(
        ax,
        bonds.bond_index[rows_periodic],
        collect(skipmissing(bonds.periodic_transformed_bond_dim)),
        color = :dodgerblue3,
        linewidth = 3,
        label = "periodic",
    )
    lines!(
        ax,
        bonds.bond_index[rows_open],
        collect(skipmissing(bonds.open_transformed_bond_dim)),
        color = :firebrick3,
        linewidth = 3,
        label = "open",
    )
        lines!(
        ax,
        bonds.bond_index[rows_input],
        collect(skipmissing(bonds.input_bond_dim)),
        color = :deeppink,
        linewidth = 3,
        label = "input", 
        linestyle = :dash,
    )
    scatter!(
        ax,
        bonds.bond_index[rows_input],
        collect(skipmissing(bonds.input_bond_dim)),
        color = :deeppink, 
        markersize = 15
    )

    add_worst_case_envelope!(ax, bonds.bond_index; base = 4, label = "worst case")
    axislegend(ax, position = :rb)
    return fig
end

function plot_operator_bond_dims(bonds)
    fig = Figure(size = (1200, 650), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Affine MPO bond dimensions",
        yscale = log2,
    )

    rows_periodic = .!ismissing.(bonds.periodic_operator_bond_dim)
    rows_open = .!ismissing.(bonds.open_operator_bond_dim)

    lines!(
        ax,
        bonds.bond_index[rows_periodic],
        collect(skipmissing(bonds.periodic_operator_bond_dim)),
        color = :dodgerblue3,
        linewidth = 3,
        label = "periodic MPO",
    )
    scatter!(
        ax,
        bonds.bond_index[rows_periodic],
        collect(skipmissing(bonds.periodic_operator_bond_dim)),
        color = :dodgerblue3,
        markersize = 10,
    )
    lines!(
        ax,
        bonds.bond_index[rows_open],
        collect(skipmissing(bonds.open_operator_bond_dim)),
        color = :firebrick3,
        linewidth = 3,
        label = "open MPO",
    )
    scatter!(
        ax,
        bonds.bond_index[rows_open],
        collect(skipmissing(bonds.open_operator_bond_dim)),
        color = :firebrick3,
        markersize = 10,
    )
    add_worst_case_envelope!(ax, bonds.bond_index; base = 4, label = "worst case")
    axislegend(ax, position = :rb)
    return fig
end

function main()
    ensure_dirs()
    samples = load_samples()
    bonds = load_bond_dims()
    operator_bonds = load_operator_bond_dims()

    save_plot(plot_values(samples), "qtt_affine_values")
    save_plot(plot_errors(samples), "qtt_affine_error")
    save_plot(plot_bond_dims(bonds), "qtt_affine_bond_dims")
    save_plot(plot_operator_bond_dims(operator_bonds), "qtt_affine_operator_bond_dims")

    println("wrote plots to $(PLOTS_DIR)")
end

main()
