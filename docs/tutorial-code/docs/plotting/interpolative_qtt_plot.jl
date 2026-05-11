using CairoMakie
using DelimitedFiles

const DOCS_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(DOCS_DIR, "data")
const PLOTS_DIR = joinpath(DOCS_DIR, "plots")
include("bond_envelopes.jl")

function ensure_dirs()
    isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)
end

function load_1d_samples()
    raw = readdlm(joinpath(DATA_DIR, "interpolative_qtt_1d_samples.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        case_name = rows[:, 1],
        index = parse.(Int, rows[:, 2]),
        x = parse.(Float64, rows[:, 3]),
        exact = parse.(Float64, rows[:, 4]),
        qtt = parse.(Float64, rows[:, 5]),
        abs_error = parse.(Float64, rows[:, 6]),
    )
end

function load_2d_samples()
    raw = readdlm(joinpath(DATA_DIR, "interpolative_qtt_2d_samples.csv"), ',', String)
    rows = raw[2:end, :]
    return (
        x_index = parse.(Int, rows[:, 1]),
        y_index = parse.(Int, rows[:, 2]),
        x = parse.(Float64, rows[:, 3]),
        y = parse.(Float64, rows[:, 4]),
        exact = parse.(Float64, rows[:, 5]),
        qtt = parse.(Float64, rows[:, 6]),
        abs_error = parse.(Float64, rows[:, 7]),
    )
end

function load_bonds()
    raw = readdlm(joinpath(DATA_DIR, "interpolative_qtt_bond_dims.csv"), ',', String)
    rows = raw[2:end, :]
    parse_optional_int(value) = isempty(value) ? missing : parse(Int, value)
    return (
        bond_index = parse.(Int, rows[:, 1]),
        single_scale_1d = parse_optional_int.(rows[:, 2]),
        multi_scale_1d = parse_optional_int.(rows[:, 3]),
        multi_scale_2d = parse_optional_int.(rows[:, 4]),
    )
end

function select_case(samples, case_name)
    rows = samples.case_name .== case_name
    return (
        x = samples.x[rows],
        exact = samples.exact[rows],
        qtt = samples.qtt[rows],
        abs_error = samples.abs_error[rows],
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

function plot_1d_values(samples)
    smooth = select_case(samples, "single_scale_1d")
    multiscale = select_case(samples, "multi_scale_1d")

    fig = Figure(size = (1500, 650), fontsize = 20)
    panels = [
        (
            title = "single-scale: exp(-x²)",
            data = smooth,
            color = :dodgerblue3,
            legend_position = :rb,
        ),
        (
            title = "multiscale: softened 1 / r²",
            data = multiscale,
            color = :firebrick3,
            legend_position = :rt,
        ),
    ]

    for (column, panel) in enumerate(panels)
        ax = Axis(
            fig[1, column],
            xlabel = "x",
            ylabel = "value",
            title = panel.title,
        )
        lines!(ax, panel.data.x, panel.data.exact, color = :black, linewidth = 3, label = "analytic target")
        scatter!(ax, panel.data.x, panel.data.qtt, color = panel.color, markersize = 9, label = "QTT samples")
        text!(
            ax,
            minimum(panel.data.x),
            maximum(panel.data.exact) * 0.94,
            text = "max abs error = $(round(maximum(panel.data.abs_error), sigdigits = 3))",
            align = (:left, :top),
            fontsize = 15,
            color = :gray30,
        )
        axislegend(ax, position = panel.legend_position)
    end

    return fig
end

function plot_2d_values(samples)
    xs = sort(unique(samples.x))
    ys = sort(unique(samples.y))
    exact = to_matrix(samples, samples.exact)
    qtt = to_matrix(samples, samples.qtt)
    log_error = log10.(max.(to_matrix(samples, samples.abs_error), eps(Float64)))

    xrange = (minimum(xs), maximum(xs))
    yrange = (minimum(ys), maximum(ys))

    fig = Figure(size = (1800, 600), fontsize = 20)
    titles = ["exact softened 1 / r²", "multiscale QTT", "log10 abs error"]
    matrices = [exact, qtt, log_error]
    colorbars = ["value", "value", "log10 error"]
    for i in 1:3
        ax = Axis(fig[1, 2i-1],
            xlabel = "x", ylabel = "y", title = titles[i],
            limits = (xrange..., yrange...))
        hm = heatmap!(ax, xs, ys, matrices[i]; colormap = :lipari)
        Colorbar(fig[1, 2i], hm, vertical = true, label = colorbars[i], ticks = WilkinsonTicks(6))
    end
    colgap!(fig.layout, 10)
    return fig
end

function plot_bond_dims(bonds)
    fig = Figure(size = (1200, 600), fontsize = 22)
    ax = Axis(
        fig[1, 1],
        xlabel = "bond index",
        ylabel = "bond dimension",
        title = "Bond dimensions for interpolative QTTs",
        yscale = log2,
    )

    series = [
        (bonds.single_scale_1d, :dodgerblue3, "single-scale 1D"),
        (bonds.multi_scale_1d, :firebrick3, "multiscale 1D"),
        (bonds.multi_scale_2d, :seagreen4, "multiscale 2D"),
    ]

    for (values, color, label) in series
        rows = .!ismissing.(values)
        lines!(
            ax,
            bonds.bond_index[rows],
            collect(skipmissing(values)),
            color = color,
            linewidth = 3,
            label = label,
        )
        scatter!(
            ax,
            bonds.bond_index[rows],
            collect(skipmissing(values)),
            color = color,
            markersize = 10,
        )
    end

    add_worst_case_envelope!(ax, bonds.bond_index; base = 2, label = "binary worst case")
    axislegend(ax, position = :rt)
    return fig
end

function main()
    ensure_dirs()
    samples_1d = load_1d_samples()
    samples_2d = load_2d_samples()
    bonds = load_bonds()

    save_plot(plot_1d_values(samples_1d), "interpolative_qtt_1d_values")
    save_plot(plot_2d_values(samples_2d), "interpolative_qtt_2d_values")
    save_plot(plot_bond_dims(bonds), "interpolative_qtt_bond_dims")

    println("wrote plots to $(PLOTS_DIR)")
end

main()
