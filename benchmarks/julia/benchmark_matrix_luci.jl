#!/usr/bin/env julia

import Pkg

Pkg.activate(; temp=true)
if haskey(ENV, "ACI_JL_PATH")
    Pkg.develop(path=ENV["ACI_JL_PATH"])
else
    Pkg.add(url="https://github.com/tensor4all/AlternatingCrossInterpolation.jl.git")
end

using Statistics
import AlternatingCrossInterpolation as ACI

const TCI = ACI.TCI

struct MatrixLuciTiming
    selection_ns::Int
    gather_ns::Int
    left_factor_ns::Int
    right_factor_ns::Int
    rank::Int
    last_error::Float64
    checksum::Float64
end

function parse_sizes(args)
    sizes = [4, 8, 16, 32, 64]
    i = 1
    while i <= length(args)
        if args[i] == "--sizes"
            i == length(args) && error("--sizes requires a comma-separated value list")
            sizes = parse.(Int, split(args[i + 1], ","))
            i += 2
        else
            error("unknown argument: $(args[i])")
        end
    end
    return sizes
end

function hilbert_matrix(size::Int)
    A = Matrix{Float64}(undef, size, size)
    for col in 1:size, row in 1:size
        A[row, col] = 1.0 / (row + col - 1)
    end
    return A
end

function timed_once(size::Int, leftorthogonal::Bool)
    A = hilbert_matrix(size)

    t = time_ns()
    luci = TCI.MatrixLUCI(
        A;
        leftorthogonal=leftorthogonal,
        maxrank=typemax(Int),
        reltol=0.0,
        abstol=1.0e-10,
    )
    selection_ns = time_ns() - t

    t = time_ns()
    left = TCI.left(luci)
    left_factor_ns = time_ns() - t

    t = time_ns()
    right = TCI.right(luci)
    right_factor_ns = time_ns() - t

    checksum = (isempty(left) ? 0.0 : abs(left[1, 1])) +
        (isempty(right) ? 0.0 : abs(right[1, 1]))
    return MatrixLuciTiming(
        selection_ns,
        0,
        left_factor_ns,
        right_factor_ns,
        TCI.npivots(luci),
        TCI.lastpivoterror(luci),
        checksum,
    )
end

ns_to_ms(ns::Int) = ns / 1e6
total_ns(run::MatrixLuciTiming) =
    run.selection_ns + run.gather_ns + run.left_factor_ns + run.right_factor_ns

function main()
    repeats = parse(Int, get(ENV, "T4A_MATRIX_LUCI_REPEATS", "20"))
    sizes = parse_sizes(ARGS)
    println("impl,matrix,size,repeats,left_orthogonal,selection_ms,gather_ms,left_factor_ms,right_factor_ms,total_ms,rank,last_error,checksum")
    for size in sizes
        for leftorthogonal in (true, false)
            runs = [timed_once(size, leftorthogonal) for _ in 1:repeats]
            first = runs[1]
            println(join([
                "julia",
                "hilbert",
                size,
                repeats,
                leftorthogonal,
                median(ns_to_ms.(getfield.(runs, :selection_ns))),
                median(ns_to_ms.(getfield.(runs, :gather_ns))),
                median(ns_to_ms.(getfield.(runs, :left_factor_ns))),
                median(ns_to_ms.(getfield.(runs, :right_factor_ns))),
                median(ns_to_ms.(total_ns.(runs))),
                first.rank,
                first.last_error,
                first.checksum,
            ], ","))
        end
    end
end

main()
