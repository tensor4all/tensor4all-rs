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

struct MatrixLuTiming
    inplace_ns::Int
    borrowed_ns::Int
    rank::Int
    last_error::Float64
    checksum::Float64
end

function parse_sizes(args)
    sizes = [16, 32, 64, 128]
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

    A_copy = copy(A)
    t = time_ns()
    lu_inplace = TCI.rrlu!(
        A_copy;
        leftorthogonal=leftorthogonal,
        maxrank=typemax(Int),
        reltol=0.0,
        abstol=1.0e-10,
    )
    inplace_ns = time_ns() - t

    t = time_ns()
    lu_borrowed = TCI.rrlu(
        A;
        leftorthogonal=leftorthogonal,
        maxrank=typemax(Int),
        reltol=0.0,
        abstol=1.0e-10,
    )
    borrowed_ns = time_ns() - t

    left = TCI.left(lu_inplace; permute=false)
    right = TCI.right(lu_inplace; permute=false)
    checksum = (isempty(left) ? 0.0 : abs(left[1, 1])) +
        (isempty(right) ? 0.0 : abs(right[1, 1]))

    return MatrixLuTiming(
        inplace_ns,
        borrowed_ns,
        TCI.npivots(lu_borrowed),
        TCI.lastpivoterror(lu_borrowed),
        checksum,
    )
end

ns_to_ms(ns::Int) = ns / 1e6

function main()
    repeats = parse(Int, get(ENV, "T4A_MATRIX_LU_REPEATS", "20"))
    sizes = parse_sizes(ARGS)
    println("impl,matrix,size,repeats,left_orthogonal,inplace_ms,borrowed_ms,rank,last_error,checksum")
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
                median(ns_to_ms.(getfield.(runs, :inplace_ns))),
                median(ns_to_ms.(getfield.(runs, :borrowed_ns))),
                first.rank,
                first.last_error,
                first.checksum,
            ], ","))
        end
    end
end

main()
