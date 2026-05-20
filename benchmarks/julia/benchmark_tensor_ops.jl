# Benchmark non-AD dense ITensor vector-space operations.
#
# Run:
#   BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_tensor_ops.jl
#
# Optional args:
#   julia --project=benchmarks/julia benchmarks/julia/benchmark_tensor_ops.jl <repeats> <dim1> <dim2> ...
#
# Example matching the Rust command:
#   BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_tensor_ops.jl 20000 6 2 2 6

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using ITensors
using LinearAlgebra
using Printf
using Random

ITensors.disable_warn_order()

function parse_args(args::Vector{String})
    repeats = length(args) >= 1 ? parse(Int, args[1]) : 20_000
    repeats > 0 || error("repeats must be greater than zero")
    dims = length(args) >= 2 ? parse.(Int, args[2:end]) : [6, 2, 2, 6]
    !isempty(dims) || error("at least one dimension is required")
    all(>(0), dims) || error("all dimensions must be positive")
    return repeats, dims
end

function maybe_set_blas_threads_from_env!()
    haskey(ENV, "BLAS_NUM_THREADS") || return
    nthreads = parse(Int, ENV["BLAS_NUM_THREADS"])
    nthreads > 0 || error("BLAS_NUM_THREADS must be greater than zero")
    BLAS.set_num_threads(nthreads)
end

function elapsed_seconds(f)
    start = time_ns()
    result = f()
    return (time_ns() - start) / 1.0e9, result
end

function main()
    maybe_set_blas_threads_from_env!()
    repeats, dims = parse_args(ARGS)
    rng = MersenneTwister(0x5eed1234)
    inds = [Index(dim, "bench,n=$n") for (n, dim) in enumerate(dims)]
    a = random_itensor(rng, ComplexF64, inds...)
    b = random_itensor(rng, ComplexF64, inds...)
    alpha = 0.7 - 0.2im
    beta = -0.3 + 0.4im

    for _ in 1:32
        inner(a, b)
        norm(a)
        alpha * a + beta * b
        (conj(a) * b)[]
    end

    element_count = prod(dims)
    println("=== ITensors.jl non-AD tensor ops benchmark ===")
    println("dims=$(dims) elements=$(element_count) repeats=$(repeats) dtype=ComplexF64")

    inner_seconds, inner_checksum = elapsed_seconds() do
        checksum = zero(ComplexF64)
        for _ in 1:repeats
            checksum += inner(a, b)
        end
        checksum
    end
    @printf(
        "inner_seconds = %.6f per_call_us = %.3f checksum = %.6e%+.6eim\n",
        inner_seconds,
        inner_seconds * 1.0e6 / repeats,
        real(inner_checksum),
        imag(inner_checksum),
    )

    norm_seconds, norm_checksum = elapsed_seconds() do
        checksum = 0.0
        for _ in 1:repeats
            checksum += norm(a)
        end
        checksum
    end
    @printf(
        "norm_seconds = %.6f per_call_us = %.3f checksum = %.6e\n",
        norm_seconds,
        norm_seconds * 1.0e6 / repeats,
        norm_checksum,
    )

    axpby_seconds, axpby_checksum = elapsed_seconds() do
        checksum = 0.0
        for _ in 1:repeats
            out = alpha * a + beta * b
            checksum += norm(out)
        end
        checksum
    end
    @printf(
        "axpby_seconds = %.6f per_call_us = %.3f checksum = %.6e\n",
        axpby_seconds,
        axpby_seconds * 1.0e6 / repeats,
        axpby_checksum,
    )

    conj_contract_seconds, conj_contract_checksum = elapsed_seconds() do
        checksum = zero(ComplexF64)
        for _ in 1:repeats
            checksum += (conj(a) * b)[]
        end
        checksum
    end
    @printf(
        "conj_contract_sum_seconds = %.6f per_call_us = %.3f checksum = %.6e%+.6eim\n",
        conj_contract_seconds,
        conj_contract_seconds * 1.0e6 / repeats,
        real(conj_contract_checksum),
        imag(conj_contract_checksum),
    )
end

main()
