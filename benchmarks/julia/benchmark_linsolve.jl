#!/usr/bin/env julia
#
# Benchmark: linsolve (MPO/MPS linear solve) using ITensorMPS.jl.
#
# Solve:
#   (a0*I + a1*A) * x = b
#
# Constraint (Issue #160):
# - The linear operator A must be square in the sense that it maps states/operators
#   living on the same site set/topology (input sites == output sites).
#
# This script follows the benchmark style in this repo:
# - Build fixed inputs once (reproducible with fixed RNG seed)
# - Warmup run excluded from stats
# - Multiple measured runs + mean/min/max/std summary
#

import Pkg

Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "external", "ITensorMPS.jl"))
Pkg.instantiate()

using ITensors
using ITensorMPS
using KrylovKit
using InteractiveUtils
using LinearAlgebra
using Printf
using Random
using Statistics

function check_square_operator_constraint(; sites, A0, b0, x00)
    # Issue #160 requires A to act on a space with the same site set.
    # For MPOs in ITensors, site indices typically come in primed/unprimed pairs.
    @assert length(sites) > 0
    @assert siteinds(b0) == sites "b0 siteinds != sites (benchmark requires same site set)"
    @assert siteinds(x00) == sites "x00 siteinds != sites (benchmark requires same site set)"

    op_site_pairs = collect(siteinds(A0))
    @assert length(op_site_pairs) == length(sites) "A0 site set size != sites"

    sig(i) = (dim(i), string(tags(i)))

    # Each local MPO tensor should have both input and output site indices
    # (unprimed + primed) for the same site.
    for (i, site) in enumerate(sites)
        pair = op_site_pairs[i]
        @assert length(pair) == 2 "A0[$i] should have 2 site indices (in/out)"
        site_sig = sig(site)
        pair_sigs = map(sig, pair)
        @assert all(==(site_sig), pair_sigs) "A0[$i] site signature mismatch"
    end
    return nothing
end

function envinfo()
    println("=== Environment ===")
    InteractiveUtils.versioninfo()
    println("Threads.nthreads() = ", Threads.nthreads())
    println("BLAS.get_num_threads() = ", LinearAlgebra.BLAS.get_num_threads())
    println()
end

function main()
    # Parameters (smoke defaults)
    N = 10
    chi = 20
    maxdim = 20
    cutoff = 1e-8
    rtol = sqrt(cutoff) # repo convention: rtol = sqrt(cutoff)

    nsweeps = 5
    a0 = 1.0
    a1 = 1.0

    # GMRES / KrylovKit
    gmres_tol = 1e-6
    gmres_maxiter = 20
    krylovdim = 30

    seed = 1234
    rng = MersenneTwister(seed)

    n_runs = 10

    println("=== linsolve Benchmark (Julia/ITensorMPS.jl) ===")
    println("Problem: (a0*I + a1*A) * x = b")
    println("N = ", N)
    println("d = 2 (siteinds(\"S=1/2\", N))")
    println("chi (initial linkdims) = ", chi)
    println("nsweeps = ", nsweeps)
    println("maxdim = ", maxdim)
    println("cutoff = ", cutoff)
    println("rtol (tensor4all convention) = sqrt(cutoff) = ", rtol)
    println("GMRES: tol = ", gmres_tol, ", maxiter = ", gmres_maxiter, ", krylovdim = ", krylovdim)
    println("coefficients: a0 = ", a0, ", a1 = ", a1)
    println("seed = ", seed)
    println("n_runs = ", n_runs, " (excluding warmup)")
    println()

    envinfo()

    # Fixed inputs
    sites = siteinds("S=1/2", N)
    A0 = random_mpo(rng, sites)
    b0 = random_mps(rng, sites; linkdims=chi)
    x00 = random_mps(rng, sites; linkdims=chi)
    check_square_operator_constraint(; sites=sites, A0=A0, b0=b0, x00=x00)

    # ITensorMPS forwards kwargs to alternating_update; KrylovKit kwargs go via updater_kwargs
    updater_kwargs = (; tol=gmres_tol, maxiter=gmres_maxiter, krylovdim=krylovdim)

    println("Warmup run (excluded from stats)...")
    t_warmup = @elapsed begin
        A = deepcopy(A0)
        b = deepcopy(b0)
        x0 = deepcopy(x00)
        x, info = KrylovKit.linsolve(
            A, b, x0, a0, a1;
            nsweeps=nsweeps,
            maxdim=maxdim,
            cutoff=cutoff,
            updater_kwargs=updater_kwargs,
        )
        _ = (x, info)
    end
    println(@sprintf("Warmup completed in %.3f ms", 1000 * t_warmup))
    println()

    println("Measured runs...")
    times = Float64[]
    x_last = nothing
    info_last = nothing
    for run in 1:n_runs
        t = @elapsed begin
            A = deepcopy(A0)
            b = deepcopy(b0)
            x0 = deepcopy(x00)
            x, info = KrylovKit.linsolve(
                A, b, x0, a0, a1;
                nsweeps=nsweeps,
                maxdim=maxdim,
                cutoff=cutoff,
                updater_kwargs=updater_kwargs,
            )
            x_last = x
            info_last = info
        end
        push!(times, t)
        println(@sprintf("  Run %d: %.3f ms", run, 1000 * t))
    end

    avg_t = mean(times)
    min_t = minimum(times)
    max_t = maximum(times)
    std_t = std(times)

    println()
    println("=== Results ===")
    println(@sprintf("Average: %.3f ms", 1000 * avg_t))
    println(@sprintf("Min:     %.3f ms", 1000 * min_t))
    println(@sprintf("Max:     %.3f ms", 1000 * max_t))
    println(@sprintf("Stddev:  %.3f ms", 1000 * std_t))

    if info_last !== nothing
        println()
        println("=== Solver info (last run) ===")
        println(info_last)
    end

    # Optional residual check (best-effort, may be expensive)
    if x_last !== nothing && x_last isa MPS
        println()
        println("=== Residual (best-effort) ===")
        try
            # Compute r = (a0*I + a1*A)*x - b
            Ax = apply(A0, x_last; maxdim=maxdim, cutoff=cutoff)
            r = a0 * x_last + a1 * Ax - b0
            rn = norm(r)
            println("||r|| = ", rn)
        catch err
            println("Residual computation skipped (error): ", err)
        end
    end
end

main()

