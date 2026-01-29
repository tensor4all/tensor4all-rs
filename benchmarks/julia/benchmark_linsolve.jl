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

function make_identity_mpo(sites)
    Nloc = length(sites)
    w = Vector{ITensor}(undef, Nloc)
    links = [Index(1, "Link,l=$n") for n in 1:max(0, Nloc - 1)]

    for i in 1:Nloc
        s = sites[i]
        sp = prime(s)

        if Nloc == 1
            t = ITensor(sp, s)
            for j in 1:dim(s)
                t[sp => j, s => j] = 1.0
            end
            w[i] = t
        elseif i == 1
            r = links[i]
            t = ITensor(sp, s, r)
            for j in 1:dim(s)
                t[sp => j, s => j, r => 1] = 1.0
            end
            w[i] = t
        elseif i == Nloc
            l = links[i - 1]
            t = ITensor(l, sp, s)
            for j in 1:dim(s)
                t[l => 1, sp => j, s => j] = 1.0
            end
            w[i] = t
        else
            l = links[i - 1]
            r = links[i]
            t = ITensor(l, sp, s, r)
            for j in 1:dim(s)
                t[l => 1, sp => j, s => j, r => 1] = 1.0
            end
            w[i] = t
        end
    end

    return MPO(w)
end

function make_random_mpo(sites; linkdim, rng)
    Nloc = length(sites)
    w = Vector{ITensor}(undef, Nloc)
    links = [Index(linkdim, "Link,l=$n") for n in 1:max(0, Nloc - 1)]

    for i in 1:Nloc
        s = sites[i]
        sp = prime(s)

        inds = if Nloc == 1
            (sp, s)
        elseif i == 1
            (sp, s, links[i])
        elseif i == Nloc
            (links[i - 1], sp, s)
        else
            (links[i - 1], sp, s, links[i])
        end

        dims = Tuple(map(dim, inds))
        data = randn(rng, Float64, dims...)
        w[i] = ITensor(data, inds...)
    end

    return MPO(w)
end

function rel_residual(; A, b, x, a0, a1, maxdim, cutoff)
    Ax = apply(A, x; maxdim=maxdim, cutoff=cutoff)
    r = a0 * x + a1 * Ax - b
    bn = norm(b)
    return bn > 0 ? norm(r) / bn : norm(r)
end

function main()
    # Parameters (smoke defaults)
    N = 3
    chi = 20
    maxdim = 20
    cutoff = 1e-8
    rtol = sqrt(cutoff) # repo convention: rtol = sqrt(cutoff)

    nsweeps = 10
    # Set coefficients so solver actually uses operator A: (a0*I + a1*A)
    a0 = 0.0
    a1 = 1.0

    # GMRES / KrylovKit
    gmres_tol = 1e-6
    gmres_maxiter = 20
    krylovdim = 30

    seed = 1234

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
    println("seed = ", seed, " (used for random MPO)")
    println("n_runs = ", n_runs, " (excluding warmup)")
    println()

    envinfo()

    # Fixed inputs
    sites = siteinds("S=1/2", N)
    # Align with Rust benchmark: use a random MPO with a fixed RNG seed.
    rng = MersenneTwister(seed)
    A0 = make_random_mpo(sites; linkdim=chi, rng=rng)

    # Helper: create an MPS whose all tensor elements are 1.0
    function make_ones_mps(sites; linkdims=chi)
        Nloc = length(sites)
        v = Vector{ITensor}(undef, Nloc)
        links = [Index(linkdims, "Link,l=$n") for n in 1:max(0, Nloc - 1)]
        for i in 1:Nloc
            # determine indices for this site tensor
            inds = if Nloc == 1
                (sites[i],)
            elseif i == 1
                (sites[i], links[i])
            elseif i == Nloc
                (links[i - 1], sites[i])
            else
                (links[i - 1], sites[i], links[i])
            end

            # build dense array of ones with the proper shape
            dims = Tuple(map(j -> dim(j), inds))
            data = ones(Float64, dims...)

            # construct ITensor from dense data and indices
            t = ITensor(data, inds...)
            v[i] = t
        end
        return MPS(v)
    end

    # x_true: all-ones MPS; define b = A * x_true, and initialize x with RHS b.
    x00 = make_ones_mps(sites; linkdims=chi)
    b0 = apply(A0, x00; maxdim=maxdim, cutoff=cutoff)
    check_square_operator_constraint(; sites=sites, A0=A0, b0=b0, x00=x00)

    # ITensorMPS forwards kwargs to alternating_update; KrylovKit kwargs go via updater_kwargs
    updater_kwargs = (; tol=gmres_tol, maxiter=gmres_maxiter, krylovdim=krylovdim)

    println("Warmup run (excluded from stats)...")
    t_warmup = @elapsed begin
        A = deepcopy(A0)
        b = deepcopy(b0)
        # initialize x with RHS b (x^(0) = b)
        x0 = deepcopy(b0)

        r0 = rel_residual(; A=A, b=b, x=x0, a0=a0, a1=a1, maxdim=maxdim, cutoff=cutoff)
        x = KrylovKit.linsolve(
            A, b, x0, a0, a1;
            nsweeps=nsweeps,
            maxdim=maxdim,
            cutoff=cutoff,
            updater_kwargs=updater_kwargs,
        )
        r1 = rel_residual(; A=A, b=b, x=x, a0=a0, a1=a1, maxdim=maxdim, cutoff=cutoff)
        println(@sprintf("Warmup residual (rel): %.3e -> %.3e", r0, r1))
    end
    println(@sprintf("Warmup completed in %.3f ms", 1000 * t_warmup))
    println()

    println("Measured runs...")
    times = Float64[]
    x_last = nothing
    for run in 1:n_runs
        t = @elapsed begin
            A = deepcopy(A0)
            b = deepcopy(b0)
            # initialize x with RHS b (x^(0) = b)
            x0 = deepcopy(b0)
            x = KrylovKit.linsolve(
                A, b, x0, a0, a1;
                nsweeps=nsweeps,
                maxdim=maxdim,
                cutoff=cutoff,
                updater_kwargs=updater_kwargs,
            )
            x_last = x
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
            bn = norm(b0)
            println("||r||/||b|| = ", bn > 0 ? rn / bn : rn)
        catch err
            println("Residual computation skipped (error): ", err)
        end
    end
end

main()

