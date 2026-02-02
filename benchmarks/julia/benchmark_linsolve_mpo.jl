#!/usr/bin/env julia
#
# Benchmark: linsolve (MPO unknown) using KrylovKit GMRES + FastMPOContractions MPO×MPO apply.
#
# Solve for MPO x:
#   (a0*I + a1*A) * x = b
# where A(x) = A0 * x  (operator composition), implemented as MPO-MPO contraction.
#
# Notes:
# - ITensorMPS.jl's sweeping linsolve currently specializes on MPS, so here we benchmark
#   the KrylovKit linear solver directly on MPO objects.
# - MPO-MPO contraction uses FastMPOContractions.jl.
# - Fixed inputs once (reproducible with fixed RNG seed), warmup excluded, measured runs.
#

import Pkg

Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "external", "ITensorMPS.jl"))
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "external", "FastMPOContractions.jl"))
Pkg.instantiate()

using ITensors
using ITensorMPS
using FastMPOContractions
using KrylovKit
using InteractiveUtils
using LinearAlgebra
using Printf
using Random
using Statistics
using VectorInterface

struct MPOVec
    x::MPS
end

Base.copy(v::MPOVec) = MPOVec(copy(v.x))
Base.deepcopy(v::MPOVec) = MPOVec(deepcopy(v.x))
Base.:+(a::MPOVec, b::MPOVec) = MPOVec(a.x + b.x)
Base.:-(a::MPOVec, b::MPOVec) = MPOVec(a.x - b.x)
Base.:*(α::Number, v::MPOVec) = MPOVec(α * v.x)
Base.:*(v::MPOVec, α::Number) = MPOVec(v.x * α)
LinearAlgebra.norm(v::MPOVec) = norm(v.x)
LinearAlgebra.dot(a::MPOVec, b::MPOVec) = ITensors.inner(a.x, b.x)

VectorInterface.scalartype(::Type{MPOVec}) = Float64
VectorInterface.scalartype(::MPOVec) = Float64
VectorInterface.zerovector(v::MPOVec) = MPOVec(0.0 * v.x)
VectorInterface.inner(a::MPOVec, b::MPOVec) = LinearAlgebra.dot(a, b)
VectorInterface.scale(v::MPOVec, α::Number) = α * v
VectorInterface.scale!!(y::MPOVec, x::MPOVec, α::Number) = MPOVec(α * x.x)
VectorInterface.scale!!(x::MPOVec, α::Number) = MPOVec(α * x.x)

VectorInterface.add!!(y::MPOVec, x::MPOVec, α::Number, ::VectorInterface.One) =
    MPOVec(1.0 * y.x + α * x.x)

VectorInterface.add!!(y::MPOVec, x::MPOVec, α::Number, ::VectorInterface.Zero) =
    MPOVec(0.0 * y.x + α * x.x)

VectorInterface.add!!(y::MPOVec, x::MPOVec, α::Number, β::Number) =
    MPOVec(β * y.x + α * x.x)

function envinfo()
    println("=== Environment ===")
    InteractiveUtils.versioninfo()
    println("Threads.nthreads() = ", Threads.nthreads())
    println("BLAS.get_num_threads() = ", LinearAlgebra.BLAS.get_num_threads())
    println()
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

function make_ones_mpo(sites; linkdim)
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
        data = ones(Float64, dims...)
        w[i] = ITensor(data, inds...)
    end

    return MPO(w)
end

function check_square_mpo_constraint(; sites, A0, x0, b0)
    @assert length(sites) > 0
    @assert length(A0) == length(sites)
    @assert length(x0) == length(sites)
    @assert length(b0) == length(sites)

    sig(i) = (dim(i), string(tags(i)))
    site_sig = map(sig, sites)

    function check_operand(name, op)
        op_site_pairs = collect(siteinds(op))
        @assert length(op_site_pairs) == length(sites) "$(name) site set size != sites"
        for (i, pair) in enumerate(op_site_pairs)
            @assert length(pair) == 2 "$(name)[$i] should have 2 site indices (out/in)"
            pair_sigs = map(sig, pair)
            @assert all(==(site_sig[i]), pair_sigs) "$(name)[$i] site signature mismatch"
        end
    end

    check_operand("A0", A0)
    check_operand("x0", x0)
    check_operand("b0", b0)

    return nothing
end

function rel_residual_mpo(; applyA, b, x, a0, a1)
    Ax = applyA(x)
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

    # Linear system coefficients: (a0*I + a1*A)
    a0 = 0.0
    a1 = 1.0

    # MPO×MPO contraction settings (FastMPOContractions)
    mpo_apply_alg = "fit"
    mpo_apply_nsweeps = 1

    # GMRES / KrylovKit (direct)
    gmres_tol = 1e-6
    gmres_maxiter = 50
    krylovdim = 30

    seed = 1234
    n_runs = 10

    println("=== linsolve Benchmark (MPO unknown; Julia/KrylovKit + FastMPOContractions) ===")
    println("Problem: (a0*I + a1*A) * x = b, with x::MPO")
    println("A(x) = A0 * x (MPO×MPO contraction)")
    println("N = ", N)
    println("d = 2 (siteinds(\"S=1/2\", N))")
    println("chi (linkdim) = ", chi)
    println("maxdim = ", maxdim)
    println("cutoff = ", cutoff)
    println("rtol (tensor4all convention) = sqrt(cutoff) = ", rtol)
    println("MPO apply: alg = ", mpo_apply_alg, ", nsweeps = ", mpo_apply_nsweeps)
    println("GMRES: tol = ", gmres_tol, ", maxiter = ", gmres_maxiter, ", krylovdim = ", krylovdim)
    println("coefficients: a0 = ", a0, ", a1 = ", a1)
    println("seed = ", seed, " (used for random operator MPO A0)")
    println("n_runs = ", n_runs, " (excluding warmup)")
    println()

    envinfo()

    sites = siteinds("S=1/2", N)
    rngA = MersenneTwister(seed)

    A0 = make_random_mpo(sites; linkdim=chi, rng=rngA)
    x_true = make_ones_mpo(sites; linkdim=chi)

    applyA_mpo = x -> FastMPOContractions.apply(
        A0,
        x;
        alg=mpo_apply_alg,
        cutoff=cutoff,
        maxdim=maxdim,
        nsweeps=mpo_apply_nsweeps,
    )

    # b = A(x_true) in MPO space
    b0_mpo = applyA_mpo(x_true)
    # initial guess x^(0) = b (same as MPS benchmark)
    x00_mpo = deepcopy(b0_mpo)

    check_square_mpo_constraint(; sites=sites, A0=A0, x0=x00_mpo, b0=b0_mpo)

    # KrylovKit GMRES relies on VectorInterface; ITensorMPS implements that for MPS, not MPO.
    # Represent the unknown MPO as an MPS of MPO site-index pairs (i.e. `MPS(collect(mpo))`).
    b0 = MPOVec(MPS(collect(b0_mpo)))
    x00 = MPOVec(MPS(collect(x00_mpo)))

    applyA = v -> begin
        y_mpo = applyA_mpo(MPO(collect(v.x)))
        return MPOVec(MPS(collect(y_mpo)))
    end

    println("Warmup run (excluded from stats)...")
    t_warmup = @elapsed begin
        b = deepcopy(b0)
        x0 = deepcopy(x00)

        r0 = rel_residual_mpo(; applyA=applyA, b=b, x=x0, a0=a0, a1=a1)
        x, info = KrylovKit.linsolve(
            applyA,
            b,
            x0,
            a0,
            a1;
            tol=gmres_tol,
            maxiter=gmres_maxiter,
            krylovdim=krylovdim,
        )
        r1 = rel_residual_mpo(; applyA=applyA, b=b, x=x, a0=a0, a1=a1)
        println(@sprintf("Warmup residual (rel): %.3e -> %.3e", r0, r1))
        if hasproperty(info, :numops)
            println("Warmup GMRES ops: ", getproperty(info, :numops))
        end
    end
    println(@sprintf("Warmup completed in %.3f ms", 1000 * t_warmup))
    println()

    println("Measured runs...")
    times = Float64[]
    x_last = nothing
    for run in 1:n_runs
        t = @elapsed begin
            b = deepcopy(b0)
            x0 = deepcopy(x00)
            x, _info = KrylovKit.linsolve(
                applyA,
                b,
                x0,
                a0,
                a1;
                tol=gmres_tol,
                maxiter=gmres_maxiter,
                krylovdim=krylovdim,
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

    if x_last !== nothing && x_last isa MPOVec
        println()
        println("=== Residual (best-effort) ===")
        try
            rn = norm(a0 * x_last + a1 * applyA(x_last) - b0)
            bn = norm(b0)
            println("||r|| = ", rn)
            println("||r||/||b|| = ", bn > 0 ? rn / bn : rn)
        catch err
            println("Residual computation skipped (error): ", err)
        end
    end
end

main()
