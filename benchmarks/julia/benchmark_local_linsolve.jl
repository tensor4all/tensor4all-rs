# Benchmark prepared local linsolve using ITensorTDVP.linsolve.
#
# Run:
#   BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_local_linsolve.jl
#
# Optional args:
#   BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_local_linsolve.jl <N> <state_bond_dim> <operator_bond_dim> <nsweeps> <krylov_maxiter> <krylov_dim>

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using ITensors
using ITensorTDVP
using KrylovKit
using LinearAlgebra
using Printf
using Random

ITensors.disable_warn_order()

mutable struct LocalSolveStats
    local_updates::Int
    krylov_iterations::Int
    krylov_ops::Int
    rhs_time::Float64
    gmres_time::Float64
    apply_time::Float64
end

LocalSolveStats() = LocalSolveStats(0, 0, 0, 0.0, 0.0, 0.0)

function parse_positive_int_arg(args::Vector{String}, index::Int, default::Int, name::String)::Int
    value = index <= length(args) ? parse(Int, args[index]) : default
    value > 0 || error("$name must be greater than zero")
    return value
end

function maybe_set_blas_threads_from_env!()
    haskey(ENV, "BLAS_NUM_THREADS") || return
    nthreads = parse(Int, ENV["BLAS_NUM_THREADS"])
    nthreads > 0 || error("BLAS_NUM_THREADS must be greater than zero")
    BLAS.set_num_threads(nthreads)
end

function state_indices(
    site::Int,
    nsites::Int,
    acted_sites::Vector{Index{Int64}},
    spectator_sites::Vector{Index{Int64}},
    state_links::Vector{Index{Int64}},
)
    sites = Index{Int64}[spectator_sites[site], acted_sites[site]]
    if nsites == 1
        return sites
    elseif site == 1
        return vcat(sites, Index{Int64}[state_links[site]])
    elseif site == nsites
        return vcat(Index{Int64}[state_links[site - 1]], sites)
    else
        return vcat(Index{Int64}[state_links[site - 1]], sites, Index{Int64}[state_links[site]])
    end
end

function operator_indices(site::Int, nsites::Int, operator_links::Vector{Index{Int64}})
    if nsites == 1
        return Index{Int64}[]
    elseif site == 1
        return Index{Int64}[operator_links[site]]
    elseif site == nsites
        return Index{Int64}[operator_links[site - 1]]
    else
        return Index{Int64}[operator_links[site - 1], operator_links[site]]
    end
end

function make_state_mps(
    rng::AbstractRNG,
    nsites::Int,
    state_bond_dim::Int,
    acted_sites::Vector{Index{Int64}},
    spectator_sites::Vector{Index{Int64}},
)::MPS
    state_links = [Index(state_bond_dim, "Link,psi,l=$site") for site in 1:(nsites - 1)]
    tensors = Vector{ITensor}(undef, nsites)
    for site in 1:nsites
        indices = state_indices(site, nsites, acted_sites, spectator_sites, state_links)
        tensors[site] = random_itensor(rng, indices...)
    end
    return MPS(tensors)
end

function make_operator_mpo(
    rng::AbstractRNG,
    nsites::Int,
    operator_bond_dim::Int,
    acted_sites::Vector{Index{Int64}},
    spectator_sites::Vector{Index{Int64}},
)::MPO
    operator_links = [Index(operator_bond_dim, "Link,H,l=$site") for site in 1:(nsites - 1)]
    tensors = Vector{ITensor}(undef, nsites)
    for site in 1:nsites
        core_indices = vcat(
            operator_indices(site, nsites, operator_links),
            Index{Int64}[acted_sites[site], prime(acted_sites[site])],
        )
        core = random_itensor(rng, core_indices...)
        spectator_identity = delta(spectator_sites[site], prime(spectator_sites[site]))
        tensors[site] = core * spectator_identity
    end
    return MPO(tensors)
end

function elapsed_seconds(f)::Tuple{Float64, Any}
    start = time_ns()
    result = f()
    return ((time_ns() - start) / 1.0e9, result)
end

function timed_linsolve_updater(
    problem,
    init;
    internal_kwargs,
    coefficients,
    stats::LocalSolveStats,
    kwargs...,
)
    stats.local_updates += 1
    operator = ITensorTDVP.operator(problem)

    rhs_time, rhs = elapsed_seconds() do
        return ITensorTDVP.constant_term(problem)
    end
    stats.rhs_time += rhs_time

    function timed_operator(x)
        apply_time, y = elapsed_seconds() do
            return operator(x)
        end
        stats.apply_time += apply_time
        return y
    end

    gmres_time, solve_result = elapsed_seconds() do
        return KrylovKit.linsolve(
            timed_operator,
            rhs,
            init,
            coefficients[1],
            coefficients[2];
            kwargs...,
        )
    end
    stats.gmres_time += gmres_time
    x, info = solve_result
    stats.krylov_iterations += info.numiter
    stats.krylov_ops += info.numops
    return x, (; info)
end

function run_prepared_solve(H, rhs, init; nsweeps, cutoff, maxdim, a0, a1, tol, maxiter, krylovdim)
    stats = LocalSolveStats()
    solve_time, solution = elapsed_seconds() do
        return ITensorTDVP.linsolve(
            H,
            rhs,
            init,
            a0,
            a1;
            maxdim,
            cutoff,
            nsweeps,
            nsite=2,
            reverse_step=false,
            outputlevel=0,
            updater=timed_linsolve_updater,
            updater_kwargs=(;
                stats,
                ishermitian=false,
                tol,
                maxiter,
                krylovdim,
            ),
        )
    end
    return solve_time, solution, stats
end

function main(args::Vector{String})
    nsites = parse_positive_int_arg(args, 1, 38, "N")
    state_bond_dim = parse_positive_int_arg(args, 2, 8, "state_bond_dim")
    operator_bond_dim = parse_positive_int_arg(args, 3, 8, "operator_bond_dim")
    nsweeps = parse_positive_int_arg(args, 4, 1, "nsweeps")
    maxiter = parse_positive_int_arg(args, 5, 10, "krylov_maxiter")
    krylovdim = parse_positive_int_arg(args, 6, 30, "krylov_dim")

    nsites >= 2 || error("N must be at least 2 for a two-site local solve")
    maybe_set_blas_threads_from_env!()

    phys_dim = 2
    seed = 20260518
    cutoff = 0.0
    maxdim = state_bond_dim
    a0 = 1.0
    a1 = 0.01
    tol = 1.0e-30

    setup_time, prepared = elapsed_seconds() do
        rng = MersenneTwister(seed)
        acted_sites = [Index(phys_dim, "s=$site") for site in 1:nsites]
        spectator_sites = [Index(phys_dim, "q=$site") for site in 1:nsites]
        state = make_state_mps(rng, nsites, state_bond_dim, acted_sites, spectator_sites)
        operator = make_operator_mpo(
            rng,
            nsites,
            operator_bond_dim,
            acted_sites,
            spectator_sites,
        )
        return (; operator, rhs=deepcopy(state), init=deepcopy(state))
    end
    H = prepared.operator
    rhs = prepared.rhs
    init = prepared.init

    # Compile the relevant local solve path outside the reported solve timing.
    run_prepared_solve(
        H,
        rhs,
        init;
        nsweeps=1,
        cutoff,
        maxdim,
        a0,
        a1,
        tol,
        maxiter,
        krylovdim,
    )
    GC.gc()

    solve_time, solution, stats = run_prepared_solve(
        H,
        rhs,
        init;
        nsweeps,
        cutoff,
        maxdim,
        a0,
        a1,
        tol,
        maxiter,
        krylovdim,
    )

    println("=== Prepared local linsolve benchmark (Julia/ITensorTDVP) ===")
    println("N = $nsites")
    println("phys_dim = $phys_dim")
    println("state_bond_dim = $state_bond_dim")
    println("operator_bond_dim = $operator_bond_dim")
    println("nsweeps = $nsweeps")
    println("krylov_maxiter = $maxiter")
    println("krylov_dim = $krylovdim")
    @printf("krylov_tol = %.1e\n", tol)
    println("coefficients = ($a0, $a1)")
    println("threads = $(Threads.nthreads())")
    println("blas_threads = $(BLAS.get_num_threads())")
    println()

    println("--- Prepared solve ---")
    @printf("setup excluded from solve: %.3f ms\n", setup_time * 1000.0)
    @printf("solve total: %.3f ms\n", solve_time * 1000.0)
    println("local_updates = $(stats.local_updates)")
    println("krylov_iterations = $(stats.krylov_iterations)")
    println("krylov_ops = $(stats.krylov_ops)")
    @printf("rhs projection inside updates: %.3f ms\n", stats.rhs_time * 1000.0)
    @printf("local GMRES total: %.3f ms\n", stats.gmres_time * 1000.0)
    @printf("projected apply inside GMRES: %.3f ms\n", stats.apply_time * 1000.0)
    @printf(
        "replacebond/factorization/orthogonalization overhead: %.3f ms\n",
        max(0.0, solve_time - stats.gmres_time - stats.rhs_time) * 1000.0,
    )
    println("solution max bond dim = $(maxlinkdim(solution))")
end

main(ARGS)
