# Benchmark isolated ITensors.ProjMPO local apply calls.
#
# Run:
#   julia --project=benchmarks/julia benchmarks/julia/benchmark_projected_apply.jl
#
# Optional args:
#   julia --project=benchmarks/julia benchmarks/julia/benchmark_projected_apply.jl <N> <state_bond_dim> <operator_bond_dim> <repeats> <step_index>
#
# For a one-thread comparison with the Rust example:
#   BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_projected_apply.jl 38 32 32 3 0

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using ITensors
using LinearAlgebra
using Printf
using Random

ITensors.disable_warn_order()

function parse_positive_int_arg(args::Vector{String}, index::Int, default::Int, name::String)::Int
    value = index <= length(args) ? parse(Int, args[index]) : default
    value > 0 || error("$name must be greater than zero")
    return value
end

function parse_nonnegative_int_arg(args::Vector{String}, index::Int, default::Int, name::String)::Int
    value = index <= length(args) ? parse(Int, args[index]) : default
    value >= 0 || error("$name must be nonnegative")
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

function two_site_sweep_positions(nsites::Int, center_pos::Int)::Vector{Int}
    positions = collect(center_pos:(nsites - 1))
    append!(positions, collect((center_pos - 1):-1:1))
    return positions
end

function elapsed_seconds(f)::Tuple{Float64, Any}
    start = time_ns()
    result = f()
    return ((time_ns() - start) / 1.0e9, result)
end

function summarize(label::String, times::Vector{Float64})
    mean = sum(times) / length(times)
    min_time = minimum(times)
    max_time = maximum(times)
    @printf(
        "%s: mean=%.3f ms min=%.3f ms max=%.3f ms n=%d\n",
        label,
        mean * 1000.0,
        min_time * 1000.0,
        max_time * 1000.0,
        length(times),
    )
end

function main(args::Vector{String})
    nsites = parse_positive_int_arg(args, 1, 38, "N")
    state_bond_dim = parse_positive_int_arg(args, 2, 8, "state_bond_dim")
    operator_bond_dim = parse_positive_int_arg(args, 3, 8, "operator_bond_dim")
    repeats = parse_positive_int_arg(args, 4, 20, "repeats")
    step_index = parse_nonnegative_int_arg(args, 5, 0, "step_index")

    nsites >= 2 || error("N must be at least 2 for a two-site local step")
    maybe_set_blas_threads_from_env!()

    phys_dim = 2
    seed = 20260518
    rng = MersenneTwister(seed)

    acted_sites = [Index(phys_dim, "s=$site") for site in 1:nsites]
    spectator_sites = [Index(phys_dim, "q=$site") for site in 1:nsites]

    psi = make_state_mps(rng, nsites, state_bond_dim, acted_sites, spectator_sites)
    H = make_operator_mpo(rng, nsites, operator_bond_dim, acted_sites, spectator_sites)

    center_pos = nsites ÷ 2 + 1
    positions = two_site_sweep_positions(nsites, center_pos)
    local_pos = positions[mod(step_index, length(positions)) + 1]
    phi = psi[local_pos] * psi[local_pos + 1]

    # Compile the relevant ITensor/ProjMPO code before taking timings.
    warmup_projected = ITensors.ProjMPO(H)
    position!(warmup_projected, psi, local_pos)
    warmup_out = warmup_projected(phi)
    GC.@preserve warmup_out nothing
    GC.gc()

    println("=== ProjMPO local apply benchmark ===")
    println("N = $nsites")
    println("phys_dim = $phys_dim")
    println("state_bond_dim = $state_bond_dim")
    println("operator_bond_dim = $operator_bond_dim")
    println("repeats = $repeats")
    println("threads = $(Threads.nthreads())")
    println("blas_threads = $(BLAS.get_num_threads())")
    println("center_pos = $center_pos")
    println("step_index = $(mod(step_index, length(positions)))")
    println("local_sites = ($local_pos, $(local_pos + 1))")
    println("local_dims = $(dim.(inds(phi)))")
    println()

    projected_ref = Ref{Any}()
    cold_time, cold_out = elapsed_seconds() do
        projected = ITensors.ProjMPO(H)
        position!(projected, psi, local_pos)
        out = projected(phi)
        projected_ref[] = projected
        return out
    end
    @printf(
        "cold apply (environment build + one apply): %.3f ms, output_rank=%d\n",
        cold_time * 1000.0,
        order(cold_out),
    )

    projected = projected_ref[]
    warm_times = Float64[]
    sizehint!(warm_times, repeats)
    warm_order_sum = 0
    for _ in 1:repeats
        t, out = elapsed_seconds() do
            return projected(phi)
        end
        warm_order_sum += order(out)
        push!(warm_times, t)
    end
    summarize("warm apply (environment cache hot)", warm_times)

    cold_times = Float64[]
    sizehint!(cold_times, repeats)
    cold_order_sum = 0
    for _ in 1:repeats
        t, out = elapsed_seconds() do
            projected_cold = ITensors.ProjMPO(H)
            position!(projected_cold, psi, local_pos)
            return projected_cold(phi)
        end
        cold_order_sum += order(out)
        push!(cold_times, t)
    end
    summarize("cold apply repeated (fresh environment cache)", cold_times)

    GC.@preserve warm_order_sum cold_order_sum nothing
    return nothing
end

main(ARGS)
