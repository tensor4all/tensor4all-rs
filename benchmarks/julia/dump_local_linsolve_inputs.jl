# Dump prepared local linsolve inputs in ITensorMPS-compatible HDF5.
#
# Run:
#   BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/dump_local_linsolve_inputs.jl
#
# Optional args:
#   BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/dump_local_linsolve_inputs.jl <output.h5> <N> <state_bond_dim> <operator_bond_dim>

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using HDF5
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

function default_output_path(nsites::Int, state_bond_dim::Int, operator_bond_dim::Int)::String
    root = normpath(joinpath(@__DIR__, "..", "results"))
    return joinpath(
        root,
        "local_linsolve_inputs_N$(nsites)_b$(state_bond_dim)_o$(operator_bond_dim).h5",
    )
end

function summarize_mps(name::String, psi::MPS)
    site_counts = [length(inds(psi[i])) for i in 1:length(psi)]
    println("$name.length = $(length(psi))")
    println("$name.maxlinkdim = $(maxlinkdim(psi))")
    println("$name.tensor_index_counts = $(site_counts)")
    if length(psi) > 0
        println("$name.first_inds = $(inds(psi[1]))")
        println("$name.last_inds = $(inds(psi[end]))")
    end
end

function main(args::Vector{String})
    nsites = parse_positive_int_arg(args, 2, 38, "N")
    state_bond_dim = parse_positive_int_arg(args, 3, 32, "state_bond_dim")
    operator_bond_dim = parse_positive_int_arg(args, 4, 32, "operator_bond_dim")
    nsites >= 2 || error("N must be at least 2")
    maybe_set_blas_threads_from_env!()

    output_path = length(args) >= 1 ? args[1] : default_output_path(nsites, state_bond_dim, operator_bond_dim)
    mkpath(dirname(output_path))

    phys_dim = 2
    seed = 20260518
    rng = MersenneTwister(seed)
    acted_sites = [Index(phys_dim, "s=$site") for site in 1:nsites]
    spectator_sites = [Index(phys_dim, "q=$site") for site in 1:nsites]
    rhs = make_state_mps(rng, nsites, state_bond_dim, acted_sites, spectator_sites)
    init = deepcopy(rhs)
    operator = make_operator_mpo(rng, nsites, operator_bond_dim, acted_sites, spectator_sites)
    operator_as_mps = MPS([operator[i] for i in 1:length(operator)])

    h5open(output_path, "w") do file
        write(file, "operator_as_mps", operator_as_mps)
        write(file, "rhs", rhs)
        write(file, "init", init)

        params = create_group(file, "params")
        write(params, "N", Int64(nsites))
        write(params, "phys_dim", Int64(phys_dim))
        write(params, "state_bond_dim", Int64(state_bond_dim))
        write(params, "operator_bond_dim", Int64(operator_bond_dim))
        write(params, "seed", Int64(seed))
        write(params, "format_note", "operator_as_mps stores Julia MPO site tensors in ITensorMPS MPS schema")
    end

    println("=== Dumped local linsolve inputs (Julia/HDF5) ===")
    println("output_path = $output_path")
    println("N = $nsites")
    println("phys_dim = $phys_dim")
    println("state_bond_dim = $state_bond_dim")
    println("operator_bond_dim = $operator_bond_dim")
    summarize_mps("operator_as_mps", operator_as_mps)
    summarize_mps("rhs", rhs)
    summarize_mps("init", init)
end

main(ARGS)
