#!/usr/bin/env julia

import Pkg

Pkg.activate(; temp=true)
Pkg.add(Pkg.PackageSpec(name="BenchmarkTools", uuid="6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"))
if haskey(ENV, "ACI_JL_PATH")
    Pkg.develop(path=ENV["ACI_JL_PATH"])
else
    Pkg.add(url="https://github.com/tensor4all/AlternatingCrossInterpolation.jl.git")
end

using BenchmarkTools
using Statistics
import AlternatingCrossInterpolation as ACI

const TCI = ACI.TCI

const N_SITES = 12
const LOCAL_DIM = 2
const N_INPUTS = 2
const TOLERANCE = 1e-10
const MAX_ITERS = 20
const SAMPLE_POINTS = 64

function parse_chis(args)
    chis = [2, 4, 8, 16]
    i = 1
    while i <= length(args)
        if args[i] == "--chis"
            i == length(args) && error("--chis requires a comma-separated value list")
            chis = parse.(Int, split(args[i + 1], ","))
            i += 2
        else
            error("unknown argument: $(args[i])")
        end
    end
    return chis
end

function link_dims(n_sites::Int, local_dim::Int, chi::Int)
    return [
        min(chi, local_dim ^ min(bond, n_sites - bond))
        for bond in 1:(n_sites - 1)
    ]
end

function core_value(input_index::Int, site::Int, physical::Int, left::Int, right::Int, left_dim::Int, right_dim::Int)
    input = input_index + 1.0
    site = site + 1.0
    physical = physical + 1.0
    left = left + 1.0
    right = right + 1.0
    left_coord = left / (left_dim + 1.0)
    right_coord = right / (right_dim + 1.0)
    phase = 0.173 * input * site +
        0.193 * physical +
        0.071 * left * right +
        0.109 * input * left +
        0.131 * site * right
    bond_mix = 0.29 * sin(phase) +
        0.23 * cos(0.157 * input * physical * right + 0.211 * site * left) +
        0.17 * (left_coord - right_coord) * physical
    site_value = 0.31 + bond_mix
    scale = (left_dim * right_dim) ^ 0.25
    return site_value / scale
end

function deterministic_tt(input_index::Int, n_sites::Int, local_dim::Int, chi::Int)
    links = link_dims(n_sites, local_dim, chi)
    tensors = Vector{Array{Float64,3}}(undef, n_sites)
    for site in 1:n_sites
        left_dim = site == 1 ? 1 : links[site - 1]
        right_dim = site <= length(links) ? links[site] : 1
        tensor = Array{Float64,3}(undef, left_dim, local_dim, right_dim)
        for right in 1:right_dim, physical in 1:local_dim, left in 1:left_dim
            tensor[left, physical, right] = core_value(
                input_index,
                site - 1,
                physical - 1,
                left - 1,
                right - 1,
                left_dim,
                right_dim,
            )
        end
        tensors[site] = tensor
    end
    return TCI.TensorTrain(tensors)
end

function deterministic_inputs(chi::Int)
    return [deterministic_tt(input_index, N_SITES, LOCAL_DIM, chi) for input_index in 0:(N_INPUTS - 1)]
end

function deterministic_initial_guess(chi::Int)
    return deterministic_tt(N_INPUTS, N_SITES, LOCAL_DIM, chi)
end

function sample_index(sample::Int, chi::Int)
    state = UInt64(sample + 1) * UInt64(0x9E3779B97F4A7C15) ⊻
        UInt64(chi + 17) * UInt64(0xBF58476D1CE4E5B9)
    index = Vector{Int}(undef, N_SITES)
    for site in 0:(N_SITES - 1)
        state = state ⊻ (state >> 30)
        state *= UInt64(0xBF58476D1CE4E5B9)
        state = state ⊻ (state >> 27)
        state *= UInt64(0x94D049BB133111EB)
        index[site + 1] = Int((state ⊻ UInt64(site)) % UInt64(LOCAL_DIM)) + 1
    end
    return index
end

function sampled_max_abs_error(inputs, output, chi::Int)
    max_error = 0.0
    for sample in 0:(SAMPLE_POINTS - 1)
        index = sample_index(sample, chi)
        expected = prod(input(index) for input in inputs)
        actual = output(index)
        max_error = max(max_error, abs(actual - expected))
    end
    return max_error
end

function run_aci(inputs, initial_guess)
    truncationparameters = ACI.TruncationParameters(typemax(Int), TOLERANCE, false)
    return ACI.elementwise(
        *,
        inputs;
        max_iters=MAX_ITERS,
        min_iters=2,
        truncationparameters=truncationparameters,
        initial_guess=initial_guess,
    )
end

function milliseconds(trial)
    times = trial.times ./ 1e6
    return (
        median=median(times),
        minimum=minimum(times),
        mean=mean(times),
        maximum=maximum(times),
    )
end

function main()
    chis = parse_chis(ARGS)
    println("impl,n_sites,local_dim,chi,tolerance,median_ms,min_ms,mean_ms,max_ms,output_max_chi,n_sweeps,final_error,sampled_max_abs_error")
    for chi in chis
        inputs = deterministic_inputs(chi)
        initial_guess = deterministic_initial_guess(chi)
        output, ranks, errors = run_aci(inputs, initial_guess)
        sampled_error = sampled_max_abs_error(inputs, output, chi)
        sampled_error < 1e-8 || error("sampled max abs error for chi=$chi was $sampled_error")

        trial = @benchmark run_aci($inputs, $initial_guess) samples = 10 seconds = 1 evals = 1
        stats = milliseconds(trial)
        final_error = isempty(errors) ? 0.0 : last(errors)
        println(join([
            "julia",
            N_SITES,
            LOCAL_DIM,
            chi,
            TOLERANCE,
            stats.median,
            stats.minimum,
            stats.mean,
            stats.maximum,
            TCI.rank(output),
            length(ranks),
            final_error,
            sampled_error,
        ], ","))
    end
end

main()
