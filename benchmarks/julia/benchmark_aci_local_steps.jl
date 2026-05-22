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

const N_SITES = 12
const LOCAL_DIM = 2
const N_INPUTS = 2
const TOLERANCE = 1e-10
const MAX_ITERS = 20
const MIN_ITERS = 2

mutable struct StepTiming
    setup_ns::Int
    setup_shape_ns::Int
    setup_dims_ns::Int
    setup_left_factor_ns::Int
    setup_right_factor_ns::Int
    input_values_ns::Int
    operator_ns::Int
    matrix_luci_ns::Int
    core_update_ns::Int
    frame_update_ns::Int
    updates::Int
    sweeps::Int
    final_rank::Int
    final_error::Float64
end

StepTiming() = StepTiming(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

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

function deterministic_tt(input_index::Int, chi::Int)
    links = link_dims(N_SITES, LOCAL_DIM, chi)
    tensors = Vector{Array{Float64,3}}(undef, N_SITES)
    for site in 1:N_SITES
        left_dim = site == 1 ? 1 : links[site - 1]
        right_dim = site <= length(links) ? links[site] : 1
        tensor = Array{Float64,3}(undef, left_dim, LOCAL_DIM, right_dim)
        for right in 1:right_dim, physical in 1:LOCAL_DIM, left in 1:left_dim
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
    return [deterministic_tt(input_index, chi) for input_index in 0:(N_INPUTS - 1)]
end

function timed_pitensor(problem, inputindex::Integer, bondindex::Integer, timing::StepTiming)
    left = problem.leftframes[inputindex, bondindex-1]
    Tleft = problem.inputs[inputindex][bondindex]
    Tright = problem.inputs[inputindex][bondindex+1]
    right = problem.rightframes[inputindex, bondindex+2]

    t = time_ns()
    L = ACI.contract(left, [2], Tleft, [1])
    elapsed = time_ns() - t
    timing.setup_left_factor_ns += elapsed
    timing.setup_ns += elapsed

    t = time_ns()
    R = ACI.contract(Tright, [3], right, [1])
    elapsed = time_ns() - t
    timing.setup_right_factor_ns += elapsed
    timing.setup_ns += elapsed

    t = time_ns()
    Π = ACI.contract(L, [3], R, [1])
    timing.input_values_ns += time_ns() - t
    return Π
end

function timed_localupdate!(op, problem, bondindex::Integer; leftorthogonal::Bool, truncationparameters, timing::StepTiming)
    Πs = [timed_pitensor(problem, k, bondindex, timing) for k in ACI.eachinputindex(problem)]

    t = time_ns()
    Π = op.(Πs...)
    timing.operator_ns += time_ns() - t

    t = time_ns()
    luci = TCI.MatrixLUCI(
        reshape(Π, prod(size(Π)[1:2]), prod(size(Π)[3:4]));
        leftorthogonal=leftorthogonal,
        maxrank=truncationparameters.maxbonddimension,
        abstol=truncationparameters.tolerance
    )
    timing.matrix_luci_ns += time_ns() - t

    t = time_ns()
    leftfactor = TCI.left(luci)
    rightfactor = TCI.right(luci)
    timing.matrix_luci_ns += time_ns() - t

    t = time_ns()
    problem.solution.sitetensors[bondindex] = reshape(leftfactor, size(Π, 1), size(Π, 2), :)
    problem.solution.sitetensors[bondindex+1] = reshape(rightfactor, :, size(Π, 3), size(Π, 4))
    timing.core_update_ns += time_ns() - t

    t = time_ns()
    if leftorthogonal
        ACI.updateleftframes!(problem, bondindex, TCI.rowindices(luci))
    else
        ACI.updaterightframes!(problem, bondindex + 1, TCI.colindices(luci))
    end
    problem.pivoterrors[bondindex] = TCI.lastpivoterror(luci)
    timing.frame_update_ns += time_ns() - t
    timing.updates += 1
    nothing
end

function timed_run(chi::Int)
    inputs = deterministic_inputs(chi)
    initial_guess = deterministic_tt(N_INPUTS, chi)
    truncationparameters = ACI.TruncationParameters(typemax(Int), TOLERANCE, false)
    problem = ACI.ElementwiseProblem(inputs, initial_guess)
    timing = StepTiming()
    ranks = Int[]
    errors = Float64[]

    for iteration in 1:MAX_ITERS
        forward = isodd(iteration)
        for bondindex in ACI.sweep(collect(ACI.eachbondindex(problem)); forward)
            timed_localupdate!(
                *,
                problem,
                bondindex;
                leftorthogonal=forward,
                truncationparameters,
                timing,
            )
        end

        push!(ranks, TCI.rank(problem.solution))
        push!(errors, maximum(problem.pivoterrors))
        timing.sweeps += 1
        timing.final_rank = last(ranks)
        timing.final_error = last(errors)

        if iteration >= MIN_ITERS &&
           errors[iteration] <= truncationparameters.tolerance &&
           !any(last(ranks, MIN_ITERS) .> ranks[iteration-MIN_ITERS+1])
            break
        end
    end
    return timing
end

ns_to_ms(ns::Int) = ns / 1e6
setup_other_ns(run::StepTiming) = max(
    0,
    run.setup_ns - run.setup_shape_ns - run.setup_dims_ns - run.setup_left_factor_ns - run.setup_right_factor_ns,
)

function main()
    repeats = parse(Int, get(ENV, "T4A_STEP_TIMING_REPEATS", "10"))
    chis = parse_chis(ARGS)
    println("impl,chi,repeats,n_sweeps,n_updates,setup_ms,setup_shape_ms,setup_dims_ms,setup_left_factor_ms,setup_right_factor_ms,setup_other_ms,input_values_ms,operator_ms,matrix_luci_ms,core_update_ms,frame_update_ms,total_ms,final_rank,final_error")
    for chi in chis
        runs = [timed_run(chi) for _ in 1:repeats]
        total_ms = [
            ns_to_ms(run.setup_ns + run.input_values_ns + run.operator_ns + run.matrix_luci_ns + run.core_update_ns + run.frame_update_ns)
            for run in runs
        ]
        first = runs[1]
        println(join([
            "julia",
            chi,
            repeats,
            first.sweeps,
            first.updates,
            median(ns_to_ms.(getfield.(runs, :setup_ns))),
            median(ns_to_ms.(getfield.(runs, :setup_shape_ns))),
            median(ns_to_ms.(getfield.(runs, :setup_dims_ns))),
            median(ns_to_ms.(getfield.(runs, :setup_left_factor_ns))),
            median(ns_to_ms.(getfield.(runs, :setup_right_factor_ns))),
            median(ns_to_ms.(setup_other_ns.(runs))),
            median(ns_to_ms.(getfield.(runs, :input_values_ns))),
            median(ns_to_ms.(getfield.(runs, :operator_ns))),
            median(ns_to_ms.(getfield.(runs, :matrix_luci_ns))),
            median(ns_to_ms.(getfield.(runs, :core_update_ns))),
            median(ns_to_ms.(getfield.(runs, :frame_update_ns))),
            median(total_ms),
            first.final_rank,
            first.final_error,
        ], ","))
    end
end

main()
