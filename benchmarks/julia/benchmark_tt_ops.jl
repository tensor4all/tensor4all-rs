#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Statistics
using ITensors
using ITensorMPS

const DEFAULT_L = 32
const DEFAULT_D = 2
const DEFAULT_ZIPUP_L = 10
const DEFAULT_CHIS = [4, 8, 16, 32, 64]
const DEFAULT_WARMUP_SECONDS = 1.0
const DEFAULT_MEASUREMENT_SECONDS = 2.0
const DEFAULT_MIN_SAMPLES = 10

function usage()
    println("""
Usage: julia --project=benchmarks/julia benchmarks/julia/benchmark_tt_ops.jl [options]

Options:
  --L N                         MPS length (default: $(DEFAULT_L))
  --d N                         Physical dimension (default: $(DEFAULT_D))
  --zipup-L N                   MPO zipup length (default: $(DEFAULT_ZIPUP_L))
  --chis LIST                   Comma-separated bond dimensions (default: $(join(DEFAULT_CHIS, ",")))
  --warm-up-time SECONDS        Warm-up time after first JIT call (default: $(DEFAULT_WARMUP_SECONDS))
  --measurement-time SECONDS    Measurement time per case (default: $(DEFAULT_MEASUREMENT_SECONDS))
  --min-samples N               Minimum samples per case (default: $(DEFAULT_MIN_SAMPLES))
  --blas-threads N              Julia BLAS threads (default: 1)
  --help                        Show this help text
""")
end

function parse_args(args)
    opts = Dict{String, Any}(
        "L" => DEFAULT_L,
        "d" => DEFAULT_D,
        "zipup_L" => DEFAULT_ZIPUP_L,
        "chis" => copy(DEFAULT_CHIS),
        "warm_up_time" => DEFAULT_WARMUP_SECONDS,
        "measurement_time" => DEFAULT_MEASUREMENT_SECONDS,
        "min_samples" => DEFAULT_MIN_SAMPLES,
        "blas_threads" => 1,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            usage()
            exit(0)
        elseif arg == "--L"
            i += 1
            opts["L"] = parse(Int, args[i])
        elseif arg == "--d"
            i += 1
            opts["d"] = parse(Int, args[i])
        elseif arg == "--zipup-L"
            i += 1
            opts["zipup_L"] = parse(Int, args[i])
        elseif arg == "--chis"
            i += 1
            opts["chis"] = parse.(Int, split(args[i], ","))
        elseif arg == "--warm-up-time"
            i += 1
            opts["warm_up_time"] = parse(Float64, args[i])
        elseif arg == "--measurement-time"
            i += 1
            opts["measurement_time"] = parse(Float64, args[i])
        elseif arg == "--min-samples"
            i += 1
            opts["min_samples"] = parse(Int, args[i])
        elseif arg == "--blas-threads"
            i += 1
            opts["blas_threads"] = parse(Int, args[i])
        else
            error("unknown argument: $arg")
        end
        i += 1
    end

    return opts
end

function deterministic_value(idx::Int, seed::Int)::ComplexF64
    real = ((idx * 17 + seed * 13 + 3) % 97) / 97 - 0.5
    imag = ((idx * 29 + seed * 7 + 5) % 89) / 89 - 0.5
    return ComplexF64(real, imag)
end

function deterministic_itensor(inds::Tuple, seed::Int)::ITensor
    dims = map(dim, inds)
    data = Vector{ComplexF64}(undef, prod(dims))
    @inbounds for pos in eachindex(data)
        data[pos] = deterministic_value(pos - 1, seed)
    end
    return ITensor(reshape(data, dims...), inds...)
end

function deterministic_mps(sites::Vector{<:Index}, chi::Int, seed_offset::Int)::MPS
    nsites = length(sites)
    links = [Index(chi, "Link,mps,l=$n,seed=$seed_offset") for n in 1:(nsites - 1)]
    tensors = Vector{ITensor}(undef, nsites)

    for site in 1:nsites
        inds =
            if nsites == 1
                (sites[site],)
            elseif site == 1
                (sites[site], links[site])
            elseif site == nsites
                (links[site - 1], sites[site])
            else
                (links[site - 1], sites[site], links[site])
            end
        tensors[site] = deterministic_itensor(inds, seed_offset + site)
    end

    return MPS(tensors)
end

function deterministic_mpo(
    input_sites::Vector{<:Index},
    output_sites::Vector{<:Index},
    chi::Int,
    seed_offset::Int,
)::MPO
    nsites = length(input_sites)
    @assert length(output_sites) == nsites
    links = [Index(chi, "Link,mpo,l=$n,seed=$seed_offset") for n in 1:(nsites - 1)]
    tensors = Vector{ITensor}(undef, nsites)

    for site in 1:nsites
        inds =
            if nsites == 1
                (input_sites[site], output_sites[site])
            elseif site == 1
                (input_sites[site], output_sites[site], links[site])
            elseif site == nsites
                (links[site - 1], input_sites[site], output_sites[site])
            else
                (links[site - 1], input_sites[site], output_sites[site], links[site])
            end
        tensors[site] = deterministic_itensor(inds, seed_offset + site)
    end

    return MPO(tensors)
end

function run_for_seconds(f; warmup_seconds::Float64, measurement_seconds::Float64, min_samples::Int)
    sink = f()
    GC.gc()

    warmup_start = time_ns()
    while (time_ns() - warmup_start) / 1.0e9 < warmup_seconds
        sink = f()
    end
    GC.gc()

    times_ms = Float64[]
    measurement_start = time_ns()
    while (time_ns() - measurement_start) / 1.0e9 < measurement_seconds || length(times_ms) < min_samples
        start = time_ns()
        sink = f()
        push!(times_ms, (time_ns() - start) / 1.0e6)
    end

    return sink, times_ms
end

function print_result(case; params, times_ms, value, max_bond)
    @printf(
        "%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%d,%s\n",
        case,
        params,
        length(times_ms),
        minimum(times_ms),
        median(times_ms),
        mean(times_ms),
        maximum(times_ms),
        max_bond,
        repr(value),
    )
end

function main()
    opts = parse_args(ARGS)
    BLAS.set_num_threads(opts["blas_threads"])

    L = opts["L"]
    d = opts["d"]
    zipup_L = opts["zipup_L"]
    chis = opts["chis"]
    warmup_seconds = opts["warm_up_time"]
    measurement_seconds = opts["measurement_time"]
    min_samples = opts["min_samples"]

    println("ITensorMPS TensorTrain-level ops benchmark")
    println("  Julia:            $(VERSION)")
    println("  ITensors:         $(Base.pkgversion(ITensors))")
    println("  ITensorMPS:       $(Base.pkgversion(ITensorMPS))")
    println("  threads:          Julia=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads())")
    println("  L:                $L")
    println("  d:                $d")
    println("  zipup L:          $zipup_L")
    println("  chis:             $(join(chis, ","))")
    println("  warm-up time:     $warmup_seconds")
    println("  measurement time: $measurement_seconds")
    println("  min samples:      $min_samples")
    println()
    println("case,params,samples,min_ms,median_ms,mean_ms,max_ms,max_bond,value")

    for chi in chis
        sites = [Index(d, "Site,n=$site") for site in 1:L]
        bra = deterministic_mps(sites, chi, 0)
        ket = deterministic_mps(sites, chi, L)
        mps_params = "L_$(L)_chi_$(chi)_d_$(d)"

        inner_value, inner_times = run_for_seconds(
            () -> inner(bra, ket);
            warmup_seconds,
            measurement_seconds,
            min_samples,
        )
        print_result(
            "itensormps_inner_mps";
            params = mps_params,
            times_ms = inner_times,
            value = inner_value,
            max_bond = 0,
        )

        sum_mps, directsum_times = run_for_seconds(
            () -> +(bra, ket; alg = "directsum");
            warmup_seconds,
            measurement_seconds,
            min_samples,
        )
        print_result(
            "itensormps_directsum_mps";
            params = mps_params,
            times_ms = directsum_times,
            value = inner(sum_mps, sum_mps),
            max_bond = maxlinkdim(sum_mps),
        )

        zipup_sites_in = [Index(d, "ZipIn,n=$site") for site in 1:zipup_L]
        zipup_sites_mid = [Index(d, "ZipMid,n=$site") for site in 1:zipup_L]
        zipup_sites_out = [Index(d, "ZipOut,n=$site") for site in 1:zipup_L]
        mpo_a = deterministic_mpo(zipup_sites_in, zipup_sites_mid, chi, 2L)
        mpo_b = deterministic_mpo(zipup_sites_mid, zipup_sites_out, chi, 2L + zipup_L)
        orthogonalize!(mpo_a, 1)
        orthogonalize!(mpo_b, 1)
        zipup_params = "L_$(zipup_L)_chi_$(chi)_d_$(d)_maxdim_$(chi)"

        zipup_result, zipup_times = run_for_seconds(
            () -> contract(mpo_a, mpo_b; alg = "zipup", maxdim = chi, cutoff = 0.0);
            warmup_seconds,
            measurement_seconds,
            min_samples,
        )
        print_result(
            "itensormps_zipup_mpo_prepared";
            params = zipup_params,
            times_ms = zipup_times,
            value = inner(zipup_result, zipup_result),
            max_bond = maxlinkdim(zipup_result),
        )
    end
end

main()
