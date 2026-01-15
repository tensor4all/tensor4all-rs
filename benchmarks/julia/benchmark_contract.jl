#!/usr/bin/env julia
# Benchmark: Contract two random MPOs with bond dimension 20 using zip-up method.
#
# This script creates two random MPOs with:
# - Length: 10 sites
# - Physical dimension: 2 per site (input and output)
# - Bond dimension: 20
#
# Then contracts them using zip-up method with cutoff=0 (no truncation) and maxdim=20.

using ITensors
using ITensorMPS
using Statistics

# Helper function to create random MPO with specified bond dimension
function makeRandomMPO(sites; chi::Int = 4)
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    l = [Index(chi, "Link,l=$n") for n in 1:(N - 1)]
    for n in 1:N
        s = sites[n]
        if n == 1
            v[n] = ITensor(l[n], s, s')
        elseif n == N
            v[n] = ITensor(l[n - 1], s, s')
        else
            v[n] = ITensor(l[n - 1], s, s', l[n])
        end
        randn!(v[n])
        normalize!(v[n])
    end
    return MPO(v, 0, N + 1)
end

function main()
    # Parameters
    length = 10
    phys_dim = 2
    bond_dim = 50
    max_rank = 50
    n_runs = 10  # Number of runs for averaging (excluding first compilation run)

    println("=== Random MPO Contraction Benchmark (Julia/ITensorMPS.jl) ===")
    println("Length: $length sites")
    println("Physical dimension: $phys_dim")
    println("Bond dimension: $bond_dim")
    println("Max rank: $max_rank")
    println("Number of runs: $n_runs (excluding first compilation run)")
    println()

    # Create site indices
    sites = [Index(phys_dim, "Site,n=$i") for i in 1:length]

    # Create first MPO: A[s_i, s'_i] (input s, output s')
    # MPO structure: -s'-A-s- (s' is input, s is output)
    # Create once and keep fixed for all measurements
    println("Creating first MPO (A) [fixed for all runs]...")
    mpo_a_original = makeRandomMPO(sites; chi=bond_dim)
    println("MPO A created. Max bond dim: $(maxlinkdim(mpo_a_original))")
    println()

    # Create second MPO: B[s'_i, s''_i] (input s', output s'')
    # We need to create MPO with primed site indices (s') as input
    # and new site indices (s'') as output
    # Create once and keep fixed for all measurements
    println("Creating second MPO (B) [fixed for all runs]...")
    sites_prime = [prime(s) for s in sites]  # s' indices (shared with A's output)
    mpo_b_original = makeRandomMPO(sites_prime; chi=bond_dim)
    println("MPO B created. Max bond dim: $(maxlinkdim(mpo_b_original))")
    println()

    # Contract options: zip-up with cutoff=0 (no truncation) and maxdim=20
    # Note: cutoff=0 means no truncation, but ITensorMPS.jl uses cutoff for SVD truncation
    # To disable truncation, we use a very small cutoff value
    println("Contracting MPOs using zip-up method...")
    println("Options: alg=zipup, maxdim=$max_rank, cutoff=0.0")
    println("Note: Each run copies MPOs and includes orthogonalization time")
    println()

    # First run (compilation/warmup - excluded from average)
    println("Warmup run (excluded from average)...")
    t_warmup = @elapsed begin
        # Copy MPOs for this run
        mpo_a = copy(mpo_a_original)
        mpo_b = copy(mpo_b_original)
        # Orthogonalize (included in timing)
        orthogonalize!(mpo_a, 1)
        orthogonalize!(mpo_b, 1)
        # Contract
        result_warmup = contract(mpo_a, mpo_b; alg="zipup", maxdim=max_rank, cutoff=0.0)
    end
    println("Warmup completed in $(t_warmup*1000)ms. Result max bond dim: $(maxlinkdim(result_warmup))")
    println()

    # Multiple runs for averaging
    println("Running $n_runs iterations for averaging...")
    times = Float64[]
    result = nothing
    for run in 1:n_runs
        t_elapsed = @elapsed begin
            # Copy MPOs for this run
            mpo_a = copy(mpo_a_original)
            mpo_b = copy(mpo_b_original)
            # Orthogonalize (included in timing)
            orthogonalize!(mpo_a, 1)
            orthogonalize!(mpo_b, 1)
            # Contract
            result = contract(mpo_a, mpo_b; alg="zipup", maxdim=max_rank, cutoff=0.0)
        end
        push!(times, t_elapsed)
        println("  Run $run: $(t_elapsed*1000)ms (max bond dim: $(maxlinkdim(result)))")
    end

    # Calculate statistics
    avg_time = mean(times)
    min_time = minimum(times)
    max_time = maximum(times)
    std_time = std(times)

    println()
    println("=== Results ===")
    println("Average time: $(avg_time*1000)ms")
    println("Min time: $(min_time*1000)ms")
    println("Max time: $(max_time*1000)ms")
    println("Std dev: $(std_time*1000)ms")
    if result !== nothing
        println("Final result max bond dimension: $(maxlinkdim(result))")
        println("Final result bond dimensions: $(linkdims(result))")
    end

    return nothing
end

main()
