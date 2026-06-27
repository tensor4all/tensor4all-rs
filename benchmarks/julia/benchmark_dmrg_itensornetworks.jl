# Two-site DMRG benchmark for ITensorNetworks.jl.
#
# Prefer running against a local ITensorNetworks.jl checkout in a temporary
# benchmark project so the upstream checkout remains clean:
#   export T4A_ITENSORNETWORKS_PATH=/home/shinaoka/tensor4all/ITensor/ITensorNetworks.jl
#   export T4A_ITN_BENCH_PROJECT=/tmp/t4a-itensornetworks-bench
#   julia --project=$T4A_ITN_BENCH_PROJECT -e 'using Pkg; Pkg.develop(path=ENV["T4A_ITENSORNETWORKS_PATH"]); Pkg.add(["Graphs", "ITensors", "TensorOperations"]); Pkg.instantiate()'
#   BLAS_NUM_THREADS=1 julia --project=$T4A_ITN_BENCH_PROJECT \
#     benchmarks/julia/benchmark_dmrg_itensornetworks.jl 8 4 3
#
# Optional args:
#   julia benchmark_dmrg_itensornetworks.jl <n_sites> <nsweeps> <repeats>

using Graphs: SimpleGraph, add_edge!, edges, src, dst
using ITensorNetworks: OpSum, dmrg, random_tensornetwork, siteinds, ttn
using ITensors: disable_warn_order
using LinearAlgebra: Symmetric, eigvals
using Random: MersenneTwister
using Statistics: mean
using TensorOperations

disable_warn_order()

function make_graph(kind::Symbol, n_sites::Int)
  g = SimpleGraph(n_sites)
  if kind == :chain
    for i in 1:(n_sites - 1)
      add_edge!(g, i, i + 1)
    end
  elseif kind == :star
    for i in 2:n_sites
      add_edge!(g, 1, i)
    end
  else
    error("unknown topology kind $kind")
  end
  return g
end

function heisenberg_opsum(g::SimpleGraph)
  os = OpSum()
  for edge in edges(g)
    i = src(edge)
    j = dst(edge)
    os += 1.0, "X", i, "X", j
    os += 1.0, "Y", i, "Y", j
    os += 1.0, "Z", i, "Z", j
  end
  return os
end

root_vertex(kind::Symbol) = kind == :star ? 2 : 1

function dense_heisenberg_exact(g::SimpleGraph, n_sites::Int)
  n_sites <= 10 || error("dense exact benchmark reference is capped at n_sites <= 10")
  dim = 1 << n_sites
  h = zeros(Float64, dim, dim)
  for input_state in 0:(dim - 1)
    bits = [((input_state >> site) & 1) for site in 0:(n_sites - 1)]
    for edge in edges(g)
      left = src(edge) - 1
      right = dst(edge) - 1
      z_left = bits[left + 1] == 0 ? 1.0 : -1.0
      z_right = bits[right + 1] == 0 ? 1.0 : -1.0
      h[input_state + 1, input_state + 1] += z_left * z_right

      flipped = xor(input_state, (1 << left), (1 << right))
      yy_coeff = bits[left + 1] == bits[right + 1] ? -1.0 : 1.0
      h[flipped + 1, input_state + 1] += 1.0 + yy_coeff
    end
  end
  return minimum(eigvals(Symmetric(h)))
end

function summarize(times_ns)
  seconds = times_ns ./ 1.0e9
  return mean(seconds), minimum(seconds), maximum(seconds)
end

function make_initial_state(sites, repeat_seed::Integer)
  rng = MersenneTwister(repeat_seed)
  return ttn(random_tensornetwork(rng, sites; link_space = 1))
end

function run_dmrg_once(H, psi0, root::Int, nsweeps::Int)
  return dmrg(
    H,
    psi0;
    nsweeps,
    nsites = 2,
    root_vertex = root,
    factorize_kwargs = (; cutoff = 1.0e-12, maxdim = 32),
    eigsolve_solver_kwargs = (; tol = 1.0e-12, krylovdim = 16, maxiter = 4),
  )
end

function run_case(kind::Symbol, n_sites::Int, nsweeps::Int, repeats::Int)
  g = make_graph(kind, n_sites)
  sites = siteinds("S=1/2", g)
  root = root_vertex(kind)
  H = ttn(heisenberg_opsum(g), sites; root_vertex = root, cutoff = 1.0e-12)
  exact_energy = dense_heisenberg_exact(g, n_sites)

  # Exclude Julia compilation and ITensorNetworks.jl first-call setup from the
  # timed region. The timed loop below measures DMRG execution only.
  run_dmrg_once(H, make_initial_state(sites, 0x4a16), root, nsweeps)
  GC.gc()

  times = UInt64[]
  energy = NaN
  for repeat in 1:repeats
    psi0 = make_initial_state(sites, 0x4a17 + repeat)
    start = time_ns()
    energy, _ = run_dmrg_once(H, psi0, root, nsweeps)
    push!(times, time_ns() - start)
  end

  mean_s, min_s, max_s = summarize(times)
  println(
    "case=$(kind) n=$(n_sites) sweeps=$(nsweeps) warmups=1 repeats=$(repeats) " *
    "energy=$(round(real(energy), digits = 12)) exact=$(round(exact_energy, digits = 12)) " *
    "abs_error=$(abs(real(energy) - exact_energy)) " *
    "mean_ms=$(round(mean_s * 1000, digits = 3)) " *
    "min_ms=$(round(min_s * 1000, digits = 3)) " *
    "max_ms=$(round(max_s * 1000, digits = 3))"
  )
  return nothing
end

function main()
  n_sites = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
  nsweeps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
  repeats = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
  n_sites >= 2 || error("n_sites must be at least 2")
  repeats >= 1 || error("repeats must be at least 1")

  run_case(:chain, n_sites, nsweeps, repeats)
  run_case(:star, n_sites, nsweeps, repeats)
  return nothing
end

main()
