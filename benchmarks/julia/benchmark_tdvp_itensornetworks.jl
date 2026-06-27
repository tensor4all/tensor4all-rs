# Two-site TDVP benchmark for ITensorNetworks.jl.
#
# Prefer running against a local ITensorNetworks.jl checkout in a temporary
# benchmark project so the upstream checkout remains clean:
#   export T4A_ITENSORNETWORKS_PATH=/home/shinaoka/tensor4all/ITensor/ITensorNetworks.jl
#   export T4A_ITN_BENCH_PROJECT=/tmp/t4a-itensornetworks-bench
#   julia --project=$T4A_ITN_BENCH_PROJECT -e 'using Pkg; Pkg.develop(path=ENV["T4A_ITENSORNETWORKS_PATH"]); Pkg.add(["Graphs", "ITensors", "TensorOperations", "KrylovKit"]); Pkg.instantiate()'
#   BLAS_NUM_THREADS=1 julia --project=$T4A_ITN_BENCH_PROJECT \
#     benchmarks/julia/benchmark_tdvp_itensornetworks.jl 8 4 3 0.02
#
# Optional args:
#   julia benchmark_tdvp_itensornetworks.jl <n_sites> <time_steps> <repeats> <dt>

using Graphs: SimpleGraph, add_edge!, edges, src, dst
import ITensorNetworks
using ITensorNetworks:
  OpSum, contract, exponentiate_solver, maxlinkdim, siteinds, time_evolve, ttn
using ITensors: disable_warn_order
using LinearAlgebra: Hermitian, eigen, norm
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

initial_bit(site::Int) = isodd(site) ? 0 : 1

function make_initial_state(sites, n_sites::Int)
  state = Dict{Int, String}()
  for site in 1:n_sites
    state[site] = initial_bit(site) == 0 ? "Up" : "Dn"
  end
  return ttn(state, sites)
end

function initial_vector(n_sites::Int)
  dim = 1 << n_sites
  vector = zeros(ComplexF64, dim)
  basis = 0
  for site in 1:n_sites
    basis |= initial_bit(site) << (site - 1)
  end
  vector[basis + 1] = 1.0 + 0.0im
  return vector
end

function dense_heisenberg_matrix(g::SimpleGraph, n_sites::Int)
  n_sites <= 10 || error("dense exact benchmark reference is capped at n_sites <= 10")
  dim = 1 << n_sites
  h = zeros(ComplexF64, dim, dim)
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
  return h
end

function exact_evolve(h, initial, total_time::Float64)
  decomp = eigen(Hermitian(h))
  phases = exp.((-1.0im * total_time) .* decomp.values)
  return decomp.vectors * (phases .* (decomp.vectors' * initial))
end

function state_vector(psi, n_sites::Int)
  dense = contract(psi)
  ordered_sites = [only(siteinds(psi, site)) for site in 1:n_sites]
  return vec(Array(dense, ordered_sites...))
end

function summarize(times_ns)
  seconds = times_ns ./ 1.0e9
  return mean(seconds), minimum(seconds), maximum(seconds)
end

function run_tdvp_once(H, psi0, root::Int, time_steps::Int, dt::Float64)
  time_points = collect(range(0.0, step = dt, length = time_steps + 1))
  return time_evolve(
    H,
    time_points,
    psi0;
    order = 2,
    nsites = 2,
    root_vertex = root,
    factorize_kwargs = (; cutoff = 1.0e-12, maxdim = 32),
    update!_kwargs = (; nsites = 2, solver = exponentiate_solver),
    exponentiate_solver_kwargs = (;
      ishermitian = true,
      issymmetric = true,
      krylovdim = 30,
      tol = 1.0e-12,
      maxiter = 100,
    ),
  )
end

function run_case(kind::Symbol, n_sites::Int, time_steps::Int, repeats::Int, dt::Float64)
  g = make_graph(kind, n_sites)
  sites = siteinds("S=1/2", g)
  root = root_vertex(kind)
  # ITensorNetworks.jl's TDVP sweep planner uses the graph's default DFS root.
  # Keep these benchmark topologies pinned to the same leaf root used below.
  default_root = ITensorNetworks.GraphsExtensions.default_root_vertex(g)
  default_root == root ||
    error("benchmark root $(root) does not match ITensorNetworks default sweep root $(default_root)")
  H = ttn(heisenberg_opsum(g), sites; root_vertex = root, cutoff = 1.0e-12)
  psi0 = make_initial_state(sites, n_sites)
  exact = exact_evolve(dense_heisenberg_matrix(g, n_sites), initial_vector(n_sites), dt * time_steps)
  exact_norm = norm(exact)

  # Exclude Julia compilation, ITensorNetworks.jl first-call setup, and KrylovKit
  # initialization from the timed region. The timed loop below measures TDVP only.
  run_tdvp_once(H, psi0, root, time_steps, dt)
  GC.gc()

  times = UInt64[]
  state_norm = NaN
  error = NaN
  max_link_dim = 0
  for _ in 1:repeats
    start = time_ns()
    psi_t = run_tdvp_once(H, psi0, root, time_steps, dt)
    push!(times, time_ns() - start)
    actual = state_vector(psi_t, n_sites)
    state_norm = norm(actual)
    error = norm(actual - exact)
    max_link_dim = maxlinkdim(psi_t)
  end

  mean_s, min_s, max_s = summarize(times)
  println(
    "case=$(kind) n=$(n_sites) time_steps=$(time_steps) dt=$(round(dt, digits = 12)) " *
    "time=$(round(dt * time_steps, digits = 12)) maxlinkdim=$(max_link_dim) " *
    "warmups=1 repeats=$(repeats) norm=$(round(state_norm, digits = 12)) " *
    "exact_norm=$(round(exact_norm, digits = 12)) l2_error=$(error) " *
    "rel_l2_error=$(error / max(exact_norm, floatmin(Float64))) " *
    "mean_ms=$(round(mean_s * 1000, digits = 3)) " *
    "min_ms=$(round(min_s * 1000, digits = 3)) " *
    "max_ms=$(round(max_s * 1000, digits = 3))"
  )
  return nothing
end

function main()
  n_sites = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
  time_steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
  repeats = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
  dt = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.02
  n_sites >= 2 || error("n_sites must be at least 2")
  time_steps >= 1 || error("time_steps must be at least 1")
  repeats >= 1 || error("repeats must be at least 1")
  isfinite(dt) && dt > 0 || error("dt must be finite and positive")

  run_case(:chain, n_sites, time_steps, repeats, dt)
  run_case(:star, n_sites, time_steps, repeats, dt)
  return nothing
end

main()
