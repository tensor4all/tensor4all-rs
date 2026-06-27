# TreeTN TDVP vs ITensorNetworks.jl Benchmark

Date: 2026-06-27

Purpose: compare the initial TreeTN TDVP implementation against a local
ITensorNetworks.jl checkout on matching chain and non-chain tree topologies.

Both runners evolve an alternating product state under the Pauli Heisenberg
Hamiltonian `sum_(i,j in E) X_i X_j + Y_i Y_j + Z_i Z_j` for total time
`t = 0.08` using 4 two-site TDVP time steps of `dt = 0.02`. Both use
`order = 2`, `maxdim = 32`, an ITensors-compatible relative discarded squared
singular-value tail cutoff of `1e-12`, and Hermitian Krylov exponentials with
`krylovdim = 30` and `tol = 1e-12`. The iteration caps are set to the same
numeric value, `100`, but they are not identical semantics: Julia's
KrylovKit `maxiter` controls adaptive Krylov restarts inside one exponential,
while Rust's `max_time_splits` caps a step-halving retry schedule.

Accuracy is the dense-vector L2 error against an exact dense Hermitian
eigendecomposition reference. Dense materialization is used only by these
small benchmark validators, not by the TDVP production algorithm.

For the star topology, both runners use a leaf as the TDVP root. This matches
ITensorNetworks.jl's `ttn(OpSum, ...)` requirement that the tree root be a leaf
vertex. The Julia runner asserts that this leaf also matches
ITensorNetworks.jl's default TDVP sweep root for the benchmark graph.

## Commands

Rust:

```bash
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
cargo run -p tensor4all-treetn --example benchmark_tdvp --release -- 8 4 3 0.02
```

Julia:

```bash
export T4A_ITENSORNETWORKS_PATH=/home/shinaoka/tensor4all/ITensor/ITensorNetworks.jl
export T4A_ITN_BENCH_PROJECT=/tmp/t4a-itensornetworks-bench-issue531
julia --project=$T4A_ITN_BENCH_PROJECT -e 'using Pkg; Pkg.develop(path=ENV["T4A_ITENSORNETWORKS_PATH"]); Pkg.add(["Graphs", "ITensors", "TensorOperations", "KrylovKit"]); Pkg.instantiate()'
BLAS_NUM_THREADS=1 julia --project=$T4A_ITN_BENCH_PROJECT \
  benchmarks/julia/benchmark_tdvp_itensornetworks.jl 8 4 3 0.02
```

The Julia runner performs one untimed warmup per topology and reuses the
already-built initial state and operator in the timed loop, so reported timings
exclude Julia compilation, ITensorNetworks.jl first-call setup, KrylovKit
initialization, operator construction, and dense exact reference construction.

## Representative Results

| Implementation | Topology | N | Time steps | Total time | Repeats | Norm | L2 error | Relative L2 error | Mean ms | Min ms | Max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rust TreeTN TDVP | chain | 8 | 4 | 0.08 | 3 | 0.999999999998 | 1.375e-5 | 1.375e-5 | 205.519 | 203.475 | 209.107 |
| ITensorNetworks.jl | chain | 8 | 4 | 0.08 | 3 | 0.999999999998 | 1.375e-5 | 1.375e-5 | 148.246 | 138.167 | 156.985 |
| Rust TreeTN TDVP | star | 8 | 4 | 0.08 | 3 | 1.000000000000 | 3.999e-4 | 3.999e-4 | 1510.157 | 1507.512 | 1514.993 |
| ITensorNetworks.jl | star | 8 | 4 | 0.08 | 3 | 1.000000000000 | 3.999e-4 | 3.999e-4 | 7988.115 | 7689.838 | 8563.478 |

## Notes

- The Rust benchmark source is `benchmarks/rust/benchmark_tdvp.rs`, included by
  `crates/tensor4all-treetn/examples/benchmark_tdvp.rs`.
- The Julia benchmark source is
  `benchmarks/julia/benchmark_tdvp_itensornetworks.jl`.
- The Rust benchmark reported `sweeps_completed=4` and `local_updates=104` for
  both chain and star, confirming that all requested TDVP steps ran.
- The Rust benchmark reported max Krylov diagnostic error `2.000e-2` and max
  Krylov iterations `8` for both chain and star. The diagnostic error is a
  successive-approximation quantity and can be nonzero when convergence is
  certified by Lanczos breakdown in an invariant subspace; dense exact L2 error
  above is the benchmark accuracy metric.
- The chain case is a control for harness overhead and is faster in
  ITensorNetworks.jl for this small run. The non-chain star case is faster in
  Rust while matching ITensorNetworks.jl's dense exact error.
