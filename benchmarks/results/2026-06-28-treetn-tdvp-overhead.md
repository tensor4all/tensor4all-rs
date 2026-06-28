# TreeTN TDVP Overhead Profile and ITensorNetworks.jl Comparison

Date: 2026-06-28

Purpose: measure the TreeTN TDVP overhead reductions against `origin/main` and
record the parity comparison with a local ITensorNetworks.jl checkout. The Rust
algorithm keeps the state as a TreeTN throughout TDVP; dense materialization is
used only by the small benchmark exact-reference validator.

Both runners evolve an alternating product state under the Pauli Heisenberg
Hamiltonian `sum_(i,j in E) X_i X_j + Y_i Y_j + Z_i Z_j` for total time
`t = 0.08` using 4 two-site TDVP time steps of `dt = 0.02`. Both use
`order = 2`, `maxdim = 32`, an ITensors-compatible relative discarded squared
singular-value tail cutoff of `1e-12`, and Hermitian Krylov exponentials with
`krylovdim = 30` and `tol = 1e-12`.

## Commands

Rust optimized branch:

```bash
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
cargo run -p tensor4all-treetn --example benchmark_tdvp --release -- 8 4 3 0.02
```

Rust `origin/main` baseline was measured from a temporary detached worktree:

```bash
git worktree add --detach /tmp/t4a-origin-main-bench origin/main
(
  cd /tmp/t4a-origin-main-bench
  CARGO_TARGET_DIR=/tmp/t4a-origin-main-target \
  RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
  cargo run -p tensor4all-treetn --example benchmark_tdvp --release -- 8 4 3 0.02
)
git worktree remove --force /tmp/t4a-origin-main-bench
rm -rf /tmp/t4a-origin-main-target
```

Julia:

```bash
export T4A_ITENSORNETWORKS_PATH=/home/shinaoka/tensor4all/ITensor/ITensorNetworks.jl
export T4A_ITN_BENCH_PROJECT=/tmp/t4a-itensornetworks-bench-tdvp-overhead
julia --project=$T4A_ITN_BENCH_PROJECT -e 'using Pkg; Pkg.develop(path=ENV["T4A_ITENSORNETWORKS_PATH"]); Pkg.add(["Graphs", "ITensors", "TensorOperations", "KrylovKit"]); Pkg.instantiate()'
BLAS_NUM_THREADS=1 julia --project=$T4A_ITN_BENCH_PROJECT \
  benchmarks/julia/benchmark_tdvp_itensornetworks.jl 8 4 3 0.02
```

The Julia runner performs one untimed warmup per topology and reuses the
already-built initial state and operator in the timed loop, so reported timings
exclude Julia compilation, ITensorNetworks.jl first-call setup, KrylovKit
initialization, operator construction, and dense exact reference construction.

## Results

| Implementation | Topology | N | Time steps | Repeats | Norm | L2 error | Relative L2 error | Mean ms | Min ms | Max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rust TreeTN TDVP (`origin/main`) | chain | 8 | 4 | 3 | 0.999999999998 | 1.375e-5 | 1.375e-5 | 213.602 | 209.228 | 219.574 |
| Rust TreeTN TDVP (optimized) | chain | 8 | 4 | 3 | 0.999999999998 | 1.375e-5 | 1.375e-5 | 194.335 | 192.744 | 195.922 |
| ITensorNetworks.jl | chain | 8 | 4 | 3 | 0.999999999998 | 1.3750765420845311e-5 | 1.3750765420845311e-5 | 169.096 | 152.953 | 197.118 |
| Rust TreeTN TDVP (`origin/main`) | star | 8 | 4 | 3 | 1.000000000000 | 3.999e-4 | 3.999e-4 | 1769.085 | 1748.214 | 1780.838 |
| Rust TreeTN TDVP (optimized) | star | 8 | 4 | 3 | 1.000000000000 | 3.999e-4 | 3.999e-4 | 1521.991 | 1489.183 | 1575.377 |
| ITensorNetworks.jl | star | 8 | 4 | 3 | 1.0 | 0.00039988891002920937 | 0.0003998889100292092 | 8665.628 | 8366.429 | 8871.538 |

## Notes

- Optimized Rust vs `origin/main`: chain improves from 213.602 ms to
  194.335 ms, and star improves from 1769.085 ms to 1521.991 ms.
- Optimized Rust matches the dense-reference L2 errors reported by
  ITensorNetworks.jl for both chain and star.
- The optimized Rust benchmark reports `sweeps_completed=4` and
  `local_updates=104` for both topologies. Maximum Krylov residual estimates
  were `8.898e-13` for chain and `2.883e-22` for star.
- The chain case remains a small-problem overhead control where
  ITensorNetworks.jl is faster in mean time on this run. The non-chain star case
  is faster in Rust by about 5.7x while matching the dense exact error.
