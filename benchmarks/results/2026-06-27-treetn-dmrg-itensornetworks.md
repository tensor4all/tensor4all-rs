# TreeTN DMRG vs ITensorNetworks.jl Benchmark

Date: 2026-06-27

Purpose: compare the initial two-site TreeTN DMRG implementation against a
local ITensorNetworks.jl checkout on matching chain and non-chain tree
topologies.

Both runners use a Pauli Heisenberg Hamiltonian
`sum_(i,j in E) X_i X_j + Y_i Y_j + Z_i Z_j`, `maxdim = 32`, two-site updates,
one untimed DMRG warmup per topology, and an ITensors-compatible relative
discarded squared singular-value tail cutoff of `1e-12`.

For the star topology, both runners use a leaf as the DMRG root. This matches
ITensorNetworks.jl's `ttn(OpSum, ...)` requirement that the tree root be a leaf
vertex. The Rust runner does not enable sweep-to-sweep energy early stopping
for this benchmark and reports that all 4 requested sweeps completed.

## Commands

Rust:

```bash
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
cargo run -p tensor4all-treetn --example benchmark_dmrg --release -- 8 4 3
```

Julia:

```bash
BLAS_NUM_THREADS=1 \
julia --project=/tmp/t4a-itensornetworks-bench-issue531 \
  benchmarks/julia/benchmark_dmrg_itensornetworks.jl 8 4 3
```

The Julia runner performs one untimed warmup per topology and constructs the
repeat initial state outside the timed region, so reported timings exclude Julia
compilation, ITensorNetworks.jl first-call setup, and random initial-state
construction.

## Representative Results

| Implementation | Topology | N | Sweeps | Repeats | Energy | Exact | Abs error | Mean ms | Min ms | Max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rust TreeTN DMRG | chain | 8 | 4 | 3 | -13.499730394752 | -13.499730394752 | 5.329e-15 | 135.364 | 132.498 | 137.314 |
| ITensorNetworks.jl | chain | 8 | 4 | 3 | -13.499730394752 | -13.499730394752 | 5.151e-14 | 117.271 | 107.571 | 136.669 |
| Rust TreeTN DMRG | star | 8 | 4 | 3 | -9.000000000000 | -9.000000000000 | 9.059e-14 | 242.797 | 239.229 | 245.971 |
| ITensorNetworks.jl | star | 8 | 4 | 3 | -9.000000000000 | -9.000000000000 | 2.487e-14 | 1928.784 | 1716.034 | 2199.503 |

## Notes

- The Rust benchmark source is `benchmarks/rust/benchmark_dmrg.rs`, included by
  `crates/tensor4all-treetn/examples/benchmark_dmrg.rs`.
- The Julia benchmark source is `benchmarks/julia/benchmark_dmrg_itensornetworks.jl`.
- The Rust benchmark also reports max residual norms. In this run they were
  `2.784e-6` for the chain and `7.326e-3` for the star, while both energies
  matched the dense exact reference to roundoff.
- The Rust benchmark reported `sweeps_completed=4`, `local_updates=56`, and
  `converged=false` for both chain and star, confirming that the timed runs did
  not stop early.
- Rust reuses one deterministic initial state for timed repeats, while Julia
  uses deterministic per-repeat random initial states. Since both runners use a
  fixed sweep count, this affects trajectories but not the amount of requested
  sweep work.
- The chain case is a control for harness overhead: Rust and ITensorNetworks.jl
  are comparable on the chain, while the high-valence star is much slower in
  ITensorNetworks.jl. The exact cause is not isolated by this benchmark alone.
