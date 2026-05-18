# Projected Local Apply Benchmark

Date: 2026-05-18

Purpose: isolate the local projected operator apply used by two-site local
linsolve/TDVP-style updates. These timings are for synthetic chains with two
physical index groups per site: one acted index and one spectator index. The
Julia benchmark includes the spectator identity in the MPO, matching
QuanticsNEGF.jl's `add_dummy_indices` layout.

## Commands

Rust:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_projected_apply --release -- 38 32 32 3 0
RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_projected_apply --release -- 38 64 64 2 0
```

Julia:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_projected_apply.jl 38 32 32 3 0
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_projected_apply.jl 38 64 64 2 0
```

## Representative Results

| Implementation | N | State bond | Operator bond | Cold apply | Warm apply mean | Cold repeated mean |
|---|---:|---:|---:|---:|---:|---:|
| Rust `ProjectedOperator::apply` | 38 | 32 | 32 | 70.3 ms | 6.0 ms | 45.0 ms |
| Julia `ProjMPO` | 38 | 32 | 32 | 52.9 ms | 7.7 ms | 73.5 ms |
| Rust `ProjectedOperator::apply` | 38 | 64 | 64 | 564.0 ms | 68.2 ms | 532.8 ms |
| Julia `ProjMPO` | 38 | 64 | 64 | 807.2 ms | 159.4 ms | 759.7 ms |

Interpretation: both implementations show that the warm local projected apply
is already O(10 ms) at bond 32 and O(100 ms) at bond 64. A long local Dyson
test with thousands of local GMRES matvecs can therefore be slow from this hot
path alone, even when environment caches are effective.
