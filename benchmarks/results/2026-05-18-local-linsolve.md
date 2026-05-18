# Prepared Local Linsolve Benchmark

Date: 2026-05-18

Purpose: compare the solve body after operator/RHS/initial-state construction.
Setup time is reported separately and excluded from the prepared solve timing.

## Commands

Rust:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_local_linsolve --release -- 8 4 4 1 4 4 0
RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_local_linsolve --release -- 38 32 32 1 10 30 0
RAYON_NUM_THREADS=1 T4A_PROFILE_CONTRACT=1 cargo run -p tensor4all-treetn --example benchmark_local_linsolve --release -- 38 32 32 1 1 10 0
```

Julia:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_local_linsolve.jl 8 4 4 1 4 4
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_local_linsolve.jl 8 4 4 1 1 10
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_local_linsolve.jl 38 32 32 1 1 10
```

## Representative Results

### Small Case

| Implementation | N | Bonds | Sweep steps | Solve total | Local operator apps | Apply time | Other solve overhead |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rust | 8 | 4/4 | 14 | 47.0 ms | single step: 6 | single step: 3.1 ms | single step GMRES overhead: 1.0 ms |
| Julia `maxiter=4,krylovdim=4` | 8 | 4/4 | 14 | 16.0 ms | 196 | 4.7 ms | replacebond/factorization/orthogonalization: 2.7 ms |
| Julia `maxiter=1,krylovdim=10` | 8 | 4/4 | 14 | 14.1 ms | 154 | 3.3 ms | replacebond/factorization/orthogonalization: 2.4 ms |

### Larger Case

| Implementation | N | Bonds | Sweep steps | Solve total | Local operator apps | Apply time | RHS time | Other solve overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rust `gmres_max_restarts=10,gmres_restart_dim=30` before convention fix | 38 | 32/32 | 74 | 6.69 s | single step: 12 | single step: 139.9 ms | single step: 89.7 ms | single step GMRES overhead: 5.4 ms |
| Rust `gmres_max_restarts=1,gmres_restart_dim=10` after KrylovKit convention fix | 38 | 32/32 | 74 | 6.89 s | single step: 12 | single step: 144.0 ms | single step: 91.3 ms | single step GMRES overhead: 5.9 ms |
| Julia `maxiter=1,krylovdim=10` | 38 | 32/32 | 74 | 10.47 s | 814 | 9.85 s | 7.0 ms | 0.30 s |

`Julia maxiter=10,krylovdim=30` was intentionally interrupted after more than
two minutes before the Rust convention was fixed. KrylovKit defines `maxiter`
as the number of restart cycles, so the maximum number of expansion steps is
roughly `maxiter * krylovdim`. Rust now follows that convention:
`gmres_max_restarts` maps to `GmresOptions::max_restarts`, while
`gmres_restart_dim` maps to the restart cycle length.

The comparable one-restart case is therefore Julia `maxiter=1,krylovdim=10`
against Rust `gmres_max_restarts=1,gmres_restart_dim=10`. Both perform 74 local
updates and about 814 local operator applications in this benchmark.

## Finding

The bottleneck is local projected operator application, not setup. In the
larger Julia prepared solve, `projected apply inside GMRES` accounts for
approximately 94% of solve time. In the Rust single local GMRES measurement,
`ProjectedOperator::apply` accounts for approximately 96% of local GMRES time.

For comparable local operator application counts, the synthetic prepared
benchmark does not show Rust as slower than Julia. The larger remaining gap to
track in QuanticsNEGF should therefore be problem-specific rank/topology
distribution, vector-space operation overhead, or a difference in the local
operator applications actually performed.

The isolated Rust single-step numbers include cold environment construction, so
they intentionally overestimate one step inside a full sweep. The full `N=38`
Rust solve is consistent with warm projected apply cost: about 74 local update
steps times about 12 local matvecs per step times about 6 ms per warm apply is
already about 5.3 s, close to the measured 6.69 s before other sweep overhead.
