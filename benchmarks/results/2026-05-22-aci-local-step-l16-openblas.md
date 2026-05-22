# ACI Local Step Timing, L=16, System OpenBLAS

Date: 2026-05-22

## Host and Thread Settings

- Host CPU: Apple M5 Max
- OS: Darwin 25.3.0 arm64
- Rust toolchain: `rustc 1.94.0 (4a4ef493e 2026-03-02)`, LLVM 21.1.8
- Julia version: 1.12.5
- Rust backend feature: `tenferro-system-blas`
- Rust BLAS provider: Homebrew OpenBLAS at `/opt/homebrew/opt/openblas`
- Julia BLAS provider: `OpenBLAS_jll v0.3.29+0` through
  `libblastrampoline_jll v5.15.0+0`
- Thread settings:
  `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`,
  `BLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

This run uses 16 sites and local dimension 2. The central exact-rank bound is
`2^8 = 256`, so `chi = 128` is not clamped by the fixture. Fixed sweeps are
used to compare the same number of local updates across Rust and Julia.

The native Rust convergence rule now mirrors Julia's `convergencecriterion`
shape: run at least `min_iters = 2`, check `errors[iteration] <= tolerance`,
and reject if any rank in `last(ranks, min_iters)` is greater than
`ranks[iteration - min_iters + 1]`. Rust also has a relative `scale_tolerance`
mode, but it is disabled here. Fixed sweeps are used because small
numerical/path differences can still make Rust and Julia stop after different
numbers of sweeps.

## Commands

Rust:

```bash
OPENBLAS_ROOT=${OPENBLAS_ROOT:-$(brew --prefix openblas)}
env \
RUSTFLAGS="-L native=${OPENBLAS_ROOT}/lib -l dylib=openblas" \
DYLD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${DYLD_LIBRARY_PATH:-}" \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=20 T4A_STEP_TIMING_N_SITES=16 \
T4A_STEP_TIMING_FIXED_SWEEPS=3 T4A_STEP_TIMING_CHIS=16,32,64,128 \
cargo test --release -p tensor4all-aci \
  --no-default-features --features tenferro-system-blas \
  local_update_step_timing -- --ignored --nocapture
```

Julia:

```bash
env \
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=20 \
julia benchmarks/julia/benchmark_aci_local_steps.jl \
  --sites 16 --fixed-sweeps 3 --chis 16,32,64,128
```

## Results

All timings are medians in milliseconds over 20 repeats.

| impl | chi | sites | sweeps | updates | setup | input | operator | MatrixLUCI | core | frame | total | rank | error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rust | 16 | 16 | 3 | 45 | 0.198875 | 0.194607 | 0.071831 | 1.259810 | 0.006457 | 0.111080 | 1.841535 | 33 | 9.525310e-11 |
| julia | 16 | 16 | 3 | 45 | 0.344733 | 0.242043 | 0.030210 | 1.233481 | 0.016289 | 0.224732 | 2.115942 | 33 | 9.525310e-11 |
| rust | 32 | 16 | 3 | 45 | 0.404332 | 0.374188 | 0.105338 | 2.209271 | 0.006583 | 0.228935 | 3.330730 | 46 | 9.720931e-11 |
| julia | 32 | 16 | 3 | 45 | 0.523897 | 0.387251 | 0.026439 | 2.414964 | 0.016959 | 0.316272 | 4.085233 | 46 | 9.720948e-11 |
| rust | 64 | 16 | 3 | 45 | 1.438957 | 0.930770 | 0.168311 | 4.708000 | 0.012729 | 0.714835 | 7.975667 | 63 | 9.320186e-11 |
| julia | 64 | 16 | 3 | 45 | 1.735355 | 1.156627 | 0.045643 | 5.213416 | 0.018834 | 0.914043 | 9.584417 | 63 | 9.930868e-11 |
| rust | 128 | 16 | 3 | 45 | 3.861642 | 1.956604 | 0.245189 | 7.987332 | 0.016482 | 1.808458 | 15.889415 | 76 | 8.523082e-11 |
| julia | 128 | 16 | 3 | 45 | 4.560911 | 1.923917 | 0.054647 | 8.276169 | 0.023434 | 2.028794 | 17.230575 | 76 | 8.961109e-11 |

## Summary

| chi | Rust total / Julia total | Rust MatrixLUCI / Julia MatrixLUCI |
|---:|---:|---:|
| 16 | 0.870 | 1.021 |
| 32 | 0.815 | 0.915 |
| 64 | 0.832 | 0.903 |
| 128 | 0.922 | 0.965 |

The MatrixLUCI bucket is close across the whole range. Rust is slightly slower
at `chi = 16`, faster for `chi = 32, 64`, and very close at `chi = 128`. Rust's
operator bucket remains slower because this benchmark exercises the public
batched callback path, while Julia uses broadcasted scalar multiplication.
