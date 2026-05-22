# ACI Local Step Timing, System OpenBLAS

Date: 2026-05-22

## Host and Thread Settings

- Host CPU: Apple M5 Max
- OS: Darwin 25.3.0 arm64
- Rust toolchain: `rustc 1.94.0 (4a4ef493e 2026-03-02)`, LLVM 21.1.8
- Julia version: 1.12.5
- Rust source state during the run: `cbd16e4-dirty`
- Rust backend feature: `tenferro-system-blas`
- Rust BLAS provider: Homebrew OpenBLAS at `/opt/homebrew/opt/openblas`
- Rust binary link check:
  `/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib`
- Thread settings:
  `RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`,
  `BLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

The deterministic fixture has 12 sites and local dimension 2. Therefore
`chi=128` is effectively clamped by the exact central rank bound `2^6 = 64`,
which is why `chi=64` and `chi=128` are expected to be nearly identical.

## Commands

Rust:

```bash
env \
RUSTFLAGS='-L native=/opt/homebrew/opt/openblas/lib -l dylib=openblas' \
DYLD_LIBRARY_PATH=/opt/homebrew/opt/openblas/lib \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 T4A_STEP_TIMING_CHIS=16,32 \
cargo test --release -p tensor4all-aci \
  --no-default-features --features tenferro-system-blas \
  local_update_step_timing -- --ignored --nocapture
```

```bash
env \
RUSTFLAGS='-L native=/opt/homebrew/opt/openblas/lib -l dylib=openblas' \
DYLD_LIBRARY_PATH=/opt/homebrew/opt/openblas/lib \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 T4A_STEP_TIMING_CHIS=64,128 \
cargo test --release -p tensor4all-aci \
  --no-default-features --features tenferro-system-blas \
  local_update_step_timing -- --ignored --nocapture
```

Julia:

```bash
env \
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 \
julia benchmarks/julia/benchmark_aci_local_steps.jl --chis 16,32
```

```bash
env \
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 \
julia benchmarks/julia/benchmark_aci_local_steps.jl --chis 64,128
```

## Rust Results

| chi | setup ms | input ms | operator ms | MatrixLUCI ms | core ms | frame ms | total ms | final rank | final error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.117666 | 0.080397 | 0.027372 | 0.323687 | 0.004709 | 0.063438 | 0.614538 | 25 | 8.805722e-11 |
| 32 | 0.211792 | 0.094503 | 0.025479 | 0.323209 | 0.003999 | 0.100208 | 0.759833 | 26 | 6.172906e-11 |
| 64 | 0.309105 | 0.121538 | 0.026063 | 0.362123 | 0.004002 | 0.140831 | 0.962290 | 26 | 8.018094e-11 |
| 128 | 0.305374 | 0.120855 | 0.025335 | 0.359270 | 0.004044 | 0.137960 | 0.955769 | 26 | 8.018094e-11 |

## Julia Results

| chi | setup ms | input ms | operator ms | MatrixLUCI ms | core ms | frame ms | total ms | final rank | final error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.1879575 | 0.111789 | 0.0087505 | 0.3064995 | 0.0111035 | 0.1248325 | 0.755751 | 25 | 8.805721625343378e-11 |
| 32 | 0.2998115 | 0.138312 | 0.0093775 | 0.3468935 | 0.0110005 | 0.1685645 | 0.9774875 | 26 | 6.172906150931444e-11 |
| 64 | 0.4103360 | 0.1671675 | 0.010646 | 0.3941450 | 0.0114610 | 0.2140220 | 1.225249 | 26 | 8.018093743486929e-11 |
| 128 | 0.3965050 | 0.1657890 | 0.010666 | 0.4011880 | 0.0120445 | 0.2153550 | 1.227997 | 26 | 8.018093743486929e-11 |

## Notes

- Rust is faster in total for all recorded chi values under this system
  OpenBLAS comparison.
- The MatrixLUCI bucket is now close: Rust is slightly slower at `chi=16`,
  faster at `chi=32`, and faster at `chi=64/128`.
- The operator bucket differs because the Rust benchmark uses the batched
  public callback path while the Julia benchmark uses broadcasted scalar
  multiplication.
- The Rust bucket-timing code currently lives in the ignored unit test
  `local_update_step_timing` because it needs crate-private ACI state and
  timing hooks. If this benchmark becomes a regular artifact, extract the
  shared body into a feature-gated `bench_support` module and expose a thin
  wrapper from `benchmarks/rust/benchmark_aci_local_steps.rs`.
