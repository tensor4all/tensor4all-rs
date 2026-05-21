# ACI Elementwise Chi Scaling

Date: 2026-05-21

## Host and Thread Settings

- Host CPU: AMD EPYC 7713P 64-Core Processor
- Rust threads: `RAYON_NUM_THREADS=1`
- Julia BLAS threads: `BLAS_NUM_THREADS=1`
- Julia version: 1.12.5
- Rust toolchain: `rustc 1.94.1 (e408947bf 2026-03-25)`, LLVM 21.1.8
- Rust source state during the run: `469d612-dirty`
- Julia package source: `ACI_JL_PATH=/tmp/AlternatingCrossInterpolation.jl`
- Julia upstream SHA: `3a33e2b738c9502f7e312b79ebe45a578611d2ff`
- Relevant Julia packages resolved in a temporary project:
  `AlternatingCrossInterpolation v0.1.0`,
  `TensorCrossInterpolation v0.9.19`, and `BenchmarkTools v1.8.0`

## Commands

```bash
cargo bench --no-run -p tensor4all-aci --bench elementwise_scaling
RAYON_NUM_THREADS=1 cargo bench -p tensor4all-aci --bench elementwise_scaling -- --sample-size 10 --measurement-time 1 --warm-up-time 1
ACI_JL_PATH=/tmp/AlternatingCrossInterpolation.jl BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_aci_elementwise.jl --chis 2,4,8,16
```

The benchmark scripts use the same deterministic closed-form TT core formula.
The formula depends on physical and left/right bond coordinates so the fixture
does not merely inflate structural bond dimensions. The Rust script asserts
sampled max absolute error below `1e-8` and asserts non-rank-one output for
`chi > 2`.

The requested BenchmarkTools UUID
`6e4b80f9-dda5-5e3a-8b7e-868b0140f7fe` is not registered. The benchmark project
uses the registered UUID `6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf`.

## Rust Results

Rust metadata rows printed by the benchmark:

| chi | Criterion estimate ms | output max chi | sweeps | final error | sampled max abs error |
|---:|---:|---:|---:|---:|---:|
| 2 | 3.4510 | 1 | 2 | 8.310106e-11 | 2.990242e-11 |
| 4 | 25.551 | 7 | 4 | 9.776016e-11 | 4.295409e-11 |
| 8 | 84.297 | 14 | 2 | 9.902878e-11 | 8.034786e-11 |
| 16 | 4216.9 | 23 | 4 | 9.759426e-11 | 7.172652e-11 |

## Julia Results

CSV output:

| impl | n_sites | local_dim | chi | tolerance | median_ms | min_ms | mean_ms | max_ms | output_max_chi | n_sweeps | final_error | sampled_max_abs_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| julia | 12 | 2 | 2 | 1.0e-10 | 2.019424 | 2.012608 | 2.0394163 | 2.205166 | 1 | 2 | 8.310105747818132e-11 | 2.9902424477977914e-11 |
| julia | 12 | 2 | 4 | 1.0e-10 | 3.1609285 | 3.137227 | 3.1978732 | 3.520995 | 7 | 3 | 9.776015850916343e-11 | 4.2954087123189454e-11 |
| julia | 12 | 2 | 8 | 1.0e-10 | 2.4988745 | 2.375364 | 2.6774354 | 3.340246 | 14 | 2 | 9.573733291247284e-11 | 4.0545975613723955e-11 |
| julia | 12 | 2 | 16 | 1.0e-10 | 6.193907 | 5.741522 | 7.2255916 | 13.064673 | 25 | 3 | 8.80572239324038e-11 | 5.599156990105606e-11 |

## Interpretation

The revised fixture now exercises nontrivial output-rank growth: output max chi
increases from 1 at `chi = 2` to the low-to-mid twenties at `chi = 16`. Runtime
therefore reflects both larger input cores and larger local two-site blocks.

In this short local run, Rust scales much more steeply at `chi = 16` than Julia.
The jump is consistent with the Rust public API path spending most of the high
chi case in local block construction, cache lookups, and MatrixLUCI updates over
larger intermediate blocks. This is a smoke benchmark with only ten samples, so
the exact timing ratios should be treated as directional.

Numerical parity is within the required sampled threshold. All sampled max
absolute errors are below `1e-8`. Rust and Julia agree on output max chi through
`chi = 8`; for `chi = 16`, Rust returned output max chi 23 and Julia returned
25, with both final error metrics near `1e-10`.
