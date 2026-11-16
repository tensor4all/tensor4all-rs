# TreeTN Fourier construction

The Fourier builders generate local Chen–Lindsey cores as column-major arrays
with shape `[left_bond, 4, right_bond]`. The fused physical position is
`2 * output_bit + input_bit`. Core owns the two compression sweeps
(`compress_chain_lu_mut`, `compress_chain_lu_on_session`, and the generic
`compress_chain_cores_mut` that SimpleTT's SVD path uses), and SimpleTT
compresses its own cores through the `ChainCore` adapter. The forward sweep
uses a relative pivot floor of `1e-14` and applies no rank cap. The backward sweep uses the
Fourier options for the relative pivot threshold and rank cap, with the same
floor when the threshold is zero. The matrix unfolding order stays fixed so a
change in representation does not change pivot tie-breaking. Per-site
normalization follows compression.

QuanticsTransform binds the compressed cores to a chain of indexed tensors
through the public TreeTN constructor. The monotone multi-axis path multiplies
bond dimensions at each cut and carries inactive axes through unchanged. For
non-monotone layouts, the existing compose-and-restructure path still uses the
per-axis builder. Patched transforms keep the existing one-path selection and
lazy scheduler. Factorization works on one core at a time. The shared backend
multiplies the resulting factor into the neighboring core.

The existing non-Fourier transforms and SimpleTT conversion APIs remain under
the default `simplett-transforms` feature. A consumer that needs only Fourier
can disable defaults and select a CPU provider and, when needed, the
`partitioned` feature. The Fourier-only dependency check uses Cargo's resolved
feature graph, including the isolated fixture's test dependencies. The nightly
feature matrix runs the layout and partitioned regression suites in that fixture.

## Matched release measurements

The construction probe calls the production Fourier builders with
`FourierOptions::forward()` on the same 1D and 2D layouts as the
`patched_fourier_large` workload. It compares base `1647bc66` with candidate
`dcda800c` on an Apple M3 Pro, using Rust 1.98.1, the platform's
Accelerate provider, `RAYON_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`.
Each case ran in five fresh processes with one warmup and fifteen timed
constructions per process. Time is the median of the five process medians,
with their range in parentheses. RSS is the median of the five process peak
RSS values from macOS `/usr/bin/time -l`. The process includes provider startup
and allocator overhead. The runs used ordinary macOS scheduling. See the
[probe, commands, and process output](../experiments/2026-11-15-fourier-construction/README.md).

| Bits per axis | Layout | Base time (ms) | Shared LU time (ms) | Base RSS (MiB) | Shared LU RSS (MiB) | Maximum operator bond |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 1D | 1.057 (1.047–1.248) | 0.468 (0.412–0.866) | 13.20 | 12.94 | 12 |
| 10 | 2D grouped | 3.128 (2.227–5.166) | 0.835 (0.830–0.838) | 13.92 | 13.27 | 12 |
| 10 | 2D interleaved | 6.675 (5.582–7.399) | 1.428 (1.419–1.452) | 46.03 | 31.69 | 144 |
| 20 | 1D | 4.635 (2.797–5.451) | 1.024 (1.022–1.038) | 15.19 | 14.00 | 12 |
| 20 | 2D grouped | 8.462 (5.810–8.693) | 2.052 (2.048–2.057) | 15.95 | 14.55 | 12 |
| 20 | 2D interleaved | 31.894 (28.933–38.017) | 4.734 (4.582–5.059) | 134.55 | 83.36 | 144 |

Every bond dimension matched the base in all six cases. Running the grouped
Gaussian example with `frontend-only` gave the same sampled errors and
reconstruction bounds at the displayed precision.

| Bits per axis | Layout | Max sampled spectral error (35 samples) | Reported reconstruction bound |
| --- | --- | ---: | ---: |
| 10 | 1D | 3.570018e-14 | 7.912518e-9 |
| 10 | 2D grouped | 7.309301e-14 | 2.492751e-9 |
| 20 | 1D | 9.556589e-14 | 3.997983e-11 |
| 20 | 2D grouped | 1.373180e-13 | 2.550413e-9 |

The spectral error is a sampled comparison with the analytic discrete
spectrum. It includes input interpolation, Fourier construction,
reconstruction, and floating-point roundoff. The scheduler bound applies to
reconstruction relative to the constructed operator. The exact-DFT regression
tests check the constructed operator separately on small inputs, with
roundoff included in their existing tolerances.
