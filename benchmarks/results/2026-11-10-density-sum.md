# Density-matrix and N-ary sums (#1126)

Internal comparison of two algorithms in the same library. All 54 fresh-process samples completed, including the dense error checks. The method is opt-in: these results establish a memory/time trade-off, not a speed improvement.

Measured candidate: `a68d6547fe867f4f6836fe54a8b170059610ca3b`; base: `21f375de199bd2f8e3ea93c47ba05cb4f6402952`. The [raw record](2026-11-10-density-sum.json) contains source/binary hashes, every sample, load observations and thread settings. The harness is [density_sum_benchmark.rs](../rust/density_sum_benchmark.rs).

Apple M3 Pro, macOS, default Accelerate provider, release build. All six CPU-library thread variables were set to one. Three processes per method and case, alternating order; two seconds of identical CPU priming before each process. Reported time is the operation only. Peak RSS is per-child `wait4` maximum RSS and includes input construction and dense verification. Every timed output was checked against its dense exact sum and the documented cutoff/weight bounds.

CPU affinity and exclusive host access were unavailable. One-minute host load ranged from 12.97 to 17.85. Treat timings as local diagnostic observations, not an isolated-host speedup guarantee. No samples were omitted or selectively retried.

## Median operation time and process peak RSS

`D=10`, complex cores, binary sites, discarded-weight cutoff `1e-8`, no bond cap. The smooth/random/mixed definitions and random stream follow #1107. Errors are relative Frobenius residuals; ranks are the largest output bond. Both methods returned the same rank in all nine cases.

| R | N | Family | Density seconds | N-ary seconds | Density RSS MiB | N-ary RSS MiB | Density error | N-ary error | Rank |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | smooth | 0.017578 | 0.002546 | 18.05 | 18.53 | 1.138983e-04 | 1.138982e-04 | 4 |
| 8 | 8 | random | 0.018219 | 0.002488 | 18.56 | 18.14 | 1.629251e-15 | 5.327982e-15 | 16 |
| 8 | 8 | mixed | 0.017550 | 0.002510 | 18.09 | 18.53 | 9.445536e-05 | 9.445530e-05 | 4 |
| 12 | 16 | smooth | 0.107114 | 0.018606 | 29.11 | 31.20 | 1.245117e-04 | 1.245112e-04 | 4 |
| 12 | 16 | random | 0.148483 | 0.022127 | 33.86 | 31.08 | 4.321884e-15 | 7.406670e-15 | 64 |
| 12 | 16 | mixed | 0.105926 | 0.016818 | 29.05 | 31.64 | 1.104017e-04 | 1.104016e-04 | 4 |
| 12 | 64 | smooth | 1.608798 | 0.218533 | 177.45 | 396.67 | 1.277449e-04 | 1.277443e-04 | 4 |
| 12 | 64 | random | 2.172990 | 0.245461 | 184.72 | 396.44 | 4.538652e-15 | 8.848541e-15 | 64 |
| 12 | 64 | mixed | 1.642856 | 0.252054 | 177.98 | 396.80 | 1.411820e-04 | 1.411819e-04 | 4 |

At `R=12, N=64`, density summation used about 177–185 MiB versus 397 MiB, a 53–55% reduction in median process peak RSS. Its operation took 1.61–2.17 seconds versus 0.22–0.25 seconds, or 6.5–8.9 times longer. At `N=16` the random family used slightly more memory with density summation. The quadratic pair environments are a real cost; reduced output bonds do not make this algorithm universally cheaper.

## Verification and limits

- All processes asserted the relative cutoff budget against the dense exact sum. Density processes also asserted their absolute discarded-weight estimate, with a floating-point allowance. Nontrivial chain/tree unit tests separately cover cap-forced loss, exact-sum-then-truncate rank comparisons, cancellation and native scalar kinds.
- The sweep kernels preserve tensor metadata where their backend does; gradients of the new algorithm were not validated. The comparisons cover one CPU provider and chains, not large input bonds, thousand-term sums, GPUs or branching-tree performance.
- Benchmark setup/verification is intentionally dense and bounded to 2–16 binary sites. The production API remains local; the 65-site regression would reject full physical materialization.

Reproduce a cell with the release harness described in [benchmarks/README.md](../README.md#density-matrix-sums-1126). Run one method per process for RSS; retain all repetitions and source revisions when comparing results.

## Mutation evidence

Each of the seven new tests failed under a deliberate production defect, then the source was restored byte for byte. These were runtime assertion/error failures, not compile failures. The [complete mutation-to-failure mapping](2026-11-10-density-sum-mutations.json) preserves every observed failure.

| Test | Injected defect |
|---|---|
| `density_matrix_sum_retains_interference_and_cancellation` | `omit_cross_terms` |
| `density_matrix_sum_uses_discarded_weight_and_hard_cap` | `ignore_cap` |
| `density_matrix_sum_branches_complex_sites_and_error_bound` | `omit_bra_conjugation` |
| `density_matrix_sum_validates_before_single_node_fast_path` | `skip_cutoff_validation` |
| `density_matrix_sum_matches_exact_then_truncate_ranks` | `square_cutoff_again` |
| `density_matrix_sum_long_chain_stays_local` | `square_cutoff_again` |
| `hermitian_eigendecomposition_preserves_narrow_scalars` | `read_f32_as_f64` |

A separate RED/GREEN fixture rejects same-identity site indices with different dimensions at the shared summation boundary, including the singleton shortcut. No tolerance was loosened.
