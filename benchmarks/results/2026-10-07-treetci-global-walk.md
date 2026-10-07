# TreeTCI retained-coordinate search (#812)

## Predeclared comparison

This is a correctness repair with a diagnostic cost comparison, not an
optimization promotion experiment. A floating-zone walk and the former
original-start axis scan intentionally visit different candidates and may
produce different pivots, evaluation counts and rank/error histories. Timing
ratios describe these complete algorithms at the same requested accuracy;
they do not isolate cached-readout speed or prove a general speedup.

- Baseline: `b1bb828da13dff679684b59511151ee98358d697`.
- Candidate: the #812 implementation recorded with the results below.
- Benchmark source: `benchmarks/rust/benchmark_treetci_global_search.rs`,
  identical for both builds; release profile and the same Cargo.lock.
- Separate baseline/candidate target directories and executable SHA-256 hashes.
- Hardware: AMD Ryzen 9 6900HX, 16 logical CPUs, Linux virtualized host.
- Affinity: CPU 2. Rayon, OMP, OpenBLAS, MKL and BLAS thread variables all 1.
- Effective backend worker count: 1, verified with
  `with_default_backend(|backend| backend.num_threads())` under the same
  environment and affinity. Local Rust/Cargo: 1.98.1.
- Complete suite: `chain_cos_129`, `quantics_chain_r20`,
  `tree_3x10_plus_centre`; no case exclusions or selective retries.
- One untimed complete warm-up per executable, then three complete paired
  runs in alternating AB/BA/AB order. Each executable reports a median of
  three internal repeats; summarize the median of the three paired ratios.
- Correctness: the new coupled-residual regressions must pass, and all
  sampled relative errors of both executables must be at most `1e-6`.
  The default interpolation tolerance remains `1e-8`.
- Record ranks, error bits, evaluation counts, fingerprints, elapsed times,
  peak RSS and host load for every run. Load above 16 or an internal
  median/minimum time ratio above 1.5 flags the whole timing comparison as
  inconclusive. Frequency/steal observability is limited on this shared
  virtual host; ratios remain descriptive even without such flags.

## Results

Candidate code revision: `bccbeea1`; later commits only record evidence.
The complete raw outputs, ranks, error bits, fingerprints, host load,
RSS and SHA-256 hashes are in [the JSON record](2026-10-07-treetci-global-walk.json).
All predeclared correctness and diagnostic validity gates passed. The virtual
host limits remain; this is not a general speed claim.

| Case | Baseline median (s) | Candidate median (s) | Paired cost ratio | 95% bootstrap interval | Evaluations, baseline → candidate | Sampled error, baseline → candidate |
|---|---:|---:|---:|---|---|---|
| `chain_cos_129` | 0.3576 | 1.3457 | 3.76x | [3.70, 3.84] | 42,145 → 43,250 | 5.56e-9 → 5.56e-9 |
| `quantics_chain_r20` | 0.0180 | 0.0363 | 2.01x | [1.98, 2.02] | 17,946 → 16,247 | 1.28e-8 → 1.64e-8 |
| `tree_3x10_plus_centre` | 0.0380 | 0.0853 | 2.21x | [2.19, 2.29] | 50,253 → 44,918 | 7.30e-9 → 7.29e-9 |

The statistic is the median of the three paired candidate/baseline median
ratios. Intervals use exact resampling of these three pairs (27 combinations);
three pairs provide limited statistical evidence. All cases retain four
iterations and final ranks 2, 8 and 9 respectively, but the pivots and errors
change as intended. Median process peak RSS across the complete suite is
159,820 KiB for baseline and 155,280 KiB for candidate. RSS includes setup and
sampled validation, so it does not isolate search memory or prove a reduction.

The retained-coordinate walk is slower on these cheap-oracle workloads,
even though two cases request fewer function values. Sequential coordinate
batches and additional sweeps have a cost. The change repairs missed pivots;
it is not promoted as a speed optimization. The obsolete axis scan is not
retained as a hidden fallback.

Both release builds used the same workspace path and lockfile, with separate
targets. Dependency artifacts were copied to the candidate target and the
affected treetci crate rebuilt there. Executables were retained separately:
baseline `5acee0d5…`, candidate `81a0ffb2…`; banners identify `b1bb828d` and
`bccbeea1`. The differing recorded trajectories also establish the intended
search variants. The benchmark source hash is `babf6ce9…`.

Reproduction uses the build and run pattern in [the benchmark README](../README.md#treetci-global-pivot-search-792).
Set all five thread variables to 1 and affinity to CPU 2. Warm each executable
once with no case filter; then run the complete suite in AB, BA, AB order.
Capture `/proc/loadavg` before/after and `/usr/bin/time -f
'METRICS elapsed_s=%e max_rss_kb=%M'` stderr for every run. Never overwrite or
omit unsuccessful paired outputs.
