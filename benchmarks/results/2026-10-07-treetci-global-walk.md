# TreeTCI retained-coordinate search (#812)

## Predeclared comparison

This is a correctness repair with a diagnostic cost comparison, not an
optimization promotion experiment. A floating-zone walk and the former
original-start axis scan intentionally visit different candidates and may
produce different pivots, evaluation counts and rank/error histories. Timing
ratios describe these complete algorithms at the same requested accuracy;
they do not isolate cached-readout speed or prove a general speedup.

- Baseline: `4249c445` (#838 merged).
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

## Synchronized comparison

Candidate: `8b1ba82c`, synchronized with `4249c445`. The predeclared cases,
statistic, repetitions and validity/accuracy gates above are unchanged.
The complete experiment is rerun after synchronization; no selective retries.

All predeclared diagnostic and accuracy gates passed; no runs were excluded.
Ranks are 2, 8 and 9 respectively, with four outer iterations in every run.

| Case | Baseline median (s) | Candidate median (s) | Median paired ratio | Paired bootstrap interval | Oracle evaluations (baseline / candidate) | Sampled relative error (baseline / candidate) |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `chain_cos_129` | 0.3653 | 1.4046 | 3.86x | [3.69, 3.90] | 42145 / 43250 | 5.56e-9 / 5.56e-9 |
| `quantics_chain_r20` | 0.0193 | 0.0370 | 1.92x | [1.91, 1.94] | 17946 / 16247 | 1.28e-8 / 1.64e-8 |
| `tree_3x10_plus_centre` | 0.0404 | 0.0877 | 2.17x | [2.17, 2.26] | 50253 / 44918 | 7.30e-9 / 7.29e-9 |

The interval is the percentile 95% interval from all 27 resamples of three
paired ratios; with three pairs it equals their minimum/maximum and has limited
statistical resolution. The algorithms visit different trajectories. This
correctness repair costs more on these cheap-oracle cases, even where the
oracle evaluation count falls. Higher thread counts and expensive oracles
remain unmeasured.

Whole-process peak RSS medians are 149176 KiB (baseline) and
151932 KiB (candidate), including validation and all cases; this
does not establish a per-search memory improvement. Executable hashes, load
observations, raw stdout/stderr and every run are retained in
[the synchronized JSON](2026-10-07-treetci-global-walk.json).

## Earlier comparison before synchronization

The complete initial comparison on `b1bb828d` / `bccbeea1` is retained in
[the pre-synchronization JSON](2026-10-07-treetci-global-walk-pre-sync.json).
Its paired cost ratios were 3.76x, 2.01x and 2.21x; every diagnostic/accuracy
gate passed. This is retained evidence, not a substituted favorable subset.
The rerun is required because #838 changed materialization on main.

## Reproduction

Use the build/run pattern in [the benchmark README](../README.md#treetci-global-pivot-search-792)
with separate target directories. Dependency artifacts may be copied; rebuild
the affected crates for each revision, retain executable hashes and verify
build banners. Set all five thread variables to 1 and affinity to CPU 2.
Warm each executable once with no case filter, then run the complete suite
in AB, BA, AB order. Capture `/proc/loadavg` before/after and `/usr/bin/time -f
'METRICS elapsed_s=%e max_rss_kb=%M'` stderr for every run. Never overwrite or
omit unsuccessful paired outputs.
