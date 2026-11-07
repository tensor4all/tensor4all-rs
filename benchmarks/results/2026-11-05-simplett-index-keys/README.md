# SimpleTT index keys: before/after measurements (#627)

**Status: INCONCLUSIVE. Performance signoff is pending a quiet-host run.**
Both complete paired experiments are retained, including regressions and failed
noise checks. A subsequent fixed 120-second idle check failed, so no third timed
experiment was started. None of these runs certifies a speedup or passes the
performance promotion gate.

## Implementations and protocol

- Original: `097e2927d640ea68ff365ac56886be7676e79c8d`. The identical new benchmark harness was applied to
  this revision without changing its library implementation.
- Shared keys only: `6527870794612c9ef0dfcf659082f89e2af787c5`.
- Final: `a740d6c7f287090d7681eb3c3d7d2bcbe031938b` (persistent integer keys, borrowed temporary split counts,
  singleton-field boundary fix).
- Harness: `crates/tensor4all-simplett/benches/cache_overhead.rs`, SHA-256
  `790df1082fd016ecdaaeb42f540330103580af9538c15dacfe50342ec7168125`.
- Apple M3 Pro, macOS 27.0.1, default Apple Accelerate provider, Cargo `bench`
  profile (optimization level 3, debug info disabled). Affinity is unavailable.
- Thread environment: `RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.
- Six cases `(sites, stored bond dimension, left/right samples)`:
  `(32,1,8)`, `(100,1,8)`, `(100,1,32)`, `(100,8,32)`, `(300,1,8)`, `(300,8,8)`.
  Batch size is the square of the last value. Physical dimensions are binary.
- Five routes: cold fixed split, warm fixed split, warm automatic split,
  left environment hit, right environment hit. The stored cores are dense at
  bond dimension 1 or 8 and represent a known product function; this isolates
  cache costs rather than testing approximation rank or large GEMM throughput.
- Every timed case is checked against `(1025/1024)^sum(indices)` before timing.
  Partial vectors are checked against the corresponding product divided by the
  bond dimension on the left, and the product on the right. Inputs and full
  numeric results pass through `black_box`.
- Cold timing excludes construction and destruction of the TTCache, while
  including lazy indexer preparation and filling the cache. Warm routes reuse
  the populated cache. The automatic route includes split selection.
- Twenty Criterion samples per cell, 0.2-second warm-up, 0.5-second measurement.
  Three paired rounds, with arm order original/shared/final,
  final/original/shared, shared/final/original. All 30 cases run in each arm:
  270 measured cells per experiment. No builds or unrelated commands were
  launched by this task during timing.
- Tables show the median of three Criterion mean estimates. Ratios are the
  median of the three paired final/original ratios; their min/max are variation
  across rounds, **not confidence intervals**. Per-run 95% intervals and all raw
  iteration/time samples remain in the linked evidence.
- Predeclared numerical gates: each batch case at most 1.10 times original;
  final warm-hit geometric mean at least 5% lower than shared-only, with no
  batch case more than 10% slower than shared-only.
- Predeclared noise limits: one-minute load at most 8 at run boundaries,
  relative Criterion mean CI half-width at most 25%, per-arm/case mean CV across
  repetitions at most 20%, and no CPU speed limit below 100 when available.
  macOS reported no thermal/performance warning and no CPU power status; CPU
  speed-limit verification is therefore unavailable.

## Validity and decisions

Experiment 1 (final arm before the split-count correction) is inconclusive:
load reached 8.13. Automatic-split medians were 14–20% slower than original in
five cases, motivating removal of disposable candidate-key encoding. It was not
promoted as a successful result.

Experiment 2 is also inconclusive: its initial load was 22.49 after validation,
the largest relative CI half-width was 25.77%, and the largest repetition CV
was 24.04%. The observed paired ratios meet the numerical timing thresholds,
but **the failed noise gates invalidate a performance conclusion**. The final
idle check recorded load 5.28 before and
6.86 after 120 seconds, above its maximum of 6.
No samples from different experiments are combined or selectively retained.

In experiment 2, observed cold-batch ratios span 0.40–0.67, automatic-split
ratios 0.66–0.89, fixed-split warm ratios 0.86–1.05, and environment-hit ratios
0.53–0.89. These are diagnostics for choosing the implementation, not certified
speedups. A fixed-split warm case was about 5% slower; no regression is omitted.

## Complete experiment 2 observations

All times are microseconds per call; lower is better. The last column is the
observed paired ratio range across all three rounds.

| Case | Original | Shared only | Final | Final/original | Paired range |
|---|---:|---:|---:|---:|---:|
| cold_fixed/n100_chi1_batch1024 | 1249.5650 | 1282.5362 | 550.8476 | 0.446 | 0.422–0.449 |
| cold_fixed/n100_chi1_batch64 | 275.5955 | 259.5954 | 111.1084 | 0.400 | 0.396–0.457 |
| cold_fixed/n100_chi8_batch1024 | 2333.7187 | 2401.0404 | 1605.4112 | 0.671 | 0.663–0.698 |
| cold_fixed/n300_chi1_batch64 | 1724.8189 | 1740.7158 | 675.7585 | 0.397 | 0.329–0.400 |
| cold_fixed/n300_chi8_batch64 | 2568.9811 | 2723.3496 | 1393.4829 | 0.538 | 0.525–0.546 |
| cold_fixed/n32_chi1_batch64 | 50.5009 | 51.1939 | 32.8485 | 0.664 | 0.645–0.694 |
| left_hit/n100_chi1_batch1024 | 0.2205 | 0.2306 | 0.1194 | 0.536 | 0.511–0.544 |
| left_hit/n100_chi1_batch64 | 0.2259 | 0.2249 | 0.1231 | 0.535 | 0.512–0.545 |
| left_hit/n100_chi8_batch1024 | 0.2309 | 0.2297 | 0.1232 | 0.533 | 0.527–0.534 |
| left_hit/n300_chi1_batch64 | 0.5626 | 0.5897 | 0.4662 | 0.832 | 0.804–0.872 |
| left_hit/n300_chi8_batch64 | 0.5662 | 0.5870 | 0.4756 | 0.860 | 0.804–0.899 |
| left_hit/n32_chi1_batch64 | 0.1068 | 0.1036 | 0.0590 | 0.545 | 0.545–0.569 |
| right_hit/n100_chi1_batch1024 | 0.2249 | 0.2294 | 0.1204 | 0.538 | 0.506–0.538 |
| right_hit/n100_chi1_batch64 | 0.2264 | 0.2356 | 0.1192 | 0.531 | 0.456–0.532 |
| right_hit/n100_chi8_batch1024 | 0.2295 | 0.2304 | 0.1222 | 0.534 | 0.533–0.557 |
| right_hit/n300_chi1_batch64 | 0.5606 | 0.5652 | 0.4614 | 0.824 | 0.819–0.829 |
| right_hit/n300_chi8_batch64 | 0.5558 | 0.5724 | 0.5055 | 0.892 | 0.843–0.955 |
| right_hit/n32_chi1_batch64 | 0.1103 | 0.1055 | 0.0592 | 0.537 | 0.534–0.555 |
| warm_auto/n100_chi1_batch1024 | 451.6654 | 546.4850 | 388.8888 | 0.849 | 0.811–0.909 |
| warm_auto/n100_chi1_batch64 | 33.7816 | 39.8668 | 26.2989 | 0.778 | 0.608–0.788 |
| warm_auto/n100_chi8_batch1024 | 442.5647 | 563.4520 | 393.6059 | 0.885 | 0.881–0.911 |
| warm_auto/n300_chi1_batch64 | 145.7726 | 170.2141 | 96.9801 | 0.659 | 0.655–0.671 |
| warm_auto/n300_chi8_batch64 | 143.9775 | 170.9023 | 95.9193 | 0.665 | 0.639–0.683 |
| warm_auto/n32_chi1_batch64 | 17.8265 | 16.6465 | 11.8474 | 0.666 | 0.662–0.669 |
| warm_fixed/n100_chi1_batch1024 | 208.0778 | 226.9675 | 214.2968 | 1.037 | 1.027–1.176 |
| warm_fixed/n100_chi1_batch64 | 18.3411 | 18.9715 | 17.3868 | 0.935 | 0.930–0.950 |
| warm_fixed/n100_chi8_batch1024 | 214.4640 | 229.2867 | 217.4916 | 0.993 | 0.992–1.026 |
| warm_fixed/n300_chi1_batch64 | 63.4761 | 69.9019 | 66.2604 | 1.044 | 1.036–1.050 |
| warm_fixed/n300_chi8_batch64 | 62.9716 | 73.3752 | 67.9086 | 1.050 | 1.049–1.141 |
| warm_fixed/n32_chi1_batch64 | 8.9891 | 8.5239 | 7.7554 | 0.861 | 0.844–0.898 |

## Evidence and reproduction

- [Experiment 1 per-run means and 95% intervals](experiment1.csv)
- [Experiment 1 complete protocol, binary hashes, host observations and raw samples](experiment1.json.gz)
- [Experiment 2 per-run means and 95% intervals](experiment2.csv)
- [Experiment 2 complete protocol, binary hashes, host observations and raw samples](experiment2.json.gz)
- [Failed fixed idle check](idle-check.json)
- [Generated summary](summary.json)

Build `cache_overhead` at each pinned revision with the identical harness
above. Build once before measuring, retain each executable separately, then run
these commands for all three arm orders, with the thread environment above:

```sh
cargo bench -p tensor4all-simplett --bench cache_overhead --no-run
# Run each retained executable; labels must be unique per arm/round.
<retained-executable> cache_keys --bench --noplot --save-baseline <arm-round>
```

Before timing, a quiet-host run must pass the recorded idle and noise checks.
Preserve full suites, confidence intervals, host observations and invalid runs;
do not keep rerunning selected cases until they pass. Kernel changes, larger
ranks, non-binary benchmark dimensions, memory consumption and multithread
scaling were not measured. Correctness covers non-binary/singleton dimensions,
zero-dimensional partial reads, and keys beyond 1024 bits separately.
