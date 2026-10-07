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

Pending the complete paired comparison. No candidate timings have been run.
