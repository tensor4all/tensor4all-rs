# Compression-check paired experiment

This compares two versions of the same library operation. The reference is
`compression_change` from the #1310 parent commit, retained verbatim as a test-only
helper. Both versions run on the same compressed train and child node values in
one release test binary. Setup, interpolation-core construction and SVD are
outside the timed region. No vendor or LAPACK source was consulted.

## Predeclared settings

- CPU: AMD Ryzen 9 6900HX; native/faer provider; `f64`; one backend thread.
- Measured CPU: 14; its SMT sibling 15 is reserved. Observer processes use 0-13.
- Build: Cargo `release`, locked dependencies, `parallel` feature enabled.
- Source: `patched::tests::compression::compression_change_timing`.
- Cases `(D, sites, degree, calls per sample)`: `(1,4,3,100)`, `(1,12,6,100)`,
  `(2,4,3,10)`, `(2,12,3,10)`, `(3,4,2,3)`, `(3,10,2,3)`.
- Each fixture uses the same target, bounds `[-0.3,0.9]^D`, compression tolerance
  `1e-10`, and rank cap 4. Report the actual resulting rank.
- Prime for one second per case and warm each implementation once. Take 11 pairs,
  alternating reference/candidate order; retain every pair.
- Before timing, require bitwise equality of both computed residuals for every
  case. Inputs and scalar residuals pass through `black_box`.
- Primary statistic: geometric mean of the six median paired candidate/reference
  time ratios. Acceptance requires a ratio below 1.0 and all correctness gates.
  Individual cases are descriptive, including regressions; no selective retries.
- Report each case's median per-call times and paired ratio with a deterministic
  95% bootstrap interval (10,000 resamples, seed 1311).
- Sample `/proc/stat` for one second before and after the complete suite. If
  non-idle time on either reserved logical CPU exceeds 10%, or steal time
  increases, mark the entire experiment INCONCLUSIVE. Record load averages,
  affinity, thread settings, source pins, binary hash and raw output.
- Run no other build, test or benchmark during timing. Do not edit or commit
  measured sources while the run is in flight. Restore observer affinities after.

The recorder resolves source pins with Git before executing the binary; they
are not transcribed into this protocol by hand. Its output accompanies the
complete raw CSV and generated results page.

## Candidate refinement

The first complete valid experiment missed the primary gate (ratio 1.326).
Its complete evidence is retained under `initial/`; it is not discarded or
reclassified. Full-left `evaluate_many` paid for two-sided index mapping and
persistent hash/vector entries despite the compression check's call-local owner.
The revised candidate specializes that natural split boundary in simplett with
a lexicographic batch traversal and one transient prefix environment per level.
It preserves input result order and the full left-to-right arithmetic. The
complete original case grid, statistic, gates and repetition policy apply again;
no cases or thresholds change.
