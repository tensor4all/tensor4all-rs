# Completing CI latency investigation #769

## Decisions

- Keep the improvements merged in #782 and evaluate the remaining build reuse,
  fixture size, redundant execution, and integration-binary work against the
  acceptance criteria in [#769](https://github.com/tensor4all/tensor4all-rs/issues/769).
- Start binary consolidation with the five index/tag integration targets in
  `tensor4all-core`. They share one public API responsibility and do not use a
  numerical backend, environment mutation, HDF5, or per-binary setup. Preserve
  modules and individual test names so failures remain attributable. The
  [controlled local experiment](../experiments/ci769-index-suite.md) supports
  this limited grouping; it does not establish a workspace-wide speedup.
- Keep numerical regressions when a smaller fixture with equivalent failure
  detection has not been demonstrated. Unchanged line coverage alone is not
  evidence of equivalent regression protection.

## Verification conclusions and constraints

- The [fixture experiment](../experiments/ci769-fixtures.md) preserves numerical
  tolerances and all tutorial flows. Interpolative QTT improves in repeated 1T
  measurements; the tutorial package-wide difference is inconclusive.
- The final focused local Nextest selection passes 105 tests: 54 index cases,
  27 interpolative cases, and 24 tutorial cases. Changed-package all-target
  Clippy passes with the hosted error/panic documentation warning policy.
- The [hosted build evidence](../experiments/ci769-build.md) proves heavy
  dependency reuse and identifies unchanged workspace source timestamps as a
  cause of recompilation. No timestamp manipulation is introduced.
- Hosted acceptance, the book harness edition evaluation, and final runner-cost
  accounting are in progress; issue completion is not yet established.
