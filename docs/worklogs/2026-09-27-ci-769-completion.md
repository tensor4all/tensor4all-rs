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
- After the review follow-up, the final focused local Nextest selection passes
  106 tests: 54 index cases, 27 interpolative cases, and 25 tutorial cases,
  including the default depth-15 QTT sweep. Changed-package all-target
  Clippy passes with the hosted error/panic documentation warning policy.
- The follow-up PR head also passes hosted Test and Coverage with 3,476 and
  3,429 passing tests, respectively; coverage's 253 file thresholds pass.
  The added depth-15 test is included in both hosted test counts.
- The [hosted build evidence](../experiments/ci769-build.md) proves heavy
  dependency reuse and identifies unchanged workspace source timestamps as a
  cause of recompilation. No timestamp manipulation is introduced.
- The [book edition evaluation](../experiments/ci769-book-edition.md) supports
  Rust 2024 for both book configurations: all 47 Cargo examples and the same
  mdBook chapters remain exercised. Build preparation and post-build times are
  reported separately; the standalone timing comparison is a reference sample.
- Final hosted checks, unchanged coverage thresholds, and aggregate runner-time
  comparisons are recorded in [PR #785](https://github.com/tensor4all/tensor4all-rs/pull/785),
  alongside the exact revisions and cache/runner conditions.
