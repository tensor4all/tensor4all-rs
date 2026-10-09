# TreeACI cross retention at the tolerance boundary

## Decisions

- Reconstruct an established cross, or complete its still-available pivot
  axis, within the current Cartesian candidate matrix. Accept it only after
  measuring the full local maximum residual at the unchanged tolerance. Fresh
  LUCI still bounds the retained rank, so every rank decrease remains possible.
- Keep cross reconstruction and axis completion in Core. TreeACI maps immutable
  component samples to candidate coordinates and owns the acceptance decision.
  Extending candidate axes with historical samples would also require changing
  adjacent CI gauges; it is not a valid local substitution.
- Pin the interpolative factor to exact identity at its selected pivot axis.
  These entries are algebraic invariants; retaining solve round-off there
  changes subsequent CI gauge choices on branches. Measure the residual after
  pinning, rather than assuming this modification is harmless.
- Retention is optional under the existing working-byte budget. Reserve the
  original matrix and another conservative Core LUCI working estimate while
  fresh factors remain live; fall back to ordinary LUCI when this does not fit.
- Canonical sorting of pivot pairs did not eliminate the original cycle.
  Requiring the retained interpolation norm to be no larger than fresh LUCI
  eliminated the useful retention and reproduced the baseline cycle. Removing
  the fresh-rank ceiling did not resolve the chi=60 witness either. None of
  these experimental policies is included.
- The test-only skeleton reference now contracts a site-free scalar network.
  Explicit enumeration of rank^(2E) inverse-gauge products amplified round-off
  on the binary-tree edge-order regression. Existing accuracy thresholds and
  all existing tests remain in place; the deleted recursive reference helper
  had no production callers.

## Verification conclusions and constraints

- Work remains in progress against issue #784, based on main e0a63a31. No PR
  or issue-closing claim is justified yet.
- The original 50-site, d=3, chi=40 DMRG product changes from MaxSweeps at 20
  to Converged at 5 with cap 400, relative tolerance 1e-8, seed 1, and Guard
  enabled. Independent TT-inner-product error remains about 3e-6; the metric's
  resolution is about 1e-7. This is not a full-grid maximum-error certificate.
- A conditioned 26-site window converges at 7 with one-axis completion;
  complete-cross retention alone preserves its period-four cycle. The 28-site
  window converges at 9 with complete-cross retention.
- R=10 NBlock W, absolute tolerance 1e-4, seed 0, cap 4096, changes from
  MaxSweeps at 20 to Converged at 18. Its 1,086 independently contracted sample
  residuals satisfy the unchanged 10*tolerance diagnostic gate; the maximum
  sampled absolute residual is approximately 1.03e-4. This is sampling evidence.
- The chi=60 DMRG product at cap 600 still returns MaxSweeps at 20. The chi=80 cases at caps 160 and 640, and chi=100 at caps 200 and 800,
  also retain MaxSweeps at 20. All three/full-run prefix checks pass. In the
  chi=60 final pass, 25 of 49 updates reuse a cross, 14 reject its full residual,
  and 10 discard it because its rank exceeds fresh LUCI. Removing the rank
  ceiling also fails to converge, so these counts do not establish a missing
  rank-floor policy. The remaining work concerns replacing insufficient
  projections while preserving accuracy and nested gauges, rather than
  changing the stopping rule.
- Local checks pass: Core library (502 passed, one existing ignored), TreeACI
  library (214 passed, 10 existing ignored), train ACI (88 passed, one existing
  ignored), TreeTCI (75 passed), and both G0 convergence regressions including
  R=9. The R=9 check uses release because its contraction workload is costly
  unoptimized. New Core doctests, changed-crate Clippy including production
  panic paths, rustdoc, provider-inject compilation, public error docs, crate
  boundaries, and the repository-rules dry run pass. Hosted CI remains unrun.
  No tests or accuracy thresholds are removed or weakened.
- Only reference-repository MPS data is used. No RSI algorithm is executed.
  These experiments establish numerical behavior, not a performance claim.
