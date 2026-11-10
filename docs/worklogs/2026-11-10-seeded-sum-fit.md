# Seeded sum fit (#1129)

## Decisions

- Compose the existing pairwise seed and variational sum fit; keep exact sum
  engines unchanged. Both pairwise orders are selectable. The density-matrix
  seed is deferred to #1126; the reported fit accepts any caller-built seed.
- Keep `fit_sum`'s existing return type and cost. A reported entry point uses
  the same implementation, reconstructing only each local two-site split to
  measure its discarded norm. Reports are in sweep order, with node names.
- Fit diagnostics describe local truncation loss, not an error certificate
  against the exact sum. No new fitting or dense whole-network algorithm.

## Verification conclusions and constraints

- Regression checks cover recovery of seed loss, regional discarded norms,
  deterministic branching results and ranks, zero sweeps, one-node networks,
  invalid options, and the R=12, N=64 random family from #1107.
- All six new tests were killed by reverted production mutations: skipping
  the fit breaks recovery, branching and the heavy accuracy test; reporting
  zero loss breaks the regional-norm test; forcing a sweep breaks the
  zero-sweep test; bypassing validation breaks the invalid-options test.
- The pairwise seed's center is now validated before reduction, including a
  singleton. The extended owner regression failed before the fix and when its
  center check was temporarily disabled.
- The R=12, N=64, D=10 f64 family reaches max-absolute error 4.82e-11 versus
  1.04e-10 for exact-then-truncate on macOS Accelerate, below the documented
  factor of 10. This family is small enough for a dense test oracle; production
  fitting remains local.
- The complete ordinary treetn suite passes: 1129 tests, with three existing
  or heavy tests ignored. The new heavy regression also passes separately in
  release mode because its bond-640 exact reference needs optimization.
- Strict local Clippy is blocked by existing `nonminimal_bool` diagnostics
  (43 in core; 20 in treetn with `--no-deps`). Allowing that single lint only
  for the local command checks all remaining treetn warnings; repository lint
  policy is unchanged. The fetched main revision's hosted CI is green.
- Public error-doc checks and the deterministic repository-rules preview pass.
  The file-growth reminder is advisory: the fit module retains its shared
  update and report ownership instead of splitting the fitter implementation.
- Peak RSS is not measured. The density-matrix seed remains dependent on #1126.
  The guide edit is prose only; local mdBook is unavailable, and hosted
  documentation CI owns the complete guide execution check.
