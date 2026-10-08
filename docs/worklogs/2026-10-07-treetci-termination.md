# TreeTCI stopping diagnostics

## Decisions

- First batch: #835 and #834. Stopping diagnostics are useful to downstream
  bond-dimension schedules and independent of the pending core numerical fix
  in #819. Correctness follow-ups #779 and #812 remain higher-impact work;
  proposer RNG and continuation (#824/#833) should be coordinated afterwards.
- Replace return tuples with named results carrying an explicit reason.
  Preserve stopping order, convergence window, and global-search scheduling;
  this change reports existing decisions rather than changing trajectories.
- Keep the public contract and upstream comparison in
  [the design record](../design/treetci-termination.md), replacing local paths
  in both the implementation and its related regression-test comment.

## Verification conclusions and constraints

- The treetci and quanticstci unit/integration suites and doctests pass. New
  regressions cover all five Rust entry points with real/complex values,
  enabled/disabled global search, limit precedence, and the confirmation
  window. A seven-site dense comparison distinguishes an incomplete single
  iteration from a converged run; a constrained local proposer verifies that
  accepted global pivots prevent sampled convergence.
- Strict changed-crate Clippy, warning-free rustdoc, public error docs,
  deterministic repository-rules review, and relative-link checks pass.
  Nextest is unavailable, so the suites use `cargo test`.
- The prototype Python adapter compiles against the named result. Its Python
  return tuple is unchanged; runtime pytest validation is unavailable because
  pytest/maturin are not installed. The standalone benchmark call site is
  migrated, but no performance measurements are part of this batch.
- Existing rank/error trajectory regressions pass. No algorithmic branches
  or existing tests were removed, and no numerical tolerances were relaxed.
- `Converged` reports the sampled criterion, not a global accuracy guarantee.
  No performance improvement or resolution of #779/#812 is claimed.
