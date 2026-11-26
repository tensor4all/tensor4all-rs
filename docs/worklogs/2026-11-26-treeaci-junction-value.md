# Tree ACI locally fixed junctions (#1435)

## Decisions

- Adopt the junction-scoped predicate accepted in
  [#1435](https://github.com/tensor4all/tensor4all-rs/issues/1435#issuecomment-7182530985):
  degree at least 3 and at least two committed pivots agreeing at the local
  coordinate, with the measured incoming-rank slack and neighbour choice.
  Candidate expansion keeps the full local index; agreement is a heuristic.
- Reuse the unit-vertex expansion's limits, transactional rollback and
  cut-back. No environment switch, new public API, target evaluation of retained
  pivots or history replay is needed. See the
  [design](../design/tree-aci.md#local-agreement-at-non-unit-branching-vertices).
- The scheduler regression's forward pass now already spans binary leaf 5
  at rank 2. The injected Guard point grows only `4 -> 1`, so the return walk
  needs that excursion and no longer needs `4 -> 5 -> 4`. The test asserts
  the full leaf basis, the new walk, failure of the old spine-only walk and
  restoration of the pivot basis without relaxing its tolerance. Recursive
  child-first excursion planning remains covered by the path-cover tests.

## Verification conclusions and constraints

- The [accepted prototype measurement](https://github.com/tensor4all/tensor4all-rs/issues/1435#issuecomment-7181759972)
  establishes the case B accuracy/point-count tradeoff and unchanged G0/star
  trajectories. These are evaluation counts, not runtime measurements.
- On base `9b106ef8`, case B reproduces all 11 accepted prototype results:
  ranks (12, 1, 12), dense relative error at most 6.14e-16, and 259,746 points
  versus the base's 213,415. Case A's 11 results are identical to the base,
  including its 232,778 points (the base already includes #1466).
- All 213 G0/star results preserve the base's recorded errors, ranks, pass
  counts and terminations; 212 also preserve point counts on the first run.
  One r=7 cttn row (seed 42, tolerance 3e-4) uses 5,345,576 instead of
  5,329,448 points. A repeat of the same finished candidate binary's entire
  21-case r=7 cttn block returns the lower count; its read-only diagnostic
  build selects no junction expansion in that block. The cause of this
  cross-process count variability is not localized; no deterministic
  point-count identity or runtime improvement is claimed. Both full runs
  and the diagnostic/repeat results were retained, without replacing the
  first observations. All 180 `min_sweeps = 1` rows are identical to the base.
- The 345 ordinary crate tests, five heavy numerical tests, 26 doctests,
  strict Clippy and deterministic maintenance checks pass. The heavy dense
  oracles and r=9 G0 regression use release builds for tractable numerical
  validation. Seven new unit tests cover the predicate and expansion, and
  two public regressions cover three seeds each for f64 and Complex64.
  Disabling the new predicate makes both public regressions fail; restoring
  it returns them to green. No existing tolerance changed.
- The inner-loop review adds only a rank-sized scan of pivot coordinates at
  branching endpoints, with no allocation or tensor evaluation. TreeTCI's
  corresponding junction-scoped predicate already matches this contract.
  The predicate does not certify full-domain accuracy; `Converged` remains
  a sampled stopping condition. Hosted CI has not run for this branch.
