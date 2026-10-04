# Interpolation checkpoint review

## Decisions

The M1–M3 and M5 interpolation line was reviewed as one integrated branch,
starting from `e91038683fb4a604840d28f5dd38d9bb399ab97f` against its merge base
with `origin/main`. M4 remains deferred. The checkpoint found two major
boundary defects, which are corrected without changing defaults or the
acceptance/error contract:

- Dimension-derived exact point lists, candidate lists, and split child
  collections now reserve fallibly. Unrepresentable byte lengths return
  typed input/problem errors rather than panicking or attempting an
  impossible enumeration. Candidate targets are checked after clamping to
  the patch domain, so `usize::MAX` remains usable on a small domain.
- The TreeTCI engine validates scalar components and magnitudes in every
  evaluator batch, including sweeps and materialization. A finite initial
  pivot followed by NaN can no longer produce a `Converged` exact tensor
  with error zero. Diagnostics preserve evaluator classification and identify
  the failing batch point and its coordinates.

The allocation defect prompted an adjacent audit of candidate growth and
cache splitting; their reservations use the same driver helper. Tests cover
the rejected boundaries and preserve exact small-domain behavior, cached
sample transfer, and the frozen M2 outputs. No tolerance was relaxed.

Documentation distinguishes resolved TreeTN construction ordering (#791)
and the closed tracking/test gap (#795) from the remaining generic evaluator
planner limitation (upstream tenferro-rs#1963). The sampled corner-miss and
generic-path determinism regressions remain ignored with their existing
reasons. The consolidated [follow-up list](../design/tree-patching-findings.md#7-interpolation-checkpoint-follow-ups)
records milestone ownership; it does not authorize those deferred tasks.

## Verification conclusions and constraints

The original candidate passed the affected-crate suites and doctests (1527
tests, 18 explicitly ignored), guide examples, formatting, and maintenance
gates. The quantics caller affected by the M1 optimizer report change also
passed its crate suite and doctests (98 tests). The boundary reproductions independently confirmed the immediate
TreeTCI capacity panic and the NaN `Converged` result before the corrections.
Clippy reported only five existing warnings in the fixed-depth benchmark.

The corrected candidate passed the affected two-crate suites and doctests
(517 tests, 12 explicitly ignored), guide examples, formatting, and maintenance
gates. The quantics caller affected by the M1 optimizer report change also
passed its crate suite and doctests (98 tests). The
independent re-review resolved both major findings and found no new blocker,
major, minor, or nit. Removed validation paths were reviewed for coverage
impact: non-finite checks moved into the shared evaluator wrapper and remain
covered at initial and later batches. No
new static performance violation or measured performance claim was established;
downstream-scale measurements remain with M4/M9. Existing sampled-estimate and
calibrated-rounding limitations are unchanged. The branch is nine commits
behind `origin/main`; synchronizing and revalidating that integrated candidate
remain prerequisites before proposing a merge. No PR is part of this checkpoint.
