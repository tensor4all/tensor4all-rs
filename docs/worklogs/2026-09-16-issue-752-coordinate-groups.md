# Issue #752: coordinate axes and the padding decision

Session 2026-09-16 (tenth batch), batched with the error-accounting tightening into
one change as requested. Base: `c2f8e8fa` (squash of #762). Branch
`feat/752-ledger-tightening` (the branch now carries both changes).

## Coordinate axes

`MergeRefineOptions::coordinate_groups: Option<Vec<CoordinateGroup>>` lets a caller
transform several coordinate axes in one synchronized level. Each `CoordinateGroup`
lists its axis' selected indices in input significance order and, separately, in
output significance order, because the one-axis convention (reversing the whole
selected node order) does not define a multi-axis schedule on its own. The plan
requires exactly that explicitness: "axis schedules must be specified rather than
inferred from node order".

- The geometry is now per-axis: input and output regions are keyed by a
  `(prefix length, prefix value)` pair per axis, one level advances every
  non-exhausted axis by one bit, so two axes combine four input children and produce
  four output children, and the grandchildren that reach the same parent input key are
  merged pairwise as they arrive with each sum measured individually.
- Validation is per axis: a patch must fix a contiguous prefix on every axis, leaves
  must not overlap (no leaf's per-axis prefixes may all be prefixes of another's), and
  the coverage contract is the joint Kraft sum over the summed per-axis depths.
- `None`, the default, is one axis whose output order reverses its input order, so
  every earlier behavior and test is unchanged (124 crate tests and 52 doctests passed
  unchanged through the refactor).
- Two regressions: a two-axis schedule over a composed product operator (four input
  children per level, four regions after one level, the fully refined strict partition
  matching the two one-dimensional transforms applied in sequence), and the rejection
  of groups whose outputs are not their inputs or that do not assign every selected
  index exactly once.

The composition path was verified separately before implementing this: one-node `X`
and `Z` composed with `compose_exclusive_linear_operators` reproduce the dense `X ⊗ Z`
exactly, which is why a two-axis operator can be built from per-axis one-dimensional
operators in the regression.

## Padding decision

Automatic zero-padding is **not provided**, and an operator whose input and output
bit counts differ is out of scope: the entry points require the operator's output legs
to be the selected input indices, so `replace_mapping_true_indices` rejects a differing
output dimension. The state-side padding proposal is withdrawn; a caller that needs a
different output space owns that embedding outside this API, and the library never
pads silently. The guide, design record, README, and skill reference now say so, and
the design record's "padding needs an explicit domain/embedding contract" sentence was
replaced accordingly.

## Verification

- 126 crate tests plus 55 doctests, including the two new coordinate-axis regressions.
- Release Clippy `--all-targets --no-deps -- -D warnings`, `cargo fmt --check`, the
  runnable example, the mdBook snippet suite and HTML build, and the repository-rules
  dry-run review.

## Limits

Per-input-branch refinement depths with lazy reconciliation and benchmark evidence in
the intended patched-input regime remain follow-up work, as does attributing residual
components to the final regions they touch so that residuals of nested levels could
combine by the Euclidean norm instead of the triangle inequality.
