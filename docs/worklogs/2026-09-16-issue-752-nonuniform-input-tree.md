# Issue #752: nonuniform dyadic input trees

Session: 2026-09-16 (seventh batch). Base: `eb73c585` (squash of #760). Branch:
`feat/752-nonuniform-leaves`. Second slice of plan step 5: "Support nonuniform input
prefixes with a real dyadic tree, not arbitrary adjacent-term pairing. Merge only
genuine siblings/unions in the selected geometry." plus "Specify coarse/deep sibling
handling explicitly".

## What changed

- Input regions are keyed by `(prefix length, prefix value)` instead of a single
  uniform-depth prefix integer, so leaves may have different depths.
- A patch support is read as a *contiguous prefix* of the selected indices. A
  support that fixes a later selected index while leaving an earlier one free is
  rejected, because the level structure cannot merge it.
- Validation became a dyadic prefix-code check: leaves must not overlap (no leaf may
  fix a prefix of another leaf's indices) and the coverage contract is a Kraft sum in
  integer units of `2^-depth`, which distinguishes an overlap from a missing
  assignment exactly. `Complete` requires the sum to be exactly `2^d`;
  `ZeroForMissingLeaves` accepts a strict subset.
- Every item ascends one selected bit per level. A sibling pair is summed; a leaf
  whose sibling is absent keeps its own region, which is already the union of its
  subtree, so ascending alone is exact and free. This is the plan's "coarse/deep
  sibling handling": no re-application of the transform and no invented partner.
- Because each refined region keeps its own copy of the still-unmerged prefixes, the
  live item count can grow with the number of regions on nonuniform trees. The work
  limit is therefore enforced on every level and reported as
  `PartitionedTreeTNError::ResourceLimit`, and the l1 budget moved from a single
  `2^d * output_depth` split to a per-level split of `remaining / level_merges` with
  `remaining` reduced by the residuals actually accepted. The derivation is
  unchanged in spirit (each accepted residual is at most its level's share, and the
  spend cannot exceed the shares), but it now covers merge counts that
  `2^d * output_depth` does not bound.

## Verification

- 122 crate tests plus 52 doctests. New regressions: the asymmetric three-bit tree
  `{0, 10, 110, 111}` merges to the whole DFT with `4` applications, `14` additions
  (2 + 4 + 8, repeated per refined region), and a live-item trajectory of
  `[4, 6, 8, 8]`; a sparse single-prefix preimage is rejected under `Complete` and
  matches the DFT of the masked state under `ZeroForMissingLeaves`; and a work limit
  of five is reported as `ResourceLimit { value: 6 }` instead of being silently
  exceeded.
- Release Clippy `--all-targets --no-deps -- -D warnings`, `cargo fmt --check`, the
  runnable example, the mdBook snippet suite, and the repository-rules dry-run
  review.

## Limits

Multi-coordinate groups with a synchronized multidimensional level are still
follow-up: this change makes the input tree nonuniform along one ordered axis, not
multi-axis. Per-input-branch refinement depths with lazy reconciliation and item
dropping under the global policy also remain open, and automatic padding still awaits
a domain/embedding contract decision.
