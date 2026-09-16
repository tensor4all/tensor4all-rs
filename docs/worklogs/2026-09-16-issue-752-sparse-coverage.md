# Issue #752: explicit sparse-leaf coverage contract

Session: 2026-09-16 (sixth batch). Base: `bf2754c2` (squash of #759). Branch:
`feat/752-nonuniform-geometry`. First slice of plan step 5: the plan's requirement
that "sparse/missing leaves represent zero only under an explicit validated coverage
contract".

## What changed

- `MergeRefineOptions::coverage: CoverageContract` with
  `CoverageContract::{Complete, ZeroForMissingLeaves}`, defaulting to `Complete`, so
  every existing behavior is unchanged.
- `Complete` keeps the previous requirement: every `2^d` selected-coordinate
  assignment must be present exactly once, and an incomplete preimage is rejected
  with the repair suggestion to select the other contract.
- `ZeroForMissingLeaves` accepts a sparse preimage. Present leaves must still fix
  every selected index exactly once and agree on their spectator constraints, and
  every omitted assignment contributes exactly zero. An entirely empty preimage is
  the zero target under this contract: the schedule returns no region, the pinned
  scale is zero, and the counters stay at zero.
- The option's rustdoc, the design record, the crate README, the guide, and the skill
  reference document the decision, because an absent leaf is only a zero when the
  caller says so.

## Why this is a bounded slice

Sparse coverage is the part of step 5 that does not require a new geometry engine:
the uniform level structure, the merge order, and the error ledger are unchanged, and
a missing leaf is simply absent from the level-zero map, which the existing
`restrict_and_merge` already handles. What remains for step 5 is genuinely
nonuniform: unequal leaf depths with coarse/deep sibling handling, a real dyadic
input tree instead of the full cover, multi-coordinate groups with a synchronized
multidimensional level, and the corresponding bit-significance tests. Those are not
approximated or half-implemented here.

## Verification

- 119 crate tests plus 52 doctests. New regressions: a sparse two-bit preimage whose
  omitted leaf is zero is accepted under `ZeroForMissingLeaves` and matches the
  complete run exactly, while the default `Complete` contract rejects the same
  preimage with the repair guidance; and an entirely empty preimage returns the zero
  target with zero counters.
- Release Clippy `--all-targets --no-deps -- -D warnings`, `cargo fmt --check`, the
  runnable example, the mdBook snippet suite, and the repository-rules dry-run
  review.
