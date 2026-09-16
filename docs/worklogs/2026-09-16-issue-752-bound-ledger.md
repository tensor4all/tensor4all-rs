# Issue #752: counting each measured error component once

Session 2026-09-16 (ninth batch). Base: `c2f8e8fa` (squash of #762). Local branch
`feat/752-ledger-tightening`, held for review rather than pushed, because the change
touches a contract-visible number and the previous batch was criticized for opening
too many PRs.

## Problem

The reported bound copied an error component into every refined child region and then
combined the per-region sums by the Euclidean norm, so one leaf's application error
was counted once per region it spread into. Measured with `I^⊗r + 0.1·X^⊗r` truncated
to bond dimension one, the bound exceeded the true L2 deviation by 3.9x, 7.2x and
12.8x at two, three and four levels, i.e. it grew like `2^(levels/2)`.

## Change

Every component is measured where it happens and recorded once, grouped by the level
and the region that measured it:

- application errors belong to level zero, which has the single root region;
- a truncation residual or a dropped norm belongs to the child region it was accepted
  in, at that level.

Components of one level live in disjoint regions, so their disjoint supports make the
Euclidean norm the correct combination; components of different levels can be nested,
so they add by the triangle inequality. `error_bound` is the sum over levels of the
Euclidean norm over the regions of that level. The same probe now reports 1.9x, 2.5x
and 3.2x, and the absolute bound is two to four times smaller.

The change also deletes the machinery it replaced: `Term`, `Item::dropped`, the
per-region bound, and the residual-spend counter. Budget allocation is unchanged in
substance (level shares from what the application error leaves, spent by what the
level measures, including discarded probes), so `error_bound <= absolute_tolerance`
follows from the same argument; the level arithmetic now clamps floating-point
rounding at the boundary instead of failing a run for it.

## Verification

- 124 crate tests plus 52 doctests. The new regression `merge_refine_schedule_bound_
  does_not_grow_with_the_level_count` asserts both validity (the deviation stays
  within the bound) and tightness (the bound is at most 3x at two levels and 4x at
  three), which the previous accounting failed at 3.9x and 7.2x.
- Every existing deviation-within-bound, adaptive, cancellation, dropping, sparse,
  nonuniform-tree and work-limit regression passes unchanged.
- Release Clippy `--all-targets --no-deps -- -D warnings`, `cargo fmt --check`, the
  runnable example, the mdBook snippet suite, and the repository-rules dry-run
  review.
- Two stale public claims in the same guide paragraph were corrected here: the
  sentence describing the old per-region combination, and a follow-up list that still
  called nonuniform geometry and benchmark evidence pending after #759 and #761
  landed them.

## Limits

The remaining looseness (1.9x to 3.2x, growing slowly with the level count) comes from
adding different levels by the triangle inequality although some are disjoint.
Attributing residuals to the final regions they touch would allow the Euclidean
combination there as well; that is recorded as the remaining follow-up rather than
approximated. Multi-coordinate groups and automatic padding are unchanged blockers.
