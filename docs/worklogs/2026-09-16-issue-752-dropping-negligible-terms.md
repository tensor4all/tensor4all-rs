# Issue #752: dropping negligible contributions under the global policy

Session: 2026-09-16 (eighth batch). Base: `2d1b962b` (squash of #761). Branch:
`feat/752-drop-negligible`. Closes plan section 6.5: "Drop negligible objects only
under the global error policy ... Discarding a term costs its measured norm in
addition to inherited error already carried by the work item."

## What changed

- A work item now carries a `dropped` bound next to its terms, because a dropped
  contribution has no term to attach its error to.
- `merge_items` drops a paired or unpaired contribution only when its measured norm
  fits that item's share of the global allowance. Dropping charges the inherited
  bound plus the measured norm to the item and to the level's spend, so the
  conservative l1 argument is unchanged: a level's spend now covers accepted
  residuals *and* dropped norms.
- The final assembly adds each item's dropped bound to the region it belonged to.
  A region whose terms were all dropped is omitted from the result, because it has
  nothing to expose, while its bound still enters `error_bound`: it joins the
  Euclidean combination as a region-scale bound rather than being forgotten.
- `MergeRefineReport::dropped_terms` and `dropped_error` report how many
  contributions were removed and how much of the bound they account for.

## Verification

- 123 crate tests plus 52 doctests. The new regression uses a three-bit state whose
  upper half is negligible: a loose tolerance drops the merged contributions that
  cover only that half (`dropped_terms > 0`, `dropped_error > 0`) and the reported
  bound covers the deviation from the exact trajectory, while a near-exact tolerance
  drops nothing (`dropped_terms == 0`) and reproduces the exact result. Every earlier
  bound, adaptive, cancellation, nonuniform-tree, and work-limit test still passes.
- Release Clippy `--all-targets --no-deps -- -D warnings`, `cargo fmt --check`, the
  runnable example, the mdBook snippet suite, and the repository-rules dry-run
  review.

## Limits

Dropping is deliberately conservative: it requires the *measured* norm to fit the
level share, and it never applies when the share is zero, so an exact-tolerance run
removes nothing. Multi-coordinate groups remain the last open step-5 item and need an
explicit axis schedule plus a per-axis output placement that the current one-axis
operator contract does not define; automatic padding still awaits its
domain/embedding contract.
