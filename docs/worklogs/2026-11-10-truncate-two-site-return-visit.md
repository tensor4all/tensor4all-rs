# The return visit of the two-site tour in `truncate`

## Decisions

- **The return visit is not removable.** It is the only place where the edge is
  truncated against the sub-tree that the descent has compressed in the
  meantime, and it is where the norm moves back to the parent, which is the
  orthogonality centre the next step needs. Measured per edge on four states, it
  removes 0-2 bond dimensions, but it does remove them: two edges each on the two
  application states (B1: 265->264 and 256->255; IFT: 129->128 and 135->134).
  Dropping it unconditionally therefore changes the result, which is what the
  earlier 416/923 reading seemed to license and the per-edge count does not.
- **The candidate skip rule is safe on the measured states but almost worthless
  there.** "Skip the return visit on an edge whose outbound visit removed no
  rank" never skipped a rank-removing return visit on the four states, and it
  covers 0.1 % of the sweep's SVD time on the plain comb, 4 % on the reporter's
  state, 9 % on B1 and 50 % only on B2, a state that has already been compressed
  once. It is not shipped: the rule is a property of the states rather than of
  the algorithm (a state where truncating a sub-tree lowers the rank across an
  edge above it would break it), and the payload is the cheap steps, not the
  expensive ones - on B1 the rule catches the single most expensive return step
  (edge 57-60, where the outbound visit removed nothing) and misses the next two
  (59-60, 58-60).
- **What the return visit spends its time on is the gauge move, so the gauge move
  was made cheaper.** A two-site truncation has to factorize the block; the
  block's row dimension is the product of the left node's other indices, which on
  a comb is the other branches (up to 1.2e4 here) against a bond of a few hundred.
  `TruncateUpdater` now splits the left operand with an exact QR first (a QR that
  cannot read the operand's storage, or that drops a row, keeps the old route):
  `A = Q R` with `Q` isometric, so `AB = Q (R B)` and `R B` carries the block's
  singular values, its retained rank and its truncated value. The truncating SVD
  then runs on a `bond x columns` matrix. The conditioning is used only when the
  bond is smaller than both block dimensions, which is exactly when it is cheaper,
  and it is skipped when the QR has to drop a row (the operand is rank-deficient on
  the bond), so no rank-decision is taken by the QR.
- The one path stays one path: both directions of the tour go through the same
  `TruncateUpdater::update`, and the conditioned and unconditioned routes share the
  factorization call, the truncation options and the empty-side helper.
- The result is *not* bit-identical: the left factor is `Q U` instead of `U`, an
  equivalent isometry. Bond dimensions and removed units are identical, and the
  state agrees with the unconditioned route to 1e-13 on every measured state.

## Verification conclusions and constraints

- `test_truncate_updater_conditioned_route_matches_the_direct_one` builds a
  two-site block whose bond (8) is smaller than both its sides (32 rows, 16
  columns) and whose eight bond directions have singular values a factor 5 apart
  (`sum_p 0.2^p u_p v_p^T` over DCT-II bases; the block's own spectrum, `A B`, is
  0.667 / 0.267 / 0.0424 / 0.0102 / ... so a 1e-2 relative cutoff keeps four
  directions and the cap of 3 binds), runs the updater, and compares the block it
  produces against the one produced by a single SVD of the contracted pair: same
  bond dimension, same rank, same values to 1e-12 — for a relative cutoff, an
  absolute cutoff, squared values and a bond cap.
  It is an **equivalence test, not a route test**: both routes compute the same
  truncated block, so it also passes with the conditioning switched off. That the
  conditioned branch is entered was measured separately, by a counter inside the
  branch run on the reproducer's four states: A 84 steps, B1 74, B2 160, IFT 0
  (on IFT the bond is not smaller than both sides of the block, so every step
  takes the fallback, as designed). A counter-based assertion would pin the route
  better than this note; it is not in this PR.
- The whole crate suite passes (702 unit tests, 419 + 212 integration tests);
  `cargo fmt --all --check` and `cargo clippy --all-targets` are clean.
- Speedups on the four states, base `a1a02af6` against the branch, two runs per
  side alternating: 2.24x (comb A), 1.50x (B1), 1.72x (B2, an already compressed
  state), 1.26x (the reporter's three-branch state). Truncation accuracy is
  unchanged to the printed precision, and the deviation from the unconditioned
  route is 7e-14 to 2.8e-13. Numbers, harness and the command output that was
  captured:
  [the return visit evidence](../experiments/2026-11-10-treetn-truncate-return-visit.md).
- The wall times were taken on a host at load average 12-26, so the ratios are
  what the numbers support; the bond dimensions and removed units are exact.
- Not covered: production constructs `TruncateUpdater` only in `truncate.rs`
  (`fit`, DMRG and TDVP run their own factorizations), so the change reaches the
  truncation API, which includes the context-scoped `truncate_in` and therefore a
  CUDA-reachable path; no GPU was available here and that route was not exercised.
  A state where the conditioning is not taken (bond larger than both block sides)
  keeps the old route and was not measured; nor was a zero cutoff, where the
  retained rank is a roundoff artifact and the conditioned route caps it at the
  bond instead of letting it grow.
