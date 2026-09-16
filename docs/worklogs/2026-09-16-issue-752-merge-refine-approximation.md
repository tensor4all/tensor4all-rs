# Issue #752: controlled approximation in the merge-refine schedule

Session: 2026-09-16 (second batch). Base: `dcb56ac5` (squash of #755, which added
operator-node merging and the exact merge-refine schedule). Branch:
`feat/752-merge-refine-approximation`. Implements plan step 3, "Controlled
approximation: local residuals, inherited-error ledger, conservative global
budgets, report fields, zero/tight-tolerance and fanout regressions".

## What changed

- `MergeRefineOptions::target_bond_dim: Option<usize>` (default `None`, `Some(0)`
  invalid) is the soft rank goal. It reuses the greedy engine's rank-goal
  vocabulary and its `None` default, so every previous trajectory is unchanged.
- Each live work item carries a measured deviation bound. Restriction
  (`SubDomainTreeTN::project`) is nonexpansive, so an inherited bound transfers to
  the child without being charged against local budget and without the invalid
  `delta / sqrt(fanout)` split.
- A merge measures its local residual against the sum of the *actual restricted
  parent approximations*, exactly as the plan requires, and carries
  `e_left + e_right + e_local`.
- Budgets are a conservative l1 split: at most `2^d * output_depth` merges can
  truncate, so each gets `delta / (2^d * output_depth)`. The total of all measured
  residuals therefore stays inside the allowance with no orthogonality assumption
  between separate compressions. The sum of bounds inside a region follows the
  triangle inequality; disjoint regions combine by the Euclidean norm.
- A truncation candidate is accepted only when it strictly lowers the bond
  dimension and its measured residual fits its share; otherwise the exact sum is
  kept at zero cost. A soft rank goal can never force an accuracy violation.
- `MergeRefineReport` gains `compression_attempts`, `compressions`, and
  `max_transient_bond_dim` (largest merged bond dimension before compression), and
  `error_bound` now documents the measured bound.
- The truncation step is shared with the greedy engine as
  `truncate_toward_allowance`, so both paths build the SVD policy, remask the
  candidate, and measure residuals identically. The engine's `compress` keeps its
  drop-under-allowance behavior on top of that helper; the refactor is
  behavior-preserving (all existing engine tests pass unchanged).

Not implemented, matching the plan: dropping whole items under the global policy,
and exploiting the norm-preservation of an exact split into disjoint children
through a shared ledger. The design record states both as the next accounting
steps and keeps the current l1 split as the conservative baseline.

## Verification

- `cargo test -p tensor4all-partitionedtreetn --release`: 113 tests, including 20
  QFT integration tests.
- New regressions: a rank-limited run that truncates inside the allowance and whose
  reported bound covers both the dense transform oracle and the exact trajectory;
  a zero-tolerance run whose probes are rejected for free so the result equals the
  exact trajectory with `error_bound == 0`; a cancellation run with a rank goal that
  keeps zero items and a zero bound; and a `Some(0)` rank goal rejected as invalid.
- 51 crate doctests, release Clippy `--all-targets --no-deps -- -D warnings`,
  `cargo fmt --check`, the runnable example, the mdBook snippet suite and book
  build, and the repository-rules dry-run review.

## Risk and limits

The l1 split is deliberately conservative: it dilutes the allowance by the number
of possible merges, so a tight tolerance can reject affordable candidates. No
performance claim is made; the new counters describe schedule shape and rank
behavior only. The bound excludes backend roundoff and any error from building the
operator outside this crate, so an approximate Fourier operator still needs its own
construction-error accounting.

An incidental measurement during test design: the *exact* four-bit trajectory
(rank-32 merged sums at the last level) dominated the test file's runtime (~39 s
in release) while the same trajectory with a rank goal of two, which truncates 48
merged items inside the allowance, ran in under a second. That is schedule shape
and rank behavior, not a benchmark, but it is the first concrete indication that
the merge-refine trajectory's cost lives in the merged ranks rather than in the
per-leaf transforms.
