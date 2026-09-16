# Issue #752: adaptive refinement in the merge-refine schedule

Session: 2026-09-16 (third batch). Base: `2950e531` (squash of #756). Branch:
`feat/752-adaptive-schedule`. Implements plan step 4, "Adaptive execution: lazy
prefix refinement, no-gain stop, merge-gain lists, resource limits, mismatched
output refinements; spike/comb tests".

## What changed

- A work item is now a superposition: a list of retained terms, each with its own
  measured bound and its input prefix. The region decision stays per output region
  rather than per input branch, which keeps the final region set prefix-free and
  removes the need for a post-hoc reconciliation of mismatched depths; the plan's
  "mismatched output refinements" case therefore does not arise in this design.
- An output region is refined only when its retained terms exceed the rank goal and
  refining strictly lowers its maximum retained rank. Without that gain the region
  stops with the input prefixes it already holds, and the discarded probe's
  residuals are never charged. Without a rank goal the trajectory is unchanged: it
  stays uniform and exact, so every earlier test keeps its meaning.
- A merged pair becomes one term only when the combination strictly lowers the
  bond dimension against the naive sum; otherwise both operands stay as separate
  terms. This is the plan's merge-gain list policy and it is what keeps a rank-one
  object out of a preset output tiling.
- `MergeRefineOptions::max_terms` (default `1 << 20`) bounds the retained term
  count across live and finalized regions and reports the new
  `PartitionedTreeTNError::ResourceLimit` variant when exhausted. A budget that
  cannot be kept is an explicit error rather than a silent accuracy relaxation.
- `MergeRefineReport` gains `refined_regions` and `stopped_regions`, and the
  final assembly groups terms by output region so `regions()` exposes a
  superposition and `into_partition()` correctly rejects one.

## Verification

- 115 crate tests plus 51 doctests, including new adaptive regressions: a spike
  input that reaches rank one after one level and therefore stops at two regions
  instead of the preset `2^r`, and a two-term budget rejected with
  `ResourceLimit`. The earlier bounds, zero-tolerance, cancellation, and
  intermediate-oracle tests still pass with the adaptive policy.
- Release Clippy `--all-targets --no-deps -- -D warnings`, `cargo fmt --check`,
  the runnable example, the mdBook snippet suite and book build, and the
  repository-rules dry-run review.

## Decisions and limits

The per-region (rather than per-branch) refinement decision is a deliberate
simplification: it removes the mismatched-depth reconciliation the plan lists, and
it still satisfies the plan's goal that a simple object is not forced into `M`
output blocks. Per-branch depths would need a lazy common-output-region
reconciliation and are left as follow-up.

Dropping whole items under the global error policy is still unimplemented; the
current policy only keeps, compresses, or splits terms, so the reported bound never
depends on a term that was removed. Input coalescing is not performed either: a
region that stops early keeps its unmerged input prefixes, which is what keeps its
terms small, and callers consume them through `regions()`.
