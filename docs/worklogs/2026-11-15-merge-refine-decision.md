# Merge-refine split decision compared against the merged candidate

## Decisions

- The rank-goal decision of `schedule_merge_refine` now compares the candidate a
  stopped region keeps — the merged, unsplit region — with the merged split children,
  and stops as soon as that candidate fits the goal. The old test compared the
  pre-merge items' rank with the post-merge children's rank, so it could stop a
  region exactly when the split would have restored its rank to the goal, and because
  a stop is one-way the region could never refine again.
- The useless-split guard is the total stored parameters of the kept children against
  the merged candidate's, i.e. the issue's `Σ_children ≤ ρ · Σ_unsplit` with a fixed
  `ρ = 1` ("refine only if storage does not grow"), not a new option. A projected
  child keeps every output site axis, so the children of one split store almost as much
  as the candidate each; a rank drop by one therefore does not pay for `2^axes` regions.
  Rejected: a public growth factor (API shape, and the issue leaves its default to a
  measurement programme outside this change) and the per-region variant
  `Σ_children ≤ k · Σ_unsplit` (a tie then always refines, which is the coarse
  behaviour the issue removes).
- The charge accounting keeps the existing invariant: a kept result records its
  measured component at its own region, a discarded probe is charged to the level's
  share but not reported. `level_charge_bound` already reserves
  `own merge + one probe per child`, which covers the new order unchanged.

## Verification conclusions and constraints

- `tensor4all-partitionedtreetn`: full crate suite green.
- `tensor4all-quanticstransform --features partitioned --test qft_reconstruction`
  (the file is `#![cfg(feature = "partitioned")]`; without the feature it compiles to
  no tests): green, including three decision-boundary tests.
  `…stops_at_the_root_when_the_merged_candidate_fits_the_goal` is the case the old
  rule split unnecessarily (it now keeps one region at the goal instead of two).
  `…stops_above_the_goal_when_the_split_stores_more` pins the guard's reject
  direction. `…refines_when_the_split_stores_no_more_than_the_candidate` pins the
  accept direction.
- The accept-direction test was first written on four unit peaks at `r = 6`. That
  input is a product state whose DFT is a product state, so its exact merged candidate
  is rank 1 and already fits the goal; it split on x86_64 only because rounding left
  a bond above the tolerance, and it stopped at rank 1 on macOS/arm64 (Accelerate
  backend). With point leaves and a goal of 1, a merge that does not fit the goal
  keeps its two rank-1 operands, so the root candidate is rank 1 in exact arithmetic
  and no point-leaf input can exercise the accept direction. The test now uses
  `w(x) = 1 + 0.6i e^{-2 pi i x / N}` split into its two input halves: the exact DFT
  is two deltas differing in every output bit, so the merged root candidate is exactly
  rank 2 (second singular value `0.6 sqrt(N)` against an allowance of about `1e-8`)
  and each child is exactly rank 1. Measured on x86_64 with a temporary print in the
  guard (not committed): candidate 40 stored parameters at rank 2, children 24 at
  ranks 1 and 1, matching `4 + 8 (r - 2) + 4` and `2 * 2r`.
- Consequence of `ρ = 1` in the existing small-`r` test regime: a projected child
  stores about as much as the merged candidate, so a split whose retained rank does not
  drop is refused. Measured on x86_64 with the same temporary print: `r = 4`, goal 1,
  `rtol = 1e-2`, candidate 192 parameters at rank 2 against 288 for two rank-2
  children, so the region stays above the soft goal with one region instead of two.
  The split pays when restricting to a child lowers each child's rank enough to cover
  the extra region (the accept test above).
- The soft goal decides splitting, not truncation: a region that stops on the goal
  keeps truncating toward its share of the allowance on every later level, so a
  retained rank above the goal is expected and bounded, not a contract violation;
  `max_work_items` and `max_terms` remain the backstop.
- Fail/pass against the old decision (tests run against `origin/main`'s scheduler with
  the new tests kept): `…stops_at_the_root_when_the_merged_candidate_fits_the_goal` and
  `…stops_above_the_goal_when_the_split_stores_more` both obtain
  `refined_regions = 1` from the old rule and pass with the new one, which reports
  `0` for both.
- Not verified here: the issue's planned 2D tight-binding measurement for the default
  growth factor, and any large-rank regime where a split does reduce total storage.
  The growth factor stays an open question for the maintainer.

## Review

- Plan review before implementation: `ollama-cloud/glm-5.3-flash:high`
  (`reviewer-glm`, read-only) — `VERDICT: APPROVE`, no blocker; its two IMPORTANT
  findings (do not state an unmeasured outcome for the r=3 boundary case; include the
  two goal-using `patched_fourier` tests in the re-derivation list) and two MINOR
  findings (two comments describing the old probe order) are addressed in this change.
