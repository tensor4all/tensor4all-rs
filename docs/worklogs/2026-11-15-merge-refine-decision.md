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
- `tensor4all-quanticstransform --features partitioned --test qft_reconstruction`:
  green, including two decision-boundary tests. `…stops_at_the_root_when_the_merged_
  candidate_fits_the_goal` is the case the old rule split unnecessarily (it now
  keeps one region at the goal instead of two). `…stops_above_the_goal_when_the_split_
  stores_more` pins the guard's reject direction.
- Consequence of `ρ = 1` in the existing small-`r` test regime: a projected child
  stores about as much as the merged candidate, so a split whose retained rank only
  halves is refused (measured: `r = 4`, goal 1, `rtol = 1e-2`, the two children store
  1.5x the candidate, so it keeps one region above the soft goal instead of two). The
  same regime also has a case where the split does pay, when the region holds more
  peaks than one child (four peaks at `r = 6`, goal 1: two regions, rank one).
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
