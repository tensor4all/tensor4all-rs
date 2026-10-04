# M5 patch-size bounds implementation

## Decisions

The [patch-size plan](../design/tree-pqtci-patch-size-bounds.md) was
implemented with every proposal of its open issues. The capped bound is the
option (c) of open issue 14: `capped_patches: CappedPatches` with the
variants `Split` (default) and `AcceptUpTo { bits }`, set by
`with_capped_patches`. The minimum is `min_patch_bits: Option<usize>`, set by
`with_min_patch_bits`. Both count generalized bits, one per active site.

A capped bound below the minimum is rejected as `InvalidInput` (open issue
13), validated after `max_patches` and before the verification options, so
the documented order of the other checks is unchanged. A network that the
driver would use for a capped-eligible or retained run and that exceeds the
cap or does not match the layout is an engine error (open issue 10); the
`InterpolationOutcome::network` rustdoc now states that bonds never exceed
the cap whatever the termination. `TreeTciInterpolator` already satisfies
this.

Open issue 1 remains provisional. The code retains a blocked patch whose
last run did not converge by measuring it once, and its rustdoc describes
that rule. The user's decision may still change the wording or the rule.

The policy (size classification and the retention of a blocked patch) lives
in the private module `adaptive_interpolation/acceptance.rs`, so the driver
file does not grow further. The engine loop now records whether its last run
was judged; `resolve` replaces `split` and chooses between splitting,
retaining, and the M3 errors. `VerificationFailed` is chosen when the last
run was judged and measured, which replaces the M3 test "last run converged"
and keeps `a_capped_rerun_after_a_failed_verification_keeps_no_split_index_left`
unchanged.

`verify::global_error` keeps the M3 classification in a private `classify`
and wraps it into `GlobalL2Error::ToleranceNotMet` when a contribution missed
its allowance. The audit option is global, so sampled contributions either
all have audits or none, and "best available measurement per contribution"
equals the M3 classification.

## Deviations from the test plan

- Tests 4 and its `IterationLimit` variant, test 12's four cases, and test
  15's three layouts share one test function each.
- Test 21 has two scenarios. With `max >= min`, a blocked capped TreeTCI run
  is always judged already, so the minimum acts only through
  `ToleranceNotMet` patches (or `IterationLimit`, which TreeTCI rarely
  reaches here). The first scenario (cap 2, minimum 4, bound 5) exercises
  both bounds and compares `measured_rms` with the dense residual; the second
  (cap 3, minimum 3, bound 5) exercises capped acceptance with every patch
  within tolerance and checks the M3 certificate against the dense error.
  The parameters were chosen by trying a small grid and are recorded in the
  test.
- Open issue 10 proposed a test engine that truncates at the cap. It was
  not added: the dense test engine's untruncated over-cap network is exactly
  the engine fault the new engine-error path needs, and every other test
  splits such a patch without using it. The engine's documentation says so.
- The `l2_given` and `tau` helpers moved from `tests/adaptive_l2.rs` into
  `tests/adaptive_common`, next to the shared `never` evaluator, so the new
  test file does not duplicate them.

## Review

An independent review found no blocker or major issue. Its minor findings
were fixed: `classify` returns the M3 RMS value and basis, so no unreachable
match arm remains; a test covers a capped-eligible run that fails its
estimate at an exhausted `patch_order` under `SampledMax`
(`NoSplitIndexLeft`); a passing retained measurement is asserted not to count
as a verification failure; the test engine documentation and the
`Certified` rustdoc match the amended contract; several rustdoc wordings
were corrected.

## Verification conclusions and constraints

All crate tests pass unchanged with the defaults, including the frozen M2
golden outputs and every M3 test; the new file has 20 tests whose expected
values are closed forms (or the independent stream implementation for the
sampled cases) and were not tuned to the output. Workspace doctests of the
crate and `./scripts/test-mdbook.sh` pass. Clippy reports nothing new; its five
warnings are in the pre-existing M5 fixed-depth benchmark example.

No performance or overpatching measurement was made. The bounds' effect on
patch counts, evaluations, and stored size at downstream scale is part of
the remaining M5 measurements, and no default value is proposed.

## Question 5: cache candidates

`PatchedInterpolationOptions::cache_candidates` (default off) adds, for
every patch, the at most `n_initial_pivots` points of its inherited cache
with the largest nonzero `|f|` to its start candidates, after the user,
recycled, and worst points and before the random fill. The cache already
holds every value the parent evaluated inside the patch, so no evaluation is
added. `PatchCache::largest_points` orders equal magnitudes by coordinates,
so the choice does not depend on the hash-map order or the key packing.

The first proposal, changing only the split coordinate of parent points,
was dropped before implementation: flipping a quantics bit moves a point by
half the patch rather than across the split face, and a true reflection
needs the variable structure, which the driver does not know.

On the corner-miss reproduction (release build, seeds 2, 3, 4) the true
`E / delta` fell from 680, 880, 880 to 51, 230, 16 with the option, and no
rank-1 never-sampled patch was accepted; misses of the sampled acceptance
remain, so the option mitigates and does not fix the limitation. Both
reproductions stay ignored tests that fail while the limitation exists.

With question 5 implemented, question 3 moved to M6, and the early exit
closed, M5 is complete by its completion rule; the checkpoint review of the
interpolation line is next.
