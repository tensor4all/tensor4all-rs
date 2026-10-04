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
- The `l2_given` and `tau` helpers moved from `tests/adaptive_l2.rs` into
  `tests/adaptive_common`, next to the shared `never` evaluator, so the new
  test file does not duplicate them.

## Verification conclusions and constraints

All crate tests pass unchanged with the defaults, including the frozen M2
golden outputs and every M3 test; the new file has 20 tests whose expected
values are closed forms (or the independent stream implementation for the
sampled cases) and were not tuned to the output. Workspace doctests of the
crate and `./scripts/test-mdbook.sh` pass. Clippy reports nothing new; its six
warnings are in the pre-existing M5 fixed-depth benchmark example.

No performance or overpatching measurement was made. The bounds' effect on
patch counts, evaluations, and stored size at downstream scale is part of
the remaining M5 measurements, and no default value is proposed.
