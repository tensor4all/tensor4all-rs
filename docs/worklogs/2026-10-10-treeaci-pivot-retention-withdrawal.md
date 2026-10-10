# Withdraw TreeACI pivot retention (#885; reopen #784)

## Decision and scope

The user requested removing the severe performance overhead and reopening the
original cycling issue instead of retaining a costly repair. Withdraw both
complete previous-cross retention and partial preferred-pivot restoration from
TreeACI. Every edge update uses fresh LUCI at the original tolerance and rank
ceiling. Remove the now-unused prescribed/preferred-cross Core APIs and the
Guard preference-refresh flag; no fallback option retains the expensive path.

Preserve the independent numerical/resource repairs, per-edge growth window,
Guard search, injection and revalidation, and stable scalar-network skeleton
construction. The #874 real/complex public fixtures keep their original
accuracy, pass-count and evaluation-count assertions. The former refresh-flag
lifecycle test now checks that injected candidates survive a failed pass and
fresh LUCI can subsequently reduce rank. New all-scalar/orientation controls
reconstruct a known near-threshold perturbation with the fresh cross.

## Evidence and limits

The original complete R=8 profile found preferred search in 177/274 native
samples (64.60%), exceeding the registered 40% need threshold. Priority-group
screening failed its aggregate gate (527.205 -> 392.317 seconds; ratio 0.744,
required <= 0.70). A second candidate retained exact outputs/histories in all
15 archived controls but remained far slower than the pre-umbrella run:
R=8 chain 69.216 vs 9.902 seconds, R=9 chain 57.229 vs 33.170, and warm R=9
11.534 vs 1.132. All 12 completed observations are retained; the user cancelled
its remaining cases when requesting withdrawal. Neither candidate is promoted.

Issue #784 is reopened. Withdrawal does not repair tolerance-boundary cycling
and does not reinterpret `MaxSweeps` as success. Historical #870/#875 logs
retain their original results with an explicit withdrawn-status note.

The independently registered withdrawal study is pending. It requires all
eight candidate G0 cases to converge, pass exact full-grid checks at unchanged
settings, and restore each retained historical timing within 10% plus 0.03s.
Baseline and candidate are built at the same current-main dependency graph;
historical dependencies differ, so historical comparisons are not isolated
causal attribution. No performance recovery, merged withdrawal or #885 closure
is claimed until the complete study and required hosted checks pass.

Removed-test coverage attestation: prescribed-cross, preferred replacement and
TreeACI retention-only tests cover code removed in this withdrawal. Independent
scale, dtype, finite-value, resource, Guard, skeleton and growth regressions
remain. Their thresholds and CI coverage targets are unchanged. RSI is excluded.
