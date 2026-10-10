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

The complete [paired experiment](../experiments/treeaci-885-withdrawal.json)
measures main `32c76331` against withdrawal production commit `23042b92` at
identical dependency/feature resolution. Six-case construction time falls
from 527.930 to 50.100 seconds (ratio 0.0949, a 90.51% reduction). Both R=10
baselines exceed the predeclared 600-second deadline; both withdrawal cases
return, in 12.618 seconds for the chain and 216.069 seconds for CTTN. Every
declared case ran once in its registered order. Observable host-noise gates,
the primary aggregate, all current-base non-regression gates, and all eight
candidate exact-grid accuracy bounds pass. One pass provides descriptive
ratios without confidence intervals. Historical dependencies differ, so those
comparisons do not isolate a single commit's effect.

**The full registered study fails.** It also required all candidate cases to
report `Converged` and every historical timing to recover within 10% plus
0.03s. Low-temperature R=9 reports `MaxSweeps`; R=6 and warm R=9 remain
slower than history. The original protocol, all observations and failed gates
are retained unchanged. This is the explicitly requested withdrawal of the
costly repair with the original issue reopened, not a successful promotion
under the broader all-restoration protocol. Remaining historical timing
differences stay under umbrella #854; their causes are not established.

The original #784 fixture still reports `MaxSweeps` at 20 passes, with
independent relative Frobenius error about 3.0e-6 and no Guard discoveries.
No stopping threshold or success label changed. All retained Core/TreeACI
release tests and doctests pass (1,537 passed; 13 existing ignored), as do
strict changed-crate Clippy, the explicit heavy R=9 regression, current-source
API inventory and local repository-rules review. Public #874 real/complex
fixtures are unchanged. Hosted CI must pass before auto-merge; merged status
and issue closure are reported only when verified.

Removed-test coverage attestation: prescribed-cross, preferred replacement and
TreeACI retention-only tests cover code removed in this withdrawal. Independent
scale, dtype, finite-value, resource, Guard, skeleton and growth regressions
remain. Their thresholds and CI coverage targets are unchanged. RSI is excluded.
