# Tree pQTCI patch-size bounds: minimum patch size and capped outcomes

## Status

Implementation plan for the two decided M5 items of
[tree-pqtci-split-selection.md](./tree-pqtci-split-selection.md#open-questions-for-the-user):
open question 2 (minimum patch size) and open question 4 (capped outcomes,
without the engine's early exit). Written on `feat/tree-adaptive-patching` at
`a70a127d`. Nothing is implemented.

Writing the plan uncovered gaps in the recorded decisions. The user has since
decided two of them, recorded under [Decided](#decided). The others are under
[Open issues](#open-issues). Open issue 1 states what it blocks; every other
open issue carries a proposal that the implementation follows unless the
user changes it. Behavior marked
**provisional** is implemented as described here unless the user decides
otherwise.

The implementation phase keeps a work log under `docs/worklogs/`, following
the repository's "Work Logs And Design Records" rule. It records the
decisions taken while coding, the deviations from this plan, and the
verification conclusions.

## The decisions as recorded

From the user's answers of 2026-10-03 (quoted from the split-selection
record):

- **Q2.** "When a patch fails its tolerance and every remaining split would
  go below the minimum, the patch is accepted with its measured error and
  explicitly reported as not meeting the tolerance. Such a patch is never
  counted as certified. The run does not stop with an error. The minimum is
  given as a number of active quantics bits."
- **Q4.** "A `BondCapReached` patch that passes its error check is accepted
  only if its domain is at most a maximum size; a larger capped patch is
  split. A patch that converges below the cap is never split for its size.
  The maximum is given as a number of quantics bits: the count of the patch's
  unfixed (active) bits." A global maximum patch size is rejected. The early
  exit of the engine at the first saturated sweep is still open and outside
  this plan.

This plan implements Q4 as "if and only if", with the single exception of
the precedence rule for blocked patches
([open issues 1 and 7](#open-issues)). A `BondCapReached` run is accepted exactly
when the patch is at most the maximum size and the run passes its error
check. Every other capped run is not accepted.

## How a patch is decided today

Verified in `crates/tensor4all-partitionedtreetn/src/adaptive_interpolation.rs`
at `a70a127d`:

1. `process` computes the active positions (sites not fixed). A patch with at
   most one active **site** goes to `exact_patch` (line 1097). That path never
   calls the engine and needs no acceptance decision; it fails only on
   evaluator or construction errors.
2. Candidates are sampled. Under L2, an all-zero candidate set triggers the
   zero screen, which can produce a zero patch (lines 1149-1201).
3. The engine loop (lines 1214-1300) runs attempt `a = 0..=retries`:
   - any termination other than `Converged` breaks out **unmeasured** and
     splits (line 1230). `BondCapReached` and `IterationLimit` are treated
     alike;
   - a `Converged` run is layout-checked, re-embedded, and checked strictly
     below the cap (an engine error otherwise, line 1237). Under `SampledMax`
     it is accepted directly. Under L2 it is measured on verification stream
     `a` and accepted if `rms <= tau`. Otherwise `verification_failures` is
     incremented and the patch reruns with worst points and outcome pivots
     while `a < retries`, or breaks to the split.
4. `split` (lines 1316-1390) takes the first unfixed site of
   `layout.split_order`. If none exists, it returns `VerificationFailed`
   when the last run converged and failed its measurement, and
   `NoSplitIndexLeft` otherwise. A split fixes exactly **one site**, giving
   one child per coordinate.
5. `norm_report` and `verify::global_error` (verify.rs lines 321-410) combine
   all contributions. `certified_fraction` counts every exact or exhaustive
   contribution. `Certified` is produced whenever every contribution is
   exact or exhaustive. Its documented promise
   `rms_error <= tau (1 + GLOBAL_ROUNDING_MARGIN)` holds only because every
   accepted measurement had `rms <= tau`.

## Definitions

### Patch size: generalized bits (unit A)

**Decided by the user:** a patch's size is its number of **generalized
bits**: every active (unfixed) site counts as one bit, whatever its
dimension. Multi-site nodes and site-free nodes do not matter, because
counting is per site.

This is the patching level of Grosso et al. (paper and thesis). The level
`ℓ̄` is the number of fixed indices, one per index regardless of its local
dimension; a fused site of dimension `d = 2^N` counts as one level. A patch's
size is the number of sites minus `ℓ̄`. Every split fixes one site, so every
split removes exactly one bit. "The next split goes below the minimum" and
"every remaining split goes below the minimum" are therefore the same
condition.

Consequences:
- A fused quantics site of dimension 4 or 8 counts as one bit.
- A site of dimension 3, or of dimension 1, also counts as one bit. No layout
  is rejected. Telling the user about such sites is deferred
  ([open issue 3](#open-issues)).
- No unit distinguishes a quantics bit from another site, such as the binary
  flag site of the M3 diagnostic tree or a spin index. Each counts as one bit.

All active sites count, including sites missing from a partial `patch_order`
that the driver will never split ([open issue 4](#open-issues)).

### Deferred: spatial size (unit B)

Unit B is shelved by the user. It is recorded here so that it can be added
later; no code path or test for it is planned now.

- **Definition.** The size is `log2 |P|`, the logarithm of the patch's
  number of points. Fused sites of dimension `2^N` then count `N` bits.
  Comparing the point count with `2^bits` directly (rather than rejecting
  dimensions that are not powers of two) keeps it defined for every layout.
- **Literature.** Grosso's thesis, §3.2, in the overpatching discussion
  (around Fig. 3.8), proposes "Minimum patch size: impose a lower bound
  `|Ω|min` on the spatial size of any patch; if a candidate split would
  produce sub-domains smaller than this threshold, the recursion is halted
  and the current patch is accepted as is." This is a suggestion for future
  versions, not an implemented feature.
- **Why it matters.** The user notes that fused sites matter for MPO
  compression, where the spatial size is the relevant quantity.
- **Its open problem (reviewer major 1).** Under B, "the next split goes
  below the minimum" and "every remaining split goes below the minimum"
  diverge on mixed layouts. Example: a 5-bit patch made of one fused 4-bit
  site (dimension 16) and one binary site, with a minimum of 2. Splitting the
  fused site leaves 1 bit (below the minimum); splitting the binary site
  leaves 4 bits (allowed). With the fixed split order, the result depends on
  which site comes first. The "every remaining split" reading would require
  skipping ahead in `patch_order` or a split selector (M5 Q1). This must be
  decided before B is implemented.
- **Compatibility.** `PatchedInterpolationOptions` is `#[non_exhaustive]`.
  B can later be added as a unit field whose default is A, without breaking
  A. The option names below say "bits", not "log2" or "volume", and their
  rustdoc defines a bit as "one active site".

### Blocked split

With `min_patch_bits = Some(m)`, the split of a patch `P` is **blocked** when
its next split site `s` (the first unfixed site of `split_order`) exists and
the child would have fewer than `m` bits: `active_sites(P) - 1 < m`. When no
`s` exists, `patch_order` is exhausted. That case keeps its M3 errors and is
not a blocked split ([open issue 5](#open-issues)).

### Capped eligibility

With a maximum `M` set, an engine run is **capped-eligible** when its
termination is exactly `BondCapReached` and `active_sites(P) <= M`.
`IterationLimit` and any future variant are not eligible
([open issue 9](#open-issues)).

Under A, if both `M` and `m` are set and `M >= m`, every blocked patch has
`active_sites(P) <= m <= M`, so a blocked capped run is always eligible.

## Semantics

### Per engine run

For engine run `a` of a patch with at least two active sites:

| Outcome | L2 | SampledMax |
|---|---|---|
| `Converged` | as M3: checks, measure on stream `a`; pass: accept `WithinTolerance`; fail: retry while `a < retries`, then **unaccepted** | accept `WithinTolerance` (unchanged) |
| `BondCapReached`, capped-eligible | layout check, re-embed, bond `<= cap` check; measure on stream `a`; pass: accept `WithinTolerance` (termination `BondCapReached`); fail: **unaccepted**, no retry ([open issue 11](#open-issues)) | same checks; `engine_error_estimate <= engine tolerance`: accept `WithinTolerance`; otherwise unaccepted ([open issue 8](#open-issues)) |
| anything else | unaccepted, unmeasured (as M3) | unaccepted (as M3) |

A failed measurement of a capped-eligible run counts in
`verification_failures`, and its worst points go to the children exactly as
in M3 step 10.

### An unaccepted patch

In this order:

1. **Split allowed** (`s` exists and the split is not blocked): split as in
   M3, passing the worst points of the last failed measurement.
2. **Split blocked by the minimum**: accept the **last** engine run's network
   ([open issue 12](#open-issues)).
   - If that run was already judged (converged or capped-eligible, and
     failed: measured under L2, estimate-checked under SampledMax), it is
     accepted as `ToleranceNotMet` with that result. No second measurement
     runs.
   - **Provisional ([open issue 1](#open-issues)).** If that run was not
     measured (`BondCapReached` without eligibility, or `IterationLimit`),
     the driver layout-checks it, re-embeds it, and checks its bond against
     the cap. Under L2 it measures the run once on verification stream `a`,
     with no retry. If `rms <= tau` the patch is `WithinTolerance`;
     otherwise it is `ToleranceNotMet`. Under SampledMax it is
     `WithinTolerance` if `engine_error_estimate <= engine tolerance`,
     whatever the non-converged termination; otherwise it is
     `ToleranceNotMet`. The record's `termination` shows that the engine
     did not converge.
   - A sampled measurement of an accepted blocked patch gets an audit, as
     every sampled contribution does (`audit = true`). The audit removes the
     upward selection bias of a measurement that was kept because it failed.
3. **No split site left** (`patch_order` exhausted): the M3 errors remain.
   `VerificationFailed` is returned when the last run was measured and
   failed (now also for a capped-eligible run; its rustdoc and message change
   from "converged below the cap" to "measured"). `NoSplitIndexLeft` is
   returned when the last run was not measured; its rustdoc changes from
   "did not converge" accordingly.

### Edge cases

- **Exact patches** (at most one active site) are decided before any of this
  and are always `WithinTolerance`. A minimum of 0 or 1 changes nothing,
  because a two-site patch's split yields exact one-site children. With a
  minimum `m >= 2`, a failing two-site patch is accepted as
  `ToleranceNotMet` although splitting it would give exact patches. That
  follows from the decision, and the rustdoc states it.
- **The root** can itself be blocked (`active_sites(X) - 1 < m`,
  equivalently `m >= active_sites(X)`). The whole run is then one patch,
  possibly `ToleranceNotMet`. This is allowed and not validated; the rustdoc
  warns about it.
- **Zero patches** are not affected. The zero screen runs before the engine,
  and a zero verdict requires a passing measurement. A failed zero screen on
  a blocked patch proceeds to the engine and then to the rules above.
- **A maximum of 0 or 1** never makes a run eligible: a patch that reaches
  the engine has at least two active sites. For configurations that pass
  validation, the behavior then equals no maximum. Open issue 13 may reject
  a maximum below the minimum before any run takes place.
- **`max < min`**: every capped patch with `M < active_sites <= m` would be
  handled by rule 2, so the maximum has no effect there. Whether to reject
  this is [open issue 13](#open-issues).
- **Retries** apply only to a failed `Converged` run, as in M3. A blocked
  patch whose runs all converge and fail therefore makes `1 + retries`
  engine calls, and the accepted network is that of the last run. A run at
  attempt `a >= 1` that is capped-eligible or blocked-and-unmeasured is
  measured on stream `a`, like a converged run.
- **A network that is above the cap or malformed** is used only when a
  non-converged run is capped-eligible or blocked. The handling is
  [open issue 10](#open-issues). Provisionally it is
  `Interpolation { source: Engine }`, like `Converged` at the cap today.
- **`max_patches`** is unchanged: blocked acceptances count as processed
  patches.
- **Determinism**: the new decisions compare a measurement with `tau` in the
  same way as the M3 decisions, under the same scope and caveats (#795).
  Classification uses only integers (site counts) and the termination.

## Accounting and report

### What breaks in the M3 contract

Accepting a patch whose measured error exceeds `tau` invalidates these
documented statements:

- the error contract, Budget (line 430): "summing ... gives `E <= delta`".
  The module docs of `adaptive_interpolation` (line 32), the README, and the
  guide repeat it;
- error contract line 829 and the `GlobalL2Error` rustdoc (report.rs line
  318): "Every acceptance measurement satisfies `rms <= tau`, so a certified
  `rms_error` and `acceptance_statistic_rms` do not exceed `tau`";
- `GlobalL2Error::Certified` (report.rs line 336): "`rms_error` is at most
  `tau * (1 + GLOBAL_ROUNDING_MARGIN)`", together with the certificate
  `E <= delta (1 + margin) + rounding` in the module docs, README, guide, and
  skill reference;
- error contract line 373: "certified when every contribution is exact or
  exhaustive", which defines certification by method only. The same
  definition appears in `MeasurementMethod::Exhaustive` ("a certificate",
  report.rs line 69), in the `VerificationOptions` rustdoc ("a certificate
  up to rounding", options.rs line 274), and in the error-contract row of
  `docs/design/index.md`;
- `certified_fraction` ("fraction of `|X|` whose contribution is `Exact` or
  `Exhaustive`"): Q2 says such a patch is "never counted as certified";
- the error contract Semantics (line 893), the roadmap M2 scope (line 140),
  driver record step 10 (line 259), the M1 record (lines 35-36, "only the
  first is accepted"), the `PatchedInterpolationOptions::max_bond_dim`
  rustdoc (options.rs line 67), and the `PatchRecord::termination` rustdoc
  ("always `Converged` for an accepted patch"). Q4 breaks these as well;
- error contract Algorithm step 6 (line 947): "No measurement runs on a patch
  that is not accepted anyway". Capped-eligible and blocked runs are now
  measured;
- `VerificationOptions::retries` ("before the patch splits") and the
  `PatchedInterpolationOptions` rustdoc (options.rs lines 26-27, "splits down
  to exact patches"), which no longer hold with a minimum.

The capped acceptances of Q4 meet their allowance by measurement, so they
break no accounting statement. They do widen the selection effects of
sampled acceptance that M3 open question 4 warned about. They also accept
more large patches on uniform samples, which is the patch class in which the
corner-localized misses were found ([open issue 15](#open-issues)).

### Report design (decided: Option 1)

**Decided by the user:** "certified" means *within the allowance and
measured exactly* (exact or exhaustive). A patch that does not meet its
allowance is never certified, even when its error is known exactly.

- `certified_fraction` excludes `ToleranceNotMet` patches.
- A new variant `GlobalL2Error::ToleranceNotMet` is produced whenever some
  contribution is `ToleranceNotMet`, and takes precedence over the three M3
  variants. Those variants keep their exact M3 promises, which hold again
  because they occur only when every patch meets its allowance.

Shape of the new variant:

- **`measured_rms`.** It combines every contribution's best available
  measurement: the exact or exhaustive value, else the audit, else the
  acceptance statistic. The name avoids `rms_error`, which belongs to
  `Certified` by the existing naming convention (`rms_error` for
  `Certified`, `rms_error_estimate` for `Audited`,
  `acceptance_statistic_rms` for `AcceptanceOnly`).
- **`unmet_fraction`.** The fraction of `|X|` covered by `ToleranceNotMet`
  patches.
- **`basis`.** How `measured_rms` is known, so that information is not lost.
  It uses the M3 classification rules over all contributions:
  - `ExactOrExhaustive { rounding_allowance_rms, rounding_limited,
    relative_error_bound }`.
    `measured_rms` is then the measured `E / sqrt(|X|)`. The re-based
    certificate `E <= sqrt(|X|) (measured_rms (1 + margin) + rounding)`
    still holds up to the rounding model. The relative bound is kept: the
    formula in `verify.rs` lines 337-347 uses the measured value and
    `approximation_rms` only, never `tau`, so it remains valid. It is
    `None` under the same conditions as in M3. This is a bound on the
    error, not a statement that the tolerance was met. `rounding_limited`
    keeps its M3 meaning (the rounding allowance reaches `tau`): a run that
    misses its tolerance by rounding alone is worth distinguishing from one
    that misses it by approximation error.
  - `Audited { mean_square_rel_std_error, relative_bound_estimate }`, with
    the M3 meanings.
  - `AcceptanceOnly`, with no relative statement.

Per patch: `PatchRecord::status: PatchStatus`, with variants
`WithinTolerance` and `ToleranceNotMet`. The enum is `#[non_exhaustive]`, so
another acceptance reason (for example a sibling merge) can be added later.

Report level: `PatchedInterpolationReport::tolerance_met(&self) -> bool`,
true when every accepted record is `WithinTolerance`. It is the only global
indicator under `SampledMax`, which has no `GlobalL2Error`.

The global error can exceed `delta` without limit. The budget of accurate
patches is not redistributed to compensate (M3 rule). The rustdoc says this.

## Public API changes

All in `tensor4all-partitionedtreetn::adaptive_interpolation`. Names are
proposals. The block shows the maximum as option (a) of
[open issue 14](#open-issues), and the rest of this plan writes it as `M`,
until that issue is decided. Under the proposed option (c),
`max_capped_patch_bits: Some(M)` becomes
`capped_patches: CappedPatches::AcceptUpTo { bits: M }` and `None` becomes
`CappedPatches::Split`; the semantics are unchanged.

```rust
// PatchedInterpolationOptions (already #[non_exhaustive]); two new fields.
/// Smallest patch the driver may create, in generalized bits (one bit per
/// active site, whatever its dimension). A split whose children would have
/// fewer bits is not made; a patch whose split is blocked this way is
/// retained instead of split. Its status follows whether it meets its
/// allowance: the measured error under L2, the engine estimate under
/// `SampledMax` (provisional rules in open issues 1 and 6).
/// `None`: no minimum.
pub min_patch_bits: Option<usize>,
/// Size bound, in generalized bits, for accepting `BondCapReached` patches
/// on their error check; larger capped patches are split. Patches that
/// converge below the cap are never split for their size.
/// NOTE: `None` (default) disables capped acceptance through this option
/// (the M3 behavior), unlike `max_patches: None` (no limit). The provisional
/// minimum rule can still retain a capped patch (open issues 1 and 7).
pub max_capped_patch_bits: Option<usize>,
// Builders: with_min_patch_bits(bits), with_max_capped_patch_bits(bits).

/// Whether an accepted patch meets its allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PatchStatus { WithinTolerance, ToleranceNotMet }

// PatchRecord (already #[non_exhaustive]): new field `status: PatchStatus`.
// PatchedInterpolationReport: new method `tolerance_met(&self) -> bool`.
// GlobalL2Error (already #[non_exhaustive]): new #[non_exhaustive] variant
//   ToleranceNotMet { measured_rms: f64, unmet_fraction: f64,
//                     basis: ToleranceNotMetBasis }
//   rms_value() returns measured_rms.
// New #[non_exhaustive] enum ToleranceNotMetBasis {
//   ExactOrExhaustive { rounding_allowance_rms: Option<f64>,
//                       rounding_limited: Option<bool>,
//                       relative_error_bound: Option<f64> },
//   Audited { mean_square_rel_std_error: f64,
//             relative_bound_estimate: Option<f64> },
//   AcceptanceOnly }
```

- **Defaults.** `None` and `None` reproduce M3 exactly: the M2 golden test
  and every existing test pass unchanged ([open issue 2](#open-issues)).
- **Values.** Any value is accepted. Under A, a patch has at most as many
  bits as the problem has sites. A larger maximum acts as "no size bound
  for capped patches"; a larger minimum blocks the root.
- **Validation.** Nothing is validated unless open issue 13 adopts the
  `max >= min` rule. No layout is rejected.
- **Errors.** None are added or removed. `VerificationFailed` widens from
  "converged" to "measured", and `NoSplitIndexLeft` narrows to "not
  measured" (rule 3). The rustdoc of `Interpolation` names the new
  engine-error cases (open issue 10).
- **Changed rustdoc meaning under the same names.**
  - `PatchRecord::termination` may be `BondCapReached` or `IterationLimit`
    for an accepted patch.
  - `PatchRecord::acceptance` of a `ToleranceNotMet` patch is its last
    (failed) measurement.
  - `max_bond_dim` no longer says "accepted only when the engine converges".
  - `verification_failures` counts every measured engine run with
    `rms > tau`.

  No field changes its type, so these are documentation changes. But
  `termination` and `certified_fraction` change meaning, so the PR body
  lists them as deliberate breaking changes (early development).
- **No diagnostic counters** are added. Capped acceptances and blocked
  acceptances are visible in the records (`termination`, `status`), and
  tests use the scripted engines' call logs. A count of capped runs split
  for size, if the M5 measurements need one, would be a separate
  diagnostic-only addition.

## Implementation steps

1. `layout.rs`: a helper for `active_sites(P)` from `fixed`, and the
   validation of open issue 13 if it is adopted.
2. New private module `adaptive_interpolation/acceptance.rs` for the policy:
   classify a run (eligible, measured, unaccepted) and resolve an unaccepted
   patch (split, blocked, exhausted). `adaptive_interpolation.rs` is 1488
   lines, and the policy is a separate behavior, so it gets its own file
   under the file-organization rule.
3. `process`: the engine loop records the last run's network and whether it
   was measured. Capped-eligible runs follow the measurement path, and
   retries stay limited to `Converged` failures. `split` receives the
   resolution result. The choice between `NoSplitIndexLeft` and
   `VerificationFailed` uses "last run measured and failed" in place of
   "last run converged".
4. `report.rs`: `PatchStatus`, `PatchRecord::status`, `tolerance_met()`, the
   new variant and its basis enum, and the rustdoc listed under
   Documentation.
5. `verify.rs`: `Contribution` gains `within_tolerance`. `global_error`
   checks for `ToleranceNotMet` first, computes its basis with the existing
   classification, and excludes unmet patches from `certified_fraction`.
6. Tests (below), then documentation. Two PRs are possible: Q4 first (no
   report type change), then Q2 with the report changes and the interaction
   rules. One PR is also fine, since the interaction tests need both.

## Test plan

New integration file `tests/adaptive_patch_size.rs`, reusing
`tests/adaptive_common`. Shared test helpers:

- `ScriptedEngine` gets a per-step `error_estimate` and a
  `Step::capped_with(network)` constructor.
- The `never` evaluator, now duplicated in `tests/adaptive_l2.rs:89` and
  `tests/adaptive_interpolation.rs:698`, moves into `adaptive_common`, and
  both files use it. A third copy is not added.
- `check_invariants` gains the status checks: under L2, `WithinTolerance`
  implies `acceptance.rms <= tau`, and `ToleranceNotMet` implies `rms > tau`.
- `assert_same_run` (`adaptive_common/mod.rs:801`) also compares the new
  `status` field.

**Shared setup** unless stated otherwise:
- The problem is `chain("s", 4, 2)`: 16 points, derived order `s0..s3`.
- `f(x) = 1 + x0 + 2 x1 + 4 x2 + 8 x3`: values 1 to 16, `sum f^2 = 1496`.
  It has rank 2 across every bond.
- `L2Reference::Given(sqrt(1496))`, `rtol = 1e-12`. A test that uses another
  problem or another `f` passes `L2Reference::Given` with the L2 norm of its
  own `f` and states it.
- `max_bond_dim = 3`, because an `Exact` network of `f` has rank 2, and a
  `Converged` rank-2 network at a cap of 2 is an engine error (line 1237).
- `retries = 0`, seed 0, no user pivots, `recycle_pivots = false`.
- Every patch has at most 16 points, which is at most
  `max(max_exhaustive_points, samples) = 1024`, so every patch is measured
  exhaustively. RMS values are compared with closed forms to a few ulps.

1. **Accept at the minimum (converged, failed).** `min = 3`; every step is
   `converged(Constant(0))`. The root fails and splits at `s0` (the child has
   3 bits, which is at least 3). Both children fail and are blocked.
   - Expect no error, `splits = 1`, and two `ToleranceNotMet` records: rms
     `sqrt(85)` for the odd values and `sqrt(102)` for the even values.
   - The global error is `ToleranceNotMet { measured_rms = sqrt(93.5),
     unmet_fraction = 1, basis: ExactOrExhaustive { relative_error_bound:
     None, .. } }`. The bound is `None` because the approximation is zero.
   - `certified_fraction = 0`, `tolerance_met() == false`,
     `function_evaluations = 16`, and the dense check from the materialized
     partition gives `||f - f~|| = sqrt(1496)`.
   - Control: without the minimum, the same run reaches exact patches and is
     `Certified`.
2. **Retries before the minimum; the last run is kept.** As test 1 with
   `retries = 1`. The step constants are `[0, 0, 0, 8, 0, 9]`, in call order:
   root run 0, root run 1, child 0 run 0, child 0 run 1, child 1 run 0,
   child 1 run 1.
   - Expect 6 engine calls, `retries_used = 1`, and both children at
     `rms = sqrt(21)`: the constants 8 and 9 center the values.
   - `approximation_rms = sqrt(72.5)`.
   - `relative_error_bound` is `Some` and equals the M3 formula with
     `measured_rms = sqrt(21)`, to `1e-6` relative (the margins are `1e-8`).
3. **Mixed status.** Steps: root `Constant(0)`, child 0 `Exact`, child 1
   `Constant(0)`; `min = 3`, `retries = 0`.
   - Expect child 0 `WithinTolerance` (exhaustive, rms at rounding level) and
     child 1 `ToleranceNotMet` with `sqrt(102)`.
   - `certified_fraction = 0.5`, `unmet_fraction = 0.5`,
     `measured_rms = sqrt(51)` up to rounding.
   - `measured_rms^2` equals the sum of `(|P|/|X|) rms_P^2` over the records.
4. **Blocked unmeasured run (provisional, open issue 1).** Every step is
   `capped()` (`Constant(0)`, `BondCapReached`); no maximum; `min = 3`.
   - The root splits unmeasured. Only the two blocked children are
     measured, so `verification_failures = 2` (it would be 3 if the root had
     been measured).
   - Both children are `ToleranceNotMet`, with termination `BondCapReached`
     and the same RMS as test 1; `function_evaluations = 16`.
   - Variant: `DenseEngine::with_fault(IterationLimit)` with `f = 2^(sum x)`
     (rank 1). The blocked children pass and are `WithinTolerance` with
     termination `IterationLimit`.
5. **Capped run accepted at or below the maximum.** `DenseEngine`,
   `max_bond_dim = 2`, `f = 2^(sum x) + 3^(sum x)` (rank 2 across every
   bond, and in every child), `rtol = 1e-10`. The exact network has rank 2,
   at the cap. With `max = 4`:
   - the root is accepted, `splits = 0`;
   - termination `BondCapReached`, `max_bond_dim = 2`, `Certified`;
   - the dense residual is at most `delta`.
6. **Capped run split above the maximum.** As test 5 with `max = 3`:
   - `splits = 1`, the root is unmeasured, and the two capped children are
     accepted;
   - the partition matches the dense function within `delta`.
   - Control without a maximum: every capped patch splits, down to exact
     patches, giving `splits = 7` and 8 exact patches.
7. **A converged large patch is never split for its size.** `DenseEngine`,
   cap 2, `f = 2^(sum x)` (rank 1), `max = 0`: accepted at the root,
   `splits = 0`.
8. **A capped-eligible failure splits without retry.** Root step `capped()`,
   `max = 4`, `retries = 1`, children `converged(Exact)`.
   - Expect `verification_failures = 1` and `engine_retries = 0`.
   - The children's recorded initial pivots begin with the worst points of
     the root measurement, as in the M3 helper `split_children`. This holds
     only because there are no compatible user pivots and recycling is off:
     user and recycled pivots come first (`sampling.rs` lines 288-297).
9. **A capped-eligible failure at an exhausted `patch_order`.**
   `patch_order = [s0]`, every step `capped()`, `max = 4`. The root is
   measured, fails and splits at `s0`. The queue is FIFO: the child `s0 = 0`
   is measured, fails, has no split site left, and the run returns, so the
   child `s0 = 1` is never processed. Expect `VerificationFailed` with
   projector `s0 = 0`, where M3 returns `NoSplitIndexLeft`.
10. **A capped-eligible failure that is also blocked.** `min = 4`, `max = 4`,
    root step `capped()`. The root is measured once, fails, and is blocked.
    - Expect a single record: `ToleranceNotMet`, termination
      `BondCapReached`, `rms = sqrt(93.5)`.
    - `verification_failures = 1` shows that no second measurement ran.
11. **`IterationLimit` with a maximum set is split, not eligible.**
    `DenseEngine::with_fault(IterationLimit)`, `f = 2^(sum x)`, `max = 4`,
    no minimum. Every multi-site patch splits unmeasured: `splits = 7`,
    8 exact patches, `verification_failures = 0`.
12. **A network above the cap or malformed (provisional, open issue 10).**
    Each case returns `Interpolation { source: Engine }` naming the
    violation:
    - `DenseEngine`, cap 2, `f = 1^(sum x) + 2^(sum x) + 3^(sum x)`
      (rank 3 at the middle bond): capped-eligible with `max = 4`, and,
      separately, blocked with `min = 4` and no maximum;
    - `Fault::CapAfterPivots` (an empty network): blocked with `min = 4`;
    - `Fault::WrongLayout` with the rank-2 `f` of test 5 at cap 2 (so the
      termination is `BondCapReached`): blocked with `min = 4`.
13. **A retry run measured on stream 1.** `chain("s", 6, 2)` with
    `f = 1 + sum_k 2^k x_k` (values 1 to 64), `ScriptedEngine`, cap 3,
    `samples = 4`, `max_exhaustive_points = 0`, `retries = 1`, `min = 6` (the
    root is blocked). Steps: run 0 `converged(Constant(0))`, which fails;
    run 1 `capped()`.
    - With `max = 6`, run 1 is capped-eligible. Its acceptance points equal
      `streams::draw(&[2; 6], 4, streams::verify(streams::path_state(0,
      &[]), 1))`, and its RMS is computed from `f` at those points.
    - The audit uses `streams::audit`.
    - Without a maximum, the blocked-unmeasured path gives the same
      measurement (provisional, open issue 1).
14. **SampledMax.** Steps with error estimates:
    - a capped-eligible run with an estimate at or below the engine
      tolerance is accepted, with acceptance `None`; above it, the patch
      splits;
    - a blocked run with an estimate above the tolerance is
      `ToleranceNotMet` and `tolerance_met() == false`; a blocked
      `IterationLimit` run with an estimate at or below it is
      `WithinTolerance`;
    - `NormReport::SampledMax` is unchanged.
15. **A fused site counts as one bit.**
    - `chain("q", 3, 4)` (three sites of dimension 4, 64 points), `ScriptedEngine`
      `converged(Constant(0))`, cap 3, `f = 1 + x0 + 4 x1 + 16 x2` (values 1
      to 64), `min = 2`. The root has 3 bits and splits at `q0` into four
      2-bit children, which are blocked.
      - Expect four `ToleranceNotMet` records with `rms^2` = 1301, 1364, 1429,
        and 1496 for `q0` = 0 to 3.
      - `measured_rms^2 = 1397.5`, `splits = 1`.
      - Counting `log2` of the volume would give the children 4 bits and let
        them split; this test pins unit A.
    - Maximum: `DenseEngine`, cap 2, `chain("q", 2, 4)` with
      `f = 2^(x0 + x1) + 3^(x0 + x1)` (rank 2), `max = 2`. The root (2 bits)
      is capped-eligible and accepted, `splits = 0`.
    - Dimension 3 (no rejection under A): a three-site chain with
      dimensions (3, 2, 2), the dimension-3 site `q0` first in
      `patch_order`, `ScriptedEngine` `converged(Constant(0))`, cap 3,
      `f = 1 + x0 + 3 x1 + 6 x2` (values 1 to 12, `sum f^2 = 650`),
      `min = 2`, `max = 3`. The root (3 bits) fails and splits at `q0` into
      three 2-bit children, which fail and are blocked.
      - Expect three `ToleranceNotMet` records with
        `rms^2 = a^2 + 9a + 31.5` for `a = q0 + 1`: 41.5, 53.5 and 67.5.
      - `measured_rms^2 = 650 / 12`, `splits = 1`.
16. **Invalid options before any evaluation.** Only if open issue 13 adopts
    `max >= min`: `InvalidInput` with the shared `never` evaluator, at its
    documented position in the validation order.
17. **`patch_order` exhaustion versus the minimum.**
    - `patch_order = [s0]`, `min = 1`, `Constant(0)`: `VerificationFailed` at
      a child (unchanged).
    - `patch_order = [s0, s1]`, `min = 3`: the children are blocked at `s1`
      and accepted as `ToleranceNotMet`.
18. **Settings with no effect equal `None`.** A minimum of 0 or 1, and a
    maximum of 0 or 1, each varied with the other option unset, produce runs
    equal to the unset run under `assert_same_run`. For combinations, this
    equivalence applies only to configurations that pass the validation
    chosen in open issue 13. This holds on every layout under A: a patch that
    reaches the engine has at least two active sites, and a split of a
    two-site patch yields exact one-site patches.
19. **Zero patches under the minimum.** `f = x0 (1 + x1 + 2 x2 + 4 x3)`
    (zero on `s0 = 0`), `ScriptedEngine` `converged(Constant(0))`, cap 3,
    `retries = 0`, `min = 3`, seed 0.
    - Child `s0 = 0` is a zero patch with an exhaustive acceptance.
    - Child `s0 = 1` is `ToleranceNotMet` with `rms^2 = 25.5` (values 1 to 8).
    - `measured_rms^2 = 12.75`, `certified_fraction = 0.5` (the zero patch),
      `unmet_fraction = 0.5`.
20. **A sampled blocked patch with an audit.** `chain("s", 6, 2)`, the `f` of
    test 13, `ScriptedEngine` `converged(Constant(0))`, cap 3,
    `retries = 0`, `samples = 4`, `max_exhaustive_points = 0`, `min = 5`.
    - The 32-point children are sampled, fail, are blocked, and are audited.
    - The expected acceptance and audit RMS values come from
      `streams::draw` with `streams::verify(.., 0)` and `streams::audit`.
    - The basis is `Audited` and uses the audits; with `audit = false` it is
      `AcceptanceOnly` and uses the acceptance statistics.
21. **A branched tree with TreeTCI** (`quantics_tree`, junction of degree
    three), with a maximum and a minimum that both trigger:
    - `check_invariants` passes;
    - at least one accepted `BondCapReached` record has `max_bond_dim <=
      cap`;
    - dense comparison: if `tolerance_met()`, the dense error satisfies
      the existing M3 certificate
      `E <= delta * (1 + GLOBAL_ROUNDING_MARGIN) + sqrt(|X|) * rounding_allowance_rms`;
      otherwise `measured_rms` equals the dense `E / sqrt(|X|)`
      (every patch exhaustive) to `1e-12` relative.
22. **Regression.** `adaptive_m2_golden.rs` and all M3 tests pass unchanged
    with the defaults. `a_capped_rerun_after_a_failed_verification_keeps_no_split_index_left`
    keeps its meaning, because no maximum is set.
23. **Rustdoc.** Runnable, asserted examples for both builders,
    `PatchStatus`, `tolerance_met`, the new variant, and its basis (a
    two-bit blocked patch with a known RMS).

## Documentation updates

- **Rustdoc:**
  - module docs of `adaptive_interpolation`: Error contract, Algorithm steps
    4-6, and the `E <= delta` sentence;
  - `PatchedInterpolationOptions`: the new fields, `max_bond_dim`, the
    `patch_order` failure sentence, and the "splits down to exact patches"
    sentence (lines 26-27);
  - `VerificationOptions` (lines 274ff., the "certificate" wording) and
    `VerificationOptions::retries`;
  - `MeasurementMethod::Exhaustive`;
  - `PatchedInterpolationError::{Interpolation, VerificationFailed,
    NoSplitIndexLeft}` and the `# Errors` section of `patched_interpolate`;
  - `PatchRecord`, `GlobalL2Error` (the `rms <= tau` paragraph and
    `Certified`), `L2ErrorReport::certified_fraction`;
  - the crate root (`lib.rs`).
- `crates/tensor4all-partitionedtreetn/README.md`, adaptive interpolation
  section: the certificate sentence and the acceptance rule.
- `docs/book/src/guides/partitioned-treetn.md`, Adaptive patched
  interpolation: acceptance, the two options, generalized bits,
  `ToleranceNotMet`, and a runnable asserted snippet. Run
  `./scripts/test-mdbook.sh`.
- `skills/use-tensor4all-rs/references/crates.md`, `patched_interpolate`
  entry: the options and the new variant. Replace "only `Certified` is a
  bound": `ToleranceNotMet::ExactOrExhaustive` also retains a rigorous
  measured-error bound, while `Certified` additionally states that every
  patch met its allowance.
- `llms.txt` line 18: add "minimum patch size and capped-patch bound" to the
  guide summary.
- **Design records:**
  - error contract: Budget, the definition of certified, Records and report,
    Errors, Semantics, Algorithm step 6, and a dated amendment under OQ4;
  - roadmap: M2 scope note and M5 status;
  - driver record: step 10 ("superseded" note);
  - M1 record: lines 35-36, plus the cap sentence if open issue 10 is decided
    as proposed;
  - split-selection: Q2 and Q4 point here;
  - `docs/design/index.md`: the error-contract row's definition of
    certified.
- If open issue 10 is decided as proposed: the rustdoc of
  `InterpolationOutcome::network` or `InterpolationTermination` in
  `tensor4all-treetn`.
- A work log under `docs/worklogs/` for the implementation phase.

## Issues for the user

### Decided

- **Certification (2026-10-03).** Option 1: "certified" means within the
  allowance and measured exactly. A new `GlobalL2Error::ToleranceNotMet`
  variant with a `basis` is added, as specified under
  [Report design](#report-design-decided-option-1). The rejected Option 2
  kept "certified" as a statement about the measurement method only, so a
  run that missed its tolerance could still be labelled `Certified`.
- **Unit of "bits" (2026-10-03).** Unit A (generalized bits = patching level)
  only. Unit B is deferred, with its open problem recorded under
  [Deferred: spatial size](#deferred-spatial-size-unit-b).

### Open issues

A blocking issue identifies the phase that must wait for an answer (for
example, final rustdoc in issue 1). The other issues have a proposal that
the implementation follows unless the user changes it.

1. **Blocked patches with an unconverged run (reviewer major 2; open,
   provisional behavior).** Q2 says "when a patch fails its tolerance". A
   blocked patch whose last run is `BondCapReached` (not eligible) or
   `IterationLimit` has not failed any measurement; it was never measured.
   - Provisional rule ("minimum wins"): measure the run and accept it. It is
     `WithinTolerance` if it passes; the record's `termination` shows that
     the engine did not converge.
   - Conflict: with no maximum set, this accepts capped patches although Q4
     accepts capped patches only when they are small enough. Under unit A
     with a maximum set and `max >= min`, every blocked capped run is
     eligible anyway, so the conflict arises only when the maximum is unset
     (or below the minimum, open issue 13).
   - The alternatives are not consistent with the decisions:
     - accepting without measuring contradicts "with its measured error";
     - splitting below the minimum contradicts Q2;
     - stopping with an error contradicts "the run does not stop with an
       error";
     - always labelling such a patch `ToleranceNotMet` even when it passes
       would be possible, but it reports a tolerance failure that the
       measurement does not show.

   Blocking for the final rustdoc; the code can proceed with the provisional
   rule.
2. **Default values.** Neither decision fixes a default. Proposed: `None` for
   both, which is exactly M3 and keeps the golden outputs. Q4's direction
   then applies only when the user opts in. A numeric default (for example
   10, see open issue 15) is a candidate once M5 measurements exist.
3. **Informing the user about sites of dimension other than 2 (decided:
   deferred; no API and no tests now).** Under unit A such a site counts as
   one generalized bit. A user who sets `min_patch_bits` or
   `max_capped_patch_bits` and thinks in spatial bits should learn that. Library
   code must not print: the repository has no logging framework, and the
   bindings cannot capture stderr. Two options are recorded:
   - (1) a structured notice in the report that lists each such active site
     and its dimension;
   - (2) a pre-run check function that validates the layout and options
     before any evaluation and returns the same notice, using the driver's
     own validation code (no second copy).

   The precedent for a structured notice is `warnings: Vec<String>` in the
   treetn linsolve updater (`tensor4all-treetn/src/linsolve/square/updater.rs`
   line 37). A typed notice would fit this crate's typed reports better than
   strings. Reviewer major 1 applies only to unit B and is recorded with it.
4. **Sites outside a partial `patch_order` count toward the size.** The
   driver never splits them. Proposed: they count, because the size is the
   patch's domain.
5. **Vacuous truth when `patch_order` is exhausted.** Read literally, "every
   remaining split would go below the minimum" is true when no split
   remains. Under that reading, the minimum turns `NoSplitIndexLeft` and
   `VerificationFailed` into acceptances whenever it is set. Proposed: no.
   Exhaustion keeps its errors, and only a split that exists but is blocked
   leads to acceptance. As a result, a blocked patch is accepted while an
   exhausted `patch_order` stops with an error, although the two situations
   are similar.
6. **Q2 under `SampledMax`.** No measured error exists there. Proposed:
   allowed. The record carries the engine estimate, and a blocked patch is
   `WithinTolerance` exactly when `engine_error_estimate <= engine
   tolerance`, for every non-converged termination. This matches L2, where a
   passing measurement makes a blocked patch `WithinTolerance`. Alternative:
   the minimum is L2-only (`InvalidInput` under SampledMax).
7. **Q2 versus Q4 precedence.** When a split is blocked, the minimum wins:
   the patch is accepted instead of being split. Under unit A this matters
   only in the cases of open issue 1 (no maximum set) and open issue 13
   (`max < min`). The previous draft wrongly limited it to layouts with
   dimensions above 2.
8. **Q4 under `SampledMax`.** The driver performs no error check there.
   Proposed: the engine estimate against the engine tolerance. For TreeTCI at
   the cap, that is the last pivot error. Alternative: Q4 is L2-only.
9. **`IterationLimit` is not covered by Q4.** Proposed: it is never
   capped-eligible (it splits as now) and is accepted only when blocked
   (open issue 1).
10. **A capped or blocked network that is above the cap or malformed.** The
    M1 contract (`tensor4all-treetn` rustdoc) requires every outcome network
    to be over the active sites, but says nothing about its bond for a
    non-converged termination. TreeTCI's ranks never exceed the cap
    (M1 record, `tree-interpolation-engine-seam.md` lines 313-318). The test
    `DenseEngine` returns the exact network, whose rank can exceed the cap.
    - Proposed: an engine error whenever the driver would use such a
      network, plus a one-sentence M1 amendment ("an outcome network's bonds
      never exceed the cap") and a test engine that truncates at the cap.
    - Honest cost: for a blocked patch this stops the run with an error,
      which contradicts "the run does not stop with an error". The
      justification is that the cause is a broken engine, not a missed
      tolerance, as with the existing engine errors.
    - The alternative "treat it as not eligible and split" works for a
      capped-eligible patch, but it is undefined for a blocked one. A
      blocked patch cannot be split, and accepting a malformed or over-cap
      network would violate the bond cap or the layout. So no error-free
      resolution exists for the blocked case.
11. **No retry after a failed capped measurement.** Proposed, because the
    rank is already exhausted and added pivots only raise the starting rank.
    The alternative uses the M3 retry rule.
12. **Which network a blocked patch keeps.** Proposed: the last run, which is
    simple and involves no selection. Keeping the best of several sampled
    measurements would select the luckiest sample. With exhaustive
    measurement, keeping the best would be safe, but it needs every run's
    network.
13. **The `max >= min` validation (a new constraint).** With `max < min`,
    capped patches with `max < active_sites <= min` reach the "minimum wins"
    rule, so the maximum has no effect there. Proposed: reject it as
    `InvalidInput` before any evaluation. Alternative: allow it and document
    the precedence.
14. **The maximum's name and `None` semantics.**
    `max_capped_patch_bits: None` disables acceptance through the capped-size
    option; the provisional minimum-wins rule can still retain a capped
    patch. This differs from `max_patches: None` ("no limit"). Options:
    - (a) keep the name, with the emphatic rustdoc shown above;
    - (b) rename it to `accept_capped_up_to_bits: Option<usize>`;
    - (c) a field `capped_patches: CappedPatches`, with the variants
      `CappedPatches::Split` (default) and
      `CappedPatches::AcceptUpTo { bits }`.

    Proposed: (c), which makes the default self-describing. Its rustdoc
    states, as for `min_patch_bits`, that a bit is one active site whatever
    its dimension. The engine's early exit is not a reason for (c): it would
    be an engine-side hint in the M1 problem, not a driver acceptance
    policy, and stays out of this field.
15. **Capped acceptance on sampled measurements.** M3 OQ4 warned about it.
    The corner-localized misses were large patches (2 to 3 fixed sites)
    accepted on uniform samples, and Q4 accepts more such patches. On binary
    layouts, if the maximum is at most `log2(max(max_exhaustive_points,
    samples))` (10 with the defaults), every acceptance through the
    capped-size option has an exhaustive measurement. This does not cover
    an oversized patch retained by the minimum-wins rule if `max < min`
    remains allowed. It is certified only when the patch meets its
    allowance; a blocked, failed patch retained as `ToleranceNotMet` keeps
    its measured-error bound without certification. Under unit A this
    relation is inexact on fused layouts, because a bit can stand for more than two
    points. Should this be a documented recommendation, a default, or a hard
    constraint?
16. **Early exit (not decided; interaction only).** For a capped run that is
    not eligible (above the maximum, or no maximum), the sweeps after the
    first saturation are wasted except for the recycled pivots, because the
    patch splits whatever its error. Only eligible and blocked patches
    benefit from full sweeps. The engine does not know the patch size, so an
    early exit restricted to such patches needs a hint in the M1 problem. An
    early exit also lowers the quality of the capped networks that Q4
    measures.
17. **Unbounded overrun and mixed biases.** A run with `ToleranceNotMet`
    patches can exceed `delta` without limit, and unused budget is not
    redistributed. With audits off, `measured_rms` mixes downward-biased
    acceptance statistics with upward-biased failed ones. The report states
    this; no further mitigation is proposed.
18. **Sibling merging (Q3, undecided)** must respect both bounds: a merge
    must not create a capped patch above the maximum. Whether a
    `ToleranceNotMet` patch may be merged is not decided.

## Out of scope

The split-site selector (Q1), sibling merging (Q3), the engine's early exit,
unit B, the notice about non-binary sites (open issue 3), the corner-miss fix
(M9), and any change to TreeTCI.
