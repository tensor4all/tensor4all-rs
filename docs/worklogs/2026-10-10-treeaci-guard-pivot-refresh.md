# TreeACI Guard discoveries and old-pivot preference

[AI Supplied] Continues #854 after #870 merged. #874 is a confirmed regression
in the new retention behavior, distinct from #794's unconfirmed claim of
mixed-capacity searches whose pivots cannot be injected. No RSI algorithm or
workload is executed.

## Defect and chosen repair

The 162-entry five-node analytic target is a sum of three separable products,
with one weak term. At relative tolerance `1e-9`, cap 3, seed 17 and the default
Guard, pre-#870 main `cd8a1f8a` converges in 3 passes with guard history `[1,0,0]`
and dense maximum relative error `5.3851e-10`. Main `72d02e0e` reaches `MaxSweeps(20)`,
returning a failing pivot every pass, with dense maximum relative error
1.04499e-8. The cap is sufficient for exact rank 3 representation; the local
matrices stay below their threshold. This is not false convergence.

An owning-module observation with 8 starts/4 search sweeps shows point
`[0,1,1,1,2]` repeatedly injected into cut 3-4, whose available capacity is 1;
the other cuts have no headroom. Subsequent updates discard it and recover
exactly the same output after two passes. Retaining a locally admissible old
cross prevents the fresh neighboring pivot movement that previously repaired
this residual. These observations concern an actually injectable point,
not #794's original non-injectable-pivot acceptance condition.

Successful injection now requests fresh LUCI for the next whole directional
pass, including saturated neighbors. A complete successful pass consumes the
request and restores ordinary old-pivot preference. A failed pass leaves it
pending; empty/no-op/error injections cannot create or erase a pending request.
The flag is published only in injection's no-failure commit block, preserving
rollback. One Boolean adds no sample scans, histories or tensor-sized work.

Refreshing only the padded cut is insufficient because its candidate samples
feed neighboring matrices. Keeping rank padding as a floor or relaxing the
Guard criterion would evade the underlying pivot-selection problem. Neither
is used. Configured tolerance, rank cap, Guard, RNG, revalidation, and per-cut
stability criteria are unchanged. Core's retention APIs are unchanged.

## Investigation limits

A new current-main public-entry scan of 6000 seeded cases covers a chain and
two branching trees, physical dimensions `[2,3,3,3,3]`, four analytic/perturbed
target families, relative tolerances `1e-3`/`1e-6`/`1e-9`, and Guard margins 1/10.
The production scheduler performs 16704 searches, 12598 at mixed capacity,
with 14851 returned pivots and 2591063 target evaluations. No non-injectable
search is observed. Terminations are 3990 Converged, 1661 RankLimited,
349 MaxSweeps. This is negative evidence for #794, not proof of absence;
MaxSweeps alone is not a bug certificate. The scan remains an ignored local
diagnostic rather than a committed parameter-search test.

## Validation and limits

The new default-Guard public regression now returns Converged in 3 passes,
guard history `[1,0,0]`, 1069 evaluations, and independent dense maximum
relative error `5.385121771281499e-10`. Both f64 and Complex64 regressions check
termination, full dense residual, bounded passes/evaluations, and cap
preservation. The injection lifecycle regression checks no-op/error behavior
and failed versus successful passes.

Changed-crate debug tests and doctests: **270 passed**, 10 existing opt-in
profiles ignored. Both existing slow G0 numerical regressions pass separately
in release. All-target Clippy with warnings/error-doc/panic-doc lints denied,
production Clippy with unwrap/expect/panic/todo/unimplemented denied,
formatting, public-error-doc checks and deterministic repository-rules review
pass. Hosted CI remains authoritative for complete coverage and rules review.

All 15 original #784 DMRG/window/well public replays still converge with their
unchanged settings and exact public history-prefix checks. All six full n50
DMRG outputs are byte-identical to the previously validated #870 outputs:
passes 6/9/6/6/6/12 for chi40/60/80-cap160/80-cap640/100-cap200/100-cap800.
These cases have no Guard discoveries, so ordinary pivot preference is
unaffected.

| Conditioned window | #870 passes | Refresh passes | Independent relative Frobenius error |
|---|---:|---:|---:|
| chi40 n26 | 4 | 7 | 1.49239e-6 |
| chi40 n28 | 4 | 5 | 1.68408e-6 |
| chi60 n33 | 7 | 9 | 7.07888e-7 |

The independent TT inner-product calculation has about `1e-7` relative error
resolution; these stay at comparable scales. Some cases require more passes;
there is no blanket speed or accuracy improvement claim. Raw/RMS well seeds
1/2/3 converge in 4/5/8 passes. Exhaustive 65536-entry comparisons give maximum
relative errors `1.38303e-8` to `2.07400e-8`, within the unchanged Guard-margin
criterion `1e-7`; Frobenius errors range `7.83135e-9` to `9.58894e-9`.

The actual R=10 NBlock W replay converges in 6 passes instead of 8, with the
same absolute tolerance `1e-4`, seed0 and cap4096. Its independent 1086-point
seed671 check has all finite values and maximum absolute residual
`1.2135898812378802e-4`, satisfying the unchanged `10 * tolerance` gate.
This is sampled evidence, not a full-grid maximum-error certificate.

Generated traces, binaries, hashes, input/output cores and independent
quality calculations remain in the ignored downstream experiment directory.
No coverage or existing numerical test threshold is relaxed. No production
paths or existing tests are removed.
