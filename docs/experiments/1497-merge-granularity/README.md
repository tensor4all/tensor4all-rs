# Serial merge granularity diagnostic (#1497)

## Scope and validity

This follows the [whole-region baseline](../1497-baseline/README.md). It is a
single-revision serial diagnostic, not a parallel implementation or a speedup
comparison. Source is pinned to `5fbd73e5`, which integrates #1407/#1511 and
contains upstream `62c3b606`. [Protocol](protocol.json), [source/binary/tool
pins](raw/manifest.json), [all matrix outcomes](raw/attempts.json).

All six processes in the declared two-cell matrix complete and are CLEAN.
The accepted preflight also has three CLEAN processes: R4 tracing off/on and
the counts-only tiny fixture. Off/on signatures agree on every reported field
except the declared frontend/producer seconds. The first preflight is retained
under [raw-preflight-contaminated-1](raw-preflight-contaminated-1/manifest.json):
its tiny fixture passed numerically but was CONTAMINATED (4.1% foreign CPU,
3% limit). The entire preflight was rerun before starting the matrix, with
unchanged cases, thresholds and repetitions. Its timings do not enter tables.

The tiny two-bit Exact fixture directly records incoming widths `[1, 2]` and
asserts four final regions and dense values `[0.5, 0.5, 0.5, 0.5]`. This pins
the count's meaning; it has no input-sibling additions and establishes no
performance opportunity. One original leaf still means one leaf application.

Both real-workload-derived cells keep the original M6a generator, seed 42,
T=0.1, `rtol=1e-6` and certified share 0.5. R4 uses G0 cap 8 / transform goal
16 and full 4096-point DFT checking; R6 uses G0 cap 16 / goal 64 and the existing
4096-sample/four-slice check. IR and production R10 are outside this follow-up;
their original failures remain in the earlier baseline.

## Measurement definition

One incoming region job is one `(spectator cell, per-axis prefix)` map entry
before an executed transition. Its scope contains the complete decision for
that key. Candidate scopes cover own-region work, an axis probe including its
children, or joint/uniform children. These scopes are serial in this run.
An own candidate can return early when it meets the rank goal.

Within each child `restrict_and_merge` call, a parent group is all source
items mapping to one input parent. The probe records each original item's
restriction and any sibling addition in the original traversal order, then
aggregates their durations by parent for inspection. It never reorders the
input loop or balances a sum. Per-probe item/addition totals and nested time
accounting are checked, including agreement with the public additions counter.

"Eligible addition-parent share" sums parent groups containing additions,
only in child probes with at least two such groups, divided by that process's
Fourier timer. It implements the declared 20% need screen with actual same-probe
width. It is a screen for further design, not a scaling result or memory-admission
proof. Sequential child probes/candidates are never pooled into concurrent width.

## Results

Times/shares are medians of the three per-process values, not pooled job medians.
Widths are observed maxima across the retained processes.

| Case (G0 cap / transform goal) | Fourier seconds | Region seconds | Parent-operation share | Eligible addition-parent share | Max incoming regions | Max addition-parent groups in one probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m6a-r4-cap8 | 0.861 | 0.278 | 32.7% | 30.0% | 1 | 35 |
| m6a-r6-cap16 | 33.542 | 29.510 | 88.1% | 74.8% | 1 | 38 |

The following is the fixed first repetition, not the fastest sample. The
hottest eligible child probe is chosen by its own elapsed time. All repetitions
and every group duration are retained in raw stderr; [summary.json](summary.json)
contains per-process aggregates and the same hotspot selection.

| Case, fixed repetition 0 | Hottest eligible child probe | Parent groups / with additions | Parent seconds | Largest parent seconds | Largest parent's fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| m6a-r4-cap8 | level 2, region 0, own | 14 / 12 | 0.118 | 0.020 | 16.8% |
| m6a-r6-cap16 | level 2, region 0, own | 19 / 12 | 5.754 | 0.973 | 16.9% |

| Case, fixed repetition 0 | Candidate label | Calls | Seconds |
| --- | --- | ---: | ---: |
| m6a-r4-cap8 | own | 4 | 0.277 |
| m6a-r6-cap16 | axis-0 | 1 | 5.932 |
| m6a-r6-cap16 | axis-1 | 1 | 5.198 |
| m6a-r6-cap16 | own | 6 | 18.517 |

- m6a-r4-cap8: axis-candidate width 0; same-region multi-axis candidate work 0.0% of Fourier time.
- m6a-r6-cap16: axis-candidate width 2; same-region multi-axis candidate work 33.1% of Fourier time.

All formal processes finish with `region_count = 1` and `pruned_regions = 0`.
No refinement was selected: R4 meets the own goal; R6 probes two axes but keeps
own. Output keys never re-join. Starting with one root, this explains width one;
a selected split or multiple initial spectator cells can change that conclusion.
No repeated realistic multi-region case was measured here.

Both cells pass the declared parent-work need screen. Prioritize a design of
input-parent jobs within one child probe: that opportunity includes the own
candidate as well as axis probes. R6 also has a narrower axis-candidate option;
its two candidates belong to one region decision, rather than two whole-region
jobs. R4 takes the own-candidate early exit without axis work. These observed
durations do not make either option safe to dispatch concurrently.

## Interpretation and limits

The observed parent-group distribution can justify refining that seam's design;
it cannot establish a lane speedup. A parent job would need to preserve its
original sibling fold, while the coordinator replays scalar contributions in
canonical source/term order. Group subtotals can change floating-point results.
Dispatch/join must fit each child probe's admission and early-exit boundary.
No worker borrows a peer engine, and aggregate memory admission remains unsolved.

Parent times exclude collector-map updates and logging; candidate/region and
frontend times include nested diagnostic overhead. Explicit emission durations
are retained, but clock/metadata overhead is not fully removed. The largest
parent fraction describes a cost distribution, not an ideal speedup prediction.
Thread counts at region boundaries and actual command affinities are recorded;
OS thread counts are not a numerical worker budget. One logical CPU was used.

The new serial source `5fbd73e5` is the head merged by #1407 as `9ddffd17`.
All consumed crates, benchmarks and the lockfile match at those revisions.
These numbers do not re-time the original `8ce72e4e` baseline or the separate
[#1130 acceptance campaign](../issue-1130-implementation/acceptance-2026-11-20/README.md), and do not
promise production-resolution or IR performance.

## Reproduction

Create a detached checkout at the protocol's full baseline commit, then apply
[profiling.patch](profiling.patch) (which includes the diagnostic lazy reader),
or run [instrument.py](instrument.py) on that clean checkout. Numerical calls
and their order are preserved. The helper uses only the standard library; no
new production dependency or lane implementation is added.

Run `taskset -c 0-7 python3 run.py --checkout /path/to/detached-checkout
--target-dir /path/to/owned-target` from this experiment directory, in a fresh
copy without existing `raw`. Builds use `ai-run build --mem 6G`; all finished
binary executions use exclusive `ai-run measure --mem 8G`, CPU 8, with 300-second
process caps. `ai-run` is external; its entry-point hash is in the manifest.
The current controller resolves it through `PATH` before building. The exact
controller used for these records is retained as [run-recorded.py](run-recorded.py);
its bytes still match the historical controller hash. Raw paths remain the
original execution provenance; no measurement record or pin was rewritten.
Choose and declare appropriate CPU sets before a campaign on another host.

`python3 report.py` regenerates tables and derived per-process summaries from
the committed stdout/stderr/guard records. Those derived summaries and Cargo's
full artifact listing are local caches; the authoritative measurement records,
build logs and pins are committed. Keep the contaminated preflight too.

`python3 verify.py` checks retained controls, tool/source-patch pins and process
counts. Add `--checkout /path/to/instrumented-checkout` to compare all captured
source bytes, and `--binaries` while the original binaries remain available.
Both patch application and instrumentation generation were independently
reproduced at the pinned baseline and matched all eight captured source hashes.
