# ACI stagnation investigation and conditional repair

Status: executed. Outcomes, evidence and classification are in the
[worklog](../worklogs/2026-09-25-aci-stagnation.md); confirmed defects are
tracked in [#776](https://github.com/tensor4all/tensor4all-rs/issues/776).
The text below is the original planning handoff.

Branch: `investigate/aci-stagnation`, created from freshly fetched
`origin/main` at `076a75e02735d0e8a2db55904e4f669cb212b91b` (includes #774).
Use this branch's dedicated worktree, not the dirty RSI worktree. The
[worklog](../worklogs/2026-09-25-aci-stagnation.md) records scope and constraints.

## Scope and authority

Investigate and repair bugs ONLY in native `tensor4all-treeaci`. The
SimpleTT-based `tensor4all-aci` is obsolete and is not an investigation,
comparison, repair or acceptance target. Throughout this plan, a "chain"
means a path-shaped native TreeTN processed by TreeACI, not the legacy crate.
Do not implement RSI or change its benchmark acceptance rules here. SRC is
not a comparator.

The author's Julia ACI is an optional historical/algorithmic reference, not a
correctness oracle or a quality ceiling. Maintainer context supplied by the
user: that implementation is maintained by someone else and its bug fixes
are less active than in this project. Do not assume it contains all necessary
fixes, copy defects for parity, or conclude that TreeACI is satisfactory
because Julia has the same or worse behavior. Our acceptance is based on
independent numerical evidence, invariants, useful progress and honest
termination semantics. Improving those properties beyond the reference is
part of the task; maintaining or repairing the Julia project is not.

The user authorizes the receiving implementer to investigate, then, IF a bug
is demonstrated, file a tensor4all-rs issue and fix it on this branch. First
search existing issues and use the existing issue if it already covers the
same defect. This is conditional issue-creation authority, not authority to
push, create/merge a PR, contact other upstream projects, or relax existing
test tolerances. This planning turn does none of those actions.

Preserve other worktrees and uncommitted work. Refresh `origin/main` before
execution, record any base change, and reproduce on that base before relying
on older results. Follow repository rules, API-first exploration, local build
resource policy and proportionate validation. Do not spawn agents unless the
user explicitly requests delegation for the execution.

## Primary sources and evidence discipline

Read the full [ACI paper v2](https://arxiv.org/pdf/2604.00037v2), including
Appendix B, before diagnosing algorithmic convergence. Record the PDF hash.
If a claim uses the author's
[AlternatingCrossInterpolation.jl](https://github.com/tensor4all/AlternatingCrossInterpolation.jl),
read the relevant initialization, local updates, factorization call,
frame/index updates and stopping logic in full and pin the actual Git
revision. Follow delegated numerical routines when the claim depends on
them. A fetched file on a moving default branch is not an immutable reference.
State whether each observed behavior comes from the paper, this reference
revision or a TreeACI-specific policy; reference parity is not acceptance.

Important locators: paper sections 2.3-2.6 (local updates and index sets), 3.1
(experimental numerical accuracy), and Appendix B, Algorithm 1, PDF lines
10-22 (sweeps and stopping). PDF line 21 is the stopping criterion; HTML
numbering differs. This is not a theorem that unchanged ranks imply unchanged
index sets or a fixed point, nor an unconditional global convergence theorem.

Initial Rust map, pinned to the base above; verify ranges before citing:

| File | Location | What must be checked |
|---|---|---|
| `crates/tensor4all-treeaci/src/global_guard.rs` | `find_global_pivots`, lines 35-198; scale at 92-102 | Starting-point scale, walk threshold, discovered candidates |
| Same file | `inject_global_pivots`, from line 200 | Duplicate cuts, accepted additions, capacity, rollback and output padding |
| `crates/tensor4all-treeaci/src/schedule.rs` | `run_local_sweeps`, lines 72-190 | Guard scheduling; found count versus injection result; timing of histories |
| Same file | lines 350-419 | Rank stability, convergence, rank limitation and local error scaling |
| `crates/tensor4all-treeaci/src/local_update.rs` | `materialize_and_factor_edge`; lines 313-345 | Local rrLU tolerance and rank-zero handling |
| `crates/tensor4all-core/src/matrixlu.rs` | lines 768-773 and surrounding pivot loop | Absolute pivot floor; distinguish from #774's rank-zero skeleton fix |

Record claims in an evidence table: claim, observed/inferred/hypothesis,
revision, file/function/lines or paper locator, fixture/configuration hash,
command, raw output, and counterexample/limitation. A source range existing
does not establish that it supports a conclusion. Do not invent pseudocode,
theorems, passing tests, numerical thresholds, timing or measurement coverage.

## Prior observations: leads, not acceptance evidence

The Claude session `3bd978bb-97d2-4da3-86c7-7934f89483a8` and a subsequent
audit used RSI-branch diagnostic binaries. Their source and logs are leads;
the receiving model must reproduce using a freshly built TreeACI-only harness on
this branch. Do not import the uncommitted RSI crate to make a reproducer run.

Seed 1, two-input Hadamard, unit-RMS input copies, global guard enabled:

| Fixture | Output cap | Local relative tolerance | Observed behavior |
|---|---:|---:|---|
| `rand_chi64` | 128 | `1e-14` | 20 and 200 sweeps both end at rank 94, relative Frobenius error about `6.289e-14`; guard continues finding pivots |
| Same | 128 | `1e-3` / `1e-4` | Both hit 200 sweeps, final ranks 10 / 14; final relative Frobenius errors about `4.609e-3` / `5.751e-4` |
| `gauss_0.1_0.9` | 12 | `1e-14` | Hits 200 sweeps, final rank 11 |
| Same | 12 | `1e-3` / `1e-4` | Converged in 3 sweeps, ranks 4 / 6; relative Frobenius errors about `1.374e-3` / `1.352e-4` |

The practical-tolerance random cases demonstrate that exhaustion is not
exclusive to round-off settings. They do NOT prove that every intermediate
iterate is identical, that the guard's candidates are false positives, or
that local tolerances certify a global Frobenius bound.

For exact reproduction, locate these historical files in the existing RSI
worktree (not expected to exist in this main-based worktree):

| Historical relative path | SHA-256 |
|---|---|
| `target/tree-rsi-bench/cases-v3-rms/gauss_0.1_0.9.json` | `84e990d78a6f25b345e7deb82c53194f45b5802602966ac24371987991aa56fe` |
| `target/tree-rsi-bench/cases-v4-rms/rand_chi64.json` | `7a7b8c5b01064295a0312c636354b1a7d557b35bdde1bad8f006a86c92d71122` |
| `benchmarks/tree-rsi/gate_p.py` | `5223710af7c988b358ba8d6edc15fdc4a72fe78d0698de09da758131ff1b23f2` |
| `benchmarks/tree-rsi/bonds.py` | `0ffef469339532bf3aa250abd433c35f76e8d37222d88eee9cd7a23eb789b3d1` |
| `benchmarks/rust/benchmark_treersi_bonds.rs` | `532966aae04a107aad8daf81a65964cebd865a1d1e180c6b6b6411919f0b564b` |

Copy fixtures only into owned diagnostic storage after hash verification;
never change the originals. Record layout, shape, physical-index order and
input scaling. Reduce to a small deterministic generator/fixture that can be
committed with the eventual regression, without local absolute paths or RSI
dependencies. If historical files are unavailable, report that exact replay
is unavailable and label newly generated cases as new, not reproduced.

## Classification: normal behavior versus a bug

Define stagnation operationally over a declared observation window using
output/residual changes, not rank alone. Initially trace all 20 passes and
selected full 200-pass runs; do not introduce a new production stopping window
before establishing the cause.

| Class | Evidence needed | Correct interpretation/action |
|---|---|---|
| Normal progress at fixed rank | Index sets/frames or gauge-invariant output change; independently evaluated residual improves | Not stagnation; do not stop just because ranks are constant |
| Normal cap limitation | Necessary approximation capacity is exhausted; oracle or controlled cap increase supports this; status is honest | `RankLimited` is expected; more sweeps alone need not help |
| Normal heuristic failure/slow progress | Finite local/random search misses structure or has not found useful directions; no violated invariant/contract demonstrated | An honest `MaxSweeps` is not by itself a bug; report limitations and unresolved cause |
| Precision-limited behavior | Residuals are comparable to a measured, scale/conditioning-dependent rounding uncertainty; controlled better-accuracy evaluation or rescaling supports it | Do not use a universal `1e-14` floor; finite-budget termination may be expected |
| Correct guard rejection | Independent evaluation verifies residual above the intended guard threshold | The guard must not be silenced merely to report convergence, even if local ranks are stable |
| Scale-policy inconsistency bug | Trace shows the intended relative-error contract changes with harmless rescaling, or incompatible thresholds cause reproducible ineffective updates away from round-off | File a reproducible issue; fix at the owner, preserving a meaningful residual contract |
| Injection/update bug | A genuinely new admissible pivot is reported but not represented/used as required, or a frame/skeleton/cache invariant fails | File an issue; repair the invariant, not the stopping label |
| Avoidable repeated no-op/termination bug | Measured repeated work has no state/residual benefit; trace proves a bookkeeping or scheduling defect against a stated contract | Distinguish found, accepted and useful pivots; fix wasted work while retaining failure information |
| False convergence bug | A status claims its documented checks passed when they did not, or the proposed fix suppresses a verified unresolved residual | Reject the shortcut; preserve non-success status and evidence |
| Harness/metric bug | Layout, scaling, seed, dtype, stale executable, norm mismatch or missing termination data explains the observation | Fix the reproducer/report; do not change ACI to accommodate invalid evidence |

Several classes can coexist. Reaching 200 sweeps proves only budget
exhaustion. Finding pivots proves neither actual injection nor useful progress.
Not adding new pivots proves neither accuracy nor convergence. A plateau can
be expected under current numerical settings while its handling is wasteful.

Keep three different quantities explicit: local factorization tolerance,
absolute guard threshold, and independently measured output error. Record
both absolute max error and relative Frobenius error for bounded dense cases;
they are not interchangeable. For zero/near-zero targets use an explicit
absolute criterion. Sampled checks on large cases are labeled sampled, never
global certificates.

## Execution phases

### A. Baseline and bounded reproductions

1. Record current base, source/build hashes, backend, CPU, thread settings,
   actual RNG algorithm/seed, fixture hashes and full options. Inspect the
   current API inventory before implementation sources. Read historical tests
   for #774 and guard/candidate handling so the investigation preserves them.
2. Write a minimal TreeACI-only harness. Print termination reason, per-edge rank
   history and per-pass local/guard metrics; an `Ok` result is not convergence.
   Use normal local test profiles first. Keep dense oracles bounded and
   materialize each whole result once per inspected snapshot, not per point.
3. Reproduce both historical leads, then minimize them. Freeze a diagnostic
   matrix before testing a fix: practical tolerances `1e-3`, `1e-4`, plus
   `1e-8`, `1e-12`, `1e-14` diagnostic lanes; seeds 1, 2, 3; sweep limits
   20 and 200; cap-limited and generous-cap runs. If the cross-product is too
   costly, declare a bounded subset and its rationale before candidate results.
4. Include normal early convergence, zero and sparse/isolated-feature inputs,
   cancellation, compressible random inputs, separated peaks, and real/complex
   paths. Include a native branch (not only chains). Control root/traversal
   and input scaling independently. Require the guard to retain its ability to
   discover missed features; include a fixture where later sweeps help at
   constant rank to protect against an unsafe rank-only stop.

### B. Trace the causal chain

Use private/test-only, bounded instrumentation, not a new public tracing API
or an unbounded cache. Count and distinguish:

- Local matrix scale, retained pivot cutoff, absolute floor, Schur residual,
  chosen indices and rank before/after every affected edge update.
- Guard starts and their target values, scale estimate, absolute threshold,
  candidate coordinates and independently checked residuals. Keep searches
  and random streams aligned when comparing diagnostic variants.
- Found candidates, unique candidates, existing projected cuts, available
  capacity, actually accepted additions (the injection return value), frame
  changes and post-injection ranks. The current scheduler records `found`
  although injection reports a separate accepted count; examine this without
  assuming that replacing one count with the other is a valid fix.
- Whether each accepted candidate survives the next local update; evaluate
  residuals at those SAME coordinates before injection and after the next
  pass. Inspect canonicalization and cache validity where the trace points.
- Output changes on fixed holdouts or bounded dense oracles, separately from
  index-set changes and gauge-dependent core changes. Record peak intermediate
  ranks including post-injection padding, not only end-of-pass maxima.
- Phase time and evaluation/update counts. Diagnostic tracing is outside
  performance acceptance; profile again with tracing disabled.

Explicitly test competing hypotheses: underestimated guard scale; valid guard
residual but local truncation too coarse; duplicate/rejected candidates counted
as progress; cap/growth mismatch; candidates discarded by updates; stale frames
or caches; absolute floor/underflow; initialization or local-search limitations;
and metric/harness mistakes. A source-code suspicion is not causal proof.

### C. Controlled diagnostics and bug decision

Vary one factor at a time. Diagnostic-only guard-off runs may isolate cost,
but cannot be promoted as a fix. Compare the existing scale to a clearly
defined alternative and, on small cases, the exact target max norm. Do not
assume a cumulative maximum is correct until its semantics and regression
behavior are checked. Exercise homogeneous Hadamard rescaling with output
scale restored; do not assume arbitrary nonlinear callbacks are homogeneous.

Use high-accuracy or independent residual evaluation where rounding is the
hypothesis. Rank and fixed-seed replay alone cannot establish a fixed point.
Never assume error is monotone in cap or sweeps, perform binary searches on
that assumption, or turn interpolated ranks into measured integer ranks.

At the decision gate, deliver a table classifying each case, the violated
contract if any, minimal reproducer, causal trace, competing explanations
ruled out, and remaining uncertainty. If evidence is insufficient, label it
unresolved. Do not file a confirmed-bug issue or implement a speculative fix
just to finish the plan. If all behavior is normal, finish with that finding
and supported usage guidance; no forced code change or issue.

### D. Confirmed bug: issue, then correction on this branch

Search existing issues, including #773/#774 and related guard investigations.
Do not confuse the already-fixed rank-zero interpolation identity with a new
scale or stagnation defect. File or update a tensor4all-rs issue BEFORE the
production fix, using the bug-report template. Include immutable base/source
links, a small runnable reproducer, expected versus observed behavior, options,
raw trace excerpts, root cause, affected surfaces and explicit acceptance.
Record its URL in the worklog. Split unrelated root causes into distinct issues
and commits; do not expand into a wholesale numerical-backend rewrite.

Add a focused failing regression, verify it fails on the unmodified base,
then implement the smallest owner-level repair. Do not silently enlarge ranks,
loosen existing tolerances, disable guard, change callback semantics or mark
unresolved error as `Converged`. If justified, a documented non-success
stagnation status may avoid wasted work; first check existing status/API
semantics and bindings. It must not hide an update bug or replace accuracy.

Inspect related paths within TreeACI (native chains and branches, scalar
variants, guard and local updates) for the demonstrated defect. Shared
numerical helpers may be read or tested as dependencies to establish the
cause, not as a separate repair campaign. If a sound fix requires modifying
a shared owner outside TreeACI, report the dependency and request explicit
scope approval before doing so; do not add a TreeACI-local workaround that
violates layering. Do not audit or repair the obsolete SimpleTT ACI or the
author's Julia implementation. Keep unrelated defects outside this branch.
Preserve generic scalars, physical-index identity, configured backend/context,
bounded memory and AD boundaries. Update public docs if a status/option changes.

### E. Numerical, performance and final acceptance

- Focused regression fails before and passes after the fix; it asserts the
  relevant residual/invariant/status and productive-work counts, not just a
  smaller runtime, bounded rank or successful return.
- Normal, cap-limited, precision-limited, sparse-feature and branching cases
  remain correctly classified. No new false convergence or lost guard rescue.
  Run affected crate tests, Clippy, formatting and relevant docs/invariant
  checks. A separately approved shared-owner change additionally requires
  affected-dependent validation; it is not implicitly authorized here.
- For numerical accuracy, keep oracle/error definitions and acceptance limits
  fixed before comparing implementations. Improved runtime with worse accuracy
  or a misleading success status fails. No interpolated timing/bond estimate
  can replace the actual accepted run.
- Freeze a proportionate release performance protocol before candidate timing:
  exact cases, matched error requirements, seeds, paired order/repeats,
  providers, pinned threads/affinity, warmup and host validity criteria.
  Include affected cases and normal early-convergence controls. Separate
  fixed-iteration per-sweep overhead from legitimate earlier non-success
  termination; never call the latter improved accuracy convergence. Report
  raw paired results, variation and all regressions; invalid/noisy results
  remain inconclusive. Do not rerun only favorable cases.
- Review source attribution and final diff, update the issue/worklog, and
  report what is fixed, normal, unresolved or unavailable. No push/PR/merge
  without further approval. If later requested, synchronize main and rerun
  affected validation before declaring PR readiness.

## Handoff deliverables

- [ ] Fresh-base, reproducible TreeACI-only diagnostic harness and small fixtures.
- [ ] Pinned paper/Rust source map and raw evidence; pin Julia only if cited.
- [ ] Case-by-case normal/precision/bug/unresolved classification.
- [ ] Found/accepted/surviving pivot trace and independent residual evidence.
- [ ] If bug confirmed: issue URL, regression demonstrated failing on base,
      owner-level repair and passing focused validation.
- [ ] Guard rescue and normal early-exit behavior preserved on native TreeTN
      chains and branches; no legacy SimpleTT implementation involved.
- [ ] Acceptance justified independently of Julia behavior; no upstream
      deficiency used to excuse a TreeACI defect or incomplete handling.
- [ ] Honest performance/accuracy/status report and remaining limitations.

The receiving model starts with phase A, not with a preselected guard fix.
