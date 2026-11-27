# Merge-refine parallel integration plan (#1497)

## Scope and decision order

This accompanies the [design proposal](./merge-refine-parallel-execution.md)
and the [whole-region](../experiments/1497-baseline/README.md) and
[finer merge](../experiments/1497-merge-granularity/README.md) diagnostics. It is a plan
for maintainer review, not authorization to implement the lanes variant.

The original inspected baseline is `8ce72e4e`; the finer diagnostic uses
`5fbd73e5`, integrating #1407 and merged #1511 over main `62c3b606`. The [symbol inventory](../experiments/1497-baseline/dependency-inventory.json)
records baseline source locations, not instrumentation-shifted line numbers.
The scope is the numerical call chain consumed by patched Fourier; unrelated
parts of the workspace's engine migration need not be completed for their own
sake before this work.

This step has no speedup acceptance gate. The 20% phase share is a baseline
need screen; useful independent width means at least two numerical jobs in
one dispatch wave. A later candidate campaign requires maintainer-approved
numeric speedup, non-regression and resource thresholds before measurements.
No CLEAN run can substitute for those missing performance criteria.

| Order | Owner / seam | Required change or decision | Evidence before advancing |
| --- | --- | --- | --- |
| 0 | Maintainers, #1497 / #1407 | Review input-parent job design as the next scope; keep leaf-only execution optional; settle numerical profiles, memory semantics and final merge-policy admission rules | Explicit design review; current PR's source and ledger read after its final update |
| 1 | Benchmark owner | Use the repeated M6a parent-work screen for further design; obtain repeated IR evidence before claiming an IR need; recheck affected costs if the consumed policy changes | Declared valid inputs, complete CLEAN samples, the 20% need screen and useful independent width; a feasibility observation alone cannot pass the median gate |
| 2 | tensorbackend / core, #1376 | Carry a borrowed engine and compatible runtime/placement through the canonical tensor operations used here | Routing tests with global defaults unavailable; typed mismatch errors; no new implicit-engine site |
| 3 | TreeTN numerical owners | Propagate that engine through exact/certified/measured application, restructuring, projection, canonicalization, sums, truncation and residual/norm operations | High-level operation tests on the consumed paths; no downstream storage access or duplicated algorithm |
| 4 | M7 numerical owners | Establish deterministic contraction plans and decompositions on the supported profiles | Bitwise cross-thread/lane tests, including site-free/multi-site and 32/64-bit real/complex paths; fresh internal IDs cannot determine order |
| 5 | reconstruction | Design and review one-child-probe input-parent grouping, ordered accounting events and admission barriers; only then extract the shared serial seam | Serial equivalence of tensors, scalar folds, counters, selected candidates, errors and admission; independent small numerical oracle |
| 6 | reconstruction lane adapter | Implement only the reviewed seam on supplied engines; leaf-only staging remains an alternative if separately selected | One/several lanes, uneven costs, bounded windows, full joins, canonical errors, allocation-domain compatibility and output lifetime |
| 7 | Benchmark owner | Test a complete serial/one-lane/multilane campaign | Predeclared acceptance thresholds, all cases retained, CLEAN reports, startup/tail/RSS data; no extrapolated speedup |
| Completed diagnostic | Merge/probe owner | Record finer own/axis/child/input-parent costs without changing numerical order | Six CLEAN processes: parent-work screen passes in R4/R6; ordering, engine and aggregate memory proofs remain prerequisites |

These are suggested change boundaries, not assignments to other contributors.
Each owner keeps its existing algorithm, typed errors and public abstraction.
No new public crate, matrix container, scalar-specific Rust entry point or
direct downstream tenferro dependency is needed.

## Existing seams and gaps

- **Frontend/preparation:** `patched_fourier_transform` constructs the Fourier
  operator, calls `ReconstructionTarget::from_partition`, then schedules it.
  Its current signature has no engine. Target preparation restructures once,
  materializes terms, chooses centers and fixes certified budgets before the
  leaf loop. All of these numerical steps need the supplied engine too.
- **Applications:** `apply_linear_operator[_to_indices]` and
  `apply_linear_operator_certified[_to_indices]` currently take no engine.
  Certified contraction also calls canonicalization and QR/SVD. Changing only
  the patch loop would leave those numerical descendants implicit.
- **Merge tail:** `SubDomainTreeTN::{project,add,truncate,norm}` reach TreeTN
  sums, truncation and norm/canonicalization. Residual measurement builds a
  direct-sum difference. Existing context/session variants are migration
  surfaces, not a reason to add more twins or adopt tensors per lane.
- **Core:** the crate-private `with_eager_on` already validates runtime,
  engine and placement, then calls `Runtime::with_eager` with a borrowed
  engine. This is useful infrastructure, not proof that the public numerical
  call chain uses it. The current ordinary calls still name `EngineRoute::Legacy`.
- **Determinism:** existing cached-evaluator parallelism tests compare bits on
  selected raw f64 chain paths. They do not establish generic site-free or
  multi-site contraction, f32/c32, complete application, or SVD/QR reproducibility.
  Use the [M7 prerequisite](./tree-adaptive-patching-roadmap.md#m7-shared-memory-parallel-execution-l),
  not issue #1475, which concerns key encoding/hashing.
- **Placement/AD:** compatible CPU allocation domains and shared runtime
  identity need validation across stages. The ordinary serial path's gradient
  metadata must survive; interpolation's plain-host publication convention is
  not a substitute. A shared runtime's metadata locks also need scaling checks.
- **Resources:** Hataori's adapter owns lane dispatch; the caller's engines
  retain compute teams. There is no aggregate memory admission contract here.
  `CertifiedApplication::estimated_bytes` is returned after work; it is not an
  existing preflight reservation API for a patch scheduler.

## Dependency on the evolving tolerance policy

The optional-candidate resource failure at the earlier inspected `ddc0c617`
head is fixed in #1407 and covered by the nonuniform two-bit reproduction,
work/term limits and related axis/joint paths. `RegionTrial::probe` maps only a
typed `ResourceLimit` to an infeasible optional candidate; other errors still
propagate, and the mandatory unsplit baseline must fit. This is verified source
at `5fbd73e5`, not a claim based on an earlier discussion comment.

Accuracy budgets are still computed before the region loop, while
`RetainedWork` from each chosen region carries into the next `RegionTrial`.
Admission therefore remains canonical and serial. Fixed accuracy budgets do
not make different region decisions independent resource transactions.

[#1511](https://github.com/tensor4all/tensor4all-rs/pull/1511) is merged as
`1ad44e07` and integrated into #1407. The shared private
`MergeCounters::record_refinement` counts selected coverage cuts only. Dense
one-/two-axis tests exercise Exact, RankGoal and Tolerance through both the
scheduler and public Fourier frontend, including unselected output, retained
error bounds and Tolerance's per-level ledger. Discarded probes are not added
to selected coverage cuts. The two-bit Exact fan-out fixture additionally
asserts the smallest multi-region trajectory. #1407 remains under review;
freeze the consumed seam after that review, without waiting for unrelated
engine migrations elsewhere in the workspace.

Tolerance also charges only the kept candidate's error; the baseline rank-goal
schedule conservatively spends on discarded probes. Parallel integration
must preserve the selected policy's own ledger, not transplant one policy's
accounting into the other. The leaf-only design keeps that whole tail serial
and avoids prematurely fixing a public merge-job contract.

## What a finer merge investigation must decide

Whole output regions have width one in every completed baseline. This rejects
that scheduling unit for these cases, not every possible parallel merge.

The [declared finer diagnostic](../experiments/1497-merge-granularity/README.md)
now supplies that cost screen at the integrated source. Median eligible
addition-parent shares are 30.0% (R4) and 74.8% (R6), with observed same-probe
addition-parent widths up to 35 and 38. Whole-region width remains one. R6
also executes two axis candidates within one region decision; R4 takes the
own-candidate early exit. These are serial instrumented costs, not speedups.

The alternatives are:

1. **Complete probe/candidate jobs.** Finish the unsplit candidate and its
   early-stop test first. Independent axis candidates may then run, with a
   join before selection and any joint candidate. Keep each candidate's child
   order and early resource exits inside its job. Extra speculative work must
   not silently alter executed-work counters or error accounting.
2. **Input-parent groups inside a probe.** Preserve each parent's original
   sibling fold, but retain original input/term ordinals for scalar replay:
   different parent groups can be interleaved in the serial traversal.
   Form groups once; do not rescan every input for every parent. This split
   has more accounting and scheduling machinery. The measured repeated cost
   supports designing this seam next, including own candidates that axis-only
   dispatch cannot distribute.

In both cases a serial coordinator owns decisions and canonical admission.
It dispatches from outside engine callbacks, joins before dependent steps,
and never tries to borrow idle peer engines. A width-one or dominant sibling
fold can remain the critical path; a new task graph does not remove it.

## Proposed input-parent seam: review before extraction

1. At entry to one child probe, traverse the ordered input map once. Assign
   each source its original ordinal and group references by `input_parent`;
   retain each group's source order. No repeated full scans or global tensor
   copies. A job owns only that parent's evolving `Item`, while immutable
   source items and projector belong to the joined probe scope.
2. Keep restriction, sibling addition, term traversal, truncation and residual
   decisions in each parent's original sequence. Return the resulting item
   and ordered scalar/counter events tagged by source and term/operation
   ordinal. A worker's subtotal is insufficient: parents may interleave in
   the original traversal. The serial coordinator must replay every floating
   contribution, checked integer update and fallible finite check in original
   order, including dropped-error and incurred-error updates.
3. Define and test that event seam on the serial path first. A local fold can
   change rounding or report a later overflow before an earlier canonical
   error. Select the first error in original operation order and fully join
   outstanding work before returning. Speculative physical work needs explicit
   telemetry separate from the existing serial algorithm's logical counters;
   agree on that distinction before exposing a parallel report.
4. Join and replay before the child probe's admission check. Preserve the
   candidate's child order, resource early exits, own-candidate early stop,
   axis selection and selected-candidate ledger. Do not dispatch a later child
   or another region speculatively under the current admission contract.
   Candidate-level parallelism remains an alternative requiring its own proof.
5. Dispatch bounded parent windows from outside engine callbacks. Each lane
   uses only its supplied engine; no nested lane pool or borrowed idle peer.
   Engine teams count toward the total CPU budget. Validate runtime identity,
   allocation domains, AD metadata and output lifetimes on these exact paths.
   One lane and insufficient width take the same serial algorithm.
6. Account for resident inputs, unsplit/retained candidates, in-flight parent
   scratch, completed items and event buffers together. A job count bounds
   concurrency only. Without an owner-provided reservation/admission contract,
   no hard byte guarantee is available; keep the affected path serial where
   that guarantee is required. The measured serial profile proves neither
   concurrent peak memory nor safe engine/metadata sharing.

This is an internal design boundary, not a new public job API. Do not add an
accounting recorder or lanes implementation just to make the diagram real.
First review the ordering/admission contract, then extract a serial seam with
known-value and trajectory tests, then evaluate a parallel implementation only
with agreed numerical, runtime and resource acceptance thresholds.

## Review requests and limits

- Review input-parent work within one child probe as the next design scope,
  given its repeated cost and width. Leaf-only execution stays optional.
- Confirm whether the ordered-event seam preserves policy-specific reports
  and canonical errors, including discarded probes and resource early exits.
- Confirm engine profiles, lane/window CPU budgets and the memory contract;
  bounded windows alone provide no aggregate byte limit.
- Confirm #1407's final reviewed integration point before signatures or
  report semantics are frozen. Whole-region lanes remain deferred until a
  relevant repeated workload actually has both region cost and width.

The original R=10 process cap and invalid R=6 IR inputs limit the evidence.
The IR R=8 observation is not repeated or a full DFT error check. The new
six-process campaign covers only the declared downscaled M6a cases. There is
no parallel candidate or scaling result. README, guides and the usage skill
should advertise lanes only after implementation and numerical validation.
