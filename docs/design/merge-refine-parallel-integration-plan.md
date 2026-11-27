# Merge-refine parallel integration plan (#1497)

## Scope and decision order

This accompanies the [design proposal](./merge-refine-parallel-execution.md)
and [serial diagnostic](../experiments/1497-baseline/README.md). It is a plan
for maintainer review, not authorization to implement the lanes variant.

The inspected baseline is `8ce72e4e`. The [symbol inventory](../experiments/1497-baseline/dependency-inventory.json)
records baseline source locations, not instrumentation-shifted line numbers.
The scope is the numerical call chain consumed by patched Fourier; unrelated
parts of the workspace's engine migration need not be completed for their own
sake before this work.

| Order | Owner / seam | Required change or decision | Evidence before advancing |
| --- | --- | --- | --- |
| 0 | Maintainers, #1497 / #1407 | Agree on the initial leaf-only scope, numerical profiles, memory semantics and final merge-policy admission rules | Explicit design review; current PR's source and ledger read after its final update |
| 1 | Benchmark owner | Confirm a relevant repeated IR baseline and recheck phase shares if #1407 changes the consumed policy | Declared valid inputs, complete CLEAN samples, the 20% need screen and useful independent width; a feasibility observation alone cannot pass the median gate |
| 2 | tensorbackend / core, #1376 | Carry a borrowed engine and compatible runtime/placement through the canonical tensor operations used here | Routing tests with global defaults unavailable; typed mismatch errors; no new implicit-engine site |
| 3 | TreeTN numerical owners | Propagate that engine through exact/certified/measured application, restructuring, projection, canonicalization, sums, truncation and residual/norm operations | High-level operation tests on the consumed paths; no downstream storage access or duplicated algorithm |
| 4 | M7 numerical owners | Establish deterministic contraction plans and decompositions on the supported profiles | Bitwise cross-thread/lane tests, including site-free/multi-site and 32/64-bit real/complex paths; fresh internal IDs cannot determine order |
| 5 | reconstruction / quanticstransform | Extract preparation, leaf application and serial tail as shared private stages | Old-versus-extracted results, full reports, projectors and per-level trajectory; independent small numerical oracle |
| 6 | reconstruction lane adapter | Add ordered bounded leaf windows and singleton preparation/tail stages on supplied engines | One/several lanes, uneven costs, windows, full joins, canonical errors, allocation-domain compatibility and output lifetime |
| 7 | Benchmark owner | Test a complete serial/one-lane/multilane campaign | Predeclared acceptance thresholds, all cases retained, CLEAN reports, startup/tail/RSS data; no extrapolated speedup |
| Later | Merge/probe owner | Profile finer probe/candidate/input-parent units before designing parallel merging | Demonstrated independent cost and sufficient width; canonical accounting, early-exit and resource-admission proof |

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

At #1407 head `ddc0c617`, `tolerance::advance` computes region accuracy budgets
before its loop, but carries `RetainedWork` from each chosen region into the
next `RegionTrial`. `candidate` checks that running count, and `choose`
propagates probe resource failures. Fixed accuracy budgets therefore do not
imply independent candidate admission.

The [source review](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7210164923)
and [regression follow-up](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7210703006)
ask for a feasible unsplit/over-limit split regression. The inspected remote
source still contains the error propagation; a comment describing a fix is
not evidence that it is present at that head. Re-read the final source and
tests before relying on the result. The same admission seam serves work and
term limits and single-axis/joint candidates, so the follow-up audit should
cover those related paths as well.

The [latest maintainer reply inspected](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7212551191)
confirms the workspace migration is confined to the two defining/consuming
crates. External-consumer evidence, named performance cases and their benefit
trade-off remain requested there, along with the same regression test. That
reply does not change the inspected remote head or settle its admission rule.

Tolerance also charges only the kept candidate's error; the baseline rank-goal
schedule conservatively spends on discarded probes. Parallel integration
must preserve the selected policy's own ledger, not transplant one policy's
accounting into the other. The leaf-only design keeps that whole tail serial
and avoids prematurely fixing a public merge-job contract.

## What a finer merge investigation must decide

Whole output regions have width one in every completed baseline. This rejects
that scheduling unit for these cases, not every possible parallel merge.

Two alternatives deserve a separately declared diagnostic if merging remains
the target bottleneck:

1. **Complete probe/candidate jobs.** Finish the unsplit candidate and its
   early-stop test first. Independent axis candidates may then run, with a
   join before selection and any joint candidate. Keep each candidate's child
   order and early resource exits inside its job. Extra speculative work must
   not silently alter executed-work counters or error accounting.
2. **Input-parent groups inside a probe.** Preserve each parent's original
   sibling fold, but retain original input/term ordinals for scalar replay:
   different parent groups can be interleaved in the serial traversal.
   Form groups once; do not rescan every input for every parent. This split
   has more accounting and scheduling machinery and needs measured granularity.

In both cases a serial coordinator owns decisions and canonical admission.
It dispatches from outside engine callbacks, joins before dependent steps,
and never tries to borrow idle peer engines. A width-one or dominant sibling
fold can remain the critical path; a new task graph does not remove it.

## Review requests and limits

- Agree whether the first lane surface should parallelize only application,
  given the measured M6a bottleneck and the single IR feasibility observation.
- Confirm the lane-count window default and the documented absence of a hard
  total-memory bound. A strict byte budget needs an owner-provided reservation
  contract including residents and outputs.
- Confirm the #1407 integration point and the supported engine profiles before
  signatures or report semantics are frozen.
- Decide whether to commission finer merge profiling after those decisions;
  no performance implementation is justified by whole-region width alone.

The R=10 process cap and invalid R=6 IR inputs limit the current evidence.
The IR R=8 observation is not a repeated performance distribution or a full
DFT error check. There is no parallel candidate and no lane scaling result.
The public README, guides and usage skill should advertise lanes only after
the implementation and its relevant numerical examples are validated.
