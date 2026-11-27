# Parallel merge-refine execution

## Status and decisions for review

Proposal for [#1497](https://github.com/tensor4all/tensor4all-rs/issues/1497).
The serial baseline is `8ce72e4e`; its numerical sources match the initial
inspection at `379868e0`. Hataori is pinned to `092eecdd`, tenferro to
`5df83672`. No `_on_lanes` implementation is included. Maintainer agreement
precedes implementation, as the issue requires.

The [serial evidence](../experiments/1497-baseline/README.md) changes the
initial decomposition proposal: complete output regions are too coarse for
the measured workloads. Each completed schedule has one region at every
level, even when it has many input leaves. A large merge-time share does not
make whole-region fan-out useful.

| Issue question | Proposed answer | Reason and trade-off |
| --- | --- | --- |
| Unit of parallelism | Leaf applications are the first candidate. Keep the complete merge/refine tail serial initially. Investigate probe/candidate or input-parent work before proposing parallel merging. | Leaves are independent. Whole-region jobs have width one in these measurements; finer jobs need additional ordering and admission design. |
| Accuracy allocation | Preserve the existing allocation and arithmetic order. | Application budgets and each level's share can be fixed before its numerical work. Workers need no peer's measurements. |
| Sum/report order | Keep source, sibling, term and scalar-fold orders. | Sorting completed outputs alone cannot preserve intermediate truncation decisions or the next level's budget. |
| M7 prerequisite | Design now; expose lanes only after the consumed numerical paths are deterministic and engine-explicit. | A cached-evaluator topology/dtype allowlist alone does not prove application, QR, SVD or residual determinism. |
| Memory | Bound submitted work with a window; retain each application's existing limit. | This controls concurrency and buffered outcomes. Per-application estimates do not bound process memory or retained outputs. |

The [integration plan](./merge-refine-parallel-integration-plan.md) lists
owners, blocking contracts and acceptance evidence. In particular,
[#1407](https://github.com/tensor4all/tensor4all-rs/pull/1407) changes merge
policy, accounting and resource admission. Its final semantics must be
integrated before freezing a parallel merge seam.

## What the baseline establishes

The six-case diagnostic retained all fourteen processes, including failures;
every process received `verdict=CLEAN`. The report distinguishes repeated
measurements from the single IR feasibility observation.

- Small M6a has useful application work, but at R=6 application is only about
  11% of Fourier time and merging about 88%. Leaf parallelism alone has a
  limited opportunity there.
- The original IR R=8/cap-64 case has four applications and substantial costs
  in both phases. It is one observation, requiring repeated confirmation
  before an implementation priority is established for that workload.
- M6a R=10 reached application but exceeded the declared process time cap.
  The downscaled IR R=6 inputs were rejected by their producer. These are
  explicit limits of the evidence, not completed transform measurements.

No parallel speedup, production-resolution efficiency or hard memory ceiling
is established. The preregistered 20% phase-share screen is a need screen for
the measured Fourier front end, not an acceptance threshold for a candidate's
speedup. A job family also needs independent work to distribute.

## Numerical contract to preserve

The owners are
[`schedule.rs`](../../crates/tensor4all-partitionedtreetn/src/reconstruction/schedule.rs),
[`target.rs`](../../crates/tensor4all-partitionedtreetn/src/reconstruction/target.rs),
[`engine.rs`](../../crates/tensor4all-partitionedtreetn/src/reconstruction/engine.rs)
and the [Fourier front end](../../crates/tensor4all-quanticstransform/src/patched_fourier.rs).
These facts describe the baseline's exact/rank-goal schedule; the prospective
Tolerance policy in #1407 has a different ledger.

Preparation validates full indices, dyadic coverage and spectator geometry,
restructures the operator, and pins the reference scale. It applies once per
original leaf, then restricts images to disjoint spectator cells. An original
application error is charged once, even if its image belongs to several cells.

The allowance is `A = max(atol, rtol * reference_scale)`. Certified application
gets `share * A / max(original_leaf_count, 1)` per leaf. Application errors
are added in canonical source order. The remaining allowance is
`A - application_error`.

Each level first computes its charge bound `C` from the whole incoming level.
With a rank goal, its share is evaluated as `(remaining / levels_left) / C`;
zero charges give zero. Without a rank goal the share is zero. Preserve that
operation order. Unused reservations do not become a peer's extra budget.

The serial region body then merges genuine input siblings, preserves unpaired
leaves, probes its unsplit representation and axes, and may reuse one axis's
probes or construct joint children. A stopped axis keeps merging inputs in
later levels. In this baseline, discarded probes spend allowance but appear
in no retained error component. Kept components combine by an ordered `hypot`
fold within a level and add in level order across levels.

Consequently, balanced sums, reordered truncation, skipped probes, worker
subtotals or scale remeasurement would change the algorithm. For example,
positive binary64 contributions `a=1`, `b=c=2^-53` satisfy
`(a+b)+c != a+(b+c)`. Preserve the inherited scalar folds directly. Operator
construction error and backend roundoff remain outside `report.error_bound`.

## First implementation candidate: three stages on supplied engines

Use one common orchestration path with private stages at the existing owners.
The sequential entry borrows the caller's engine. The lanes entry executes
the same stages using the caller's `TenferroLanes`:

| Stage | Execution | Why |
| --- | --- | --- |
| Prepare | One supplied engine performs the existing numerical preparation after metadata validation | Construction, scale and restructuring run once, with no implicit coordinator engine or per-lane operator preparation. |
| Apply leaves | Ordered windows distribute complete leaf applications over lanes | Budgets are pinned. A job owns its application, center and temporaries. Exact/certified/measured bodies remain shared with serial. |
| Finish schedule | One supplied engine runs initialization, spectator normalization, all merge/refine levels and final assembly | Preserves the complete ledger and region-decision sequence, including future policy-specific resource admission. |

Singleton stages use the same lane adapter as wider stages. A lane callback
returns before the next stage is dispatched; it never nests another fan-out
while borrowing its engine. There is no lane-count-specific numerical branch.
The adapter need not pick the same physical lane for singleton stages, so all
accepted engines must have compatible numerical and allocation profiles.

Leaf jobs return ordered prepared images, provenance and typed results. Under
certified application they do not construct an exact reference. Measured
application keeps its exact reference, approximate image and residual in one
job. Preserve explicit seeds and RNG algorithms; none depends on lane number.

All applications succeed before the old level-zero error fold and spectator
normalization begin, matching the current preparation boundary. Hataori's
input-order output contract lets that fold stay unchanged. The serial tail
keeps existing sibling/term orders, `BTreeMap<OutputKey, ...>` region order,
axis and child generation order, finite checks and checked counters.

This first scope needs no new per-region accounting event recorder or generic
task graph. Do not implement those before finer merge parallelism is justified.
Extract serial stages first and compare with the old implementation; agreement
between two new executors alone cannot establish preservation.

The proposed names follow #1495: `schedule_merge_refine_on_lanes` and a thin
`patched_fourier_transform_on_lanes`. They are not callable APIs yet. Serial
base-name functions follow #1376, taking `exec: &mut dyn ExecutionEngine` last.
Do not add context wrappers, `_in`/`_on_session` twins or direct downstream
tenferro dependencies to bypass a missing high-level operation.

## Execution ownership and thread competition

Complete engine propagation through the consumed construction, application,
projection, addition, truncation, canonicalization and residual/norm paths.
Existing explicit low-level helpers do not establish that the whole call
chain uses the lane's engine. Require a routing oracle with global defaults
unavailable; a callback's thread identity is insufficient.

The numerical prerequisite includes the roadmap's deterministic cached
evaluator work, plus every application and decomposition path actually used.
Start reproducibility acceptance with homogeneous serial CPU engines. More
inner workers are a separate numerical profile requiring their own evidence.
Unsupported profiles return a typed refusal; no implicit engine substitution
or kernel fallback establishes the bitwise promise.

At the pins, Hataori starts one OS lane thread per engine per run. A serial
CPU engine computes there without a pool. A multiworker engine retains its
worker team for its lifetime, including idle stages. Budget for all lane
threads, reserved workers and the coordinator. Reducing admitted jobs does not
release the idle engines' teams.

The caller owns engine widths, lifetime, placement and affinity. The driver
creates or resizes no pools, changes no thread environment and borrows no
peer's engine. Provider nesting stays within that engine's existing placement
budget. The coordinator performs no simultaneous large numerical work.
This follows the [placement ownership rules](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/core-boundary-and-execution-ownership.md#where-placement-is-decided).

Serial engines are the initial recommendation for outer fan-out. A narrow
merge tail does not automatically gain the other lanes' workers. Using wider
engines instead can improve inner work only under a validated profile, and
their idle teams may compete during other stages. Dynamic outer/inner budget
rebalancing belongs at the execution owner; it is not a patch-driver fix.

Share prepared immutable operands and metadata. Existing application-local
canonicalization copies remain at their owner. A new driver cache or one
deep operator copy per lane would multiply storage and traffic. Even shared
reads can saturate bandwidth or cross NUMA domains; more lanes need evidence.

## Windows, memory and failures

Propose an execution-only `max_batch_items: Option<NonZeroUsize>`:
`None` resolves to the supplied lane count; `Some(K)` bounds pending outcomes.
It is separate from accuracy and Fourier options. Supplying several engines
is the caller's opt-in to concurrent working sets.

For resolved `K`, submit at most `K` canonical leaf jobs, join the window,
consume results in source order and release its discarded temporaries.
There are at most `min(lanes, K, jobs_left)` active callbacks. `K > lanes`
lets a lane take another cheap job while a peer is busy; smaller windows
reduce concurrency and buffering but can create a slow tail at every join.
Keep Hataori's existing dynamic queue rather than a second scheduler.

Do not use a memory semaphore inside callbacks. No callback waits for a peer,
a coordinator action or a permit while holding its engine; the coordinator
holds no shared lock during joins. This avoids circular waits. Lane startup
and bounded wasted work on failure are costs to measure, not hide.

Keep the certified `memory_limit` as the same per-application estimate.
With `q` simultaneous applications, plan for their simultaneous factor/sweep
working sets, plus residents, outputs, stacks and engine/runtime resources.
The bound is not a complete allocator or RSS cap. Exact and measured modes
have no equivalent complete byte bound. Retained images accumulate across
windows; a window does not bound the final result by `K`.

A hard aggregate limit needs composable estimates or allocation-backed
admission from the numerical owners, including retained outputs and caches.
Dividing a process limit by lane count can reject serially valid inputs and
does not provide that contract. Preserve #1279's bounded-block exact zip-up
and certified path without an exact reference. Parallelism cannot make a
single refused application fit.

Return typed job results as successful Hataori payloads, avoiding loss of the
source through its message-only callback error. Finish/join the current
window and return the first failing source ordinal; start no later window.
Speculative later jobs have private mutations and publish no partial result.
Executor start/join failure has its own typed error. Reuse the adapter's
unwind/join behavior without high-level `catch_unwind`.

## Values, allocation domains and AD

Every stage output must be readable by every engine that can receive it.
CPU placement alone does not prove this: validate input runtime/domain
compatibility and engine output-allocation policy through core/backend
contracts, then check outcomes before the next stage. Reject incompatible
domains rather than silently copying or adopting values.

Preserve the serial path's AD metadata. Do not copy interpolation's plain-host
publication convention for tracked values or create one eager runtime per
lane. Scalar norms used for decisions remain explicit boundaries. Missing
tracked/cross-engine operations return errors, without silent detach.
Gradient correctness or bitwise gradient accumulation requires separate AD
oracle coverage; forward preservation is not a new AD claim. Shared runtime
locks also need contention evidence before any scaling claim.

## Further merge parallelism: a separate decision

The measured width-one regions still contain numerical work. Candidate axes,
child probes and input-parent groups are possible finer units, with different
constraints. The current diagnostic does not measure their individual cost
balance or dispatch overhead.

A viable exploration keeps a serial coordinator: finish the unsplit candidate
before deciding whether probes are needed; expose only the probes the serial
algorithm would execute; join before axis/joint selection and resource
admission. Within an input-parent job, preserve the original sibling fold.
Never nest lane fan-out inside a callback or turn it into a balanced sum.

Such a split must reconstruct `incurred`, global dropped norms and counters
in the original input/term/probe order, preserving early-error boundaries.
Worker subtotals are insufficient. Extra speculative probes must not change
the report's executed-work counters. In #1407 Tolerance, earlier regions'
retained work affects candidate admission even though their accuracy shares
are fixed. Independent region decisions against stale admission snapshots
would change results.

Profile these finer units under a new declared protocol before choosing that
architecture. Do not require a general event recorder, task graph, persistent
lane service or aggregate allocator for the initial leaf-only candidate.

## Acceptance and rollout

This design/baseline step sets no numerical speedup or aggregate-memory
acceptance gate. CLEAN means an uncontaminated observation, not a performance
pass. Before a lane implementation is measured, maintainers must approve a
separate protocol with numeric speedup/non-regression and resource thresholds;
without that protocol no candidate can pass performance acceptance. Useful
independent width here means at least two numerical jobs available in one wave,
not a guarantee that their cost amortizes dispatch.

The integration plan separates these steps into reviewable changes:

1. Settle policy/admission integration and complete only the numerical engine
   paths consumed here, with the deterministic and placement prerequisites.
   Before performance implementation, confirm relevant repeated baseline
   evidence on the settled policy; the single IR feasibility run is insufficient.
2. Extract shared serial stages and prove old-versus-extracted results,
   report fields, partition and per-level trajectory.
3. Add one-lane execution, then several lanes/windows for leaf application;
   the serial tail uses the same body throughout.
4. Run a declared complete correctness/performance campaign on appropriate
   real-workload inputs, including failures, startup, tail and memory costs.

Tests cover exact/certified/measured application, zero allowances/depth,
cancellation, sparse/complete coverage, nonuniform input depths, stopped axes,
discarded probes, selected output regions, spectator normalization and
same-ID indices with different tags/prime levels. Cover real/complex 32/64-bit
data and supported trees, including degree-three and site-free/multi-site
nodes. Certified remains chain-only; Fourier keeps its path-selection
contract and rejects an empty input with no site topology.

Compare every floating report field by bits, structural counters and
`work_items_per_level` exactly, and coefficient bits under deterministic
node/leg correspondence. Allow only consistent renaming of fresh internal
bond IDs, never changed external full indices or a new gauge normalization.
Keep an independent numerical oracle: small whole-result comparisons
materialize each result once; large cases use scalable checks. Relax no tests.

Exercise lane counts 1/2/4 where available, windows below/equal/above the lane
count, a final partial window, uneven jobs, canonical ordering, typed multiple
failures, incompatible runtimes/domains/profiles, full joins and post-return
value usability. Existing defaults must be unavailable in routing tests.
Compare cold and reused engines with different prior cache histories under
the same fixed numerical profile.

For performance, pin source/dependencies, numerical profile and CPU allocation
before measurement. Include one leaf, several leaves and uneven costs, plus
both motivating workloads. Compare old serial, extracted serial and one lane
before several lanes. More inner workers are a separate comparison. Only
`ai-run measure` CLEAN results support claims; retain complete raw evidence.
Cross-process reproducibility has its own test gate; no cross-machine promise
is added. The current report establishes no lane speedup.
