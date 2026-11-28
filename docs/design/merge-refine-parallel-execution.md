# Parallel merge-refine execution

## Status and decisions for review

Proposal for [#1497](https://github.com/tensor4all/tensor4all-rs/issues/1497).
The serial baseline is `8ce72e4e`; its numerical sources match the initial
inspection at `379868e0`. Hataori is pinned to `092eecdd`, tenferro to
`5df83672`.

The [maintainer decision in #1497](https://github.com/tensor4all/tensor4all-rs/issues/1497#issuecomment-7229306374)
selects level-zero leaf applications as the first parallel stage. It supersedes
the earlier ordering suggested by the serial investigations below. A completed
serial run at #1364's size remains evidence for choosing the next stage.

The serial subset-image path separates preparation from application.
`plan_subset_images` fixes the operator bindings, reference scale, per-leaf
budget and centers, and checks every exact block before the first leaf
application. The serial loop consumes those jobs in canonical source order
and returns the first application error in that order. This boundary gives
the future lane driver a place to supply its engines and collect leaf results
without changing the schedule's error fold.

The [serial evidence](../experiments/1497-baseline/README.md) changes the
initial decomposition proposal: jobs for whole incoming regions have width
one in the measured completed schedules, even with many input leaves.
That scheduling unit cannot distribute their dominant merge work. Leaf
applications remain independent; the region result does not say otherwise.

The [finer serial investigation](../experiments/1497-merge-granularity/README.md)
is complete at integrated #1407/#1511 source `5fbd73e5`. Input-parent work in
child probes with multiple addition-bearing parents occupies median 30.0%
(R4) and 74.8% (R6) of Fourier time; observed same-probe widths reach 35 and 38.
Both pass the declared need screen. Design that internal seam next, preserving
canonical folds and per-child admission; this does not establish parallel speedup. Keep leaf-only execution
as an option, rather than a committed implementation step: its opportunity
is limited in the larger repeated M6a case, and the IR observation needs
repeats. Defer whole-region lane execution until a relevant repeated workload
has both costly region jobs and width above one. A tiny multi-region example
establishes what the count means, not a reason to implement parallelism.

| Issue question | Proposed answer | Reason and trade-off |
| --- | --- | --- |
| Unit of parallelism | Design input-parent work within one child probe next; retain candidate-level work as an alternative. Keep leaf-only execution as a possible later scope with a serial tail. | Leaves are independent, but their R=6 share is small. Whole-region jobs have width one here; finer parent costs now pass the need screen, while ordering and admission remain prerequisites. |
| Accuracy allocation | Preserve the existing allocation and arithmetic order. | Application budgets and each level's share can be fixed before its numerical work. Workers need no peer's measurements. |
| Sum/report order | Keep source, sibling, term and scalar-fold orders. | Sorting completed outputs alone cannot preserve intermediate truncation decisions or the next level's budget. |
| M7 prerequisite | Design now; expose lanes only after the consumed numerical paths are deterministic and engine-explicit. | A cached-evaluator topology/dtype allowlist alone does not prove application, QR, SVD or residual determinism. |
| Memory | Bound submitted work with a window; retain each application's existing limit. | This controls concurrency and buffered outcomes. Per-application estimates do not bound process memory or retained outputs. |

The [integration plan](./merge-refine-parallel-integration-plan.md) lists
owners, blocking contracts and acceptance evidence. In particular,
[#1407](https://github.com/tensor4all/tensor4all-rs/pull/1407) merged as `9ddffd17` with merge policy, accounting and resource admission.
Its head `5fbd73e5` is the measured source; all consumed crates, benchmarks
and the lockfile are identical at that head and merge. Preserve those settled
semantics when designing the parallel merge seam.

## Code-level answers at `281f5122`

This section checks the five issue questions against `main` at `281f5122`
(tenferro pinned at `5df83672`). Paths are relative to
`crates/tensor4all-partitionedtreetn/src/reconstruction/` unless named.
One finding corrects the prerequisite stated in #1497: see question 4.

**1. Unit of parallelism.** Leaf applications are numerically independent:
the restructured operator, each term's center and exact-block check, and the
certified budget are all fixed before the first application
(`target.rs:591-629`). The loop body reads only its own term
(`target.rs:631-685`). Region bodies of one level share no tensors. Under
Tolerance they do share admission (question 2). Amdahl ceilings follow from the
CLEAN medians, assuming unlimited lanes and no dispatch cost. They are
upper bounds, not speedups:

| Case | (a) leaf share, ceiling | (b) region width | (b') input-parent share, ceiling | (a)+(b') ceiling |
| --- | --- | ---: | --- | ---: |
| m6a-r4-cap8 | 60.3% (98 leaves), 2.5x | 1 | 30.0% (≤ 35 per probe), 1.4x | 10x |
| m6a-r6-cap16 | 11.3% (118 leaves), 1.13x | 1 | 74.8% (≤ 38 per probe), 4.0x | 7.2x |
| ir-r8-cap64 (one run) | 37.6% (4 leaves), 1.6x | 1 | not measured | — |

Shares come from the [baseline](../experiments/1497-baseline/README.md) and
[granularity](../experiments/1497-merge-granularity/README.md) campaigns. Their
Fourier times agree within 1%, so the last column, which adds the two, is approximate.
Whole-region jobs, (b), have width one in every completed case and buy nothing.
In the hottest R6 probe, the largest parent takes 16.9% of the parent time, which caps
that probe near 6x. Neither campaign is decision grade. Both are downscaled
(T = 0.1, caps 8/16). At the resolution of #1364 (R = 10, T = 0.01, 113 patches),
no serial merge-refine run completes: exact is killed at 12 GB, and certified
times out after 60 min. The serial algorithm is the first bottleneck there
(#1373, #1130). The deciding measurement is a completed serial run of that
workload with the same phase split (leaf application against eligible
parent work) and its leaf count.

**2. Budget allocation.** Every accuracy share is fixed before a level's
items run, so no item reads a peer's measurement:

- Rank goal: `charges` and `share` are computed from the whole incoming level
  before the region loop (`schedule.rs:1882-1900`). `spent` is accumulated during the
  loop (`schedule.rs:1946`, `1967`, `2002`, `2059`) but is read only after it
  (`schedule.rs:2095`), when it sets the next level's `remaining`. Exact
  uses share zero (`schedule.rs:1896-1897`).
- Tolerance: `total_pairs` and each region's budget come from whole-level
  shapes (`tolerance.rs:326-338`). A child's share is fixed from key and pair
  counts before its tensor work (`tolerance.rs:213-217`).

Two running quantities remain. `spent`, `incurred` and
`counters.dropped_error` (`schedule.rs:2543`) are binary64 sums in traversal
order. A parallel split must replay them in that order; worker subtotals do
not suffice. Tolerance admission `RetainedWork` carries from region to region
(`tolerance.rs:331`, `350`) and from child to child (`tolerance.rs:228`). A
`ResourceLimit` there drops an optional candidate (`tolerance.rs:250`). This is
a decision, not a float: a later region's chosen split depends on earlier
regions' retained work whenever `max_work_items` (default 4096) or
`max_terms` (default 2^20) binds. Fixing a per-region quota up front would
change serial results in exactly those cases. Keeping admission serial
changes nothing. No change to the allocation is needed.

**3. Order of sums and of the report.** The serial path is already free of
hash order:

- Leaves follow the canonical patch order: the metadata key with projector
  coordinates, ties broken by content digest. This is the order #1366 made public
  (`target.rs:126-133`, `subdomain_tree_tn.rs:156-212`).
  `application_error` is folded in that order (`schedule.rs:1804-1814`).
- Live work is `BTreeMap<OutputKey, BTreeMap<InputKey, Item>>`
  (`schedule.rs:1797`). The keys are derived-`Ord` prefix tuples
  (`schedule.rs:1074-1082`). Regions run in `OutputKey` order, then own, axis
  and joint candidates, then child keys in `child_keys` order.
- Sibling sums are a left fold in `InputKey` order
  (`schedule.rs:2401-2410`), paired by term index (`schedule.rs:2466`).
- Level components fold by `hypot` in `OutputKey` order
  (`schedule.rs:2088-2090`). Levels add in level order to `application_error`
  (`schedule.rs:2144-2149`).

That traversal is the canonical order. A parallel run must produce the same
tensors and replay the same scalar sequence. Sorting finished outputs is not
enough, because truncation decisions and the next level's share depend on
intermediate values.

**4. Dependency on the M7 prerequisite.** The roadmap's steps (c) and (a)
change `TreeTNCachedEvaluator` only. This call chain never uses it:
`CachedEvaluator` does not occur in `reconstruction/`, in
`tensor4all-quanticstransform/src`, or in `tensor4all-treetn/src/operator`.
Level zero contracts through the zip-up. At internal nodes it makes one N-ary
`T::contract` (`tensor4all-treetn/src/treetn/contraction.rs:1834-1843`,
`297-300`). Core labels its operands positionally
(`tensor4all-core/src/defaults/contract.rs:1230-1275`), and tenferro plans
them with `EinsumOptions::default()`
(`tensor4all-tensorbackend/src/tenferro_bridge.rs:269`). At the pinned
tenferro, default planning does not anneal and uses the deterministic greedy
planner (`tenferro-einsum/src/planning/tree.rs:49-57`, `207-215`;
tenferro-rs#2014, which closed tenferro-rs#1963). The planner tie that the M7
prerequisite targets therefore does not apply to this path.
[tree-patching-findings.md](./tree-patching-findings.md) records #1963 as
closed upstream by #2014. This is a code-reading argument; no cross-lane test has been run. Two
inputs remain unproven. The first is the operand order of each N-ary call: it
comes from a BFS over the graph (`contraction.rs:1706`,
`node_name_network.rs:492-502`), so it depends on node insertion order. The
second is the SVD/QR/GEMM kernels, which the error contract records as
thread-count dependent. Option (a) therefore need not wait for the
cached-evaluator steps, and a topology/dtype allowlist is the wrong gate. Its
gates are an explicit engine on the consumed path, which does not exist
today (`patched_fourier.rs:154-160`, `schedule.rs:1672-1680`; #1376), and a
cross-lane bitwise test at a fixed per-lane engine profile. That test covers
a chain, a degree-3 tree with a site-free node, and f32/c32. The certified
application stays chain-only (`application.rs:52-55`).

**5. Memory.** `memory_limit` bounds one certified application's estimated
working set (`tensor4all-treetn/src/operator/certified.rs:41-48`). The
scheduler passes it unchanged to every leaf (`target.rs:657`). The exact
path's #1279 guard refuses an oversized non-chain block before any
application (`target.rs:595-617`, limit `2^28` elements at `target.rs:804`). Both
are per application. With `q` applications in flight, the worst case is `q`
working sets plus the retained images, which serial already holds until the
loop ends (`target.rs:630`, `schedule.rs:1804`). At the 4 GiB default, `q = 3`
already admits 12 GiB of estimates. A single certified
application at R = 8 already peaks at 1.3 GB (#1373). In the measured M6a cases,
the merge phase sets the peak, not level zero (R6: 111.7 MiB after
applications, 888.8 MiB high-water mark). Parallel level zero therefore needs
the window `K` of the section on windows, memory and failures. Dividing
`memory_limit` by `q` would refuse inputs that fit serially.

**Implementation outline (idiom of #1495).** First, extract the serial stages
unchanged and compare them with the old function. Then add
`schedule_merge_refine_on_lanes` and `patched_fourier_transform_on_lanes`, in which:

1. Preparation (`target.rs:441-629`) runs once on one supplied engine.
2. Leaf applications run in windows of `K` canonical jobs over the lanes.
   Results are consumed in source order, so the fold at `schedule.rs:1814`
   is unchanged.
3. The level loop stays serial on one engine. Any later parent-level split
   replays the scalar events of question 2 in traversal order.
4. One lane uses the same code as several lanes; there is no
   `lanes.len() == 1` branch.

Tests that pin this, after `tests/adaptive_lanes.rs` and
`tests/routing_oracle.rs`:

- Lane counts 1, 2 and 4 with identical serial per-lane engines produce the
  same bits for every floating `MergeRefineReport` field, including per-level
  reports. They also produce equal counters and `work_items_per_level`, and
  bitwise-equal node tensors under a consistent renaming of bond IDs.
- The serial entry matches one lane only when the global engine has the same
  profile.
- The suite covers Exact, Measured and Certified application, Rank goal and
  Tolerance, and a Tolerance case where `max_work_items` binds and rejects an
  optional split.
- Leaf costs are very uneven: one leaf is empty or tiny, and `K` is below,
  equal to and above the lane count, with a partial last window.
- It runs on a chain and a degree-3 tree with a site-free node, with f64, c64,
  f32 and c32 data.
- A routing test makes the global defaults unavailable.
- Two failing leaves return the first failure in source order.

## What the baseline establishes

An **incoming region job** is one iteration over the incoming level's
`BTreeMap<OutputKey, BTreeMap<InputKey, Item>>`. Its `OutputKey` identifies a
spectator cell and one `(depth, prefix)` per coordinate axis. The job contains
all input items and retained terms for that key, its unsplit work, probes and
selection. It is not one leaf, term, child candidate, final output region or
a claim that its tensors cover the whole output grid.

For executed level `t`, define `W_t` as the number of keys in that incoming
map, before the transition. The diagnostic emits exactly one
`schedule/region` event per iteration, tagged with `level` and ordinal.
`largest_region_count` is `max_t W_t`, obtained by counting those events per
level. "Whole" describes the job's scope; "completed" describes the process
finishing the transform. `work_items_per_level` counts input items and is
generally different from `W_t`; `report.region_count` counts final outputs.

The smallest example in the single-axis, single-root, no-spectator family is
the two-bit Exact transform at depth two: one input leaf with dense values
`[1, 0, 0, 0]` has unitary DFT `[0.5, 0.5, 0.5, 0.5]`. The tiny diagnostic directly records incoming widths `[1, 2]`, while the
test asserts four final regions and `work_items_per_level = [1,2,4]`.
With one original input item per region in this fixture, those work-item
counts imply the same widths; this equivalence does not hold generally. One bit has only one transition
from its sole root, so it cannot have two incoming jobs in this family.
Multiple initial spectator cells can instead provide several first-level
jobs; these are separate keys in the same definition.
The numerical/work-item fixture is
[`minimal_two_bit_exact_fanout_has_two_second_level_region_jobs`](https://github.com/tensor4all/tensor4all-rs/blob/c3b08cb08fd2d44459d4476ca8e806a090d3bfd1/crates/tensor4all-quanticstransform/tests/qft_reconstruction/output_selection.rs).
It is in scope for correctness/definition, with one original leaf and no
genuine input-sibling additions; it is not a representative performance case.

Every completed measured process ends with `region_count = 1`; the finer
M6a records also report zero output pruning. Starting from one root, selected
splits add output keys and those keys are never re-joined, so here width one
means no refinement was selected. R4's own candidate meets the goal; R6 probes
two axes and retains own; the single IR observation also ends in one region.
A goal that causes a selected split, or several initial spectator cells,
can provide wider levels. No repeated realistic multi-region case is
established by these runs, and a tighter goal need not make a split pay.

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
These facts describe the baseline's exact/rank-goal schedule; the merged
Tolerance policy from #1407 has a different ledger.

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

## Possible leaf-only executor: three stages on supplied engines

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
(Correction: the cached-evaluator steps are not on this call chain. See
question 4 under "Code-level answers".)
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
lane service or aggregate allocator if the leaf-only option is selected.

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
