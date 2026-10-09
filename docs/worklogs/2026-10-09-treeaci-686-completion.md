# Complete the TreeACI redundant-work issue

Tracking: [#686](https://github.com/tensor4all/tensor4all-rs/issues/686), under
[#854](https://github.com/tensor4all/tensor4all-rs/issues/854).

## Decisions

Production individual `candidate_frame` calls occur only in the leaf candidate
batch. That batch now lazily prepares one scalar layout at its first cache miss,
bound to the input and directed cut. Empty batches and fully cached batches
prepare none. Each multiincoming physical group also reuses one incoming-slice
adapter for all scalar candidates, then drops it before the next group. The
existing scalar accumulator, candidate order, cache admission and exact
multiply/add reduction order are unchanged. Layout and adapter metadata remain
batch/group-local, with no persistent per-cut cache or quadratic retained
storage. Builder batches already use shared layouts through merged #863.

The final metadata-scan item explicitly asks to **evaluate** fusion. Retain the
separate lazy scans: rank classification checks residuals/scales, computes
limits only for bad cuts, and returns at the first growable bad cut; injection
capacities are allocated only inside an eligible Guard. An eager fused pass
would compute all limits and allocate a vector even when Guard is disabled,
search count or retained pivot count is zero, or local residuals are already
rank-limited. A bad saturated cut can also be locally rank-limited despite
headroom on an accurate cut. Fusing these semantic questions would need extra
state or a second pass to preserve the current laziness. Both scans are O(E)
and small relative to contractions. The focused mixed-cut/tolerance policy
regression supports this completed evaluation, not a claim that fusion landed.

## Closure accounting

Every item in #686 has a supported disposition on top of merged #863:

| Original item | Resolution/evidence |
| --- | --- |
| Repeated Guard start evaluation | Merged #856; shared floating-zone kernel and point accounting regressions; low-temperature R=9 witness retained. |
| Empty global injection clones/checkpoint | Earlier remediation; validated empty return and no-mutation regression. |
| Duplicate evaluator hint | Earlier remediation; shared checked batch hint. |
| Repeated physical offset | Earlier remediation; hoisted, budgeted table and physical-axis-order oracle. |
| Transaction sample cloning | Earlier remediation; consumed owned samples, rollback boundary retained. |
| Batched incoming frame copies | Borrowed slices and `extend_from_slice` at matrix boundary. |
| Duplicate FrameBuilder priming | Non-copying memo-hit `ensure_computed` seam; owned-returning `compute` retained for actual consumers. |
| Dead ranks/rank-changed bookkeeping | Removed in earlier remediation; schedule trace remains test-only. |
| Production updated-edge vector | Boolean production tracking; full update order test-only. |
| Duplicate algebraic bounds | Computed once and supplied to initial-rank selection. |
| Duplicate compact candidate keys | Carried from validated lookup through admission. |
| Overlapping component traversal | Merged #856; compact ordered axis suffixes, no O(E^2) full-subtree cache. |
| Repeated scalar layout/adapter | Merged #861/#863 plus this leaf/group repair cover every production batch route. Individual test reference calls intentionally prepare independently. |
| Projection scratch evaluation | Merged #856; one reused dependency-ordered scratch vector. |
| Schedule cloning | Merged #861; shared immutable schedule storage. |
| Rank/capacity scan fusion evaluation | Completed here; retain early classification and lazy capacity allocation for the reasons above. |

The earlier remediation evidence is in
[the original work log](2026-08-25-treeaci-redundancy-remediation.md);
[#856 follow-up](2026-10-09-treeaci-followups.md),
[#861 follow-up](2026-10-09-treeaci-known-guard-failures.md), and
[#863 builder repair](2026-10-09-treeaci-scalar-frame-layout.md) retain their
own correctness and performance boundaries.

Intentionally retained costs remain necessary under their owning contracts:
output snapshots/arena checkpoints and candidate clones protect atomicity;
candidate IDs and pivot IDs have independent ownership; projection validates
points at its safety boundary; activation-mask contents implement selective
injection; scalar `compute` returns owned memo values; two-incoming batching
may compute a Cartesian superset; and the 3+ incoming scalar path preserves
its numerical reference order. This issue does not authorize removing them.

## Verification conclusions and constraints

All 261 changed-crate checks pass (258 debug tests/doctests, two diagnostics
checks and one focused R=9 release regression), as do deny-warning Clippy,
rustdoc, formatting and deterministic repository-rules preview. The leaf effort regression
independently fails on pristine merged `378ca500` with 12 axis lookups instead
of 2. Four scalar types exercise exact leaf values, row/column packing,
duplicates, cache hits, zero cache headroom and empty batches. Separate tests
reject cross-input/cross-cut layout reuse. Existing wide-cut and malformed
layout tests preserve reduction and budget semantics.

The costly R=9 low-temperature third-slice witness is checked in release mode;
ordinary changed-crate checks use debug mode. Paired release evidence uses
an isolated pristine baseline and distinct source-stamped binaries, with no
compilation/numerical overlap during timing. The initial full matrix is
inconclusive because one baseline cold-query timing exceeds the dispersion
gate. The separately declared complete confirmation (five pairs/repetitions,
unchanged gates) has zero oracle/validity failures. Counts and maximum oracle
errors match exactly in all 120 cases in both runs. Timings remain descriptive
and do not establish a general speedup or formal non-regression bound.
[Complete initial and confirmation results](../../benchmarks/results/2026-10-09-treeaci-686-completion.md). The closure concerns confirmed
redundant production work and the requested scan evaluation; it does not
resolve #671's downstream cost attribution, #784's rank oscillation or #794's
unreproduced mixed-capacity no-op report. #854 therefore remains open. RSI is
excluded, and no tolerance or coverage requirement is relaxed.
