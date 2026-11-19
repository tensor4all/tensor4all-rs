# TreeTCI stopping diagnostics

`optimize_default`, `optimize_with_proposer`, and
`optimize_with_proposer_with_rng` return `TreeTciOptimizationResult`.
`crossinterpolate2` and `crossinterpolate2_with_rng` return `TreeTciRunResult`,
which also owns the materialized `TreeTN`. Both results expose `ranks`,
`errors`, and `termination`; callers use fields rather than tuple destructuring.

## Stopping contract

The reasons describe the optimization loop, in its decision order:

| Reason | Condition | Consequence |
| --- | --- | --- |
| `SampledErrorFloor { at_bond_cap: true }` | Three trailing maximum ranks reach the cap and a positive measured error fails the strict tolerance comparison. | The final error records the sampled floor. The state remains swept and materializable. |
| `MaxBondDimension` | Three trailing maximum ranks reach the cap without the measured floor above. | Still takes precedence over convergence; not every edge need be saturated. |
| `SampledErrorFloor { at_bond_cap: false }` | Three trailing completed-sweep pivot snapshots are equal while their positive measured errors fail the strict tolerance comparison. | The error records the measured stall, not a proof that no other approximation could improve it. |
| `Converged` | Three trailing errors are strictly below `tolerance`, the final maximum rank equals the minimum in that window, and no global pivots were accepted in that window. | A sampled criterion, not an exhaustive bound over the full index domain. |
| `DegenerateCandidateSet` | The convergence window holds, but an edge's most recent local update could not measure its bond error. | The approximation may be inaccurate. |
| `MaxIterations` | The requested iterations completed without an earlier stop. | A small final error alone does not establish convergence. |

The recorded error is the maximum of local bond residuals, residuals at the
**bounded historical pool and all current retained pivot pairs**, and new
global-search observations.
Known points are checked against the current approximation even if they were
pruned from the pivot sets. With normalization enabled and a positive sampled
scale, divide this maximum by `max_sample_value`; otherwise use it unnormalized.
The observer receives this same value and the evaluation accounting after the
search. Pivot injection happens only if the loop will continue, so every
returned state has completed a sweep since its last injection.

The uncapped floor window consists of snapshots `S_k = S_(k+1) = S_(k+2)`
at the ends of three completed sweeps, with two equality comparisons between
those snapshots. The first sweep may have changed its input pivot state; that
new snapshot starts the count at one. The condition does not require three
individually unchanged input-to-output sweeps.

### Local truncation and measured floors

Local LU truncation remains `tolerance * error_scale`, independently of
`enable_global_pivots`. This correction does not divide that threshold by the
number of edges, disable absolute truncation, or introduce a recovery phase.
Those numerical strategies belong to the separate #1371 experiment.

If three consecutive completed-sweep pivot snapshots (including their order)
are equal while their recorded errors are all positive and fail the strict
tolerance comparison, the run reports `SampledErrorFloor`.
The last recorded error is the measured stall. This is an observed lack of
progress, not proof that no other approximation or future trajectory can improve
it. The stopping tolerance and the numerical relative cutoff remain unchanged.

At `tolerance = 0`, a zero recorded residual is neither strictly below tolerance
nor a positive sampled violation. An exact uncapped run therefore reaches
`MaxIterations`; an exact capped run can report `MaxBondDimension`. Positive
residuals can still produce a floor at zero tolerance.

### Validation ownership and search coverage

Historical validation belongs to one initialization/optimization call,
independently of optional oracle memoization. Compact integer keys index a
bounded pool of point/value payloads. The default `validation_cache_bytes` budget
is 32 MiB of accounted sample storage, including both packed-key copies and
estimated container overhead, excluding spare capacity, fixed indexing metadata
and process RSS. Zero disables historical storage. The pool is frozen before the
requested final materialization and dropped at return.

At capacity, ordinary oracle traffic replaces old unprotected slots in rotating slot
order, reusing their point buffers. The strongest residual witness measured on
the current approximation is protected from ordinary insertion. Each new
approximation first replays the retained pool, including the previous witness;
a stronger newly measured point may replace it. This avoids silently discarding
the largest observed violation while bounding retained memory. It is still a
subset of historical samples, not a record of every oracle request ever made.

Every current retained pivot pair is validated independently of that pool.
Missing values are queried in bounded batches; pairs already replayed need no
second read. This also runs with zero history budget, zero random starts, zero
insertion allowance, and on final or capped sweeps. The budget limits historical
coverage rather than terminating interpolation. `validation_samples`,
`validation_bytes`, `validation_limit_reached` and `validation_evictions` expose
the pool's occupancy and pressure. Target values must be fixed during a call.

One cached tree evaluator serves historical replay, current pivots and fresh
search. Replaying known values needs no oracle queries. The largest validation
violation can become a global pivot, even below the discovery margin. Fresh
search uses
`tolerance * error_scale`, preferring discoveries above
`tol_margin_global_search` times that threshold; when none cross the margin,
observed tolerance violations still qualify.

If no point qualifies and `nsearch > 0`, confirmation walks every current
retained pair on every edge, newest first, walking duplicate full points only
once. The walk count and compact deduplication set are bounded by the sum of
bond ranks, with borrowed pivot tables and reusable full-site scratch. If
confirmation finds no pivot, every edge with a dimension-one endpoint gets
up to `nsearch` restarts in **each** direction: hold one observed subtree fixed
and randomize the other. This preserves a sparse branch while exploring an
independent branch. Those restarts total at most twice `nsearch` times the
number of unit-endpoint edges. All stages share the cached approximation
readout. They improve sampled coverage without enumerating the full domain.

Fresh discovery and injection are skipped on final or saturated sweeps, but
known-sample validation still runs. It also runs when `nsearch` or
`max_nglobal_pivot` is zero. `enable_global_pivots = false` disables the history
and all global checks. Immediate materialization after an unswept injection
can have inconsistent bond ranks
([#692](https://github.com/tensor4all/tensor4all-rs/issues/692)); floor stops therefore discard pending injections and return the measured state.

The named `TreeInterpolator` adapter maps a cap-caused floor to
`BondCapReached`, preserving the common contract's cap precedence. An uncapped
floor maps to `InterpolationTermination::SampledErrorFloor`. MPI reports preserve
that verdict.

## Unmeasured bond errors

An edge update is marked unmeasured when a candidate side has only one entry,
or the selected pivots exhaust the candidates at an endpoint of local dimension
1, while both full subtrees admit more configurations than the stored rank.
The dimension-one endpoint has no local variation to probe beyond the
neighboring pivots. The LU then records a zero residual because no candidate
row or column remains, even if unsampled configurations would increase the rank. Such an error does not count toward
`Converged`; the run reports `DegenerateCandidateSet` instead. If the rank
reaches the full configuration count of either subtree, candidate exhaustion
is legitimate: the bond cannot grow further. A rank below both candidate counts
also has a residual to measure. At endpoints of local dimension greater than
one, exhausting the local expansion alone does not invalidate the ordinary
sampled stopping criterion; this diagnostic is not a global error certificate.

The rank-1 case occurs around a vertex of local dimension 1 whose adjacent
bonds have rank 1: the junction-side candidates are the product of the other
bonds' single pivots, so none of those bonds can grow
([#1153](https://github.com/tensor4all/tensor4all-rs/issues/1153)). The same
problem occurs at larger ranks on chains containing a dimension-1 site. Its
neighbor-product candidates can stop at rank 14 even though the full
bipartition admits rank 16; neither a zero local error nor a global search
that finds no new pivots establishes convergence
([#1352](https://github.com/tensor4all/tensor4all-rs/issues/1352)). The diagnostic
qualifies the recorded error; it does not enrich candidates or guarantee that
the returned approximation meets the requested tolerance.

To let those bonds grow, each optimization call first enriches the pivots
around vertices of local dimension 1, following xfac's
`TensorTreeCI::addPivotsAllBonds(enrich_pivot1())`: it collects the vertices of
local dimension greater than 1 adjacent to a vertex of local dimension 1 and,
for every unordered pair of them, adds the base pivot with those two sites set
to every pair of local values. The base pivot is the first pivot pair of the
first edge adjacent to a vertex of local dimension 1. The enrichment runs only
when some bond around such a vertex has a side with at most one pivot.

A function that is exactly rank 1 around the vertex collapses back to rank 1
after the first update of each bond. If the enriched pivots lived only in the
pivot sets, the junction side would again be a single candidate and the run
would end in `DegenerateCandidateSet` although rank 1 is the right answer
([#1159](https://github.com/tensor4all/tensor4all-rs/issues/1159)). xfac
avoids this because its candidate set is
`set_union(I0[{from,to}], kronecker(from,to))`, where the history `I0` holds
the enriched pivots. Candidate generation retains current pivots rather than
an `I0` history (the run's validation history is separate). The first call that builds the enrichment pivots records their projections onto the
junction side (the side holding the vertex of local dimension 1) of each
adjacent bond, whether or not the pivots are added, and `DefaultProposer`
(and through it `TruncatedDefaultProposer`) appends them to that side's
candidates on every update. The junction side then keeps at least as many
candidates as there are distinct projections, so a rank-1 bond error is
measured. Later calls record nothing more, so the candidate count grows by a
fixed amount. A custom proposer that does not offer them can still reach
`DegenerateCandidateSet`.

## Continued optimization

A `TreeTCI2` can be optimized repeatedly with a larger `max_bond_dim` or with
its cap removed, for the same function, topology and local dimensions. Current
pivot sets and the maximum sampled magnitude carry over; bond errors are
refreshed during subsequent passes. Each pass visits each edge once; its pivot
sets remain unchanged until its own update, since other edges use distinct
canonical subtree keys. Proposers retain those current pivots directly. No
historical full pivot maps are copied or retained; `ijset_history` is removed.
A caller-supplied `EdgeVisitor`
must return a permutation of the graph's edges on every pass; the loop rejects a
missing, duplicated, or unknown edge before that pass updates any state, so the
stale-bond-error and false-convergence cases cannot arise from an invalid order.

Each call starts fresh ranks/errors/global-pivot diagnostics, a new convergence
window, bounded validation pool and iteration budget. Final
iterations validate known samples without injecting new pivots on every call. Several calls need not reproduce
one longer call, nor a fresh uncapped run's pivot trajectory or ranks at each
iteration. Increasing a cap permits growth; it does not guarantee convergence
or certify error over the whole network.

Seeded high-level calls restart their candidate and global RNG streams.
Caller-stream entry points advance the same generator when the caller reuses
it, including for random proposers. The [random-stream contract](treetci-random-streams.md)
defines that distinction. A failed evaluator can leave partial edge updates;
continuation is not a transactional rollback mechanism.

The #833 regression applies χ=4 then uncapped to real/complex three-arm trees
with a site-free junction, with global search enabled and disabled. It checks
all dense entries against the target, checks the capped result is inaccurate,
and compares the recovered result with a fresh uncapped run after aligning
physical indices. This fixture establishes recovery in that case, not a
universal equivalence between optimization schedules.

## Relationship to TreeTCI.jl

The Rust stopping logic follows the `TreeTCI.jl` convergence criterion cited
by the original implementation: branch `local-fix-convergence`, commit
`06563dd`. Rust applies an early-convergence break rather than exhausting
`max_iter` unconditionally, compares the error on the same normalized or
absolute scale as `tolerance`, and checks bond saturation before injecting
global pivots. The early-stop regression and capped chain/star regressions in
`crates/tensor4all-treetci/src/optimize/tests.rs` preserve those decisions.
This reference describes the implementation's recorded provenance; it is
not a claim about the behavior of every current upstream branch.

This document replaces the inaccessible local-file references reported in
[#834](https://github.com/tensor4all/tensor4all-rs/issues/834) and records the
public stopping diagnostics requested by
[#835](https://github.com/tensor4all/tensor4all-rs/issues/835).
