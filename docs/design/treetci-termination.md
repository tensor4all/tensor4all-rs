# TreeTCI stopping diagnostics

`optimize_default`, `optimize_with_proposer`, and
`optimize_with_proposer_with_rng` return `TreeTciOptimizationResult`.
`crossinterpolate2` and `crossinterpolate2_with_rng` return `TreeTciRunResult`,
which also owns the materialized `TreeTN`. Both results expose `ranks`,
`errors`, and `termination`; callers use fields rather than tuple destructuring.

## Stopping contract

Default stopping is the current-main rule at `35174fdd`. The recorded error is
`max(local bond residual, ordinary global-search sampled maximum)`, divided by
`max_sample_value` when normalization is enabled and the scale is positive.
The observer receives the recorded error and accounting after any confirmation.

| Reason | Condition | Consequence |
| --- | --- | --- |
| `MaxBondDimension` | Three trailing maximum ranks reach the cap. | Takes precedence over convergence. No confirmation runs. |
| `Converged` | Three trailing errors are strictly below tolerance, final maximum rank equals the minimum in that window, no global pivots were accepted in the window, and every latest bond error is measured. | With `confirm_convergence`, confirmation must also pass. Still a sampled criterion. |
| `DegenerateCandidateSet` | The ordinary convergence window holds but an edge's latest local error is unmeasured. | No confirmation runs. Accuracy is not established. |
| `SampledErrorFloor` | Three opt-in failed confirmations have identical completed-sweep pivot snapshots (including order), each with a positive measured residual at or above tolerance. | Records an observed stall, not an optimality bound; returns without pending injection. |
| `MaxIterations` | Iteration limit reached without an earlier stop, including failure of a final confirmation. | A small final error alone does not establish convergence. |

Local LU truncation remains `tolerance * error_scale`. No per-edge tolerance
allocation, absolute-cutoff removal or recovery phase is introduced. At zero
tolerance, the strict ordinary convergence comparison cannot pass, so confirmation
and its floor cannot run. An exact uncapped run reaches `MaxIterations`; a capped
run can report `MaxBondDimension`.

### Opt-in convergence confirmation

`TreeTciOptions::confirm_convergence` defaults to `false`. The off path has no
history storage or replay and preserves ordinary errors, ranks, verdicts and
oracle counts. Enable it to check only an approximation that would otherwise
return `Converged`, independently of `enable_global_pivots`. Confirmation does
not run on ordinary non-converged or capped stops, nor on every completed sweep.
A sampled maximum that keeps the ordinary criterion above tolerance can prevent
confirmation entirely. The per-bond / Julia-compatible alternative and #1371's
cost issue remain separate work in #1405. `501ebb4a` is a historical cost
reference, not the default behavior contract.

Each confirmation materializes one approximation and shares one cached evaluator
across all checks. It replays the bounded historical pool and validates every
current retained pivot pair; missing values are queried in bounded batches.
Known values need no new target query. Zero history budget or zero `nsearch`
still checks current pairs. If no violation is known and `nsearch > 0`, every
distinct retained full point seeds a walk, newest first. If these find none,
each unit-endpoint edge gets at most `nsearch` restarts in each direction,
rotating retained anchor pairs while randomizing the complementary subtree.
These finite searches improve coverage without certifying the full domain.

A positive measured residual at or above the absolute tolerance rejects the
candidate. Its maximum is recorded in that iteration's error. Unless the run
stops, up to `max_nglobal_pivot` violators are injected and counted as accepted
global pivots. The ordinary convergence window then requires three fresh sweeps;
there is no one-sweep retry. Zero injection allowance still rejects a failed
confirmation. A final failed confirmation returns `MaxIterations` before floor
detection or injection. Otherwise, the third consecutive failed confirmation at
an identical completed snapshot returns `SampledErrorFloor`. Snapshot comparison
happens only at failed confirmations. Fixed rank alone does not imply a floor.
Every returned state has completed a sweep since its last injection
([#692](https://github.com/tensor4all/tensor4all-rs/issues/692)).

### History ownership and accounting

History belongs to one enabled initialization/optimization call, independently
of oracle memoization. Its `validation_cache_bytes` budget defaults to 32 MiB of
accounted point/value and table storage, excluding spare capacity, fixed indexing
metadata and process RSS. Zero disables historical storage. The pool is frozen
before the requested final materialization and dropped at return. Target values
must remain fixed throughout the call.

At capacity, ordinary oracle traffic replaces old unprotected slots in rotating
order, reusing point buffers. The strongest currently measured residual witness
is protected until the next confirmation replays it; a stronger measured witness
can replace it. Eviction limits historical coverage without stopping the run.
`validation_samples`, `validation_bytes`, `validation_limit_reached` and
`validation_evictions` report occupancy and pressure. `confirmation_checks`,
`confirmation_failures` and `confirmation_pivots` report attempted checks, rejected
candidates and points passed to injection. All history and confirmation counters
are zero with confirmation disabled.

The named `TreeInterpolator` adapter maps `SampledErrorFloor` to the common
`InterpolationTermination::SampledErrorFloor`; MPI reports preserve that verdict.
Ordinary `MaxBondDimension` still maps to `BondCapReached`.

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
window and iteration budget, plus a new bounded pool when confirmation is enabled.
A final convergence candidate can be confirmed; a failed confirmation never
injects at the iteration limit. Several calls need not reproduce
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
