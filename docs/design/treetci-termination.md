# TreeTCI stopping diagnostics

`optimize_default`, `optimize_with_proposer`, and
`optimize_with_proposer_with_rng` return `TreeTciOptimizationResult`.
`crossinterpolate2` and `crossinterpolate2_with_rng` return `TreeTciRunResult`,
which also owns the materialized `TreeTN`. Both results expose `ranks`,
`errors`, and `termination`; callers use fields rather than tuple destructuring.

## Stopping contract

The reasons describe the existing optimization loop, in its decision order:

| Reason | Condition | Consequence |
| --- | --- | --- |
| `MaxBondDimension` | The maximum rank is at the configured cap throughout three trailing iterations. | Checked before global search and convergence, even if the sampled error is small. It does not mean all edges are saturated. |
| `Converged` | Three trailing errors are strictly below `tolerance`, the final maximum rank equals the minimum in that window, and no global pivots were accepted in that window. | Describes the sampled criterion, not an exhaustive error bound. |
| `DegenerateCandidateSet` | The `Converged` condition holds on the recorded errors, but the most recent update of at least one edge could not measure its bond error. | The run stops where it would have reported `Converged`. The approximation may be inaccurate. |
| `MaxIterations` | The requested iterations completed without either stopping condition. | A small final error alone does not establish convergence. |

The recorded error is the maximum of local bond residuals and residuals
actually observed by the global search during that iteration. With a positive
sampled scale and normalization enabled, divide that maximum by
`max_sample_value`; otherwise use it unnormalized. The observer receives
this same value after search, along with evaluation accounting including the
search. New pivots are processed in the next sweep.

Global search uses `tolerance * max_sample_value` in normalized mode and
`tolerance` otherwise. Discoveries above `tol_margin_global_search` times that
absolute tolerance take priority; when none cross the margin, already observed
tolerance violations still qualify. If ordinary starts yield no pivot, edges
with a dimension-one endpoint supply retained tail pivot pairs and restarts
that preserve one observed subtree while randomizing its complement. Each
extra family is bounded by `nsearch` walks. If those families find nothing
either, a third stage enumerates around the same retained tail pivots: every
configuration of each maximal edge side with at most 32 local configurations,
the other sites held at the pivot, in one batch and without random draws
([#1433](https://github.com/tensor4all/tensor4all-rs/issues/1433)). A walk
needs a nonzero residual one coordinate move from its start; the enumeration
reaches a defect several moves away as long as it lies in such a side. On
binary quantics chains that side is the aligned block of width at most 32
around the pivot's value of one variable; wider blocks are not searched. These
checks can escape sparse support that uniform random starts miss; they do not
exhaust the domain or turn sampled convergence into an error bound.

The third stage runs in every search that finds no pivot, and `Converged`
needs three trailing iterations without an accepted global pivot. So when
those three iterations all searched, none of their searches saw a residual
above the tolerance inside an enumerated side around a retained tail pivot.
This is a bounded guarantee: it says nothing about blocks wider than 32, and
an iteration whose search was skipped (see below) contributes no search.

Global search is skipped after the final requested iteration and at a
bond-saturation stop. An injected pivot must be processed by a later sweep:
immediate materialization of an unswept pivot set can have inconsistent ranks
across an edge ([#692](https://github.com/tensor4all/tensor4all-rs/issues/692)).
Disabled or skipped searches contribute zero to the convergence window. Thus
`Converged` can be reported on the final iteration without searching again;
it does not assert that an independent final residual search took place.
The same applies when search is enabled but `nsearch` or `max_nglobal_pivot`
is zero. If the convergence window is not yet available, the result is
`MaxIterations`, including a one-iteration call on an exactly representable
function.

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

The diagnostic fires only when the candidates are fully used. A unit vertex can
also lock its bonds while a few spare candidates remain. In the three-branch
table of #1352, the centre has local dimension 1, and the x and z bonds into it
capped each other at rank 11 (or 10) below the true rank 12. The bond into the
third branch had rank 1, so each side offered only the other bond's pivots.
The #1159 candidates kept the count just above the rank. Every retained pivot
pair had residual near 1e-16, and the run stopped `Converged` with a dense error
of 6.3e-10 at tolerance 1e-10.

`DefaultProposer` therefore makes a unit vertex transparent while it is locked.
On the unit side of an edge, let r be the edge's current rank. If the product of
the other bonds' pivots holds fewer than 2r candidates, the side also offers a
second set: the other bond with the most pivots has those pivots replaced by its
neighbour's own side candidates, expanded over the neighbour's local index. This
is what a two-site update across the unit vertex would see.

- The threshold 2r is the slack a binary site would add. It is not tuned.
- The largest-rank bond is the one entangled with the edge. A nearly separable
  bond has small rank and cannot carry the missing direction.
- Above the threshold, the plain product is used. High-rank junctions, such as
  the #1153 comb, therefore pay nothing extra after their first sweeps.

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
the enriched pivots. tensor4all-rs keeps no pivot history; instead the first
call that builds the enrichment pivots records their projections onto the
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
window and an independent iteration budget. The final-iteration global-search
skip and stop precedence apply on every call. Several calls need not reproduce
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
