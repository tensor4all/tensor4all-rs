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

The error is `max_bond_error / max_sample_value` when normalization is enabled
and the sampled scale is positive; otherwise it is the raw bond error. The
global search uses `tolerance * max_sample_value` in normalized mode and
`tolerance` otherwise, with `tol_margin_global_search` as its acceptance margin.

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

An edge update measures nothing when one side of its candidate matrix has a
single candidate while both sides of the edge have more than one
configuration: the first pivot exhausts that side and the LU stops with a zero
residual whatever the function is. Around a vertex of local dimension 1 whose
adjacent bonds have rank 1, the junction-side candidate set of each adjacent
bond is the product of the other bonds' single pivots, so every one of those
bonds is in this state and none can grow
([#1153](https://github.com/tensor4all/tensor4all-rs/issues/1153)). Such an
error does not count toward `Converged`; the run reports
`DegenerateCandidateSet` instead. An edge whose one side has only one
configuration (all its sites of local dimension 1) has rank 1 exactly and is
measured.

To let those bonds grow, each optimization call first enriches the pivots
around vertices of local dimension 1, following xfac's
`TensorTreeCI::addPivotsAllBonds(enrich_pivot1())`: it collects the vertices of
local dimension greater than 1 adjacent to a vertex of local dimension 1 and,
for every unordered pair of them, adds the base pivot with those two sites set
to every pair of local values. The base pivot is the first pivot pair of the
first edge adjacent to a vertex of local dimension 1. The enrichment runs only
when some bond around such a vertex has a side with at most one pivot. A
function that is exactly rank 1 around the vertex collapses back to rank 1
after the first sweep and is reported as `DegenerateCandidateSet`.

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
