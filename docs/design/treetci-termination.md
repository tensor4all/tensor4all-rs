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

## Continued optimization

A `TreeTCI2` can be optimized repeatedly with a larger `max_bond_dim` or with
its cap removed, for the same function, topology and local dimensions. Current
pivot sets and the maximum sampled magnitude carry over; bond errors are
refreshed during subsequent passes. Each pass visits each edge once; its pivot
sets remain unchanged until its own update, since other edges use distinct
canonical subtree keys. Proposers retain those current pivots directly. No
historical full pivot maps are copied or retained; `ijset_history` is removed.

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
