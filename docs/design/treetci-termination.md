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

Each optimizer call starts its own diagnostic and convergence window. These
results do not establish an equivalence between several calls and one longer
call, nor change the continuation/RNG contracts tracked by
[#833](https://github.com/tensor4all/tensor4all-rs/issues/833) and
[#824](https://github.com/tensor4all/tensor4all-rs/issues/824).

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
