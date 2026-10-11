# F3 step 3a: a caller-owned execution context for the TCI path's factorization (#670 / #859 F3)

Short design for the first implementation step of F3, after the diagnosis in
`docs/design/859-f3-g0-stage-budget.md` (§1, §1.1) and its control-surface inventory (§2, step 1-3).

## 1. What the diagnosis leaves, and the smallest first step

The G0-shaped stage over-parallelizes: more workers make it slower in both the global pivot search and the
rest, while most of a 64-core machine stays idle. Giving a *stage* its own budget needs a control surface, and
the inventory in §2 records where the path reaches the process-global context: `tensor4all-treetci` has no
execution-context parameter, and its per-sweep factorization calls `matrix_luci_factors_from_matrix` on the
global path.

One fact makes the first step small: **the core seam already exists.** `tensor4all-core` exposes
`matrix_luci_factors_from_matrix_owned_in(matrix, options, context: &CpuExecutionContext)`, documented as
consulting no process-global context. The missing piece is only the propagation from the TCI entry point to
that call.

## 2. This step

```rust
// tensor4all-treetci; the context is appended, matching `matrix_luci_factors_from_matrix_owned_in`,
// `contract_src_with_rng_in` and `norm_in`
pub fn crossinterpolate2_in<T, F, P>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
    context: &tensor4all_tensorbackend::CpuExecutionContext,
) -> TreeTciResult<TreeTciRunResult>;

// and the caller-randomness counterpart, so a caller can keep advancing its own stream:
pub fn crossinterpolate2_with_rng_in<T, F, P, R>(..., proposer: &P, rng: &mut R, context: &CpuExecutionContext)
    -> TreeTciResult<TreeTciRunResult>;
```

- **The algorithm and its successful-run numerics are unchanged, with the conditions stated.** Pivot
  selection happens before the factor assembly and does not read it, so for finite inputs, identical options
  and the same random stream the pivots, ranks, errors and termination are the same. **Two behaviours are not
  promised to be identical**: the factor assembly runs through the *concrete* session entry, which rejects a
  Rayon worker rather than serializing or falling back to the global context, so a worker caller that would
  have succeeded on the legacy route can fail typed here; and the owned core path validates its input
  (`validate_owned_luci_input`) where the borrowed one classifies through `rrlu`, so a nonfinite derived
  tolerance can be reported as a different error variant. The rustdoc states the worker restriction, and the
  record does not promise identical failure behaviour.
- **Only the factorization moves in this step.** The context is threaded to `update_edge`'s LUCI call and used
  through the existing `..._owned_in` variant. The evaluator, the materialization and anything else the path
  reaches stay on the process-global context, and the record says so: this is step 1 of the inventory's
  ordered plan ("then measure what remains"), not a claim that the stage is fully budget-controlled.
- **Optional, not a replacement.** `crossinterpolate2` keeps its signature and its behaviour; the new entry
  is the opt-in one. No option field is added to `TreeTciOptions`, so nothing changes for existing callers or
  for the types that derive from those options.

## 3. Evidence plan

1. **Parity with materialized values**: a seeded run through both entries agrees on termination, ranks and
   errors, and the two materialized networks agree at points that were not training pivots, against an
   analytic oracle - real and complex, plus a rank cap and an all-zero sample that fails the same way through
   both.
2. **The context is the one used, at the edge boundary**: an in-crate test calls `update_edge` with a caller
   context *while the process-global backend is held*, which a route resolving its factorization through the
   global backend could not survive, and compares the selected pivots and pivot errors against the legacy
   route. The fixture is rank-deficient and rectangular, so it cannot take the rank-zero shortcut. The source
   guard is kept as a *propagation* check (the caller parameter reaches each shared boundary), not as proof.
3. **Both budgets measured separately**: with the process-global context fixed at one worker by
   `RAYON_NUM_THREADS`, the caller context's worker count is varied (1, 4) and the same-count legacy-versus-new
   pair is included. The bench reports requested and observed counts for both domains, complete ranks and
   error histories, peak process threads and CPU ticks. An unchanged curve is an honest result and does not
   prove the context was used - the dispatch test above does that - nor that the evaluator is the only
   remaining cause.
4. Gates: fmt, clippy with the repository flags, both crates' tests and the new doctests, rustdoc
   `-D warnings`, `scripts/check-public-error-docs.py`, the panic audit and
   `scripts/repository-rules-review.py`.

## 4. Non-goals and non-claims

- No change to `crossinterpolate2`, its options type or any other public signature.
- No change to the TCI algorithm, the pivot search or the stopping rule.
- No claim that the stage now has a stage-local budget: only its factorization does.
- No claim about the downstream `R=10` behaviour, which remains unmeasured here.

## 5. What the measurement decided

The step-3 measurement (`859-f3-g0-stage-budget.md` §2) says the factorization's budget is **not** where the
stage's worker penalty lives: moving it to a caller context with four workers made the stage slower, not
faster, while peak threads grew and every configuration stayed numerically identical. The next threading
target is therefore the evaluator, the global pivot search's readout and the final materialization - not the
remaining full-pivot-LU/SVD/QR sites, which this stage barely reaches.
