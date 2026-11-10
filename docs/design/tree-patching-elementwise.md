# Patched element-wise operations (M6c)

## Status

Design for milestone M6c of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md),
approved by the maintainer on 2026-11-10 (see
[Decisions](#decisions-2026-11-10)). Steps 1 to 3 of the
[implementation order](#implementation-order) and the TreeACI adapter of
step 4 are implemented (see [Implementation notes](#implementation-notes));
the tree RSI adapter and the benchmark are open. It answers the two open questions the roadmap left to this
record (acceptance and driver).

The 2026-11-09 ruling stands: Hadamard is the case `k = 2, f = product` of a
general element-wise operation; the dedicated kernels (TreeACI for any `f`,
tree RSI for products) are the route, not a diagonal-MPO zip-up. The
product-only prototype and `broadcast_site_indices` of the retired
`feat/tree-patching-m6-m10` branch are not reused.

## Problem

Given `k` partitions `x_1, ..., x_k` on one site space and a pointwise map
`f`, compute a partition `g ~ f(x_1, ..., x_k)` whose patches stay below a
bond cap. Two facts shape the design.

1. **Neither kernel certifies its result.** TreeACI's `Converged` is "not a
   certificate of the full-grid maximum error"; tree RSI's tolerances
   "control local pivot selection only; validate actual product accuracy
   separately" (`treersi/src/api.rs`).
2. **The target is computable pointwise.** `g(x) = f(x_1(x), ..., x_k(x))`
   at any point costs one evaluation of each input network, and
   `PartitionedTreeTN::evaluate_batched` (#1233, PR #1234) does exactly that
   on patched inputs. A contraction has no such cheap pointwise value; an
   element-wise operation does.

Fact 2 means the M3 measurement machinery applies unchanged: the target is a
black-box function the driver can sample, exhaustively on small patches
(certified) and by an estimator on large ones.

## Scope and non-goals

In scope: `k >= 1` inputs, real or complex, any `f` through TreeACI, the
product through TreeACI or tree RSI, the M3 L2 contract on the result,
branched trees.

Not in scope:

- **Broadcasting.** Every input must have the same full site space (same
  identities and dimensions) and the same topology, as both kernels require.
  Inputs whose equivalent sites carry different identities are renamed first
  with `TreeTN::replace_indices` (a simultaneous rename since #1230). A map
  over fewer sites than the output is a different operation.
- **Error propagation from the inputs.** The contract measures
  `g - f(x_1, ..., x_k)` for the given input networks; how the inputs' own
  errors propagate through `f` is the caller's (it depends on `f`).
- **The error of the patched algebra (M3b).** Unchanged.

## Acceptance (roadmap question 1)

Proposed: **(c), a library measurement under the M3 contract**, on the
target `f(x_1, ..., x_k)`.

- Each output patch is measured against the target exactly as M2 measures an
  interpolated patch: same allowance `tau`, same estimators
  (`Exact`, `Exhaustive`, `Sampled`, `ResidualTt`), same report, same
  `GlobalL2Error` classification, and M6b's `merge_siblings` applies to the
  result unchanged.
- A mandatory caller callback (the prototype) would push onto every caller
  work the library can do itself, and would contradict the M3 decision that
  the driver measures the L2 error. An optional callback adds nothing the
  L2 measurement does not already give; norms other than L2 already have
  their placeholders in `ErrorNorm`.
- The reference `S` of the relative tolerance comes from the same
  `L2Reference` choices as M2 (given, Monte Carlo, or required). For a
  product it could be computed exactly from the exact Hadamard product of
  the restricted inputs; that is an optional later addition, not part of
  the first version.

## Driver (roadmap question 2)

| Option | Fit |
|---|---|
| A. M6a's project-first driver with the kernel swapped | Poor. Its machinery (summing several pair contributions into one output, the `Sequential` and `ExactParameterGain` probes, bounded overlap refinement) exists because a contraction output receives many pair contributions. An element-wise output patch receives exactly one input tuple, so none of it applies, and M6a has no measurement to reuse. |
| **B. M2's wave driver with a per-patch engine seam** | Best. M2's per-patch flow (engine, measurement, accept or split, report) is what M6c needs, and M2 already runs on lanes (`patched_interpolate_on_lanes`) and MPI (`patched_interpolate_mpi`). |
| C. A driver of its own | Duplicates M2's queue, measurement and report; only if B fails. |

Proposed: **B**. M6c then inherits M2's parallel entries, so M7 adds new
parallel work only for M6a: the roadmap's "one driver or two" becomes one
new driver.

### What B means concretely

The black-box form of M6c already works without any new driver code: run
`patched_interpolate` with an evaluator that calls `evaluate_batched` on
every input and applies `f`. Call it **B0**. It uses a black-box engine
(TreeTCI) and none of the inputs' structure, so it pays one pointwise input
evaluation per sample, but it is correct under the full M3 contract. B0 is
the test oracle and the benchmark baseline of M6c, and it is a usable
fallback today.

B adds what B0 cannot: an engine that receives the inputs' networks
restricted to the patch, which is what TreeACI and tree RSI take. The
measurement still uses the black-box evaluator of B0.

### Regions and structural splits

A kernel engine needs, for each output patch `P`, one network per input:
`P` must lie inside one patch of every input, or outside all of its patches
(the input is zero on `P`). Call such a `P` *resolvable*.

- The first wave is the root. A patch that is not resolvable is split at
  the next site of `patch_order` without running the engine (a *structural*
  split), so the waves descend exactly until every patch is resolvable. This
  needs every site fixed by an input projector to appear in `patch_order`;
  otherwise the run fails before evaluation with a typed error.
- **Zero-preserving operators.** For a product, `f` vanishes when any
  argument does, so a resolvable patch on which some input is zero is a
  zero patch: it is recorded as one and measured like any M2 zero patch,
  without an engine run. Other operators may declare the same property; a
  wrong declaration is caught by that measurement under L2, because a zero
  patch is accepted only when its measured error fits.
- **General `f`.** Where an input is zero, the engine receives a zero
  network for it, so `f(0, x_2)` and `f(0, ..., 0)` are represented. This
  is why the output layout cannot be the inputs' intersections alone (the
  Grosso et al. rule is for products): for `G = 1 / (G0^-1 - Sigma)`,
  `G = G0` where `Sigma` has no patch. `PartitionedTreeTN::common_refinement`
  covers only the union of the inputs' regions, so it does not give this
  layout either; the structural descent does.
- With M6a's prefix-tree projector index, finding the input patch of a
  region is a lookup, not a scan of `N_1 * ... * N_k` pairs.

A black-box engine needs no resolvability and may start at the root, as B0
does.

### The engine seam

M1's `TreeInterpolator::interpolate(problem, ctx, evaluate)` sees only a
black-box function. Proposed: a second trait next to it in
`tensor4all-treetn::interpolation`, so that `tensor4all-treeaci` and
`tensor4all-treersi` implement it without depending on
`tensor4all-partitionedtreetn` (the layering of M1's Decision 1):

```rust
pub trait ElementwiseInterpolator<T, Op> {
    fn interpolate_elementwise<V>(
        &self,
        problem: &InterpolationProblem<V>,   // active sites, cap, tau, seed
        inputs: &[TreeTN<IdxTensor, V>],     // restricted to the active sites
        operator: &Op,
        ctx: &mut EvalContext<'_>,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>;
}
```

- `Op` is a type parameter: TreeACI implements the trait for every operator
  type `Op: ElementwiseOperator<T>`; tree RSI implements it only for the
  marker `Product`. A non-product operator with RSI therefore does not
  compile, as the roadmap requires.
- The outcome is M1's `InterpolationOutcome`, so the driver's accept, split,
  embed and report code is shared. `termination` maps from the kernels:
  TreeACI `Converged` to `Converged`, `RankLimited` to `BondCapReached`,
  `MaxSweeps` to `IterationLimit`; tree RSI reports `BondCapReached` when an
  edge reached the cap (`rank_limit_reached`), otherwise `ErrorNotMeasured`,
  since it estimates no error.
- **Restriction to active sites.** The driver removes the fixed sites from
  each input network by contracting their legs with one-hot vectors, the
  inverse of M2's `embed_fixed_sites`, and re-embeds them in the output as
  M2 does. The engine then works on the patch's own domain; running it on
  the full domain of masked inputs would waste work on the zero region and,
  for a general `f`, sample `f` where the result is masked away anyway.

### Engine options

- TreeACI: `max_bond_dim` = the patch cap, `scale_tolerance = false`,
  `tolerance = tau`, so its local criterion uses the same absolute units as
  the M2 engines; `initial_guess = None`.
- Tree RSI: `max_bond_dim` = the patch cap. Its `rel_tol` (default `1e-14`)
  is relative to the largest pivot and has no absolute counterpart, so with
  the default every edge tends to reach the cap and the run splits down to
  small patches. The adapter needs a mapping from `tau` to `rel_tol`; which
  one works is an M9 measurement, and the first version exposes `rel_tol`
  in the adapter's options rather than guessing.

### Accept and split

- **Rank at the cap** (decided 2026-11-09): split when the engine's rank
  reaches the cap, equality included, as in M6a.
- **Measurement failure**: under B the M2 rule also splits a patch whose
  measured error exceeds `tau` after its retries. That is a second split
  trigger beside the decided one ([decision 3](#decisions-2026-11-10)).
  Without it, a patch below the cap that fails its measurement could only be
  retained as `ToleranceNotMet` or fail the run.
- **Retries.** M2 reruns a failed engine with the worst measured points as
  extra pivots. Neither kernel takes pivots: TreeACI takes an
  `initial_guess` network, tree RSI nothing. Proposed: a kernel engine
  retries only with a seed derived from the attempt number (TreeACI's global
  guard and random initial output, RSI's sketch), and the worst points are
  not passed. Whether retries pay off at all for kernels is measured in M9;
  `VerificationOptions::retries = 0` turns them off.

### Execution context

Every M2 stage runs on the `Session` of the caller's `EvalContext`, and M2's
publication rule depends on it. The kernels take other handles:
`tree_elementwise_batched_with_rng_in` takes a `&CpuExecutionContext` and
a caller RNG, `hadamard_many_in` an `&ExecutionContext`, and RSI requires its
inputs to belong to that context. A `Session` cannot be turned into either.
This is a prerequisite outside the patching crate
([decision 4](#decisions-2026-11-10)):

- (a) session-scoped entry points in `tensor4all-treeaci` and
  `tensor4all-treersi`, like `TreeTN::norm_on_session`; or
- (b) an `EvalContext` that also carries the `ExecutionContext` the session
  belongs to, so an adapter can pass it on.

(a) keeps one handle per run and matches M2's design; (b) is smaller but
gives the engine two handles to the same resource. Without either, B can
only call the kernels' global-default entries, which the roadmap rules out
("no route through `global-defaults`").

## Parallel execution

Under B the lanes and MPI entries of M2 carry over: the engine runs per
patch on the lane's session, as M2's engines do. Two additions: the
operator must be `Sync` for the lanes entry, as M2's evaluator already is;
and the MPI entry must make the input partitions available on every rank
before the first wave, since M2's wire format carries only patch results
and the inputs of a black-box run never travel.

## Tests

- Dense oracle on branched trees (degree-3 hub, as the M6a layout tests):
  a nonlinear `f` (for example `1 / (a - x_1 x_2)`) through TreeACI, and the
  product through TreeACI and tree RSI, each against the dense `f` of the
  dense inputs, within `delta`.
- B0 and B agree within the contract on the same inputs.
- Zero handling: a product where one input lacks a patch gives a measured
  zero patch; a general `f` with `f(0, x) != 0` covers the gap.
- Structural splits: a region spanning two input patches is split before
  any engine run; an input projector fixing a site absent from
  `patch_order` fails before evaluation.
- Rank at the cap splits; a measurement failure splits; `merge_siblings` applies to the result.
- The compile-time restriction: a `compile_fail` doctest shows that tree
  RSI with a non-product operator does not compile; the product case is
  tested positively.

Exit (roadmap): tests on branched trees against a dense oracle, for a
nonlinear `f` through TreeACI and for products through TreeACI and tree RSI;
a benchmark against the unpatched element-wise operation (and against B0)
at matched measured accuracy, on a workload chosen under the M9 rules.

## Implementation order

1. B0 as a documented recipe and the test oracle (needs PR #1234).
2. The M2 seam: resolvability and structural splits, zero-preserving
   operators, input restriction, and the `ElementwiseInterpolator` trait,
   tested with a dense reference engine inside the crate's tests.
3. The execution-context prerequisite (decision 4), in the kernel crates.
4. The TreeACI adapter, then the tree RSI adapter.
5. The benchmark (M9 rules).

Steps 1 and 2 do not depend on step 3.

## Decisions (2026-11-10)

The maintainer approved every proposal:

1. **Acceptance**: (c), the M3 library measurement on `f(x_1, ..., x_k)`.
2. **Driver**: B, M2's driver with a per-patch engine seam, with B0 as oracle
   and fallback. M2's per-patch body is generalized, not copied.
3. **Second split trigger**: a patch below the cap that fails its measurement
   splits, as in M2.
4. **Execution context**: route (a), session-scoped entry points in
   `tensor4all-treeaci` (#1235) and `tensor4all-treersi` (#1236).
5. **No broadcasting**: inputs share the full site space; the caller renames
   equivalent sites beforehand.

## Implementation notes

Steps 1 and 2, on 2026-11-10.

- **Seam.** The driver's engine is the private trait `PatchEngine<T, V>`
  (`adaptive_interpolation/engine.rs`): `runs_on(projector)`, `run(..)`, and
  the black-box `residual_engine()`. `BlackBox` wraps a `TreeInterpolator`
  for `patched_interpolate` and its lanes and MPI entries; `Elementwise`
  wraps an `ElementwiseInterpolator` for `patched_elementwise`. The per-patch
  body of M2 is unchanged apart from the engine call and the structural
  split; M2's existing tests, its golden outputs included, pass unchanged.
- **Residual engine.** The `ResidualTt` estimator interpolates the residual
  `f - f~` as a black box, which an element-wise kernel cannot do (tree RSI
  computes products only). `patched_elementwise` therefore takes a second,
  black-box engine for the estimator (for example TreeTCI); it is unused
  under `UniformMonteCarlo`. The record did not foresee this argument.
- **Zero-preserving operators.** No declaration was needed. Where an input
  is zero for a product, every candidate sample is zero, so M2's existing
  zero screen measures the zero approximation and records a zero patch
  without an engine run. A map with `f(0, ...) != 0` samples nonzero values
  there and runs the engine with a zero input network.
- **Restriction.** `embed::restrict_fixed_sites` is the inverse of
  `embed_fixed_sites`. An input with no patch on the region is restricted
  through any of its patches, which gives an exact zero network (the mask
  zeroes the selected coordinates); an empty input partition takes
  `zeros_like` of another restricted input.
- **Target evaluation.** The black-box values of `f(x_1, ..., x_k)` used for
  sampling and measurement come from `PatchLocator` (shared with
  `PartitionedTreeTN::evaluate_batched`, #1234) and one
  `TreeTNCachedEvaluator` per input patch, evaluated on the run's session.
- **`patch_order`.** Structural splits descend through `patch_order` in
  order, so an order that lists other sites before the sites the inputs fix
  splits deeper than the inputs need: with the default order (every site),
  the test inputs split to depth five instead of three. The documentation
  asks callers to list the inputs' fixed sites first; the order is not
  reordered automatically, because M6b's `merge_siblings` relies on the
  patches forming a prefix tree over `patch_order`.
- **Engines provided.** `tensor4all-treetn::interpolation` has
  `ElementwiseOperator` (closures implement it), the `Product` marker,
  `ElementwiseInterpolator<T, Op>`, and `PointwiseElementwise<E>`, which
  turns any black-box engine into an element-wise one by evaluating the
  restricted inputs pointwise. `PointwiseElementwise(TreeTciInterpolator)`
  remains the reference engine of the driver's tests.
- **Entries.** `patched_elementwise`, `patched_elementwise_on_lanes`, and
  `patched_elementwise_mpi` (2026-11-10) share one preparation (`prepare`)
  and the bodies of the interpolation entries, now factored as
  `run_serial`, `run_on_lanes`, and `run_mpi`. The cached input evaluators
  of the target live in a pool behind a mutex: a lane takes one out while it
  evaluates and returns it afterwards, so the lanes never share an evaluator
  and the lock is held only to take or return one. Every MPI rank passes the
  same inputs, held in its own memory; a rank's preparation error goes
  through the same validation collective as a plan error. The lanes entry is
  tested against the serial one (same layout, splits, and evaluations,
  values within `1e-12`); the MPI entry is added to the MPI smoke test, which
  runs only under `scripts/check-mpi-smoke.sh` with an MPI wrapper library
  and was compiled but not run here.
- **B0.** The black-box route is documented in `patched_elementwise` and
  tested against the same oracle
  (`the_black_box_route_meets_the_same_contract`).

Step 3 landed on main as #1245 (TreeACI, #1235) and #1240 (tree RSI, #1236).
The TreeACI adapter, on 2026-11-10:

- **Engine.** `tensor4all_treeaci::TreeAciInterpolator` implements
  `ElementwiseInterpolator<T, Op>` for every `Op: ElementwiseOperator<T>`
  and runs `tree_elementwise_batched_with_rng_on_session` on the session of
  the caller's `EvalContext`. The problem supplies the tolerance (absolute,
  `scale_tolerance = false`), the cap, and the random stream (ChaCha8 seeded
  with the problem's seed); the engine's own options supply the rest, as
  [Engine options](#engine-options) proposed. `initial_guess` and `root` name
  nodes of one network, so the constructor rejects them; the options are
  stored node-free.
- **Pivots.** Tree ACI takes none, so the problem's initial pivots and
  global search starts are unused and the outcome carries no pivots, as
  [Accept and split](#accept-and-split) proposed. A retry therefore differs
  from the first run only by its seed.
- **Outcome.** `termination` follows the mapping of
  [The engine seam](#the-engine-seam), with one addition taken from the
  TreeTCI adapter: `Converged` at a final rank equal to the cap is
  `BondCapReached`, as the contract requires. `error_estimate` is the last
  sweep's largest local error, which with an absolute tolerance is in the
  units of `f` (zero on the exact single-node path).
- **Sample scale.** The contract's `max_sample_magnitude` excludes samples
  taken only for global pivot search, so it cannot be recorded by wrapping
  the operator (the global guard calls the same operator). Tree ACI now
  reports `TreeAciResult::max_sample_magnitude`, the running maximum of
  `|f|` over the local matrices it committed (all values on the single-node
  path), and the adapter passes it on.
- **Errors.** An operator error reaches the caller as `Evaluator` with the
  operator's own error as the source (the adapter keeps it beside tree ACI's
  string-only `Callback`); a non-finite operator value is also `Evaluator`,
  as for TreeTCI. Input and topology mismatches, and a zero tolerance (tree
  ACI's criterion needs a positive one), are `InvalidProblem`; everything
  else is `Engine`.
- **`TreeAciNode` without `'static`.** The contract's node type carries no
  `'static` bound, and nothing in tree ACI needed one, so the bound was
  dropped from `TreeAciNode`; this only widens the accepted node types.
- **Tests.** In `tensor4all-treeaci`: on the G0 fixture (a CTTN with a
  degree-3 junction) the engine's network equals the session entry's bit for
  bit with the problem's settings, while the process-global engine is
  borrowed, and meets the exact function; plus the termination mapping (cap
  reached, rank limited, sweep limit), the single-node scale, and each error
  class. In `tensor4all-partitionedtreetn`: the product and the nonlinear
  map of the star tests through `patched_elementwise` with tree ACI as the
  kernel, against the dense oracle within `delta`, with certified L2 error.

