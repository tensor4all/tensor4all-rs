# Tree interpolation engine seam

## Status

Approved and implemented in M1 of
[tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md). The
implementation consists of:

- the TreeTCI termination report (step 1) and the matching
  `tensor4all-quanticstci` update;
- the contract module `tensor4all_treetn::interpolation`;
- the TreeTCI engine `TreeTciInterpolator` (steps 2-6) and its tests,
  including the test-only mock engine;
- the amendment of [partitioned-treetn.md](./partitioned-treetn.md) (last
  section);
- review fixes (the cap precedence, non-finite initial samples).

The decisions taken during implementation are recorded under
[Implementation decisions](#implementation-decisions).

A separate proposed amendment,
[optional edge pivots](./tree-interpolation-edge-pivots.md), describes opt-in
export of each edge's selected side coordinates for a later recursive
split-selection study. That amendment is not implemented or approved; the
M1 seed pivots described below cover the problem's active sites and do not
preserve per-edge selections.

## Goal

A patch driver in `tensor4all-partitionedtreetn` (milestone M2) must run a
tree interpolation engine on one patch at a time without depending on any
engine crate. This record defines the contract between the driver and an
engine, and its TreeTCI implementation.

## What the driver needs from one patch

For a patch whose projected sites are fixed, the driver hands the engine the
remaining (active) sites and needs back:

1. a network over the active sites, with the caller's node names and site
   identities;
2. a verdict that separates "converged below the bond cap" from "stopped at
   the bond cap" and "stopped at the iteration limit", because only the first
   is accepted by default (since the M5 patch-size bounds of 2026-10-04 the
   driver can also accept a small capped patch or retain a patch at its
   minimum size, see
   [tree-pqtci-patch-size-bounds.md](./tree-pqtci-patch-size-bounds.md); an
   outcome network's bonds therefore never exceed the cap, whatever the
   termination);
3. the error estimate and the maximum sampled magnitude, unnormalized;
4. optionally, full-domain pivots of the result for seeding children.

## Findings that shape the design

Verified against `main` at `b63e13de`.

- `tensor4all_treetci::crossinterpolate2` returns only
  `(TreeTN<IdxTensor, usize>, ranks_per_iter, errors_per_iter)`.
- `tensor4all_treetci::optimize_with_proposer` returns `(ranks, errors)`. It
  stops when, over the last three sweeps, the errors are below tolerance, no
  global pivots were added, and the last rank equals the window minimum; or
  when the rank reached `max_bond_dim` in all of the last three sweeps; or
  when `max_iter` is exhausted. The return value does not say which.
- **The bond-cap stop can return an unswept state on `main`**
  ([#692](https://github.com/tensor4all/tensor4all-rs/issues/692)). Global
  pivots are injected after a sweep, and the saturation stop could then exit
  before a sweep re-paired the two sides of each edge, so
  `crossinterpolate2` failed with "bond ranks disagree across edge". A
  reproduction on a four-vertex tree whose center has degree three failed for
  all 40 seeds tried. The fix
  ([PR #790](https://github.com/tensor4all/tensor4all-rs/pull/790)) decides
  the saturation stop before the global pivot search and skips the search when
  it fires; the convergence stop is unchanged and still requires that the
  current search added no pivots. With it, every exit of the loop happens after
  a sweep.
- `TreeTCI2` keeps `max_sample_value` and per-subtree pivot sets `ijset`
  (`SubtreeKey -> [n_subtree_sites, n_pivots]`). The two keys of an edge
  partition all vertices, and a completed edge update writes both sides in
  LU pairing order, so an edge's pivots can be joined column by column into
  full-domain points on chains and on branched trees.
- With `normalize_error`, convergence uses `tolerance` times the patch-local
  `max_sample_value`; with it disabled the raw bond error is compared.
- `tensor4all_treetci::to_treetn` names nodes by vertex id
  (`TreeTN<IdxTensor, usize>`) and creates one fresh site index per vertex.
  A `TreeTciGraph` vertex has one local dimension; dimension-one vertices are
  accepted by the graph, `TreeTCI2::new`, and materialization. `TreeTCI2`
  requires at least two vertices.
- `TreeTN::rename_node` cannot change the node-name type, and
  `TreeTN::replace_site_index_with_indices` cannot remove an index (it rejects
  an empty replacement).
- If every initial pivot evaluates to zero, `crossinterpolate2` returns an
  error; with no initial pivots it silently uses the all-zero point.
- When an LU step selects nothing, the edge update falls back to candidate 0,
  so recycled pivots may be points where the function is zero.

## Dependency: #692 fix

The M1 implementation starts after
[PR #790](https://github.com/tensor4all/tensor4all-rs/pull/790) is merged. The
seam relies on its guarantee that every exit of `optimize_with_proposer` leaves
a swept state, so that a run stopped at the bond cap has consistent pivot sets
and can be materialized. A capped run may be an inaccurate approximation; the
seam only needs it to stop cleanly and report `BondCapReached`.

## Contract in `tensor4all-treetn`

A new module `tensor4all_treetn::interpolation` (names are proposals). Node
names require `V: Ord` in addition to the usual `TreeTN` node-name bounds;
`tensor4all-partitionedtreetn` already requires it.

```rust
/// One interpolation problem. Built only through `InterpolationProblem::new`,
/// which validates the invariants below; fields are private with accessors.
pub struct InterpolationProblem<V> { /* private */ }

impl<V: Clone + Hash + Eq + Ord + Debug + Send + Sync> InterpolationProblem<V> {
    /// `node_sites` gives the active site indices of every node of
    /// `topology` (possibly none). The site order used by batches and
    /// pivots is derived: nodes in ascending name order, each node's sites in
    /// the given order.
    pub fn new(
        topology: NodeNameNetwork<V>,
        node_sites: BTreeMap<V, Vec<DynIndex>>,
        initial_pivots: ColMajorArray<usize>,
        absolute_tolerance: f64,
        max_bond_dim: Option<NonZeroUsize>,
        seed: u64,
    ) -> Result<Self, InterpolationError>;
    // accessors: topology(), node_sites(), site_order(), initial_pivots(),
    // absolute_tolerance(), max_bond_dim(), seed()

    /// The site order `new` derives from `node_sites`, so callers can lay
    /// out `initial_pivots` before constructing the problem.
    pub fn derive_site_order(node_sites: &BTreeMap<V, Vec<DynIndex>>) -> Vec<DynIndex>;
}

/// Topology and site checks of the contract, shared with callers such as the
/// M2 patch driver (added in M2; see "Validation" below).
pub fn validate_layout<V>(
    topology: &NodeNameNetwork<V>,
    node_sites: &BTreeMap<V, Vec<DynIndex>>,
) -> Result<(), InterpolationError>;

#[non_exhaustive]
pub enum InterpolationTermination { Converged, BondCapReached, IterationLimit }

pub struct InterpolationOutcome<V> {
    pub network: TreeTN<IdxTensor, V>,
    pub termination: InterpolationTermination,
    pub error_estimate: f64,
    pub max_sample_magnitude: f64,
    /// Shape `[n_active_sites, n_pivots]` in site order, or `None` when the
    /// engine does not produce pivots.
    pub pivots: Option<ColMajorArray<usize>>,
}
// A plain struct with public fields: engines in other crates build it with a
// struct literal, so it is not `#[non_exhaustive]` and has no constructor.

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InterpolationError {
    InvalidProblem { message: String },
    Evaluator { source: anyhow::Error },
    AllSamplesZero,
    Engine { source: anyhow::Error },
}

pub trait TreeInterpolator<T> {
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>;
}
```

### Validation in `InterpolationProblem::new`

The topology is a tree whose node set equals the keys of `node_sites`; every
site index appears once and has a positive dimension; at least one active
site exists; `initial_pivots` has one row per active site, at least one
column, and coordinates within the site dimensions; `absolute_tolerance` is
finite and nonnegative. Engines rely on these invariants and do not repeat
the checks. The topology and site checks (tree shape, node set, distinct full
site identities, positive dimensions, at least one site) are performed by the
public `validate_layout`, which `new` calls; M2 added it so that the patch
driver validates its inputs with the same code before any evaluation. Sizes an
engine derives for itself (for TreeTCI, the product of a
node's site dimensions) are validated by that engine with checked arithmetic
and reported as `InvalidProblem`.

### Semantics

- **Tolerance.** `absolute_tolerance` is compared with the engine's raw error
  estimate. The driver computes it from its relative tolerance and a
  reference scale (pinned for all patches in M3), so no engine normalizes on
  its own.
- **Termination precedence.** `Converged` means the engine's error criterion
  holds and the final maximum bond dimension is strictly below
  `max_bond_dim` (or no cap is set). An error criterion met at a rank equal to
  the cap is `BondCapReached`. The driver accepts only `Converged` and treats
  every other variant, including future ones, as not accepted.
- **Randomness.** `seed` is the only source of randomness; an engine overrides
  any seed or seeded component held in its own configuration.
- **Batch.** `evaluate` receives a `ColMajorArrayRef<usize>` of shape
  `[n_active_sites, n_points]` in site order; no new batch type is added.
- **Samples.** `max_sample_magnitude` covers the samples used by the
  interpolation sweeps, not samples taken only for global pivot search or
  materialization.
- **Zero patches.** If every initial pivot evaluates to exactly zero the
  engine returns `AllSamplesZero`. A non-finite initial sample (NaN or
  infinite) is an invalid evaluator value and is reported as `Evaluator`, never
  as `AllSamplesZero`. The sampled-zero policy itself (screening candidates
  before calling the engine) belongs to the driver (M2).
- **Pivots.** Returned pivots are valid points but may include points where
  the function is zero; they seed children and carry no other meaning.
- **Network form.** The outcome network has the problem's node names and
  topology and carries only the active site indices; a node without active
  sites carries no site index. Re-embedding fixed sites into the patch form
  used by `partitionedtreetn` is the driver's job through one shared helper
  (M2), not repeated per engine.
- **Scalar bounds.** The trait leaves `T` unbounded; each implementation
  states the bounds its engine needs.

## TreeTCI implementation in `tensor4all-treetci`

1. **Termination report.** `optimize_with_proposer` returns a report with
   `ranks`, `errors`, and a reason (`Converged`, `MaxBondDimension`,
   `MaxIterations`), mapped to `InterpolationTermination` with the precedence
   above. `crossinterpolate2` keeps its current signature. The other caller of
   `optimize_with_proposer`, `tensor4all-quanticstci`, is updated in the same
   M1 PR. The implementation runs with `normalize_error = false` and requires
   `max_iter >= 3`, because convergence needs three sweeps of history.
2. **Evaluator errors.** TreeTCI wraps callback errors into its own operation
   error. The implementation wraps each evaluator error in a private marker
   type before returning it to TreeTCI, and on failure searches the error
   chain with `downcast_ref` for that marker: found maps to `Evaluator`,
   otherwise `Engine`. The same wrapper checks that the evaluator returned
   exactly one value per point and reports a wrong length as `Evaluator`.
3. **Vertices.** Each node becomes one vertex whose local dimension is the
   product of its active site dimensions, fused column-major in the node's
   site order, or one for a node without active sites. Vertex coordinates
   are translated to site-order rows for `evaluate`; this is a copy, not a
   view.
4. **Pivots.** Initial pivots are converted to vertex coordinates and passed
   through `add_global_pivots`. After optimization, each edge's two pivot sets
   are joined column by column (after checking equal column counts), converted
   back to site order, and deduplicated.
5. **Network.** A crate-internal generalization of `to_treetn` materializes
   directly with the problem's node names and final site indices, splitting
   fused vertices and omitting the dimension-one index of nodes without active
   sites, instead of renaming and replacing indices after the fact.
6. **Single-node topology.** `TreeTCI2` needs two vertices; a single-node
   problem first applies the zero-patch rule (all initial pivots zero returns
   `AllSamplesZero`, as for every topology); otherwise it is evaluated exactly
   on its full index set and returns `Converged`, an error estimate of zero,
   the largest evaluated magnitude, and no pivots.

## Other engines

The contract is designed so another engine is added in its own crate. For
TreeACI this is plausible but not free: its current entry point interpolates
an element-wise operator over input networks, seeds from an initial network
rather than pivots, and reports normalized errors without pivots or a maximum
sample. A TreeACI implementation needs a new site-coordinate entry point in
`tensor4all-treeaci`; `pivots` is optional for that reason. The test-only mock
engine shows that the trait and the driver need no engine-specific code; it
does not show that TreeACI needs no work.

## Tests

- Chain and branched trees (at least one node of degree three or more),
  compared with dense references on small cases.
- A node with several active sites (fused vertex), a node without active
  sites, and a single-node topology.
- Each termination variant, including an error criterion met at a rank equal
  to the cap (`BondCapReached`), a capped run on a branched tree that stops
  cleanly with `BondCapReached` (no accuracy assertion), and an iteration
  limit.
- An evaluator failure and an evaluator returning the wrong number of values
  are both reported as `Evaluator`, not `Engine`.
- A single-node problem with all-zero initial pivots returns
  `AllSamplesZero`.
- Returned pivots are valid full-domain points and seed a second run.
- Node names and site identities of the result equal the problem's, including
  site indices that share an ID but differ in prime level or tags.
- `InterpolationProblem::new` rejects each invalid input with
  `InvalidProblem`; all-zero initial samples return `AllSamplesZero`.
- A test-only mock engine runs through the same generic test helper.
- Rustdoc: runnable, asserted examples for every new public item, with
  `# Errors` naming the variants.

## Amendment to partitioned-treetn.md

The migration record excludes adaptive interpolation, TreeTCI termination
changes, pivot recycling, and sampled-zero inference from
`tensor4all-partitionedtreetn`, and keeps the TreeTCI prototype branch
separate. The reason was to keep that crate free of TCI dependencies. With
this seam the driver depends only on the trait in `tensor4all-treetn`, so the
reason still holds. The M1 implementation PR amends both statements. The M2
driver derives its patch queue from TCIAlgorithms.jl through
`tensor4all-partitionedtt`; the M2 PR records that in
`docs/PROVENANCE_AND_CITATION_POLICY.md` and follows the license and
derivation-notice obligations that
[partitioned-treetn.md](./partitioned-treetn.md) states for code derived from
that crate (`LICENSE-TCIALGORITHMS-MIT`).

## Names

The review kept the proposed names: `TreeInterpolator`,
`InterpolationProblem`, `InterpolationOutcome`, `InterpolationTermination`,
and `InterpolationError` in `tensor4all_treetn::interpolation`. The TreeTCI
side adds `TreeTciInterpolator`, `TreeTciOptimizationResult`, and
`TreeTciTermination`.

## Implementation decisions

- **Report from both optimizers.** `optimize_default`, the thin wrapper of
  `optimize_with_proposer` with `DefaultProposer`, returns the same
  `TreeTciOptimizationResult { ranks, errors, termination, evaluation }`.
  `TreeTciTermination` is `#[non_exhaustive]`.
- **Cap precedence in TreeTCI.** The saturation stop is checked before the
  convergence criterion, ranks never exceed the cap after a sweep, and
  convergence requires the last rank to be the window minimum. A
  `TreeTciTermination::Converged` run with a cap therefore ends strictly below
  it, and an error criterion met at the cap surfaces as `MaxBondDimension`,
  mapped to `BondCapReached`. The engine keeps the "rank at the cap is
  `BondCapReached`" guard for `Converged` as defence in depth; it is covered by
  a unit test of the mapping.
- **Engine configuration.** `TreeTciInterpolator::new` takes
  `TreeTciOptions`. On every run the problem overrides `tolerance` (the
  absolute tolerance), `max_bond_dim` (the cap), `normalize_error` (always
  `false`), and `seed` (the problem seed); `max_iter`, the global pivot
  search settings and `evaluation_cache_bytes` come from the engine. The
  optional memo covers optimization; named materialization evaluates through
  the adapter separately. `new` validates the options and
  requires `max_iter >= TreeTciInterpolator::MIN_MAX_ITER` (public constant,
  3). `Default` uses `TreeTciOptions::default()`.
- **Proposer.** The engine always uses `DefaultProposer`, which is
  deterministic. A seeded proposer could not have its own seed overridden
  generically, which the randomness rule requires.
- **Evaluator error source.** An error whose chain contains the marker
  anywhere is classified as `Evaluator`. When the marker is the error itself
  or sits only under `anyhow` context layers added inside the engine,
  `downcast` recovers it: the caller's original error becomes the source and
  those engine-internal context layers are dropped. When the marker sits
  under another typed error, the whole chain is kept as the source.
- **Test-only mock engine.** It evaluates the dense domain and factorizes it
  with `factorize_tensor_to_treetn`, which rejects a node without site
  indices, so the mock runs on a chain, a star of degree three, and a single
  node. The TreeTCI engine is tested on all topologies, including internal and
  leaf nodes without sites.
- **Batch translation fast path.** Vertex batches are still translated to
  site-order rows, but not always copied (a deviation from item 3 of the
  TreeTCI implementation). When every vertex has at most one site, a site
  coordinate equals its vertex coordinate and the site rows are the vertex
  rows in order (the site order concatenates the node site lists in node-name
  order, which is the vertex order). The batch is then passed to `evaluate`
  without a copy when no vertex is site-free, and otherwise the site rows are
  gathered past the site-free vertices without the column-major division. A
  vertex that fuses several sites takes the general split. Every path checks
  the batch's vertex count and every vertex coordinate against its local
  dimension (a site-free vertex must have coordinate 0), so a malformed batch
  still fails as an engine error before `evaluate` sees it. On the branched
  quantics tree workload of the tree-patching runner, whose site-free root
  and fixed nodes take the gather path, the translation time fell from 2.2 s
  to 1.2 s (`R = 7`) and from 5.8 s to 3.0 s (`R = 8`), measured with
  `patched_interpolate` (L2, `rtol = 1e-4`, `eta = 0.3`, cap 32; release, one
  pinned core, all thread variables set to 1).
