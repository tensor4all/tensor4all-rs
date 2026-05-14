# TreeTN Cached Evaluator Design

## Goal

Add an allocation-light TreeTN batch evaluator that extends the `TTCache::evaluate_many` idea from chain tensor trains to arbitrary tree topologies. The evaluator should reuse subtree environments across repeated batch points, choose a useful center automatically, and include benchmarks that compare scaling against the chain `TTCache` baseline.

This design primarily addresses issue #464. It also provides the efficient approximate-evaluation building block needed by TreeTCI global pivot search in issues #380 and #383.

## Context

`TreeTN::evaluate` already accepts a column-major batch coordinate array and returns one scalar per point. Internally, however, `TreeTNEvaluator::evaluate_batch` loops over points and each point builds one-hot tensors, contracts each local tensor with those one-hots, constructs a temporary `TreeTN`, and contracts that temporary network to a scalar.

`tensor4all-simplett::TTCache` is more efficient for chain tensor trains. It caches left and right partial contractions, groups repeated left and right index assignments in a batch, and evaluates each point by combining cached environments. The TreeTN version should preserve that scaling property for chains while supporting general trees.

## Non-Goals

- Do not change `TreeTN::evaluate` semantics in the first pass.
- Do not implement center-edge evaluation unless benchmarks show center-node evaluation is insufficient.
- Do not add dense full-network materialization to production evaluation paths.
- Do not introduce row-major batch layout. Batch coordinates remain column-major with shape `[n_sites, n_points]`.

## API Shape

Add a new reusable evaluator in `tensor4all-treetn`, separate from the current point-loop evaluator:

```rust
pub struct TreeTNCachedEvaluator<T, V>
where
    T: TensorLike,
{
    center: V,
    // private precomputed node/site/topology data
}

impl<T, V> TreeTNCachedEvaluator<T, V>
where
    T: TensorLike,
{
    pub fn new(tree: &TreeTN<T, V>, indices: &[T::Index], options: CachedEvaluatorOptions<V>) -> Result<Self>;

    pub fn center(&self) -> &V;

    pub fn evaluate_batch(&mut self, values: ColMajorArrayRef<'_, usize>) -> Result<Vec<AnyScalar>>;
}
```

Add an options type:

```rust
pub struct CachedEvaluatorOptions<V> {
    pub center: Option<V>,
    pub initial_centers: Vec<V>,
    pub max_greedy_steps_per_start: Option<usize>,
}
```

When `center` is provided, the evaluator uses it directly. When it is absent, it runs greedy center search from `initial_centers`. A later helper can populate deterministic default starts from the canonical center, minimum node name, diameter endpoints, and diameter midpoint.

Keep `TreeTNEvaluator` unchanged in this first pass. The cached evaluator is a new API so existing behavior remains stable.

## Center Cost Model

Given a batch with `B` points and a TreeTN with `N` tensor nodes, define a directed edge component count:

```text
component_count[(u, v)] =
    number of unique batch assignments restricted to the component containing u
    after removing edge u-v
```

For a center node `c`, define:

```text
center_cost(c) = sum over neighbors n of c: component_count[(n, c)]
```

This approximates the number of unique subtree environments that must be computed and cached when all environments are collected into `c`.

The scaling target for center search is:

```text
precompute: O(B * N)
greedy:     O(S * H * average_degree), with no batch rescan
memory:     O(B * N) or lower
```

where `S` is the number of initial centers and `H` is the number of greedy steps per start. The implementation must avoid recomputing full batch projections for every center candidate, which would be `O(B * N^2)` on large trees.

## Component Cost Index

Introduce an internal precomputed structure:

```rust
struct ComponentCostIndex<V> {
    counts: HashMap<(V, V), usize>,
}
```

It exposes:

```rust
fn component_count(&self, from: &V, to: &V) -> Option<usize>;
fn center_cost(&self, center: &V, topology: &NodeNameNetwork<V>) -> Result<usize>;
```

The construction should read the batch coordinates a bounded number of times and produce all directed edge component counts. A straightforward first implementation may be clearer than maximally optimized code, but it must not perform an independent full projection scan per candidate center.

## Greedy Center Search

Keep greedy search separate from evaluator construction:

```rust
pub struct GreedyCenterSearch<V> {
    pub initial_centers: Vec<V>,
    pub max_steps_per_start: Option<usize>,
}

pub struct CenterSearchResult<V> {
    pub center: V,
    pub cost: usize,
    pub start: V,
    pub steps: usize,
}
```

For each start:

```text
c = start
loop:
    current = center_cost(c)
    best = neighbor of c with minimum center_cost(neighbor)
    if center_cost(best) < current:
        c = best
    else:
        stop
```

Return the terminal result with the lowest cost across all starts. Tie-breaking must be deterministic, using node ordering where available.

This search is a heuristic. It can stop at a local minimum, so the API should keep the initial-center list explicit. Tests should cover multi-start behavior rather than claiming global optimality.

## Cached Evaluation Algorithm

The cached evaluator stores node ownership, local site-axis mapping, topology, and selected center. It evaluates a batch as follows:

1. Validate `values` has shape `[input_count, n_points]`.
2. For each neighbor subtree of the center:
   - Project each batch point to the site rows in that subtree.
   - Deduplicate projected assignments.
   - Compute one environment tensor per unique assignment.
   - Store a mapping from point index to unique environment index.
3. For the center node:
   - Slice or contract the center tensor according to center-owned site coordinates for each point.
   - Combine the center slice with the cached neighbor environments to produce a scalar.

The cache key must be based on full site assignment values for the relevant subtree. It does not need to persist across unrelated TreeTN objects. A single evaluator may retain its environment cache between calls when the same assignments recur.

## Correctness Requirements

- Results must match `TreeTN::evaluate` for:
  - one-node networks,
  - two-node chains,
  - longer chains,
  - star trees,
  - balanced trees.
- Batch coordinate layout must remain column-major.
- Site index identity must use full `Index` equality, not index IDs.
- Invalid values shape, duplicate indices, missing indices, unknown center, and out-of-range coordinates must return errors.
- Tests should compare whole result vectors against the existing evaluator for small networks.

## Benchmark Requirements

Add benchmark coverage that demonstrates both absolute runtime and size dependence:

- Chain `TensorTrain` evaluated by `TTCache::evaluate_many`.
- Equivalent linear-chain `TreeTN` evaluated by `TreeTNCachedEvaluator`.
- Current `TreeTN::evaluate` as the allocation-heavy baseline.
- Optional star and balanced-tree cases to show non-chain behavior.

Use sizes such as:

```text
N in {16, 32, 64, 128}
B in {100, 1_000, 10_000}
physical_dim = 2
controlled bond dimension
TCI-like batches with repeated subtree assignments
```

Acceptance criteria:

- Linear-chain `TreeTNCachedEvaluator` should show the same qualitative scaling as `TTCache::evaluate_many`.
- Current `TreeTN::evaluate` should be visibly worse for large `B` because it allocates per point.
- Center search overhead should be small compared with cached evaluation for large batches.
- Benchmark output should include Criterion medians or equivalent timing tables for `N` and `B` dependence.

## Test Strategy

Use test-driven development. Start with a failing behavior test that calls the planned public cached evaluator API on a small chain and compares against `TreeTN::evaluate`. Then add tests for center selection and error paths before broadening to more topologies.

Focused verification commands:

```bash
cargo test --release -p tensor4all-treetn treetn::cached_evaluator
cargo bench -p tensor4all-treetn --bench cached_evaluator
cargo test --release -p tensor4all-simplett cache
```

Full pre-PR verification remains the repository standard:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --release --workspace
cargo doc --workspace --no-deps
```
