# Design: `tensor4all-treetci` MVP Port from `TreeTCI.jl`

**Date:** 2026-03-26  
**Scope:** Add a new `crates/tensor4all-treetci` crate that ports the core `TreeTCI.jl` algorithm on top of the newly introduced `matrixluci` substrate and `tensor4all-treetn`.  
**Related issues:** `#332`, `#338`

## 1. Context

The original feature request is issue `#332`, which targets a Rust port of
`TreeTCI.jl`.

That work was previously blocked on the LUCI substrate. PR `#338` introduced
`matrixluci`, migrated `TensorCI2` to it, and removed the main low-level blocker.

This document defines the first TreeTCI porting stage:

- introduce a dedicated `tensor4all-treetci` crate
- port the tree-specific state, graph utilities, candidate assembly, and
  optimization loop from `TreeTCI.jl`
- use `matrixluci` directly for pivot selection
- materialize the final approximation as a `TreeTN`

## 2. Upstream Attribution and Copyright

This port must preserve upstream attribution explicitly.

- The upstream Julia package is `TreeTCI.jl`.
- `TreeTCI.jl/Project.toml` currently lists:
  - `Ryo Watanabe <https://github.com/Ryo-wtnb11>`
- `README.md`, crate-level docs, and porting comments must not omit this
  authorship.
- The new crate should also state that its dense/lazy pivot substrate depends on
  `matrixluci`, whose dense path is itself a tensor4all-owned port informed by
  `faer`.

This is not optional bookkeeping. The Rust port should make the provenance and
 ownership chain visible.

## 3. Goals

### 3.1 Functional goals

- Add `crates/tensor4all-treetci` in this monorepo.
- Port the MVP algorithm from `TreeTCI.jl`:
  - tree bipartition utilities
  - `SimpleTCI`-like state
  - edge visitation with `AllEdges`
  - pivot candidate generation with `DefaultProposer`
  - edge update loop using `matrixluci`
  - final `TreeTN` materialization
- Use a batch callback interface with:
  - global site order `0..n_sites-1`
  - shape `(n_sites, n_points)`
  - column-major storage
- Keep the Rust surface generic enough that additional strategies can be added
  without reworking the state model.

### 3.2 Validation goals

- Reproduce direct `TreeTCI.jl` parity on small tree examples.
- Add tree-graph utility tests corresponding to `TreeTCI.jl/test/treegraph_utils_tests.jl`.
- Add end-to-end interpolation tests corresponding to
  `TreeTCI.jl/test/simpletci_tests.jl`.
- Reserve a final validation stage that imports stronger integration tests from:
  - `TensorCrossInterpolation.jl`
  - `QuanticsTCI.jl`
- Use `quanticsgrids-rs` where appropriate for the advanced quantics-derived tests.

## 4. Non-goals for PR1

- Strategy completeness:
  - `RandomEdges`
  - `LocalAdjacent`
  - `Adaptive`
  - `TruncatedProposer`
  - `SimpleProposer`
- Binding work (C API / Python / Julia)
- Performance optimization beyond what is needed for a clean MVP
- Porting every upstream Julia test in the first PR

These remain follow-up work after the MVP is working and covered by direct
parity tests.

## 5. Algorithmic Model

### 5.1 Bonds are bipartitions, not chain-adjacent sites

In TreeTCI, a bond is defined by an edge-induced bipartition of the site set.
This differs from chain TCI and is the central design constraint.

For an undirected edge `(u, v)`:

- removing the edge splits the site set into two connected components
- each component is represented as a canonical ordered subtree key
- pivot sets are stored per subtree key, not per adjacent-site bond index

This gives the desired invariance:

- reconnecting sites within one partition does not invalidate the partition's
  pivot interpretation

### 5.2 No forward/backward sweep

The optimizer does not model left-to-right or right-to-left sweeps.

Instead:

- choose an ordered list of edges to visit
- update pivots on each edge in that order
- repeat until convergence or iteration limit

This matches the upstream `TreeTCI.jl` structure and the issue `#332`
discussion.

## 6. Core Types

### 6.1 Subtree key

TreeTCI needs a canonical, hashable representation for one side of an
bipartition:

```rust
pub struct SubtreeKey(Box<[usize]>);
```

Semantics:

- stores site ids in ascending global site order
- acts as the key for pivot sets and bond errors
- must be deterministic for both directions of every edge

### 6.2 Tree graph metadata

Introduce a lightweight graph helper owned by `tensor4all-treetci`.

Responsibilities:

- validate that the input graph is a tree
- map user-facing node names to internal `usize` site ids
- precompute both subtree keys for every oriented edge
- expose adjacency, candidate-edge, and edge-distance utilities needed by
  upstream algorithms

This helper should stay algorithm-focused. It should not duplicate the full
 responsibilities of `tensor4all-treetn`.

### 6.3 TreeTCI state

The Rust state should mirror upstream `SimpleTCI` closely:

```rust
pub struct SimpleTreeTci<T> {
    pub local_dims: Vec<usize>,
    pub graph: TreeTciGraph,
    pub ijset: HashMap<SubtreeKey, Vec<MultiIndex>>,
    pub bond_errors: HashMap<UndirectedEdge, f64>,
    pub pivot_errors: Vec<f64>,
    pub max_sample_value: f64,
    pub ijset_history: Vec<HashMap<SubtreeKey, Vec<MultiIndex>>>,
}
```

Additional internal metadata may be added, but the algorithm should stay close
to the Julia state layout to reduce semantic drift.

## 7. Strategy Traits

Issue `#332` already fixed the trait shape conceptually. The MVP only implements
the default choices, but the public API should already reflect the intended
extension points.

### 7.1 Edge visitor

```rust
pub trait EdgeVisitor {
    fn visit_order(
        &self,
        graph: &TreeTciGraph,
        state: &TreeTciStateView<'_>,
    ) -> Vec<UndirectedEdge>;
}
```

PR1 implementation:

- `AllEdges`

Planned follow-ups:

- `RandomEdges`
- `LocalAdjacent`
- `Adaptive`

### 7.2 Pivot candidate proposer

```rust
pub trait PivotCandidateProposer {
    fn candidates(
        &self,
        graph: &TreeTciGraph,
        state: &TreeTciStateView<'_>,
        edge: UndirectedEdge,
    ) -> crate::Result<(Vec<MultiIndex>, Vec<MultiIndex>)>;
}
```

PR1 implementation:

- `DefaultProposer`

Planned follow-ups:

- `TruncatedProposer`
- `SimpleProposer`

The default proposer must follow the upstream logic exactly:

- gather neighboring subtree pivot sets on both sides
- form their Cartesian products
- insert the local site dimension for the edge endpoint
- union with the previous-iteration history for non-strictly-nested candidate
  growth

## 8. Batch Evaluation Interface

The user callback should see only global site-order data.

```rust
pub struct GlobalIndexBatch<'a> {
    pub data: &'a [usize], // column-major
    pub n_sites: usize,
    pub n_points: usize,
}
```

Semantics:

- logical shape: `(n_sites, n_points)`
- storage: column-major
- one sample point is one column
- site order is always `0, 1, ..., n_sites - 1`

Internal subtree-local pivot tuples are implementation details. TreeTCI must
assemble them into global site-order batches before calling the user function.

This preserves the intended API alignment with chain TCI and tenferro-side
column-major conventions.

## 9. Optimization Flow

PR1 should implement the following flow:

1. Construct `SimpleTreeTci`.
2. Seed global pivots into per-subtree `ijset`.
3. Repeatedly:
   - obtain edge visit order from `AllEdges`
   - build `(I_candidates, J_candidates)` with `DefaultProposer`
   - assemble a candidate matrix callback in global site order
   - run `matrixluci` pivot selection
   - update `ijset`
   - update bond and pivot errors
   - append history snapshots as needed by the proposer
4. Materialize a `TreeTN`.
5. Return `(TreeTN, ranks, errors)` or the Rust-equivalent result object.

PR1 may start with dense candidate materialization per edge update if that keeps
the port faithful and small. A later optimization pass can introduce lazy edge
assembly if profiling shows it is needed.

## 10. TreeTN Materialization

The final representation should be a `TreeTN`, not a custom tree tensor type.

Responsibilities:

- compute site tensors from the converged subtree pivot sets
- preserve the original tree topology
- allow an explicit center choice or use a deterministic default

TreeTCI should own the TreeTCI-specific tensor assembly logic. It should use
`tensor4all-treetn` only as the final network representation layer.

## 11. Testing Strategy

### 11.1 PR1 tests

Port direct upstream tests first:

- tree graph utilities:
  - `subtreevertices`
  - `subregionvertices`
  - `adjacentedges`
  - `candidateedges`
  - `distanceedges`
- simple interpolation:
  - the same small 7-site tree shape used in `TreeTCI.jl`
  - value checks for `f(v) = 1 / (1 + v'v)` or the Rust equivalent
- batch semantics:
  - verify callback sees global site order
  - verify column-major point packing

### 11.2 Final validation stage

After the MVP is working, add stronger integration tests inspired by
`TensorCrossInterpolation.jl` and `QuanticsTCI.jl`.

These tests should focus on invariants that still make sense for tree TCI:

- interpolation accuracy on nontrivial functions
- initial-pivot handling
- integral/sum checks on quantized grids where applicable
- rank growth behavior on structured low-rank functions
- topology changes within one bipartition not changing evaluation semantics

For the quantics-derived cases, use `quanticsgrids-rs` instead of reimplementing
grid logic.

## 12. PR Structure

### PR1: MVP port

- add `crates/tensor4all-treetci`
- port tree utilities and `SimpleTreeTci`
- implement `AllEdges`
- implement `DefaultProposer`
- integrate `matrixluci`
- materialize `TreeTN`
- add direct `TreeTCI.jl` parity tests

### PR2: strategy expansion and advanced validation

- add additional visitors and proposers
- add advanced integration tests from upstream Julia packages
- optimize hot paths based on benchmark evidence

This split keeps the first PR algorithmically meaningful while avoiding a very
large initial review surface.
