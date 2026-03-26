# TreeTCI MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `crates/tensor4all-treetci` crate that ports the core
`TreeTCI.jl` algorithm on top of `matrixluci` and `tensor4all-treetn`, with
direct parity tests for the upstream Julia package.

**Architecture:** Model TreeTCI around edge-induced bipartitions, canonical
subtree keys, an upstream-like `SimpleTreeTci` state, and a batch callback that
always uses global site order in column-major layout. Use `matrixluci` for the
per-edge pivot selection kernel and materialize the result as a `TreeTN`.

**Tech Stack:** Rust workspace crate, `petgraph` or existing graph support,
`matrixluci`, `tensor4all-treetn`, `cargo nextest`, upstream `TreeTCI.jl`
reference code, and later `quanticsgrids-rs` for advanced tests.

**Attribution requirement:** Preserve the upstream `TreeTCI.jl` authorship in
crate docs and README. Do not omit `Ryo Watanabe
<https://github.com/Ryo-wtnb11>`.

---

### Task 1: Scaffold `tensor4all-treetci` crate and crate-level docs

**Files:**
- Create: `crates/tensor4all-treetci/Cargo.toml`
- Create: `crates/tensor4all-treetci/README.md`
- Create: `crates/tensor4all-treetci/src/lib.rs`
- Modify: workspace `Cargo.toml`

**Implementation notes:**
- Depend on:
  - `anyhow`
  - `thiserror`
  - `matrixluci`
  - `tensor4all-treetn`
  - supporting tensor/core crates as needed
- State clearly in crate docs and README:
  - this is a Rust port of `TreeTCI.jl`
  - upstream authorship includes `Ryo Watanabe`
  - pivot selection uses `matrixluci`

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci`

### Task 2: Port tree graph utility layer

**Files:**
- Create: `crates/tensor4all-treetci/src/graph.rs`
- Create: `crates/tensor4all-treetci/src/key.rs`
- Create: `crates/tensor4all-treetci/src/graph/tests.rs`

**Implementation notes:**
- Introduce:
  - canonical undirected edge type
  - `SubtreeKey`
  - `TreeTciGraph`
- Port the behavior of:
  - `subtreevertices`
  - `subregionvertices`
  - `adjacentedges`
  - `candidateedges`
  - `distanceedges`
- Keep site ids in ascending global order for subtree keys

**Tests to port first:**
- the 7-site example tree from `TreeTCI.jl/test/treegraph_utils_tests.jl`
- exact equality on subtree keys, candidate edges, and edge-distance maps

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci graph::tests`

### Task 3: Add batch-index and candidate-assembly primitives

**Files:**
- Create: `crates/tensor4all-treetci/src/batch.rs`
- Create: `crates/tensor4all-treetci/src/index.rs`
- Create: `crates/tensor4all-treetci/src/assemble.rs`
- Create: `crates/tensor4all-treetci/src/assemble/tests.rs`

**Implementation notes:**
- Define:
  - `GlobalIndexBatch`
  - subtree-local `MultiIndex`
  - helpers that assemble subtree-local pivots into global site-order batches
- Match the upstream `filltensor` semantics:
  - user callback sees all sites
  - output layout is column-major `(n_sites, n_points)`
- Support single-point and batch evaluation helpers without exposing subtree-local
  ordering publicly

**Tests:**
- verify assembled global indices match the expected site-ascending placement
- verify nontrivial subtree splits are packed column-major

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci assemble::tests`

### Task 4: Port `SimpleTreeTci` state and default seeding

**Files:**
- Create: `crates/tensor4all-treetci/src/state.rs`
- Create: `crates/tensor4all-treetci/src/options.rs`
- Create: `crates/tensor4all-treetci/src/state/tests.rs`

**Implementation notes:**
- Add Rust equivalents for:
  - local dims
  - graph metadata
  - `ijset`
  - bond errors
  - pivot errors
  - max sample value
  - history snapshots
- Port global pivot seeding into per-subtree pivot sets
- Keep state layout intentionally close to upstream `SimpleTCI`

**Tests:**
- constructor rejects non-tree graphs
- initial global pivots propagate to each edge bipartition as expected

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci state::tests`

### Task 5: Add strategy traits and MVP implementations

**Files:**
- Create: `crates/tensor4all-treetci/src/visitor.rs`
- Create: `crates/tensor4all-treetci/src/proposer.rs`
- Create: `crates/tensor4all-treetci/src/proposer/tests.rs`

**Implementation notes:**
- Define public traits:
  - `EdgeVisitor`
  - `PivotCandidateProposer`
- Implement only:
  - `AllEdges`
  - `DefaultProposer`
- Ensure `DefaultProposer` matches Julia:
  - neighbor subtree pivot collection
  - Cartesian product
  - local-dimension insertion
  - history union

**Tests:**
- candidate generation on a small branchy tree
- history union is preserved
- edge visit order covers all edges deterministically

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci proposer::tests`

### Task 6: Implement per-edge update with `matrixluci`

**Files:**
- Create: `crates/tensor4all-treetci/src/update.rs`
- Create: `crates/tensor4all-treetci/src/update/tests.rs`
- Modify: `crates/tensor4all-treetci/src/state.rs`

**Implementation notes:**
- For each edge:
  - generate candidates
  - assemble a candidate matrix callback
  - call `matrixluci`
  - map selected row/col indices back to subtree-local pivots
  - update `ijset`, bond error, pivot error, and `max_sample_value`
- Preserve the upstream semantics as closely as possible rather than inventing a
  Rust-only update model

**Tests:**
- simple diagonal/separable functions choose non-empty pivots
- repeated edge update is stable on an already-converged simple case
- pivot error bookkeeping remains finite and deterministic

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci update::tests`

### Task 7: Implement the MVP optimization loop

**Files:**
- Create: `crates/tensor4all-treetci/src/optimize.rs`
- Create: `crates/tensor4all-treetci/src/optimize/tests.rs`
- Modify: `crates/tensor4all-treetci/src/options.rs`

**Implementation notes:**
- Add:
  - options for tolerance, max iterations, and initialization
  - `SimpleTreeTci::optimize`
- PR1 uses:
  - `AllEdges`
  - `DefaultProposer`
- Do not add sweep-direction concepts
- Stop on bond-error convergence or iteration limit

**Tests:**
- optimization loop converges on a small low-rank example
- ranks/errors history has expected length and monotonic sanity properties

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci optimize::tests`

### Task 8: Materialize the result as a `TreeTN`

**Files:**
- Create: `crates/tensor4all-treetci/src/materialize.rs`
- Create: `crates/tensor4all-treetci/src/materialize/tests.rs`

**Implementation notes:**
- Port the upstream tensor assembly logic needed to produce the final tree tensor
  network
- Build a `TreeTN` using public `tensor4all-treetn` APIs only
- Keep the materialization center deterministic

**Tests:**
- a small hand-checkable tree contracts to the expected dense result
- output topology matches the requested tree shape

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci materialize::tests`

### Task 9: Add the public high-level API and direct parity tests

**Files:**
- Create: `crates/tensor4all-treetci/src/api.rs`
- Create: `crates/tensor4all-treetci/tests/simple_parity.rs`
- Modify: `crates/tensor4all-treetci/src/lib.rs`

**Implementation notes:**
- Add a high-level entry point similar to upstream `crossinterpolate`
- Return the TreeTN approximation plus rank/error diagnostics
- Port the direct upstream 7-site example from
  `TreeTCI.jl/test/simpletci_tests.jl`

**Tests:**
- evaluate the same sample points as the Julia test
- verify batch callback sees global site order and column-major point packing

**Verification:**
- `cargo nextest run --release -p tensor4all-treetci --test simple_parity`

### Task 10: Integrate docs and workspace verification

**Files:**
- Modify: root `README.md`
- Modify: any crate-level docs touched by the new public API

**Implementation notes:**
- Add the new crate to the project structure overview
- Mention the upstream port and authorship in the new crate README
- Keep README statements aligned with the implemented PR1 scope

**Verification:**
- `cargo fmt --all`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release -p tensor4all-treetci -p tensor4all-treetn -p tensor4all-tensorci`

### Task 11: Follow-up phase after PR1

This is not part of the MVP PR, but it should already be tracked.

**Follow-up work:**
- add:
  - `RandomEdges`
  - `LocalAdjacent`
  - `Adaptive`
  - `TruncatedProposer`
  - `SimpleProposer`
- add advanced integration tests inspired by:
  - `TensorCrossInterpolation.jl`
  - `QuanticsTCI.jl`
- use `quanticsgrids-rs` for quantics-derived grid and integral tests
- profile and optimize edge update hot paths only after correctness is pinned down
