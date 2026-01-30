# Linsolve (TreeTN) — Control Flow & Key Components

This document summarizes how the TreeTN linsolve implementation is structured and how the main
components interact during a sweep.

The goal of linsolve is to approximately solve a linear system in TreeTN form:

- \((a_0 I + a_1 A)\,x = b\)

where:
- `A` is represented as an MPO-like `TreeTN` (operator network)
- `x` and `b` are represented as `TreeTN` (state networks)

## Module Structure

The linsolve implementation is organized into:

- **`src/linsolve/common/`**: Shared infrastructure
  - `environment.rs` - `EnvironmentCache` for caching contracted environments
  - `options.rs` - `LinsolveOptions` configuration
  - `projected_operator.rs` - `ProjectedOperator` for 3-chain environment management

- **`src/linsolve/square/`**: V_in = V_out solver (input/output spaces are the same)
  - `updater.rs` - `SquareLinsolveUpdater` implementing `LocalUpdater`
  - `projected_state.rs` - `ProjectedState` for 2-chain RHS environments
  - `local_linop.rs` - `LocalLinOp` wrapper for GMRES

## Entry Points

- **Sweep driver (generic local-update infrastructure)**:
  - `crates/tensor4all-treetn/src/treetn/localupdate.rs`
  - Main helper used by examples/tests: `apply_local_update_sweep(...)`
- **Linsolve implementation (a `LocalUpdater`)**:
  - `crates/tensor4all-treetn/src/linsolve/square/updater.rs`
  - Type: `SquareLinsolveUpdater<T, V>`

## High-level Call Graph (one sweep step)

During a sweep, the generic sweep driver repeatedly calls the linsolve updater:

1. `apply_local_update_sweep(full_treetn, plan, updater)`
2. For each `LocalUpdateStep { nodes, new_center }`:
   - `SquareLinsolveUpdater::before_step(step, full_treetn_before)` (lazy init / pre-step hooks)
   - extract a local subtree (region)
   - call `SquareLinsolveUpdater::update(subtree, step, full_treetn)`
   - insert the updated subtree back
   - move canonical center to `new_center`
   - call `SquareLinsolveUpdater::after_step(step, full_treetn_after)` (reference state sync + cache invalidation)

## What happens inside `SquareLinsolveUpdater::update`

Source: `crates/tensor4all-treetn/src/linsolve/square/updater.rs`

For a region `step.nodes` (typically 2-site / 2-node update in current examples):

### 1) Region "cut-out": contract region tensors into a single local tensor

- The updater takes the extracted `subtree: TreeTN` and forms a *single* tensor `init_local: T`
  representing the degrees of freedom of the update region.
- This is the "local vector" seen by GMRES.

Conceptually:
- `init_local := contract( tensors in region )`

### 2) Local linear solve (GMRES)

The updater solves a *local* linear system:

- \((a_0 I + a_1 H_{\text{local}})\,x_{\text{local}} = b_{\text{local}}\)

Implementation details:

- `SquareLinsolveUpdater::solve_local(region, init_local, full_treetn)`
  - Builds a `LocalLinOp` (local linear operator wrapper)
  - Defines `apply_a = |x_local: &T| -> Result<T> { linop.apply(x_local) }`
  - Calls `tensor4all_core::krylov::gmres(apply_a, rhs_local, init_local, options)`

Relevant files:
- `crates/tensor4all-treetn/src/linsolve/square/updater.rs`
- `crates/tensor4all-treetn/src/linsolve/square/local_linop.rs`

### 3) Local operator application: `LocalLinOp` → `ProjectedOperator`

`LocalLinOp::apply(x_local)` computes:

- `hx_local = (a_0 I + a_1 H_local) x_local`

It delegates the expensive part to `ProjectedOperator::apply(...)`:

- `ProjectedOperator` contracts:
  - operator tensors on the open region
  - precomputed environments from outside the region
  - the input local tensor `x_local` (with index mappings if needed)

Relevant files:
- `crates/tensor4all-treetn/src/linsolve/square/local_linop.rs`
- `crates/tensor4all-treetn/src/linsolve/common/projected_operator.rs`
- `crates/tensor4all-treetn/src/linsolve/common/environment.rs` (cache structure)

## Reference State Separation (current implementation)

To avoid unintended bra↔ket contractions while still reusing cached environments,
the square linsolve keeps a persistent reference state:

- `SquareLinsolveUpdater` owns a `reference_state: TreeTN<T, V>` whose **link indices use different IDs** from
  the ket (`full_treetn`).
- On each step, `after_step` updates the reference state region to track the updated ket region while keeping
  **boundary bond IDs stable** (region ↔ outside), so cached environments remain connectable.
- The boundary bonds for a region are computed via the shared helper:
  - `get_boundary_edges(treetn, region)` in `crates/tensor4all-treetn/src/treetn/localupdate.rs`

Operationally:
- `solve_local` constructs `LocalLinOp` with a reference state, so
  `ProjectedOperator::apply(..., ket_state=state, reference_state=self.reference_state, ...)` uses distinct
  ket/reference link namespaces and avoids spurious traces when contracting with `AllowedPairs::All`.

### 4) Local RHS construction: `ProjectedState`

The local RHS tensor `b_local` is constructed via environments of the RHS state network:

- `ProjectedState::local_constant_term(region, ket_state, topology)` (2-chain contraction)

Relevant file:
- `crates/tensor4all-treetn/src/linsolve/square/projected_state.rs`

### 5) "Embed back": factorize solved local tensor into a TreeTN on the region

Once GMRES returns `solved_local: T`, the updater factorizes it back into a small TreeTN matching
the region topology:

- Build a `TreeTopology` describing how indices in `solved_local` correspond to region nodes/edges
- `factorize_tensor_to_treetn_with(&solved_local, &topology, FactorizeAlg::SVD, root=&new_center)`

Relevant files:
- `crates/tensor4all-treetn/src/linsolve/square/updater.rs`
- `crates/tensor4all-treetn/src/treetn/decompose.rs`

### 6) Replace tensors/bonds in the subtree (preserve appearance)

The decomposed region TreeTN is copied back into the extracted `subtree`, updating:

- node tensors
- the internal region bond index on the edge(s)
- canonical-center bookkeeping

Relevant code lives in:
- `crates/tensor4all-treetn/src/linsolve/square/updater.rs` (copy back logic)
- `crates/tensor4all-treetn/src/treetn/mod.rs` (e.g. `replace_edge_bond`)

### 7) Cache invalidation after each step

After the updated subtree is inserted into the global TreeTN, the sweep infrastructure calls:

- `SquareLinsolveUpdater::after_step(step, full_treetn_after)`

This performs (1) reference state synchronization for the updated region and then (2) invalidates environment
caches in both:
- `ProjectedOperator` (3-chain)
- `ProjectedState` (2-chain)

Source:
- `crates/tensor4all-treetn/src/linsolve/square/updater.rs`

## Index mapping (MPO internal indices vs state site indices)

Operators may use internal site indices (e.g. `s_in_tmp`, `s_out_tmp`) whose IDs differ from
the "true" site indices in the state network. In that case:

- `ProjectedOperator::apply` transforms the input local tensor's site indices into the MPO's
  internal input indices before contraction, and transforms the output back afterwards.

Source:
- `crates/tensor4all-treetn/src/linsolve/common/projected_operator.rs`

## Known pitfall: unintended contraction due to shared index IDs

The contraction engine treats indices as contractable if they share the same ID (plus compatible
`ConjState`, dimensions). When bra/ket share the same index IDs, `AllowedPairs::All` can contract
more aggressively than intended if the implementation does not explicitly separate bra/ket index
namespaces or restrict contraction pairs.

This is the direction for "root-cause" fixes:
- separate bra/ket index identities (e.g. via priming / directed indices), and/or
- constrain contractions (e.g. `AllowedPairs::Specified`) so environments do not introduce
  unintended traces.

Related reproducer examples:
- `crates/tensor4all-treetn/examples/test_linsolve_identity_residual_n3_variants.rs`

## Call Graph (current implementation)

The following diagram shows the main runtime call chain during a sweep step.

```text
apply_local_update_sweep(state: TreeTN, plan, updater)
  └─ for step in plan.steps:
       ├─ SquareLinsolveUpdater::before_step(step, full_treetn_before)
       │    └─ ensure_reference_state_initialized(full_treetn_before)   // lazy init
       ├─ (extract subtree for step.nodes)            // localupdate infrastructure
       ├─ SquareLinsolveUpdater::update(subtree, step, full_treetn)
       │    ├─ contract_region(subtree, step.nodes) -> init_local: T
       │    ├─ solve_local(region, init_local, state=full_treetn) -> solved_local: T
       │    │    ├─ ProjectedState::local_constant_term(...) -> rhs_local: T
       │    │    ├─ linop = LocalLinOp::new(projected_operator, region, state.clone(), reference_state.clone(), a0, a1)
       │    │    └─ gmres(apply_a, rhs_local, init_local, gmres_options)
       │    │         └─ apply_a(x_local: &T) = LocalLinOp::apply(x_local)
       │    │              ├─ ProjectedOperator::apply(v=x_local, region, ket_state=state, reference_state, topology)
       │    │              │    ├─ ensure_environments(...)
       │    │              │    │    └─ compute_environment(from, to, ket_state, reference_state, topology)   // recursive
       │    │              │    └─ contract([v, op(region), envs(outside)]) -> H_local * v
       │    │              └─ return (a0 * x_local + a1 * (H_local * x_local))
      │    ├─ factorize_tensor_to_treetn_with(solved_local, topology, SVD, root=new_center) -> decomposed_region: TreeTN
       │    └─ copy_decomposed_to_subtree(subtree, decomposed_region, step.nodes, full_treetn)
       ├─ (insert updated subtree back into full_treetn)
       └─ SquareLinsolveUpdater::after_step(step, full_treetn_after)
            ├─ sync_reference_state_region(step, full_treetn_after)     // keep boundary bonds stable
            ├─ ProjectedOperator::invalidate(step.nodes, topology)
            └─ ProjectedState::invalidate(step.nodes, topology)
```
