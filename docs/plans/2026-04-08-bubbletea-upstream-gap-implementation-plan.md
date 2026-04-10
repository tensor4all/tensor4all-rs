# BubbleTeaCI Upstream Gap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the remaining general-purpose backend APIs in `quanticsgrids-rs` and `tensor4all-rs` so that `BubbleTeaCI-rs` can complete its migration without dense fallbacks or BubbleTeaCI-specific backend hacks.

**Architecture:** This is a single pre-PR batch. The work starts in `../quanticsgrids-rs` for grid semantics, then updates `tensor4all-rs` crates that should consume those grid primitives instead of re-deriving layout ad hoc. `BubbleTeaCI-rs` is the downstream verifier, but it must stay on local path dependencies until the migration is complete; no `tensor4all-rs` PR should be opened before the downstream cutover succeeds end-to-end.

**Tech Stack:** Rust 2021, `quanticsgrids-rs`, `tensor4all-quanticstransform`, `tensor4all-treetn`, `tensor4all-quanticstci`, `cargo test --release`, `cargo nextest run --release`, `cargo run -p api-dump --release -- . -o docs/api`, `BubbleTeaCI-rs` local-path verification.

---

## Derived From

- Workspace boundary design:
  - `/home/shinaoka/tensor4all/docs/plans/2026-04-08-bubbletea-migration-boundary-design.md`
- Umbrella issue:
  - `tensor4all-rs#405`

## Execution Rules

1. Do **not** open or update a `tensor4all-rs` PR until `BubbleTeaCI-rs` has been migrated against local backend changes and its test suite passes.
2. During this batch, `tensor4all-rs` should temporarily depend on local `../quanticsgrids-rs` instead of the pinned git revision.
3. When downstream verification starts, `BubbleTeaCI-rs` must also temporarily depend on local `../quanticsgrids-rs` so that `quanticsgrids` types come from the same crate source as `tensor4all-rs`.
4. Keep the boundary strict:
   - general grid / TT / operator primitives upstream
   - `TTFunction`, `BasicContractOrder`, `times_dV`, result-grid semantics downstream
5. Because `tensor4all-rs` is early-stage and does not promise backwards compatibility, remove or rename bad APIs outright instead of carrying deprecated aliases.

## Non-Goals For This Batch

- Upstreaming `TTFunction`
- Upstreaming `BasicContractOrder`
- Upstreaming BubbleTeaCI variable-name DSLs or physics semantics
- AD-related work
- Adding downstream-only pack/unpack helpers for component axes

## Current Status

Backend Tasks 1-9 are complete for the current migration slice.

Resolved upstream in this batch:

- transform-layer dense fallbacks are no longer needed downstream
- manual operator alignment rewrites are no longer needed downstream
- ad hoc simple-TT -> `TreeTN` conversion is gone
- exact index unfusing landed for packed component-axis materialization
- backend-native component-valued transforms now pass downstream after the
  topology-preservation fix in `factorize_tensor_to_treetn(...)`
- `quanticsgrids-rs` now provides structural `PartialEq` limited to
  constructor-visible metadata
- `quanticsgrids-rs` no longer exports Julia-absent public layout-query or
  metadata-algebra APIs; consumer crates use internal helpers instead
- the public transform selector API now uses
  `shift_operator_on_grid_by_variable_name(...)`
- grouped multi-variable affine application on higher-resolution states now
  works through the public operator-apply path
- `tensor4all-quanticstransform` now provides
  `affine_pullback_operator_on_grid(...)`
- open-boundary 1D shift semantics are now correct for both positive and
  negative offsets, with release-mode regression coverage at the grid-aware and
  integration layers

Fresh downstream verification result:

```bash
cd /home/shinaoka/tensor4all/BubbleTeaCI-rs
cargo test
```

Status:

- full downstream suite passes for the current migration slice
- the original cross-topology contraction pattern from `basic_operations.jl`
  now lowers backend-natively through generalized `partial_contract(...)`
- the higher-dimensional affine embedding slice now passes downstream through
  backend-native grouped pullback transforms

## Task 10: Generalize Operator Apply For Grouped Multi-Site Affine Operators

**Status:** complete

**Delivered:**

- grouped/interleaved affine operators now preserve a chain-compatible topology
  when factorized into binary local sites
- the public apply path succeeds on grouped higher-resolution states
- grid-aware pullback semantics are available through
  `affine_pullback_operator_on_grid(...)`
- the regression is covered in
  `crates/tensor4all-quanticstransform/tests/integration_test.rs`

## Task 1: Local Dependency Wiring And Acceptance Harness

**Files:**
- Modify: `Cargo.toml`
- Read only for later downstream verification: `../BubbleTeaCI-rs/Cargo.toml`

**Step 1: Switch `tensor4all-rs` to local `quanticsgrids-rs`**

Edit `Cargo.toml`:

```toml
[workspace.dependencies]
quanticsgrids = { path = "../quanticsgrids-rs" }
```

Do not touch any other workspace dependency yet.

**Step 2: Run the narrow pre-change acceptance set**

Run:

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
cargo test --release

cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-quanticstransform
cargo nextest run --release -p tensor4all-treetn
cargo nextest run --release -p tensor4all-quanticstci
```

Expected:

- all current tests pass
- no downstream code is touched yet

**Step 3: Record the downstream verification rule in the working notes**

Before touching implementation code, note in the execution log or working notes:

- `BubbleTeaCI-rs` will stay on local path dependencies
- no upstream PR before downstream migration succeeds

**Step 4: Commit local dependency wiring**

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
git add Cargo.toml docs/plans/2026-04-08-bubbletea-upstream-gap-implementation-plan.md
git commit -m "chore: switch quanticsgrids to local path for BubbleTeaCI migration"
```

## Task 2: Add Structural Equality To `quanticsgrids-rs`

**Status:** complete

**Files:**
- Modify: `../quanticsgrids-rs/src/lib.rs`
- Modify: `../quanticsgrids-rs/src/inherent_discrete_grid.rs`
- Modify: `../quanticsgrids-rs/src/discretized_grid.rs`
- Test: `../quanticsgrids-rs/src/inherent_discrete_grid/tests/mod.rs`
- Test: `../quanticsgrids-rs/src/discretized_grid/tests/mod.rs`

**APIs to add in this task:**

- `impl PartialEq for InherentDiscreteGrid`
- `impl PartialEq for DiscretizedGrid`

**Scope restriction:**

- do not add layout-query APIs in this batch
- do not add `same_layout`, `variable_id(s)`, `variable_sites`, or
  `unfolding_scheme()` getters unless the Julia grid library first grows a
  matching public concept

**Step 1: Write failing tests for equality**

Add unit tests that express the intended public behavior:

```rust
#[test]
fn test_discretized_grid_partial_eq_tracks_bounds_and_layout() {
    let a = DiscretizedGrid::builder(&[2, 2])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    let b = DiscretizedGrid::builder(&[2, 2])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    assert_eq!(a, b);
}

```

Also add tests that check:

- differing bounds are not equal
- differing `indextable` values are not equal
- differing `includeendpoint` values are not equal

**Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
cargo test --release discretized_grid
```

Expected:

- compile failure or missing-trait failures for `PartialEq`

**Step 3: Implement the minimal public API**

Implementation requirements:

- `PartialEq` must be structural, not approximate
- compare constructor-visible metadata, not derived caches
- do not use `PartialEq` as a back door for layout-compatibility semantics

**Step 4: Add doc comments with runnable examples**

Every new public API must get examples with assertions. Follow the repository rule for public API docs.

**Step 5: Run tests to verify they pass**

Run:

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
cargo test --release
```

Expected:

- all `quanticsgrids-rs` tests pass

**Step 6: Commit**

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
git add src/lib.rs src/inherent_discrete_grid.rs src/discretized_grid.rs src/inherent_discrete_grid/tests/mod.rs src/discretized_grid/tests/mod.rs
git commit -m "feat: add quantics grid structural equality"
```

**Implemented outcome:**

- `PartialEq` landed for both `InherentDiscreteGrid` and `DiscretizedGrid`
- equality is structural and exact
- equality ignores derived caches and hidden requested-scheme state
- degenerate layouts that collapse to the same public metadata compare equal
- no new public layout-query APIs were introduced alongside `PartialEq`

## Task 3: Grid Metadata Algebra

**Status:** out of scope for this batch

`project_grid`, `reduce_grid`, `refine_grid`, and `rename_variables` should not
be added to `quanticsgrids-rs` in this batch. `QuanticsGrids.jl` `origin/main`
does not currently expose matching public APIs, so adding them only on the Rust
side would create an avoidable cross-language divergence.

These helpers remain downstream for now.

**Files:** none in this batch

**Note:**

- do not execute this task unless the Julia grid library grows matching public
  APIs first

## Task 9: Preserve Spectator Site Space During Partial Operator Application

**Status:** complete

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/apply.rs`
- Modify: `crates/tensor4all-treetn/src/operator/linear_operator.rs`
- Test: `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs`
- Downstream verifier: `../BubbleTeaCI-rs/tests/transform.rs`

**Problem:**
- `shift_operator_on_grid(...)` / `affine_operator_on_grid(...)` build operators
  on the continuous subspace only
- downstream states may contain untouched component/spectator legs on other nodes
- applying the operator with `apply_linear_operator(...)` should act as identity
  on untouched nodes
- current behavior can leak internal operator indices into the result site-index
  network instead of preserving the original spectator site space exactly

**Required behavior:**
- if an operator is applied to only a subset of nodes, the result must keep
  exactly the same external site indices on untouched nodes
- on acted-upon nodes, the result should expose only the operator output site
  indices, not extra internal operator indices
- downstream should be able to reconstruct its wrapper with the original
  spectator site indices and the transformed continuous site indices without
  manual cleanup

**Minimum reproducer to upstream as a test:**
- build a 3-node chain state
- nodes 0 and 1 are the transform target subspace
- node 2 is an untouched spectator site
- apply a 2-node operator on nodes 0 and 1
- assert that the result's external/site index set is exactly the original
  three site indices, with node 2 unchanged

**Acceptance:**
- the new upstream test passes
- downstream `BubbleTeaCI-rs/tests/transform.rs::periodic_shift_preserves_component_legs`
  passes without dense fallback or downstream index cleanup

## Task 4: Rebase Grid-Aware Transforms On The Grid Layer And Fix API Naming

**Status:** complete with adjusted implementation strategy

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/grid_aware.rs`
- Modify: `crates/tensor4all-quanticstransform/src/lib.rs`
- Test: `crates/tensor4all-quanticstransform/src/grid_aware/tests/mod.rs`

**APIs to change in this task:**

- rename:
  - `shift_operator_on_grid_by_tag`
  - to `shift_operator_on_grid_by_variable_name`

Implementation note:

- after the Julia parity review, `quanticsgrids-rs` intentionally did **not**
  gain public layout-query APIs
- therefore `grid_aware.rs` now keeps local internal layout/variable helpers
  instead of depending on grid-query methods from `quanticsgrids-rs`
- the public `tag` wording was still removed from the transform API

**Step 1: Write failing tests for the renamed selector API**

Add tests like:

```rust
#[test]
fn test_shift_operator_on_grid_by_variable_name_grouped() {
    let grid = DiscretizedGrid::builder(&[3, 2])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();

    let op = shift_operator_on_grid_by_variable_name(
        &grid,
        &[("x", 1, BoundaryCondition::Periodic)],
    )
    .unwrap();

    assert_eq!(op.mpo.node_count(), grid.len());
}
```

Also add a failing test that asserts `grid_aware` no longer contains a local layout detector path.

**Step 2: Run the crate tests to confirm the rename breaks callers**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-quanticstransform
```

Expected:

- failures in old tests and docs until callers are updated

**Step 3: Implement the rename and keep layout inference internal**

Implementation requirements:

- bad naming using `tag` for variable names must disappear from public API
- internal layout inference in `grid_aware.rs` is acceptable as long as it
  stays non-public and `quanticsgrids-rs` keeps Julia-aligned public parity
- update examples and re-exports in `src/lib.rs`

**Step 4: Re-run tests**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-quanticstransform
```

Expected:

- all transform tests pass with the new names and no local layout decoder

**Step 5: Commit**

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
git add crates/tensor4all-quanticstransform/src/grid_aware.rs crates/tensor4all-quanticstransform/src/lib.rs crates/tensor4all-quanticstransform/src/grid_aware/tests/mod.rs
git commit -m "feat: rebase grid-aware transforms on quantics grid APIs"
```

## Task 5: Complete Grid-Aware Transform Coverage Needed By BubbleTeaCI-rs

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/grid_aware.rs`
- Test: `crates/tensor4all-quanticstransform/src/grid_aware/tests/mod.rs`

**Coverage to add in this task:**

- native grouped-layout affine path
- multi-variable interleaved shift path
- any remaining layout path currently returning `not yet implemented` that is still needed for BubbleTeaCI-rs basic transforms

Do not add BubbleTeaCI-specific subset-of-`TTFunction` semantics. Only fill out the grid-aware operator constructors.

**Step 1: Add failing structural and numeric tests**

Add tests that assert:

- grouped affine operator matches expected coordinate remapping
- interleaved multi-variable shift operator preserves site count and contracts correctly
- unsupported custom layouts still return a precise error instead of silently falling back

Example failing test skeleton:

```rust
#[test]
fn test_affine_operator_on_grid_grouped_matches_expected_remap() {
    let grid = DiscretizedGrid::builder(&[3, 3])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();

    let params = AffineParams::from_integers(vec![0, 1, 1, 0], vec![0, 0], 2, 2).unwrap();
    let op = affine_operator_on_grid(
        &grid,
        &params,
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();

    assert_eq!(op.mpo.node_count(), grid.len());
}
```

**Step 2: Run tests to verify the gaps are real**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-quanticstransform
```

Expected:

- failures on grouped affine and/or interleaved multi-variable paths

**Step 3: Implement the minimal backend-native coverage**

Implementation requirements:

- no dense fallback logic
- no downstream-specific assumptions
- use grid-variable site grouping supplied by `quanticsgrids`
- if a path remains impossible for a principled reason, return a narrow error that explains the missing general case

**Step 4: Re-run tests**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-quanticstransform
```

Expected:

- all tests pass

**Step 5: Commit**

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
git add crates/tensor4all-quanticstransform/src/grid_aware.rs crates/tensor4all-quanticstransform/src/grid_aware/tests/mod.rs
git commit -m "feat: complete grid-aware transform coverage for BubbleTeaCI migration"
```

## Task 6: Add TreeTN Site-Space Alignment And Partial-Contraction Output Control

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/addition.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/addition/tests/mod.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/partial_contraction.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/partial_contraction/tests/mod.rs`
- Modify if needed: `crates/tensor4all-treetn/src/treetn/mod.rs`

**APIs to add in this task:**

- a general site-space alignment helper for states, for example:

```rust
pub fn reindex_site_space_like(&self, template: &Self) -> Result<Self>;
```

- an ergonomic aligned-add helper if the first API proves useful for direct downstream use:

```rust
pub fn add_aligned(&self, other: &Self) -> Result<Self>;
```

- result ordering for partial contraction, for example by extending:

```rust
pub struct PartialContractionSpec<I: IndexLike> {
    pub contract_pairs: Vec<(I, I)>,
    pub multiply_pairs: Vec<(I, I)>,
    pub output_order: Option<Vec<I>>,
}
```

The exact field name is flexible, but the capability is not: callers must be able to prescribe the order of surviving external site indices without encoding BubbleTeaCI semantics in the backend.

**Step 1: Write failing tests for aligned add**

Add a unit test like:

```rust
#[test]
fn test_add_aligned_accepts_equivalent_site_space_with_different_ids() {
    let a = make_chain_state_with_site_ids(&[2, 2], 0);
    let b = make_chain_state_with_site_ids(&[2, 2], 100);
    let sum = a.add_aligned(&b).unwrap();
    assert!(sum.share_equivalent_site_index_network(&a));
}
```

**Step 2: Write failing tests for partial contraction result order**

Add a test like:

```rust
#[test]
fn test_partial_contract_honors_output_order() {
    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        multiply_pairs: vec![],
        output_order: Some(vec![idx_b.clone(), idx_a.clone()]),
    };
    let result = partial_contract(&tn_a, &tn_b, &spec, &center, ContractionOptions::default()).unwrap();
    let (indices, _) = result.all_site_indices().unwrap();
    assert_eq!(indices[0].id(), idx_b.id());
    assert_eq!(indices[1].id(), idx_a.id());
}
```

**Step 3: Run tests to verify they fail**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-treetn
```

Expected:

- aligned add missing
- partial contraction output ordering missing

**Step 4: Implement the minimal helper APIs**

Implementation requirements:

- keep APIs site/index based, not variable-name based
- use existing `share_equivalent_site_index_network`, `replaceind(s)`, and `swap_site_indices_by_index` as building blocks where possible
- do not embed BubbleTeaCI result semantics into `PartialContractionSpec`
- preserve current `partial_contract` behavior when no output order is requested

**Step 5: Re-run tests**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-treetn
```

Expected:

- all TreeTN tests pass

**Step 6: Commit**

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
git add crates/tensor4all-treetn/src/treetn/addition.rs crates/tensor4all-treetn/src/treetn/addition/tests/mod.rs crates/tensor4all-treetn/src/treetn/partial_contraction.rs crates/tensor4all-treetn/src/treetn/partial_contraction/tests/mod.rs crates/tensor4all-treetn/src/treetn/mod.rs
git commit -m "feat: add site-space alignment and ordered partial contraction"
```

## Task 7: Refresh Docs, API Dumps, And Downstream Verification Gate

**Files:**
- Modify public doc comments in every touched public API
- Refresh: `docs/api/`
- Modify for local downstream validation only: `../BubbleTeaCI-rs/Cargo.toml`

**Step 1: Add/finish doc examples for all new public APIs**

Every new public type and function from Tasks 2-6 must have:

- a doc comment
- a short example with assertions

Do not leave public APIs undocumented.

**Step 2: Rebuild API docs**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo run -p api-dump --release -- . -o docs/api
cargo doc --workspace --no-deps
```

Expected:

- API dump succeeds
- rustdoc succeeds

**Step 3: Run the full backend verification set**

Run:

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
cargo test --release

cd /home/shinaoka/tensor4all/tensor4all-rs
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
```

Expected:

- formatting clean
- clippy clean
- full backend tests pass

**Step 4: Point `BubbleTeaCI-rs` at local backend sources**

Edit `../BubbleTeaCI-rs/Cargo.toml` so that, during downstream migration, it uses:

```toml
quanticsgrids = { path = "../quanticsgrids-rs" }
tensor4all-core = { path = "../tensor4all-rs/crates/tensor4all-core" }
tensor4all-quanticstransform = { path = "../tensor4all-rs/crates/tensor4all-quanticstransform" }
tensor4all-treetn = { path = "../tensor4all-rs/crates/tensor4all-treetn" }
```

This is a local verification step, not a PR step.

**Step 5: Execute the downstream BubbleTeaCI migration plan**

Use the existing downstream plan:

- `../BubbleTeaCI-rs/docs/plans/2026-04-08-bubbletea-rs-redesign-plan.md`

Success condition before opening any backend PR:

- `BubbleTeaCI-rs` no longer needs dense transform fallbacks
- `BubbleTeaCI-rs` no longer needs manual operator `true_index` rewrites
- `BubbleTeaCI-rs` can use backend-native grid/layout APIs instead of handwritten shims
- if downstream hits a remaining backend gap, stop and add it to this plan rather than patching around it downstream

**Step 6: Only after downstream success, prepare the publishable state**

At this point, and only at this point:

- replace temporary local path dependency in `tensor4all-rs/Cargo.toml` with the final `quanticsgrids-rs` revision
- keep commits separated by repo
- then prepare PRs in the correct order

Do **not** do this earlier.

**Step 7: Commit the final verification artifacts**

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
git add docs/api
git commit -m "docs: refresh api after BubbleTeaCI migration backend work"
```

## Task 8: Add Exact Index-Unfusing Primitives For Component-Axis Materialization

**Status:** complete

**Files:**
- Modify: `crates/tensor4all-core/src/...` for tensor-level exact unfuse support
- Modify: `crates/tensor4all-treetn/src/treetn/...` for TreeTN-level site-index wrapper
- Test: tensor-level unit tests
- Test: TreeTN-level unit tests
- Downstream verification: `../BubbleTeaCI-rs/tests/contract.rs`

**Goal:**
- add a general exact operation that replaces one index with multiple indices
- require the caller to specify linearization order explicitly
- expose a TreeTN wrapper for site indices so downstream code can materialize
  multiple physical component legs without dense or application-specific logic

**API direction:**

At the tensor level, add something equivalent to:

```rust
pub enum LinearizationOrder {
    ColumnMajor,
    RowMajor,
}

fn unfuse_index(
    &self,
    old_index: &Self::Index,
    new_indices: &[Self::Index],
    order: LinearizationOrder,
) -> Result<Self>;
```

At the TreeTN level, add a site-index wrapper such as:

```rust
fn replace_site_index_with_indices(
    &self,
    old_index: &T::Index,
    new_indices: &[T::Index],
    order: LinearizationOrder,
) -> Result<Self>;
```

The exact names are flexible, but the semantics are not:

- `dim(old_index) == product(dim(new_indices))`
- exact reshape, no approximation
- no hidden default linearization order
- works for general tensor/site-index manipulation, not just batched QTCI

**Why this belongs upstream:**
- the operation is a general tensor-network primitive
- the need appears in BubbleTeaCI migration, but the primitive is reusable by
  any upper layer that needs to unfuse packed component/state indices
- solving this downstream would require ad hoc dense unpack/re-factorization,
  which violates the workspace boundary

**Step 1: Write failing tensor-level tests**

Add tests that check:

- unfusing a length-4 index into two length-2 indices with `ColumnMajor`
  produces the expected reshaped tensor
- `RowMajor` gives a different but deterministic mapping
- dimension mismatch is rejected

**Step 2: Write failing TreeTN-level tests**

Add tests that check:

- a 1-node TreeTN with one site index of dim 4 can be transformed into a 1-node
  TreeTN with two site indices of dims 2 and 2
- `all_site_indices()` returns the two new indices
- `evaluate_at(...)` on the unfused network matches the original fused network
  under the specified linearization order

**Step 3: Run the narrow failing tests**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-core
cargo nextest run --release -p tensor4all-treetn
```

Expected:

- missing-method or behavior failures for the new unfuse operation

**Step 4: Implement the minimal exact primitive**

Implementation requirements:

- do not implement this as dense materialize -> factorize -> truncate
- do not rely on implicit workspace-wide column-major assumptions
- keep the tensor-level implementation exact and reusable
- the TreeTN wrapper should update site-index metadata consistently

**Step 5: Re-run tests**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo nextest run --release -p tensor4all-core
cargo nextest run --release -p tensor4all-treetn
```

Expected:

- all touched backend tests pass

**Step 6: Re-run downstream verification at the contraction boundary**

Run:

```bash
cd /home/shinaoka/tensor4all/BubbleTeaCI-rs
cargo test --test contract
```

Expected:

- downstream can begin materializing component axes without ad hoc repacking
- if further contract-order gaps remain, they should be semantic/downstream, not
  component-site-layout blockers

**Step 7: Commit**

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
git add crates/tensor4all-core crates/tensor4all-treetn docs/plans/2026-04-08-bubbletea-upstream-gap-implementation-plan.md
git commit -m "feat: add exact index unfusing primitives"
```

Plan complete and saved to `docs/plans/2026-04-08-bubbletea-upstream-gap-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
