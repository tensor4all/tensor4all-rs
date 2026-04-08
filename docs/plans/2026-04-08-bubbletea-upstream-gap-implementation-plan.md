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
- Reworking `quanticscrossinterpolate_batched` component layout unless downstream validation proves the current last-site contract is insufficient

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

## Task 2: Add Structural Equality And Layout Query APIs To `quanticsgrids-rs`

**Files:**
- Modify: `../quanticsgrids-rs/src/lib.rs`
- Modify: `../quanticsgrids-rs/src/inherent_discrete_grid.rs`
- Modify: `../quanticsgrids-rs/src/discretized_grid.rs`
- Test: `../quanticsgrids-rs/src/inherent_discrete_grid/tests/mod.rs`
- Test: `../quanticsgrids-rs/src/discretized_grid/tests/mod.rs`

**APIs to add in this task:**

- `impl PartialEq for InherentDiscreteGrid`
- `impl PartialEq for DiscretizedGrid`
- explicit layout query API on both grid types:
  - `unfolding_scheme(&self) -> Option<UnfoldingScheme>` or a stronger replacement such as `layout_kind()`
  - `variable_id(&self, name: &str) -> Result<usize>`
  - `variable_ids(&self, names: &[&str]) -> Result<Vec<usize>>`
  - `variable_sites(&self, names: &[&str], do_sort: bool) -> Result<Vec<usize>>`
  - `same_layout(&self, other: &Self) -> bool`

The exact enum name for the layout query is flexible, but it must let downstream code distinguish:

- grouped
- interleaved
- fused
- layout not representable as one of the above, if custom index tables require that distinction

**Step 1: Write failing tests for equality and layout queries**

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

#[test]
fn test_discretized_grid_variable_sites_grouped() {
    let grid = DiscretizedGrid::builder(&[2, 2])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    assert_eq!(grid.variable_sites(&["x"], true).unwrap(), vec![0, 1]);
    assert_eq!(grid.variable_sites(&["y"], true).unwrap(), vec![2, 3]);
}
```

Also add one test each for:

- fused layout
- interleaved layout
- custom index table if layout is no longer one of the standard schemes
- `same_layout()` ignoring bounds and variable renaming but respecting site structure

**Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
cargo test --release discretized_grid
```

Expected:

- compile failure or missing-method failures for the new APIs

**Step 3: Implement the minimal public API**

Implementation requirements:

- do not duplicate layout-decoding logic between inherent/discretized grids; share one internal helper
- `PartialEq` must be structural, not approximate
- `same_layout()` must compare site structure and base/unfolding semantics, but not bounds or variable names
- `variable_sites()` must be expressed in terms of public grid structure, not by re-deriving from external callers

If layout cannot always be represented by `UnfoldingScheme`, add a dedicated public layout enum instead of forcing a lossy answer.

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
git commit -m "feat: add quantics grid equality and layout query APIs"
```

## Task 3: Add Grid Metadata Algebra To `quanticsgrids-rs`

**Files:**
- Modify: `../quanticsgrids-rs/src/discretized_grid.rs`
- Modify: `../quanticsgrids-rs/src/inherent_discrete_grid.rs`
- Modify: `../quanticsgrids-rs/src/lib.rs`
- Test: `../quanticsgrids-rs/src/discretized_grid/tests/mod.rs`
- Test: `../quanticsgrids-rs/src/inherent_discrete_grid/tests/mod.rs`

**APIs to add in this task:**

- `rename_variables(...)`
- `project_grid(...)`
- `reduce_grid(...)`
- `refine_grid(...)`

Keep the names close to Julia/BubbleTeaCI, but make them methods on the grid types for Rust-side consistency. The public shape should look like:

```rust
impl DiscretizedGrid {
    pub fn rename_variables(&self, new_names: &[&str]) -> Result<Self>;
    pub fn project_grid(&self, variables_to_remove: &[&str]) -> Result<Self>;
    pub fn reduce_grid(&self, vars: &[usize], n_legs: &[usize]) -> Result<Self>;
    pub fn refine_grid(&self, template: &Self, vars: &[usize], n_new_legs: &[usize]) -> Result<Self>;
}
```

Mirror the same algebra for `InherentDiscreteGrid` where it is still general.

**Step 1: Write failing tests**

Add tests that cover:

- variable rename preserves layout
- projection removes named variables
- projection down to zero dimensions is supported if the type model permits it
- reduction removes low-significance bits correctly
- refinement reconstructs an intermediate grid from a higher-resolution template

Start with tests like:

```rust
#[test]
fn test_project_grid_removes_named_variables() {
    let grid = DiscretizedGrid::builder(&[2, 3, 1])
        .with_variable_names(&["x", "y", "z"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();

    let projected = grid.project_grid(&["y"]).unwrap();
    assert_eq!(projected.variable_names(), &["x".to_string(), "z".to_string()]);
}
```

Add at least one test ported from BubbleTeaCI grid behavior:

- reduction that drops a variable completely
- refinement with preserved origin/bounds

**Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
cargo test --release project_grid
```

Expected:

- missing-method or behavior failures

**Step 3: Implement the minimal API**

Implementation requirements:

- use the new layout query API from Task 2 rather than downstream-style reverse engineering
- preserve site layout unless the operation logically changes it
- keep methods purely metadata-based; do not introduce BubbleTeaCI result-order semantics
- preserve or explicitly define behavior for zero-dimensional grids

**Step 4: Add doc comments with assertions**

Public methods must document:

- what is preserved
- what changes
- 1-indexed vs 0-indexed variable numbering, if applicable

**Step 5: Run the full crate test suite**

Run:

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
cargo test --release
```

Expected:

- all tests pass

**Step 6: Commit**

```bash
cd /home/shinaoka/tensor4all/quanticsgrids-rs
git add src/lib.rs src/inherent_discrete_grid.rs src/discretized_grid.rs src/inherent_discrete_grid/tests/mod.rs src/discretized_grid/tests/mod.rs
git commit -m "feat: add quantics grid metadata algebra"
```

## Task 4: Rebase Grid-Aware Transforms On The Grid Layer And Fix API Naming

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/grid_aware.rs`
- Modify: `crates/tensor4all-quanticstransform/src/lib.rs`
- Test: `crates/tensor4all-quanticstransform/src/grid_aware/tests/mod.rs`

**APIs to change in this task:**

- stop manually detecting layout from `index_table`
- consume the new `quanticsgrids` layout/query APIs
- rename:
  - `shift_operator_on_grid_by_tag`
  - to `shift_operator_on_grid_by_variable_name`

If there is any second API that refers to variable names as “tags”, rename it in the same pass.

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

**Step 3: Implement the rename and remove local layout inference**

Implementation requirements:

- `grid_aware.rs` should depend on `quanticsgrids` public APIs, not re-parse the grid structure itself
- bad naming using `tag` for variable names must disappear from public API
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

Plan complete and saved to `docs/plans/2026-04-08-bubbletea-upstream-gap-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
