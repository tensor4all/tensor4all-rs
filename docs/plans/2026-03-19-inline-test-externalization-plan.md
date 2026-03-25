# Inline Test Externalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move Rust test coverage out of `src/` inline test blocks and into `tests/`-style modules, following the `tenferro-rs` split between public integration tests and crate-local test modules.

**Architecture:** Keep behavior-oriented regression tests in `tests/` and move implementation-focused tests into `src/<module>/tests/` submodules where they can still access crate-private internals. Use small `pub(crate)` test seams only when a test cannot be expressed through the public API. The end state should make test ownership obvious: public contract tests live outside `src/`, while module-specific unit tests stay close to the code they exercise.

**Tech Stack:** Rust 2021, Cargo workspace, `cargo fmt --all`, `cargo clippy --workspace`, `cargo nextest run --release`.

---

### Task 1: Establish the crate-by-crate test layout contract

**Files:**
- Modify: `README.md`
- Create: `crates/tensor4all-core/tests/mod.rs`
- Create: `crates/tensor4all-simplett/tests/mod.rs`
- Create: `crates/tensor4all-itensorlike/tests/mod.rs`
- Create: `crates/tensor4all-treetn/tests/mod.rs`
- Create: `crates/tensor4all-partitionedtt/tests/mod.rs`
- Create: `crates/tensor4all-tensorci/tests/mod.rs`
- Create: `crates/tensor4all-quanticstci/tests/mod.rs`
- Create: `crates/tensor4all-quanticstransform/tests/mod.rs`
- Create: `crates/tensor4all-tensorbackend/tests/mod.rs`
- Create: `crates/matrixci/tests/mod.rs`
- Create: `crates/quanticsgrids/tests/mod.rs`
- Create: `crates/tensor4all-hdf5/tests/mod.rs`
- Create: `crates/tensor4all-capi/tests/mod.rs`

**Step 1: Write the failing layout checks**

Add tiny organization tests that assert the intended split exists for crates that currently have only inline tests. These tests should verify that the crate exposes a `tests/` entry point and that the old inline-test-only layout is not still the only home for coverage.

**Step 2: Verify the baseline**

Run:
```bash
cargo nextest run --release -p tensor4all-core
```
Expected: existing coverage still passes before moving any logic.

**Step 3: Write the minimal layout scaffolding**

Add empty `mod.rs` test entry points where needed, plus shared helper modules such as `tests/support/mod.rs` for repeated constructors and dense fixtures.

**Step 4: Verify the scaffolding**

Run:
```bash
cargo nextest run --release -p tensor4all-core
```
Expected: PASS, with no behavior change yet.

**Step 5: Commit**

```bash
git add README.md crates/*/tests/mod.rs docs/plans/2026-03-19-inline-test-externalization-plan.md
git commit -m "docs: plan inline test externalization"
```

### Task 2: Move foundational unit tests first

**Files:**
- Modify: `crates/tensor4all-core/src/block_tensor.rs`
- Modify: `crates/tensor4all-core/src/defaults/contract.rs`
- Modify: `crates/tensor4all-core/src/defaults/direct_sum.rs`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Modify: `crates/tensor4all-core/src/defaults/index.rs`
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Modify: `crates/tensor4all-core/src/global_default.rs`
- Modify: `crates/tensor4all-core/src/krylov.rs`
- Modify: `crates/tensor4all-core/src/smallstring.rs`
- Modify: `crates/tensor4all-core/src/tensor_like.rs`
- Modify: `crates/tensor4all-core/src/truncation.rs`
- Modify: `crates/tensor4all-core/tests/*.rs`
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Modify: `crates/tensor4all-tensorbackend/src/backend.rs`
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`

**Step 1: Write the failing migrated tests**

Move one representative test file per crate family into `tests/` or `src/<module>/tests/`, keeping the assertions identical. Use public APIs where possible, and expose only the minimum extra test-only helpers when the assertion currently depends on private state.

**Step 2: Run the focused crate tests**

Run:
```bash
cargo nextest run --release -p tensor4all-core
cargo nextest run --release -p tensor4all-tensorbackend
```
Expected: PASS after the first migrated batch.

**Step 3: Finish the rest of the foundational moves**

Continue moving the remaining inline tests in these crates until `src/` only contains production code and any necessary `#[cfg(test)] pub(crate)` seams.

**Step 4: Remove the old inline test blocks**

Delete the now-empty `#[cfg(test)] mod tests;` declarations and inline test modules from the moved source files.

**Step 5: Commit**

```bash
git add crates/tensor4all-core crates/tensor4all-tensorbackend
git commit -m "test: externalize core and backend tests"
```

### Task 3: Move simplett and itensorlike test coverage

**Files:**
- Modify: `crates/tensor4all-simplett/src/arithmetic.rs`
- Modify: `crates/tensor4all-simplett/src/cache.rs`
- Modify: `crates/tensor4all-simplett/src/canonical.rs`
- Modify: `crates/tensor4all-simplett/src/compression.rs`
- Modify: `crates/tensor4all-simplett/src/contraction.rs`
- Modify: `crates/tensor4all-simplett/src/tensortrain.rs`
- Modify: `crates/tensor4all-simplett/src/types.rs`
- Modify: `crates/tensor4all-simplett/src/vidal.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/*.rs`
- Modify: `crates/tensor4all-simplett/tests/*.rs`
- Modify: `crates/tensor4all-itensorlike/src/contract.rs`
- Modify: `crates/tensor4all-itensorlike/src/options.rs`
- Modify: `crates/tensor4all-itensorlike/src/tensortrain.rs`
- Modify: `crates/tensor4all-itensorlike/tests/*.rs`

**Step 1: Move the highest-value regression tests first**

Start with the tests that caught recent bugs or verify public behavior, then move the purely internal edge-case tests into module-local `tests/` files.

**Step 2: Add support helpers**

Create `tests/support/mod.rs` or `tests/common.rs` only if a crate needs repeated tensor constructors, dense fixtures, or shared assertions.

**Step 3: Remove inline test blocks**

Delete the `#[cfg(test)]` blocks from the source files once the equivalent external tests are green.

**Step 4: Run crate verification**

Run:
```bash
cargo nextest run --release -p tensor4all-simplett
cargo nextest run --release -p tensor4all-itensorlike
```
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-simplett crates/tensor4all-itensorlike
git commit -m "test: externalize simplett and itensorlike tests"
```

### Task 4: Move treetn and partitionedtt coverage

**Files:**
- Modify: `crates/tensor4all-treetn/src/algorithm.rs`
- Modify: `crates/tensor4all-treetn/src/link_index_network.rs`
- Modify: `crates/tensor4all-treetn/src/named_graph.rs`
- Modify: `crates/tensor4all-treetn/src/node_name_network.rs`
- Modify: `crates/tensor4all-treetn/src/options.rs`
- Modify: `crates/tensor4all-treetn/src/random.rs`
- Modify: `crates/tensor4all-treetn/src/site_index_network.rs`
- Modify: `crates/tensor4all-treetn/src/operator/*.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/*.rs`
- Modify: `crates/tensor4all-treetn/src/linsolve/common/*.rs`
- Modify: `crates/tensor4all-treetn/src/linsolve/square/*.rs`
- Modify: `crates/tensor4all-treetn/tests/*.rs`
- Modify: `crates/tensor4all-partitionedtt/src/contract.rs`
- Modify: `crates/tensor4all-partitionedtt/src/partitioned_tt.rs`
- Modify: `crates/tensor4all-partitionedtt/src/patching.rs`
- Modify: `crates/tensor4all-partitionedtt/src/projector.rs`
- Modify: `crates/tensor4all-partitionedtt/src/subdomain_tt.rs`
- Modify: `crates/tensor4all-partitionedtt/tests/*.rs`

**Step 1: Preserve public regressions in `tests/`**

Keep existing integration tests as the source of truth for API behavior, especially bug regressions and usage examples.

**Step 2: Move internal module tests next**

Split the large source-side test blocks into `src/<module>/tests/mod.rs` files so the implementation-focused assertions stay close to the code they cover.

**Step 3: Add crate-private seams only where required**

If a migrated test can only pass through an internal helper, expose the smallest possible `pub(crate)` helper and keep it test-only.

**Step 4: Run the treetn and partitionedtt suites**

Run:
```bash
cargo nextest run --release -p tensor4all-treetn
cargo nextest run --release -p tensor4all-partitionedtt
```
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn crates/tensor4all-partitionedtt
git commit -m "test: externalize treetn and partitionedtt tests"
```

### Task 5: Move tensorci, quantics, hdf5, capi, and utility crate coverage

**Files:**
- Modify: `crates/tensor4all-tensorci/src/cached_function.rs`
- Modify: `crates/tensor4all-tensorci/src/indexset.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`
- Modify: `crates/tensor4all-quanticstci/src/options.rs`
- Modify: `crates/tensor4all-quanticstci/src/quantics_tci.rs`
- Modify: `crates/tensor4all-quanticstransform/src/*.rs`
- Modify: `crates/quanticsgrids/src/*.rs`
- Modify: `crates/matrixci/src/*.rs`
- Modify: `crates/tensor4all-hdf5/src/index.rs`
- Modify: `crates/tensor4all-hdf5/src/schema.rs`
- Modify: `crates/tensor4all-capi/src/*.rs`
- Modify: `crates/tensor4all-capi/tests/*.rs`

**Step 1: Move the remaining inline tests into local test modules**

For the smaller utility crates, create `src/<module>/tests/mod.rs` files only where the tests are genuinely implementation-focused. Use `tests/` for the cross-crate or user-facing regressions.

**Step 2: Align the C API and HDF5 checks with the new layout**

Keep boundary behavior tests out of `src/` and put them in `tests/` so the public contract stays easy to audit.

**Step 3: Remove the remaining inline test declarations**

Delete the last `#[cfg(test)]` blocks from the source files once equivalent coverage exists outside the production modules.

**Step 4: Run the remaining crate suites**

Run:
```bash
cargo nextest run --release -p tensor4all-tensorci
cargo nextest run --release -p tensor4all-quanticstci
cargo nextest run --release -p tensor4all-quanticstransform
cargo nextest run --release -p quanticsgrids
cargo nextest run --release -p matrixci
cargo nextest run --release -p tensor4all-hdf5
cargo nextest run --release -p tensor4all-capi
```
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-tensorci crates/tensor4all-quanticstci crates/tensor4all-quanticstransform crates/quanticsgrids crates/matrixci crates/tensor4all-hdf5 crates/tensor4all-capi
git commit -m "test: externalize remaining crate tests"
```

### Task 6: Remove leftover inline-test declarations and verify the workspace

**Files:**
- Modify: all source files that still contain `#[cfg(test)]` after Tasks 2-5
- Modify: `README.md` if the test layout needs a short note for contributors

**Step 1: Audit for leftovers**

Run:
```bash
rg -n "#\\[cfg\\(test\\)\\]|mod tests;|pub\\(crate\\) mod tests;" crates
```
Expected: only intentional `src/<module>/tests` module declarations and test-only seam exports remain.

**Step 2: Run formatting**

Run:
```bash
cargo fmt --all
```
Expected: PASS.

**Step 3: Run lint**

Run:
```bash
cargo clippy --workspace --all-targets -- -D warnings
```
Expected: PASS.

**Step 4: Run the full test suite**

Run:
```bash
cargo nextest run --release --workspace
```
Expected: PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "test: externalize inline tests"
```
