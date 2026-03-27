# Absorb matrixluci Into tensor4all-tcicore Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the standalone `matrixluci` crate into `tensor4all-tcicore` as an internal `matrixluci` module, update all downstream crates and benches to use the new paths, remove the old crate, and verify the workspace still passes formatting, lint, and release-mode tests.

**Architecture:** `tensor4all-tcicore` will own two LUCI layers: the existing row-major `MatrixLUCI` wrapper and the absorbed low-level `matrixluci` substrate. To avoid a module-name collision, the current wrapper module will be renamed internally while preserving the public `MatrixLUCI` re-export at crate root. The copied low-level source tree will live under `src/matrixluci/` with the same internal module layout as the old crate, but its internal `crate::` references will be rewritten to point at `crate::matrixluci`.

**Tech Stack:** Rust workspace, Cargo, criterion benches, `faer`, `cargo fmt`, `cargo clippy`, `cargo nextest --release`

---

### Task 1: Snapshot Current API And Dependency Surface

**Files:**
- Modify: `docs/plans/2026-03-27-absorb-matrixluci.md`
- Inspect: `docs/api/matrixluci.md`
- Inspect: `docs/api/tensor4all_tcicore.md`
- Inspect: `crates/matrixluci/src/lib.rs`
- Inspect: `crates/tensor4all-tcicore/src/lib.rs`
- Inspect: `crates/tensor4all-tcicore/src/matrixluci.rs`

**Step 1: Confirm public exports from the standalone crate**

Run: `sed -n '1,240p' crates/matrixluci/src/lib.rs`
Expected: module declarations plus re-exports for `LazyBlockRookKernel`, `DenseFaerLuKernel`, `MatrixLuciError`, `Result`, `CrossFactors`, `PivotKernel`, `Scalar`, `CandidateMatrixSource`, `DenseMatrixSource`, `LazyMatrixSource`, `DenseOwnedMatrix`, `PivotKernelOptions`, and `PivotSelectionCore`.

**Step 2: Confirm module-name collision in tcicore**

Run: `sed -n '1,260p' crates/tensor4all-tcicore/src/lib.rs && sed -n '1,260p' crates/tensor4all-tcicore/src/matrixluci.rs`
Expected: `tensor4all-tcicore` already exposes `pub mod matrixluci;` for the higher-level wrapper.

**Step 3: Record downstream dependency and import sites**

Run: `rg -n "matrixluci::|matrixluci" crates --glob '*.rs' --glob '*.toml' Cargo.toml`
Expected: hits in downstream crates, benches, and workspace membership.

**Step 4: Commit the exploration-only plan artifact if desired**

Run: `git add docs/plans/2026-03-27-absorb-matrixluci.md`
Expected: plan file staged without code changes.

### Task 2: Move The High-Level MatrixLUCI Wrapper Aside

**Files:**
- Create: `crates/tensor4all-tcicore/src/matrix_luci.rs`
- Delete: `crates/tensor4all-tcicore/src/matrixluci.rs`
- Modify: `crates/tensor4all-tcicore/src/lib.rs`

**Step 1: Rename the existing wrapper module internally**

Implementation:
- Copy the contents of `crates/tensor4all-tcicore/src/matrixluci.rs` to `crates/tensor4all-tcicore/src/matrix_luci.rs`.
- Delete `crates/tensor4all-tcicore/src/matrixluci.rs`.
- Update `lib.rs` to declare `pub mod matrixluci;` for the absorbed low-level module and `mod matrix_luci;` for the row-major wrapper.
- Re-export `MatrixLUCI` from `matrix_luci`.

**Step 2: Update the wrapper imports to use the new in-crate low-level module**

Implementation:
- Replace `::matrixluci::...` imports and trait bounds with `crate::matrixluci::...`.
- Keep the public behavior of `MatrixLUCI` unchanged.

**Step 3: Move existing MatrixLUCI tests if needed**

Implementation:
- Keep wrapper tests compiling under the renamed module, either inline or under `src/matrix_luci/tests/mod.rs`.

**Step 4: Commit the internal rename**

Run: `git add crates/tensor4all-tcicore/src/lib.rs crates/tensor4all-tcicore/src/matrix_luci.rs crates/tensor4all-tcicore/src/matrixluci.rs crates/tensor4all-tcicore/src/matrixluci/tests/mod.rs && git commit -m "refactor(tcicore): rename high-level MatrixLUCI module"`
Expected: one commit containing only the wrapper-module relocation and path updates.

### Task 3: Absorb The Low-Level matrixluci Source Tree

**Files:**
- Create: `crates/tensor4all-tcicore/src/matrixluci/mod.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/block_rook.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/block_rook/tests.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/dense.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/dense/tests.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/error.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/factors.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/factors/tests.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/kernel.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/kernel/tests.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/scalar.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/scalar/tests.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/source.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/source/tests.rs`
- Create: `crates/tensor4all-tcicore/src/matrixluci/types.rs`

**Step 1: Copy all source files preserving the old layout**

Implementation:
- Recreate the old `matrixluci` source tree under `crates/tensor4all-tcicore/src/matrixluci/`.
- Preserve module names and test file locations.

**Step 2: Rewrite internal module paths for submodule context**

Implementation:
- Replace `crate::...` references inside the copied files with `crate::matrixluci::...` or `super::...` as appropriate.
- Keep public item names unchanged within the new `tensor4all_tcicore::matrixluci` namespace.

**Step 3: Preserve the old public exports under the new namespace**

Implementation:
- In `src/matrixluci/mod.rs`, mirror the old crate-root `pub use` surface from `crates/matrixluci/src/lib.rs`.
- Do not re-export `matrixluci::Result` or `matrixluci::Scalar` at `tensor4all-tcicore` crate root because those names already exist there; instead keep them under `tensor4all_tcicore::matrixluci`.

**Step 4: Commit the absorbed source tree**

Run: `git add crates/tensor4all-tcicore/src/matrixluci && git commit -m "feat(tcicore): absorb low-level matrixluci module"`
Expected: one commit containing only the copied low-level source subtree and the new module re-exports.

### Task 4: Update tcicore Cargo Metadata And Benches

**Files:**
- Modify: `crates/tensor4all-tcicore/Cargo.toml`
- Create: `crates/tensor4all-tcicore/benches/dense_vs_faer.rs`
- Create: `crates/tensor4all-tcicore/benches/end_to_end_chain_tci.rs`
- Create: `crates/tensor4all-tcicore/benches/lazy_block_rook.rs`
- Modify: existing bench files in `crates/tensor4all-tcicore/benches/` if they import `matrixluci`

**Step 1: Remove the old path dependency and add direct requirements**

Implementation:
- Delete `matrixluci = { path = "../matrixluci" }` from `tensor4all-tcicore/Cargo.toml`.
- Add non-dev `faer = "^0.23"` if the absorbed module needs it in normal library code.
- Keep or adjust dev-dependencies for benches as needed.

**Step 2: Copy criterion benches from the old crate**

Implementation:
- Move all bench source files from `crates/matrixluci/benches/` into `crates/tensor4all-tcicore/benches/`.
- Update imports to `tensor4all_tcicore::matrixluci::...` or root re-exports where appropriate.

**Step 3: Commit tcicore manifest and bench changes**

Run: `git add crates/tensor4all-tcicore/Cargo.toml crates/tensor4all-tcicore/benches && git commit -m "build(tcicore): inline matrixluci dependencies and benches"`
Expected: one commit containing only tcicore manifest and bench updates.

### Task 5: Update Downstream Crates To The New Namespace

**Files:**
- Modify: `crates/tensor4all-core/Cargo.toml`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Modify: `crates/tensor4all-simplett/Cargo.toml`
- Modify: `crates/tensor4all-simplett/src/compression.rs`
- Modify: `crates/tensor4all-simplett/src/compression/tests/mod.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/factorize.rs`
- Modify: `crates/tensor4all-tensorci/Cargo.toml`
- Modify: `crates/tensor4all-tensorci/src/error.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`
- Modify: `crates/tensor4all-quanticstci/Cargo.toml`
- Modify: `crates/tensor4all-quanticstci/src/quantics_tci.rs`
- Modify: `crates/tensor4all-treetci/Cargo.toml`
- Modify: `crates/tensor4all-treetci/src/api.rs`
- Modify: `crates/tensor4all-treetci/src/materialize.rs`
- Modify: `crates/tensor4all-treetci/src/optimize.rs`
- Modify: `crates/tensor4all-treetci/src/update.rs`
- Modify: `crates/tensor4all-treetci/src/update/tests.rs`

**Step 1: Replace dependency edges**

Implementation:
- Remove `matrixluci` from affected `Cargo.toml` files.
- Ensure each affected crate already depends on `tensor4all-tcicore`; if not, add it.

**Step 2: Rewrite all code imports and trait bounds**

Implementation:
- Replace `matrixluci::...` with `tensor4all_tcicore::matrixluci::...` or `tensor4all_tcicore::...` when the symbol is re-exported at crate root.
- Update error conversions and generic bounds consistently.

**Step 3: Search for stragglers before deleting the old crate**

Run: `rg -n "matrixluci::|matrixluci" crates --glob '*.rs' --glob '*.toml'`
Expected: only hits inside `crates/tensor4all-tcicore/` and the soon-to-be-deleted old crate.

**Step 4: Commit downstream path updates**

Run: `git add crates/tensor4all-core crates/tensor4all-simplett crates/tensor4all-tensorci crates/tensor4all-quanticstci crates/tensor4all-treetci && git commit -m "refactor: point downstream crates at tcicore matrixluci"`
Expected: one commit containing all downstream import and manifest rewrites.

### Task 6: Remove The Old Crate From The Workspace

**Files:**
- Modify: `Cargo.toml`
- Delete: `crates/matrixluci/`
- Modify: `README.md` if the workspace structure or crate documentation table still mentions the standalone crate

**Step 1: Drop workspace membership**

Implementation:
- Remove `crates/matrixluci` from the root workspace members list.

**Step 2: Delete the old crate directory**

Implementation:
- Remove `crates/matrixluci/` after all imports and benches no longer rely on it.

**Step 3: Keep README accurate**

Implementation:
- Update project-structure and crate-documentation sections if they still list `matrixluci` as a standalone crate.

**Step 4: Commit workspace cleanup**

Run: `git add Cargo.toml README.md crates/matrixluci && git commit -m "chore: remove standalone matrixluci crate"`
Expected: one commit removing the old crate and workspace references.

### Task 7: Verify Formatting, Lints, Tests, And Final Search

**Files:**
- Modify only if fixes are required by tooling

**Step 1: Format**

Run: `cargo fmt --all`
Expected: no diff after formatting or only mechanical formatting diff.

**Step 2: Lint the workspace**

Run: `cargo clippy --workspace`
Expected: success without introducing new warnings that block CI.

**Step 3: Run the full release-mode test suite**

Run: `cargo nextest run --release --workspace`
Expected: full pass in release mode.

**Step 4: Confirm no downstream references remain**

Run: `grep -r 'matrixluci' crates/ --include='*.rs' --include='*.toml' | grep -v target/ | grep -v tensor4all-tcicore/`
Expected: no output.

**Step 5: Commit any verification-driven fixes**

Run: `git add -A && git commit -m "fix: address absorb-matrixluci verification fallout"`
Expected: only if tooling required follow-up fixes after the structural commits.
