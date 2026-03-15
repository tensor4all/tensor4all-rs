# Tenferro Core `mdarray` Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `mdarray` from `tensor4all-core` and `tensor4all-tensorbackend` execution paths by routing `TensorDynLen`, einsum, and linalg through `tenferro::Tensor`.

**Architecture:** Make `tenferro::Tensor` the only canonical execution payload in the core/backend layer. Keep `Storage` only as a materialization boundary, replace `AnyScalar` with a rank-0 wrapper over `tenferro::Tensor`, delete storage-level einsum/linalg adapters, and rework `TensorDynLen` to delegate all numeric execution to `tenferro`.

**Tech Stack:** Rust, `tensor4all-core`, `tensor4all-tensorbackend`, `tenferro`, `tenferro-tensor`, `tenferro::snapshot`, `cargo fmt`, `cargo clippy`, `cargo nextest --release`

---

### Task 1: Update the upstream dependency baseline and surface scan

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/tensor4all-tensorbackend/Cargo.toml`
- Test: workspace compile and API dump output

**Step 1: Update workspace `tenferro-*` dependencies to the reviewed upstream revision**

Point all `tenferro-*` git dependencies at the reviewed `origin/main` revision and add a direct
workspace dependency on the `tenferro` crate.

**Step 2: Remove assumptions tied to `tenferro-dyadtensor` as a public frontend**

Update dependency comments and crate-level descriptions so they refer to `tenferro::Tensor`
instead of `DynAdTensor`/`DynAdScalar`.

**Step 3: Regenerate API docs to expose compile failures early**

Run:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Expected: either successful regeneration or compile errors pointing at old frontend assumptions.

**Step 4: Run a narrow compile check to inventory breakage**

Run:

```bash
cargo check -p tensor4all-tensorbackend
cargo check -p tensor4all-core
```

Expected: FAIL. The failures should identify every remaining direct dependency on old native types
or `mdarray` execution helpers.

**Step 5: Commit**

```bash
git add Cargo.toml crates/tensor4all-tensorbackend/Cargo.toml docs/api
git commit -m "build: update core backend to current tenferro frontend"
```

### Task 2: Replace `AnyScalar` and the backend facade with `tenferro::Tensor`-based wrappers

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Test: new or updated scalar/backend tests in `tensor4all-tensorbackend`

**Step 1: Write failing tests for rank-0 scalar semantics**

Add or update tests so they assert:

- `AnyScalar` is a wrapper type, not a `DynAdScalar` alias
- `AnyScalar::from_real` / `from_complex` produce rank-0 values
- `AnyScalar::real`, `imag`, `conj`, `is_real`, `is_complex` work without matching on backend enums

**Step 2: Replace `AnyScalar = DynAdScalar` with a real `Scalar` newtype**

Implement a public `Scalar` wrapper over rank-0 `tenferro::Tensor`, then keep:

```rust
pub type AnyScalar = Scalar;
```

Internal validation must reject non-rank-0 tensors.

**Step 3: Rewrite backend helpers to use `tenferro::Tensor`**

Replace old `DynAdTensor`/`DynAdScalar`-centric helpers with backend functions that accept and
return:

- `tenferro::Tensor`
- `Scalar`
- `snapshot::DynTensor` only at materialization boundaries

**Step 4: Remove public re-exports of low-level native types and `mdarray`**

Delete re-exports of:

- `DynAdScalar`
- `DynAdTensor`
- `tenferro_dyadtensor`
- `mdarray`

from [lib.rs](/sharehome/shinaoka/projects/tensor4all/tensor4all-rs/crates/tensor4all-tensorbackend/src/lib.rs).

**Step 5: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-tensorbackend
```

Expected: scalar/backend tests pass or fail only on downstream callers not yet updated.

**Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/any_scalar.rs crates/tensor4all-tensorbackend/src/lib.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs
git commit -m "refactor(tensorbackend): switch core facade to tenferro tensor wrappers"
```

### Task 3: Rebuild `TensorDynLen` around `tenferro::Tensor` and remove core `mdarray` usage

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/tensor_like.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Test: `crates/tensor4all-core/tests/*` covering scalar semantics and tensor algebra

**Step 1: Write failing tests for the new public surface**

Add or update tests so they assert:

- `sum`, `only`, `inner_product` return `Scalar`
- `scale` and `axpby` accept `Scalar`
- `TensorDynLen` no longer exposes `from_native`, `as_native`, `into_native` publicly

**Step 2: Change `TensorDynLen` to store `tenferro::Tensor`**

Remove `DynAdTensor` from the struct and replace it with `tenferro::Tensor`.

**Step 3: Remove public native escape hatches**

Delete or demote to internal helpers:

- `from_native`
- `as_native`
- `into_native`

**Step 4: Delete `DTensor`-based helper paths from `tensordynlen.rs`**

Remove imports and helper functions that convert unfolded tensors into `mdarray::DTensor`.

**Step 5: Update `TensorLike` signatures**

Change trait signatures so tensor/scalar mixed operations are expressed entirely in terms of the
new `Scalar` wrapper rather than old native scalar aliases.

**Step 6: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
cargo nextest run --release -p tensor4all-core --test scalar_public_api
```

Expected: PASS or only fail in einsum/linalg code paths not yet migrated.

**Step 7: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/src/tensor_like.rs crates/tensor4all-core/src/lib.rs crates/tensor4all-core/tests
git commit -m "refactor(core): rebuild tensor boundary on tenferro tensor"
```

### Task 4: Delete storage-level einsum and route all contraction execution through `tenferro::Tensor::einsum`

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/contract.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Delete or drastically shrink: `crates/tensor4all-tensorbackend/src/einsum.rs`
- Test: contraction and einsum regression tests in `tensor4all-core`, `tensor4all-treetn`, `tensor4all-itensorlike`

**Step 1: Write failing contraction regressions that exercise multi-tensor and hyperedge cases**

Cover:

- repeated labels / hyperedges
- output axis ordering
- diagonal inputs
- complex/real promotion at frontend level

**Step 2: Move the only execution path to `tenferro::Tensor::einsum`**

Keep tensor4all-owned index/label preparation in `contract.rs`, but once labels are prepared,
execute only through the dynamic frontend einsum.

**Step 3: Remove storage-level einsum facade**

Delete `einsum_storage` and related dense conversion helpers if they are no longer used by any
core/backend path.

**Step 4: Remove old workaround layers that only existed for the storage/native split**

If helper code exists solely to choose between storage and native einsum execution, collapse it
to the single `tenferro` path.

**Step 5: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core contract
cargo nextest run --release -p tensor4all-treetn
cargo nextest run --release -p tensor4all-itensorlike
```

Expected: PASS for contraction behavior; failures should now reflect genuine semantic regressions,
not missing `mdarray` adapters.

**Step 6: Commit**

```bash
git add crates/tensor4all-core/src/defaults/contract.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs crates/tensor4all-tensorbackend/src/einsum.rs crates/tensor4all-core/tests crates/tensor4all-treetn/tests crates/tensor4all-itensorlike/tests
git commit -m "refactor(core): route all einsum execution through tenferro frontend"
```

### Task 5: Delete `DTensor`/`DSlice` linalg adapters and route all QR/SVD/factorize execution through `tenferro`

**Files:**
- Delete or drastically shrink: `crates/tensor4all-tensorbackend/src/backend.rs`
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Test: linalg and factorization regressions in `tensor4all-core`, `tensor4all-itensorlike`, `tensor4all-treetn`

**Step 1: Write failing linalg regressions against the public tensor API**

Cover:

- `qr` reconstruction and truncation
- `svd` reconstruction, singular values, and truncation
- `factorize` rank/bond-index semantics
- complex and real cases

**Step 2: Remove `backend.rs` as a matrix-adapter layer**

Delete `DTensor`/`DSlice` wrappers and use `tenferro::Tensor::{qr, svd, ...}` directly.

**Step 3: Rework `qr.rs` and `svd.rs` to unfold with `TensorDynLen` + native reshape only**

Keep index bookkeeping in tensor4all, but do not materialize `mdarray` matrices at any point.

**Step 4: Rework `factorize.rs` to consume `TensorDynLen` outputs directly**

If any code still reads `Storage` only to rebuild matrix-like intermediates, delete that path and
use native linalg outputs instead.

**Step 5: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core qr svd factorize
cargo nextest run --release -p tensor4all-itensorlike
cargo nextest run --release -p tensor4all-treetn
```

Expected: PASS. No linalg execution path should rely on `mdarray`.

**Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/backend.rs crates/tensor4all-core/src/defaults/qr.rs crates/tensor4all-core/src/defaults/svd.rs crates/tensor4all-core/src/defaults/factorize.rs crates/tensor4all-core/tests crates/tensor4all-itensorlike/tests crates/tensor4all-treetn/tests
git commit -m "refactor(core): route all linalg through tenferro frontend"
```

### Task 6: Shrink `Storage` to an explicit snapshot/materialization boundary

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-capi/src/tensor.rs` if needed for compile recovery
- Test: storage/materialization regressions and C API compile checks

**Step 1: Write failing tests for `to_storage()` / `from_storage()` as explicit boundaries**

Cover:

- round-trip for dense tensors
- diagonal snapshot behavior
- scalar snapshot behavior

**Step 2: Remove execution helpers from `Storage` that are no longer used by core/backend**

Delete or isolate methods whose only purpose was to support storage-based einsum/linalg execution.

**Step 3: Keep only the materialization-oriented helpers required by current downstream crates**

This includes explicit snapshot paths used by C API and HDF5, but not parallel execution logic.

**Step 4: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core
cargo check -p tensor4all-capi
```

Expected: PASS for core tests and successful C API compile recovery.

**Step 5: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-capi/src/tensor.rs
git commit -m "refactor(storage): keep only snapshot and interop boundaries"
```

### Task 7: Final cleanup review for DRY, KISS, and layering

**Files:**
- Review: `crates/tensor4all-core/src/defaults/*.rs`
- Review: `crates/tensor4all-tensorbackend/src/*.rs`
- Update: docs and comments that still describe removed paths

**Step 1: Search for forbidden remnants**

Run:

```bash
rg -n \"mdarray|DTensor|DSlice|DynAdTensor|DynAdScalar|einsum_storage|svd_backend|qr_backend|pub use mdarray\" crates/tensor4all-core crates/tensor4all-tensorbackend
```

Expected: no remaining execution-path references in the targeted crates.

**Step 2: Review for DRY violations**

Check that promotion, scalar extraction, reshape/materialization, and contraction label logic are
implemented in one place only.

**Step 3: Review for KISS violations**

Delete any remaining storage/native branching that no longer has a caller.

**Step 4: Review for layering violations**

Ensure:

- core does not manipulate low-level native internals directly
- backend owns native execution details
- storage is only a boundary type

**Step 5: Run final verification**

Run:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo nextest run --release --workspace
```

Expected: all commands succeed.

**Step 6: Commit**

```bash
git add crates/tensor4all-core crates/tensor4all-tensorbackend crates/tensor4all-capi docs/api
git commit -m "refactor(core): remove mdarray from core execution paths"
```
