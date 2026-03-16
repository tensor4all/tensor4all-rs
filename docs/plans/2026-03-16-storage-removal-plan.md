# Storage Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `Storage` completely from the tensor4all core/backend layer and replace it with generic public constructors over `tenferro::Tensor`.

**Architecture:** Keep `TensorDynLen` and `Scalar` as thin tensor4all-owned wrappers over `tenferro::Tensor`. Replace `Storage`-based construction/materialization with generic typed constructors (`from_dense<T>`, `from_diag<T>`, scalar generic construction) and direct `tenferro` bridge helpers. Delete the storage module and any remaining storage-centric execution code.

**Tech Stack:** Rust, `tensor4all-core`, `tensor4all-tensorbackend`, `tenferro`, `num-complex`, `cargo fmt`, `cargo clippy`, `cargo nextest --release`

---

### Task 1: Introduce generic public constructors and lock their semantics with tests

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Test: `crates/tensor4all-core/tests/*`

**Step 1: Write failing tests for generic dense construction**

Add tests covering:

- `TensorDynLen::from_dense::<f32>(...)`
- `TensorDynLen::from_dense::<f64>(...)`
- `TensorDynLen::from_dense::<Complex32>(...)`
- `TensorDynLen::from_dense::<Complex64>(...)`

Check:

- dims follow index order
- row-major input interpretation is preserved
- mismatched lengths fail

**Step 2: Write failing tests for generic diagonal construction**

Add tests covering:

- valid diagonal construction for all supported element types
- rejection when index dimensions differ
- rejection when diagonal payload length is wrong

**Step 3: Write failing tests for generic scalar construction**

Add tests covering:

- `Scalar::from_value::<f32>(...)`
- `Scalar::from_value::<f64>(...)`
- `Scalar::from_value::<Complex32>(...)`
- `Scalar::from_value::<Complex64>(...)`

**Step 4: Implement a backend element trait**

Create a single internal/public trait for supported tensor element types so dtype dispatch exists
in one place.

**Step 5: Implement `TensorDynLen::from_dense<T>` and `TensorDynLen::from_diag<T>`**

Construct native tensors directly without `Storage`.

**Step 6: Implement `Scalar::from_value<T>`**

Construct rank-0 native tensors directly without storage materialization.

**Step 7: Run focused tests**

Run:

```bash
cargo test --release -p tensor4all-core from_dense -- --nocapture
cargo test --release -p tensor4all-core from_diag -- --nocapture
cargo test --release -p tensor4all-tensorbackend scalar -- --nocapture
```

Expected: new constructor tests pass.

### Task 2: Remove `Storage` from `TensorDynLen` constructors and public core APIs

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Modify: `crates/tensor4all-core/src/tensor_like.rs`
- Test: `crates/tensor4all-core/tests/*`

**Step 1: Replace `TensorDynLen::new(indices, Arc<Storage>)` uses in tests with generic constructors**

Update tests to stop constructing tensors through `Storage`.

**Step 2: Delete `TensorDynLen::new(indices, Arc<Storage>)` and `from_storage(...)`**

Keep only native and generic typed construction paths.

**Step 3: Remove any public re-exports that imply `Storage`-first construction**

Update crate exports and docs accordingly.

**Step 4: Run focused core regressions**

Run:

```bash
cargo test --release -p tensor4all-core --test linalg_qr test_qr_reconstruction_with_unit_dim_axis -- --exact --nocapture
cargo test --release -p tensor4all-core --test linalg_svd test_svd_reconstruction_with_unit_dim_axis -- --exact --nocapture
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
```

Expected: PASS.

### Task 3: Replace remaining `Storage`-based backend helpers with direct native construction/materialization

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Test: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`

**Step 1: Identify helpers that still accept or return `Storage`**

Inventory and group them into:

- construction helpers
- export/materialization helpers
- dead code

**Step 2: Replace construction helpers with typed direct builders**

Bridge code should accept typed `Vec<T>`/slice inputs and build native tensors directly.

**Step 3: Redesign export helpers around snapshots or typed extraction**

Do not recreate a new storage enum under another name.

**Step 4: Delete dead helpers**

Remove any helper that exists only to translate between legacy `Storage` and native tensors.

**Step 5: Run backend tests**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend --lib -- --nocapture
```

Expected: PASS.

### Task 4: Delete the `storage` module and its public re-exports

**Files:**
- Delete: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/Cargo.toml`
- Modify: `Cargo.toml`

**Step 1: Remove the module from `lib.rs`**

Delete the module declaration and all `pub use` exports from `storage.rs`.

**Step 2: Remove `mdarray` dependency**

Delete `mdarray` from the crate dependency list if nothing else still needs it.

**Step 3: Remove storage-driven docs**

Regenerate API docs after deleting the module.

**Step 4: Run compile checks**

Run:

```bash
cargo check -p tensor4all-tensorbackend
cargo check -p tensor4all-core
```

Expected: FAIL only at downstream callers still importing `Storage`.

### Task 5: Update downstream core callers to the new construction/export model

**Files:**
- Modify: `crates/tensor4all-core/tests/*`
- Modify: `crates/tensor4all-itensorlike/tests/*`
- Modify: `crates/tensor4all-simplett/src/mpo/factorize.rs`
- Modify: any core/backend caller still importing `Storage`

**Step 1: Replace direct `Storage::*` constructors with `from_dense<T>` / `from_diag<T>`**

Do this crate-by-crate so the migration stays mechanical.

**Step 2: Remove `Storage` from helper utilities and fixtures**

Delete utility functions that build `Arc<Storage>` just to seed tensors.

**Step 3: Re-run focused downstream tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
cargo test --release -p tensor4all-core --test linalg_qr test_qr_reconstruction_with_multiple_unit_dims -- --exact --nocapture
cargo test --release -p tensor4all-core --test linalg_svd test_svd_reconstruction_with_multiple_unit_dims -- --exact --nocapture
cargo nextest run --release -p tensor4all-itensorlike --test tensortrain_native_ad
```

Expected: PASS or only fail in crates not yet migrated.

### Task 6: DRY / KISS / layering cleanup review

**Files:**
- Review: all changed files in `tensor4all-core` and `tensor4all-tensorbackend`

**Step 1: Check DRY**

Verify there is one place for:

- supported element type dispatch
- dense construction validation
- diagonal construction validation
- row-major boundary reshape logic

**Step 2: Check KISS**

Verify:

- no second tensor representation remains
- no shadow storage cache was introduced
- no constructor family duplicates behavior by dtype

**Step 3: Check layering**

Verify:

- core owns index semantics only
- backend owns the tenferro bridge only
- tests use public constructors rather than internal native/stateful shortcuts

**Step 4: Final verification**

Run:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --release
cargo nextest run --release --workspace
cargo run -p api-dump --release -- . -o docs/api
```

Expected: PASS, or a short explicit list of remaining downstream crates still blocked by the
`Storage` removal.
