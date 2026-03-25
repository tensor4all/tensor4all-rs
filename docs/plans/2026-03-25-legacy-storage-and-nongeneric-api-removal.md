# Legacy Storage Removal & Non-Generic API Cleanup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove legacy DenseStorage/DiagStorage types and mdarray dependency (#321), then remove non-generic f64/c64 public Rust APIs (#300).

**Architecture:** All external code paths already create StructuredStorage (col-major). Legacy DenseStorage/DiagStorage match arms in Storage operations are dead code in production—they only survive via test code that explicitly creates them. TensorDynLen already routes through the tenferro native path for all compute. The migration is therefore: (1) add Structured arms where missing, (2) delete legacy types/arms, (3) update tests, (4) remove mdarray. For #300, add generic APIs, migrate callers, delete scalar-specific wrappers.

**Tech Stack:** Rust, tenferro, tenferro_tensor, num_complex

---

## File Map

### #321 – tensorbackend

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/tensor4all-tensorbackend/src/storage.rs` | **Heavy modify** | Remove ~800 lines of legacy types (DenseStorage, DiagStorage, DenseScalar), legacy StorageRepr variants, all legacy match arms, legacy constructors, helper functions (promote_dense_to_c64, promote_diag_to_c64, contract_diag_dense_impl, contract_dense_diag_impl). Add StructuredStorage-based arms for operations that lack them (try_add, try_sub, combine_to_complex, axpby, Add impl, Mul impls). |
| `crates/tensor4all-tensorbackend/src/storage/tests/mod.rs` | **Heavy modify** | Rewrite all tests to use Structured constructors instead of legacy DenseStorage/DiagStorage |
| `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs` | **Light modify** | Remove `#[cfg(test)] use crate::storage::StorageRepr;` if unused |
| `crates/tensor4all-tensorbackend/src/tenferro_bridge/tests/mod.rs` | **Modify** | Update tests referencing StorageRepr legacy variants |
| `crates/tensor4all-tensorbackend/Cargo.toml` | **Modify** | Remove `mdarray` dependency |

### #321 – simplett

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/tensor4all-simplett/src/types.rs` | **Rewrite** | Replace `DTensor<T, 3>` with own `Tensor<T, N>` wrapper over `Vec<T>` + shape |
| `crates/tensor4all-simplett/src/mpo/types.rs` | **Rewrite** | Replace `DTensor<T, 4>` with `Tensor<T, 4>` |
| `crates/tensor4all-simplett/src/mpo/mod.rs` | **Modify** | Replace `DTensor<T, 2>` + `matrix2_zeros` with shared `Tensor<T, 2>` |
| `crates/tensor4all-simplett/src/mpo/environment.rs` | **Modify** | Remove duplicate `Matrix2<T>` + `matrix2_zeros`, use shared definition |
| All other simplett source files using Tensor3/4/Matrix2 | **Modify** | Update imports, adapt to new `Tensor<T, N>` API |
| `crates/tensor4all-simplett/Cargo.toml` | **Modify** | Remove `mdarray` dependency |

### #300 – non-generic API removal

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/tensor4all-core/src/defaults/tensordynlen.rs` | **Modify** | Add generic methods, remove *_f64/*_c64 wrappers |
| `crates/tensor4all-tensorbackend/src/storage.rs` | **Modify** | Add generic Storage constructors, remove *_f64/*_c64 methods |
| `crates/tensor4all-core/src/defaults/qr.rs` | **Modify** | Remove `qr_c64` |
| `crates/tensor4all-core/src/defaults/svd.rs` | **Modify** | Remove `svd_c64` |
| `crates/tensor4all-treetn/src/random.rs` | **Modify** | Remove `random_treetn_f64/c64`, keep generic impl |
| `crates/tensor4all-treetn/src/operator/identity.rs` | **Modify** | Remove `build_identity_operator_tensor_c64`, add generic |
| Many test/example files | **Modify** | Migrate callers to generic APIs |
| `crates/tensor4all-core/src/defaults/mod.rs`, `lib.rs` | **Modify** | Update re-exports |

---

## Task 1: Remove legacy DenseStorage/DiagStorage from storage.rs (#321)

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`

### Strategy

All public Storage constructors (`from_dense_f64_col_major`, `from_diag_f64_col_major`, etc.) already create `StructuredF64/C64`. The native bridge (`native_tensor_primal_to_storage`) also creates Structured. Legacy match arms are dead code in production—only test code creates legacy storage.

Approach:
1. Add Structured support to operations that only have legacy arms
2. Remove legacy types, constructors, match arms
3. Simplify StorageRepr to just `F64(StructuredStorage<f64>)` and `C64(StructuredStorage<Complex64>)`

- [ ] **Step 1: Add StructuredStorage-based arms to operations missing them**

Operations needing Structured arms: `try_add`, `try_sub`, `combine_to_complex`, `axpby`, `Add<&Storage>` impl.

For element-wise operations on two StructuredStorages:
```rust
// try_add for Structured
(StorageRepr::StructuredF64(a), StorageRepr::StructuredF64(b)) => {
    if a.len() != b.len() { return Err(...); }
    let sum: Vec<f64> = a.data().iter().zip(b.data()).map(|(&x, &y)| x + y).collect();
    Ok(Storage::structured_f64(StructuredStorage::new(
        sum, a.payload_dims().to_vec(), a.strides().to_vec(), a.axis_classes().to_vec()
    )?))
}
```

Also add Structured handling for `SumFromStorage` if missing (it already has it).

- [ ] **Step 2: Verify new Structured arms compile and pass basic smoke test**

Run: `cargo nextest run --release -p tensor4all-tensorbackend`

- [ ] **Step 3: Remove DenseStorage<T> type and all code (lines ~35-275)**

Remove:
- `DenseScalar` trait (lines 14-33)
- `DenseStorage<T>` struct and all impl blocks (lines 35-380)
- `DenseStorageF64`, `DenseStorageC64` type aliases (lines 273, 277)
- `promote_dense_to_c64` function
- `DenseStorage::contract` method and related helpers

- [ ] **Step 4: Remove DiagStorage<T> type and all code (lines ~673-835)**

Remove:
- `DiagStorage<T>` struct and all impl blocks (lines 673-826)
- `DiagStorageF64`, `DiagStorageC64` type aliases (lines 831, 835)
- `promote_diag_to_c64` function
- `contract_diag_dense_impl` function
- `contract_dense_diag_impl` function

- [ ] **Step 5: Remove legacy StorageRepr variants and simplify**

Change:
```rust
pub(crate) enum StorageRepr {
    F64(StructuredStorage<f64>),
    C64(StructuredStorage<Complex64>),
}
```

Update all remaining match arms to use `StorageRepr::F64` / `StorageRepr::C64`.

- [ ] **Step 6: Remove legacy constructors and update Storage API**

Remove: `dense_f64_legacy`, `dense_c64_legacy`, `diag_f64_legacy`, `diag_c64_legacy`, `structured_f64`, `structured_c64`.

Replace with direct construction or simplified helpers.

- [ ] **Step 7: Remove all legacy match arms from operations**

Each operation (permute_storage, extract_real_part, extract_imag_part, to_complex_storage, conj, combine_to_complex, to_dense_f64_col_major_vec, to_dense_c64_col_major_vec, native_payload_f64, native_payload_c64, SumFromStorage, max_abs, len, is_dense, is_diag, is_f64, is_c64, Mul impls, contract_storage) should only have F64/C64 arms.

- [ ] **Step 8: Remove mdarray import and row_major helpers**

Remove `use mdarray::{DynRank, Shape, Tensor};` and `row_major_to_col_major_values` if no longer needed.

- [ ] **Step 9: Remove mdarray from Cargo.toml**

- [ ] **Step 10: Compile check**

Run: `cargo build --release -p tensor4all-tensorbackend`

---

## Task 2: Update tensorbackend tests (#321)

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/storage/tests/mod.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge/tests/mod.rs`

- [ ] **Step 1: Rewrite storage tests to use Structured constructors**

Replace all `DenseStorage::from_vec_with_shape(...)` / `Storage::dense_f64_legacy(...)` with `Storage::from_dense_f64_col_major(...)`.

Replace all `DiagStorage::from_vec(...)` / `Storage::diag_f64_legacy(...)` with `Storage::from_diag_f64_col_major(...)`.

**Important**: Legacy DenseStorage was ROW-MAJOR. Structured is COLUMN-MAJOR. Test data must be adjusted for column-major layout.

- [ ] **Step 2: Update tenferro_bridge tests**

Remove any `StorageRepr::DenseF64` / `DiagF64` pattern matching.

- [ ] **Step 3: Run all tensorbackend tests**

Run: `cargo nextest run --release -p tensor4all-tensorbackend`

- [ ] **Step 4: Commit**

---

## Task 3: Replace mdarray DTensor in simplett (#321)

**Files:**
- Create: `crates/tensor4all-simplett/src/tensor.rs` (shared `Tensor<T, N>` wrapper)
- Modify: `crates/tensor4all-simplett/src/types.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/types.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/mod.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/environment.rs`
- Modify: All simplett files using Tensor3/Tensor4/Matrix2
- Modify: `crates/tensor4all-simplett/Cargo.toml`

- [ ] **Step 1: Create Tensor<T, N> wrapper**

```rust
// crates/tensor4all-simplett/src/tensor.rs

/// Rank-N tensor backed by a flat Vec<T> with row-major layout.
/// Rank is fixed at compile time via const generic.
#[derive(Debug, Clone)]
pub struct Tensor<T, const N: usize> {
    data: Vec<T>,
    dims: [usize; N],
}

pub type Tensor2<T> = Tensor<T, 2>;
pub type Tensor3<T> = Tensor<T, 3>;
pub type Tensor4<T> = Tensor<T, 4>;
```

Implement: `new`, `from_elem`, `from_fn`, `dim`, `dims`, `Index<[usize; N]>`, `IndexMut<[usize; N]>`, `as_slice`, `as_mut_slice`.

- [ ] **Step 2: Migrate types.rs (Tensor3)**

Replace `pub type Tensor3<T> = DTensor<T, 3>;` with `pub use crate::tensor::Tensor3;`.
Update `Tensor3Ops` impl to use new Tensor API.
Update `tensor3_zeros` and `tensor3_from_data`.

- [ ] **Step 3: Migrate mpo/types.rs (Tensor4)**

Same pattern as Tensor3.

- [ ] **Step 4: Migrate mpo/mod.rs and mpo/environment.rs (Matrix2)**

Remove duplicate `Matrix2` definition. Use `crate::tensor::Tensor2` everywhere.

- [ ] **Step 5: Update all simplett source files**

Grep for `DTensor`, `mdarray`, `.dim(` and migrate to new API.

- [ ] **Step 6: Update simplett tests**

- [ ] **Step 7: Remove mdarray from simplett Cargo.toml**

- [ ] **Step 8: Run all simplett tests**

Run: `cargo nextest run --release -p tensor4all-simplett`

- [ ] **Step 9: Commit**

---

## Task 4: Add generic APIs and remove non-generic wrappers (#300)

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Modify: `crates/tensor4all-treetn/src/random.rs`
- Modify: `crates/tensor4all-treetn/src/operator/identity.rs`
- Modify: `crates/tensor4all-core/src/defaults/mod.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Modify: `crates/tensor4all-treetn/src/lib.rs`

- [ ] **Step 1: Add generic Storage constructors**

```rust
// In storage.rs - replace from_dense_f64_col_major / from_dense_c64_col_major
impl Storage {
    pub fn from_dense_col_major<T: StorageScalar>(data: Vec<T>, logical_dims: &[usize]) -> Result<Self> {
        T::build_dense_storage(data, logical_dims)
    }
    pub fn from_diag_col_major<T: StorageScalar>(diag_data: Vec<T>, logical_rank: usize) -> Result<Self> {
        T::build_diag_storage(diag_data, logical_rank)
    }
    pub fn new_dense<T: StorageScalar>(size: usize) -> Self { ... }
    pub fn new_diag<T: StorageScalar>(diag_data: Vec<T>) -> Self { ... }
}
```

Add `StorageScalar` trait (or extend existing) with `build_dense_storage`/`build_diag_storage`.

- [ ] **Step 2: Add generic TensorDynLen methods**

```rust
// Replace to_vec_f64/c64, as_slice_f64/c64
impl TensorDynLen {
    pub fn to_vec<T: TensorElement>(&self) -> Result<Vec<T>> { ... }
    pub fn scalar<T: TensorElement>(value: T, ...) -> Self { ... }
    pub fn zeros<T: TensorElement>(...) -> Self { ... }
    pub fn random<T: TensorElement, R: Rng>(...) -> Self { ... }
}
```

- [ ] **Step 3: Add generic free functions**

```rust
// Replace qr_c64, svd_c64, diag_tensor_dyn_len_c64
pub fn diag_tensor_dyn_len<T: TensorElement>(...) -> TensorDynLen { ... }
// qr::<T> and svd::<T> already exist as generic

// Replace random_treetn_f64/c64
pub fn random_treetn<T: TreeTnScalar, R: Rng, V: ...>(...) -> TreeTN<V> { ... }

// Replace build_identity_operator_tensor_c64
pub fn build_identity_operator_tensor<T: TensorElement>(...) -> TensorDynLen { ... }
```

- [ ] **Step 4: Migrate all callers in library code**

Search-and-replace across all crates (excluding capi which is out of scope).

- [ ] **Step 5: Migrate all callers in test/example code**

This is the bulk of the work (~200+ call sites for to_vec_f64/c64 alone).

- [ ] **Step 6: Remove non-generic wrappers**

Delete: `from_dense_f64`, `from_dense_c64`, `as_slice_f64`, `as_slice_c64`, `to_vec_f64`, `to_vec_c64`, `sum_f64`, `scalar_f64`, `scalar_c64`, `zeros_f64`, `zeros_c64`, `random_f64`, `random_c64`, `qr_c64`, `svd_c64`, `diag_tensor_dyn_len_c64`, `random_treetn_f64`, `random_treetn_c64`, `build_identity_operator_tensor_c64`, `Storage::new_dense_f64`, `Storage::new_dense_c64`, `Storage::new_diag_f64`, `Storage::new_diag_c64`, `Storage::sum_f64`, `Storage::sum_c64`, `Storage::from_dense_f64_col_major`, `Storage::from_dense_c64_col_major`, `Storage::from_diag_f64_col_major`, `Storage::from_diag_c64_col_major`.

Keep: `is_f64`, `is_complex`, `is_c64` (runtime type queries, still useful).

- [ ] **Step 7: Update re-exports in mod.rs/lib.rs**

- [ ] **Step 8: Run full workspace tests**

Run: `cargo nextest run --release --workspace`

- [ ] **Step 9: Commit**

---

## Task 5: Final verification and PR

- [ ] **Step 1: Run full lint and test suite**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --release --workspace
```

- [ ] **Step 2: Verify mdarray removed from both Cargo.toml files**

- [ ] **Step 3: Verify README.md is still accurate**

- [ ] **Step 4: Create PR and merge**
