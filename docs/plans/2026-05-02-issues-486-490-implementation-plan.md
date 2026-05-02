# Issues 486-490 Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the AGENTS.md audit cleanup for issues #486, #487, #488, #489, and #490 in one branch while keeping each change reviewable.

**Architecture:** Treat the work as four independently reviewable chunks: layering cleanup, core public-surface cleanup, TCI public-surface cleanup, and error/documentation cleanup. Each chunk starts with a focused regression or API check, then makes the minimal code change, then runs crate-local verification before moving to the next chunk.

**Tech Stack:** Rust workspace, Cargo release-mode tests, rustdoc examples, `api-dump`, `cargo fmt`, `rg`, and existing tensor4all crate APIs.

---

## Execution Rules

- Work from branch `issues-486-490-spec-design` or a branch based on current `origin/main`.
- Do not push or create a PR without explicit user approval.
- Keep each task as a separate commit if committing during execution.
- Run `cargo fmt --all` before any commit and before final verification.
- Regenerate `docs/api` after public API changes:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

- Do not close #484 or #485 from this PR. Mention them only as partially addressed if local cleanup removes relevant `unwrap`, `expect`, or `panic` paths.

## Task 1: Establish Baseline And Public Surface Inventory

**Files:**
- Read: `crates/tensor4all-core/src/lib.rs`
- Read: `crates/tensor4all-tensorci/src/lib.rs`
- Read: `crates/tensor4all-tcicore/src/lib.rs`
- Read: `crates/tensor4all-core/src/defaults/index.rs`
- Read: `crates/tensor4all-tensorbackend/src/storage.rs`

**Step 1: Capture baseline search output**

Run:

```bash
rg -n "pub mod storage|RandomScalar|TensorAccess|pub use tensor4all_tcicore|DenseLuKernel|LazyBlockRookKernel|PivotKernel|Result<[^>]*String|allow\\(missing_docs\\)|Err\\(\\(code, msg\\)\\)" crates
```

Expected: matches in the issue docs, including `tensor4all-core/src/lib.rs`, `tensor4all-tensorci/src/lib.rs`, `tensor4all-tcicore/src/lib.rs`, `tensor4all-tensorbackend/src/storage.rs`, `tensor4all-core/src/any_scalar.rs`, and `tensor4all-capi/src/quanticstransform.rs`.

**Step 2: Check current compile before edits**

Run:

```bash
cargo check --release -p tensor4all-core
cargo check --release -p tensor4all-tcicore
cargo check --release -p tensor4all-tensorci
```

Expected: PASS. If baseline fails, stop and record the failure before editing.

**Step 3: Commit design docs only if requested**

If the design docs should be committed separately:

```bash
git add docs/plans/2026-05-02-issue-487-layering-cleanup-design.md \
        docs/plans/2026-05-02-issue-488-core-public-surface-design.md \
        docs/plans/2026-05-02-issue-489-tci-public-surface-design.md \
        docs/plans/2026-05-02-issues-486-490-error-doc-cleanup-design.md \
        docs/plans/2026-05-02-issues-486-490-implementation-plan.md
git commit -m "docs: plan AGENTS audit cleanup"
```

Expected: commit succeeds. Skip this step if the user wants all docs and implementation in one commit.

## Task 2: Add `DynId::value` And Layering Tests (#487)

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/index.rs`
- Modify: `crates/tensor4all-hdf5/tests/test_hdf5.rs`
- Test: `crates/tensor4all-core/src/defaults/index/tests/mod.rs`

**Step 1: Add a failing test for public ID access**

Add to `crates/tensor4all-core/src/defaults/index/tests/mod.rs`:

```rust
#[test]
fn test_dyn_id_value_accessor() {
    let id = DynId(42);
    assert_eq!(id.value(), 42);
}
```

Run:

```bash
cargo test --release -p tensor4all-core test_dyn_id_value_accessor
```

Expected: FAIL with no method named `value`.

**Step 2: Implement `DynId::value`**

In `crates/tensor4all-core/src/defaults/index.rs`, add below `pub struct DynId(pub u64);`:

```rust
impl DynId {
    /// Return the numeric ID value used for ITensors-compatible serialization.
    ///
    /// Prefer full [`Index`](crate::Index) equality for tensor index identity.
    /// This accessor exists for stable serialization and FFI query paths that
    /// must expose the raw numeric identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynId;
    ///
    /// let id = DynId(42);
    /// assert_eq!(id.value(), 42);
    /// ```
    pub fn value(&self) -> u64 {
        self.0
    }
}
```

Run:

```bash
cargo test --release -p tensor4all-core test_dyn_id_value_accessor
cargo test --doc --release -p tensor4all-core DynId
```

Expected: PASS.

**Step 3: Add HDF5 full-index identity regression**

Add a test to `crates/tensor4all-hdf5/tests/test_hdf5.rs`:

```rust
#[test]
fn test_roundtrip_preserves_same_id_distinct_metadata_indices() -> anyhow::Result<()> {
    use tensor4all_core::{DynId, Index, TagSet, TensorDynLen};
    use tensor4all_hdf5::{load_itensor, save_itensor};

    let tags_a = TagSet::from_str("Site,A")?;
    let tags_b = TagSet::from_str("Site,B")?;
    let i = Index::new_with_tags(DynId(7), 2, tags_a);
    let j = Index::new_with_tags(DynId(7), 3, tags_b).set_plev(1);
    assert_ne!(i, j);

    let tensor = TensorDynLen::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("same_id_metadata.h5");
    let path = path.to_str().unwrap();

    save_itensor(path, "tensor", &tensor)?;
    let loaded = load_itensor(path, "tensor")?;

    assert_eq!(loaded.indices(), tensor.indices());
    assert_eq!(loaded.to_vec::<f64>()?, tensor.to_vec::<f64>()?);
    Ok(())
}
```

Run:

```bash
cargo test --release -p tensor4all-hdf5 test_roundtrip_preserves_same_id_distinct_metadata_indices
```

Expected: PASS before the production cleanup may already pass, but it must remain passing after removing field access.

**Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-core/src/defaults/index.rs crates/tensor4all-hdf5/tests/test_hdf5.rs
git commit -m "feat: add public DynId value accessor"
```

Expected: commit succeeds.

## Task 3: Remove Downstream Direct Field Access (#487, part of #490)

**Files:**
- Modify: `crates/tensor4all-hdf5/src/index.rs`
- Modify: `crates/tensor4all-hdf5/src/lib.rs`
- Modify: `crates/tensor4all-capi/src/index.rs`
- Modify: `crates/tensor4all-capi/src/tensor.rs`
- Modify: `crates/tensor4all-capi/src/treetn.rs`

**Step 1: Replace HDF5 index field reads/writes**

In `crates/tensor4all-hdf5/src/index.rs`, use this pattern:

```rust
use tensor4all_core::IndexLike;
```

Replace direct fields:

```rust
id_ds.as_writer().write_scalar(&index.id().value())?;
dim_ds.as_writer().write_scalar(&(index.dim() as i64))?;
plev_ds.as_writer().write_scalar(&index.plev())?;
write_tagset(&tags_group, index.tags())?;
```

Replace deserialization mutation:

```rust
let idx = Index::new_with_tags(DynId(id), dim as usize, tags).set_plev(plev);
Ok(idx)
```

Run:

```bash
cargo test --release -p tensor4all-hdf5 test_roundtrip_preserves_same_id_distinct_metadata_indices
```

Expected: PASS.

**Step 2: Update HDF5 rustdoc example**

In `crates/tensor4all-hdf5/src/lib.rs`, replace ID-vector comparison with:

```rust
/// // Index identity and metadata are preserved
/// assert_eq!(loaded.indices(), tensor.indices());
```

Run:

```bash
cargo test --doc --release -p tensor4all-hdf5
```

Expected: PASS.

**Step 3: Replace C API index direct field reads**

In `crates/tensor4all-capi/src/index.rs`, import `IndexLike` if needed and replace:

```rust
Ok(index.id().value())
```

and:

```rust
*out_plev = (*ptr).inner().plev();
```

Run:

```bash
cargo test --release -p tensor4all-capi index
```

Expected: PASS.

**Step 4: Replace tensor index field reads**

In `crates/tensor4all-capi/src/tensor.rs`, replace:

```rust
(*ptr).inner().indices.len()
```

with:

```rust
(*ptr).inner().indices().len()
```

and replace:

```rust
let indices = &(*ptr).inner().indices;
```

with:

```rust
let indices = (*ptr).inner().indices();
```

In `crates/tensor4all-capi/src/treetn.rs`, replace any `tensor.indices` access on `TensorDynLen` values with `tensor.indices()`.

Run:

```bash
rg -n "\\.id\\b|\\.dim\\b|\\.plev\\b|\\.tags\\b|\\.indices\\b" crates/tensor4all-hdf5/src crates/tensor4all-capi/src
cargo test --release -p tensor4all-capi tensor
```

Expected: remaining `rg` matches should be legitimate methods like `.dim()` or tests using local structs, not direct representation reads from `tensor4all-core`; tests pass.

**Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-hdf5/src/index.rs crates/tensor4all-hdf5/src/lib.rs \
        crates/tensor4all-capi/src/index.rs crates/tensor4all-capi/src/tensor.rs \
        crates/tensor4all-capi/src/treetn.rs
git commit -m "fix: use public index and tensor accessors"
```

Expected: commit succeeds.

## Task 4: Remove Core Accidental Re-exports (#488)

**Files:**
- Modify: `crates/tensor4all-core/src/lib.rs`
- Modify: downstream files reported by `cargo check`

**Step 1: Remove accidental exports from core root**

In `crates/tensor4all-core/src/lib.rs`, delete the public `storage` module:

```rust
pub mod storage {
    //! Re-export of snapshot storage utilities.
    pub use tensor4all_tensorbackend::{
        make_mut_storage, mindim, AnyScalar, Storage, StorageKind, StorageScalar,
        StructuredStorage, SumFromStorage,
    };
}
```

Delete the root-level storage re-export:

```rust
pub use storage::{
    make_mut_storage, mindim, Storage, StorageKind, StructuredStorage, SumFromStorage,
};
```

Change the tensor export block from:

```rust
pub use defaults::tensordynlen::{
    compute_permutation_from_indices, diag_tensor_dyn_len, unfold_split, RandomScalar,
    TensorAccess, TensorDynLen,
};
```

to:

```rust
pub use defaults::tensordynlen::{
    compute_permutation_from_indices, diag_tensor_dyn_len, unfold_split, TensorDynLen,
};
```

Run:

```bash
cargo check --release -p tensor4all-core
```

Expected: compile errors identify internal imports that relied on root re-exports.

**Step 2: Fix imports without restoring public root exports**

For code inside `tensor4all-core`, import backend types directly from `tensor4all_tensorbackend`:

```rust
use tensor4all_tensorbackend::{Storage, StorageKind, StructuredStorage};
```

For workspace crates outside core, prefer high-level `TensorDynLen` methods. If a crate genuinely manipulates backend storage, add or use its direct dependency on `tensor4all-tensorbackend`.

Run:

```bash
cargo check --release --workspace
```

Expected: PASS or only failures in #489-related public bounds that will be handled next.

**Step 3: Decide flattened module aliases**

Run:

```bash
rg -n "tensor4all_core::(direct_sum|factorize|qr|svd)::|crate::(direct_sum|factorize|qr|svd)::" crates docs/book README.md
```

If no user-facing examples require these module aliases, remove the `pub mod direct_sum`, `pub mod factorize`, `pub mod qr`, and `pub mod svd` compatibility modules while keeping top-level function/type re-exports. If many docs use them, leave module alias removal for a follow-up and record that in the PR.

**Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-core/src/lib.rs
git add <downstream-import-files>
git commit -m "fix: trim tensor4all-core public re-exports"
```

Expected: commit succeeds.

## Task 5: Add MatrixLUCI Public Facade And Hide Kernel Types (#489)

**Files:**
- Modify: `crates/tensor4all-tcicore/src/lib.rs`
- Modify: `crates/tensor4all-tcicore/src/matrix_luci.rs`
- Modify: `crates/tensor4all-tcicore/src/matrixluci/mod.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`

**Step 1: Add public facade result in `matrix_luci.rs`**

Add after `pub struct MatrixLUCI<T...>`:

```rust
/// High-level MatrixLUCI factors used by tensor cross interpolation.
///
/// This type exposes the selected pivot metadata and reconstructed dense
/// factors without exposing the low-level pivot kernel implementation.
#[derive(Debug, Clone)]
pub struct MatrixLuciFactors<T: Scalar> {
    /// Selected row indices.
    pub row_indices: Vec<usize>,
    /// Selected column indices.
    pub col_indices: Vec<usize>,
    /// Pivot error history.
    pub pivot_errors: Vec<f64>,
    /// Selected rank.
    pub rank: usize,
    /// Column factor multiplied by the inverse pivot block.
    pub cols_times_pivot_inv: Matrix<T>,
    /// Inverse pivot block multiplied by the row factor.
    pub pivot_inv_times_rows: Matrix<T>,
}
```

Add a private converter near existing helper functions:

```rust
fn factors_to_public<T>(
    selection: crate::matrixluci::PivotSelectionCore,
    factors: crate::matrixluci::CrossFactors<T>,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::matrixluci::Scalar,
{
    Ok(MatrixLuciFactors {
        row_indices: selection.row_indices,
        col_indices: selection.col_indices,
        pivot_errors: selection.pivot_errors,
        rank: selection.rank,
        cols_times_pivot_inv: to_row_major(&factors.cols_times_pivot_inv()?),
        pivot_inv_times_rows: to_row_major(&factors.pivot_inv_times_rows()?),
    })
}
```

Run:

```bash
cargo check --release -p tensor4all-tcicore
```

Expected: PASS or missing imports that are fixed in the same file.

**Step 2: Add dense and lazy facade functions**

Add public functions in `crates/tensor4all-tcicore/src/matrix_luci.rs`:

```rust
/// Factorize a dense row-major candidate matrix with MatrixLUCI.
pub fn matrix_luci_factors_from_matrix<T>(
    a: &Matrix<T>,
    options: Option<RrLUOptions>,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::matrixluci::Scalar,
{
    let options = options.unwrap_or_default();
    let (selection, factors) = dense_selection_from_matrix(a, options)?;
    factors_to_public(selection, factors)
}

/// Factorize a lazy candidate matrix with MatrixLUCI block-rook pivot search.
pub fn matrix_luci_factors_from_blocks<T, F>(
    nrows: usize,
    ncols: usize,
    fill_block: F,
    options: RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::matrixluci::Scalar,
    F: Fn(&[usize], &[usize], &mut [T]),
{
    let source = crate::matrixluci::LazyMatrixSource::new(nrows, ncols, fill_block);
    let kernel_options = crate::matrixluci::PivotKernelOptions {
        max_rank: options.max_rank,
        rel_tol: options.rel_tol,
        abs_tol: options.abs_tol,
        left_orthogonal: options.left_orthogonal,
    };
    let selection = crate::matrixluci::LazyBlockRookKernel
        .factorize(&source, &kernel_options)
        .map_err(map_backend_error)?;
    let factors = crate::matrixluci::CrossFactors::from_source(&source, &selection)
        .map_err(map_backend_error)?;
    factors_to_public(selection, factors)
}
```

If the exact `RrLUOptions` fields differ, use the existing field names from `dense_selection_from_matrix`.

Run:

```bash
cargo check --release -p tensor4all-tcicore
```

Expected: PASS.

**Step 3: Re-export only high-level facade**

In `crates/tensor4all-tcicore/src/lib.rs`, replace:

```rust
pub use self::matrixluci::{
    CandidateMatrixSource, CrossFactors, DenseLuKernel, DenseMatrixSource, DenseOwnedMatrix,
    LazyBlockRookKernel, LazyMatrixSource, MatrixLuciError, PivotKernel, PivotKernelOptions,
    PivotSelectionCore,
};
pub use self::matrixluci::{Result as MatrixLuciResult, Scalar as MatrixLuciScalar};
```

with:

```rust
pub use self::matrixluci::Scalar as MatrixLuciScalar;
```

and extend the MatrixLUCI re-export:

```rust
pub use matrix_luci::{
    matrix_luci_factors_from_blocks, matrix_luci_factors_from_matrix, MatrixLUCI,
    MatrixLuciFactors,
};
```

In `crates/tensor4all-tcicore/src/matrixluci/mod.rs`, keep submodules usable inside the crate. If direct external module access remains public through `pub mod matrixluci`, convert implementation submodules to `pub(crate) mod` where possible. If this causes rustdoc link breakage, keep the module public temporarily but ensure low-level items are not re-exported from crate root or public signatures.

Run:

```bash
cargo check --release -p tensor4all-tcicore
```

Expected: PASS.

**Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-tcicore/src/lib.rs crates/tensor4all-tcicore/src/matrix_luci.rs \
        crates/tensor4all-tcicore/src/matrixluci/mod.rs
git commit -m "feat: add MatrixLUCI public facade"
```

Expected: commit succeeds.

## Task 6: Remove Kernel Bounds From Public Signatures (#489)

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Modify: `crates/tensor4all-simplett/src/compression.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/factorize.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`
- Modify: `crates/tensor4all-tensorci/src/integration.rs`
- Modify: `crates/tensor4all-quanticstci/src/quantics_tci.rs`
- Modify: `crates/tensor4all-quanticstci/src/batched/mod.rs`

**Step 1: Remove public where-clause leaks**

Replace bounds like:

```rust
tensor4all_tcicore::DenseLuKernel: tensor4all_tcicore::PivotKernel<T>,
```

and:

```rust
DenseLuKernel: PivotKernel<T>,
LazyBlockRookKernel: PivotKernel<T>,
```

with no bound. Keep scalar capability bounds:

```rust
T: tensor4all_tcicore::MatrixLuciScalar
```

Run:

```bash
rg -n "DenseLuKernel:|LazyBlockRookKernel:|PivotKernel<" crates/tensor4all-core/src/defaults/factorize.rs crates/tensor4all-simplett/src crates/tensor4all-tensorci/src crates/tensor4all-quanticstci/src
```

Expected: only internal `tcicore` implementation files should mention these types after the task is complete.

**Step 2: Update dense MatrixLUCI callers**

Replace direct `MatrixLUCI::from_matrix` usage only where it no longer compiles because of hidden kernel bounds. Prefer:

```rust
let luci = MatrixLUCI::from_matrix(matrix, Some(options))?;
```

if `MatrixLUCI::from_matrix` no longer exposes the kernel bound. Otherwise use:

```rust
let factors = tensor4all_tcicore::matrix_luci_factors_from_matrix(matrix, Some(options))?;
```

and consume `factors.row_indices`, `factors.col_indices`, `factors.pivot_errors`,
`factors.cols_times_pivot_inv`, and `factors.pivot_inv_times_rows`.

Run:

```bash
cargo check --release -p tensor4all-core
cargo check --release -p tensor4all-simplett
```

Expected: PASS.

**Step 3: Update TensorCI2 pivot update**

In `crates/tensor4all-tensorci/src/tensorci2.rs`, replace the direct low-level imports:

```rust
use tensor4all_tcicore::{
    rrlu, AbstractMatrixCI, CrossFactors, DenseLuKernel, DenseMatrixSource, LazyBlockRookKernel,
    LazyMatrixSource, MatrixLUCI, PivotKernel, PivotKernelOptions, RrLUOptions,
};
```

with high-level imports:

```rust
use tensor4all_tcicore::{
    matrix_luci_factors_from_blocks, matrix_luci_factors_from_matrix, rrlu, AbstractMatrixCI,
    MatrixLUCI, RrLUOptions,
};
```

In `update_pivots`, replace the low-level dense branch with:

```rust
let factors = matrix_luci_factors_from_matrix(&pi, Some(RrLUOptions {
    max_rank: context.options.max_bond_dim,
    rel_tol: context.options.tolerance,
    abs_tol: 0.0,
    left_orthogonal: context.left_orthogonal,
}))?;
```

Replace the lazy branch with:

```rust
let factors_result = matrix_luci_factors_from_blocks(
    i_combined.len(),
    j_combined.len(),
    |rows, cols, out: &mut [T]| {
        evaluator.fill_block(rows, cols, out);
    },
    RrLUOptions {
        max_rank: context.options.max_bond_dim,
        rel_tol: context.options.tolerance,
        abs_tol: 0.0,
        left_orthogonal: context.left_orthogonal,
    },
);
if let Some(err) = evaluator.take_error() {
    return Err(err);
}
let factors = factors_result?;
```

Then replace references to `selection` and `CrossFactors` with fields on `factors`.

Run:

```bash
cargo check --release -p tensor4all-tensorci
```

Expected: PASS.

**Step 4: Remove TensorCI transitive re-export**

In `crates/tensor4all-tensorci/src/lib.rs`, delete:

```rust
pub use tensor4all_tcicore::{
    CacheKey, CacheKeyError, CachedFunction, IndexInt, IndexSet, LocalIndex, MultiIndex, Scalar,
};
```

Update rustdoc examples or tests that imported these from `tensor4all_tensorci`:

```rust
use tensor4all_tcicore::MultiIndex;
```

Run:

```bash
cargo test --doc --release -p tensor4all-tensorci
cargo check --release -p tensor4all-tensorci
```

Expected: PASS.

**Step 5: Verify public-surface removal**

Run:

```bash
rg -n "tensor4all_tcicore::(DenseLuKernel|LazyBlockRookKernel|PivotKernel)|DenseLuKernel:|LazyBlockRookKernel:|PivotKernel<" crates docs/book README.md
```

Expected: matches only inside `crates/tensor4all-tcicore/src/matrixluci/` implementation and internal tests. No public workspace crate signature should mention these names.

**Step 6: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-core/src/defaults/factorize.rs \
        crates/tensor4all-simplett/src/compression.rs \
        crates/tensor4all-simplett/src/mpo/factorize.rs \
        crates/tensor4all-tensorci/src/tensorci2.rs \
        crates/tensor4all-tensorci/src/integration.rs \
        crates/tensor4all-tensorci/src/lib.rs \
        crates/tensor4all-quanticstci/src/quantics_tci.rs \
        crates/tensor4all-quanticstci/src/batched/mod.rs
git commit -m "fix: hide MatrixLUCI kernel implementation types"
```

Expected: commit succeeds.

## Task 7: Convert Storage String Errors To `StorageError` (#486)

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Test: `crates/tensor4all-tensorbackend/src/storage/tests.rs` or existing storage test module

**Step 1: Add failing typed-error tests**

Add tests near existing storage tests:

```rust
#[test]
fn payload_f64_reports_scalar_kind_mismatch() {
    let storage = Storage::from_dense_col_major(
        vec![num_complex::Complex64::new(1.0, 0.0)],
        &[1],
    )
    .unwrap();
    let err = storage.payload_f64_col_major_vec().unwrap_err();
    assert!(err.to_string().contains("expected f64 storage"));
}

#[test]
fn try_add_reports_length_mismatch() {
    let a = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_col_major(vec![3.0_f64], &[1]).unwrap();
    let err = a.try_add(&b).unwrap_err();
    assert!(err.to_string().contains("addition"));
    assert!(err.to_string().contains("2 != 1"));
}
```

Run:

```bash
cargo test --release -p tensor4all-tensorbackend payload_f64_reports_scalar_kind_mismatch try_add_reports_length_mismatch
```

Expected: tests may pass with `String`, but they become the safety net for the typed conversion.

**Step 2: Define `StorageError`**

In `crates/tensor4all-tensorbackend/src/storage.rs`, add:

```rust
/// Errors returned by snapshot storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// Storage scalar kind did not match the requested operation.
    #[error("expected {expected} storage, got {actual}")]
    ScalarKindMismatch {
        /// Expected scalar kind.
        expected: &'static str,
        /// Actual scalar kind.
        actual: &'static str,
    },
    /// Storage payload lengths must match for an elementwise operation.
    #[error("storage lengths must match for {operation}: {left} != {right}")]
    LengthMismatch {
        /// Operation name.
        operation: &'static str,
        /// Left storage payload length.
        left: usize,
        /// Right storage payload length.
        right: usize,
    },
    /// Structured storage metadata was invalid after an operation.
    #[error("invalid structured storage: {0}")]
    InvalidStructuredStorage(String),
}

type StorageResult<T> = std::result::Result<T, StorageError>;
```

Re-export in `crates/tensor4all-tensorbackend/src/lib.rs`:

```rust
pub use storage::{
    contract_storage, make_mut_storage, mindim, Storage, StorageError, StorageKind,
    StorageScalar, StructuredStorage, SumFromStorage,
};
```

**Step 3: Change target method signatures**

Change:

```rust
pub fn payload_f64_col_major_vec(&self) -> Result<Vec<f64>, String>
```

to:

```rust
pub fn payload_f64_col_major_vec(&self) -> StorageResult<Vec<f64>>
```

Repeat for:

- `payload_c64_col_major_vec`
- `to_dense_f64_col_major_vec`
- `to_dense_c64_col_major_vec`
- `try_add`
- `try_sub`
- `axpby`

Replace `Err("...".to_string())` with `StorageError` variants. Replace `.map_err(|err| err.to_string())?` with:

```rust
.map_err(|err| StorageError::InvalidStructuredStorage(err.to_string()))?
```

Run:

```bash
cargo test --release -p tensor4all-tensorbackend
cargo check --release --workspace
```

Expected: PASS after downstream `?` conversions are adjusted.

**Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-tensorbackend/src/lib.rs
git commit -m "fix: use typed storage errors"
```

Expected: commit succeeds.

## Task 8: Convert Linsolve And TreeTN String Errors (#486)

**Files:**
- Modify: `crates/tensor4all-itensorlike/src/linsolve.rs`
- Modify: `crates/tensor4all-treetn/src/node_name_network.rs`
- Modify: `crates/tensor4all-treetn/src/named_graph.rs`
- Modify: `crates/tensor4all-treetn/src/site_index_network.rs`

**Step 1: Convert `infer_index_mappings`**

In `crates/tensor4all-itensorlike/src/linsolve.rs`, change:

```rust
) -> std::result::Result<SiteMappings, String> {
```

to:

```rust
) -> crate::error::Result<SiteMappings> {
```

Replace string errors with:

```rust
return Err(TensorTrainError::InvalidStructure {
    message: "explain the incompatible index mapping".to_string(),
});
```

Remove redundant call-site wrappers that convert `String` into `OperationError`.

Run:

```bash
cargo test --release -p tensor4all-itensorlike linsolve
```

Expected: PASS.

**Step 2: Convert TreeTN graph helper returns**

In the three TreeTN graph helper files, add:

```rust
use anyhow::{bail, ensure, Result};
```

Change signatures from:

```rust
pub fn add_node(...) -> Result<NodeIndex, String>
```

to:

```rust
pub fn add_node(...) -> Result<NodeIndex>
```

Replace:

```rust
return Err(format!("node {node_name:?} already exists"));
```

with:

```rust
bail!("node {node_name:?} already exists");
```

Replace boolean checks with `ensure!` where clearer.

Run:

```bash
cargo test --release -p tensor4all-treetn node_name_network named_graph site_index_network
```

Expected: PASS.

**Step 3: Check no remaining targeted `Result<_, String>`**

Run:

```bash
rg -n "Result<[^>]*String|std::result::Result<[^>]*String" \
  crates/tensor4all-itensorlike/src/linsolve.rs \
  crates/tensor4all-treetn/src/node_name_network.rs \
  crates/tensor4all-treetn/src/named_graph.rs \
  crates/tensor4all-treetn/src/site_index_network.rs
```

Expected: no matches.

**Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-itensorlike/src/linsolve.rs \
        crates/tensor4all-treetn/src/node_name_network.rs \
        crates/tensor4all-treetn/src/named_graph.rs \
        crates/tensor4all-treetn/src/site_index_network.rs
git commit -m "fix: replace string errors in graph and linsolve helpers"
```

Expected: commit succeeds.

## Task 9: Document `AnyScalar` Public Methods (#490)

**Files:**
- Modify: `crates/tensor4all-core/src/any_scalar.rs`

**Step 1: Remove the suppression**

Delete:

```rust
#[allow(missing_docs)]
```

from the public `impl AnyScalar` block.

Run:

```bash
cargo check --release -p tensor4all-core
```

Expected: FAIL with missing docs for public methods in `AnyScalar`.

**Step 2: Add runnable docs with assertions**

For constructors, use snippets like:

```rust
/// Create a real scalar value.
///
/// # Arguments
///
/// * `x` - Real value to store.
///
/// # Returns
///
/// A scalar whose real part is `x` and imaginary part is zero.
///
/// # Examples
///
/// ```
/// use tensor4all_core::AnyScalar;
///
/// let x = AnyScalar::new_real(2.5);
/// assert_eq!(x.real(), 2.5);
/// assert_eq!(x.imag(), 0.0);
/// assert!(!x.is_complex());
/// ```
```

For complex constructors:

```rust
/// ```
/// use tensor4all_core::AnyScalar;
///
/// let z = AnyScalar::new_complex(2.0, -3.0);
/// assert_eq!(z.real(), 2.0);
/// assert_eq!(z.imag(), -3.0);
/// assert!(z.is_complex());
/// ```
```

For arithmetic helpers:

```rust
/// ```
/// use tensor4all_core::AnyScalar;
///
/// let z = AnyScalar::new_complex(3.0, 4.0);
/// assert_eq!(z.abs(), 5.0);
/// assert_eq!(z.conj().imag(), -4.0);
/// ```
```

Document every public method in the impl. Do not use `ignore` or `no_run`.

Run:

```bash
cargo test --doc --release -p tensor4all-core AnyScalar
cargo check --release -p tensor4all-core
```

Expected: PASS.

**Step 3: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-core/src/any_scalar.rs
git commit -m "docs: document AnyScalar public methods"
```

Expected: commit succeeds.

## Task 10: Deduplicate QuanticsTransform C API Layout Handling (#490)

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`
- Test: existing `crates/tensor4all-capi/src/quanticstransform/tests/` or module tests

**Step 1: Add helper or macro**

Prefer a helper if lifetimes compile:

```rust
fn require_layout_or_status(
    layout: *const t4a_qtt_layout,
) -> std::result::Result<&'static InternalQttLayout, StatusCode> {
    match require_layout(layout) {
        Ok(layout) => Ok(layout),
        Err((code, msg)) => {
            set_last_error(&msg);
            Err(code)
        }
    }
}
```

If the borrow lifetime is tied to the input pointer, use:

```rust
fn require_layout_or_status<'a>(
    layout: *const t4a_qtt_layout,
) -> std::result::Result<&'a InternalQttLayout, StatusCode> {
    match require_layout(layout) {
        Ok(layout) => Ok(layout),
        Err((code, msg)) => {
            set_last_error(&msg);
            Err(code)
        }
    }
}
```

If neither compiles cleanly, use a local macro:

```rust
macro_rules! require_layout_or_return {
    ($layout:expr) => {
        match require_layout($layout) {
            Ok(layout) => layout,
            Err((code, msg)) => {
                set_last_error(&msg);
                return code;
            }
        }
    };
}
```

**Step 2: Replace repeated match blocks**

Replace each repeated pattern:

```rust
let layout_ref = match require_layout(layout) {
    Ok(layout) => layout,
    Err((code, msg)) => {
        set_last_error(&msg);
        return code;
    }
};
```

with:

```rust
let layout_ref = match require_layout_or_status(layout) {
    Ok(layout) => layout,
    Err(code) => return code,
};
```

or:

```rust
let layout_ref = require_layout_or_return!(layout);
```

Run:

```bash
rg -n "Err\\(\\(code, msg\\)\\)" crates/tensor4all-capi/src/quanticstransform.rs
cargo test --release -p tensor4all-capi quanticstransform
```

Expected: no repeated `Err((code, msg))` matches remain in this file; tests pass.

**Step 3: Add last-error regression if missing**

Add a test that calls one materialize function with a null layout pointer and asserts:

```rust
assert_ne!(status, T4A_SUCCESS);
let msg = last_error_string();
assert!(msg.contains("layout") || msg.contains("null"));
```

Use the existing test helper for `t4a_last_error_message` rather than adding a new C API helper.

Run:

```bash
cargo test --release -p tensor4all-capi quanticstransform
```

Expected: PASS.

**Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-capi/src/quanticstransform.rs
git commit -m "refactor: deduplicate qtransform layout errors"
```

Expected: commit succeeds.

## Task 11: Regenerate API Docs And Fix Drift

**Files:**
- Modify: `docs/api/*.md`
- Modify: stale docs or examples found by search

**Step 1: Regenerate API dump**

Run:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Expected: generated files update.

**Step 2: Search for stale public paths**

Run:

```bash
rg -n "tensor4all_tensorci::(CacheKey|CachedFunction|IndexSet|MultiIndex|Scalar)|tensor4all_core::storage|DenseLuKernel|LazyBlockRookKernel|PivotKernel|PivotSelectionCore|RandomScalar|TensorAccess" README.md docs crates --glob '!target/**'
```

Expected:

- No user-facing docs import `tcicore` types through `tensor4all_tensorci`.
- No docs advertise `tensor4all_core::storage`.
- Low-level MatrixLUCI kernel names appear only in internal implementation docs/tests if they remain visible at all.

**Step 3: Update docs if needed**

For each stale import, change it to the owning crate:

```rust
use tensor4all_tcicore::MultiIndex;
```

or remove the import entirely if it is no longer public.

Run:

```bash
cargo test --doc --release --workspace
```

Expected: PASS.

**Step 4: Commit**

```bash
cargo fmt --all
git add docs/api README.md docs/book crates
git commit -m "docs: update API references after cleanup"
```

Expected: commit succeeds. If no docs changed beyond `docs/api`, commit only generated docs.

## Task 12: Final Verification

**Files:**
- All changed files

**Step 1: Format and dry-check formatting**

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
```

Expected: PASS.

**Step 2: Run crate checks**

Run:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: PASS.

**Step 3: Run targeted release tests**

Run:

```bash
cargo test --release -p tensor4all-core
cargo test --release -p tensor4all-hdf5
cargo test --release -p tensor4all-capi
cargo test --release -p tensor4all-tensorbackend
cargo test --release -p tensor4all-itensorlike linsolve
cargo test --release -p tensor4all-treetn
cargo test --release -p tensor4all-tcicore
cargo test --release -p tensor4all-tensorci
cargo test --release -p tensor4all-simplett
cargo test --release -p tensor4all-quanticstci
cargo test --release -p tensor4all-partitionedtt
```

Expected: PASS.

**Step 4: Run doc and guide checks**

Run:

```bash
cargo test --doc --release --workspace
./scripts/test-mdbook.sh
cargo doc --workspace --no-deps
```

Expected: PASS.

**Step 5: Run full suite if time allows**

Run:

```bash
cargo nextest run --release --workspace
```

Expected: PASS.

**Step 6: Final public-surface audit**

Run:

```bash
rg -n "tensor4all_core::storage|pub mod storage|RandomScalar|TensorAccess|DenseLuKernel:|LazyBlockRookKernel:|PivotKernel<|pub use tensor4all_tcicore" crates docs/api README.md
```

Expected:

- No `tensor4all_core::storage` user-facing path.
- No root `RandomScalar` or `TensorAccess` re-export.
- No public where-clause leak of MatrixLUCI kernels.
- No `tensor4all-tensorci` re-export of `tcicore` foundational types.

**Step 7: Prepare PR summary**

Draft PR body:

```markdown
Closes #486.
Closes #487.
Closes #488.
Closes #489.
Closes #490.

Partially addresses #484 and #485 only where local cleanup removed existing unwrap/expect/panic patterns.

## Summary
- Removed downstream direct field access to core index/tensor representations.
- Trimmed accidental core and TCI public re-exports.
- Hid MatrixLUCI kernel internals behind public facade APIs.
- Replaced targeted `Result<_, String>` paths with typed/contextual errors.
- Added rustdoc for AnyScalar and deduplicated qtransform layout error handling.

## Verification
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --doc --release --workspace`
- `./scripts/test-mdbook.sh`
- `cargo nextest run --release --workspace`
```

Do not push or create the PR until the user approves.
