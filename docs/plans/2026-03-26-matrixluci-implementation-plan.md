# MatrixLUCI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a new `crates/matrixluci` substrate for dense/lazy LUCI kernels, then migrate `matrixci` and `tensor4all-tensorci` to depend on it.

**Architecture:** Add a new low-level crate that owns candidate-matrix sources, pivot kernels, dense/lazy implementations, and benchmarks. Keep pivot-only results as the primary contract, make factor reconstruction optional, then rebuild higher-level consumers on top of that substrate.

**Tech Stack:** Rust workspace crate, `faer`, `criterion`, `cargo nextest`, existing `matrixci`/`tensor4all-tensorci` tests.

---

### Task 1: Scaffold `matrixluci` crate and core types

**Files:**
- Create: `crates/matrixluci/Cargo.toml`
- Create: `crates/matrixluci/src/lib.rs`
- Create: `crates/matrixluci/src/error.rs`
- Create: `crates/matrixluci/src/scalar.rs`
- Create: `crates/matrixluci/src/types.rs`
- Create: `crates/matrixluci/src/source.rs`
- Modify: `Cargo.toml`
- Test: `crates/matrixluci/src/source/tests.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn dense_source_get_block_is_column_major() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let mut out = [0.0; 4];
    src.get_block(&[0, 1], &[0, 1], &mut out);
    assert_eq!(out, [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn dense_source_get_block_uses_cross_product_for_noncontiguous_indices() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3);
    let mut out = [0.0; 4];
    src.get_block(&[1, 0], &[2, 0], &mut out);
    assert_eq!(out, [6.0, 3.0, 4.0, 1.0]);
}

#[test]
fn scalar_get_delegates_to_get_block() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    assert_eq!(src.get(1, 0), 3.0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p matrixluci`
Expected: FAIL because package `matrixluci` does not exist yet

**Step 3: Write minimal implementation**

```rust
pub trait CandidateMatrixSource<T: Scalar> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    /// Fill `out` with the cross-product A[rows, cols] in column-major order.
    /// `out.len()` must equal `rows.len() * cols.len()`.
    fn get_block(&self, rows: &[usize], cols: &[usize], out: &mut [T]);

    fn get(&self, row: usize, col: usize) -> T {
        let mut out = [T::zero(); 1];
        self.get_block(&[row], &[col], &mut out);
        out[0]
    }
}

pub struct DenseMatrixSource<'a, T> {
    data: &'a [T],
    nrows: usize,
    ncols: usize,
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p matrixluci`
Expected: PASS for source semantics tests

**Step 5: Commit**

```bash
git add Cargo.toml crates/matrixluci
git commit -m "feat: add matrixluci crate scaffold"
```

### Task 2: Add pivot-only kernel interfaces

**Files:**
- Modify: `crates/matrixluci/src/lib.rs`
- Modify: `crates/matrixluci/src/types.rs`
- Create: `crates/matrixluci/src/kernel.rs`
- Test: `crates/matrixluci/src/kernel/tests.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn pivot_selection_core_stores_rank_and_indices() {
    let selection = PivotSelectionCore {
        row_indices: vec![0, 2],
        col_indices: vec![1, 3],
        pivot_errors: vec![1e-3, 1e-6],
        rank: 2,
    };
    assert_eq!(selection.rank, 2);
    assert_eq!(selection.row_indices, vec![0, 2]);
}

#[test]
fn pivot_kernel_options_no_truncation_uses_canonical_values() {
    let opts = PivotKernelOptions::no_truncation();
    assert_eq!(opts.rel_tol, 0.0);
    assert_eq!(opts.abs_tol, 0.0);
    assert_eq!(opts.max_rank, usize::MAX);
    assert!(opts.left_orthogonal);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p matrixluci kernel::tests::pivot_selection_core_stores_rank_and_indices`
Expected: FAIL because `PivotSelectionCore` / `PivotKernelOptions` / `PivotKernel` are missing

**Step 3: Write minimal implementation**

```rust
pub struct PivotKernelOptions {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub max_rank: usize,
    pub left_orthogonal: bool,
}

pub struct PivotSelectionCore {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub pivot_errors: Vec<f64>,
    pub rank: usize,
}

impl PivotKernelOptions {
    pub fn no_truncation() -> Self {
        Self {
            rel_tol: 0.0,
            abs_tol: 0.0,
            max_rank: usize::MAX,
            left_orthogonal: true,
        }
    }
}

pub trait PivotKernel<T: Scalar> {
    fn factorize<S: CandidateMatrixSource<T>>(
        &self,
        source: &S,
        options: &PivotKernelOptions,
    ) -> crate::Result<PivotSelectionCore>;
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p matrixluci`
Expected: PASS including kernel-type tests

**Step 5: Commit**

```bash
git add crates/matrixluci/src/lib.rs crates/matrixluci/src/types.rs crates/matrixluci/src/kernel.rs
git commit -m "feat: add matrixluci kernel interfaces"
```

### Task 3: Implement dense no-truncation kernel on top of `faer`

**Files:**
- Create: `crates/matrixluci/src/dense.rs`
- Modify: `crates/matrixluci/src/lib.rs`
- Modify: `crates/matrixluci/src/source.rs`
- Test: `crates/matrixluci/src/dense/tests.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn dense_kernel_recovers_identity_pivots() {
    let src = DenseMatrixSource::from_column_major(&[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ], 3, 3);
    let kernel = DenseFaerLuKernel::default();
    let out = kernel.factorize(&src, &PivotKernelOptions::no_truncation()).unwrap();
    assert_eq!(out.rank, 3);
    assert_eq!(out.row_indices.len(), 3);
    assert_eq!(out.col_indices.len(), 3);
}

#[test]
fn dense_kernel_reports_zero_rank_for_zero_matrix() {
    let src = DenseMatrixSource::from_column_major(&[0.0; 9], 3, 3);
    let kernel = DenseFaerLuKernel::default();
    let out = kernel.factorize(&src, &PivotKernelOptions::no_truncation()).unwrap();
    assert_eq!(out.rank, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p matrixluci dense::tests`
Expected: FAIL because `DenseFaerLuKernel` and `no_truncation()` are not implemented

**Step 3: Write minimal implementation**

```rust
#[derive(Default)]
pub struct DenseFaerLuKernel;

impl PivotKernel<f64> for DenseFaerLuKernel {
    fn factorize<S: CandidateMatrixSource<f64>>(
        &self,
        source: &S,
        options: &PivotKernelOptions,
    ) -> Result<PivotSelectionCore> {
        // 1. Materialize source into a column-major dense buffer
        // 2. Run tensor4all-owned rrLU control flow using faer primitives
        // 3. Return pivot-only result
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p matrixluci dense::tests`
Expected: PASS for basic dense-kernel behavior

**Step 5: Commit**

```bash
git add crates/matrixluci/src/dense.rs crates/matrixluci/src/source.rs crates/matrixluci/src/lib.rs
git commit -m "feat: add dense faer-based matrixluci kernel"
```

### Task 4: Add truncation and pivot-error semantics

**Files:**
- Modify: `crates/matrixluci/src/dense.rs`
- Modify: `crates/matrixluci/src/types.rs`
- Test: `crates/matrixluci/src/dense/tests.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn dense_kernel_honors_max_rank() {
    let src = low_rank_test_source();
    let kernel = DenseFaerLuKernel::default();
    let opts = PivotKernelOptions { max_rank: 2, ..PivotKernelOptions::no_truncation() };
    let out = kernel.factorize(&src, &opts).unwrap();
    assert!(out.rank <= 2);
}

#[test]
fn dense_kernel_matches_legacy_last_pivot_error_semantics() {
    let src = legacy_rrlu_regression_source();
    let kernel = DenseFaerLuKernel::default();
    let full = kernel.factorize(&src, &PivotKernelOptions::no_truncation()).unwrap();
    let max_rank = kernel.factorize(
        &src,
        &PivotKernelOptions { max_rank: 2, ..PivotKernelOptions::default() },
    ).unwrap();
    let abs_tol = kernel.factorize(
        &src,
        &PivotKernelOptions { abs_tol: 0.5, ..PivotKernelOptions::default() },
    ).unwrap();
    assert!(full.pivot_errors.last().unwrap().abs() < 1e-14);
    assert!(*max_rank.pivot_errors.last().unwrap() > 0.0);
    assert!(*abs_tol.pivot_errors.last().unwrap() < 0.5);
}

#[test]
fn dense_kernel_matches_legacy_pivot_error_vector_layout() {
    let src = eye_test_source(2);
    let kernel = DenseFaerLuKernel::default();
    let out = kernel.factorize(&src, &PivotKernelOptions::no_truncation()).unwrap();
    assert_eq!(out.pivot_errors.len(), out.rank + 1);
    assert!((out.pivot_errors[0] - 1.0).abs() < 1e-14);
    assert!((out.pivot_errors[1] - 1.0).abs() < 1e-14);
    assert!(out.pivot_errors[2].abs() < 1e-14);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p matrixluci dense::tests::dense_kernel_honors_max_rank`
Expected: FAIL because truncation / pivot-error semantics are incomplete

**Step 3: Write minimal implementation**

```rust
// Keep `no_truncation()` canonical:
// rel_tol = 0.0, abs_tol = 0.0, max_rank = usize::MAX, left_orthogonal = true
```

Implement:
- rank stopping via `max_rank`
- tolerance stopping via `rel_tol` / `abs_tol`
- explicit zero-matrix handling so the first near-zero pivot returns rank 0
- exact parity with current `matrixci::RrLU::{pivot_errors,last_pivot_error}` behavior,
  including the current full-rank reset to zero and the current truncated-case semantics
- port the current regression intent from `crates/matrixci/src/matrixlu/tests/mod.rs`
  before removing legacy LUCI code

**Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p matrixluci dense::tests`
Expected: PASS with truncation-aware behavior

**Step 5: Commit**

```bash
git add crates/matrixluci/src/dense.rs crates/matrixluci/src/types.rs
git commit -m "feat: add truncation semantics to matrixluci"
```

### Task 5: Implement lazy source and block-rook kernel

**Files:**
- Create: `crates/matrixluci/src/lazy.rs`
- Create: `crates/matrixluci/src/block_rook.rs`
- Modify: `crates/matrixluci/src/lib.rs`
- Test: `crates/matrixluci/src/lazy/tests.rs`
- Test: `crates/matrixluci/src/block_rook/tests.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn lazy_source_get_block_batches_requests() {
    let calls = Arc::new(AtomicUsize::new(0));
    let src = LazyMatrixSource::new(4, 4, {
        let calls = calls.clone();
        move |rows, cols, out: &mut [f64]| {
            calls.fetch_add(1, Ordering::SeqCst);
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = (row + col) as f64;
                }
            }
        }
    });
    let mut out = [0.0; 4];
    src.get_block(&[0, 1], &[2, 3], &mut out);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn block_rook_kernel_matches_dense_kernel_on_unique_pivot_matrix() {
    let dense = DenseMatrixSource::from_column_major(unique_pivot_test_matrix(), 4, 4);
    let lazy = lazy_from_dense(unique_pivot_test_matrix(), 4, 4);
    let dense_out = DenseFaerLuKernel::default()
        .factorize(&dense, &PivotKernelOptions::no_truncation())
        .unwrap();
    let lazy_out = LazyBlockRookKernel::default()
        .factorize(&lazy, &PivotKernelOptions::no_truncation())
        .unwrap();
    assert_eq!(lazy_out.row_indices, dense_out.row_indices);
    assert_eq!(lazy_out.col_indices, dense_out.col_indices);
    assert_eq!(lazy_out.rank, dense_out.rank);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p matrixluci lazy::tests block_rook::tests`
Expected: FAIL because lazy source and block-rook kernel do not exist

**Step 3: Write minimal implementation**

```rust
pub struct LazyMatrixSource<T, F> {
    nrows: usize,
    ncols: usize,
    fill_block: F,
    _marker: PhantomData<T>,
}

pub struct LazyBlockRookKernel;
```

Implement:
- lazy block fill callback
- block-rook search loop over requested row/col subsets
- correctness test against dense kernel on a deterministic unique-pivot matrix
- pivot-only output

**Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p matrixluci`
Expected: PASS including lazy and block-rook tests

**Step 5: Commit**

```bash
git add crates/matrixluci/src/lazy.rs crates/matrixluci/src/block_rook.rs crates/matrixluci/src/lib.rs
git commit -m "feat: add lazy block-rook matrixluci kernel"
```

### Task 6: Add optional factor reconstruction helpers

**Files:**
- Create: `crates/matrixluci/src/factors.rs`
- Modify: `crates/matrixluci/src/lib.rs`
- Test: `crates/matrixluci/src/factors/tests.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn reconstruct_cross_factors_matches_selected_submatrices() {
    let src = DenseMatrixSource::from_column_major(test_matrix_data(), 4, 4);
    let selection = DenseFaerLuKernel::default()
        .factorize(&src, &PivotKernelOptions::default())
        .unwrap();
    let factors = CrossFactors::from_source(&src, &selection).unwrap();
    assert!(approx_eq_dense(
        &factors.pivot,
        &dense_gather(&src, &selection.row_indices, &selection.col_indices),
    ));
    assert!(approx_eq_dense(
        &factors.pivot_cols,
        &dense_gather(&src, &(0..src.nrows()).collect::<Vec<_>>(), &selection.col_indices),
    ));
    assert!(approx_eq_dense(
        &factors.pivot_rows,
        &dense_gather(&src, &selection.row_indices, &(0..src.ncols()).collect::<Vec<_>>()),
    ));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p matrixluci factors::tests`
Expected: FAIL because factor helpers do not exist

**Step 3: Write minimal implementation**

```rust
pub struct CrossFactors<T> {
    pub pivot: DenseOwnedMatrix<T>,
    pub pivot_cols: DenseOwnedMatrix<T>,
    pub pivot_rows: DenseOwnedMatrix<T>,
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p matrixluci`
Expected: PASS with optional factor reconstruction working

**Step 5: Commit**

```bash
git add crates/matrixluci/src/factors.rs crates/matrixluci/src/lib.rs
git commit -m "feat: add optional matrixluci factor reconstruction"
```

### Task 7: Rebuild `matrixci` on top of `matrixluci`

**Files:**
- Modify: `crates/matrixci/Cargo.toml`
- Modify: `crates/matrixci/src/lib.rs`
- Modify or delete: `crates/matrixci/src/matrixaca.rs`
- Modify: `crates/matrixci/src/matrixci.rs`
- Modify: `crates/matrixci/src/matrixluci.rs`
- Modify: `crates/matrixci/src/matrixlu.rs`
- Modify or delete: `crates/matrixci/src/util.rs`
- Modify or delete: `crates/matrixci/src/matrixaca/tests/mod.rs`
- Test: `crates/matrixci/src/matrixci/tests/mod.rs`
- Test: `crates/matrixci/src/matrixluci/tests/mod.rs`
- Test: `crates/matrixci/src/matrixlu/tests/mod.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn matrixci_can_wrap_matrixluci_selection() {
    let matrix = test_matrix();
    let ci = MatrixCI::from_dense(&matrix, MatrixCIOptions::default()).unwrap();
    assert!(!ci.row_indices().is_empty());
    assert!(!ci.col_indices().is_empty());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p matrixci`
Expected: FAIL once `matrixci` is switched to the new dependency but not yet adapted

**Step 3: Write minimal implementation**

Implement:
- `matrixci` dependency on `matrixluci`
- `MatrixCI` as a higher-level result layer around pivot-only substrate output
- dropping `MatrixACA` is acceptable in this breaking refactor
- remove old row-major assumptions from public LUCI path

**Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p matrixci`
Expected: PASS for all migrated `matrixci` tests

**Step 5: Commit**

```bash
git add crates/matrixci
git commit -m "refactor: rebuild matrixci on top of matrixluci"
```

### Task 8: Migrate `tensor4all-tensorci` to direct `matrixluci` use

**Files:**
- Modify: `crates/tensor4all-tensorci/Cargo.toml`
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`
- Modify: `crates/tensor4all-tensorci/src/lib.rs`
- Test: `crates/tensor4all-tensorci/src/tensorci2/tests/mod.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn tensorci2_still_converges_with_matrixluci_backend() {
    let local_dims = vec![4, 4, 4];
    let f = |idx: &Vec<usize>| -> f64 { (idx[0] + idx[1] + idx[2]) as f64 };
    let (tci, _ranks, errors) =
        crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
            f,
            None,
            local_dims,
            vec![vec![0, 0, 0]],
            TCI2Options::default(),
        )
        .unwrap();
    assert!(tci.rank() > 0);
    assert!(!errors.is_empty());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-tensorci`
Expected: FAIL once old `matrixci::MatrixLUCI` usage is removed

**Step 3: Write minimal implementation**

Implement:
- direct dependency on `matrixluci`
- map existing TCI2 pivot-update flow onto `PivotKernel` / `PivotSelectionCore`
- keep current algorithm behavior and tests intact

**Step 4: Run tests to verify they pass**

Run: `cargo nextest run --release -p tensor4all-tensorci`
Expected: PASS for `tensorci2` test suite

**Step 5: Commit**

```bash
git add crates/tensor4all-tensorci
git commit -m "refactor: move tensorci to matrixluci substrate"
```

### Task 9: Add benchmark suite and verify acceptance criteria

**Files:**
- Create: `crates/matrixluci/benches/dense_vs_faer.rs`
- Create: `crates/matrixluci/benches/lazy_block_rook.rs`
- Create: `crates/matrixluci/benches/end_to_end_chain_tci.rs`
- Modify: `crates/matrixluci/Cargo.toml`
- Modify: `docs/plans/2026-03-26-matrixluci-design.md`

**Step 1: Write the benchmark targets**

Implement benchmark cases for:
- `32x32`
- `64x64`
- `100x100`
- `128x128`

Dense benchmark must compare:
- direct `faer` full-pivoting LU
- `matrixluci` dense no-truncation kernel

Also add:
- lazy/block-rook benchmark cases with cheap and expensive callbacks
- end-to-end chain TCI regression benchmark

**Step 2: Run benchmarks and capture the baseline**

Run: `cargo bench -p matrixluci --bench dense_vs_faer`
Expected: criterion report generated for both implementations

**Step 3: Tune until acceptance criteria are met**

Acceptance rule:

```text
median(matrixluci_dense_no_truncation) <= 1.05 * median(faer_direct_full_piv_lu)
```

for the benchmark sizes above.

Record the observed medians in the design doc or issue comment as the manual acceptance gate for this task.
CI performance gating is explicitly out of scope for this refactor.

**Step 4: Run verification suite**

Run:
- `cargo nextest run --release -p matrixluci`
- `cargo nextest run --release -p matrixci`
- `cargo nextest run --release -p tensor4all-tensorci`
- `cargo fmt --all`

Expected: all pass, benchmarks recorded, formatting clean

**Step 5: Commit**

```bash
git add crates/matrixluci docs/plans/2026-03-26-matrixluci-design.md
git commit -m "bench: validate matrixluci against faer baseline"
```
