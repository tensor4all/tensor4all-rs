# TensorCI1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port public legacy `TensorCI1` and `crossinterpolate1` from `TensorCrossInterpolation.jl`.

**Architecture:** Add a focused `tensorci1` module in `tensor4all-tensorci`. Keep `MatrixCI` as a private helper, use solve-based pivot-block operations, and convert the final state to `tensor4all-simplett::TensorTrain` for repeated evaluation.

**Tech Stack:** Rust, `tensor4all-tensorci`, `tensor4all-tcicore`, `tensor4all-tensorbackend::Matrix`, `tensor4all-simplett`.

---

## File Structure

- Create `crates/tensor4all-tensorci/src/tensorci1.rs`: public TCI1 state, options, entry point, and helpers.
- Create `crates/tensor4all-tensorci/src/tensorci1/matrix_ci.rs`: private `MatrixCI` helper.
- Create `crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs`: TCI1 tests.
- Modify `crates/tensor4all-tensorci/src/lib.rs`: expose `tensorci1`, `TensorCI1`, `TCI1Options`, `crossinterpolate1`.
- Modify `docs/api/tensor4all_tensorci.md`: regenerate via `api-dump`.

### Task 1: Add Public API Shell And Failing Tests

**Files:**
- Create: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Create: `crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs`
- Modify: `crates/tensor4all-tensorci/src/lib.rs`

- [ ] **Step 1: Add the public shell**

```rust
//! TensorCI1 - legacy one-site Tensor Cross Interpolation algorithm.

use crate::error::{Result, TCIError};
use tensor4all_simplett::{TTScalar, TensorTrain};
use tensor4all_tcicore::{MatrixLuciScalar, MultiIndex, Scalar};

mod matrix_ci;

/// Configuration for [`crossinterpolate1`].
#[derive(Debug, Clone)]
pub struct TCI1Options {
    /// Relative convergence tolerance.
    pub tolerance: f64,
    /// Maximum number of sweeps.
    pub max_iter: usize,
    /// Tolerance for accepting one new local pivot.
    pub pivot_tolerance: f64,
    /// Whether to normalize pivot errors by the maximum sampled value.
    pub normalize_error: bool,
    /// Additional global pivots inserted before sweeps.
    pub additional_pivots: Vec<MultiIndex>,
}

impl Default for TCI1Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 200,
            pivot_tolerance: 1e-12,
            normalize_error: true,
            additional_pivots: Vec::new(),
        }
    }
}

/// Legacy one-site tensor cross interpolation state.
#[derive(Debug, Clone)]
pub struct TensorCI1<T: Scalar + TTScalar> {
    local_dims: Vec<usize>,
    pivot_errors: Vec<f64>,
    max_sample_value: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T> TensorCI1<T>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
{
    /// Construct an empty TCI1 state.
    pub fn new(local_dims: Vec<usize>) -> Result<Self> {
        if local_dims.len() < 2 {
            return Err(TCIError::DimensionMismatch {
                message: "local_dims should have at least 2 elements".to_string(),
            });
        }
        Ok(Self {
            pivot_errors: vec![f64::INFINITY; local_dims.len() - 1],
            local_dims,
            max_sample_value: 0.0,
            _marker: std::marker::PhantomData,
        })
    }

    /// Number of sites.
    pub fn len(&self) -> usize {
        self.local_dims.len()
    }

    /// Whether the state has no sites.
    pub fn is_empty(&self) -> bool {
        self.local_dims.is_empty()
    }

    /// Local dimensions.
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Maximum sampled absolute value.
    pub fn max_sample_value(&self) -> f64 {
        self.max_sample_value
    }

    /// Pivot errors from the latest sweep.
    pub fn pivot_errors(&self) -> &[f64] {
        &self.pivot_errors
    }
}

/// Approximate a function with the legacy TCI1 algorithm.
pub fn crossinterpolate1<T, F>(
    _f: F,
    local_dims: Vec<usize>,
    _first_pivot: MultiIndex,
    _options: TCI1Options,
) -> Result<(TensorCI1<T>, Vec<usize>, Vec<f64>)>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
{
    let tci = TensorCI1::new(local_dims)?;
    Ok((tci, Vec::new(), Vec::new()))
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 2: Re-export from `lib.rs`**

```rust
pub mod tensorci1;
pub use tensorci1::{crossinterpolate1, TCI1Options, TensorCI1};
```

- [ ] **Step 3: Add failing behavior tests**

```rust
use super::*;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_crossinterpolate1_rank2_function() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
        f,
        vec![4, 4],
        vec![3, 3],
        TCI1Options {
            tolerance: 1e-12,
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    let tt = tci.to_tensor_train().unwrap();
    assert!((tt.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
}
```

- [ ] **Step 4: Run the failing test**

```bash
cargo test --release -p tensor4all-tensorci test_crossinterpolate1_rank2_function
```

Expected: compile failure because `to_tensor_train` does not exist, then assertion failures until the real algorithm is implemented.

### Task 2: Implement Private MatrixCI Helper

**Files:**
- Create: `crates/tensor4all-tensorci/src/tensorci1/matrix_ci.rs`

- [ ] **Step 1: Add helper structure and solve-based evaluation**

```rust
use crate::error::{Result, TCIError};
use tensor4all_tcicore::Scalar;
use tensor4all_tensorbackend::{solve_matrix, transpose, Matrix};

#[derive(Debug, Clone)]
pub(super) struct MatrixCI<T: Scalar> {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    left: Matrix<T>,
    right: Matrix<T>,
}

impl<T: Scalar> MatrixCI<T> {
    pub(super) fn new(
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        left: Matrix<T>,
        right: Matrix<T>,
    ) -> Result<Self> {
        if left.ncols() != right.nrows() {
            return Err(TCIError::DimensionMismatch {
                message: "MatrixCI left/right rank mismatch".to_string(),
            });
        }
        Ok(Self {
            row_indices,
            col_indices,
            left,
            right,
        })
    }

    pub(super) fn rank(&self) -> usize {
        self.left.ncols()
    }

    pub(super) fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    pub(super) fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub(super) fn evaluate(&self, row: usize, col: usize) -> T {
        let mut value = T::zero();
        for k in 0..self.rank() {
            value = value + self.left[[row, k]] * self.right[[k, col]];
        }
        value
    }
}
```

- [ ] **Step 2: Add helper tests**

Test that `evaluate` reconstructs a rank-1 outer product at selected entries and that mismatched factor shapes return `TCIError::DimensionMismatch`.

- [ ] **Step 3: Run helper tests**

```bash
cargo test --release -p tensor4all-tensorci matrix_ci
```

Expected: PASS.

### Task 3: Port TCI1 State And Pivot Updates

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci1/matrix_ci.rs`

- [ ] **Step 1: Replace placeholder fields with TCI1 state**

Store:

```rust
i_set: Vec<Vec<MultiIndex>>,
j_set: Vec<Vec<MultiIndex>>,
local_dims: Vec<usize>,
site_tensors: Vec<Tensor3<T>>,
pivots: Vec<Matrix<T>>,
pivot_errors: Vec<f64>,
max_sample_value: f64,
```

- [ ] **Step 2: Add initialization from first pivot**

Implement `TensorCI1::from_first_pivot(f, local_dims, first_pivot)`:

1. Validate `first_pivot.len() == local_dims.len()`.
2. Validate each coordinate is in range `0..local_dims[p]`.
3. Evaluate `f(&first_pivot)` and reject zero sample value.
4. Initialize `i_set[p] = first_pivot[..p]` and `j_set[p] = first_pivot[p+1..]`.
5. Fill initial site tensors and pivot matrices with function samples.

- [ ] **Step 3: Add solve helpers**

Add private helpers equivalent to Julia `AtimesBinv` and `AinvtimesB`:

```rust
fn right_solve<T: Scalar>(a: &Matrix<T>, p: &Matrix<T>) -> Result<Matrix<T>> {
    let solved_t = solve_matrix(&transpose(p), &transpose(a))?;
    Ok(transpose(&solved_t))
}

fn left_solve<T: Scalar>(p: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    solve_matrix(p, b).map_err(Into::into)
}
```

- [ ] **Step 4: Add `add_pivot_at_bond` and sweep logic**

Implement forward/backward sweeps that add one pivot per bond by evaluating local error candidates, updating I/J sets, site tensors, pivot matrix, pivot errors, and `max_sample_value`.

- [ ] **Step 5: Run rank-2 test**

```bash
cargo test --release -p tensor4all-tensorci test_crossinterpolate1_rank2_function
```

Expected: PASS.

### Task 4: Add TensorTrain Conversion And Evaluation

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs`

- [ ] **Step 1: Add `to_tensor_train`**

```rust
pub fn to_tensor_train(&self) -> Result<TensorTrain<T>> {
    TensorTrain::new(self.site_tensors.clone()).map_err(Into::into)
}
```

Call `right_solve` on each non-final site tensor matrix and its corresponding
pivot matrix before reshaping it back to `Tensor3<T>`. Leave the final site
tensor unchanged, matching Julia's `sitetensors(tci::TensorCI1)` behavior.

- [ ] **Step 2: Add single-point `evaluate`**

```rust
pub fn evaluate(&self, index: &[usize]) -> Result<T> {
    self.to_tensor_train()?.evaluate(index).map_err(Into::into)
}
```

- [ ] **Step 3: Add tests for `evaluate` and `to_tensor_train`**

Check every point of a `4 x 4` rank-2 function by materializing the tensor train once and evaluating all grid points through `tt.evaluate`.

- [ ] **Step 4: Run TCI1 tests**

```bash
cargo test --release -p tensor4all-tensorci tensorci1
```

Expected: PASS.

### Task 5: Port Julia Coverage

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs`

- [ ] **Step 1: Add complex Lorentz test**

Use `num_complex::Complex64` and function:

```rust
let f = |idx: &MultiIndex| {
    let norm: f64 = idx.iter().map(|&x| (x + 1) as f64).map(|x| x * x).sum();
    num_complex::Complex64::new(0.5, -1.0) / num_complex::Complex64::new(1.0 + norm, 0.0)
};
```

Assert sampled reconstruction error below `1e-8`.

- [ ] **Step 2: Add additional pivots test**

Use `TCI1Options { additional_pivots: vec![vec![3, 3, 3]], ..Default::default() }` and assert the final tensor train evaluates the added pivot accurately.

- [ ] **Step 3: Add invalid first-pivot tests**

Assert wrong length, out-of-range coordinate, and zero-valued initial sample return typed errors.

- [ ] **Step 4: Run crate tests**

```bash
cargo test --release -p tensor4all-tensorci tensorci1
```

Expected: PASS.

### Task 6: Docs, API Dump, And Commit

**Files:**
- Modify: `crates/tensor4all-tensorci/src/lib.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Modify: `docs/api/tensor4all_tensorci.md`

- [ ] **Step 1: Add rustdoc examples**

Add runnable examples for `TCI1Options`, `TensorCI1::new`, and `crossinterpolate1`. Each example must include assertions.

- [ ] **Step 2: Verify**

```bash
cargo fmt --all
cargo test --release -p tensor4all-tensorci tensorci1
cargo test --doc --release -p tensor4all-tensorci
cargo run -p api-dump --release -- . -o docs/api
```

- [ ] **Step 3: Commit**

```bash
git add crates/tensor4all-tensorci/src/lib.rs \
  crates/tensor4all-tensorci/src/tensorci1.rs \
  crates/tensor4all-tensorci/src/tensorci1/matrix_ci.rs \
  crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs \
  docs/api/tensor4all_tensorci.md
git commit -m "feat(tensorci): add public TensorCI1"
```
