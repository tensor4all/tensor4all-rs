# matrixci

Low-rank matrix approximation via cross interpolation algorithms.

## Overview

This crate provides two matrix cross interpolation (CI) backends and a common trait:

| Type | Algorithm | Typical use |
|------|-----------|-------------|
| `MatrixLUCI` | Rank-Revealing LU with full pivoting | Batch decomposition of a fully available matrix |
| `MatrixACA` | Adaptive Cross Approximation | Incremental pivot addition with ACA heuristic |

Both implement the `AbstractMatrixCI` trait, so downstream code can be written generically.

## Modules

### `MatrixLUCI` — LU-based Cross Interpolation

A thin wrapper around `RrLU` (Rank-Revealing LU). Given a matrix `A`, it computes:

```
P_row * A * P_col = L * U
```

and exposes `left()` / `right()` matrices for a low-rank factorization `A ≈ left * right`.

```rust
use matrixci::{MatrixLUCI, AbstractMatrixCI, RrLUOptions, from_vec2d};

let m = from_vec2d(vec![
    vec![1.0, 2.0, 3.0],
    vec![2.0, 4.0, 6.0],
    vec![3.0, 5.0, 7.0],
]);

let opts = RrLUOptions { rel_tol: 1e-12, ..Default::default() };
let ci = MatrixLUCI::from_matrix(&m, Some(opts)).unwrap();

assert_eq!(ci.rank(), 2); // rank-2 approximation
let left = ci.left();     // shape (nrows, rank)
let right = ci.right();   // shape (rank, ncols)
```

Key methods beyond the trait:

- `from_matrix(a, options)` — construct from a dense matrix
- `from_rrlu(lu)` — construct from a precomputed `RrLU`
- `left()` / `right()` — low-rank factors (orthogonalization side depends on `RrLUOptions::left_orthogonal`)
- `pivot_errors()` — per-pivot error estimates
- `col_matrix()` / `row_matrix()` — raw L*U sub-blocks
- `cols_times_pivot_inv()` / `pivot_inv_times_rows()` — explicit pivot-inverse products

### `MatrixACA` — Adaptive Cross Approximation

Stores the approximation as `A ≈ U * diag(alpha) * V` and supports incremental pivot addition.

```rust
use matrixci::{MatrixACA, AbstractMatrixCI, from_vec2d};

let m = from_vec2d(vec![
    vec![1.0, 2.0, 3.0],
    vec![2.0, 4.0, 6.0],
    vec![3.0, 5.0, 7.0],
]);

let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
aca.add_best_pivot(&m).unwrap(); // add second pivot via ACA heuristic

println!("Rank: {}", aca.rank());
println!("Value at (1,2): {}", aca.evaluate(1, 2));
```

Key methods beyond the trait:

- `from_matrix_with_pivot(a, (i, j))` — construct with an initial pivot
- `add_pivot(a, (i, j))` — add a specific pivot
- `add_best_pivot(a)` — add pivot via ACA heuristic (max in last row of V, then max in corresponding column of U)
- `add_pivot_row(a, i)` / `add_pivot_col(a, j)` — add row or column independently
- `u()` / `v()` / `alpha()` — access internal factors
- `set_rows(cols, perm)` / `set_cols(rows, perm)` — update with permuted indices

### `RrLU` — Rank-Revealing LU Decomposition

The core algorithm behind `MatrixLUCI`. Performs LU decomposition with full pivoting and rank truncation.

```rust
use matrixci::{rrlu, RrLUOptions, from_vec2d};

let m = from_vec2d(vec![
    vec![1.0, 2.0, 3.0],
    vec![2.0, 4.0, 6.0],
    vec![3.0, 5.0, 7.0],
]);

let opts = RrLUOptions {
    max_rank: 10,
    rel_tol: 1e-12,
    abs_tol: 0.0,
    left_orthogonal: true, // L has 1s on diagonal
};
let lu = rrlu(&m, Some(opts)).unwrap();

println!("Pivots: {}", lu.npivots());
println!("Row perm: {:?}", lu.row_permutation());
println!("Col perm: {:?}", lu.col_permutation());
```

Also available: `rrlu_inplace` for in-place decomposition (avoids cloning the input matrix).

### `AbstractMatrixCI` Trait

Common interface for all CI backends:

```rust
pub trait AbstractMatrixCI<T: Scalar>: Sized {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn rank(&self) -> usize;
    fn row_indices(&self) -> &[usize];  // pivot row indices (I set)
    fn col_indices(&self) -> &[usize];  // pivot column indices (J set)
    fn is_empty(&self) -> bool;
    fn evaluate(&self, i: usize, j: usize) -> T;
    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<T>;
    fn row(&self, i: usize) -> Vec<T>;
    fn col(&self, j: usize) -> Vec<T>;
    fn to_matrix(&self) -> Matrix<T>;
    fn available_rows(&self) -> Vec<usize>;
    fn available_cols(&self) -> Vec<usize>;
    fn local_error(&self, a: &Matrix<T>, rows: &[usize], cols: &[usize]) -> Matrix<T>;
    fn find_new_pivot(&self, a: &Matrix<T>) -> Result<((usize, usize), T)>;
    fn find_new_pivot_in(&self, a: &Matrix<T>, rows: &[usize], cols: &[usize]) -> Result<((usize, usize), T)>;
}
```

### Scalar Types

Supported: `f64`, `f32`, `Complex64`, `Complex32`.

The `Scalar` trait provides a unified interface (`conj`, `abs_sq`, `from_f64`, etc.) and the `scalar_tests!` macro generates f64/Complex64 test variants from a single generic test function.

### Utilities

The `util` module provides `Matrix<T>` (row-major 2D matrix) and helper functions: `zeros`, `eye`, `from_vec2d`, `mat_mul`, `submatrix`, `get_row`, `get_col`, `transpose`, `append_row`, `append_col`, `swap_rows`, `swap_cols`, `a_times_b_inv`, `a_inv_times_b`, `submatrix_argmax`, etc.

## License

MIT License
