# Tensor Cross Interpolation

Tensor Cross Interpolation (TCI) approximates a high-dimensional function as a
tensor train by sampling only a small fraction of its entries. This crate
provides two levels of API:

- **`tensor4all-tensorci`** — low-level TCI algorithm working directly on
  integer indices.
- **`tensor4all-quanticstci`** — high-level wrapper that maps continuous or
  discrete grids to quantics representations before running TCI.

## Low-Level TCI (`tensor4all-tensorci`)

Use `crossinterpolate2` when you already know the local dimensions and want
direct control over the algorithm.

### Defining the function

The function must accept a slice of multi-indices and return the corresponding
values. Here is a simple 2D example — the sum of the (0-indexed) indices:

```rust,ignore
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

// Define a function to interpolate.
let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;

// Local dimension for each mode.
let local_dims = vec![4, 4];

// Initial pivot (required — at least one pivot must be provided).
let initial_pivots = vec![vec![1, 1]];

// Run TCI2 with a tight tolerance.
let (tci, ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,             // optional batch callback; None uses the scalar callback
    local_dims,
    initial_pivots,
    TCI2Options {
        tolerance: 1e-10,
        ..Default::default()
    },
)?;

assert!(tci.rank() > 0);
assert!(errors.last().is_some());
```

### Interpreting the results

`crossinterpolate2` returns a triple:

| Value | Type | Description |
|-------|------|-------------|
| `tci` | `TensorCI2<T>` | Completed TCI object; call `.to_tensor_train()` to get a `TensorTrain`. |
| `ranks` | `Vec<usize>` | Bond dimensions after each sweep. |
| `errors` | `Vec<f64>` | Error estimate after each sweep; convergence when last value is below tolerance. |

Convert to a tensor train for further manipulation:

```rust,ignore
let tt = tci.to_tensor_train();
```

## High-Level Quantics TCI (`tensor4all-quanticstci`)

The quantics representation encodes each grid index in binary and arranges the
bits across the tensor-train sites. This often yields much lower bond dimensions
than a naive encoding.

### Important conventions

- **1-indexed grid indices**: grid points are numbered starting at 1, matching
  the Julia `QuanticsTCI.jl` convention. For example, the first grid point of a
  2D grid is `[1, 1]`.
- **Equal dimensions**: `quanticscrossinterpolate_discrete` requires all
  dimensions to have the same number of points.
- **Power-of-2 grid sizes**: all grid dimensions must be powers of 2 (4, 8,
  16, 32, …).

### Discrete grid interpolation

Use `quanticscrossinterpolate_discrete` when your function is naturally defined
on an integer grid. Indices are passed as `&[i64]` and are 1-indexed.

```rust,ignore
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

// f(i, j) = i + j on a 16 × 16 grid (1-indexed).
let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;

let sizes = vec![16, 16];   // must be equal powers of 2

let (qtci, ranks, errors) = quanticscrossinterpolate_discrete(
    &sizes,
    f,
    None,   // auto-select initial pivot
    QtciOptions::default().with_tolerance(1e-10),
)?;

// Evaluate at a specific grid point (1-indexed).
let value = qtci.evaluate(&[5, 10])?;   // ≈ 15.0
assert!((value - 15.0).abs() < 1e-8);

// Sum over all grid points.
let total = qtci.sum()?;
```

### Continuous grid interpolation with `DiscretizedGrid`

For functions on continuous domains, build a `DiscretizedGrid` that maps grid
indices to physical coordinates. The number of quantics bits per dimension is
set via the builder.

```rust,ignore
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions,
};

// 1D grid: 2^4 = 16 points on [0.0, 1.0).
let grid = DiscretizedGrid::builder(&[4])   // &[bits_per_dim]
    .with_lower_bound(&[0.0])
    .with_upper_bound(&[1.0])
    .build()
    .unwrap();

// Interpolate f(x) = x^2.
let f = |x: &[f64]| x[0] * x[0];

let (qtci, _ranks, _errors) = quanticscrossinterpolate(
    &grid,
    f,
    None,
    QtciOptions::default(),
)?;
```

### Integration

`integral()` computes a **left Riemann sum**:

```
integral ≈ Σ f(xᵢ) × Δx
```

This has O(h) convergence where h is the grid spacing. The result depends on
whether `include_endpoint` is set on the `DiscretizedGrid`.

```rust,ignore
let integral = qtci.integral()?;
// For f(x) = x^2 on [0, 1) with 2^4 points, integral ≈ 1/3.
```

For discrete grids created without a continuous domain, `integral()` returns
the plain sum identical to `sum()`.
