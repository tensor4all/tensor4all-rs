# Tensor Cross Interpolation

Tensor Cross Interpolation (TCI) approximates a high-dimensional function as a tensor train by sampling only a small fraction of its entries. This crate provides two levels of API:

- `tensor4all-tensorci` for low-level TCI directly on integer indices.
- `tensor4all-quanticstci` for quantics interpolation on discrete or continuous grids.

## Low-Level TCI (`tensor4all-tensorci`)

Use `crossinterpolate2` when you already know the local dimensions and want direct control over the algorithm.

### Defining the function

The function must accept a slice of multi-indices and return the corresponding values. Here is a simple 2D example - the sum of the 0-indexed indices plus one.

```rust,ignore
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
let local_dims = vec![4, 4];
let initial_pivots = vec![vec![3, 3]];

let (tci, _ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    local_dims,
    initial_pivots,
    TCI2Options {
        tolerance: 1e-10,
        ..Default::default()
    },
)?;

assert!(*errors.last().unwrap() < 1e-10);

let tt = tci.to_tensor_train()?;
let value = tt.evaluate(&[2, 3])?;
assert!((value - 6.0).abs() < 1e-10);
assert!(tt.rank() >= 1);
```

### Interpreting the results

`crossinterpolate2` returns a triple:

| Value | Type | Description |
|-------|------|-------------|
| `tci` | `TensorCI2<T>` | Completed TCI object; call `.to_tensor_train()` to get a `TensorTrain`. |
| `ranks` | `Vec<usize>` | Bond dimensions after each sweep. |
| `errors` | `Vec<f64>` | Error estimate after each sweep; convergence when the last value is below tolerance. |

Convert to a tensor train for further manipulation:

```rust,ignore
use tensor4all_simplett::AbstractTensorTrain;

let tt = tci.to_tensor_train()?;
assert!(tt.rank() >= 1);
```

## High-Level Quantics TCI (`tensor4all-quanticstci`)

The quantics representation encodes each grid index in binary and arranges the bits across tensor-train sites. This often yields much lower bond dimensions than a naive encoding.

### Important conventions

- Indexing differs between the two APIs:
  - `crossinterpolate2` (low-level): indices and pivots are 0-indexed (`0..local_dim`)
  - `quanticscrossinterpolate_discrete` (high-level): grid indices are 1-indexed (`1..=grid_size`), matching the Julia `QuanticsTCI.jl` convention
- Equal dimensions: `quanticscrossinterpolate_discrete` requires all dimensions to have the same number of points.
- Power-of-2 grid sizes: all grid dimensions must be powers of 2 (4, 8, 16, 32, ...).

### Discrete grid interpolation

Use `quanticscrossinterpolate_discrete` when your function is naturally defined on an integer grid. Indices are passed as `&[i64]` and are 1-indexed.

```rust,ignore
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;
let sizes = vec![16, 16];

let (qtci, ranks, errors) = quanticscrossinterpolate_discrete(
    &sizes,
    f,
    None,
    QtciOptions::default().with_tolerance(1e-10),
)?;

assert!(*errors.last().unwrap() < 1e-10);
assert!(!ranks.is_empty());

let value = qtci.evaluate(&[5, 10])?;
assert!((value - 15.0).abs() < 1e-8);

let total = qtci.sum()?;
assert!((total - 4096.0).abs() < 1e-8);
```

### Continuous grid interpolation with `DiscretizedGrid`

For functions on continuous domains, build a `DiscretizedGrid` that maps grid indices to physical coordinates. The number of quantics bits per dimension is set via the builder.

```rust,ignore
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions,
};

let grid = DiscretizedGrid::builder(&[4])
    .with_lower_bound(&[0.0])
    .with_upper_bound(&[1.0])
    .build()
    .unwrap();

let f = |x: &[f64]| x[0] * x[0];

let (qtci, _ranks, errors) = quanticscrossinterpolate(
    &grid,
    f,
    None,
    QtciOptions::default(),
)?;

assert!(*errors.last().unwrap() < 1e-8);

let sample_idx = grid.grididx_to_quantics(&[9])?;
let sample_x = grid.quantics_to_origcoord(&sample_idx)?[0];
let value = qtci.evaluate(&[9])?;
assert!((value - sample_x * sample_x).abs() < 1e-8);
```

### Integration

`integral()` computes a left Riemann sum:

```text
integral ≈ Σ f(xᵢ) × Δx
```

This has O(h) convergence where h is the grid spacing. The result depends on whether `include_endpoint` is set on the `DiscretizedGrid`.

```rust,ignore
let integral = qtci.integral()?;
assert!((integral - 1.0 / 3.0).abs() < 5e-3);

let sum = qtci.sum()?;
assert!((sum * grid.grid_step()[0] - integral).abs() < 1e-12);
```

For discrete grids created without a continuous domain, `integral()` returns the plain sum identical to `sum()`.

## Practical Example: Multi-scale 1D Function

This section corresponds to the Julia [Quantics TCI of univariate function](https://tensor4all.org/T4APlutoExamples/quantics1d.html) notebook.

The function below mixes several length scales:

`f(x) = cos(x/B) * cos(x/(4*sqrt(5)*B)) * exp(-x^2) + 2*exp(-x)`

with `B = 2^-30` on `[0, ln(20))` using `R = 40` quantics bits.

```rust,ignore
use tensor4all_quanticstci::{quanticscrossinterpolate, DiscretizedGrid, QtciOptions};

let r = 40;
let b = 2f64.powi(-30);
let x_max = 20.0_f64.ln();
let grid = DiscretizedGrid::builder(&[r])
    .with_lower_bound(&[0.0])
    .with_upper_bound(&[x_max])
    .include_endpoint(false)
    .build()
    .unwrap();

let f = move |coords: &[f64]| {
    let x = coords[0];
    (x / b).cos() * (x / (4.0 * 5.0_f64.sqrt() * b)).cos() * (-x * x).exp()
        + 2.0 * (-x).exp()
};

let tol = 1e-10;
let (qtci, _ranks, errors) = quanticscrossinterpolate(
    &grid,
    f,
    None,
    QtciOptions::default()
        .with_tolerance(tol)
        .with_maxbonddim(64)
        .with_nrandominitpivot(8),
)?;

assert!(*errors.last().unwrap() < tol);

for &grid_idx in &[1_i64, 1_i64 << 10, 1_i64 << 20, 1_i64 << 30, 1_i64 << 40] {
    let quantics = grid.grididx_to_quantics(&[grid_idx])?;
    let x = grid.quantics_to_origcoord(&quantics)?[0];
    let exact = (x / b).cos() * (x / (4.0 * 5.0_f64.sqrt() * b)).cos() * (-x * x).exp()
        + 2.0 * (-x).exp();
    let got = qtci.evaluate(&[grid_idx])?;
    assert!((got - exact).abs() < 1e-6);
}

let integral = qtci.integral()?;
assert!((integral - 1.9).abs() < 1e-4);
```

## Multivariate (2D) Example

This section corresponds to the Julia [Quantics TCI of multivariate function](https://tensor4all.org/T4APlutoExamples/quantics2d.html) notebook.

```rust,ignore
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

let sizes = vec![256, 256];
let f = |idx: &[i64]| {
    let x = idx[0] as f64;
    let y = idx[1] as f64;
    (x / 24.0).cos() + (y / 17.0).cos() + 0.1 * ((x + y) / 13.0).sin()
};

let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete(
    &sizes,
    f,
    None,
    QtciOptions::default()
        .with_tolerance(1e-10)
        .with_maxbonddim(64)
        .with_nrandominitpivot(8),
)?;

assert!(*errors.last().unwrap() < 1e-10);

for &(i, j) in &[(1_i64, 1_i64), (17, 33), (128, 200), (256, 256)] {
    let exact = (i as f64 / 24.0).cos()
        + (j as f64 / 17.0).cos()
        + 0.1 * (((i + j) as f64) / 13.0).sin();
    let got = qtci.evaluate(&[i, j])?;
    assert!((got - exact).abs() < 1e-8);
}
```
