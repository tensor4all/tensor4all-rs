# Quantum Fourier Transform

This guide corresponds to the Julia [Quantum Fourier Transform](https://tensor4all.org/T4APlutoExamples/qft.html) notebook.

## 1D QFT

We start from a one-dimensional function on `[0, 1)`, build a quantics tensor train, convert it to a `TreeTN`, apply the quantics Fourier operator, and compare the resulting coefficients against the analytic discrete Fourier transform.

```rust,ignore
use std::f64::consts::PI;

use num_complex::Complex64;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_quanticstci::{quanticscrossinterpolate, DiscretizedGrid, QtciOptions};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetci::materialize::to_treetn;
use tensor4all_treetn::{apply_linear_operator, ApplyOptions};

let r = 4;
let n = 1usize << r;
let a = 2.0;
let grid = DiscretizedGrid::builder(&[r])
    .with_lower_bound(&[0.0])
    .with_upper_bound(&[1.0])
    .include_endpoint(false)
    .build()
    .unwrap();

let f = move |coords: &[f64]| (-a * coords[0]).exp();
let (qtci, _ranks, errors) = quanticscrossinterpolate(
    &grid,
    f,
    None,
    QtciOptions::default().with_tolerance(1e-12),
)?;

assert!(*errors.last().unwrap() < 1e-10);

let batch_eval = move |batch: tensor4all_treetci::GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
    let mut values = Vec::with_capacity(batch.n_points());
    for p in 0..batch.n_points() {
        let quantics = (0..batch.n_sites())
            .map(|site| batch.get(site, p).unwrap() as i64 + 1)
            .collect::<Vec<_>>();
        let coords = grid.quantics_to_origcoord(&quantics).unwrap();
        values.push((-a * coords[0]).exp());
    }
    Ok(values)
};

let state = to_treetn(qtci.tci(), batch_eval, Some(0))?;
let fourier_op = quantics_fourier_operator(r, FourierOptions::default())?;
let result = apply_linear_operator(&fourier_op, &state, ApplyOptions::default())?;

let (index_ids, _) = result.all_site_index_ids()?;

let coefficient = |k: usize| -> Complex64 {
    let bits: Vec<usize> = (0..r).map(|shift| (k >> shift) & 1).collect();
    let values = ColMajorArrayRef::new(&bits, &[r, 1]);
    let vals = result.evaluate(&index_ids, values).unwrap();
    Complex64::new(vals[0].real(), vals[0].imag())
};

for k in [0usize, 1, 3] {
    let q = Complex64::new(-a / n as f64, -2.0 * PI * k as f64 / n as f64).exp();
    let exact = (Complex64::new(1.0, 0.0) - Complex64::new(-a, -2.0 * PI * k as f64).exp())
        / (Complex64::new(1.0, 0.0) - q)
        / (n as f64).sqrt();
    let got = coefficient(k);
    assert!((got - exact).norm() < 1e-8);
}
```

## 2D QFT via Partial Apply

A dedicated multivar Fourier API is not yet available.

`apply_linear_operator` supports partial application when the operator nodes are a subset of the state nodes, but the operator nodes must form a connected subtree. That means:

- If the x bits are stored contiguously, a 1D QFT can act on the x block and leave the y block untouched.
- If the state is stored as interleaved sites `[x1, y1, x2, y2, ...]`, the x sites are not connected, so the current operator application path does not support that layout directly.

The supported workaround is to regroup the state so the target variable occupies a contiguous block before applying the 1D Fourier operator.

```rust,ignore
use num_complex::Complex64;
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{apply_linear_operator, ApplyOptions};

let r = 4;
let x_fourier = quantics_fourier_operator(r, FourierOptions::default())?;

// Supported layout: x bits are contiguous, so the operator nodes form a connected subtree.
let result = apply_linear_operator(&x_fourier, &state_with_grouped_x_bits, ApplyOptions::default())?;

let vals = result.evaluate(&output_index_ids, output_values)?;
let got = Complex64::new(vals[0].real(), vals[0].imag());
let exact = Complex64::new(0.0, 0.0);
assert!((got - exact).norm() < 1e-8);
```

For an interleaved layout, regroup the x and y bits first, or wait for a dedicated multivariate Fourier constructor.
