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

> **Note:** A dedicated multivar Fourier API is not yet available. This example
> uses partial apply to achieve 2D transforms manually.

`apply_linear_operator` supports partial application when the operator nodes are
a subset of the state nodes — even when the operator nodes are **non-contiguous**
(e.g., interleaved quantics encoding). Identity tensors are automatically inserted
at intermediate nodes via Steiner tree expansion.

### Approach

For a 2D function in interleaved quantics encoding with R bits per variable,
the TreeTN has 2R sites: `[x₁, y₁, x₂, y₂, ..., xᵣ, yᵣ]` (node names
`0, 1, 2, 3, ..., 2R-1`).

To apply a 1D Fourier transform to the x-variable:

1. Build the 1D Fourier operator (R sites, node names 0..R-1)
2. Rename operator nodes to match x-variable sites: `0→0, 1→2, 2→4, ...`
3. Set input/output mappings from the state
4. Call `apply_linear_operator` — Steiner tree partial apply handles the gaps

```rust,ignore
use std::f64::consts::PI;
use tensor4all_quanticstci::{
    quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme,
};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetci::materialize::to_treetn;
use tensor4all_treetn::operator::{apply_linear_operator, ApplyOptions};
use tensor4all_treetn::Operator;

let r = 3;
let n = 1usize << r; // 8

// f(x, y) = cos(2π(x-1)/N), 1-indexed — depends only on x
let f = move |idx: &[i64]| -> f64 {
    let x = (idx[0] - 1) as f64;
    (2.0 * PI * x / n as f64).cos()
};

// Build QTT with interleaved encoding
let sizes = vec![n, n];
let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete::<f64, _>(
    &sizes, f, None,
    QtciOptions::default()
        .with_tolerance(1e-12)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved),
)?;
assert!(*errors.last().unwrap() < 1e-10);

// Convert to TreeTN (6 sites: 0,1,2,3,4,5)
let tci_state = qtci.tci();
let batch_eval = move |batch: tensor4all_treetci::GlobalIndexBatch<'_>|
    -> anyhow::Result<Vec<f64>>
{
    let mut values = Vec::with_capacity(batch.n_points());
    for p in 0..batch.n_points() {
        let mut x_val = 0usize;
        for bit in 0..r {
            x_val |= batch.get(2 * bit, p).unwrap() << bit;
        }
        values.push((2.0 * PI * x_val as f64 / n as f64).cos());
    }
    Ok(values)
};
let state = to_treetn(tci_state, batch_eval, Some(0))?;

// Build 1D Fourier operator (nodes 0,1,2)
let mut fourier_op = quantics_fourier_operator(
    r, FourierOptions { normalize: true, ..Default::default() },
)?;

// Rename nodes to x-variable sites: 0→0, 1→2, 2→4
// Use two-phase rename to avoid collisions
let offset = 1_000_000;
for i in 0..r {
    fourier_op.mpo.rename_node(&i, i + offset)?;
}
for i in 0..r {
    fourier_op.mpo.rename_node(&(i + offset), 2 * i)?;
}
// Update mappings
let mut new_input = std::collections::HashMap::new();
for (k, v) in fourier_op.input_mapping.drain() {
    new_input.insert(2 * k, v);
}
fourier_op.input_mapping = new_input;
let mut new_output = std::collections::HashMap::new();
for (k, v) in fourier_op.output_mapping.drain() {
    new_output.insert(2 * k, v);
}
fourier_op.output_mapping = new_output;

// Match operator's true indices to state's site indices
fourier_op.set_input_space_from_state(&state)?;
fourier_op.set_output_space_from_state(&state)?;

// Apply — Steiner tree inserts identity at y-sites {1, 3, 5}
let result = apply_linear_operator(
    &fourier_op, &state, ApplyOptions::default(),
)?;
assert_eq!(result.node_count(), 2 * r);
```

The same approach works for applying the Fourier transform to the y-variable
(rename operator nodes to `1, 3, 5` instead).
