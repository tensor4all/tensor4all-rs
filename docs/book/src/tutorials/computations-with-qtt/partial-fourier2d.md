# 2D Partial Fourier Transform

A partial Fourier transform applies Fourier only along one coordinate of a
multivariate function. Here the function is `f(x, t)`, and only the `x`
direction is transformed. The `t` direction passes through unchanged.

Runnable source: [`docs/tutorial-code/src/bin/qtt_partial_fourier2d.rs`](../../../../tutorial-code/src/bin/qtt_partial_fourier2d.rs)

## Key API Pieces

For an interleaved two-variable QTT, the state nodes are ordered
`x0, t0, x1, t1, ...`. A one-dimensional Fourier MPO has only `x` nodes, so the
operator nodes must be renamed onto the even state nodes before the operator is
expanded with identity tensors on the `t` nodes. The runnable source linked
above performs that expansion in `transform_x_dimension`. The source function
is `f(x, t) = exp(-x^2 / 2) * cos(2πt)`.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstci::{
#     quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
# };
# use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
# use tensor4all_simplett::AbstractTensorTrain;
let bits = 7;
let grid = DiscretizedGrid::builder(&[bits, bits])
    .with_variable_names(&["x", "t"])
    .with_lower_bound(&[-4.0, 0.0])
    .with_upper_bound(&[4.0, 1.0])
    .include_endpoint(true)
    .with_unfolding_scheme(UnfoldingScheme::Interleaved)
    .build()?;

let f = |coords: &[f64]| -> f64 {
    let x = coords[0];
    let t = coords[1];
    (-0.5 * x * x).exp() * (2.0 * std::f64::consts::PI * t).cos()
};
let options = QtciOptions::default()
    .with_unfoldingscheme(UnfoldingScheme::Interleaved)
    .with_verbosity(0);

let (state, _ranks, _errors) = quanticscrossinterpolate(&grid, f, None, options)?;

let operator = quantics_fourier_operator(bits, FourierOptions::forward())?;
assert_eq!(operator.mpo.node_count(), bits);

let x_site_mapping: Vec<_> = (0..bits).map(|site| (site, 2 * site)).collect();

assert_eq!(state.tensor_train().len(), 2 * bits);
assert_eq!(x_site_mapping.len(), bits);
assert_eq!(x_site_mapping[0], (0, 0));
assert_eq!(x_site_mapping[bits - 1], (bits - 1, 2 * (bits - 1)));
# Ok(())
# }
```

The full source then renames the operator nodes with this mapping, expands the
operator with identity tensors on the odd `t` nodes, aligns the resulting
operator to the state, and applies it. Passing `None` for `initial_pivots` is
the best starting point for tutorial code because it keeps QTCI on its default
initialization path. Explicit pivot lists are a later tuning tool for cases
where you already know important grid points.

## What It Computes

The example builds an interleaved two-dimensional QTT, applies a one-dimensional
Fourier operator to the x-sites, and compares the result with an analytic
partial transform.

![Partial Fourier values](qtt_partial_fourier2d_values.png)

![Partial Fourier error](qtt_partial_fourier2d_error.png)

Only the x-sites receive the operator, so the implementation must map the
one-dimensional operator nodes onto the even nodes of the interleaved state.

![Bond dimensions for the partial Fourier result](qtt_partial_fourier2d_bond_dims.png)
