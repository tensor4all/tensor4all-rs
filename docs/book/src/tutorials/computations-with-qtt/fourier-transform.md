# Fourier Transform

This tutorial applies the quantics Fourier operator to a QTT representation of
a Gaussian. A Gaussian is a helpful first check because its Fourier transform
is also a Gaussian, so the result has a simple reference.

Runnable source: [`docs/tutorial-code/src/bin/qtt_fourier.rs`](../../../../tutorial-code/src/bin/qtt_fourier.rs)

## Key API Pieces

`quantics_fourier_operator` creates the operator. The tutorial binary then
converts the state to TreeTN, aligns site indices, and applies it via
`apply_linear_operator`.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstci::{
#     quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme,
# };
# use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
# use tensor4all_treetn::{apply_linear_operator, tensor_train_to_treetn, ApplyOptions};
let bits = 3;
let sizes = [8usize];
let options = QtciOptions::default()
    .with_nrandominitpivot(0)
    .with_unfoldingscheme(UnfoldingScheme::Interleaved)
    .with_verbosity(0);
let pivots = vec![vec![1_i64], vec![8]];

let (state, _, _) = quanticscrossinterpolate_discrete::<f64, _>(
    &sizes,
    |idx| (-((idx[0] as f64 - 4.0) / 2.0).powi(2)).exp(),
    Some(pivots),
    options,
)?;

let mut operator = quantics_fourier_operator(bits, FourierOptions::forward())?;
assert_eq!(operator.mpo.node_count(), bits);

let tt = state.tensor_train();
let (state_tn, _indices) = tensor_train_to_treetn(&tt)?;

operator.align_to_state(&state_tn)?;
let result = apply_linear_operator(&operator, &state_tn, ApplyOptions::naive())?;

assert!(result.node_count() > 0);
# Ok(())
# }
```

The plotted frequency axis is scaled back to physical units, so the curve can
be compared with the analytic Gaussian transform.

## What It Computes

The example builds a QTT for the input Gaussian, applies the Fourier operator,
and compares selected output values with the analytic transform.

![Fourier transform of the Gaussian](qtt_fourier_transform.png)

The next plots show the bond dimensions for the input QTT and the operator. The
operator dimensions are part of the cost of applying the transform.

![Bond dimensions for the Gaussian QTT](qtt_fourier_bond_dims.png)

![Bond dimensions for the Fourier operator](qtt_fourier_operator_bond_dims.png)
