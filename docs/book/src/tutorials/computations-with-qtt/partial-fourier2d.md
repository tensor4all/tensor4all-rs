# 2D Partial Fourier Transform

A partial Fourier transform applies Fourier only along one coordinate of a
multivariate function. Here the function is \(f(x, t)\), and only the \(x\)
direction is transformed. The \(t\) direction passes through unchanged.

Runnable source: [`docs/tutorial-code/src/bin/qtt_partial_fourier2d.rs`](../../../../tutorial-code/src/bin/qtt_partial_fourier2d.rs)

## What It Computes

The example builds an interleaved two-dimensional QTT, applies a one-dimensional
Fourier operator to the x-sites, and compares the result with an analytic
partial transform.

![Partial Fourier values](qtt_partial_fourier2d_values.png)

![Partial Fourier error](qtt_partial_fourier2d_error.png)

Only the x-sites receive the operator, so the implementation must map the
one-dimensional operator nodes onto the even nodes of the interleaved state.

![Bond dimensions for the partial Fourier result](qtt_partial_fourier2d_bond_dims.png)

## Key API Pieces

For an interleaved two-variable QTT, x-sites live at positions `0, 2, 4, ...`.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
let bits = 4;
let x_site_mapping: Vec<(usize, usize)> =
    (0..bits).map(|site| (site, 2 * site)).collect();

let operator = quantics_fourier_operator(bits, FourierOptions::forward())?;
assert_eq!(operator.mpo.node_count(), bits);
assert_eq!(x_site_mapping, vec![(0, 0), (1, 2), (2, 4), (3, 6)]);
# Ok(())
# }
```

The tutorial code renames the operator nodes with this mapping, then applies
the operator while leaving the t-sites in place.
