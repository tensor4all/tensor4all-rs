# Definite Integrals

After a QTT has been built on a physical interval, it can approximate an
integral by summing its grid values and multiplying by the grid spacing. For a
smooth function this is a compact way to keep the sampled values and an
integral estimate together.

Runnable source: [`docs/tutorial-code/src/bin/qtt_integral.rs`](../../../../tutorial-code/src/bin/qtt_integral.rs)

## Key API Pieces

`integral()` is available when the QTT was built from a `DiscretizedGrid`.
This example uses the same target function as the interval tutorial,
`f(x) = x^2` on `[-1, 2]`.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstci::{quanticscrossinterpolate, DiscretizedGrid, QtciOptions};
let grid = DiscretizedGrid::builder(&[7])
    .with_lower_bound(&[-1.0])
    .with_upper_bound(&[2.0])
    .include_endpoint(true)
    .build()?;

let f = |coords: &[f64]| -> f64 {
    let x = coords[0];
    x.powi(2)
};
let options = QtciOptions::default()
    .with_nrandominitpivot(3)
    .with_verbosity(0);
let (qtt, _ranks, _errors) = quanticscrossinterpolate(&grid, f, None, options)?;

let integral = qtt.integral()?;
let exact = 3.0;
assert!((integral - exact).abs() < 8e-2);
# Ok(())
# }
```

For non-constant functions, compare the result against an analytic integral or
a trusted high-resolution reference.

## What It Computes

The tutorial builds the same interval QTT as before and calls `integral()` on
it. The plot below comes from the bit-depth sweep and shows how the integral
error changes as the grid is refined.

![Integral error over bit depth](qtt_integral_sweep.png)

The integral is still a grid approximation. More bits give more grid points,
but they can also increase the work needed to build the QTT.
