# Multivariate Functions

A multivariate QTT stores a function such as `f(x, y)`. The sites can be
grouped by variable or interleaved. Grouped means all bits for one variable
come first; interleaved means the first bit of each variable appears before the
second bit of each variable. The best choice depends on the function.

Runnable source: [`docs/tutorial-code/src/bin/qtt_multivariate.rs`](../../../../tutorial-code/src/bin/qtt_multivariate.rs)

## Key API Pieces

Use one bit-depth entry per variable when building a `DiscretizedGrid`.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstci::{
#     quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
# };
let grid = DiscretizedGrid::builder(&[2, 2])
    .with_lower_bound(&[0.0, 0.0])
    .with_upper_bound(&[1.0, 1.0])
    .with_unfolding_scheme(UnfoldingScheme::Interleaved)
    .build()?;

let f = |coords: &[f64]| -> f64 { coords[0] + coords[1] };
let options = QtciOptions::default()
    .with_nrandominitpivot(0)
    .with_unfoldingscheme(UnfoldingScheme::Interleaved)
    .with_verbosity(0);
let pivots = vec![vec![1_i64, 1], vec![4, 4]];
let (qtt, _ranks, _errors) = quanticscrossinterpolate(&grid, f, Some(pivots), options)?;

assert!((qtt.evaluate(&[1, 1])? - 0.0).abs() < 1e-12);
# Ok(())
# }
```

The tutorial binary uses the same function with enough output data to compare
the two layouts visually.

## What It Computes

The example builds a two-dimensional QTT with both layouts and compares the
values, errors, and bond dimensions.

![Two-dimensional QTT values](qtt_multivariate_values.png)

![Two-dimensional QTT error](qtt_multivariate_error.png)

![Bond dimensions for grouped and interleaved layouts](qtt_multivariate_bond_dims.png)
