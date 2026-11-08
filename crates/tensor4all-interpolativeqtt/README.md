# tensor4all-interpolativeqtt

Interpolative Quantics Tensor Train construction for tensor4all-rs.

This crate ports the tested public algorithms from
`InterpolativeQTT.jl` to Rust and returns `tensor4all-simplett`
`SimpleTensorTrain<f64>` values.

`invert_qtt` recovers multiresolution node matrices from binary trains,
including compressed trains, using a two-sample linear blend at the finest
level. For an untouched single-scale train built with `tolerance = 0.0` and
`max_bond_dim = None`, use `invert_qtt_uncompressed` to recover the parent
interpolant at every finest-level node without the blend's extra error.

```rust
use tensor4all_interpolativeqtt::prelude::*;

let tt = interpolate_single_scale(
    |x| (-x * x).exp(),
    -2.0,
    2.0,
    5,
    12,
    &InterpolativeQttOptions::default(),
).unwrap();

let value = tt.evaluate(&[0, 0, 0, 0, 0]).unwrap();
let expected = (-4.0_f64).exp();
assert!((value - expected).abs() < 1e-10);
```
