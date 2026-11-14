# tensor4all-interpolativeqtt

Interpolative Quantics Tensor Train construction for tensor4all-rs.

This crate ports the tested public algorithms from
`InterpolativeQTT.jl` to Rust and returns `tensor4all-simplett`
`SimpleTensorTrain<T>` values, where `T` is the value type of the sampled
function: `f64` or `Complex64`.

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

For unknown localized features, `interpolate_patched_nd` splits failed dyadic
boxes into independent `InterpolativePatch<T>` values. Each patch carries its
fixed leading bits, the local train, and a checked absolute interpolation error
estimate including SVD truncation. Options bound rank, split depth, patch count
and Chebyshev stencil size. Sampling can miss features between validation
points, so the estimate is not a certificate for arbitrary functions.

The optional `parallel` feature adds `interpolate_patched_nd_on_lanes` for
caller-owned `hataori_tenferro::TenferroLanes`. Evaluator batches receive a
`threadteam::Exec` token from the same engine used for algebra and compression.
It selects faer; on macOS disable default features to avoid combining it with
Accelerate. A BLAS provider plus `parallel` is rejected.

The [guide tutorial](../../docs/book/src/tutorials/quantics-basics/interpolative-qtt.md#independent-dyadic-patches)
links an executable complex 2D construction, TreeTN conversion and patched
Fourier reference check.
