# tensor4all-interpolativeqtt

Interpolative Quantics Tensor Train construction for tensor4all-rs.

This crate ports the tested public algorithms from
`InterpolativeQTT.jl` to Rust. Single-scale, multiscale, sparse and adaptive
constructors return a TreeTN and its physical indices in most-significant
bit order. Each index fuses one bit per axis, with axis zero varying fastest.
The TreeTN retains the sampled function's `f64` or `Complex64` dtype.

```rust
use tensor4all_interpolativeqtt::prelude::*;

let (tree, sites) = interpolate_single_scale(
    |x| (-x * x).exp(),
    -2.0,
    2.0,
    5,
    12,
    &InterpolativeQttOptions::default(),
).unwrap();

let value = tree.evaluate_point(&sites, &[0, 0, 0, 0, 0]).unwrap().real();
let expected = (-4.0_f64).exp();
assert!((value - expected).abs() < 1e-10);
```

For unknown localized features, `interpolate_patched_nd` splits failed dyadic
boxes into independent `InterpolativePatch` values. It runs in a
`CpuExecutionContext` the caller configures and passes in. Each patch carries
its fixed leading bits, the local compressed TreeTN and its ordered site indices, and an absolute per-value
error estimate of that train (interpolation residual plus compression change),
which decides acceptance. Options bound rank,
split depth and patch count. Sampling can miss features
between validation points, so the estimate is not a certificate for arbitrary
functions.

The optional `parallel` feature adds `interpolate_patched_nd_on_lanes` for
caller-owned `hataori_tenferro::TenferroLanes`. Evaluator batches receive a
`threadteam::Exec` token from the same engine used for algebra and compression.
Each lane's linear algebra runs on tenferro's faer-backed CPU stack within its
own thread budget.

`interpolate_chebyshev_patches` builds a plain piecewise Chebyshev interpolant
instead of a QTT: an accepted dyadic box keeps its Chebyshev-Lobatto node values
and nothing is compressed. A box that fails its sampled L-infinity residual
splits into `2^D` children, whose node values are exactly the parent's validation
points, so a child never resamples its node grid: it inherits those values and
samples only its own validation points. (`ChebyshevPatches`)
evaluates the result on the continuous box, and `ExtraValidation` optionally
samples points beyond the inherited grid for a residual peak that lies between
them. `parallel` adds `interpolate_chebyshev_patches_on_lanes` with the same
lane contract; the inherited node values travel inside each lane's work item.

The [guide tutorial](../../docs/book/src/tutorials/quantics-basics/interpolative-qtt.md#independent-dyadic-patches)
links an executable complex 2D construction, TreeTN embedding and patched
Fourier reference check.
