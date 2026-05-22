# tensor4all-aci

Alternating Cross Interpolation (ACI) elementwise operations for tensor trains.

This crate ports the public Rust API for
[AlternatingCrossInterpolation.jl](https://github.com/tensor4all/AlternatingCrossInterpolation.jl),
originally authored by Marc Ritter and contributors.

## Key Types

- `elementwise()` - approximate an elementwise operation on tensor trains
- `elementwise_batched()` - batch-evaluation entry point for amortized function calls
- `ElementwiseBatch` - column-major batch of tensor-train input values
- `AciOptions` - tolerance, sweep, rank, and pivot-search controls
- `AciResult` - output tensor train and convergence diagnostics

## Example

```rust
use tensor4all_aci::{elementwise, AciOptions};
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

let a = TensorTrain::<f64>::constant(&[2, 3], 2.0);
let b = TensorTrain::<f64>::constant(&[2, 3], 4.0);

let result = elementwise(
    |xs: &[f64]| xs[0] * xs[1],
    &[a, b],
    &AciOptions::default(),
)
.unwrap();

assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
assert!((result.tensor_train.evaluate(&[1, 2]).unwrap() - 8.0).abs() < 1e-10);
```

## Citation

If you use ACI in research, please cite the original ACI paper:

> M. K. Ritter, "Fast elementwise operations on tensor trains with alternating
> cross interpolation", arXiv:2604.00037 (2026).

## Documentation

- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_aci/)
