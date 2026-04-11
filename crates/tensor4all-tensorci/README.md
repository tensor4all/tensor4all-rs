# tensor4all-tensorci

Tensor Cross Interpolation algorithms. TCI2 (primary, actively maintained) and TCI1 (legacy).

## Key Types

- `crossinterpolate2()` — main entry point for two-site TCI
- `TCI2Options` — tolerance, pivot strategy, and convergence settings
- `CachedFunction` — wrapper that caches function evaluations to avoid redundant calls

## Example

```rust
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};
use tensor4all_simplett::AbstractTensorTrain;

// Function to interpolate: f(i, j) = (i + j + 1) as f64
let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;

// Run TCI on a 4x4 grid
let (tci, _ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    vec![4, 4],         // local dimensions
    vec![vec![3, 3]],   // initial pivot (0-indexed); pick where f is large
    TCI2Options {
        tolerance: 1e-10,
        ..Default::default()
    },
).unwrap();

// Verify convergence
assert!(*errors.last().unwrap() < 1e-10);

// Verify interpolation accuracy
let tt = tci.to_tensor_train().unwrap();
let val = tt.evaluate(&[2, 3]).unwrap();  // f(2, 3) = 2 + 3 + 1 = 6.0
assert!((val - 6.0).abs() < 1e-10);
```

## Documentation

- [User Guide: TCI](https://tensor4all.org/tensor4all-rs/guides/tci.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_tensorci/)
