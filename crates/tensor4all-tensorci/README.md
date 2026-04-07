# tensor4all-tensorci

Tensor Cross Interpolation algorithms. TCI2 (primary, actively maintained) and TCI1 (legacy).

## Key Types

- `crossinterpolate2()` — main entry point for two-site TCI
- `TCI2Options` — tolerance, pivot strategy, and convergence settings
- `CachedFunction` — wrapper that caches function evaluations to avoid redundant calls

## Example

```rust,ignore
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

// Function to interpolate: f(i, j) = (i + j + 1) as f64
let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;

// Run TCI on a 4x4 grid
let (tci, _ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    vec![4, 4],         // local dimensions
    vec![vec![1, 1]],   // initial pivot
    TCI2Options {
        tolerance: 1e-10,
        ..Default::default()
    },
)?;

assert!(tci.rank() > 0);
assert!(*errors.last().unwrap() < 1e-10);
```

## Documentation

- [User Guide: TCI](https://tensor4all.github.io/tensor4all-rs/guides/tci.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_tensorci/)
