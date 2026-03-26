# tensor4all-tensorci

Tensor Cross Interpolation (TCI) algorithms for efficiently approximating high-dimensional tensors as tensor trains.

## Features

- **TensorCI2**: Primary two-site TCI algorithm
- **TensorCI1**: Legacy one-site TCI algorithm retained for compatibility
- **CachedFunction**: Wrapper for caching function evaluations
- **IndexSet**: Efficient management of pivot index sets
- Supports both `f64` and `Complex64` scalar types

## Status

- `TensorCI2` is the actively maintained path and uses `matrixluci` directly.
- `TensorCI1` remains available as legacy support and continues to rely on the older ACA-based matrix code.
- `PivotSearchStrategy::Rook` now uses lazy block-rook evaluation through batch callbacks.
- With `normalize_error = true`, `Rook` normalizes by the maximum observed sample value from the lazily requested entries rather than by a full-grid scan.

## Usage

```rust
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Define a function to interpolate
let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;

// Local dimensions for each index
let local_dims = vec![4, 4];

// Run TCI
let (tci, ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    local_dims,
    vec![vec![1, 1]],
    TCI2Options {
        tolerance: 1e-10,
        ..Default::default()
    },
)?;

assert!(tci.rank() > 0);
assert!(errors.last().is_some());
# Ok(())
# }
```

## License

MIT License
