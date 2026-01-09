# tensor4all-tensorci

Tensor Cross Interpolation (TCI) algorithms for efficiently approximating high-dimensional tensors as tensor trains. Provides both one-site and two-site TCI methods.

## Features

- **TensorCI1**: One-site TCI algorithm
- **TensorCI2**: Two-site TCI algorithm (more accurate)
- **CachedFunction**: Wrapper for caching function evaluations
- **IndexSet**: Efficient management of pivot index sets
- Supports both `f64` and `Complex64` scalar types

## Usage

```rust
use tensor4all_tensorci::{crossinterpolate1, TCI1Options};

// Define a function to interpolate
let f = |idx: &[usize]| (idx[0] + idx[1] + 1) as f64;

// Local dimensions for each index
let local_dims = vec![4, 4];

// Run TCI
let (tci, ranks, errors) = crossinterpolate1(
    f,
    local_dims,
    vec![1, 1],  // initial pivot
    TCI1Options::default().with_tolerance(1e-10)
)?;

// Evaluate the approximation
let value = tci.evaluate(&[2, 3])?;
```

## License

MIT License
