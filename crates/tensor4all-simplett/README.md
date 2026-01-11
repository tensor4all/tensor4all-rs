# tensor4all-simplett

Simple, efficient Tensor Train (MPS) implementation focused on practical computational methods. Provides multiple canonical forms and compression algorithms.

## Features

- **TensorTrain**: Basic MPS with `f64` and `Complex64` support
- **SiteTensorTrain**: Center-canonical MPS with specified orthogonality center
- **VidalTensorTrain**: Vidal canonical form with explicit singular values
- **TTCache**: Caching mechanism for fast repeated evaluation
- **Compression**: LU, CI, and SVD compression methods

## Usage

```rust
use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};

// Create a constant tensor train
let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

// Evaluate at a specific multi-index
let value = tt.evaluate(&[0, 1, 2])?;

// Compute sum over all indices
let total = tt.sum();

// Compress with tolerance
let compressed = tt.compressed(1e-10, Some(20))?;
```

## License

MIT License
