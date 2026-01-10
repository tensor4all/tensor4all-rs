# matrixci

Matrix Cross Interpolation (MCI) algorithms for low-rank matrix approximations. Provides multiple algorithms including standard cross interpolation, Adaptive Cross Approximation (ACA), and LU-based methods.

## Features

- **MatrixCI**: Standard matrix cross interpolation
- **MatrixACA**: Adaptive Cross Approximation variant
- **RrLU**: Rank-Revealing LU decomposition
- **MatrixLUCI**: LU-based Cross Interpolation
- Supports `f32`, `f64`, `Complex32`, and `Complex64`

## Usage

```rust
use matrixci::{MatrixCI, crossinterpolate, AbstractMatrixCI, from_vec2d};

// Create a matrix
let m = from_vec2d(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
    vec![7.0, 8.0, 9.0],
]);

// Perform cross interpolation
let ci = crossinterpolate(&m, None);
println!("Rank: {}", ci.rank());

// Access row and column indices
let rows = ci.row_indices();
let cols = ci.col_indices();

// Evaluate approximation at a point
let value = ci.evaluate(1, 2);
```

## License

MIT License
