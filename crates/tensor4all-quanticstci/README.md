# tensor4all-quanticstci

High-level wrapper for Quantics Tensor Train (QTT) interpolation. Provides a user-friendly interface for interpolating functions in quantics representation by combining TCI with quantics grid transformations. This is a Rust port of QuanticsTCI.jl.

## Features

- **QuanticsTensorCI2**: Main class for quantics TCI results
- **Discrete grids**: Interpolate on integer grids
- **Continuous grids**: Interpolate on floating-point domains
- **Integration**: Compute integrals over the interpolated function
- Builder pattern for configuration

## Usage

```rust
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

// Define a function on a 2D grid
let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;

// Grid sizes (powers of 2)
let sizes = vec![16, 16];

// Run quantics TCI
let (qtci, ranks, errors) = quanticscrossinterpolate_discrete(
    &sizes,
    f,
    None,  // auto-select initial pivot
    QtciOptions::default().with_tolerance(1e-10)
)?;

// Evaluate at a point
let value = qtci.evaluate(&[5, 10])?;  // approximately 15.0

// Compute sum over all grid points
let total = qtci.sum();
```

## License

MIT License
