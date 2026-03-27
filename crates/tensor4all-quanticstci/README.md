# tensor4all-quanticstci

High-level wrapper for Quantics Tensor Train (QTT) interpolation. Provides a user-friendly interface for interpolating functions in quantics representation by combining TCI with quantics grid transformations. This is a Rust port of QuanticsTCI.jl.

## Features

- **QuanticsTensorCI2**: Main class for quantics TCI results
- **Discrete grids**: Interpolate on integer grids
- **Continuous grids**: Interpolate on floating-point domains via `DiscretizedGrid`
- **Integration**: Compute integrals over the interpolated function
- Builder pattern for configuration

## Important Conventions

- **1-indexed grid indices**: All grid indices passed to and returned from this crate are **1-indexed** (following the Julia QuanticsTCI.jl convention). For example, the first grid point is `[1, 1]`, not `[0, 0]`.
- **Equal dimensions**: `quanticscrossinterpolate_discrete` and `quanticscrossinterpolate_from_arrays` currently require all dimensions to have the **same number of points** (same power of 2). Use `quanticscrossinterpolate` with an explicit `DiscretizedGrid` for non-uniform grids.
- **Power-of-2 grid sizes**: All grid dimensions must be powers of 2 (e.g., 4, 8, 16, 32, ...).

## Usage

### Discrete Grid

```rust
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

// Define a function on a 2D grid
// Discrete indices are 1-indexed and passed as `&[i64]`.
let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;

// Grid sizes (must be equal powers of 2)
let sizes = vec![16, 16];

// Run quantics TCI
let (qtci, ranks, errors) = quanticscrossinterpolate_discrete(
    &sizes,
    f,
    None,  // auto-select initial pivot
    QtciOptions::default().with_tolerance(1e-10)
)?;

// Evaluate at a point (1-indexed)
let value = qtci.evaluate(&[5, 10])?;  // approximately 15.0

// Compute sum over all grid points
let total = qtci.sum()?;
# Ok::<(), anyhow::Error>(())
```

### Continuous Grid with `DiscretizedGrid`

For interpolation on continuous domains, use `DiscretizedGrid` to define the mapping
from grid indices to physical coordinates:

```rust
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions,
};

// Build a 1D grid: 2^4 = 16 points on [0.0, 1.0]
let grid = DiscretizedGrid::builder(&[4])
    .with_lower_bound(&[0.0])
    .with_upper_bound(&[1.0])
    .build()
    .unwrap();

// Interpolate f(x) = x^2
let f = |x: &[f64]| x[0] * x[0];

let (qtci, _ranks, _errors) = quanticscrossinterpolate(
    &grid,
    f,
    None,
    QtciOptions::default(),
)?;

// integral() computes sum * grid_step (left Riemann sum, O(h) convergence)
let integral = qtci.integral()?;
# Ok::<(), anyhow::Error>(())
```

### Integration Semantics

`integral()` computes a **left Riemann sum**: `sum(f(x_i)) * product(step_sizes)`.
This has O(h) convergence where h is the grid spacing. The result depends on the
`include_endpoint` setting of the `DiscretizedGrid`.

For inherent discrete grids (no continuous domain), `integral()` returns the plain sum.

## License

MIT License
