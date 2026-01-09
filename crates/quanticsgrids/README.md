# quanticsgrids

Grid structures for efficient conversion between quantics representation, grid indices, and original coordinates. This is a Rust port of [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl).

## Features

- **InherentDiscreteGrid**: Low-level grid for integer coordinates
- **DiscretizedGrid**: High-level grid for continuous domains
- **UnfoldingScheme**: Fused (default) or Interleaved index ordering
- Efficient O(R * D) coordinate conversions

## Usage

```rust
use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};

// Create a 2D grid with 3 bits for x and 2 bits for y
let grid = DiscretizedGrid::builder(&[3, 2])
    .with_lower_bound(&[0.0, 0.0])
    .with_upper_bound(&[1.0, 2.0])
    .build()?;

// Convert coordinates to quantics indices
let quantics = grid.origcoord_to_quantics(&[0.5, 1.0])?;

// Convert back to coordinates
let coords = grid.quantics_to_origcoord(&quantics)?;

// Get local dimensions for TCI
let local_dims = grid.local_dimensions();
```

## Coordinate Conversions

| From | To | Method |
|------|-----|--------|
| Original coordinates | Quantics indices | `origcoord_to_quantics()` |
| Quantics indices | Original coordinates | `quantics_to_origcoord()` |
| Grid indices | Quantics indices | `grididx_to_quantics()` |
| Quantics indices | Grid indices | `quantics_to_grididx()` |

## License

MIT License
