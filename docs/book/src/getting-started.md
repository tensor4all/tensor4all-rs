# Getting Started

## Prerequisites

You need the Rust toolchain. If you do not have it installed, follow the instructions at <https://rustup.rs/>.

## Adding tensor4all-rs to Your Project

tensor4all-rs is a collection of crates. Add the ones you need to your `Cargo.toml`:

```toml
[dependencies]
# Basic tensor train construction and manipulation
tensor4all-simplett = "0.1"

# Tensor Cross Interpolation (TCI)
tensor4all-tensorci = "0.1"

# Quantics TCI (combines quantics encoding with TCI)
tensor4all-quanticstci = "0.1"

# Tree tensor networks
tensor4all-treetn = "0.1"
```

You do not need to add all of them — only include the crates relevant to your use case.

## First Example: Tensor Trains

The following example uses `tensor4all-simplett` to create a constant tensor train, evaluate it at a specific index, and compress it.

```rust,ignore
use tensor4all_simplett::{AbstractTensorTrain, CompressionOptions, TensorTrain};

fn main() {
    // Create a constant tensor train with local dimensions [2, 3, 4].
    // Every entry of the represented tensor equals 1.0.
    let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

    // Evaluate at a specific multi-index.
    let value = tt.evaluate(&[0, 1, 2]).unwrap();
    assert!((value - 1.0).abs() < 1e-12);

    // Sum over all indices (2 * 3 * 4 = 24 elements, all 1.0).
    let total = tt.sum();
    assert!((total - 24.0).abs() < 1e-12);

    // Compress with a truncation tolerance.
    let options = CompressionOptions {
        tolerance: 1e-10,
        max_bond_dim: 20,
        ..Default::default()
    };
    let compressed = tt.compressed(&options).unwrap();
    assert!((compressed.sum() - 24.0).abs() < 1e-10);

    println!("sum = {}", compressed.sum());
}
```

Run it with:

```bash
cargo run
```

## Next Steps

- [Concepts](concepts.md) — learn about tensor trains, bond dimensions, and TCI before diving deeper.
- [Guides](guides/tensor-basics.md) — step-by-step walkthroughs for tensor basics, TCI, quantics transforms, and tree tensor networks.
