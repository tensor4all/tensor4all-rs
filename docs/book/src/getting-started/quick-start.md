# Quick Start

## Simple Tensor Train (MPS)

Create and manipulate tensor trains in Rust:

```rust
use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};

// Create a constant tensor train with local dimensions [2, 3, 4]
let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

// Evaluate at a specific multi-index
let value = tt.evaluate(&[0, 1, 2])?;

// Compute sum over all indices
let total = tt.sum();

// Compress with tolerance (rtol=1e-10, maxrank=20)
let compressed = tt.compressed(1e-10, Some(20))?;
```

## Tensor Cross Interpolation (TCI)

Construct a tensor train from a function using TCI:

```rust
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

// Define a function to interpolate
let f = |idx: &Vec<usize>| -> f64 {
    ((1 + idx[0]) * (1 + idx[1]) * (1 + idx[2])) as f64
};

// Perform cross interpolation
let local_dims = vec![4, 4, 4];
let initial_pivots = vec![vec![0, 0, 0]];
let options = TCI2Options { tolerance: 1e-10, ..Default::default() };

let (tci, ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f, None, local_dims, initial_pivots, options
)?;

// Convert to tensor train
let tt = tci.to_tensor_train()?;
println!("Rank: {}, Final error: {:.2e}", tci.rank(), errors.last().unwrap());
```

## What's Next?

- Learn about [Tensor Train](../core-concepts/tensor-train.md) concepts
- Explore [Tensor Cross Interpolation](../core-concepts/tci.md) in depth
- Check out the [Language Bindings](../bindings.md) for Julia and Python
