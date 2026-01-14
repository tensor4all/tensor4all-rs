# Tensor Cross Interpolation (TCI)

**Tensor Cross Interpolation** is an algorithm for constructing tensor trains directly from a function, without ever forming the full tensor.

## The Problem

Given a function `f(i₁, i₂, ..., iₙ)` that can be evaluated at any point, construct a tensor train approximation with error below a specified tolerance.

## The Algorithm

TCI works by:
1. Selecting pivot points (multi-indices) that capture the essential structure
2. Building a tensor train from function evaluations at these pivots
3. Iteratively refining the pivots to minimize approximation error

The key advantage is that TCI only evaluates the function at a small subset of points, making it feasible for high-dimensional problems where the full tensor is intractable.

## Usage

```rust
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

// Define your function
let f = |idx: &Vec<usize>| -> f64 {
    // Your computation here
    idx.iter().map(|&i| i as f64).sum::<f64>()
};

// Set up interpolation
let local_dims = vec![10, 10, 10, 10];  // 10^4 = 10000 points
let initial_pivots = vec![vec![0, 0, 0, 0]];
let options = TCI2Options {
    tolerance: 1e-10,
    max_rank: Some(50),
    ..Default::default()
};

// Run TCI
let (tci, ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f, None, local_dims, initial_pivots, options
)?;

// Get the tensor train
let tt = tci.to_tensor_train()?;
```

## Parameters

- **tolerance**: Target relative error `||A - A_approx||_F / ||A||_F`
- **max_rank**: Maximum allowed bond dimension
- **max_iter**: Maximum number of sweeps

## When to Use TCI

TCI is most effective when:
- The underlying function has low-rank structure
- The full tensor is too large to store
- Function evaluations are relatively cheap

For functions without low-rank structure, TCI may require high bond dimensions, reducing its efficiency.
