# tensor4all-quanticstransform

Quantics transformation operators for tensor train methods.

This crate provides `LinearOperator` constructors for various transformations in Quantics representation. It is a Rust port of transformation functionality from [Quantics.jl](https://github.com/tensor4all/Quantics.jl).

## Available Transformations

| Operator | Description | Function |
|----------|-------------|----------|
| Flip | f(x) = g(2^R - x) | `flip_operator` |
| Shift | f(x) = g(x + offset) mod 2^R | `shift_operator` |
| Phase Rotation | f(x) = exp(i*theta*x) * g(x) | `phase_rotation_operator` |
| Cumulative Sum | y_i = sum_{j < i} x_j | `cumsum_operator` |
| Fourier Transform | Quantics Fourier Transform (QFT) | `quantics_fourier_operator` |
| Binary Operation | f(x, y) with first variable transformed to a*x + b*y | `binaryop_single_operator` |
| Affine Transform | y = A*x + b with rational coefficients | `affine_operator` |

All transformations return `LinearOperator` from `tensor4all-treetn` for consistent operator application to tensor train states.

## Usage

```rust
use tensor4all_quantics_transform::{
    flip_operator, shift_operator, phase_rotation_operator,
    cumsum_operator, quantics_fourier_operator,
    BoundaryCondition, FourierOptions,
};

// Create operators for 8-bit quantics representation
let r = 8;

// Flip operator: f(x) = g(2^R - x)
let flip_op = flip_operator(r, BoundaryCondition::Periodic)?;

// Shift operator: f(x) = g(x + 10) mod 2^R
let shift_op = shift_operator(r, 10, BoundaryCondition::Periodic)?;

// Phase rotation: f(x) = exp(i*pi/4*x) * g(x)
let phase_op = phase_rotation_operator(r, std::f64::consts::PI / 4.0)?;

// Cumulative sum
let cumsum_op = cumsum_operator(r)?;

// Fourier transform (forward)
let options = FourierOptions::forward();
let ft_op = quantics_fourier_operator(r, options)?;

// Affine transform: y = A*x + b with A = [[1, 1], [1, -1]], b = [0, 0]
use tensor4all_quantics_transform::{affine_operator, AffineParams};
use num_rational::Rational64;

let a = vec![
    Rational64::from_integer(1), Rational64::from_integer(1),
    Rational64::from_integer(1), Rational64::from_integer(-1),
];
let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
let params = AffineParams::new(a, b, 2, 2)?; // 2 outputs, 2 inputs
let bc = vec![BoundaryCondition::Periodic; 2];
let affine_op = affine_operator(r, &params, &bc)?;
```

## Reference

The implementation is based on:

- [Quantics.jl](https://github.com/tensor4all/Quantics.jl) - Julia implementation
- J. Chen and M. Lindsey, "Direct Interpolative Construction of the Discrete Fourier Transform as a Matrix Product Operator", arXiv:2404.03182 (2024) - QFT algorithm
