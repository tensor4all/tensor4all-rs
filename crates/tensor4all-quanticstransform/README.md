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

## Applying Operators to TreeTN

Operators are applied to tensor networks using `apply_linear_operator` from `tensor4all-treetn`. The key concept is **index mapping**: operators define abstract site indices (0, 1, 2, ...), and you map your tensor network's indices to these.

### Basic Example: Applying to a Full MPS

```rust
use tensor4all_treetn::{apply_linear_operator, ApplyOptions, LinearOperator};
use tensor4all_core::TensorIndex;

// Create a shift operator for R=4 sites
let r = 4;
let shift_op = shift_operator(r, 3, BoundaryCondition::Periodic)?;

// Your TreeTN (e.g., converted from a TensorTrain)
let treetn: TreeTN<Complex64> = /* ... */;
let site_indices: Vec<DynIndex> = treetn.external_indices();

// Remap indices: connect your TreeTN's indices to the operator's input indices
let mut treetn_remapped = treetn;
for i in 0..r {
    let op_input = shift_op
        .get_input_mapping(&i)
        .expect("Missing input mapping")
        .true_index
        .clone();
    treetn_remapped = treetn_remapped
        .replaceind(&site_indices[i], &op_input)?;
}

// Apply the operator
let result = apply_linear_operator(&shift_op, &treetn_remapped, ApplyOptions::default())?;

// The result has new external indices (the operator's output indices)
let output_indices: Vec<DynIndex> = (0..r)
    .map(|i| shift_op.get_output_mapping(&i).unwrap().true_index.clone())
    .collect();
```

### Partial Application: Operating on a Subset of Indices

A powerful feature is applying an operator to only **some** indices of a tensor network. For example, if you have a 2D function f(x, y) represented as a TreeTN with indices for both x and y, you can apply a Fourier transform to only the x variable.

```rust
// Suppose we have a TreeTN representing f(x, y) with:
// - x_indices: [x0, x1, x2, x3] (4 sites for x variable)
// - y_indices: [y0, y1, y2, y3] (4 sites for y variable)
let r = 4;
let treetn_2d: TreeTN<Complex64> = /* ... */;
let x_indices: Vec<DynIndex> = /* indices for x variable */;
let y_indices: Vec<DynIndex> = /* indices for y variable */;

// Create Fourier operator for the x variable only
let fourier_x = quantics_fourier_operator(r, FourierOptions::forward())?;

// Remap ONLY the x indices to the operator
let mut treetn_remapped = treetn_2d;
for i in 0..r {
    let op_input = fourier_x
        .get_input_mapping(&i)
        .expect("Missing input mapping")
        .true_index
        .clone();
    treetn_remapped = treetn_remapped
        .replaceind(&x_indices[i], &op_input)?;
}
// Note: y_indices are NOT remapped, so they pass through unchanged

// Apply the operator - it acts only on the x indices
let result = apply_linear_operator(&fourier_x, &treetn_remapped, ApplyOptions::default())?;

// Result now represents F_x[f](k_x, y):
// - k_x indices: from fourier_x.get_output_mapping(i)
// - y indices: unchanged (still y_indices)
```

### Index Mapping Flow

```
Original TreeTN          Operator            Result TreeTN
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  site_idx_0  │───▶│  input_0     │    │  output_0    │
│  site_idx_1  │───▶│  input_1     │───▶│  output_1    │
│  site_idx_2  │───▶│  input_2     │    │  output_2    │
│  other_idx   │    │              │    │  other_idx   │ (passes through)
└──────────────┘    └──────────────┘    └──────────────┘
      │                                        │
      └──────── replaceind() ─────────────────►│
```

The `apply_linear_operator` function:
1. Contracts the operator's tensors with the TreeTN
2. Replaces input indices with output indices
3. Leaves unrelated indices unchanged

### Truncation Options

Control the accuracy/efficiency tradeoff when applying operators:

```rust
// Naive application (no truncation, exact but may be expensive)
let options = ApplyOptions::naive();

// ZipUp algorithm with truncation
let options = ApplyOptions::zipup()
    .with_max_bond_dim(64)
    .with_tolerance(1e-10);

// Fit algorithm for more control
let options = ApplyOptions::fit()
    .with_max_bond_dim(32)
    .with_max_iterations(100);
```

## Big-Endian Convention

All operators use **big-endian** bit ordering, matching Julia Quantics.jl:
- Site 0 = Most Significant Bit (MSB)
- Site R-1 = Least Significant Bit (LSB)
- Integer value: x = Σ_n x_n * 2^(R-1-n)

For example, with R=3 sites, the value 5 = 101₂ is represented as:
- Site 0: bit = 1 (contributes 2² = 4)
- Site 1: bit = 0 (contributes 0)
- Site 2: bit = 1 (contributes 2⁰ = 1)

## Boundary Conditions

Two boundary conditions are supported:

- **Periodic**: Results wrap around modulo 2^R
- **Open**: Results outside [0, 2^R) produce zero vectors

```rust
// Periodic: shift(7, 2) = 9 mod 8 = 1
let shift_periodic = shift_operator(3, 2, BoundaryCondition::Periodic)?;

// Open: shift(7, 2) = 9 >= 8, so result is zero vector
let shift_open = shift_operator(3, 2, BoundaryCondition::Open)?;
```

## Reference

The implementation is based on:

- [Quantics.jl](https://github.com/tensor4all/Quantics.jl) - Julia implementation
- J. Chen and M. Lindsey, "Direct Interpolative Construction of the Discrete Fourier Transform as a Matrix Product Operator", arXiv:2404.03182 (2024) - QFT algorithm
