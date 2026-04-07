# tensor4all-quanticstransform

Quantics transformation operators for tensor train methods. Port of Quantics.jl.

## Key Types

- `LinearOperator` — operator in TreeTN form; returned by all constructor functions
- `shift_operator()` — cyclic shift: f(x) = g(x + offset) mod 2^R
- `flip_operator()` — reflection: f(x) = g(2^R - x)
- `quantics_fourier_operator()` — Quantics Fourier Transform (QFT)
- `affine_operator()` — affine transform y = A·x + b with rational coefficients

## Example

```rust,ignore
use tensor4all_quanticstransform::{shift_operator, flip_operator, BoundaryCondition};

// Create operators for R=4 sites (2^4 = 16 grid points)
let r = 4;
let shift_op = shift_operator(r, 3, BoundaryCondition::Periodic)?;
let flip_op = flip_operator(r)?;

// Operators are LinearOperator (TreeTN form) — apply to a TreeTN with
// tensor4all_treetn::apply_linear_operator()
```

## Documentation

- [User Guide: Quantics](https://tensor4all.github.io/tensor4all-rs/guides/quantics.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_quanticstransform/)
