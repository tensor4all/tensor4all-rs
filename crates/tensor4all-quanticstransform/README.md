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
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
use tensor4all_quanticstransform::{shift_operator, BoundaryCondition};
use tensor4all_treetn::{apply_linear_operator, ApplyOptions};

// Create a shift operator for R=4 sites (2^4 = 16 grid points)
let r = 4;
let shift_op = shift_operator(r, 3, BoundaryCondition::Periodic)?;

// Build a tensor train representing f(x) = x, then apply the shift
// (result represents g(x) = f(x + 3) = x + 3 mod 16)
let f = |idx: &[i64]| idx[0] as f64;
let (qtci, _ranks, _errors) = quanticscrossinterpolate_discrete(
    &[16],
    f,
    None,
    QtciOptions::default().with_tolerance(1e-12),
)?;

let treetn = qtci.to_treetn()?;
let result = apply_linear_operator(&shift_op, &treetn, ApplyOptions::default())?;
assert_eq!(result.node_count(), r);
```

## Documentation

- [User Guide: Quantics](https://tensor4all.github.io/tensor4all-rs/guides/quantics.html)
- [API Reference](https://tensor4all.github.io/tensor4all-rs/rustdoc/tensor4all_quanticstransform/)
