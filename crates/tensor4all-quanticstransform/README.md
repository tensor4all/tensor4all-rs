# tensor4all-quanticstransform

Quantics transformation operators for tensor train methods. Port of Quantics.jl.

## Key Types

- `LinearOperator` — operator in TreeTN form; returned by all constructor functions
- `shift_operator()` — basis shift `|x> -> |x + offset>`; as a matrix action, `(M g)[x] = g[x - offset]`
- `flip_operator()` — reflection: f(x) = g(2^R - x)
- `quantics_fourier_operator()` — Quantics Fourier Transform (QFT)
- `affine_operator()` — affine transform y = A·x + b with rational coefficients

## Conventions

- Operators carry their own input and output indices. Replace a state site's
  index with the operator input index before applying it.
- Multi-variable operators use interleaved bit encoding:
  `site = bit_var0 + 2 * bit_var1 + ...`.
- QFT output is in bit-reversed frequency order.
- Affine matrices are column-major.
- Dense materialization is for small reference/debug checks only.

## Example

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_quanticstransform::{flip_operator, shift_operator, BoundaryCondition};

// Create operators for R=4 sites (2^4 = 16 grid points)
let r = 4;
let shift_op = shift_operator(r, 3, BoundaryCondition::Periodic)?;
let flip_op = flip_operator(r, BoundaryCondition::Periodic)?;

// Operators are LinearOperator (TreeTN form).
assert_eq!(shift_op.mpo.node_count(), r);
assert_eq!(flip_op.mpo.node_count(), r);
# Ok(())
# }
```

## Documentation

- [User Guide: Quantics](https://tensor4all.org/tensor4all-rs/guides/quantics.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_quanticstransform/)
