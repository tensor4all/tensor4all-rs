# tensor4all-quanticstransform

Quantics transformation operators for tensor train methods. Port of Quantics.jl.

## Key Types

- `LinearOperator` — operator in TreeTN form; returned by all constructor functions
- `shift_operator()` — basis shift `|x> -> |x + offset>`; as a matrix action, `(M g)[x] = g[x - offset]`
- `flip_operator()` — reflection: f(x) = g(2^R - x)
- `quantics_fourier_operator()` — Quantics Fourier Transform (QFT)
- `affine_operator()` — affine transform y = A·x + b with rational coefficients
- `LinearConstraintRow` — primitive integer row normalization for scale-invariant constraints

## Cargo features

Fourier operators, including `FTCore` and the multi-axis constructor, use
TreeTN directly. For a Fourier-only build, disable default features. Add
`partitioned` for patched Fourier.

```toml
[dependencies]
tensor4all-quanticstransform = { path = "../tensor4all-rs/crates/tensor4all-quanticstransform", default-features = false, features = ["partitioned"] }
```

`tensor4all-treetn` has no default features, so importing it directly keeps
its SimpleTT bridge disabled. The
`simplett-transforms` feature is enabled by default and provides the
non-Fourier transforms and SimpleTT conversion APIs.

## Conventions

- Operators carry their own input and output indices. Replace a state site's
  index with the operator input index before applying it.
- Multi-variable operators use interleaved bit encoding:
  `site = bit_var0 + 2 * bit_var1 + ...`.
- QFT output is in bit-reversed frequency order.
- Affine matrices are column-major.
- `LinearConstraintRow` is for equality or halfspace constraints such as
  `16*x <= 64`; it is not a simplification step for affine maps such as
  `y = A*x + b`.
- Dense materialization is for small reference/debug checks only.

## Example

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};

let op = quantics_fourier_operator(2, FourierOptions::forward())?;
// The normalized four-point DFT has Frobenius norm sqrt(4).
assert!((op.mpo().clone().norm()? - 2.0).abs() < 1e-10);
# Ok(())
# }
```

## Documentation

- [User Guide: Quantics](https://tensor4all.org/tensor4all-rs/guides/quantics.html)
- [User Guide: Quantum Fourier Transform](https://tensor4all.org/tensor4all-rs/guides/qft.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_quanticstransform/)

## Multi-axis and patched Fourier transforms

`quantics_fourier_operator_multiaxis` accepts an explicit `AxisBit` layout.
Grouped and interleaved layouts are built exactly without site swaps; the bond
at each cut is the product of the per-axis bonds.
Enable the optional `partitioned` feature for `patched_fourier_transform`,
which transforms selected axes of existing TreeTN patches and returns lazy
output regions. Selections must lie on one tree path; inverse calls reverse
each axis's index order. Reconstruction bounds exclude operator construction
error and backend roundoff. See the [QFT guide](https://tensor4all.org/tensor4all-rs/guides/qft.html).
