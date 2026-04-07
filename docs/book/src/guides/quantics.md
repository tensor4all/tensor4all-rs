# Quantics Transform

The `tensor4all-quanticstransform` crate provides `LinearOperator` constructors for
applying transformations to functions represented as quantics tensor trains.
It is a Rust port of the transformation functionality from
[Quantics.jl](https://github.com/tensor4all/Quantics.jl).

Add it to your `Cargo.toml`:

```toml
[dependencies]
tensor4all-quanticstransform = "0.1"
tensor4all-treetn = "0.1"
```

---

## Operator Overview

Every constructor returns a `LinearOperator` from `tensor4all-treetn`, so all
operators are applied in the same way regardless of their mathematical meaning.

| Operator | Mathematical effect | Constructor |
|----------|---------------------|-------------|
| Flip | f(x) = g(2^R − x) | `flip_operator` |
| Shift | f(x) = g(x + offset) mod 2^R | `shift_operator` |
| Phase Rotation | f(x) = exp(i·θ·x) · g(x) | `phase_rotation_operator` |
| Cumulative Sum | y_i = Σ_{j < i} x_j | `cumsum_operator` |
| Fourier Transform | Quantics Fourier Transform (QFT) | `quantics_fourier_operator` |
| Binary Operation | f(x, y), first variable → a·x + b·y | `binaryop_single_operator` |
| Affine Transform | y = A·x + b (rational coefficients) | `affine_operator` |

The parameter `r` that appears in every constructor is the number of quantics
bits (sites) per variable.  A single variable is discretized on `2^r` grid
points.

### Error Conditions

Constructors return `Err` for invalid inputs:

- `r == 0` — no sites to operate on
- `r == 1` for `cumsum_operator`, `triangle_operator`, `quantics_fourier_operator` — requires at least 2 sites
- `r >= 64` for `shift_operator` — would overflow a 64-bit integer
- NaN or Inf `theta` for `phase_rotation_operator` — invalid rotation angle
- `BinaryCoeffs(-1, -1)` for `binaryop_single_operator` — this combination is not supported

---

## Creating Operators

```rust,ignore
use tensor4all_quanticstransform::{
    flip_operator, shift_operator, phase_rotation_operator,
    cumsum_operator, quantics_fourier_operator,
    BoundaryCondition, FourierOptions,
};

// 8-bit quantics representation (2^8 = 256 grid points)
let r = 8;

// Flip: f(x) = g(2^R - x)
let flip_op = flip_operator(r, BoundaryCondition::Periodic)?;

// Shift by 10: f(x) = g(x + 10) mod 2^R
let shift_op = shift_operator(r, 10, BoundaryCondition::Periodic)?;

// Phase rotation: f(x) = exp(i*pi/4*x) * g(x)
let phase_op = phase_rotation_operator(r, std::f64::consts::PI / 4.0)?;

// Cumulative sum
let cumsum_op = cumsum_operator(r)?;

// Fourier transform (forward)
let ft_op = quantics_fourier_operator(r, FourierOptions::forward())?;
```

### Affine Transform

For transformations of the form **y = A·x + b** with rational coefficients:

```rust,ignore
use tensor4all_quanticstransform::{affine_operator, AffineParams, BoundaryCondition};
use num_rational::Rational64;

// A = [[1, 1], [1, -1]], b = [0, 0]  (2 outputs, 2 inputs)
let a = vec![
    Rational64::from_integer(1), Rational64::from_integer(1),
    Rational64::from_integer(1), Rational64::from_integer(-1),
];
let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
let params = AffineParams::new(a, b, 2, 2)?;
let bc = vec![BoundaryCondition::Periodic; 2];
let affine_op = affine_operator(r, &params, &bc)?;
```

---

## Applying Operators to a Tensor Train

### Index Mapping Flow

Operators define abstract *input* and *output* indices labeled `0, 1, …, r-1`.
To apply an operator, replace the tensor network's site indices with the
operator's input indices using `replaceind`, then call `apply_linear_operator`.

```
Original TreeTN          Operator            Result TreeTN
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  site_idx_0  │───▶│  input_0     │    │  output_0    │
│  site_idx_1  │───▶│  input_1     │───▶│  output_1    │
│  site_idx_2  │───▶│  input_2     │    │  output_2    │
│  other_idx   │    │              │    │  other_idx   │  (passes through)
└──────────────┘    └──────────────┘    └──────────────┘
```

`apply_linear_operator`:
1. Contracts the operator's tensors with the TreeTN.
2. Replaces input indices with output indices.
3. Leaves all unrelated indices unchanged.

### Basic Application

```rust,ignore
use tensor4all_treetn::{apply_linear_operator, ApplyOptions};

let r = 4;
let shift_op = shift_operator(r, 3, BoundaryCondition::Periodic)?;

// Obtain the TreeTN's site indices (one per bit site)
let site_indices: Vec<DynIndex> = treetn.external_indices();

// Remap the TreeTN's site indices to the operator's input indices
let mut treetn_remapped = treetn;
for i in 0..r {
    let op_input = shift_op
        .get_input_mapping(&i)
        .expect("missing input mapping")
        .true_index
        .clone();
    treetn_remapped = treetn_remapped.replaceind(&site_indices[i], &op_input)?;
}

// Apply the operator
let result = apply_linear_operator(&shift_op, &treetn_remapped, ApplyOptions::default())?;
```

### Partial Application (Multi-Variable Tensor Trains)

A key feature is applying an operator to **only some indices** of a tensor
network.  For example, to Fourier-transform only the x-variable of a 2D
function f(x, y):

```rust,ignore
let r = 4;
// treetn_2d has interleaved or separate indices for x and y
let x_indices: Vec<DynIndex> = /* 4 bit indices for x */;
let y_indices: Vec<DynIndex> = /* 4 bit indices for y */;

let fourier_x = quantics_fourier_operator(r, FourierOptions::forward())?;

// Remap ONLY the x indices
let mut treetn_remapped = treetn_2d;
for i in 0..r {
    let op_input = fourier_x
        .get_input_mapping(&i)
        .expect("missing input mapping")
        .true_index
        .clone();
    treetn_remapped = treetn_remapped.replaceind(&x_indices[i], &op_input)?;
}
// y_indices are NOT remapped — they pass through unchanged

let result = apply_linear_operator(&fourier_x, &treetn_remapped, ApplyOptions::default())?;
// result represents F_x[f](k_x, y)
```

### Truncation Options

Control the accuracy/efficiency tradeoff:

```rust,ignore
// No truncation — exact but potentially expensive
let options = ApplyOptions::naive();

// ZipUp algorithm with bond dimension and tolerance limits
let options = ApplyOptions::zipup()
    .with_max_bond_dim(64)
    .with_tolerance(1e-10);

// Fit algorithm for more control over iterations
let options = ApplyOptions::fit()
    .with_max_bond_dim(32)
    .with_max_iterations(100);

let result = apply_linear_operator(&op, &treetn_remapped, options)?;
```

---

## Bit Ordering and Encoding

### Big-Endian Convention

All operators use **big-endian** bit ordering, matching Julia's Quantics.jl:

- Site 0 = Most Significant Bit (MSB)
- Site R−1 = Least Significant Bit (LSB)
- Integer value: x = Σ_n x_n · 2^(R−1−n)

For example, with R = 3 sites the value 5 = 101₂ is stored as:

| Site | Bit | Contribution |
|------|-----|-------------|
| 0 | 1 | 2² = 4 |
| 1 | 0 | 0 |
| 2 | 1 | 2⁰ = 1 |

### Multi-Variable Encoding

The `_multivar` variants (`flip_operator_multivar`, `shift_operator_multivar`,
`phase_rotation_operator_multivar`) use **interleaved bit encoding** for
multiple variables.  Each site simultaneously encodes one bit from each
variable:

```
site index s_n encodes: bit_var0 + 2 * bit_var1 + 4 * bit_var2 + ...
```

Each site's local dimension is `2^num_vars`.  This interleaved layout is the
standard quantics multi-variable representation and is equivalent to
interleaving the bit planes of all variables.

---

## Boundary Conditions

Two boundary conditions are supported for operators that wrap indices (flip,
shift):

- **Periodic** — results wrap modulo 2^R.
- **Open** — indices that fall outside [0, 2^R) produce a zero vector.

```rust,ignore
// Periodic: shift(7, 2) = 9 mod 8 = 1
let shift_periodic = shift_operator(3, 2, BoundaryCondition::Periodic)?;

// Open: shift(7, 2) = 9 >= 8, so the result is the zero vector
let shift_open = shift_operator(3, 2, BoundaryCondition::Open)?;
```

---

## Fourier Transform Convention

`quantics_fourier_operator` produces output in **bit-reversed order**.  This is
inherent to the QFT algorithm: the output site ordering corresponds to the
bit-reversal of the frequency index.  If you need natural frequency ordering,
apply a bit-reversal permutation after the transform.

```rust,ignore
// Forward QFT (bit-reversed output)
let fwd = quantics_fourier_operator(r, FourierOptions::forward())?;

// Inverse QFT
let inv = quantics_fourier_operator(r, FourierOptions::inverse())?;
```

---

## Reference

- [Quantics.jl](https://github.com/tensor4all/Quantics.jl) — Julia implementation that this crate ports.
- J. Chen and M. Lindsey, "Direct Interpolative Construction of the Discrete Fourier Transform as a Matrix Product Operator", arXiv:2404.03182 (2024) — QFT algorithm.
