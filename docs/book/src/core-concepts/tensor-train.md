# Tensor Train (MPS)

A **Tensor Train** (TT), also known as a **Matrix Product State** (MPS) in physics, is a factorized representation of a high-dimensional tensor.

## Definition

An N-dimensional tensor `A[i₁, i₂, ..., iₙ]` is decomposed as:

```
A[i₁, i₂, ..., iₙ] = T₁[i₁] · T₂[i₂] · ... · Tₙ[iₙ]
```

where each `Tₖ[iₖ]` is a matrix of size `rₖ₋₁ × rₖ`. The values `r₀, r₁, ..., rₙ` are called **bond dimensions** (with `r₀ = rₙ = 1`).

## Key Operations

### Creation

```rust
use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};

// Constant tensor train
let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

// From explicit cores
let cores = vec![/* 3D arrays */];
let tt = TensorTrain::from_cores(cores)?;
```

### Evaluation

```rust
// Get value at specific indices
let value = tt.evaluate(&[0, 1, 2])?;
```

### Arithmetic

```rust
// Addition
let sum = &tt1 + &tt2;

// Scalar multiplication
let scaled = &tt * 2.0;
```

### Compression

Truncation reduces bond dimensions while maintaining accuracy:

```rust
// Compress with relative tolerance 1e-10 and maximum rank 20
let compressed = tt.compressed(1e-10, Some(20))?;
```

### Canonical Forms

Tensor trains can be put into left-canonical, right-canonical, or mixed-canonical forms for numerical stability:

```rust
// Orthogonalize to position 2
tt.orthogonalize(2)?;
```

## Site Dimensions vs Bond Dimensions

- **Site dimensions**: The physical indices `[d₁, d₂, ..., dₙ]`
- **Bond dimensions**: The internal indices `[1, r₁, r₂, ..., rₙ₋₁, 1]`

The compression quality depends on the bond dimensions. Lower bond dimensions mean more compression but potentially less accuracy.
