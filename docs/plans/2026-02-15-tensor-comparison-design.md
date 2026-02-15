# Tensor Comparison Utilities Design

## Problem

Test code across tensor4all-rs uses verbose element-wise loops to verify tensor accuracy:

```rust
let data_a = a.to_vec_f64().unwrap();
let data_b = b.to_vec_f64().unwrap();
for (i, (&x, &y)) in data_a.iter().zip(data_b.iter()).enumerate() {
    assert!((x - y).abs() < 1e-10, "Element {} mismatch: {} vs {}", i, x, y);
}
```

ITensors.jl uses `isapprox(A, B; atol, rtol)` and `norm(A - B)` for concise comparison.

## Design

### TensorLike Trait Additions

One new **required** method:

```rust
/// Maximum absolute value of all elements (L-infinity norm).
fn maxabs(&self) -> f64;
```

Three new **provided** methods (default implementations using existing `axpby`, `scale`, `norm`):

```rust
/// Element-wise subtraction: self - other.
/// Indices are automatically permuted to match via axpby.
fn sub(&self, other: &Self) -> Result<Self> {
    self.axpby(AnyScalar::new_real(1.0), other, AnyScalar::new_real(-1.0))
}

/// Negate all elements.
fn neg(&self) -> Result<Self> {
    self.scale(AnyScalar::new_real(-1.0))
}

/// Approximate equality check (Julia isapprox semantics).
/// Returns true if ||self - other|| <= max(atol, rtol * max(||self||, ||other||))
fn isapprox(&self, other: &Self, atol: f64, rtol: f64) -> bool {
    let diff = match self.sub(other) { Ok(d) => d, Err(_) => return false };
    diff.norm() <= atol.max(rtol * self.norm().max(other.norm()))
}
```

### TensorDynLen Implementation

- `maxabs()`: iterate over storage elements, return max absolute value
- Operator traits: `Sub` (4 variants), `Neg` (2 variants) delegating to trait methods

### Impact on Existing Implementations

Only `maxabs()` must be implemented by each `TensorLike` implementor. All other methods are provided.

### Usage After

```rust
// Before (8 lines)
let data_a = a.to_vec_f64().unwrap();
let data_b = b.to_vec_f64().unwrap();
for (i, (&x, &y)) in data_a.iter().zip(data_b.iter()).enumerate() {
    assert!((x - y).abs() < 1e-10, ...);
}

// After (1 line)
assert!(result.isapprox(&expected, 1e-10, 0.0));
```

### Tests

- `sub`: difference of identical tensors is zero
- `maxabs`: known tensor returns expected max
- `isapprox`: boundary tests for atol/rtol
- Existing element-wise loops migrated in follow-up PR
