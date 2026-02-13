# Quanticstransform Test Coverage & Julia Alignment Design

## Goal

Strengthen the test coverage and Julia (Quantics.jl) alignment of the
`tensor4all-quanticstransform` crate. Focus on numerical correctness
verification rather than structural smoke tests.

## Context

The comparison document (`docs/design/quanticstransform_julia_comparison.md`)
identified several critical gaps between the Rust and Julia implementations.
This design addresses the test coverage gaps and includes implementing
binaryop multi-variable support to enable the full Julia test suite.

## Scope

- **In scope**: Test coverage for all 6 existing operators + binaryop
  multi-variable implementation
- **Out of scope**: imaginarytime, mul modules, antisymmetric BC,
  BigFloat precision, Fourier origin shift

---

## P0: binaryop — Numerical Correctness + Multi-Variable Implementation

### Gap

- Rust has zero numerical correctness tests for binaryop (all are smoke tests)
- Julia tests 81 combinations of (a,b,c,d) with brute-force verification
- Rust's `coeffs2` is incomplete — multi-variable (2-output) support missing

### Phase 1: Single-Variable Brute-Force Tests

Add dense-matrix verification for the existing single-variable binaryop.

```
Parameters:
  (a, b) in {(1,0), (0,1), (1,1), (1,-1), (-1,1), (0,0)}
  R in {2, 3}
  BC in {Periodic, Open}

Method:
  1. Generate binaryop MPO
  2. Convert MPO to dense 2^R x (2^R x 2^R) matrix
  3. For each (x, y): verify M[z, (x,y)] == 1 where z = (a*x + b*y) mod 2^R
```

### Phase 2: Multi-Variable binaryop Implementation

Implement the 2-output-variable binaryop following Julia's
`_binaryop_tensor_multisite`.

```
Input:  (x, y)          — 2 input variables, R bits each
Output: (z1, z2)        — 2 output variables, R bits each
Transform: z1 = a*x + b*y,  z2 = c*x + d*y

Key implementation points:
  - Carry states {-1, 0, 1} for each output (bond dim = 3 per output)
  - Combined bond dim = 9 (3 x 3) for 2 outputs
  - Support (-1, -1) coefficients via flip composition
  - Site tensor: T[cin1, cin2, x, y, z1, z2, cout1, cout2]
```

### Phase 3: 81-Combination Test

```
Parameters:
  (a, b, c, d) in {-1, 0, 1}^4 = 81 combinations
  R in {2, 3}
  BC: Periodic

Method:
  1. Generate 2-output binaryop MPO
  2. Convert to dense (2^R x 2^R) x (2^R x 2^R) matrix
  3. For each (x, y): verify matrix maps to (z1, z2) = (a*x+b*y, c*x+d*y) mod 2^R
```

### Test Helpers

```rust
/// Convert binaryop MPO to dense matrix for verification
fn binaryop_mpo_to_matrix(mpo: &[Tensor], r: usize) -> Matrix

/// Compute expected binaryop result
fn expected_binaryop(a: i64, b: i64, x: usize, y: usize, r: usize, bc: BC) -> usize
```

---

## P1: Fourier — Phase Verification

### Gap

- Rust tests only check magnitude (1/sqrt(N)), not phase
- Julia compares against full DFT reference matrix
- Rust tests only R=3; Julia tests R=2,3,4

### Tests

**test_fourier_phase_correctness**:
```
Parameters: R in {2, 3, 4}, sign in {-1, +1}
Method:
  1. Generate Fourier MPO
  2. Convert to dense N x N matrix F[k, x]
  3. Compute reference: F_ref[k, x] = (1/sqrt(N)) * exp(sign * 2*pi*i * k*x / N)
  4. Assert element-wise: |F[k,x] - F_ref[k,x]| < 1e-10
```

**test_fourier_inverse_roundtrip**:
```
Parameters: R in {2, 3, 4}
Method:
  1. Compute F and F_inv as dense matrices
  2. Assert F * F_inv ≈ I (tolerance 1e-10)
```

---

## P2: Multi-Variable flip/shift/phase_rotation

### Gap

- Julia tests 2D and 3D versions of flip (reverseaxis), shift (shiftaxis),
  phase_rotation
- Rust tests only 1D

### Tests

**test_flip_multivariable_2d**:
```
Parameters: R=2, 2 variables
Method: Apply flip to each axis of a 2D quantics MPS, verify brute-force
Note: Requires multi-variable operator support. If not yet available,
      test by composing single-variable operators on interleaved sites.
```

**test_shift_multivariable_2d**:
```
Parameters: R=2, 2 variables, various offsets per axis
Method: Shift each axis independently, verify against expected permutation
```

**test_phase_rotation_multivariable**:
```
Parameters: R=2, 2 variables, theta1=pi/4, theta2=pi/3
Method: Apply independent phase rotations, verify factorization
```

---

## P3: cumsum — Lower Triangle

### Gap

- Julia supports both `:upper` and `:lower` triangle
- Rust implements upper only

### Tests

**test_cumsum_lower_triangle** (if lower is implemented or implementable):
```
Parameters: R=3
Method: Verify y_i = sum_{j > i} x_j (lower triangle)
        Compare against upper triangle with reversed ordering
```

---

## P4: Random MPS Linearity Tests

### Gap

- All Rust tests use product states
- Julia uses random MPS inputs to verify linearity

### Tests

**test_operator_linearity**:
```
Operators: flip, shift, phase_rotation, cumsum, fourier, binaryop
Parameters: R=3

Method:
  1. Generate 2 random MPS: a, b (bond dim 2-4)
  2. Pick random scalars alpha, beta
  3. Compute Op(alpha*a + beta*b)
  4. Compute alpha*Op(a) + beta*Op(b)
  5. Assert equality (tolerance 1e-8)

Purpose: Verify operators are correct linear maps, not just correct on
         product states
```

---

## Priority and Dependencies

```
P0 Phase 1 (single-var binaryop tests)
    |
P0 Phase 2 (multi-var binaryop impl)  -->  P0 Phase 3 (81-combo tests)

P1 (Fourier phase)  -- independent --

P2 (multi-var flip/shift/phase)  -- depends on multi-var infrastructure --

P3 (cumsum lower)  -- independent --

P4 (random MPS linearity)  -- independent, can run in parallel --
```

## Success Criteria

1. All binaryop numerical correctness tests pass (single + multi-variable)
2. Fourier tests verify phase, not just magnitude
3. Fourier tests cover R=2,3,4
4. At least 2D multi-variable tests exist for flip, shift, phase_rotation
5. Random MPS linearity tests pass for all operators
6. No existing tests are broken or weakened

---

*Date: 2026-02-13*
*Crate: tensor4all-quanticstransform*
*Reference: docs/design/quanticstransform_julia_comparison.md*
