# Affine Anti-Periodic Boundary and Difference Kernel Design

## Context

`tensor4all-quanticstransform` currently exposes one shared
`BoundaryCondition` enum with `Periodic` and `Open`. The affine implementation
converts that enum to a boolean periodic/open flag, so it cannot represent
anti-periodic signs. `Quantics.jl` has anti-periodic precedent in the older
`binaryop.jl` and shift/flip APIs via `bc = -1`, with signs determined by the
integer wrap quotient. The newer `Quantics.jl/src/affine.jl` API has only
periodic and open boundary condition structs.

The requested end goal is to build a difference-kernel MPO
`A(x, x') = f(x - x')` from a QTT representation of `f`, supporting periodic
and anti-periodic boundary choices. The clean dependency order is to generalize
affine boundary handling first, then build the kernel on top of affine delta
tensors.

## Phasing

Phase 1 extends affine and boundary-condition infrastructure:

- Add `BoundaryCondition::AntiPeriodic` to the shared quantics transform enum.
- Generalize affine matrix, MPO, and unfused-tensor construction from
  periodic/open booleans to boundary weights on final carry states.
- Extend shift and flip matches so the shared enum remains usable crate-wide.
- Extend the C API boundary enum and round-trip conversions.
- Extend the Tensor4all.jl wrapper to accept an anti-periodic symbol.
- Extend the existing affine tutorial with one anti-periodic example.

Phase 2 adds the difference-kernel MPO:

- Add an API that constructs an MPO/operator for `A(x, x') = f(x - x')`.
- Use the Phase 1 affine delta tensor internally.
- Support `Periodic` and `AntiPeriodic`; reject `Open` initially.
- Add focused tests and a tutorial example after the affine foundation is in
  place.

## Boundary Semantics

For affine maps, use the scaled integer equation

```text
A_int x + b_int - s y = 2^R c
```

where `s` is the common denominator scale and `c` is the final carry vector.
Each output dimension contributes one scalar boundary weight:

```text
Periodic:     1
AntiPeriodic: (-1)^c
Open:         1 if c == 0 else 0
```

The total value is the product of the per-output weights. Negative carries must
be handled by parity of the carry itself, not by treating negative carries as
invalid. In code, the anti-periodic sign should be computed with Euclidean
parity, for example:

```rust
if carry.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
```

This gives `+1` for `c = 0, 2, -2` and `-1` for `c = 1, -1`.

High bits of `b` must not be discarded for anti-periodic boundaries. For
example, `y = x + 2^R` is the identity under periodic boundaries, but equals
`-I` under anti-periodic boundaries. Therefore anti-periodic affine MPO
construction must use the high-bit extension path when nonzero high bits remain
after the `R` active bits have been processed.

## Rust API

Extend the existing enum rather than adding a difference-kernel-specific
boundary enum:

```rust
pub enum BoundaryCondition {
    Periodic,
    AntiPeriodic,
    Open,
}
```

Existing affine APIs keep their signatures:

```rust
pub fn affine_operator(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator>;

pub fn affine_operator_interleaved(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator>;

pub fn affine_transform_matrix(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<CsMat<f64>>;

pub fn affine_transform_tensors_unfused(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<Vec<Tensor3<Complex64>>>;
```

Internal affine construction should pass the boundary conditions through to a
small helper that computes final-carry weights. This replaces the current
`bc_periodic: Vec<bool>` representation.

## C API and Julia Wrapper

Extend `t4a_boundary_condition` with a third variant:

```rust
pub enum t4a_boundary_condition {
    Periodic = 0,
    Open = 1,
    AntiPeriodic = 2,
}
```

Keep existing numeric values for ABI compatibility. Add conversion tests that
round-trip all three variants.

In Tensor4all.jl, extend `_bc_code` to accept `:antiperiodic`. It may also
accept `:anti_periodic` as an alias, but the canonical spelling in docs should
be `:antiperiodic` to match a single enum-style word.

## Affine Tutorial

Extend the current affine tutorial instead of adding a separate anti-periodic
tutorial. The existing example uses the pullback

```text
u = x + y
v = y
```

so only `u` wraps when `x + y >= N`. This makes the anti-periodic effect clear:

```text
Periodic:
  g((x + y) mod N, y)

AntiPeriodic:
  +g((x + y) mod N, y)  if x + y < N
  -g((x + y) mod N, y)  if x + y >= N

Open:
  g(x + y, y)           if x + y < N
  0                     otherwise
```

Use boundary conditions:

```rust
&[BoundaryCondition::AntiPeriodic, BoundaryCondition::Periodic]
```

for the anti-periodic case, so the sign belongs only to the wrapped `u`
coordinate. Update the value, error, transformed-bond, and operator-bond plots
to include the anti-periodic series.

## Phase 1 Tests

Add affine tests for anti-periodic high-bit behavior:

- `y = x + 2^R` gives `-I`.
- `y = x - 2^R` gives `-I`.
- `y = x + 2 * 2^R` gives `+I`.
- `y = x + 2^R + 1` gives periodic shift by one with an overall `-1`.

Add a multivariable affine case matching the future difference-kernel delta:

```text
z = x - x'
bc[z] = AntiPeriodic
```

For `R = 2` or `R = 3`, verify that `x < x'` receives sign `-1` and
`x >= x'` receives sign `+1`. Compare dense affine matrix construction with the
MPO contraction path.

Add C API round-trip coverage for `AntiPeriodic` and Julia wrapper tests for
the new symbol.

## Phase 2 Difference Kernel

The difference-kernel construction starts from a QTT of `f(z)` and builds an
MPO over `(x, x')`:

```text
A[x, x'] = f((x - x') mod 2^R)
```

For anti-periodic boundaries, multiply by the affine boundary sign:

```text
A[x, x'] = +f((x - x') mod 2^R)  if x >= x'
A[x, x'] = -f((x - x') mod 2^R)  if x < x'
```

Implementation strategy:

- Build the affine delta tensor for `z = x - x'`.
- Contract the `z` physical leg sitewise with the QTT core of `f`.
- Return a TT-MPO with local physical dimension `4`, where output is `x` and
  input is `x'`.

The affine delta for `z = x - x'` has carry bond dimension at most `2`, so the
resulting MPO bond dimensions are bounded by twice the QTT bond dimensions of
`f`. Use exact local contraction first; do not introduce truncation or SVD in
the initial API.

The Phase 2 API should reject `BoundaryCondition::Open` until open-boundary
difference kernels are explicitly designed.
