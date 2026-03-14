# Design: Tensor4all Scalar/Tensor API over Tenferro origin/main

**Date:** 2026-03-14
**Status:** Approved in chat

## Goal

Adapt tensor4all-rs to the current `tenferro-rs` `origin/main` model where the native backend treats
everything as tensors, while preserving a clearer tensor4all public API with distinct scalar and
tensor types.

The user-facing rule is:

- normal users primarily see `TensorDynLen`
- scalar-valued results are represented by a dedicated `Scalar` type
- tenferro native payload types are not part of tensor4all's public API

## Upstream Baseline

The reviewed upstream target is:

- `tenferro-rs` `origin/main`
- commit `26c4cb16d3de81ab624212250d3c37393ecf7609`
- date `2026-03-13`

This upstream now models values through tensor-native AD payloads rather than a separate dynamic
scalar surface that tensor4all can re-export directly.

## Public Type Model

tensor4all-rs keeps separate semantic types even if tenferro stores both with tensor-native payloads.

### Tensor

`TensorDynLen` remains the main public tensor type.

Its visible shape stays centered on tensor4all semantics:

```rust
#[derive(Clone)]
pub struct TensorDynLen {
    pub indices: Vec<DynIndex>,
    native: DynAdTensor,
}
```

Users should continue to work mostly with `TensorDynLen`.

### Scalar

Add a new public `Scalar` newtype.

```rust
#[derive(Clone)]
pub struct Scalar {
    native: DynAdTensor,
}
```

`Scalar` is a semantic scalar, not an arbitrary one-element tensor.

Its invariant is strict:

- `Scalar` must always wrap a rank-0 native tensor
- one-element rank-1 tensors do not qualify

### Compatibility Alias

Keep:

```rust
pub type AnyScalar = Scalar;
```

This preserves the existing tensor4all type name while changing its implementation to the new
rank-0 scalar wrapper.

## Native Boundary Policy

tenferro native payloads are implementation details, not part of the stable tensor4all public API.

The following `TensorDynLen` APIs should be removed from the public API surface:

- `from_native`
- `as_native`
- `into_native`

Equivalent helpers may remain `pub(crate)` for internal backend wiring and tests.

`to_storage()` remains as an explicit interop boundary because it is a tensor4all-owned concept
used by C API, HDF5, debugging, and legacy inspection.

## Tensor and Scalar API Semantics

### Tensor-returning operations

Operations whose result is mathematically a tensor continue returning `TensorDynLen`.

Examples:

- `permute`
- `contract`
- `outer_product`
- `conj`
- `replaceind`
- `qr`
- `svd`
- `factorize`

### Scalar-returning operations

Operations whose result is mathematically a scalar must return `Scalar`.

Examples:

- `TensorDynLen::sum(&self) -> Scalar`
- `TensorDynLen::only(&self) -> Scalar`
- `TensorDynLen::inner_product(&self, other: &Self) -> Result<Scalar>`

### Tensor-scalar mixed operations

Operations that accept scalar coefficients should take `Scalar`/`AnyScalar`.

Examples:

- `TensorDynLen::scale(&self, scalar: Scalar) -> Result<Self>`
- `TensorDynLen::axpby(&self, a: Scalar, other: &Self, b: Scalar) -> Result<Self>`

## Scalar Public API

`Scalar` should expose only scalar-meaningful operations.

Recommended public surface:

- constructors:
  - `Scalar::from_real(f64) -> Self`
  - `Scalar::from_complex(re: f64, im: f64) -> Self`
- inspectors:
  - `real() -> f64`
  - `imag() -> f64`
  - `abs() -> f64`
  - `is_real() -> bool`
  - `is_complex() -> bool`
  - `is_zero() -> bool`
  - `mode() -> AdMode`
- transforms:
  - `conj() -> Self`
- optional AD accessors:
  - `tangent() -> Option<Scalar>`

Compatibility conversions should remain where they are already part of the current user model:

- `From<f64> for Scalar`
- `From<Complex64> for Scalar`
- `TryFrom<Scalar> for f64`
- `From<Scalar> for Complex64`

Native conversion helpers for `Scalar` should be internal:

- `pub(crate) fn from_native(native: DynAdTensor) -> Result<Self>`
- `pub(crate) fn as_native(&self) -> &DynAdTensor`
- `pub(crate) fn into_native(self) -> DynAdTensor`

## AD Design

tensor4all should expose an AD API that preserves immutable value semantics.

### Core rule

AD annotations produce new values. They do not mutate tensors in place and do not attach mutable
`.grad` fields.

### Reverse-mode API

Expose a tensor4all-owned tape handle:

```rust
pub struct GradTape { ... }
```

Recommended public API:

- `GradTape::new() -> Self`
- `TensorDynLen::requires_grad(&self, tape: &GradTape) -> Result<Self>`
- `Scalar::requires_grad(&self, tape: &GradTape) -> Result<Self>`

Gradient queries should return values, not mutate hidden buffers.

### Forward-mode API

Expose explicit tangent seeding:

- `TensorDynLen::with_tangent(&self, tangent: &TensorDynLen) -> Result<Self>`
- `Scalar::with_tangent(&self, tangent: &Scalar) -> Result<Self>`
- `TensorDynLen::tangent(&self) -> Option<TensorDynLen>`
- `Scalar::tangent(&self) -> Option<Scalar>`

### Mixed forward/reverse policy

General mixed-mode values should not be part of the ordinary public API contract.

Public rule:

- normal user code chooses either reverse-mode or forward-mode annotation per value
- explicit public APIs should reject building mixed forward+reverse values directly

Internal rule:

- mixed forward-over-reverse is allowed inside dedicated higher-order derivative helpers

## HVP Design

Hessian-vector product is the reason to keep internal mixed-mode support available.

Public HVP should be exposed as a dedicated API instead of general mixed-mode construction.

Approved calling pattern:

```rust
let tape = GradTape::new();
let x = x.requires_grad(&tape)?;
let loss = f(&x)?.sum();
let hv = loss.hvp_tensors(&[(&x, &v)])?;
```

Recommended semantics:

- receiver is a scalar loss
- loss must be reverse-mode
- each `(wrt, direction)` pair must match in shape and indices
- `wrt` values must belong to the same `GradTape` as the loss
- `direction` values are ordinary primal tensors
- return type is `Vec<TensorDynLen>` in argument order

Implementation detail:

- HVP is implemented internally with forward-over-reverse
- this mixed-mode detail is hidden from the ordinary public API

## Error Handling

Construction and AD boundary errors should be explicit `Result` failures, not panics.

Important error classes:

- non-rank-0 payload provided to `Scalar`
- tensor/scalar native shape mismatch
- forward tangent shape mismatch
- reverse tape mismatch
- requesting gradients from non-reverse outputs
- requesting HVP on a non-scalar loss

## Testing Requirements

Add or update tests to cover:

- `Scalar` rank-0 invariant
- `AnyScalar = Scalar` compatibility
- scalar-returning tensor ops now return `Scalar`
- tensor-scalar mixed ops preserve expected values
- public API no longer exposes `from_native` / `as_native` / `into_native`
- forward-mode tangent preservation through scalar-returning ops
- reverse-mode gradient queries through `GradTape`
- HVP public API on scalar losses
- rejection of illegal mixed-mode public construction

Related-file follow-up to inspect during implementation because of similar risks:

- `crates/tensor4all-core/tests/tensor_native_ad.rs`
- `crates/tensor4all-itensorlike/tests/tensortrain_native_ad.rs`
- `crates/tensor4all-core/src/defaults/svd.rs`
- `crates/tensor4all-core/src/defaults/qr.rs`

## Migration Strategy

1. Introduce `Scalar` and internal native conversion helpers.
2. Switch `AnyScalar` alias to `Scalar`.
3. Update `TensorDynLen` scalar-returning and scalar-taking APIs.
4. Restrict native payload accessors to internal visibility.
5. Introduce `GradTape`, reverse-mode helpers, and forward tangent helpers.
6. Add dedicated HVP API backed by internal forward-over-reverse.
7. Update tests and docs to the new public model.

## Rationale

This design keeps tensor4all's user model cleaner than the raw upstream backend:

- tensors stay tensors
- scalar results are typed as scalars
- internal backend details do not leak into normal user code
- value semantics stay immutable
- higher-order AD remains possible through dedicated APIs without forcing all users to understand
  mixed-mode internals
