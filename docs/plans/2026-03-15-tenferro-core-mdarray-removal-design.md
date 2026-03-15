# Design: Remove `mdarray` from tensor4all Core Execution Paths

**Date:** 2026-03-15
**Status:** Approved in chat

## Goal

Remove `mdarray` from the `tensor4all-core` and `tensor4all-tensorbackend` execution paths,
starting with:

- `TensorDynLen`
- einsum / contraction
- linalg (`qr`, `svd`, `factorize`)

The new canonical execution backend is `tenferro::Tensor`.

## Scope

This design is intentionally limited to the core/backend layer.

In scope:

- `crates/tensor4all-core`
- `crates/tensor4all-tensorbackend`
- public tensor/scalar API used by core crates
- storage/materialization boundaries needed by core, HDF5, and C API

Out of scope for this phase:

- `tensor4all-simplett`
- `tensor4all-quanticstransform`
- other downstream crates that still use `mdarray` as a local dense container
- complete workspace-wide `mdarray` removal

## Upstream Baseline

The reviewed upstream target is `tenferro-rs` `origin/main` after:

- frontend API gap closure (`permute`, `conj`, `primal_snapshot`, `try_scalar_value`)
- `tenferro-ndarray` bridge crate merge

Important assumptions:

- `tenferro::Tensor` is the public compute object
- dynamic einsum exists as `Tensor::einsum(subscripts, &[&Tensor])`
- linalg lives on `tenferro::Tensor`
- `snapshot::DynTensor` and `to_dense()` are available for materialization boundaries

`#516` (consuming einsum frontend) is useful but not a blocker for this phase.

## Architectural Decision

### Canonical execution object

`tensor4all` core execution should operate on `tenferro::Tensor`, not on:

- `mdarray::DTensor`
- `mdarray::DSlice`
- storage-backed dense intermediates
- old `DynAdTensor` / `DynAdScalar` public surfaces

This means:

- `TensorDynLen` stores `native: tenferro::Tensor`
- `Scalar` stores a rank-0 `tenferro::Tensor`
- einsum and linalg delegate directly to `tenferro`

### Storage boundary

`Storage` remains temporarily, but only as a boundary for:

- `from_storage`
- `to_storage`
- C API / HDF5 / debugging / explicit extraction

`Storage` is no longer an execution backend.

### Index semantics remain in tensor4all

`tenferro` owns numeric execution.

`tensor4all` still owns:

- `DynIndex`
- index equality / order / replacement
- contraction pairing logic
- tensor-network semantics
- `TensorDynLen` vs `Scalar` type separation

## Public API Model

### Tensor

`TensorDynLen` remains the main user-facing tensor type.

Its responsibilities:

- own tensor4all indices
- validate index uniqueness and order-sensitive operations
- delegate numeric work to `tenferro::Tensor`

It should no longer expose native payload escape hatches publicly.

### Scalar

`Scalar` is a rank-0 wrapper over `tenferro::Tensor`.

Keep:

```rust
pub type AnyScalar = Scalar;
```

This preserves the existing name while moving to the new implementation.

### Native boundary policy

The following APIs should not remain public:

- `TensorDynLen::from_native`
- `TensorDynLen::as_native`
- `TensorDynLen::into_native`

Equivalent helpers may remain internal for backend wiring and tests.

## Execution Rules

### TensorDynLen

`TensorDynLen` methods should follow these rules:

- `permute`, `conj`, `reshape`, `contract`, `outer_product`, `diag`, `sum`, `only`,
  `scale`, `axpby`, `inner_product` delegate to `tenferro::Tensor`
- `sum`, `only`, `inner_product` return `Scalar`
- no `mdarray` matrix/tensor conversion is allowed in these methods

### Einsum and contraction

There should be a single execution path:

- tensor4all computes label/index mapping
- backend calls `tenferro::Tensor::einsum`

The storage-level einsum facade in `tensor4all-tensorbackend` becomes dead code and should be
removed in this phase.

### Linalg

There should be a single execution path:

- tensor4all prepares unfolding / index bookkeeping
- backend calls `tenferro::Tensor::{qr, svd, ...}`
- tensor4all wraps outputs back into `TensorDynLen` / `Scalar`

The `DTensor`/`DSlice` wrappers and `backend.rs` adapter layer should be removed.

## `mdarray` Removal Target for This Phase

At the end of this phase, the following should be true:

- `tensor4all-core` no longer imports `mdarray`
- `tensor4all-tensorbackend` no longer imports or re-exports `mdarray`
- no core einsum path relies on storage-level dense conversion
- no core linalg path relies on `DTensor` / `DSlice`

`mdarray` may still remain in downstream crates outside this phase.

## Testing Strategy

Testing should shift from low-level native/storage assumptions to public semantics.

Core regression coverage should emphasize:

- `TensorDynLen` shape/index semantics still work
- scalar-returning methods return `Scalar`
- einsum/contraction preserve tensor4all external index semantics
- `qr`, `svd`, and `factorize` still preserve reconstruction/truncation behavior
- materialization through `to_storage()` still works where intentionally supported

Old tests that depend on public `DynAdTensor` access should be rewritten or removed.

## Cleanup Standard

This phase ends with an explicit review pass for:

- **DRY**: no duplicated promotion/shape/materialization logic across core and backend
- **KISS**: no parallel storage-based fallback execution path remains for einsum/linalg
- **Layering**: core does not reach into low-level native details directly; all native execution
  goes through the backend facade

If a remaining code path violates those constraints, it should be fixed before the phase is
considered complete.
