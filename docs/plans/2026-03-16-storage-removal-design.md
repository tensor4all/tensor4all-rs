# Design: Remove `Storage` from tensor4all Core Tensor APIs

**Date:** 2026-03-16
**Status:** Approved in chat

## Goal

Remove `Storage` as a first-class tensor representation from the tensor4all core layer.

After this change:

- `TensorDynLen` stores only `indices + native: tenferro::Tensor`
- `Scalar` stores only a rank-0 `tenferro::Tensor`
- public tensor construction uses generic constructors such as `from_dense<T>` and `from_diag<T>`
- `Storage` is deleted completely rather than retained as a boundary enum

## Why This Change

The current state is already close to a single-source-of-truth model:

- `TensorDynLen`'s canonical numeric payload is `tenferro::Tensor`
- `Storage` exists mostly for legacy constructors and materialization

Keeping both concepts alive creates avoidable tension:

- two representations for the same tensor value
- bridge code that must translate between them
- tests and docs that keep referring to the legacy representation
- pressure to add storage/native fallback branches again

Removing `Storage` finishes the transition to a single canonical compute object.

## Architectural Decision

### Canonical tensor representation

`TensorDynLen` is the only public tensor wrapper in tensor4all core.

Its internal shape is:

```rust
pub struct TensorDynLen {
    pub indices: Vec<DynIndex>,
    native: tenferro::Tensor,
}
```

This is the only representation that carries numeric state inside core tensor operations.

### Canonical scalar representation

`Scalar` remains a tensor4all-owned rank-0 wrapper:

```rust
pub struct Scalar {
    native: tenferro::Tensor,
}

pub type AnyScalar = Scalar;
```

`Scalar` remains distinct from `TensorDynLen` at the public API level even though upstream treats
scalars as rank-0 tensors.

### Constructors

Public construction becomes generic and typed by element type rather than by storage enum variant.

Required public constructors:

```rust
impl TensorDynLen {
    pub fn from_dense<T>(indices: Vec<DynIndex>, data: Vec<T>) -> Result<Self>;
    pub fn from_diag<T>(indices: Vec<DynIndex>, data: Vec<T>) -> Result<Self>;
}
```

Supported `T`:

- `f32`
- `f64`
- `Complex32`
- `Complex64`

This avoids dtype-specific names like `from_dense_f64` while still exposing the supported types.

### Scalar constructors

Scalar construction also becomes generic:

```rust
impl Scalar {
    pub fn from_value<T>(value: T) -> Result<Self>;
}
```

`From<T> for Scalar` can be added for the supported scalar types when ergonomic.

## Data Model Rules

### Dense construction

`from_dense<T>`:

- computes logical dims from `indices`
- validates index uniqueness
- validates `data.len() == product(dims)`
- imports values using tensor4all's row-major boundary semantics
- seeds a canonical `tenferro::Tensor`

### Diagonal construction

`from_diag<T>`:

- requires all index dimensions to match
- requires `data.len() == dim`
- imports the diagonal payload and constructs a diagonal native tensor

Dense vs diagonal remains a constructor-level semantic difference, not a runtime storage enum.

## Boundary Semantics

tensor4all keeps row-major boundary semantics.

tenferro now normalizes view operations to internal column-major semantics.

Therefore:

- imports from tensor4all constructors interpret input data as row-major
- native computations run entirely in `tenferro`
- row-major reinterpretation is centralized in backend helpers where needed

This matches the current upstream `origin/main` design.

## API Removals

The following APIs should be removed from the public core/backend surface:

- `TensorDynLen::new(indices, Arc<Storage>)`
- `TensorDynLen::from_storage(...)`
- `Storage`
- `DenseStorage*`
- `DiagStorage*`
- `DenseStorageFactory`
- `SumFromStorage`
- `to_storage()` style APIs that return `Storage`

If export/materialization remains necessary, it should be redesigned directly around
`tenferro::snapshot::DynTensor` or typed `Vec<T>` extraction rather than routing through `Storage`.

## Layering

The intended layering after this change is:

- `tensor4all-core`
  - index semantics
  - public tensor/scalar API
  - tensor network semantics
- `tensor4all-tensorbackend`
  - thin `tenferro` bridge
  - row-major boundary helpers
  - scalar/tensor element type traits
- `tenferro`
  - all numeric execution
  - AD
  - einsum
  - linalg
  - snapshots

There should be no second execution path through a tensor4all-owned storage enum.

## DRY / KISS / Layering Criteria

This change is complete only if:

- **DRY:** shape validation and dtype dispatch for tensor construction exist in one place
- **KISS:** there is one canonical tensor payload and one scalar payload
- **Layering:** core code no longer reaches through a storage abstraction that duplicates native data

If any core algorithm still needs `Storage`, that is a signal that export/import and execution
responsibilities are not yet separated cleanly enough.

## Testing Strategy

Tests should focus on public semantics, not on removed storage internals.

Required regression coverage:

- `from_dense<T>` validates shape and preserves row-major interpretation
- `from_diag<T>` validates diagonal rules
- QR/SVD regressions with unit dimensions still pass
- scalar/tensor AD regressions still pass
- dense and diagonal construction work for all supported scalar types

Tests should avoid matching on removed internal representations.
