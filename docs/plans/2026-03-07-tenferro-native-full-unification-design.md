# Design: Tenferro Native Full Unification for TensorDynLen

**Date:** 2026-03-07
**Status:** Approved in chat

## Goal

Make `DynAdTensor` the single source of truth for `TensorDynLen` so that tensor4all's tensor
backend is fully AD-aware and routed through tenferro.

This removes the current hybrid state where `TensorDynLen` stores both legacy `TensorData` /
`Storage` state and an optional native payload.

## Scope

This redesign changes:

- `TensorDynLen` internal representation
- tensor algebra execution paths
- linalg / factorization entry points
- extraction / materialization boundaries
- downstream crate usage that still depends on `storage()` / `tensor_data()`

This redesign does not change:

- `DynIndex` semantics
- `treetn` / `TensorTrain` high-level algorithms
- HDF5 file format
- C-API observable tensor semantics

## Final Representation

`TensorDynLen` becomes a thin wrapper over tenferro-native payload plus index semantics:

```rust
pub struct TensorDynLen {
    pub indices: Vec<DynIndex>,
    native: DynAdTensor,
}
```

There is no internal `TensorData`.
There is no internal `Storage`.
There is no optional native payload.

## Boundary APIs

The new canonical construction and extraction APIs are:

- `TensorDynLen::from_native(indices, native) -> Result<Self>`
- `TensorDynLen::from_storage(indices, storage) -> Result<Self>`
- `TensorDynLen::as_native(&self) -> &DynAdTensor`
- `TensorDynLen::into_native(self) -> DynAdTensor`
- `TensorDynLen::to_storage(&self) -> Result<Arc<Storage>>`

`Storage` becomes an explicit interop boundary for:

- HDF5
- C-API
- debug / inspection
- legacy import and export

## Removed APIs

The following APIs should be removed as part of this redesign:

- `TensorDynLen::new(indices, Arc<Storage>) -> Self`
- `TensorDynLen::from_indices(indices, Arc<Storage>) -> Self`
- `TensorDynLen::storage()`
- `TensorDynLen::try_storage()`
- `TensorDynLen::materialize_storage()`
- `TensorDynLen::tensor_data()`
- the `TensorData` type itself

Backward compatibility is not required in this repository stage, so compatibility shims should be
kept to an absolute minimum.

## Execution Model

All tensor algebra must operate directly on `DynAdTensor`:

- `permute`
- `contract`
- `tensordot`
- `outer_product`
- `axpby`
- `scale`
- `conj`
- `sum`
- `only`
- `inner_product`
- `replaceind(s)`
- `qr`
- `svd`
- `factorize`

No operation should dispatch on `Storage` to decide the backend.
If an operation needs a storage snapshot, that must be explicit and justified as an interop step.

## Diagonal and Structured Tensors

Structured tensors remain a tenferro concern.
`TensorDynLen` should not reintroduce its own representation enum or lazy outer-product graph.

`Diag` and dense tensors are carried through tenferro structured payloads. If a structured native
operation is unavailable, the fallback should happen inside tenferro or at the explicit snapshot
boundary, not by restoring `TensorData`.

## Error Handling

Construction becomes fallible because native conversion is fallible:

- invalid index / payload dimension mismatch
- unsupported storage/native conversion
- unsupported scalar/layout combinations at the tenferro boundary

Library code should return `Result` with context rather than panic for backend conversion failures.

## Testing Requirements

The redesign must preserve:

- `tensor4all-core` algebra correctness
- `qr` / `svd` / `factorize` reconstruction
- `treetn` orthogonalization and truncation behavior
- `TensorTrain` zipup contraction behavior
- existing HDF5 and C-API round-trip behavior via explicit `to_storage()`

Additional regressions must cover:

- nontrivial split linalg on native tensors
- multi-tensor native contraction equivalence
- explicit storage materialization boundaries
- `Diag` preservation through native payloads where supported

## Migration Strategy

This should be done in phases, but the end state is full unification:

1. make `native` mandatory inside `TensorDynLen`
2. move all core operations to native-only execution
3. switch extraction and linalg to native-only entry points
4. update downstream crates to explicit `to_storage()` boundaries
5. delete `TensorData`

The key rule is that each intermediate phase must keep the workspace green.

## Rationale

The current hybrid model was useful as a migration bridge, but it is not a good long-term design:

- it duplicates state
- it risks desynchronization bugs
- it obscures AD-preserving versus AD-dropping boundaries
- it keeps backend dispatch logic spread across storage and native paths

Canonical tenferro-native ownership is the simplest clean design.
