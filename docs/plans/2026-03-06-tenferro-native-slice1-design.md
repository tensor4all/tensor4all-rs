# Design: Tenferro Native Slice 1 for TensorDynLen

**Date:** 2026-03-06
**Status:** Approved in chat

## Goal

Introduce a first native tenferro-backed execution path for `TensorDynLen` without breaking
`treetn`-level behavior. The first slice targets tensor algebra operations that are central to
`treetn` and can benefit immediately from AD-aware tenferro primitives.

## Scope

This slice changes:

- tenferro dependency pin to the latest reviewed main commit
- native conversion helpers between `Storage` and `DynAdTensor`
- `TensorDynLen` tensor algebra operations to use native execution:
  - `sum`
  - `only`
  - `conj`
  - `scale`
  - `axpby`
  - `inner_product`
  - `permute`
  - `contract`
  - `outer_product`

This slice does not yet change:

- the public `TensorDynLen` shape/index API
- HDF5 / C API representation
- factorization/truncation internals
- removal of `Storage` / `TensorData` from public API

## Design Boundary

`TensorDynLen` keeps ownership of tensor4all index semantics.

Tenferro owns:
- numeric tensor payload
- dynamic scalar/tensor AD metadata
- contraction and mixed scalar/tensor primitives

Tensor4all owns:
- `DynIndex`
- index alignment and contraction pairing logic
- `treetn` semantics and algorithms

## Compatibility Strategy

The internal execution path is switched first, but `Storage`-based APIs remain available through
materialization bridges. This keeps `treetn` and surrounding crates working while reducing the
amount of old backend code touched in the first slice.

## Key Decisions

### 1. Native execution is opt-in inside `TensorDynLen` methods

We will not replace the stored representation in one step. Instead, we add native conversion
helpers and route core algebraic methods through tenferro.

This gives us:
- smaller blast radius
- immediate AD-aware execution for important operations
- continued compatibility with storage-oriented crates

### 2. `AnyScalar` remains a tensor4all re-export

`AnyScalar` stays as the tensor4all-visible name, but it continues to alias tenferro's dynamic AD
scalar type. No new scalar abstraction is introduced in this slice.

### 3. Diagonal tensors must survive the bridge

Diagonal `Storage` must convert to native tensors without losing logical diagonal semantics where
possible. If the current tenferro bridge cannot preserve diagonal structure directly, it may
materialize to dense for this slice, but the API boundary should keep room for a later diagonal-
preserving bridge.

## Testing

The required tests for this slice are:

- `TensorDynLen::scale` preserves scalar AD metadata through the native path
- `TensorDynLen::axpby` preserves scalar AD metadata through the native path
- `TensorDynLen::inner_product` returns an AD-aware scalar when driven by native execution
- `TensorDynLen::contract` and `outer_product` still match existing numeric behavior
- focused `treetn` smoke tests continue to pass

## Follow-up Slices

- replace `TensorDynLen` stored payload with `TensorRepr::Native(DynAdTensor)`
- cut factorization/truncation over to native linalg AD contracts
- demote `Storage` / `TensorData` from public API surface
