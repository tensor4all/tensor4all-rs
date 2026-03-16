# Column-Major Layout Migration Audit

**Date:** 2026-03-17

This note captures the current layout-sensitive code paths before the migration from row-major to column-major semantics.

## Backend

- `crates/tensor4all-tensorbackend/src/storage.rs`
  - `compute_strides` currently computes row-major strides.
  - Dense/diag contraction helpers directly use stride math and flat offsets.
  - Several unit tests encode row-major dense literals.
- `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
  - `reshape_row_major_native_tensor`
  - `native_tensor_primal_to_dense_*`
  - row-major boundary materialization helpers
- `crates/tensor4all-tensorbackend/src/tensor_element.rs`
  - row-major native tensor construction / materialization trait hooks

## Core

- `crates/tensor4all-core/src/defaults/tensordynlen.rs`
  - `row_major_offset`
  - `from_dense` / `to_dense_*` docstrings and semantics
  - reshape to matrix paths for factorization / unfolding
- `crates/tensor4all-core/src/defaults/qr.rs`
  - calls `reshape_row_major_native_tensor`
  - tests/comments mention row-major boundary semantics
- `crates/tensor4all-core/src/defaults/svd.rs`
  - calls `reshape_row_major_native_tensor`
  - tests/comments mention row-major boundary semantics
- `crates/tensor4all-core/src/defaults/direct_sum.rs`
  - computes strides locally instead of delegating to backend
- `crates/tensor4all-core/src/block_tensor.rs`
  - block ordering documented as row-major
- `crates/tensor4all-core/src/krylov.rs`
  - comments and test data describe row-major matrices

## Boundary / Bindings

- `crates/tensor4all-capi/src/tensor.rs`
  - dense get/set explicitly documented as row-major
- `crates/tensor4all-capi/src/simplett.rs`
  - simple TT payload documented and copied as row-major
- `python/tensor4all/src/tensor4all/tensor.py`
  - tensor dense storage described as row-major
  - imports normalize to C-contiguous arrays
- `python/tensor4all/src/tensor4all/simplett.py`
  - reshape/transpose logic assumes Rust row-major payloads
- `python/tensor4all/tests/test_tensor.py`
  - tests assert C-order behavior

## HDF5

- `crates/tensor4all-hdf5/src/lib.rs`
  - explicitly documents tensor4all-rs row-major vs ITensors.jl column-major
- `crates/tensor4all-hdf5/src/layout.rs`
  - row-major <-> column-major conversion utilities

## Documentation

- `README.md`
  - currently does not document column-major semantics yet
- `docs/CAPI_DESIGN.md`
  - entire dense-layout section assumes row-major C API semantics

## Immediate migration hotspots

The first code to flip safely is:

1. `tensor4all-tensorbackend/src/storage.rs`
2. `tensor4all-tensorbackend/src/tenferro_bridge.rs`
3. `tensor4all-core/src/defaults/tensordynlen.rs`
4. `tensor4all-core/src/defaults/qr.rs`
5. `tensor4all-core/src/defaults/svd.rs`

These files determine the effective dense layout semantics for most of the stack.

