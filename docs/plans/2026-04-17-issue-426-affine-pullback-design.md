# Issue 426 Affine Pullback C API Design

**Problem:** Tensor4all.jl can materialize the affine operator through the C API, but cannot yet materialize the symmetric affine pullback operator. This leaves `QuanticsTransform.affine_pullback_operator` as a Julia-side placeholder.

**Goal:** Add `t4a_qtransform_affine_pullback_materialize` to the C API while keeping the affine-family implementation DRY and preserving the current fused-layout contract.

## Requirements

- Export `t4a_qtransform_affine_pullback_materialize`.
- Reuse the existing affine-family parsing and validation path as much as possible.
- Keep the current fused-layout restriction for affine-family materialization.
- Document boundary-condition semantics in the pullback docstring.
- Add C API tests covering identity pullback and dense-matrix correctness.

## Design

### Public API

Add a new exported symbol in `crates/tensor4all-capi/src/quanticstransform.rs`:

- `t4a_qtransform_affine_pullback_materialize(...) -> StatusCode`

Its signature mirrors `t4a_qtransform_affine_materialize`.

### Internal structure

Extract the shared affine-family body behind a small helper that:

- validates `m > 0`, `n > 0`
- validates fused layout
- parses `a`, `b`, and `bc`
- constructs `AffineParams`
- invokes the selected Rust quantics operator builder
- wraps the result in `t4a_treetn`

The two exported C functions become thin wrappers over that helper:

- affine forward materialization
- affine pullback materialization

This avoids duplicating argument parsing, validation, and error mapping.

### Boundary semantics

The pullback docstring should explicitly state:

- `bc[i]` refers to source coordinate `i`
- when `(A * y + b)[i]` leaves the valid interval, `Periodic` wraps and `Open` zero-extends

This matches the Rust-side `affine_pullback_operator` contract, where `bc.len() == params.m`.

## Testing strategy

Add C API tests in `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs` that:

- verify identity pullback materialization matches the Rust reference operator
- verify a simple 1D pullback matches the expected dense matrix
- reuse existing helpers for dense operator extraction and matrix comparison

Keep tests fused-layout only, matching the current API contract.

## Non-goals

- No grouped/interleaved affine-family support in this issue.
- No broader refactor of non-affine QTT materialization APIs.
- No Tensor4all.jl changes in this repo; those follow after merge.
