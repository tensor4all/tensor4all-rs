# InterpolativeQTT Rust Port Implementation Plan

> **For agentic workers:** use the superpowers executing-plans style. Implement
> task-by-task, updating checkboxes as work completes.

**Goal:** Add `crates/tensor4all-interpolativeqtt` with a Rust port of the
tested `InterpolativeQTT.jl` surface.

**Spec:** `docs/superpowers/specs/2026-05-10-interpolativeqtt-rust-port-design.md`

## Task 1: Workspace Crate

- [x] Add `crates/tensor4all-interpolativeqtt/Cargo.toml`.
- [x] Register the crate in the workspace `Cargo.toml`.
- [x] Add crate-level rustdoc with runnable examples.

## Task 2: Basis And Core Helpers

- [x] Implement `LagrangePolynomials`.
- [x] Implement `get_chebyshev_grid`.
- [x] Implement `interpolation_tensor`.
- [x] Implement `direct_product_core_tensors`.

## Task 3: Interpolation Constructors

- [x] Implement `InterpolativeQttOptions`.
- [x] Implement `interpolate_single_scale`.
- [x] Implement `interpolate_single_scale_nd`.
- [x] Implement `interpolate_multi_scale`.
- [x] Implement `interpolate_multi_scale_nd`.
- [x] Implement `interpolate_adaptive`.
- [x] Implement `interpolate_adaptive_nd`.
- [x] Implement sparse single-scale constructors.
- [x] Use SVD compression through `tensor4all-simplett`.

## Task 4: Inversion And Error Estimation

- [x] Implement interval helpers.
- [x] Implement interpolation error estimators.
- [x] Implement `invert_qtt`.

## Task 5: Tests

- [x] Add 1D accuracy test against `quanticsgrids::DiscretizedGrid`.
- [x] Add N=2 fused accuracy test.
- [x] Add N=3 fused accuracy test.
- [x] Add multiscale tests.
- [x] Add `invert_qtt` tests.
- [x] Add interpolation error estimator tests.
- [x] Add direct-product core tests.
- [x] Add validation tests for invalid arguments.

## Task 6: Verification

- [x] Run `cargo fmt --all`.
- [x] Run `cargo test -p tensor4all-interpolativeqtt --release`.
- [x] Run `cargo test -p tensor4all-interpolativeqtt --doc --release`.
