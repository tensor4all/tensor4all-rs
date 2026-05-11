# Design: initial `tensor4all-interpolativeqtt` Rust port

## Goal

Add a new workspace crate, `tensor4all-interpolativeqtt`, that ports the
tested public construction surface from
[`InterpolativeQTT.jl`](https://github.com/tensor4all/InterpolativeQTT.jl)
into Rust so it can be exercised against the existing `tensor4all-simplett`
tensor train implementation.

## Scope

- Port the Chebyshev-Lobatto Lagrange basis:
  - `LagrangePolynomials`
  - `get_chebyshev_grid`
  - dense interpolation tensor construction
- Port interpolation constructors:
  - 1D interval API
  - N-dimensional fused-site API
- Port single-scale, multiscale, adaptive, and sparse single-scale variants.
- Port direct-product core construction for fused multivariate cores.
- Port `invert_qtt` and interpolation error estimation.
- Return `tensor4all_simplett::TensorTrain<f64>`.
- Use `TensorTrain::compressed` with SVD options for optional truncation,
  which routes SVD through the tensor4all/tenferro backend stack.
- Add Rust tests mirroring the Julia package tests.
- Julia/C/Python bindings.

## API Shape

The initial public API should stay small:

- `LagrangePolynomials`
- `InterpolativeQttOptions`
- `get_chebyshev_grid`
- `interpolation_tensor`
- `direct_product_core_tensors`
- `interpolate_single_scale`
- `interpolate_single_scale_nd`
- `interpolate_multi_scale`
- `interpolate_multi_scale_nd`
- `interpolate_adaptive`
- `interpolate_adaptive_nd`
- `interpolate_single_scale_sparse`
- `interpolate_single_scale_sparse_nd`
- `invert_qtt`
- `estimate_interpolation_error`
- `estimate_interpolation_error_nd`

`InterpolativeQttOptions` should include:

- `tolerance`: relative SVD compression tolerance.
- `max_bond_dim`: hard cap on TT bond dimension.

When `tolerance == 0.0` and `max_bond_dim == usize::MAX`, the uncompressed
exact construction is returned.

## Design Notes

- Julia arrays and Rust flat buffers are both treated as column-major at the
  storage boundary. Rust `Tensor3::from_fn` and `TensorTrain::evaluate` still
  use normal zero-based Rust indexing. Tests should validate behavior through
  public TT evaluation APIs.
- `quanticsgrids::DiscretizedGrid` remains the source of truth for converting
  grid indices to fused quantics indices and physical coordinates.
- The 1D interpolation construction can be expressed as a special case of the
  N-dimensional fused construction, but keep a dedicated 1D wrapper for
  ergonomic parity with Julia.
- Compression uses existing `tensor4all-simplett` SVD compression and therefore
  the same tenferro-backed linear algebra route as other crates.

## Validation

- `cargo fmt --all`
- `cargo test -p tensor4all-interpolativeqtt --release`
- `cargo test -p tensor4all-interpolativeqtt --doc --release`

## Follow-Up

- Consider a `QuanticsInterpolant` wrapper if callers need grid-aware
  coordinate evaluation rather than raw `TensorTrain` evaluation.
