# Julia Bindings

The Julia bindings for tensor4all-rs are maintained in a separate repository: **[Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl)**.

## Installation

To install Tensor4all.jl, use Julia's package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/tensor4all/Tensor4all.jl")
```

This will automatically download and build the Rust library via the C API (`tensor4all-capi`).

## Overview

The current C ABI is intentionally smaller than the full Rust workspace. It
exposes the pieces needed to build a Julia-native surface on top:

- **Indices** — immutable index handles with IDs, tags, and prime levels
- **Dense tensors** — `Float64` / complex tensor construction, export, and contraction
- **Tree tensor networks** — topology queries, orthogonalization, truncation, evaluation, and dense export
- **Canonical QTT layouts** — grouped, interleaved, and fused binary layouts
- **Quantics transform materialization** — shift, flip, phase rotation, cumsum, Fourier, binary-op, and affine operators materialized directly as `TreeTN`
- **Error reporting** — `StatusCode` plus `t4a_last_error_message`

This split is deliberate: the Rust side owns performance-critical kernels and
the Julia side owns higher-level ergonomics.

## ABI Conventions

The C API follows Julia-friendly conventions:

- Dense buffers are column-major
- Complex values are interleaved `Float64` pairs
- Variable-length outputs use a query-then-fill pattern
- Opaque handles must be released explicitly

The generated public header lives at
`crates/tensor4all-capi/include/tensor4all_capi.h`.

## Documentation

For Julia-side examples and package-level documentation, see the
**[Tensor4all.jl README](https://github.com/tensor4all/Tensor4all.jl)**.

For ABI details in this repository, see:

- [`docs/CAPI_DESIGN.md`](https://github.com/tensor4all/tensor4all-rs/blob/main/docs/CAPI_DESIGN.md)
- [`crates/tensor4all-capi/include/tensor4all_capi.h`](https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-capi/include/tensor4all_capi.h)

## Linking Rust and Julia Code

If you want to use tensor4all-rs directly in a Rust project and interoperate
with Julia, build against the generated C header and follow the conventions
documented in `docs/CAPI_DESIGN.md`.
