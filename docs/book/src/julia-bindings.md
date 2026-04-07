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

Tensor4all.jl provides a high-level Julia interface to tensor4all-rs, including:

- **Tensor operations** — creation, manipulation, and contraction
- **Tensor Trains** — construction and compression
- **Tree Tensor Networks** — general tensor network representations
- **Tensor Cross Interpolation (TCI)** — efficient sampling and interpolation
- **Quantics Transforms** — quantization and structured encodings

The Julia package bridges to the Rust core through the C API, so you benefit from the performance and correctness of the underlying Rust implementations.

## Documentation

For detailed documentation, examples, and API reference, visit the **[Tensor4all.jl README](https://github.com/tensor4all/Tensor4all.jl)** on GitHub.

## Linking Rust and Julia Code

If you want to use tensor4all-rs directly in a Rust project and interoperate with Julia, see the [C API reference](../api/capi/index.md) for FFI details.
