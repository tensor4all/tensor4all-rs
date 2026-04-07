# tensor4all-rs

[![CI](https://github.com/tensor4all/tensor4all-rs/actions/workflows/CI_rs.yml/badge.svg)](https://github.com/tensor4all/tensor4all-rs/actions/workflows/CI_rs.yml)

A Rust implementation of tensor networks: TCI, Quantics Tensor Train, and Tree Tensor Networks.

## Features

- **ITensors.jl-like dynamic tensors**: Flexible `Index` system with dynamic-rank `Tensor`
- **Tensor Cross Interpolation**: TCI2 algorithm for efficient high-dimensional function approximation
- **Quantics Tensor Train**: Binary encoding of continuous variables with transformation operators
- **Tree Tensor Networks**: Arbitrary topology with canonicalization, truncation, and contraction
- **C API**: Full functionality exposed for language bindings (Julia)

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tensor4all-simplett = "0.1"
```

```rust
use tensor4all_simplett::{AbstractTensorTrain, CompressionOptions, TensorTrain};

let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
let value = tt.evaluate(&[0, 1, 2]).unwrap();
assert!((value - 1.0).abs() < 1e-12);

let total = tt.sum();
assert!((total - 24.0).abs() < 1e-12);

let options = CompressionOptions {
    tolerance: 1e-10,
    max_bond_dim: 20,
    ..Default::default()
};
let compressed = tt.compressed(&options).unwrap();
assert!((compressed.sum() - 24.0).abs() < 1e-10);
```

## Crate Overview

| Crate | Description |
|-------|-------------|
| [tensor4all-core](crates/tensor4all-core/) | Core types: Index, Tensor, contraction, SVD, QR |
| [tensor4all-simplett](crates/tensor4all-simplett/) | Simple TT/MPS with compression |
| [tensor4all-itensorlike](crates/tensor4all-itensorlike/) | ITensors.jl-like TensorTrain API |
| [tensor4all-treetn](crates/tensor4all-treetn/) | Tree tensor networks with arbitrary topology |
| [tensor4all-tensorci](crates/tensor4all-tensorci/) | Tensor Cross Interpolation (TCI2) |
| [tensor4all-quanticstci](crates/tensor4all-quanticstci/) | High-level Quantics TCI interface |
| [tensor4all-quanticstransform](crates/tensor4all-quanticstransform/) | Quantics transformation operators |
| [tensor4all-treetci](crates/tensor4all-treetci/) | Tree-structured cross interpolation |
| [tensor4all-partitionedtt](crates/tensor4all-partitionedtt/) | Partitioned Tensor Train |
| [tensor4all-hdf5](crates/tensor4all-hdf5/) | ITensors.jl-compatible HDF5 serialization |
| [tensor4all-capi](crates/tensor4all-capi/) | C FFI for language bindings |

## Documentation

- **[User Guide](https://tensor4all.org/tensor4all-rs/)** — tutorials, architecture, conventions
- **[API Reference (rustdoc)](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_core/)** — generated API documentation
- **[Julia Bindings](https://github.com/tensor4all/Tensor4all.jl)** — Tensor4all.jl wrapper
- **[Design Documents](docs/design/index.md)** — architecture and design decisions

## Acknowledgments

Inspired by [ITensors.jl](https://github.com/ITensor/ITensors.jl). We acknowledge fruitful discussions with M. Fishman and E. M. Stoudenmire at CCQ, Flatiron Institute.

**Citation:** If you use this code in research, please cite:

> We used tensor4all-rs (https://github.com/tensor4all/tensor4all-rs), inspired by ITensors.jl.
>
> M. Fishman, S. R. White, E. M. Stoudenmire, "The ITensor Software Library for Tensor Network Calculations", arXiv:2007.14822 (2020)

## License

MIT License (see [LICENSE-MIT](LICENSE-MIT))
