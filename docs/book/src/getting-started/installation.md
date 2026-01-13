# Installation

## Rust

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
tensor4all-simplett = { git = "https://github.com/tensor4all/tensor4all-rs" }
tensor4all-tensorci = { git = "https://github.com/tensor4all/tensor4all-rs" }
```

Or clone the repository for development:

```bash
git clone https://github.com/tensor4all/tensor4all-rs.git
cd tensor4all-rs
cargo build --release
```

## Julia

The Julia bindings are available in the `julia/Tensor4all.jl` directory:

```julia
using Pkg
Pkg.add(url="https://github.com/tensor4all/tensor4all-rs", subdir="julia/Tensor4all.jl")
```

## Python

The Python bindings are available in the `python/tensor4all` directory:

```bash
pip install git+https://github.com/tensor4all/tensor4all-rs#subdirectory=python
```

## Building from Source

### Prerequisites

- Rust 1.70 or later
- A C compiler (for BLAS/LAPACK backends)

### Build Commands

```bash
# Build all crates
cargo build --release

# Run tests
cargo test --workspace

# Generate API documentation
cargo doc --workspace --no-deps --open
```
