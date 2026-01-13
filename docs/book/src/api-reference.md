# API Reference

The full Rust API documentation is generated using `rustdoc` and deployed alongside this book.

## Online Documentation

- **API Docs**: [tensor4all-rs API](../rustdoc/tensor4all_core/index.html)

## Generating Locally

```bash
# Generate HTML documentation
cargo doc --workspace --no-deps --open

# Generate API summary (Markdown)
cargo run -p api-dump --release -- . -o docs/api
```

## Crate Overview

| Crate | Purpose |
|-------|---------|
| `tensor4all-core` | Core types: Index, Tensor, SVD, QR |
| `tensor4all-simplett` | Simple TT/MPS implementation |
| `tensor4all-tensorci` | Tensor Cross Interpolation |
| `tensor4all-quanticstci` | High-level Quantics TCI |
| `tensor4all-capi` | C FFI for language bindings |
| `tensor4all-tensorbackend` | Scalar types and storage |
