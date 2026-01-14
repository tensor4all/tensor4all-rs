# Project Structure

```
tensor4all-rs/
├── crates/
│   ├── tensor4all-tensorbackend/     # Scalar types, storage backends
│   ├── tensor4all-core/              # Core: Index, Tensor, SVD, QR
│   ├── tensor4all-simplett/          # Simple TT/MPS implementation
│   ├── tensor4all-tensorci/          # Tensor Cross Interpolation
│   ├── tensor4all-quanticstci/       # High-level Quantics TCI
│   ├── tensor4all-capi/              # C API for language bindings
│   ├── matrixci/                     # Matrix Cross Interpolation
│   ├── quanticsgrids/                # Quantics grid structures
│   ├── tensor4all-treetn/            # Tree Tensor Networks (WIP)
│   ├── tensor4all-itensorlike/       # ITensor-like API (WIP)
│   └── tensor4all-quanticstransform/ # Quantics transforms (WIP)
├── julia/Tensor4all.jl/              # Julia bindings
├── python/tensor4all/                # Python bindings
├── tools/api-dump/                   # API documentation generator
└── docs/                             # Documentation
    ├── book/                         # This mdBook
    └── api/                          # Generated API docs
```

## Active Crates

| Crate | Description |
|-------|-------------|
| `tensor4all-tensorbackend` | Scalar types (f64, Complex64) and storage backends |
| `tensor4all-core` | Core types: Index, Tensor, SVD, QR, LU |
| `tensor4all-simplett` | Simple TT/MPS with multiple canonical forms |
| `tensor4all-tensorci` | Tensor Cross Interpolation algorithms |
| `tensor4all-quanticstci` | High-level Quantics TCI interface |
| `tensor4all-capi` | C FFI for language bindings |
| `matrixci` | Matrix Cross Interpolation |
| `quanticsgrids` | Quantics grid structures |

## Work-in-Progress Crates

These crates are excluded from the workspace and need updates:

| Crate | Description |
|-------|-------------|
| `tensor4all-treetn` | Tree tensor networks with arbitrary topology |
| `tensor4all-itensorlike` | ITensors.jl-like TensorTrain API |
| `tensor4all-quanticstransform` | Quantics transformation operators |

## Dependency Graph

```
tensor4all-tensorbackend
    ↓
tensor4all-core
    ↓
tensor4all-simplett ← matrixci
    ↓
tensor4all-tensorci ← quanticsgrids
    ↓
tensor4all-quanticstci
    ↓
tensor4all-capi → Julia/Python bindings
```
