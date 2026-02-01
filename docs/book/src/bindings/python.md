# Python Bindings

The Python bindings provide a Pythonic interface to tensor4all-rs.

## Installation

```bash
pip install git+https://github.com/tensor4all/tensor4all-rs#subdirectory=python
```

## Overview

The Python package is organized into submodules; each submodule has its own page with executable examples.

## Module Structure

| Module | Description |
|--------|-------------|
| `tensor4all` | Re-exports commonly used APIs |
| `tensor4all.index` | `Index` and index utilities |
| `tensor4all.tensor` | `Tensor` and `StorageKind` |
| `tensor4all.simplett` | `SimpleTensorTrain` |
| `tensor4all.tensorci` | `TensorCI2`, `crossinterpolate2` |
| `tensor4all.tensortrain` | `TensorTrain` (MPS/MPO) |
| `tensor4all.algorithm` | Algorithm enums and tolerance helpers |
| `tensor4all.hdf5` | HDF5 I/O helpers |
