# Julia Bindings

The Julia bindings provide a native Julia interface to tensor4all-rs.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/tensor4all/tensor4all-rs", subdir="julia/Tensor4all.jl")
```

## Overview

The Julia package is organized into submodules; each submodule has its own page with executable examples.

## Module Structure

| Module | Description |
|--------|-------------|
| `Tensor4all` | Core types: Index, Tensor |
| `Tensor4all.SimpleTT` | Simple tensor train operations |
| `Tensor4all.TensorCI` | Cross interpolation |
| `Tensor4all.ITensorLike` | ITensors.jl-like interface |

## Tolerance Parameters

The Julia bindings support both `rtol` (tensor4all-rs style) and `cutoff` (ITensors.jl style):

Note: `cutoff = rtol²`, so `rtol = √cutoff`.
