# TensorCI

Tensor Cross Interpolation (TCI) for approximating high-dimensional functions as tensor trains.

## Basic Usage (crossinterpolate2)

```julia
{{#include ../../../../examples/julia/tensorci.jl:basic}}
```

## Evaluating Results

```julia
{{#include ../../../../examples/julia/tensorci.jl:evaluate}}
```

## Low-Level API (TensorCI2)

```julia
{{#include ../../../../examples/julia/tensorci.jl:manual}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `crossinterpolate2` | High-level interpolation |
| `TensorCI2` | Low-level TCI object |
| `add_global_pivots!`, `to_tensor_train` | Low-level operations |

