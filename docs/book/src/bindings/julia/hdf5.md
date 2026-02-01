# HDF5

ITensors.jl-compatible HDF5 serialization helpers.

## Save/Load Tensor

```julia
{{#include ../../../../examples/julia/hdf5.jl:save_load_tensor}}
```

## Save/Load MPS

```julia
{{#include ../../../../examples/julia/hdf5.jl:save_load_mps}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `save_itensor`, `load_itensor` | Tensor I/O |
| `ITensorLike.save_mps`, `ITensorLike.load_mps` | MPS I/O |

