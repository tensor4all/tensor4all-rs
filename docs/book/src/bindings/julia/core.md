# Core (Index, Tensor)

Core types for labeled indices and dense tensors.

## Creating Indices

```julia
{{#include ../../../../examples/julia/core.jl:index_basic}}
```

## Index Utilities

```julia
{{#include ../../../../examples/julia/core.jl:index_utils}}
```

## Creating Tensors

```julia
{{#include ../../../../examples/julia/core.jl:tensor_basic}}
```

## One-Hot Tensors

```julia
{{#include ../../../../examples/julia/core.jl:tensor_onehot}}
```

## Array Conversion (with index reordering)

```julia
{{#include ../../../../examples/julia/core.jl:tensor_array}}
```

## Complex Tensors

```julia
{{#include ../../../../examples/julia/core.jl:tensor_complex}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `Index`, `dim`, `id`, `tags`, `hastag` | Index construction and accessors |
| `sim`, `copy` | Index cloning utilities |
| `commoninds`, `uniqueinds`, `replaceinds` | Index-list utilities |
| `Tensor`, `rank`, `dims`, `indices` | Tensor construction and basic metadata |
| `onehot`, `Array(t, inds)` | One-hot tensors and array conversion |

