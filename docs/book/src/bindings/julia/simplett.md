# SimpleTT

Simple tensor train (TT/MPS) interface with fixed site dimensions.

## Creating Tensor Trains

```julia
{{#include ../../../../examples/julia/simplett.jl:create}}
```

## Properties

```julia
{{#include ../../../../examples/julia/simplett.jl:properties}}
```

## Evaluation

```julia
{{#include ../../../../examples/julia/simplett.jl:evaluate}}
```

## Operations

```julia
{{#include ../../../../examples/julia/simplett.jl:operations}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `SimpleTensorTrain` | Construction (`constant`) and storage |
| `length`, `site_dims`, `link_dims`, `rank` | TT metadata |
| `evaluate`, `tt(i...)` | Point evaluation (0-based indices) |
| `sum`, `norm`, `site_tensor`, `copy` | Common operations |

