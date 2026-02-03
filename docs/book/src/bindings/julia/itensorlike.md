# ITensorLike (MPS/MPO)

ITensorMPS.jl-inspired tensor train interface with orthogonality tracking.

## Creating TensorTrain

```julia
{{#include ../../../../examples/julia/itensorlike.jl:create}}
```

## Accessors

```julia
{{#include ../../../../examples/julia/itensorlike.jl:accessors}}
```

## Orthogonalization

```julia
{{#include ../../../../examples/julia/itensorlike.jl:orthogonalize}}
```

## Truncation

```julia
{{#include ../../../../examples/julia/itensorlike.jl:truncate}}
```

## Operations

```julia
{{#include ../../../../examples/julia/itensorlike.jl:operations}}
```

## Site Indices

```julia
{{#include ../../../../examples/julia/itensorlike.jl:siteinds}}
```

## Random Tensor Trains

```julia
{{#include ../../../../examples/julia/itensorlike.jl:random}}
```

## Linear Solver

```julia
{{#include ../../../../examples/julia/itensorlike.jl:linsolve}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `TensorTrain` / `MPS` / `MPO` | Tensor train types |
| `orthogonalize!`, `truncate!` | Canonicalization and truncation |
| `contract`, `add`, `norm`, `inner`, `to_dense` | Core operations |
| `siteinds`, `findsite`, `findsites` | Site index helpers |
| `linsolve` | Solve `(a0 + a1*A) * x = b` |

