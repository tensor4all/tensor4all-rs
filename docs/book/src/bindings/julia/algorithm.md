# Algorithm

Algorithm selection constants and truncation tolerance utilities.

## Factorize Algorithms

```julia
{{#include ../../../../examples/julia/algorithm.jl:factorize}}
```

## Contraction Algorithms

```julia
{{#include ../../../../examples/julia/algorithm.jl:contraction}}
```

## Compression Algorithms

```julia
{{#include ../../../../examples/julia/algorithm.jl:compression}}
```

## Tolerance Resolution

```julia
{{#include ../../../../examples/julia/algorithm.jl:tolerance}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `Algorithm.FactorizeAlgorithm` | SVD/LU/CI selection |
| `Algorithm.ContractionAlgorithm` | Naive/ZipUp/Fit selection |
| `Algorithm.CompressionAlgorithm` | SVD/LU/CI/Variational selection |
| `get_default_svd_rtol` | Default truncation tolerance |
| `resolve_truncation_tolerance` | `cutoff` ↔ `rtol` conversion |

