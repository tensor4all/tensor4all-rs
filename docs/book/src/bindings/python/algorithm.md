# Algorithm

Algorithm selection enums and truncation tolerance utilities.

## Factorize Algorithms

```python
{{#include ../../../../examples/python/algorithm.py:factorize}}
```

## Contraction Algorithms

```python
{{#include ../../../../examples/python/algorithm.py:contraction}}
```

## Compression Algorithms

```python
{{#include ../../../../examples/python/algorithm.py:compression}}
```

## Tolerance Resolution

```python
{{#include ../../../../examples/python/algorithm.py:tolerance}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `FactorizeAlgorithm` | SVD/LU/CI selection |
| `ContractionAlgorithm` | Naive/ZipUp/Fit selection |
| `CompressionAlgorithm` | SVD/LU/CI/Variational selection |
| `get_default_svd_rtol` | Default truncation tolerance |
| `resolve_truncation_tolerance` | `cutoff` ↔ `rtol` conversion |

