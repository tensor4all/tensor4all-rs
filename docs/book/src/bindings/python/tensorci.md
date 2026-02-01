# TensorCI

Tensor Cross Interpolation (TCI) for approximating high-dimensional functions as tensor trains.

## Basic Usage (crossinterpolate2)

```python
{{#include ../../../../examples/python/tensorci.py:basic}}
```

## Evaluating Results

```python
{{#include ../../../../examples/python/tensorci.py:evaluate}}
```

## Low-Level API (TensorCI2)

```python
{{#include ../../../../examples/python/tensorci.py:manual}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `crossinterpolate2` | High-level interpolation |
| `TensorCI2` | Low-level TCI object |
| `TensorCI2.add_global_pivots`, `TensorCI2.to_tensor_train` | Low-level operations |

