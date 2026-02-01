# Core (Index, Tensor)

Core types for labeled indices and dense tensors.

## Creating Indices

```python
{{#include ../../../../examples/python/core.py:index_basic}}
```

## Index Utilities

```python
{{#include ../../../../examples/python/core.py:index_utils}}
```

## Creating Tensors

```python
{{#include ../../../../examples/python/core.py:tensor_basic}}
```

## One-Hot Tensors

```python
{{#include ../../../../examples/python/core.py:tensor_onehot}}
```

## NumPy Conversion

```python
{{#include ../../../../examples/python/core.py:tensor_numpy}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `Index` | Index construction and accessors |
| `tensor4all.index.sim` | Create a similar index with a new ID |
| `tensor4all.index.commoninds`, `uniqueinds`, `replaceinds` | Index-list utilities |
| `Tensor` | Dense tensor type |
| `Tensor.onehot`, `Tensor.to_numpy` | One-hot tensors and NumPy conversion |

