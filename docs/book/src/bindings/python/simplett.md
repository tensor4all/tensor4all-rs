# SimpleTT

Simple tensor train (TT/MPS) interface with fixed site dimensions.

## Creating Tensor Trains

```python
{{#include ../../../../examples/python/simplett.py:create}}
```

## Properties

```python
{{#include ../../../../examples/python/simplett.py:properties}}
```

## Evaluation

```python
{{#include ../../../../examples/python/simplett.py:evaluate}}
```

## Operations

```python
{{#include ../../../../examples/python/simplett.py:operations}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `SimpleTensorTrain` | Construction (`constant`, `zeros`) |
| `n_sites`, `site_dims`, `link_dims`, `rank` | TT metadata |
| `__call__` | Point evaluation (0-based indices) |
| `sum`, `norm`, `site_tensor`, `copy` | Common operations |

