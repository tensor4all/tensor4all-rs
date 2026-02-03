# TensorTrain (MPS/MPO)

ITensorMPS.jl-inspired tensor train interface with orthogonality tracking.

## Creating TensorTrain

```python
{{#include ../../../../examples/python/tensortrain.py:create}}
```

## Accessors

```python
{{#include ../../../../examples/python/tensortrain.py:accessors}}
```

## Orthogonalization

```python
{{#include ../../../../examples/python/tensortrain.py:orthogonalize}}
```

## Truncation

```python
{{#include ../../../../examples/python/tensortrain.py:truncate}}
```

## Operations

```python
{{#include ../../../../examples/python/tensortrain.py:operations}}
```

## Linear Solver

```python
{{#include ../../../../examples/python/tensortrain.py:linsolve}}
```

## API Summary

| Function/Type | Description |
|---|---|
| `TensorTrain` / `MPS` / `MPO` | Tensor train types |
| `TensorTrain.orthogonalize`, `TensorTrain.truncate` | Canonicalization and truncation |
| `TensorTrain.contract`, `TensorTrain.add`, `TensorTrain.norm`, `TensorTrain.inner` | Core operations |
| `tensor4all.tt_linsolve` | Solve `(a0 + a1*A) * x = b` |

