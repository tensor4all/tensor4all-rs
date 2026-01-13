# Python Bindings

The Python bindings provide a Pythonic interface to tensor4all-rs.

## Installation

```bash
pip install git+https://github.com/tensor4all/tensor4all-rs#subdirectory=python
```

## Basic Usage

### Tensor Cross Interpolation

```python
from tensor4all import crossinterpolate2

# Define a function to interpolate
def f(i, j, k):
    return float((1 + i) * (1 + j) * (1 + k))

# Perform cross interpolation
tt, err = crossinterpolate2(f, [4, 4, 4], tolerance=1e-10)

# Evaluate the tensor train
print(tt(0, 0, 0))  # 1.0
print(tt(1, 1, 1))  # 8.0
print(tt(3, 3, 3))  # 64.0

# Check properties
print("Rank:", tt.rank)
print("Site dims:", tt.site_dims)
print("Sum:", tt.sum())
```

## Index and Tensor (Advanced)

For more control, use the low-level Index and Tensor types:

```python
from tensor4all import Index, Tensor
import numpy as np

# Create indices
i = Index(2, tags="Site")
j = Index(3, tags="Link")

# Create tensor from NumPy array
data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
t = Tensor([i, j], data)

# Convert back to NumPy
arr = t.to_numpy()
```

## NumPy Integration

All tensor data can be converted to and from NumPy arrays:

```python
import numpy as np
from tensor4all import TensorTrain

# Create a tensor train
tt = TensorTrain.constant([2, 3, 4], 1.0)

# Get cores as NumPy arrays
cores = [tt.core(i) for i in range(tt.num_sites)]

# Sum over all indices
total = tt.sum()
print(f"Sum: {total}")  # 24.0 (= 2 * 3 * 4)
```

## Module Structure

| Module | Description |
|--------|-------------|
| `tensor4all` | Main module with high-level functions |
| `tensor4all.tensortrain` | Tensor train operations |
| `tensor4all.tci` | Cross interpolation |
