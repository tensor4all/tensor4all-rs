# Julia Bindings

The Julia bindings provide a native Julia interface to tensor4all-rs.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/tensor4all/tensor4all-rs", subdir="julia/Tensor4all.jl")
```

## Basic Usage

### Tensor Cross Interpolation

```julia
using Tensor4all
using Tensor4all.TensorCI

# Cross interpolation of a function
f(i, j, k) = Float64((1 + i) * (1 + j) * (1 + k))
tt, err = crossinterpolate2(f, [4, 4, 4]; tolerance=1e-10)

# Evaluate the tensor train
println(tt(0, 0, 0))  # 1.0
println(tt(1, 1, 1))  # 8.0
println(tt(3, 3, 3))  # 64.0

# Check properties
using Tensor4all.SimpleTT: rank, site_dims
println("Rank: ", rank(tt))
println("Site dims: ", site_dims(tt))
println("Sum: ", sum(tt))
```

## ITensorLike Interface (Advanced)

For users familiar with ITensors.jl, we provide a similar interface:

```julia
using Tensor4all.ITensorLike

# Create indices
i = Index(2, tags="Site,n=1")
j = Index(3, tags="Link")
k = Index(2, tags="Site,n=2")

# Create random tensors
t1 = Tensor([i, j], randn(2, 3))
t2 = Tensor([j, k], randn(3, 2))

# Contract tensors
result = contract(t1, t2)

# Create TensorTrain
link1 = Index(4, tags="Link,l=1")
link2 = Index(4, tags="Link,l=2")
tensors = [
    Tensor([i, link1], randn(2, 4)),
    Tensor([link1, j, link2], randn(4, 3, 4)),
    Tensor([link2, k], randn(4, 2)),
]
tt = TensorTrain(tensors)

# Orthogonalize and truncate
orthogonalize!(tt, 2)
truncate!(tt; maxdim=3, rtol=1e-10)
```

## Module Structure

| Module | Description |
|--------|-------------|
| `Tensor4all` | Core types: Index, Tensor |
| `Tensor4all.SimpleTT` | Simple tensor train operations |
| `Tensor4all.TensorCI` | Cross interpolation |
| `Tensor4all.ITensorLike` | ITensors.jl-like interface |

## Tolerance Parameters

The Julia bindings support both `rtol` (tensor4all-rs style) and `cutoff` (ITensors.jl style):

```julia
# Using rtol (relative tolerance)
truncate!(tt; rtol=1e-10)

# Using cutoff (ITensors.jl style)
truncate!(tt; cutoff=1e-20)  # equivalent to rtol=1e-10
```

Note: `cutoff = rtol²`, so `rtol = √cutoff`.
