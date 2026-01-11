# tensor4all-rs

A Rust implementation of tensor networks for **vibe coding** — rapid, AI-assisted development with fast trial-and-error cycles.

## Design Philosophy

**Vibe Coding Optimized**: tensor4all-rs is designed for rapid prototyping with AI code generation:

- **Modular architecture**: Independent crates with unified core (`tensor4all-core`) enable fast compilation and isolated testing
- **ITensors.jl-like dynamic structure**: Flexible `Index` system and dynamic-rank tensors preserve the intuitive API
- **Static error detection**: Rust's type system catches errors at compile time while maintaining runtime flexibility
- **Multi-language support via C-API**: Full functionality exposed through C-API; initial targets are Julia and Python

**Scope**: Initial focus on QTT (Quantics Tensor Train) and TCI (Tensor Cross Interpolation). The design is extensible to support Abelian and non-Abelian symmetries in the future.

## Type Correspondence

| ITensors.jl | QSpace v4 | tensor4all-rs |
|-------------|-----------|---------------|
| `Index{Int}` | — | `Index<Id, NoSymmSpace>` |
| `Index{QNBlocks}` | `QIDX` | `Index<Id, QNSpace>` (future) |
| `ITensor` | `QSpace` | `TensorDynLen<Id, Symm>` |
| `Dense` | `DATA` | `Storage::DenseF64/C64` |
| `Diag` | — | `Storage::DiagF64/C64` |
| `A * B` | — | `a.contract_einsum(&b)` |

### Truncation Tolerance

| Library | Parameter | Conversion |
|---------|-----------|------------|
| tensor4all-rs | `rtol` | — |
| ITensors.jl | `cutoff` | `rtol = √cutoff` |

## Project Structure

```
tensor4all-rs/
├── crates/
│   ├── tensor4all-tensorbackend/     # Scalar types, storage backends
│   ├── tensor4all-core/              # Core: Index, Tensor, SVD, QR
│   ├── tensor4all-simpletensortrain/ # Simple TT/MPS implementation
│   ├── tensor4all-tensorci/          # Tensor Cross Interpolation
│   ├── tensor4all-quanticstci/       # High-level Quantics TCI (QuanticsTCI.jl port)
│   ├── tensor4all-capi/              # C API for language bindings
│   ├── matrixci/                     # Matrix Cross Interpolation (internal)
│   ├── quanticsgrids/                # Quantics grid structures (internal)
│   ├── tensor4all-treetn/            # Tree Tensor Networks (WIP, excluded)
│   ├── tensor4all-itensorlike/       # ITensor-like TensorTrain API (WIP, excluded)
│   └── tensor4all-quanticstransform/# Quantics transformation operators (WIP, excluded)
├── julia/Tensor4all.jl/              # Julia bindings
├── python/tensor4all/                # Python bindings
├── tools/api-dump/                   # API documentation generator
└── docs/                             # Design documents
```

### Crate Documentation

**Active crates** (in workspace):

| Crate | Description |
|-------|-------------|
| [tensor4all-tensorbackend](crates/tensor4all-tensorbackend/) | Scalar types (f64, Complex64) and storage backends |
| [tensor4all-core](crates/tensor4all-core/) | Core types: Index, Tensor, SVD, QR, LU |
| [tensor4all-simpletensortrain](crates/tensor4all-simpletensortrain/) | Simple TT/MPS with multiple canonical forms |
| [tensor4all-tensorci](crates/tensor4all-tensorci/) | Tensor Cross Interpolation (TCI) algorithms |
| [tensor4all-quanticstci](crates/tensor4all-quanticstci/) | High-level Quantics TCI interface |
| [tensor4all-capi](crates/tensor4all-capi/) | C FFI for language bindings |
| [matrixci](crates/matrixci/) | Matrix Cross Interpolation |
| [quanticsgrids](crates/quanticsgrids/) | Quantics grid structures |

**Work-in-progress crates** (excluded from workspace, need TensorLike update):

| Crate | Description |
|-------|-------------|
| [tensor4all-treetn](crates/tensor4all-treetn/) | Tree tensor networks with arbitrary topology |
| [tensor4all-itensorlike](crates/tensor4all-itensorlike/) | ITensors.jl-like TensorTrain API |
| [tensor4all-quanticstransform](crates/tensor4all-quanticstransform/) | Quantics transformation operators |

## Usage Example (Rust)

### Simple Tensor Train (MPS)

```rust
use tensor4all_simpletensortrain::{TensorTrain, AbstractTensorTrain};

// Create a constant tensor train with local dimensions [2, 3, 4]
let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

// Evaluate at a specific multi-index
let value = tt.evaluate(&[0, 1, 2])?;

// Compute sum over all indices
let total = tt.sum();

// Compress with tolerance (rtol=1e-10, maxrank=20)
let compressed = tt.compressed(1e-10, Some(20))?;
```

## Language Bindings

### Julia

```julia
using Tensor4all
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

# ITensors.jl interop
using ITensors
it_idx = ITensors.Index(i)  # Convert to ITensors.Index
t4a_idx = Index(it_idx)     # Convert back
```

### Python

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

## Future Extensions

- GPU acceleration via PyTorch backend (tch-rs)
- Optimization for block-sparse tensors
- Automatic differentiation
- Quantum number symmetries (Abelian: U(1), Z_n)
- Non-Abelian symmetries (SU(2), SU(N))

## Acknowledgments

This implementation is inspired by **ITensors.jl** (https://github.com/ITensor/ITensors.jl). We have borrowed API design concepts for compatibility, but the implementation is independently written in Rust.

We acknowledge many fruitful discussions with **M. Fishman** and **E. M. Stoudenmire** at the Center for Computational Quantum Physics (CCQ), Flatiron Institute. H. Shinaoka visited CCQ during his sabbatical (November–December 2025), which greatly contributed to this project.

**Citation**: If you use this code in research, please cite:

> We used tensor4all-rs (https://github.com/tensor4all/tensor4all-rs), inspired by ITensors.jl.

For ITensors.jl:

> M. Fishman, S. R. White, E. M. Stoudenmire, "The ITensor Software Library for Tensor Network Calculations", arXiv:2007.14822 (2020)

## Documentation

- [Index System Design](docs/INDEX_SYSTEM.md) — Overview of the index system, QSpace compatibility, and IndexLike/TensorLike design

## References

- [ITensors.jl](https://github.com/ITensor/ITensors.jl) — M. Fishman, S. R. White, E. M. Stoudenmire, arXiv:2007.14822 (2020)
- QSpace v4 — A. Weichselbaum, Annals of Physics **327**, 2972 (2012)

## License

MIT License (see [LICENSE-MIT](LICENSE-MIT))
