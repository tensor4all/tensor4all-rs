# tensor4all-rs

A Rust implementation of tensor networks for **vibe coding** — rapid, AI-assisted development with fast trial-and-error cycles.

## Design Philosophy

**Vibe Coding Optimized**: tensor4all-rs is designed for rapid prototyping with AI code generation:

- **Modular architecture**: Independent crates (`core-common`, `core-tensor`, `core-linalg`, etc.) enable fast compilation and isolated testing
- **ITensors.jl-like dynamic structure**: Flexible `Index` system and dynamic-rank tensors preserve the intuitive API
- **Static error detection**: Rust's type system catches errors at compile time while maintaining runtime flexibility

**Scope**: Initial focus on QTT (Quantics Tensor Train) and TCI (Tensor Cross Interpolation). The design is extensible to support Abelian and non-Abelian symmetries in the future.

## Type Correspondence

| ITensors.jl | QSpace v4 | tensor4all-rs |
|-------------|-----------|---------------|
| `Index{Int}` | — | `Index<Id, NoSymmSpace>` |
| `Index{QNBlocks}` | `QIDX` | `Index<Id, QNSpace>` (future) |
| `ITensor` | `QSpace` | `TensorDynLen<Id, T, Symm>` |
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
│   ├── tensor4all/           # Umbrella + core crates
│   │   ├── core-common/      # Index, tags, utilities
│   │   ├── core-tensor/      # Tensor, storage
│   │   └── core-linalg/      # SVD, QR (FAER/LAPACK)
│   ├── tensor4all-capi/      # C API for bindings
│   ├── tensor4all-tensortrain/
│   ├── tensor4all-matrixci/
│   ├── tensor4all-tensorci/
│   ├── tensor4all-treetn/
│   └── quanticsgrids/
├── julia/Tensor4all.jl/      # Julia bindings
├── python/tensor4all/        # Python bindings
└── docs/                     # Design documents
```

## Language Bindings

### Julia

```julia
using Tensor4all
using Tensor4all.ITensorLike  # TensorTrain functionality

# Index with ITensors.jl interop
i = Index(2, tags="Site")
it_idx = ITensors.Index(i)  # Convert to ITensors.Index
```

### Python

```python
from tensor4all import Index, Tensor
import numpy as np

i = Index(2, tags="Site")
t = Tensor([i, j], np.array([[1, 2], [3, 4]]))
```

## Usage Example

```rust
use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
use tensor4all_core_tensor::{Storage, TensorDynLen};

// Create indices and tensors
let i = Index::new_dyn(2);
let j = Index::new_dyn(3);
let k = Index::new_dyn(4);

// Contract: C[i,k] = A[i,j] * B[j,k]
let c = tensor_a.contract_einsum(&tensor_b);
```

## Future Extensions

- Quantum number symmetries (Abelian: U(1), Z_n)
- Non-Abelian symmetries (SU(2), SU(N))
- Arrow/direction for fermionic systems

## Acknowledgments

This implementation is inspired by **ITensors.jl** (https://github.com/ITensor/ITensors.jl). We have borrowed API design concepts for compatibility, but the implementation is independently written in Rust.

We acknowledge many fruitful discussions with **M. Fishman** and **E. M. Stoudenmire** at the Center for Computational Quantum Physics (CCQ), Flatiron Institute. H. Shinaoka visited CCQ during his sabbatical (November–December 2025), which greatly contributed to this project.

**Citation**: If you use this code in research, please cite:

> We used tensor4all-rs (https://github.com/tensor4all/tensor4all-rs), inspired by ITensors.jl.

For ITensors.jl:

> M. Fishman, S. R. White, E. M. Stoudenmire, "The ITensor Software Library for Tensor Network Calculations", arXiv:2007.14822 (2020)

## References

- [ITensors.jl](https://github.com/ITensor/ITensors.jl) — M. Fishman, S. R. White, E. M. Stoudenmire, arXiv:2007.14822 (2020)
- QSpace v4 — A. Weichselbaum, Annals of Physics **327**, 2972 (2012)

## License

MIT License (see [LICENSE-MIT](LICENSE-MIT))
