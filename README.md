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
│   ├── tensor4all-core/              # Core: Index, Tensor, Storage, SVD, QR
│   ├── tensor4all-treetn/            # Tree Tensor Networks
│   ├── tensor4all-itensorlike/       # ITensor-like TensorTrain API
│   ├── tensor4all-simpletensortrain/ # Simple TT/MPS implementation
│   ├── tensor4all-tensorci/          # Tensor Cross Interpolation
│   ├── tensor4all-capi/              # C API for language bindings
│   ├── matrixci/                     # Matrix Cross Interpolation (internal)
│   └── quanticsgrids/                # Quantics grid structures (internal)
├── julia/Tensor4all.jl/              # Julia bindings
├── python/tensor4all/                # Python bindings
└── docs/                             # Design documents
```

## Usage Example (Rust)

### Tree Tensor Network

```rust
use tensor4all_treetn::{
    TreeTN, random_treetn_f64,
    CanonicalizationOptions, TruncationOptions,
};

// Create a random tree tensor network (4 sites, bond dim 10)
let site_dims = vec![2, 2, 2, 2];
let bond_dim = 10;
let ttn = random_treetn_f64(&site_dims, bond_dim);

// Canonicalize towards center node
let ttn = ttn.canonicalize(
    ["node_0"],
    CanonicalizationOptions::default()
)?;

// Truncate bond dimensions
let ttn = ttn.truncate(
    ["node_0"],
    TruncationOptions::default()
        .with_max_rank(5)
        .with_rtol(1e-10)
)?;

// Contract two tree tensor networks
let result = contract_zipup(&ttn1, &ttn2, options)?;
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
