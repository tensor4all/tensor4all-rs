# tensor4all-rs

[![CI](https://github.com/tensor4all/tensor4all-rs/actions/workflows/CI_rs.yml/badge.svg)](https://github.com/tensor4all/tensor4all-rs/actions/workflows/CI_rs.yml)
[![codecov](https://codecov.io/gh/tensor4all/tensor4all-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/tensor4all/tensor4all-rs)

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
| `ITensor` | `QSpace` | `TensorDynLen` |
| `Dense` | `DATA` | `Storage::DenseF64/C64` |
| `Diag` | — | `Storage::DiagF64/C64` |
| `A * B` | — | `a.contract(&b)` |

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
│   ├── tensor4all-core/              # Core: Index, Tensor, TensorLike trait, SVD, QR
│   ├── tensor4all-treetn/            # Tree Tensor Networks with arbitrary topology
│   ├── tensor4all-itensorlike/       # ITensorMPS.jl-like TensorTrain API
│   ├── tensor4all-simplett/          # Simple TT/MPS implementation
│   ├── tensor4all-tensorci/          # Tensor Cross Interpolation
│   ├── tensor4all-quanticstci/       # High-level Quantics TCI (QuanticsTCI.jl port)
│   ├── tensor4all-quanticstransform/ # Quantics transformation operators
│   ├── tensor4all-partitionedtt/     # Partitioned Tensor Train
│   ├── tensor4all-hdf5/              # ITensors.jl-compatible HDF5 serialization
│   ├── tensor4all-capi/              # C API for language bindings
│   ├── matrixci/                     # Matrix Cross Interpolation (internal)
│   ├── quanticsgrids/                # Quantics grid structures (internal)
│   └── mdarray-einsum/               # Einstein summation for mdarray (internal)
├── julia/Tensor4all.jl/              # Julia bindings
├── python/tensor4all/                # Python bindings
├── tools/api-dump/                   # API documentation generator
├── xtask/                            # Development task runner
└── docs/                             # Design documents
```

### Crate Documentation

| Crate | Description |
|-------|-------------|
| [tensor4all-tensorbackend](crates/tensor4all-tensorbackend/) | Scalar types (f64, Complex64) and storage backends |
| [tensor4all-core](crates/tensor4all-core/) | Core types: Index, Tensor, TensorLike trait, SVD, QR, LU |
| [tensor4all-treetn](crates/tensor4all-treetn/) | Tree tensor networks with arbitrary topology |
| [tensor4all-itensorlike](crates/tensor4all-itensorlike/) | ITensorMPS.jl-like TensorTrain API |
| [tensor4all-simplett](crates/tensor4all-simplett/) | Simple TT/MPS with multiple canonical forms |
| [tensor4all-tensorci](crates/tensor4all-tensorci/) | Tensor Cross Interpolation (TCI) algorithms |
| [tensor4all-quanticstci](crates/tensor4all-quanticstci/) | High-level Quantics TCI interface |
| [tensor4all-quanticstransform](crates/tensor4all-quanticstransform/) | Quantics transformation operators |
| [tensor4all-partitionedtt](crates/tensor4all-partitionedtt/) | Partitioned Tensor Train |
| [tensor4all-hdf5](crates/tensor4all-hdf5/) | ITensors.jl-compatible HDF5 serialization |
| [tensor4all-capi](crates/tensor4all-capi/) | C FFI for language bindings |
| [matrixci](crates/matrixci/) | Matrix Cross Interpolation |
| [quanticsgrids](crates/quanticsgrids/) | Quantics grid structures |
| [mdarray-einsum](crates/mdarray-einsum/) | Einstein summation for mdarray |

## Usage Example (Rust)

### Simple Tensor Train (MPS)

```rust
use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};

// Create a constant tensor train with local dimensions [2, 3, 4]
let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

// Evaluate at a specific multi-index
let value = tt.evaluate(&[0, 1, 2])?;

// Compute sum over all indices
let total = tt.sum();

// Compress with tolerance (rtol=1e-10, maxrank=20)
let compressed = tt.compressed(1e-10, Some(20))?;
```

### Tensor Cross Interpolation (TCI)

```rust
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

// Define a function to interpolate
let f = |idx: &Vec<usize>| -> f64 {
    ((1 + idx[0]) * (1 + idx[1]) * (1 + idx[2])) as f64
};

// Perform cross interpolation
let local_dims = vec![4, 4, 4];
let initial_pivots = vec![vec![0, 0, 0]];
let options = TCI2Options { tolerance: 1e-10, ..Default::default() };

let (tci, ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f, None, local_dims, initial_pivots, options
)?;

// Convert to tensor train
let tt = tci.to_tensor_train()?;
println!("Rank: {}, Final error: {:.2e}", tci.rank(), errors.last().unwrap());
```

## Language Bindings

Binding documentation in this repo is built as an mdBook (`docs/book/`). Code blocks in the book are included from standalone scripts under `docs/examples/` and are executed in CI.

### Julia

Julia bindings are maintained in a separate repository: **[Tensor4all.jl](https://github.com/tensor4all/Tensor4all.jl)**

```julia
using Pkg
Pkg.add(url="https://github.com/tensor4all/Tensor4all.jl")
```

See the [Tensor4all.jl README](https://github.com/tensor4all/Tensor4all.jl) for detailed installation and usage instructions.

#### Executable documentation examples

All Julia documentation examples live in `docs/examples/julia/` and can be run with:

```bash
for f in docs/examples/julia/*.jl; do
  julia --project=julia/Tensor4all.jl "$f"
done
```

CI runs these examples as part of `./scripts/run_julia_tests.sh`.

### Python

**Note:** Python bindings require a Rust toolchain. Install from <https://rustup.rs/>.

#### Install from GitHub

```bash
uv pip install "tensor4all @ git+https://github.com/tensor4all/tensor4all-rs#subdirectory=python/tensor4all"
```

#### Development install

```bash
cd python/tensor4all
python scripts/build_capi.py   # Build the Rust C library
uv pip install -e .
```

After Rust code changes, re-run `python scripts/build_capi.py`.

#### Quick example

```python
from tensor4all import crossinterpolate2

def f(i, j, k):
    return float((1 + i) * (1 + j) * (1 + k))

tt, err = crossinterpolate2(f, [4, 4, 4], tolerance=1e-10)
print(tt(0, 0, 0))  # 1.0
```

#### Available modules

| Module | Description |
|--------|-------------|
| `tensor4all.treetn` | Tree tensor networks (MPS, MPO, TTN) |
| `tensor4all.tensorci` | Tensor cross interpolation |
| `tensor4all.quanticsgrids` | Quantics grid representations |
| `tensor4all.quanticstci` | Quantics TCI for function interpolation |
| `tensor4all.quanticstransform` | Quantics operators (shift, flip, Fourier) |
| `tensor4all.simplett` | Simple tensor trains |

#### Executable documentation examples

All Python documentation examples live in `docs/examples/python/` and are executed from the repo root:

```bash
cd python/tensor4all
python scripts/build_capi.py
uv pip install -e .
cd ../..
for f in docs/examples/python/*.py; do
  python "$f"
done
```

CI runs these examples as part of `./scripts/run_python_tests.sh`.

## Future Extensions

- GPU acceleration via PyTorch backend (tch-rs)
- In-place operations for memory efficiency
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

## TODO / Known Issues

### Naming Convention

- **Tolerance parameters**: Standardize on `rtol` (relative tolerance) and `atol` (absolute tolerance)
  - Current inconsistency: `cutoff`, `tolerance`, `rtol` used interchangeably
  - `cutoff` (ITensors.jl style) should only appear in compatibility layers
  - Conversion: `rtol = √cutoff`

### Incomplete Implementations

- **MPO canonical forms**: VidalMPO and InverseMPO conversions not yet implemented
- **C API `t4a_treetn_evaluate`**: TreeTN evaluate function not yet exposed in C API

### Known Backend Issues

- Complex SVD: mdarray-linalg-faer returns V^T instead of V^H for complex matrices

## Development

### Development Tasks (xtask)

This project uses `cargo xtask` for common development tasks:

```bash
# Generate documentation with custom index page
cargo xtask doc

# Generate and open documentation in browser
cargo xtask doc --open

# Run all CI checks (fmt, clippy, test, doc)
cargo xtask ci
```

### Pre-commit Checks

Before committing changes, ensure that both formatting and linting pass:

```bash
# Check code formatting
cargo fmt --all -- --check

# Run clippy with all warnings as errors
cargo clippy --workspace --all-targets -- -D warnings
```

If either command fails, fix the issues before committing:

```bash
# Auto-fix formatting
cargo fmt --all

# Fix clippy warnings (some may require manual fixes)
cargo clippy --workspace --all-targets -- -D warnings
```

These checks are also enforced in CI, so ensuring they pass locally will prevent CI failures.

## Documentation

- [Index System Design](docs/INDEX_SYSTEM.md) — Overview of the index system, QSpace compatibility, and IndexLike/TensorLike design
- [Vibe Coding Workflow](docs/vibe_coding_workflow.md) — Development workflow guidelines for rapid, AI-assisted development
- mdBook user guide: `mdbook build docs/book` (output: `docs/book/book/`)

## References

- [ITensors.jl](https://github.com/ITensor/ITensors.jl) — M. Fishman, S. R. White, E. M. Stoudenmire, arXiv:2007.14822 (2020)
- QSpace v4 — A. Weichselbaum, Annals of Physics **327**, 2972 (2012)

## License

MIT License (see [LICENSE-MIT](LICENSE-MIT))
