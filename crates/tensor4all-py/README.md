# tensor4all-py

PyO3 bindings for the tensor4all-rs core and TreeTN crates. This is the initial
prototype tracked by
[issue #744](https://github.com/tensor4all/tensor4all-rs/issues/744).

The Python layer is deliberately thin: `Index`, `Tensor`, and
`TreeTensorNetwork` forward to the public Rust API of `tensor4all-core` and
`tensor4all-treetn`. No tensor or network algorithm lives on the Python side.
`TreeTensorNetwork` is the only network type; chains (MPS/MPO-like networks) are
path-shaped instances of it.

## Build and test

The crate is **not** a workspace member. It links CPython, which the Rust-only
CI matrix does not provide, so it builds from its own directory:

```bash
cd crates/tensor4all-py
python3 -m venv .venv
.venv/bin/python -m pip install maturin numpy pytest
.venv/bin/maturin develop          # builds the extension into .venv
.venv/bin/python -m pytest tests
.venv/bin/python examples/basic_operations.py
```

Use `maturin develop --release` for a smaller, faster extension to import. The
crate carries its own `[profile.*] debug = 0` settings because it is its own
workspace root; without them the dev-profile extension is built with full
debuginfo.

`maturin develop` enables the `extension-module` feature, so the extension does
not link `libpython` at runtime. Plain `cargo test` in this directory keeps the
default (non-`extension-module`) linking.

## Supported operations

- `Index(dim, tags=..., plev=...)`, plus `prime()`, `noprime()`, `same_id()`,
  and full identity semantics: equality is (id, prime level, tags).
- `Tensor(indices, data)` for `float64` and `complex128` arrays, `dims`,
  `indices`, `to_numpy()`, and pairwise `contract()`.
- `TreeTensorNetwork(tensors, names=None)`, `num_vertices`, `num_edges`,
  `node_names()`, `tensor(name)`, `contract_to_tensor()`, and `contract()` with
  the `"naive"` (dense reference) and `"zipup"` methods.

## Data contract

- Indices are connected automatically: an index occurring in exactly two node
  tensors becomes the bond between them; an index occurring once stays a site
  leg; any other multiplicity is rejected.
- NumPy input is copied into Rust-owned storage, and `to_numpy()` allocates a
  fresh array. Mutating either side never affects the other.
- Input arrays are read in logical index order for any layout, including
  F-contiguous, sliced, and negative-stride arrays. Output arrays are returned
  in logical index order as Fortran-ordered arrays. The logical tensor is never
  transposed.
- Only `float64` and `complex128` are accepted; other dtypes raise `TypeError`
  rather than silently dropping imaginary components.
- Intermediate tensors and network state stay in Rust across operations. NumPy
  is an explicit data interchange boundary, not an autograd integration.

## Not covered yet

- TCI, quantics, canonicalization, truncation, and solver APIs.
- A `TensorTrain` type or legacy `simplett` bindings (intentionally absent).
- Zero-copy interop, framework autograd, and wheel release infrastructure.
- CI wiring for this crate; it is exercised locally through the commands above.
