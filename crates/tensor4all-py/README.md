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
- `crossinterpolate(evaluate, local_dims, ...)` for discrete tree tensor cross
  interpolation; it returns `(network, ranks, errors)`.

## Cross interpolation (TreeTCI)

The evaluator boundary is **batched**. `evaluate` receives a C-contiguous
`int64` array of shape `(n_points, n_sites)` and must return a `(n_points,)`
array of `float64` or `complex128`:

```python
import numpy as np
import tensor4all as t4a

def evaluate(points):        # points[p] is one point (i, j, k)
    return (points[:, 0] + 10 * points[:, 1] + 100 * points[:, 2]).astype(np.float64)

network, ranks, errors = t4a.crossinterpolate(
    evaluate, [2, 3, 4], initial_pivots=[[0, 0, 1]], seed=0
)
assert errors[-1] < 1e-10
assert np.allclose(network.contract_to_tensor().to_numpy(), reference)
```

- **Layout**: the batch axis is axis 0 and the site axis is axis 1, for every
  call. This is the same memory as Rust's column-major `(n_sites, n_points)`
  batch, so nothing is transposed. The batch axis is present even for a single
  point: a one-point batch has shape `(1, n_sites)`, and the result is always
  one dimensional. Arguments such as `local_dims`, `edges`, and
  `initial_pivots` are plain Python sequences.
- **Contract**: the function must be pure. TreeTCI may request the same point
  repeatedly and gives no ordering guarantee. The array handed to `evaluate` is
  a fresh copy that may be modified.
- **Value type**: fixed by the first call (the initial-pivot batch) and must not
  change. A later mismatch raises `TypeError` instead of silently dropping the
  imaginary component. Rust panics inside TreeTCI surface as `RuntimeError`.
- **Errors**: Python exceptions raised by `evaluate` propagate unchanged. Not
  converging is not an error; inspect `errors`.
- **Options**: `tolerance`, `max_iter`, `max_bond_dim`, `seed`. `seed=None`
  (the default) seeds the global pivot search from OS entropy, so pass a seed
  for reproducible runs. The default initial pivot is the all-zero point, which
  must not evaluate to zero.
- **Result**: node `k` of the returned `TreeTensorNetwork` is site `k`, and its
  site leg is the first index of `network.tensor(k)`. The result is an ordinary
  `TreeTensorNetwork`, so `contract_to_tensor()` and `contract()` work on it.

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

- Quantics TCI and quantics transforms. Because the Rust quantics entry points
  are point-wise (`f(&[f64]) -> V`), extending the batched contract to quantics
  needs a batched entry point in `tensor4all-quanticstci` rather than an adapter
  in this crate.
- TCI options beyond `tolerance`, `max_iter`, `max_bond_dim`, and `seed` (for
  example `enable_global_pivots`, `nsearch`), plus `center_site` and proposer
  selection.
- Canonicalization, truncation, and solver APIs.
- A `TensorTrain` type or legacy `simplett` bindings (intentionally absent).
- Zero-copy interop, framework autograd, and wheel release infrastructure.
- CI wiring for this crate; it is exercised locally through the commands above.
