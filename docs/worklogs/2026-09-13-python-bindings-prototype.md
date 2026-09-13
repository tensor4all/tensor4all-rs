# Python bindings prototype (issue #744)

## Summary

Added `crates/tensor4all-py`, a PyO3 extension crate exposing `Index`,
`Tensor`, and `TreeTensorNetwork`, plus tests, a runnable example, a `.pyi`
stub, and a README. Updated the binding-boundary documentation so the C API is
described as the C/C++/Julia boundary rather than the only boundary.

## Context read

- `gh issue view 744` (requirements, data contract, acceptance criteria).
- `crates/tensor4all-capi/src/{index,tensor,treetn}.rs` for the existing
  binding patterns and the concrete Rust entry points they call
  (`IdxTensor::from_dense`, `contract`/`contract_pair`, `TreeTN::from_tensors`,
  `contraction::contract`).
- `crates/tensor4all-core/src/defaults/index.rs`: `Index` equality is
  (id, plev, tags), `same_id` compares ids only, `plev`/`tags` are public fields.
- `crates/tensor4all-treetn/src/treetn/contraction.rs`: `contract` dispatches on
  method; `Naive` requires an explicit dense-reference limit; `Zipup` requires
  `same_topology` (graph structure only, so MPS-like x MPO-like is allowed).
- `AGENTS.md`, `REPOSITORY_RULES.md`, `scripts/check-{crate-boundaries,public-error-docs}.py`,
  `tools/library-panic-audit`, `xtask`, `.github/workflows/CI_rs.yml`.

## Decisions

- **Direct PyO3 boundary, not the C API.** The issue asks Python to call the
  public Rust APIs; the crate depends on `tensor4all-core` and
  `tensor4all-treetn` and adds no C API surface.
- **Not a workspace member** (`workspace.exclude`). A pyo3 crate that links
  CPython would make `cargo nextest run --workspace` and `cargo test --doc
  --workspace` depend on `libpython`/dev headers on the CI runner, which the
  current Rust-only matrix does not provide. Excluding it keeps CI unchanged;
  the tradeoff is that fmt/clippy/coverage do not cover it yet, so CI wiring for
  this crate is deferred. `crates/tensor4all-py/target/` is gitignored, and the
  crate README documents the local `maturin develop` workflow.
- **`TreeTensorNetwork` = `DefaultTreeTN<usize>`.** Node names are plain
  integers so no string naming layer is needed; chains are path-shaped
  instances, so no `TensorTrain` class exists.
- **NumPy boundary is copy-in/copy-out.** Input is read through a strided
  ndarray view and copied into column-major Rust storage; `to_numpy()` allocates
  a Fortran-ordered NumPy array and copies into it. This honours C-order,
  F-order, non-contiguous, and negative-stride inputs without transposing the
  logical tensor, and guarantees storage independence in both directions.
- **`Tensor.contract` uses the connected-network entry point**
  (`tensor4all_core::contract`) rather than `contract_pair`, so a missing shared
  index is a `ValueError` instead of a silent outer product.
- **Only `float64`/`complex128`.** Other dtypes raise `TypeError`; no imaginary
  component is dropped silently.
- **`TreeTensorNetwork.contract` exposes `"naive"` and `"zipup"`** with an
  explicit `dense_reference_limit` for the dense reference path. Fit/SRC and the
  rest of the algorithm surface are out of scope for the prototype.

## Alternatives rejected

- Making the crate a workspace member: rejected for the CI reason above.
- Wrapping `itensorlike::TensorTrain` or legacy `simplett`: excluded by the
  issue.
- Reimplementing a chain type or per-node bookkeeping in Python: the issue
  requires Python to hold no algorithms.
- A Python helper package with `__init__.py` re-exports: unneeded, since the
  extension is the whole public module; type information is provided by a
  module-level `.pyi` stub shipped in the wheel.

## Verification

- `cd crates/tensor4all-py && cargo clippy --all-targets` clean, `cargo fmt
  --check` clean.
- `maturin develop` builds and installs into a local venv.
- `pytest tests` 26 passed, covering Index identity (same id with different
  plev/tags), C/F/non-contiguous/negative-stride input layouts, complex values,
  real x complex promotion, rank-0 tensors, input and output storage
  independence, unsupported dtypes and non-arrays, shape mismatch, duplicate
  indices, disconnected contraction, chain and Y-shaped tree constructions,
  `contract_to_tensor` against NumPy/einsum references, and naive/zipup
  agreement.
- `python examples/basic_operations.py` passes all assertions.
- `python3 scripts/check-crate-boundaries.py` and
  `python3 scripts/repository-rules-review.py --base main --worktree --dry-run`
  pass (the latter with the new crate staged as intent-to-add so it is in the
  diff). `check-public-error-docs.py` reports no findings for this crate.
- The crate owns its profile settings because it is its own workspace root;
  without `[profile.dev] debug = 0` the dev extension carried full debuginfo
  (1.9 GB). After the fix, `cargo clean` removed 6.8 GB of stale artifacts and a
  fresh dev build leaves a 485 MB extension and a 2.2 GB `target/`.

## Remaining risks

- No CI job builds the Python crate; the `extension-module` feature and the
  ABI/wheel matrix are only exercised locally. A release-profile build was not
  run (disk headroom on the development machine), so `maturin develop
  --release` is documented but unverified here.
- The excluded crate builds into `crates/tensor4all-py/target` instead of the
  root `target`.
- `to_numpy()` always materializes; structured/stored representations are not
  exposed.
