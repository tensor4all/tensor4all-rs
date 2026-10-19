# Architecture & Crate Guide

This page describes how tensor4all-rs is organised, what each crate does, how to
choose the right crate for your use case, and how the workspace is converging
on a single TreeTN-based interface.

## Two stacks today, one interface intended

The workspace currently contains **two stacks**, each owning its tensor-train
representation and its algorithm crates. The intended direction is a single
interface: **`TreeTN` is the unified tensor-network type**, and the positional
`SimpleTensorTrain` stack is being retired. That migration is not finished:
TCI, quantics TCI and quantics transforms still produce or consume
`SimpleTensorTrain` today. There is **no facade
crate** that wraps everything: `tensor4all-capi` is a C FFI layer, not a Rust
facade, and each crate can be used on its own.

```text
                        tensor4all-core (Index, Tensor, contraction, SVD/QR)
                                      |
              +-----------------------+------------------------+
              |                                                |
   NETWORK STACK (TreeTN, intended interface)     SIMPLETT STACK (positional, being retired)
              |                                                |
   tensor4all-treetn  (TreeTN)                    tensor4all-simplett (SimpleTensorTrain)
     |-- tensor4all-itensorlike (TensorTrain)                  |
     |-- tensor4all-treetci (tree TCI)            tensor4all-tensorci (TCI1/TCI2)
     |-- tensor4all-treeaci (tree ACI)            tensor4all-aci (chain ACI)
     |-- tensor4all-partitionedtreetn (TreeTN patches)
     `-- tensor4all-treersi (experimental RSI products)

   Crates that still use both stacks (crossing through treetn::simplett_bridge):
   tensor4all-quanticstci        (treetci + treetn + simplett)
   tensor4all-quanticstransform  (treetn + simplett)

   tensor4all-capi  (C FFI for language bindings; depends on both stacks)
   tensor4all-hdf5  (MPS serialization, ITensors.jl-compatible)
```

### The migration bridge: `treetn::simplett_bridge`

While both stacks exist, the only sanctioned place where they convert into each
other is `treetn::simplett_bridge`, a migration aid rather than a permanent
layer:

- `tensor_train_to_treetn` / `tensor_train_to_treetn_with_names` /
  `tensor_train_to_treetn_with_names_and_site_indices`: simplett
  `SimpleTensorTrain` -> network `TreeTN`;
- `treetn_to_tensor_train`: network `TreeTN` (linear chain) -> simplett
  `SimpleTensorTrain`;
- `insert_onehot_site_in_treetn_chain`,
  `weighted_remove_site_from_treetn_chain`: chain-topology helpers.

General-tree, name-addressed site fixing lives on `treetn::fix_and_remove_site`
and is not part of the simplett bridge.

Crates that need to cross stacks must route through this bridge. New ad hoc
bridges (hand-rolled conversion logic inside another crate) are rejected.
The TreeTN-native `tensor4all-partitionedtreetn`
operates directly on named `TreeTN<IdxTensor, V>` values and does not depend on
simplett, itensorlike, or TCI crates. Its adaptive interpolation driver sees
interpolation engines only through the `TreeInterpolator` trait in
`tensor4all-treetn`; engines such as `tensor4all-treetci` implement that trait.

One deliberate exception exists: `tensor4all-quanticstransform` builds
TreeTN-based `LinearOperator`s from simplett **MPO** data (two site indices
per node) via `tensortrain_to_linear_operator`. That is operator construction,
not a general stack conversion: the bridge stays MPS-shaped (one site index
per node), and the transform layer keeps the MPO-specific index bookkeeping.
No *new* ad hoc conversions may be added; MPO conversion belongs in the
transform layer, and MPS conversion belongs in the bridge.

### Which stack does a new feature crate target?

- **New features target `TreeTN`**, chains included: operators, evolution,
  DMRG/TDVP, cross interpolation and quantics kernels build on
  `tensor4all-treetn` (or `tensor4all-itensorlike` when ITensors.jl-style
  semantics are wanted). Shared numerical substrate (factorizations, matrix
  CI, `CachedFunction`) lives in `tensor4all-core`.
- **The simplett stack is maintained, not extended**: existing
  `tensor4all-simplett` / `tensor4all-tensorci` functionality stays available
  until its TreeTN replacement lands, but new capabilities are not added there.
- While a feature still needs both stacks, it crosses through
  `treetn::simplett_bridge`; it is not duplicated in both.

## Vocabulary conventions

Beyond the naming-policy suffixes (`_mut`, `_into`, `_batched`), the following
operation names are unified across crates:

- **Inner product**: `inner_product` everywhere. (`dot` was retired in
  `tensor4all-simplett`.)
- **Densification**: `full_tensor` materializes a tensor train/MPO into a flat
  column-major buffer plus shape; `to_dense` materializes a single tensor into
  a dense `IdxTensor`. The distinction is intentional (whole-TT vs
  one-tensor).
- **Canonicalization**: `treetn` uses `canonicalize` / `canonicalize_mut`;
  `itensorlike` uses `orthogonalize` (ITensors.jl-compatible vocabulary) for
  the same center-site normalization. Each stack keeps its standard term; do
  not introduce a third synonym.
- **Options**: bond caps use `max_bond_dim: Option<usize>`; truncation
  tolerances use `SvdTruncationPolicy` (`rtol`/`cutoff`); sweep counts use
  `nfullsweeps`/`nsweeps` (TreeTN) or `nhalfsweeps` (itensorlike) as documented
  in the skills guide.
  This is the target vocabulary; some older entry points still use other
  option names, and unifying them is tracked in #915.

## `TensorTrain` naming resolution

Two different types are named `TensorTrain`-family, which is the historical
naming trap that PR 3 resolves:

| Type | Crate | Representation | Use when |
|------|-------|----------------|----------|
| `SimpleTensorTrain<T>` | `tensor4all-simplett` | positional cores (`Tensor3<T>`), no named indices | lightweight create/evaluate/compress |
| `TensorTrain` | `tensor4all-itensorlike` | `TreeTN<IdxTensor, usize>` wrapper with orthogonality tracking | ITensors.jl-style interface |

Rules:

- `tensor4all-simplett::SimpleTensorTrain` is the **positional** chain type.
  Prefer it for numerical kernels.
- `tensor4all-itensorlike::TensorTrain` is the **tree-based** chain type with
  named indices, canonical forms, and orthogonality tracking.
- There are no compatibility aliases: `TensorTrain` alone always means the
  itensorlike type inside that crate, and simplett code must write
  `SimpleTensorTrain`.

The corresponding errors follow the representation names:

| Error | Owner | Covers |
|-------|-------|--------|
| `tensor4all_simplett::SimpleTensorTrainError` | `SimpleTensorTrain` | positional shapes, flat indices, and MatrixCI failures |
| `tensor4all_itensorlike::TensorTrainError` | itensorlike `TensorTrain` | named indices, TreeTN structure, orthogonality, and factorization |

They stay separate while both stacks exist; the distinct names prevent
ambiguous imports.

## Scalar capability boundaries

Scalar traits are layered capabilities rather than interchangeable numeric
aliases:

| Capability | Purpose |
|------------|---------|
| `tensor4all_core::Scalar` | common value arithmetic and conversion |
| `tensor4all_core::MatrixLuciScalar` | core `Scalar` plus MatrixLUCI backend dispatch |
| `tensor4all_tensorbackend::TensorElement` | supported Rust scalar ↔ native tensor conversion |
| `tensor4all_tensorbackend::StorageScalar` | compact storage construction |
| tensorbackend matrix/linalg traits | individual backend kernel capabilities |
| `tensor4all_simplett::TTScalar` | positional tensor-train arithmetic requirements |
| ACI/TreeACI scalar traits | sealed algorithm-supported types and algorithm-only operations |

`tensor4all_core::AnyScalar` is the only public `AnyScalar`; it may retain an
AD-tracked rank-0 tensor. The internal tensorbackend value is named `BackendScalar`
and is an untracked compact scalar. Do not replace these capability boundaries
with one all-purpose supertrait: that would force storage and algorithm
requirements onto unrelated generic code.

## Layer descriptions

### Foundation

| Crate | Description |
|-------|-------------|
| **tensorbackend** | Public low-level infrastructure: the dense `Matrix` container, the `CpuExecutionContext` execution context (built from a caller-configured `CpuBackend`, re-exported here), compact `f64`/`Complex64`/`f32`/`Complex32` storage, and tenferro-backed primitives. Most users reach it through the higher-level crates; depend on it directly when you need these building blocks. Not every `pub` item is a stability promise. |
| **core** | Foundation for everything else. Owns the shared abstractions (`Index` system, dynamic-rank `Tensor`, contraction, SVD/QR/LU factorizations) and the matrix-CI substrate (rrLU/LUCI, `CachedFunction`, `MultiIndex`). |

### Network stack

| Crate | Description |
|-------|-------------|
| **treetn** | Tree tensor networks with arbitrary graph topology. Supports canonicalization, truncation, contraction, DMRG/TDVP, and hosts the sanctioned `simplett_bridge`. |
| **itensorlike** | ITensors.jl-inspired `TensorTrain` (tree-based) with orthogonality tracking and multiple canonical forms. |
| **partitionedtreetn** | TreeTN-native eagerly masked subdomains, strict partition algebra, volume-budgeted adaptive patching, sequential adaptive patched interpolation through any `TreeInterpolator` engine, reconstruction against a fixed global L2 tolerance, and level-coupled merge-refine QFT scheduling. |
| **treersi** | Experimental CPU Hadamard products on TreeTN via recursive sketched interpolation; requires independent accuracy checks. |
| **treeaci** | Experimental Alternating Cross Interpolation directly on `TreeTN` with any validated tree topology (no path-order requirement, unlike `tensor4all-aci`). |
| **treetci** | Tree TCI: cross interpolation on tree-structured tensor networks; `TreeTciInterpolator` implements the `TreeInterpolator` engine trait. |

### Simplett stack

| Crate | Description |
|-------|-------------|
| **simplett** | Lightweight positional tensor train (`SimpleTensorTrain`) for numerical computation. |
| **tensorci** | Tensor Cross Interpolation. Contains TCI2 (primary algorithm) and TCI1 (legacy). |
| **quanticstci** | High-level Quantics TCI. Interpolates functions on discrete or continuous grids in the quantics format. |

### Quantics & transforms

| Crate | Description |
|-------|-------------|
| **quanticstransform** | Quantics transformation operators: shift, flip, Fourier, affine, and more. Consumes simplett-stack data; constructs TreeTN-based `LinearOperator`s (see the bridge exception above). |
| **interpolativeqtt** | Interpolative QTT construction on a coarse grid (simplett stack). See the [interpolative-QTT tutorial](tutorials/quantics-basics/interpolative-qtt.md). |

### Applications

| Crate | Description |
|-------|-------------|
| **aci** | Alternating Cross Interpolation (ACI) for elementwise tensor-train operations. |

### I/O & bindings

| Crate | Description |
|-------|-------------|
| **hdf5** | HDF5 serialization compatible with ITensors.jl/ITensorMPS.jl file formats. |
| **capi** | C FFI for language bindings (Julia, C, C++). Out of scope for this guide; see [Julia Bindings](julia-bindings.md). |
| **tensor4all-py** | PyO3 bindings for Python (prototype). Not a workspace member; see its `README.md`. |

## Which crate should I use?

| Goal | Recommended crate |
|------|-------------------|
| TCI on a black-box function (high level) | `tensor4all-quanticstci` |
| TCI with fine-grained control | `tensor4all-tensorci` |
| Sketched Hadamard products of existing TreeTNs | [`tensor4all-treersi`](guides/tree-rsi.md) (experimental) |
| Tree TCI | `tensor4all-treetci` |
| Simple positional tensor train (create, evaluate, compress) | `tensor4all-simplett` (`SimpleTensorTrain`) |
| Tensor train with ITensors.jl-style interface | `tensor4all-itensorlike` (`TensorTrain`) |
| Tree tensor networks | `tensor4all-treetn` |
| Subdomain decomposition on named TreeTNs | `tensor4all-partitionedtreetn` |
| Adaptive patched interpolation of a function on a named tree | `tensor4all-partitionedtreetn::adaptive_interpolation` with an engine such as `tensor4all-treetci::TreeTciInterpolator` |
| Reconstruct orthogonal patches or tensor products with a global L2 tolerance | `tensor4all-partitionedtreetn::reconstruction` |
| Quantics transform operators | `tensor4all-quanticstransform` |
| HDF5 I/O compatible with Julia | `tensor4all-hdf5` |
| Interpolative QTT on a coarse grid | `tensor4all-interpolativeqtt` |
| Elementwise TT ops via ACI | `tensor4all-aci` (chains) or `tensor4all-treeaci` (any TreeTN) |

## Error remedies

Public error types use `thiserror`, preserve the source error, carry
structured fields, and name an actionable remedy in the rustdoc when one is
documented. Recurring cases avoid opaque string payloads. `cargo clippy` is
configured with `-D clippy::missing_errors_doc -D clippy::missing_panics_doc`
so every fallible or panicking public function documents its failure modes.

## Low-level crates

`tensor4all-tensorbackend` is public low-level infrastructure; its `Matrix`
container and execution contexts are meant to be used across crates, but
lower-level items may change while the API settles. The former
`tensor4all-tcicore` crate was dissolved into `tensor4all-core` (#639): the
matrix CI / LUCI / rrLU algorithms, `CachedFunction`, and `MultiIndex` now
live in core. These interfaces may still change without notice.
