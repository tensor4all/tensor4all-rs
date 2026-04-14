# Minimal Julia-Aligned C API Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `crates/tensor4all-capi` into the smallest C API that fully supports the planned `Tensor4all.jl` frontend: real `Index`, `Tensor`, and `TreeTensorNetwork` wrappers, plus direct quantics transform materialization.

**Architecture:** Hard-cut the current “full Rust feature dump” approach. Keep only general multi-language primitives at the FFI boundary, remove Rust-side convenience subsystems that Julia does not need, and replace the current `qgrid + linop + apply` operator flow with direct quantics-transform materialization to chain-shaped `t4a_treetn` operators. Julia owns grid semantics, variable-name resolution, ITensors/HDF5 glue, and operator/state remapping.

**Tech Stack:** Rust, `crates/tensor4all-capi`, `tensor4all-core`, `tensor4all-treetn`, `tensor4all-quanticstransform`, `cbindgen`, `cargo fmt`, `cargo clippy`, `cargo nextest`, local cross-repo verification with `Tensor4all.jl`.

---

## Summary

The current `tensor4all-capi` surface is too broad for the Julia frontend that
we actually want to support. It exports `Index`, `Tensor`, `TreeTN`,
`SimpleTT`, `QuanticsGrids`, `QuanticsTCI`, `TreeTCI`, transform operators,
HDF5 helpers, and a set of extra algorithm enums/helpers. That makes the FFI
boundary larger, harder to document, and more coupled to Rust-internal
conveniences than necessary.

The redesigned C API will keep only four opaque handle families:

- `t4a_index`
- `t4a_tensor`
- `t4a_treetn`
- `t4a_qtt_layout`

Everything else moves out of the public C surface. In particular:

- `SimpleTT` is removed from the C API entirely
- `QuanticsGrids` objects are removed from the C API entirely
- `QuanticsTCI` and `TreeTCI` are removed from the C API entirely
- HDF5 is removed from the C API entirely
- `t4a_linop` is removed and replaced by direct transform materialization
- string/int algorithm helper functions are removed
- index mutation APIs are removed
- chain-specialized TreeTN convenience APIs are removed

This is a breaking redesign. Backward compatibility is explicitly out of scope.

## Locked Decisions

### 1. The C API is Julia-aligned, not “all Rust features exposed”

The C API must expose only general primitives needed by `Tensor4all.jl` after
its frontend rework. If a Rust feature is not needed by the Julia wrapper and
is not clearly valuable for another language binding, it must not remain in the
public C surface.

### 2. `TreeTN` is the only public tensor-network object in C

`Tensor4all.jl` is TreeTN-general. Therefore:

- the C API keeps `t4a_treetn`
- the C API removes `t4a_simplett_*`
- chain-specific logic remains a Julia runtime predicate/story on top of TreeTN

### 3. `Index` is immutable at the FFI boundary

The C API must not expose in-place tag or prime-level mutation. Julia can build
`sim`, `prime`, `setprime`, and `noprime` from getters plus constructors.

### 4. All constructors become `StatusCode + out`

The redesign removes “return pointer or null” constructors from the public API.
Every constructor and cloning function uses:

```c
StatusCode t4a_<type>_new(..., t4a_<type> **out);
StatusCode t4a_<type>_clone(const t4a_<type> *ptr, t4a_<type> **out);
```

This keeps error handling uniform for Julia.

### 5. The quantics bridge is direct materialization, not backend handle juggling

The public C API will not expose:

- quantics grid objects
- operator handle objects (`t4a_linop`)
- operator “align to state” helpers
- operator application helpers

Instead it exposes direct materialization:

- transform parameters in
- canonical quantics layout descriptor in
- chain-shaped operator `t4a_treetn` out

### 6. The first transform wave supports canonical binary layouts only

`t4a_qtt_layout` is a small immutable descriptor for one of three canonical
binary layouts:

- `Grouped`
- `Interleaved`
- `Fused`

Arbitrary custom index tables remain Julia-side and must be reduced to one of
these canonical layouts via Julia-side permutation or deferred.

### 7. The output of transform materialization is always MPO-like `TreeTN`

Every transform materializer returns a chain-shaped `t4a_treetn` with:

- vertices `0, 1, ..., nsites - 1`
- one tensor per layout site
- exactly two site indices per tensor
- site-index order fixed as `[output_site, input_site]`

No transform materializer returns a separate operator-handle type.

## Target File Changes

### Crate wiring and public surface

- Modify: `crates/tensor4all-capi/Cargo.toml`
- Modify: `crates/tensor4all-capi/src/lib.rs`
- Modify: `crates/tensor4all-capi/src/macros.rs`
- Modify: `crates/tensor4all-capi/src/types.rs`

### Core retained modules

- Modify: `crates/tensor4all-capi/src/index.rs`
- Modify: `crates/tensor4all-capi/src/tensor.rs`
- Modify: `crates/tensor4all-capi/src/treetn.rs`

### Quantics replacement layer

- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`

### Remove from public crate

- Delete: `crates/tensor4all-capi/src/simplett.rs`
- Delete: `crates/tensor4all-capi/src/quanticsgrids.rs`
- Delete: `crates/tensor4all-capi/src/quanticstci.rs`
- Delete: `crates/tensor4all-capi/src/treetci.rs`
- Delete: `crates/tensor4all-capi/src/hdf5.rs`
- Remove their corresponding inline test modules

### Documentation and header generation

- Create: `crates/tensor4all-capi/cbindgen.toml`
- Create: `crates/tensor4all-capi/include/tensor4all_capi.h`
- Modify: `crates/tensor4all-capi/README.md`
- Modify: `README.md`
- Modify: `docs/CAPI_DESIGN.md`
- Modify: `docs/book/src/julia-bindings.md`

## Public API Contract

### Kept opaque types

```c
typedef struct t4a_index t4a_index;
typedef struct t4a_tensor t4a_tensor;
typedef struct t4a_treetn t4a_treetn;
typedef struct t4a_qtt_layout t4a_qtt_layout;
```

### Kept enums

```c
typedef enum {
    T4A_SCALAR_F64 = 0,
    T4A_SCALAR_C64 = 1,
} t4a_scalar_kind;

typedef enum {
    T4A_CANONICAL_UNITARY = 0,
    T4A_CANONICAL_LU = 1,
    T4A_CANONICAL_CI = 2,
} t4a_canonical_form;

typedef enum {
    T4A_CONTRACT_ZIPUP = 0,
    T4A_CONTRACT_FIT = 1,
    T4A_CONTRACT_NAIVE = 2,
} t4a_contract_method;

typedef enum {
    T4A_BC_PERIODIC = 0,
    T4A_BC_OPEN = 1,
} t4a_boundary_condition;

typedef enum {
    T4A_QTT_GROUPED = 0,
    T4A_QTT_INTERLEAVED = 1,
    T4A_QTT_FUSED = 2,
} t4a_qtt_layout_kind;
```

### Common lifecycle contract

All kept opaque types provide:

```c
void t4a_<type>_release(t4a_<type> *ptr);
int32_t t4a_<type>_is_assigned(const t4a_<type> *ptr);
StatusCode t4a_<type>_clone(const t4a_<type> *ptr, t4a_<type> **out);
```

Rules:

- `release(NULL)` is a no-op
- `is_assigned(NULL)` returns `0`
- `clone(NULL, ...)` returns `T4A_NULL_POINTER`

### `Index`

Keep exactly these public functions:

```c
StatusCode t4a_index_new(
    size_t dim,
    const char *tags_csv,
    int64_t plev,
    t4a_index **out
);

StatusCode t4a_index_new_with_id(
    size_t dim,
    uint64_t id,
    const char *tags_csv,
    int64_t plev,
    t4a_index **out
);

StatusCode t4a_index_dim(const t4a_index *ptr, size_t *out_dim);
StatusCode t4a_index_id(const t4a_index *ptr, uint64_t *out_id);
StatusCode t4a_index_plev(const t4a_index *ptr, int64_t *out_plev);
StatusCode t4a_index_tags(const t4a_index *ptr, uint8_t *buf, size_t buf_len, size_t *out_len);
StatusCode t4a_index_has_tag(const t4a_index *ptr, const char *tag, int32_t *out_has_tag);
```

Rules:

- `dim` must be `> 0`
- `plev` must be `>= 0`
- `tags_csv == NULL` means “no tags”
- tags remain comma-separated UTF-8 and are returned in the Rust canonical order
- `has_tag` writes `0` or `1`; it does not return a tri-state int directly

Remove from public C:

- `t4a_index_add_tag`
- `t4a_index_set_tags_csv`
- `t4a_index_set_plev`
- `t4a_index_prime`

### `Tensor`

Keep exactly these public functions:

```c
StatusCode t4a_tensor_new_dense_f64(
    size_t rank,
    const t4a_index *const *index_ptrs,
    const double *data,
    size_t data_len,
    t4a_tensor **out
);

StatusCode t4a_tensor_new_dense_c64(
    size_t rank,
    const t4a_index *const *index_ptrs,
    const double *data_interleaved,
    size_t n_complex,
    t4a_tensor **out
);

StatusCode t4a_tensor_rank(const t4a_tensor *ptr, size_t *out_rank);
StatusCode t4a_tensor_dims(const t4a_tensor *ptr, size_t *buf, size_t buf_len, size_t *out_len);
StatusCode t4a_tensor_indices(
    const t4a_tensor *ptr,
    t4a_index **buf,
    size_t buf_len,
    size_t *out_len
);
StatusCode t4a_tensor_scalar_kind(const t4a_tensor *ptr, t4a_scalar_kind *out_kind);
StatusCode t4a_tensor_copy_dense_f64(
    const t4a_tensor *ptr,
    double *buf,
    size_t buf_len,
    size_t *out_len
);
StatusCode t4a_tensor_copy_dense_c64(
    const t4a_tensor *ptr,
    double *buf_interleaved,
    size_t n_complex,
    size_t *out_len
);
StatusCode t4a_tensor_contract(
    const t4a_tensor *a,
    const t4a_tensor *b,
    t4a_tensor **out
);
```

Rules:

- constructors derive dimensions from `index_ptrs`; there is no separate `dims` argument
- all dense buffers are column-major
- `n_complex` counts complex elements, not raw doubles
- `buf_interleaved` contains `2 * n_complex` doubles `[re, im, re, im, ...]`
- `copy_dense_*` must materialize dense data even if Rust storage is compact internally

Remove from public C:

- `t4a_tensor_get_storage_kind`
- `t4a_tensor_onehot`
- any diagonal-only constructor or export path

### `TreeTN`

Keep exactly these public functions:

```c
StatusCode t4a_treetn_new(
    const t4a_tensor *const *tensors,
    size_t n_tensors,
    t4a_treetn **out
);

StatusCode t4a_treetn_num_vertices(const t4a_treetn *ptr, size_t *out_n);
StatusCode t4a_treetn_tensor(const t4a_treetn *ptr, size_t vertex, t4a_tensor **out);
StatusCode t4a_treetn_set_tensor(t4a_treetn *ptr, size_t vertex, const t4a_tensor *tensor);
StatusCode t4a_treetn_neighbors(
    const t4a_treetn *ptr,
    size_t vertex,
    size_t *buf,
    size_t buf_len,
    size_t *out_len
);
StatusCode t4a_treetn_siteinds(
    const t4a_treetn *ptr,
    size_t vertex,
    t4a_index **buf,
    size_t buf_len,
    size_t *out_len
);
StatusCode t4a_treetn_linkind(
    const t4a_treetn *ptr,
    size_t v1,
    size_t v2,
    t4a_index **out
);
StatusCode t4a_treetn_orthogonalize(
    t4a_treetn *ptr,
    size_t vertex,
    t4a_canonical_form form
);
StatusCode t4a_treetn_truncate(
    t4a_treetn *ptr,
    double rtol,
    double cutoff,
    size_t maxdim
);
StatusCode t4a_treetn_evaluate(
    const t4a_treetn *ptr,
    const t4a_index *const *indices,
    size_t n_indices,
    const size_t *values_col_major,
    size_t n_points,
    double *out_re,
    double *out_im
);
StatusCode t4a_treetn_inner(
    const t4a_treetn *a,
    const t4a_treetn *b,
    double *out_re,
    double *out_im
);
StatusCode t4a_treetn_norm(t4a_treetn *ptr, double *out_norm);
StatusCode t4a_treetn_contract(
    const t4a_treetn *a,
    const t4a_treetn *b,
    t4a_contract_method method,
    double rtol,
    double cutoff,
    size_t maxdim,
    t4a_treetn **out
);
StatusCode t4a_treetn_to_dense(const t4a_treetn *ptr, t4a_tensor **out);
```

Rules:

- `t4a_treetn_new` still assigns backend node names `0..n_tensors-1`
- `n_tensors == 0` is rejected with `T4A_INVALID_ARGUMENT`
- `evaluate` consumes explicit `t4a_index*` handles, not raw index IDs
- `values_col_major` is column-major with shape `(n_indices, n_points)`
- `out_im == NULL` is allowed only when the result is purely real

Remove from public C:

- `t4a_treetn_num_edges`
- `t4a_treetn_bond_dim`
- `t4a_treetn_linkind_at`
- `t4a_treetn_bond_dim_at`
- `t4a_treetn_bond_dims`
- `t4a_treetn_maxbonddim`
- `t4a_treetn_ortho_center`
- `t4a_treetn_canonical_form`
- `t4a_treetn_all_site_index_ids`
- `t4a_treetn_lognorm`
- `t4a_treetn_add`
- `t4a_treetn_linsolve`
- `t4a_treetn_swap_site_indices`

### `QTT` layout descriptor

Keep exactly this new public type and constructor:

```c
StatusCode t4a_qtt_layout_new(
    t4a_qtt_layout_kind kind,
    size_t nvariables,
    const size_t *variable_resolutions,
    t4a_qtt_layout **out
);
```

Rules:

- this descriptor is for canonical binary QTT layouts only
- `variable_resolutions[i]` is the bit count of variable `i`
- `Grouped` allows different resolutions
- `Interleaved` requires all resolutions equal
- `Fused` requires all resolutions equal

Canonical site order is fixed:

- `Grouped`: variable `0` block first, then variable `1`, etc.; within each block sites run MSB to LSB
- `Interleaved`: sites run level by level from MSB to LSB; within each level variable `0..nvariables-1`
- `Fused`: one site per level from MSB to LSB; each site fuses all variables at that level

The layout descriptor exposes no grid coordinates, variable names, bounds, or
index-table mutation. Julia owns those.

### Direct transform materialization

Keep exactly these public functions:

```c
StatusCode t4a_qtransform_shift_materialize(
    const t4a_qtt_layout *layout,
    size_t target_var,
    int64_t offset,
    t4a_boundary_condition bc,
    t4a_treetn **out
);

StatusCode t4a_qtransform_flip_materialize(
    const t4a_qtt_layout *layout,
    size_t target_var,
    t4a_boundary_condition bc,
    t4a_treetn **out
);

StatusCode t4a_qtransform_phase_rotation_materialize(
    const t4a_qtt_layout *layout,
    size_t target_var,
    double theta,
    t4a_treetn **out
);

StatusCode t4a_qtransform_cumsum_materialize(
    const t4a_qtt_layout *layout,
    size_t target_var,
    t4a_treetn **out
);

StatusCode t4a_qtransform_fourier_materialize(
    const t4a_qtt_layout *layout,
    size_t target_var,
    int32_t forward,
    size_t maxbonddim,
    double tolerance,
    t4a_treetn **out
);

StatusCode t4a_qtransform_binaryop_materialize(
    const t4a_qtt_layout *layout,
    size_t lhs_var,
    size_t rhs_var,
    int8_t a1,
    int8_t b1,
    int8_t a2,
    int8_t b2,
    t4a_boundary_condition bc1,
    t4a_boundary_condition bc2,
    t4a_treetn **out
);

StatusCode t4a_qtransform_affine_materialize(
    const t4a_qtt_layout *layout,
    const int64_t *a_num,
    const int64_t *a_den,
    const int64_t *b_num,
    const int64_t *b_den,
    size_t m,
    size_t n,
    const t4a_boundary_condition *bc,
    t4a_treetn **out
);
```

Rules:

- all materializers return a chain-shaped `t4a_treetn`
- the chain’s vertex order matches the canonical site order of `layout`
- each vertex tensor has exactly two site indices `[output_site, input_site]`
- all generated operator site indices are fresh backend indices
- Julia is responsible for remapping materialized operator indices onto a target state
- Julia composes multiple one-variable transforms by contracting returned MPO-like `TreeTN`s
- no `t4a_linop` survives in public C

## Task Breakdown

### Task 1: Hard-cut the crate boundary

**Files:**

- Modify: `crates/tensor4all-capi/Cargo.toml`
- Modify: `crates/tensor4all-capi/src/lib.rs`

- [ ] Remove public module declarations and `pub use` lines for:
  - `simplett`
  - `quanticsgrids`
  - `quanticstci`
  - `treetci`
  - `hdf5`
- [ ] Remove crate dependencies for:
  - `tensor4all-itensorlike`
  - `tensor4all-simplett`
  - `tensor4all-quanticstci`
  - `tensor4all-treetci`
  - `quanticsgrids`
  - `tensor4all-hdf5`
- [ ] Keep only dependencies still needed by the reduced surface.
- [ ] Replace README claims like “full functionality exposed for Julia” with “minimal Julia-facing C API”.

**Acceptance criteria:**

- `tensor4all-capi` no longer compiles or exports the removed subsystems
- `cargo check -p tensor4all-capi` succeeds with the reduced dependency set

### Task 2: Normalize error handling and lifecycle semantics

**Files:**

- Modify: `crates/tensor4all-capi/src/lib.rs`
- Modify: `crates/tensor4all-capi/src/macros.rs`

- [ ] Replace `unwrap_catch_ptr`-style pointer-return constructor handling with a single `run_catching` helper for `StatusCode + out`.
- [ ] Keep `t4a_last_error_message`, but require every error path to preserve the Rust-side message.
- [ ] Update the lifecycle macro so `clone` also becomes `StatusCode + out`.
- [ ] Remove all `Err(_) => T4A_INTERNAL_ERROR` / message-discard patterns from retained modules.

**Acceptance criteria:**

- every retained public function uses one of two patterns only:
  - `void release`
  - `StatusCode + out`
- calling any retained function on bad input yields a non-generic `last_error_message`

### Task 3: Rebuild `types.rs` around the reduced API

**Files:**

- Modify: `crates/tensor4all-capi/src/types.rs`

- [ ] Delete all opaque-type definitions not listed in the kept-surface contract.
- [ ] Replace `t4a_storage_kind` with `t4a_scalar_kind`.
- [ ] Replace `t4a_unfolding_scheme` and `t4a_linop` with `t4a_qtt_layout_kind` and `t4a_qtt_layout`.
- [ ] Keep `t4a_canonical_form`, `t4a_contract_method`, and `t4a_boundary_condition`.
- [ ] Implement `t4a_qtt_layout` as an immutable owned wrapper storing:
  - layout kind
  - number of variables
  - per-variable resolutions
  - derived site count

**Acceptance criteria:**

- `types.rs` no longer mentions `SimpleTT`, `QGrid`, `Qtci`, `TreeTCI`, `LinOp`, or HDF5 types
- the public enum set matches the contract above exactly

### Task 4: Rebuild `index.rs` as immutable constructor/getter-only FFI

**Files:**

- Modify: `crates/tensor4all-capi/src/index.rs`
- Modify: `crates/tensor4all-capi/src/index/tests/mod.rs`

- [ ] Convert constructors to `StatusCode + out`.
- [ ] Add explicit `plev` to both constructors.
- [ ] Rename getters to the final names:
  - `t4a_index_plev`
  - `t4a_index_tags`
- [ ] Convert `has_tag` to `StatusCode + out_has_tag`.
- [ ] Delete all in-place mutators.

**Acceptance criteria:**

- constructor behavior matches the final contract exactly
- tests cover:
  - `dim == 0` rejection
  - negative `plev` rejection
  - tags round-trip
  - `has_tag` writes `0/1`
  - cloned index preserves metadata

### Task 5: Rebuild `tensor.rs` around dense I/O and contraction only

**Files:**

- Modify: `crates/tensor4all-capi/src/tensor.rs`
- Modify: `crates/tensor4all-capi/src/tensor/tests/mod.rs`

- [ ] Rename getters to the final names:
  - `t4a_tensor_rank`
  - `t4a_tensor_dims`
  - `t4a_tensor_indices`
  - `t4a_tensor_scalar_kind`
  - `t4a_tensor_copy_dense_f64`
  - `t4a_tensor_copy_dense_c64`
- [ ] Remove `dims` from both constructors and derive expected dimensions from the supplied indices.
- [ ] Ensure complex constructor/getter length parameters count complex elements, not raw doubles.
- [ ] Remove `onehot` and storage-kind export.
- [ ] Keep `contract` as the only tensor-level compute primitive.

**Acceptance criteria:**

- rank/dims/indices all support query-then-fill
- dense round-trips are column-major for both `f64` and `Complex64`
- a small contraction test matches dense matrix multiplication

### Task 6: Rebuild `treetn.rs` around the Julia-facing general primitives

**Files:**

- Modify: `crates/tensor4all-capi/src/treetn.rs`
- Modify: `crates/tensor4all-capi/src/treetn/tests/mod.rs`

- [ ] Collapse `orthogonalize` and `orthogonalize_with` into a single public function taking `t4a_canonical_form`.
- [ ] Replace the current index-ID evaluation API with handle-based evaluation using `t4a_index*`.
- [ ] Remove all chain-only helpers and all high-level convenience operations listed in the removal contract.
- [ ] Keep only:
  - constructor
  - topology/tensor access
  - link/site access
  - orthogonalize
  - truncate
  - evaluate
  - inner
  - norm
  - contract
  - to_dense

**Acceptance criteria:**

- the retained TreeTN API can support the exact Julia wrapper plan without extra C helpers
- no retained symbol in `treetn.rs` assumes chain topology
- tests cover:
  - topology queries
  - get/set tensor
  - orthogonalization with form
  - truncation via `rtol` and `cutoff`
  - dense evaluation at multiple points
  - contraction and dense materialization

### Task 7: Replace `qgrid + linop + apply` with `qtt_layout + materialize`

**Files:**

- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`
- Modify: `crates/tensor4all-capi/src/types.rs`
- Modify: `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`

- [ ] Delete every public `t4a_qtransform_*` function that returns `t4a_linop`.
- [ ] Delete `t4a_linop_apply`, `t4a_linop_set_input_space`, and `t4a_linop_set_output_space`.
- [ ] Implement `t4a_qtt_layout_new`.
- [ ] Implement one direct materializer per final public transform function.
- [ ] Materialize each transform directly as a chain-shaped `t4a_treetn`.
- [ ] Use the following implementation strategy:
  - grouped/interleaved single-variable transforms are built from existing raw kernels plus site-order permutation where needed
  - fused transforms reuse existing multi-variable kernels when layout constraints permit
  - affine and binaryop use the existing general kernels, but emit TreeTN directly and permute site order if needed
  - unsupported canonical layout/kernel combinations return `T4A_INVALID_ARGUMENT` with an explicit message

**Acceptance criteria:**

- no public symbol in the crate exposes `t4a_linop`
- no public symbol in the crate exposes a quantics grid object
- every materializer returns an MPO-like chain TreeTN with `[output_site, input_site]` ordering at every vertex
- tests compare dense operators from materialized TTNs against Rust reference operators for:
  - shift
  - flip
  - phase rotation
  - cumsum
  - Fourier
  - binaryop
  - affine

### Task 8: Delete removed modules and stale tests

**Files:**

- Delete: `crates/tensor4all-capi/src/simplett.rs`
- Delete: `crates/tensor4all-capi/src/quanticsgrids.rs`
- Delete: `crates/tensor4all-capi/src/quanticstci.rs`
- Delete: `crates/tensor4all-capi/src/treetci.rs`
- Delete: `crates/tensor4all-capi/src/hdf5.rs`
- Delete or rewrite associated test modules under `crates/tensor4all-capi/src/**/tests/mod.rs`

- [ ] Remove deleted modules from `src/lib.rs`.
- [ ] Remove tests that target deleted public APIs.
- [ ] Keep only tests for the retained surface.

**Acceptance criteria:**

- `rg "simplett|qgrid|qtci|treetci|linop|hdf5" crates/tensor4all-capi/src` finds only deleted-file references in git history, not live public code

### Task 9: Add generated header and fix documentation

**Files:**

- Create: `crates/tensor4all-capi/cbindgen.toml`
- Create: `crates/tensor4all-capi/include/tensor4all_capi.h`
- Modify: `crates/tensor4all-capi/README.md`
- Modify: `README.md`
- Modify: `docs/CAPI_DESIGN.md`
- Modify: `docs/book/src/julia-bindings.md`

- [ ] Generate a checked-in header with:

```bash
cbindgen crates/tensor4all-capi \
  --config crates/tensor4all-capi/cbindgen.toml \
  --output crates/tensor4all-capi/include/tensor4all_capi.h
```

- [ ] Update docs to say:
  - the C API is minimal and Julia-oriented
  - data layout is column-major
  - complex buffers are interleaved `[re, im]`
  - the public surface is `Index`, `Tensor`, `TreeTN`, and quantics materialization
- [ ] Remove README examples that mention `SimpleTT` or in-place index mutation.

**Acceptance criteria:**

- the checked-in header matches the reduced public API exactly
- no public doc claims that the C API exposes “full functionality”
- the C API README no longer tells users to hand-declare prototypes

### Task 10: Verify against `Tensor4all.jl`

**Files:**

- Cross-repo verification only; no file path inside `tensor4all-rs`

- [ ] Build the redesigned shared library.
- [ ] Point local `Tensor4all.jl` at that library.
- [ ] Run the Julia core and TreeTN tests.
- [ ] Implement the first real `materialize_transform(..., grid)` layer in the Julia repo after the C API lands.
- [ ] Confirm that `Tensor4all.jl` no longer needs any removed C symbol.

**Verification commands:**

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release -p tensor4all-capi
cargo test --doc --release -p tensor4all-capi

cd ../Tensor4all.jl
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --startup-file=no --project=docs docs/make.jl
```

**Acceptance criteria:**

- the Rust crate passes formatting, linting, tests, and rustdoc checks
- the local Julia wrapper passes its core and TreeTN tests against the redesigned C API
- the Julia repo needs no fallback shims for removed FFI functions

## Explicit Non-Goals

- no backward-compatibility shim layer for the old C API
- no C API for `SimpleTT`
- no C API for `QuanticsGrids` coordinate conversion
- no C API for `QuanticsTCI`
- no C API for `TreeTCI`
- no C API for HDF5
- no public `t4a_linop`
- no chain-specific TreeTN helpers in C
- no arbitrary custom quantics index-table descriptor in the first redesign wave

## Final Deliverable Checklist

- [ ] `tensor4all-capi` exports only `Index`, `Tensor`, `TreeTN`, `QTTLayout`, and direct transform materializers
- [ ] all constructors and clones use `StatusCode + out`
- [ ] all retained error paths preserve `t4a_last_error_message`
- [ ] `Index` is immutable at the FFI boundary
- [ ] `Tensor` dense constructors no longer take redundant `dims`
- [ ] `TreeTN` evaluation is handle-based rather than index-ID-based
- [ ] no removed subsystem remains in the public C surface
- [ ] a checked-in generated header documents the exact API
- [ ] Rust verification passes
- [ ] `Tensor4all.jl` core/TreeTN verification passes against the new library
