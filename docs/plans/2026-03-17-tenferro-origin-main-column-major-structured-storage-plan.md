# Tenferro origin/main Column-Major Structured Storage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebase tensor4all-rs onto the current tenferro `Tensor` frontend, replace the legacy Dense/Diag snapshot model with `axis_classes`-aware structured storage, and complete the column-major semantic flip across internal materialization boundaries and public flat-buffer APIs.

**Architecture:** Keep `TensorDynLen` as an index-semantic wrapper over live `tenferro::Tensor` values so AD metadata stays intact. Replace `Storage::Dense* / Diag*` with a tensor4all-owned `StructuredStorage<T>` snapshot model carrying `data`, `payload_dims`, `strides`, and canonical `axis_classes`, then update bridge/boundary code so column-major semantics and `axis_classes` survive roundtrips.

**Tech Stack:** Rust, tensor4all-core, tensor4all-tensorbackend, tenferro `Tensor`, tenferro `snapshot::DynTensor`, cargo fmt, cargo clippy, cargo nextest (`--release`), Python/NumPy, C API, HDF5

---

### Task 1: Pin tensor4all-rs to the current tenferro frontend and freeze breakage

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/tensor4all-tensorbackend/Cargo.toml`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Create: `docs/plans/2026-03-17-tenferro-origin-main-column-major-audit.md`

**Step 1: Update workspace dependencies**

Replace the `tenferro-dyadtensor`-era dependency set with the current `tenferro` frontend revision:

- add direct workspace dependency on `tenferro`
- remove `tenferro-dyadtensor`
- keep `tenferro-tensor`, `tenferro-prims`, `tenferro-linalg`, `tenferro-einsum`, `tenferro-algebra`, `tenferro-device` aligned to the same git revision

**Step 2: Update crate-level docs and re-exports**

Change tensorbackend crate comments so they describe:

- live frontend payloads via `tenferro::Tensor`
- snapshot/materialization via tensor4all-owned storage

Delete public re-exports of:

- `DynAdTensor`
- `DynAdScalar`
- `tenferro_dyadtensor`

**Step 3: Audit all compile breakage**

Run:

```bash
cargo check -p tensor4all-tensorbackend
cargo check -p tensor4all-core
```

Record the failing call sites in `docs/plans/2026-03-17-tenferro-origin-main-column-major-audit.md`, grouped by:

- backend live tensor API
- snapshot/materialization API
- tests relying on `DynAdTensor`
- boundary code assuming Dense/Diag only

**Step 4: Commit**

```bash
git add Cargo.toml crates/tensor4all-tensorbackend/Cargo.toml crates/tensor4all-tensorbackend/src/lib.rs docs/plans/2026-03-17-tenferro-origin-main-column-major-audit.md
git commit -m "build: update tensor4all to current tenferro frontend"
```

### Task 2: Rebuild `TensorDynLen` around `tenferro::Tensor`

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Modify: `crates/tensor4all-core/tests/tensor_native_ad.rs`
- Modify: `crates/tensor4all-itensorlike/tests/tensortrain_native_ad.rs`

**Step 1: Write/adjust failing tests**

Update native AD tests so they target the new frontend:

- use `tenferro::Tensor`
- assert forward/reverse/tangent preservation through public tensor4all APIs
- stop depending on backend-only enum names

**Step 2: Switch the canonical payload**

In `TensorDynLen`, replace:

- `native: DynAdTensor`

with:

- `native: tenferro::Tensor`

Keep:

- `from_native`
- `as_native`
- `into_native`

but retarget them to `tenferro::Tensor`, and make them `pub(crate)` if public escape hatches are no longer justified.

**Step 3: Rework scalar and tensor execution helpers**

Update:

- `sum`
- `only`
- `inner_product`
- `permute`
- `contract`
- `outer_product`
- `conj`

to use `tenferro::Tensor`-based backend helpers instead of `DynAdTensor`.

**Step 4: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
cargo nextest run --release -p tensor4all-itensorlike --test tensortrain_native_ad
```

Expected: PASS after the new frontend wiring is in place.

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/src/lib.rs crates/tensor4all-core/tests/tensor_native_ad.rs crates/tensor4all-itensorlike/tests/tensortrain_native_ad.rs
git commit -m "refactor(core): make tenferro tensor the live payload"
```

### Task 3: Introduce `StructuredStorage` and stop dropping `axis_classes`

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Create or update tests in: `crates/tensor4all-tensorbackend/src/storage.rs`

**Step 1: Write failing backend tests first**

Add focused tests in `tensor4all-tensorbackend` that prove:

- a structured snapshot such as `axis_classes = [0, 1, 1]` roundtrips through
  `native_tensor_primal_to_storage` without densification
- `storage_to_native_tensor` reconstructs the same logical dims and
  `axis_classes`
- dense and diagonal snapshots still roundtrip exactly

Run only the new tests first and confirm they fail because `Storage` still drops
`axis_classes`.

**Step 2: Introduce `StructuredStorage<T>`**

Define a tensor4all-owned snapshot type with:

- `data: Vec<T>`
- `payload_dims: Vec<usize>`
- `strides: Vec<isize>`
- `axis_classes: Vec<usize>`

with invariants:

- canonical `axis_classes`
- `payload_rank == max(axis_classes) + 1` for non-empty logical rank
- logical dims derived from `payload_dims[axis_classes[i]]`

**Step 3: Extend the legacy storage enum incrementally**

Add:

- `Storage::StructuredF64(StructuredStorage<f64>)`
- `Storage::StructuredC64(StructuredStorage<Complex64>)`

Keep the existing Dense/Diag variants temporarily as convenience/sugar so the
rest of the workspace keeps compiling while the bridge is migrated. Dense/Diag
constructors should map to structured invariants conceptually, even if the enum
still carries compatibility variants during this task.

**Step 4: Update scalar snapshots**

Rework `AnyScalar` so its materialized representation comes from rank-0 `StructuredStorage`, not typed Dense/Diag enums.

**Step 5: Update `TensorDynLen::to_storage()` / `from_storage()`**

These functions should now roundtrip:

- `tenferro::Tensor` <-> `snapshot::DynTensor` <-> `StructuredStorage`

without dropping `axis_classes` for non-dense, non-diagonal structured tensors.

**Step 6: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-tensorbackend --lib
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
```

Expected: PASS with `axis_classes` preserved across materialization.

**Step 7: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-tensorbackend/src/any_scalar.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs crates/tensor4all-tensorbackend/src/lib.rs crates/tensor4all-core/src/defaults/tensordynlen.rs
git commit -m "feat(tensorbackend): preserve axis-class structured snapshots"
```

### Task 4: Finish the column-major semantic flip on structured snapshots

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/layout.rs`
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Modify: `crates/tensor4all-core/tests/tensor_basic.rs`
- Modify: `crates/tensor4all-core/tests/tensor_permute.rs`
- Modify: `crates/tensor4all-core/tests/linalg_qr.rs`
- Modify: `crates/tensor4all-core/tests/linalg_svd.rs`

**Step 1: Make column-major the only flat-buffer semantics**

Update:

- `from_dense_*`
- `to_vec_*`
- storage import/export helpers
- reshape/flatten paths in QR/SVD

so logical linearization is column-major everywhere.

**Execution note (current slice):**

Start with the public high-level path first:

- `TensorDynLen::from_dense`
- `TensorDynLen::from_diag`
- `TensorDynLen::to_vec_*`
- `TensorDynLen::onehot`
- bridge helpers and tests that explicitly mention row-major dense flattening

Write failing tests that distinguish 2D row-major and column-major linearization
before touching implementation. Do not start with QR/SVD reshape helpers; those
can migrate after the public constructors/materializers are stable.

**Execution note (current slice, structured-first convenience path):**

Before deleting legacy `DenseStorage` kernels, move public convenience code to
`StructuredStorage`:

- add `Storage` constructors that interpret flat dense data as column-major
  structured snapshots
- add `Storage` materializers that expose logical dense values in column-major
  order without forcing callers to match `Storage::Dense*`
- route `DenseStorageFactory` / `StorageScalar::dense_storage_with_shape` through
  those structured constructors
- update tests and docs to assert logical values and `is_dense()` /
  `is_diag()` instead of matching `Storage::Dense*` whenever the exact internal
  enum variant is not the point of the test

This keeps `DenseStorage` as a temporary low-level kernel implementation detail
while shifting the public and test-facing surface toward structured snapshots.

**Step 2: Keep physical storage concerns private**

If temporary row-major conversions are still needed for tenferro-tensor or scratch buffers, keep them as private helpers with explicit names such as:

- `from_linearized_row_major_temp`
- `to_linearized_row_major_temp`

No public API should remain ambiguous.

**Step 3: Update tests**

Rewrite layout-sensitive tests so they:

- compare by logical indices when possible
- otherwise use explicit column-major expectations
- avoid `as_slice`-style assumptions

**Step 4: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-tensorbackend --lib
cargo nextest run --release -p tensor4all-core --test tensor_basic --test tensor_permute --test linalg_qr --test linalg_svd
```

Expected: PASS with column-major expectations.

**Step 5: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/layout.rs crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs crates/tensor4all-core/src/defaults/qr.rs crates/tensor4all-core/src/defaults/svd.rs crates/tensor4all-core/tests/tensor_basic.rs crates/tensor4all-core/tests/tensor_permute.rs crates/tensor4all-core/tests/linalg_qr.rs crates/tensor4all-core/tests/linalg_svd.rs
git commit -m "feat: switch structured snapshot semantics to column-major"
```

### Task 5: Update Python, C API, and HDF5 boundaries to structured column-major semantics

**Files:**
- Modify: `crates/tensor4all-capi/src/tensor.rs`
- Modify: `crates/tensor4all-capi/src/simplett.rs`
- Modify: `crates/tensor4all-hdf5/src/itensor.rs`
- Modify: `python/tensor4all/src/tensor4all/tensor.py`
- Modify: `python/tensor4all/src/tensor4all/simplett.py`
- Modify: `README.md`
- Modify: `docs/CAPI_DESIGN.md`

**Step 1: C API**

Make dense flat-buffer input/output explicitly column-major and update docs/comments accordingly.

**Step 2: Python**

Ensure:

- `from_numpy` copies by logical indices
- `to_numpy` returns `order="F"`-compatible arrays
- reshape/flatten semantics match Fortran order

**Step 3: HDF5**

Keep ITensors.jl compatibility only, but preserve `axis_classes`/structured layout when roundtripping snapshots where the format supports it. Remove obsolete row-major conversion logic.

**Step 4: README/docs**

State explicitly that:

- internal dense materialization is column-major
- public flat-buffer semantics are column-major
- NumPy users should expect `order="F"` semantics at the boundary

**Step 5: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-capi -p tensor4all-hdf5
python3 python/tensor4all/scripts/build_capi.py
cd python/tensor4all && PYTHONPATH=src uv run --with pytest --with numpy --with cffi python -m pytest tests/test_tensor.py tests/test_simplett.py
```

Expected: PASS with updated boundary semantics.

**Step 6: Commit**

```bash
git add crates/tensor4all-capi/src/tensor.rs crates/tensor4all-capi/src/simplett.rs crates/tensor4all-hdf5/src/itensor.rs python/tensor4all/src/tensor4all/tensor.py python/tensor4all/src/tensor4all/simplett.py README.md docs/CAPI_DESIGN.md
git commit -m "feat: expose structured column-major semantics at boundaries"
```

### Task 6: Workspace-wide verification and cleanup

**Files:**
- Modify as needed: remaining layout-sensitive tests/docs discovered during verification

**Step 1: Regenerate API docs**

Run:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

**Step 2: Run formatting and lint**

Run:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

**Step 3: Run the full test suite**

Run:

```bash
cargo nextest run --release --workspace
```

**Step 4: Record residual issues**

If any tests are intentionally deferred because of unrelated upstream work, record them in the branch notes with exact file names and reasons. Otherwise require zero known failures.

**Step 5: Commit**

```bash
git add docs/api
git commit -m "chore: verify tenferro column-major structured storage migration"
```
