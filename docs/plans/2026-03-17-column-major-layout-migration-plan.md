# Column-Major Layout Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert tensor4all-rs from row-major dense semantics to a clean column-major model across internal storage, public flat-buffer APIs, Python bindings, and HDF5 interoperability, while preserving ITensors.jl HDF5 compatibility.

**Architecture:** First centralize all layout-sensitive math in backend-owned helpers, then flip backend/core semantics to column-major, and finally update all external boundaries and documentation to match. Temporary adapters are allowed only at boundaries during the intermediate phase and must be removed before completion.

**Tech Stack:** Rust, mdarray, tenferro-rs, cargo fmt, cargo clippy, cargo nextest (`--release`), Python/NumPy, C API tests, HDF5 interoperability tests

---

## Task 1: Inventory and freeze layout-sensitive behavior

**Files:**
- Create: `docs/plans/2026-03-17-column-major-layout-migration-audit.md`
- Test: `crates/tensor4all-tensorbackend/src/storage.rs`
- Test: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Test: `crates/tensor4all-core/src/defaults/qr.rs`
- Test: `crates/tensor4all-core/src/defaults/svd.rs`

**Step 1: Record the current layout-sensitive call sites**

Use:

```bash
rg -n "row-major|row major|column-major|column major|reshape_row_major|compute.*strides|flat.*index|order=\"F\"|order=\"C\"" crates python docs README.md
```

Copy the relevant groups into the audit note, organized by:
- backend
- core
- boundary
- docs/tests

**Step 2: Add or tighten backend golden tests for logical offset mapping**

Add tests in `crates/tensor4all-tensorbackend/src/storage.rs` for:
- 2D and 3D `multi_index -> flat_offset`
- `flat_offset -> multi_index`
- reshape/relinearization behavior with unit dimensions

These tests should describe expected logical behavior explicitly so they can be updated during the semantic flip.

**Step 3: Run targeted tests**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend --lib
```

Expected: PASS

**Step 4: Commit**

```bash
git add docs/plans/2026-03-17-column-major-layout-migration-audit.md crates/tensor4all-tensorbackend/src/storage.rs
git commit -m "test: freeze layout-sensitive backend behavior"
```

## Task 2: Introduce a centralized backend layout module

**Files:**
- Create: `crates/tensor4all-tensorbackend/src/layout.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`

**Step 1: Write the failing compile-time refactor**

Create backend helpers with row-major semantics first:
- `dense_strides(dims: &[usize]) -> Vec<usize>`
- `multi_index_to_offset(dims: &[usize], idx: &[usize]) -> Result<usize>`
- `offset_to_multi_index(dims: &[usize], offset: usize) -> Result<Vec<usize>>`
- `reshape_linearized_dims(src_dims: &[usize], dst_dims: &[usize]) -> Result<()>`

Initially preserve current behavior.

**Step 2: Route existing stride / offset code through the new module**

Replace duplicated stride math in:
- `storage.rs`
- `tenferro_bridge.rs`

Avoid naming helpers `row_major_*`; prefer semantic names such as `dense_strides` and `multi_index_to_offset`.

**Step 3: Run targeted tests**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend --lib
```

Expected: PASS with no semantic change yet.

**Step 4: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/layout.rs crates/tensor4all-tensorbackend/src/lib.rs crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs
git commit -m "refactor: centralize dense layout helpers"
```

## Task 3: Remove explicit row-major assumptions from high-level code paths

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-simplett/src/tensortrain.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/mod.rs`

**Step 1: Rename row-major specific helper names**

Examples:
- `reshape_row_major_dyn_ad_tensor` -> a layout-neutral name such as `reshape_linearized_dyn_ad_tensor`
- row-major comments -> “current dense linearization” or equivalent where possible

**Step 2: Route all internal reshape/flatten logic through backend helpers**

No core crate should compute dense strides or linearized offsets itself after this step.

**Step 3: Update tests to validate logical tensor equality rather than ad hoc flat arrays where possible**

Prefer:
- dense materialization once
- tensor subtraction + `maxabs()`
- multi-index queries

**Step 4: Run targeted tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core
cargo nextest run --release -p tensor4all-simplett
cargo nextest run --release -p tensor4all-treetn
```

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/qr.rs crates/tensor4all-core/src/defaults/svd.rs crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-simplett/src/tensortrain.rs crates/tensor4all-treetn/src/treetn/mod.rs
git commit -m "refactor: make high-level reshape paths layout-neutral"
```

## Task 4: Flip backend dense semantics to column-major

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/layout.rs`
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/einsum.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`

**Step 1: Update layout helpers to column-major semantics**

Change:
- stride generation
- offset mapping
- reshape linearization assumptions

**Step 2: Update storage constructors/exporters**

Ensure:
- dense storage creation from flat buffers interprets them as column-major
- dense export returns column-major flat buffers

**Step 3: Update tenferro bridge reshape and native conversion helpers**

Remove row-major forcing behavior and align reshape/native conversions with column-major semantics.

**Step 4: Update backend unit tests**

Replace old row-major expectations with column-major ones.

**Step 5: Run targeted tests**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend --lib
```

Expected: PASS after updating expectations.

**Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/layout.rs crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-tensorbackend/src/einsum.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs
git commit -m "feat: switch backend dense layout to column-major"
```

## Task 5: Flip core dense constructors, reshape, and linalg semantics

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Test: `crates/tensor4all-core/tests/linalg_qr.rs`
- Test: `crates/tensor4all-core/tests/linalg_svd.rs`
- Test: `crates/tensor4all-core/tests/tensor_permute.rs`

**Step 1: Update `from_dense*` / `to_dense*` semantics to column-major**

Document and enforce that flat dense data is now column-major.

**Step 2: Update QR/SVD flattening and reconstruction paths**

Make all flatten-to-matrix and reshape-back operations use the new backend semantics.

**Step 3: Rewrite layout-sensitive tests**

Update comments and expected flat vectors to column-major where flat literals remain necessary.

**Step 4: Run targeted tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core
```

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/src/defaults/qr.rs crates/tensor4all-core/src/defaults/svd.rs crates/tensor4all-core/tests/linalg_qr.rs crates/tensor4all-core/tests/linalg_svd.rs crates/tensor4all-core/tests/tensor_permute.rs
git commit -m "feat: switch core dense semantics to column-major"
```

## Task 6: Preserve ITensors.jl HDF5 compatibility under column-major semantics

**Files:**
- Modify: `crates/tensor4all-hdf5/src/lib.rs`
- Modify: `crates/tensor4all-hdf5/src/layout.rs`
- Modify: `crates/tensor4all-hdf5/tests/*` (existing HDF5 tests)

**Step 1: Re-evaluate conversion layer**

Remove or simplify row-major/column-major conversion code where internal and ITensors.jl semantics now align.

**Step 2: Keep the public rule unchanged**

HDF5 support remains ITensors.jl-compatible only.

**Step 3: Update tests to assert roundtrip interoperability**

Cover:
- tensor4all -> HDF5 -> tensor4all
- ITensors.jl-compatible dense layout assumptions

**Step 4: Run targeted tests**

Run:

```bash
cargo nextest run --release -p tensor4all-hdf5
```

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-hdf5/src/lib.rs crates/tensor4all-hdf5/src/layout.rs crates/tensor4all-hdf5/tests
git commit -m "feat: align HDF5 layout with column-major core semantics"
```

## Task 7: Flip C API dense buffer contracts to column-major

**Files:**
- Modify: `crates/tensor4all-capi/src/tensor.rs`
- Modify: `crates/tensor4all-capi/src/simplett.rs`
- Modify: `docs/CAPI_DESIGN.md`
- Test: `crates/tensor4all-capi/tests/*` (existing C API tests if present)

**Step 1: Update function behavior**

Change dense constructors/getters to interpret and return flat buffers in column-major order.

**Step 2: Update docs and comments**

Remove row-major wording and replace with column-major wording.

**Step 3: Add/adjust tests for column-major roundtrip**

Ensure C API import/export matches the documented flat-buffer semantics.

**Step 4: Run targeted tests**

Run:

```bash
cargo test --release -p tensor4all-capi
```

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-capi/src/tensor.rs crates/tensor4all-capi/src/simplett.rs docs/CAPI_DESIGN.md crates/tensor4all-capi/tests
git commit -m "feat: switch C API dense buffers to column-major"
```

## Task 8: Flip Python import/export semantics to column-major

**Files:**
- Modify: `python/tensor4all/src/tensor4all/tensor.py`
- Modify: `python/tensor4all/src/tensor4all/simplett.py`
- Modify: `python/tensor4all/tests/test_tensor.py`
- Modify: `docs/examples/python/*` where relevant
- Modify: `README.md`

**Step 1: Update Python boundary normalization**

Make Python import/export consistent with:
- column-major flat-buffer semantics
- `order="F"` reshape/flatten rules
- F-contiguous output where appropriate

**Step 2: Update tests**

Add or update tests so Python roundtrips match:

```python
np.reshape(arr, new_shape, order="F")
```

**Step 3: Update README**

Explicitly state that tensor4all-rs uses column-major internal dense storage and column-major linearization semantics.

**Step 4: Run targeted tests**

Run:

```bash
cd python/tensor4all
python scripts/build_capi.py
uv pip install -e .
pytest tests
```

Expected: PASS

**Step 5: Commit**

```bash
git add python/tensor4all/src/tensor4all/tensor.py python/tensor4all/src/tensor4all/simplett.py python/tensor4all/tests/test_tensor.py docs/examples/python README.md
git commit -m "feat: switch Python dense semantics to column-major"
```

## Task 9: Update higher-level crates, examples, and layout-sensitive tests

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs`
- Modify: `crates/tensor4all-quanticstransform/tests/integration_test.rs`
- Modify: `crates/tensor4all-itensorlike/tests/bug_fit_elementwise.rs`
- Modify: `crates/matrixci/src/util.rs`
- Modify: layout-sensitive comments/tests found in:
  - `crates/tensor4all-simplett/src/types.rs`
  - `crates/tensor4all-core/src/block_tensor.rs`
  - `crates/tensor4all-core/src/krylov.rs`
  - `crates/tensor4all-treetn/tests/*`

**Step 1: Rewrite stale comments and test assumptions**

Any comment or assertion that encodes row-major meaning must be updated or removed.

**Step 2: Prefer logical comparisons**

When updating tests, prefer:
- multi-index assertions
- tensor subtraction + `maxabs()`
- explicit materialization once per comparison

**Step 3: Run targeted tests**

Run:

```bash
cargo nextest run --release -p tensor4all-quanticstransform
cargo nextest run --release -p tensor4all-itensorlike
cargo nextest run --release -p matrixci
cargo nextest run --release -p tensor4all-simplett
cargo nextest run --release -p tensor4all-treetn
```

Expected: PASS

**Step 4: Commit**

```bash
git add crates/tensor4all-quanticstransform/src/affine.rs crates/tensor4all-quanticstransform/tests/integration_test.rs crates/tensor4all-itensorlike/tests/bug_fit_elementwise.rs crates/matrixci/src/util.rs crates/tensor4all-simplett/src/types.rs crates/tensor4all-core/src/block_tensor.rs crates/tensor4all-core/src/krylov.rs crates/tensor4all-treetn/tests
git commit -m "test: update higher-level layout assumptions to column-major"
```

## Task 10: Full cleanup and remove transitional wording

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/CAPI_DESIGN.md`
- Modify: all remaining files reported by:

```bash
rg -n "row-major|row major|reshape_row_major|C-order|column-major priority bug" crates python docs README.md AGENTS.md
```

**Step 1: Remove transitional names and comments**

Delete any remaining row-major-specific helper names, comments, and stale migration notes that are no longer true in the final state.

**Step 2: Keep only explicit boundary documentation**

Layout documentation should remain where necessary:
- README
- C API docs
- Python import/export docs
- HDF5 docs

**Step 3: Run linting**

Run:

```bash
cargo fmt --all
cargo clippy --workspace
```

Expected: PASS

**Step 4: Commit**

```bash
git add AGENTS.md README.md docs/CAPI_DESIGN.md crates python docs
git commit -m "docs: finalize column-major layout migration"
```

## Task 11: Final verification

**Files:**
- No new files

**Step 1: Run Rust workspace tests**

Run:

```bash
cargo nextest run --release --workspace
```

Expected: PASS

**Step 2: Run Python verification**

Run:

```bash
cd python/tensor4all
python scripts/build_capi.py
uv pip install -e .
pytest tests
cd ../..
for f in docs/examples/python/*.py; do
  python "$f"
done
```

Expected: PASS

**Step 3: Spot-check HDF5 and C API behavior if they have separate commands/scripts**

Run any existing crate-local verification commands needed for those boundaries.

**Step 4: Summarize breaking changes**

Prepare release notes covering:
- internal and public dense layout now column-major
- Python/C API reshape/flatten semantics now match `order="F"`
- stale row-major assumptions removed

**Step 5: Commit final verification-only adjustments if needed**

```bash
git add -A
git commit -m "chore: finalize column-major migration verification"
```

