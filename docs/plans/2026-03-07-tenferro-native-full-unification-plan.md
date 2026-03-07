# Tenferro Native Full Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `DynAdTensor` the single canonical payload of `TensorDynLen` and remove `TensorData` / internal `Storage` from tensor4all core tensor representation.

**Architecture:** Convert `TensorDynLen` into a thin index wrapper over `DynAdTensor`, move all tensor algebra and linalg to native execution, and demote `Storage` to an explicit import/export boundary. Keep `treetn` / `TensorTrain` APIs intact while removing hybrid backend state internally.

**Tech Stack:** Rust, tenferro-rs, tensor4all-core, tensor4all-tensorbackend, tensor4all-treetn, tensor4all-itensorlike, cargo-nextest (`--release`)

---

### Task 1: Introduce native-only TensorDynLen scaffolding

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `docs/api/tensor4all_core.md` via `api-dump` after code changes
- Test: `crates/tensor4all-core/tests/tensor_native_ad.rs`

**Step 1: Write failing tests**

Cover:
- plain dense construction preserves a mandatory native payload
- plain diag construction preserves a mandatory native payload
- `from_native(...).into_native()` round-trips payload dimensions and mode

**Step 2: Run focused tests to verify failure**

Run:
```bash
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
```

**Step 3: Implement minimal scaffolding**

Change `TensorDynLen` to:
- replace `native: Option<DynAdTensor>` with `native: DynAdTensor`
- add `from_storage(indices, storage) -> Result<Self>`
- add `into_native(self) -> DynAdTensor`
- make `as_native(&self) -> &DynAdTensor`

Do not remove old constructors yet; route them through the new fallible constructor first.

**Step 4: Re-run focused tests**

Run:
```bash
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
```

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/tests/tensor_native_ad.rs
git commit -m "refactor(core): make TensorDynLen carry mandatory native payload"
```

### Task 2: Move core tensor algebra to native-only execution

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/defaults/contract.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Test: `crates/tensor4all-core/tests/tensor_contract_multi_pair_equivalence.rs`

**Step 1: Write failing tests**

Cover:
- `contract_multi` native path matches sequential binary contract
- `contract_multi` zero-masked regression stays fixed
- `permute`, `contract`, `tensordot`, and `outer_product` no longer rely on optional native checks

**Step 2: Run focused tests to verify failure**

Run:
```bash
cargo nextest run --release -p tensor4all-core --test tensor_contract_multi_pair_equivalence
```

**Step 3: Implement minimal code**

Update:
- `permute`
- `permute_indices`
- `contract`
- `tensordot`
- `outer_product`

to always execute on `DynAdTensor`.

Remove the storage fallback branches in these methods.
Keep explicit storage conversion only inside `to_storage()`.

**Step 4: Re-run focused tests**

Run:
```bash
cargo nextest run --release -p tensor4all-core --test tensor_contract_multi_pair_equivalence
```

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/src/defaults/contract.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs crates/tensor4all-core/tests/tensor_contract_multi_pair_equivalence.rs
git commit -m "refactor(core): route tensor algebra through native payload only"
```

### Task 3: Move scalar algebra and reductions to native-only execution

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Test: `crates/tensor4all-core/tests/tensor_native_ad.rs`

**Step 1: Write failing tests**

Cover:
- `sum`
- `only`
- `inner_product`
- `scale`
- `axpby`
- `conj`

with no reliance on storage snapshots in the implementation.

**Step 2: Run focused tests to verify failure**

Run:
```bash
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
```

**Step 3: Implement minimal code**

Remove storage-based branches from the scalar algebra and reduction methods.
Use explicit `to_storage()` only in tests or boundary code, not in core execution.

**Step 4: Re-run focused tests**

Run:
```bash
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
```

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs crates/tensor4all-core/tests/tensor_native_ad.rs
git commit -m "refactor(core): route scalar algebra and reductions through native payload"
```

### Task 4: Make linalg and factorization native-first by construction

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Test: `crates/tensor4all-core/tests/linalg_qr.rs`
- Test: `crates/tensor4all-core/tests/linalg_svd.rs`
- Test: `crates/tensor4all-core/tests/linalg_factorize.rs`

**Step 1: Write failing tests**

Cover:
- nontrivial split QR reconstruction
- nontrivial split SVD reconstruction
- factorize reconstruction without storage dispatch

**Step 2: Run focused tests to verify failure**

Run:
```bash
cargo nextest run --release -p tensor4all-core --test linalg_qr --test linalg_svd --test linalg_factorize
```

**Step 3: Implement minimal code**

Change:
- `qr_with`
- `svd_with`
- `factorize`

to dispatch from `DynAdTensor` directly.

Remove storage-type matching in `factorize`.
Keep explicit storage conversion only for extracting singular values or boundary snapshots.

**Step 4: Re-run focused tests**

Run:
```bash
cargo nextest run --release -p tensor4all-core --test linalg_qr --test linalg_svd --test linalg_factorize
```

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/qr.rs crates/tensor4all-core/src/defaults/svd.rs crates/tensor4all-core/src/defaults/factorize.rs crates/tensor4all-core/tests/linalg_qr.rs crates/tensor4all-core/tests/linalg_svd.rs crates/tensor4all-core/tests/linalg_factorize.rs
git commit -m "refactor(core): make linalg and factorize native-only"
```

### Task 5: Replace extraction and query APIs with explicit boundary methods

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify downstream uses in:
  - `crates/tensor4all-hdf5/`
  - `crates/tensor4all-capi/`
  - `crates/tensor4all-treetn/`
  - `crates/tensor4all-itensorlike/`

**Step 1: Write failing tests**

Cover:
- dense extraction via explicit `to_storage()`
- diag extraction via explicit `to_storage()`
- downstream call sites compile without `storage()` / `tensor_data()`

**Step 2: Run focused tests to verify failure**

Run:
```bash
cargo test --release -p tensor4all-core --no-run
cargo test --release -p tensor4all-hdf5 --no-run
cargo test --release -p tensor4all-capi --no-run
```

**Step 3: Implement minimal code**

Add:
- `to_storage(&self) -> Result<Arc<Storage>>`

Replace downstream direct storage access with explicit materialization boundaries.

Remove:
- `storage()`
- `try_storage()`
- `materialize_storage()`

from the public `TensorDynLen` API once call sites are migrated.

**Step 4: Re-run focused compile/test checks**

Run:
```bash
cargo nextest run --release -p tensor4all-core -p tensor4all-hdf5 -p tensor4all-capi
```

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-hdf5 crates/tensor4all-capi crates/tensor4all-treetn crates/tensor4all-itensorlike
git commit -m "refactor(core): make storage materialization an explicit boundary"
```

### Task 6: Delete TensorData and remove legacy construction paths

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Delete or heavily prune legacy `TensorData` code under `crates/tensor4all-core/src/defaults/`
- Update tests in `crates/tensor4all-core/tests/`

**Step 1: Write failing compile checks**

Run:
```bash
cargo test --release -p tensor4all-core --no-run
```

Expected: compile errors from remaining `TensorData` references after the API removals.

**Step 2: Remove dead code**

Delete:
- `TensorData`
- `tensor_data()` accessors
- any lazy outer-product/permutation machinery that only existed to support `TensorData`

Replace constructor usage with:
- `TensorDynLen::from_native(...)`
- `TensorDynLen::from_storage(...)`

**Step 3: Update tests**

Rewrite tests that directly inspected `TensorData` internals to use public tensor APIs.

**Step 4: Re-run focused core tests**

Run:
```bash
cargo nextest run --release -p tensor4all-core
```

**Step 5: Commit**

```bash
git add crates/tensor4all-core
git commit -m "refactor(core): remove TensorData and legacy tensor backend state"
```

### Task 7: Verify treetn and TensorTrain compatibility after full unification

**Files:**
- No new files unless regressions require changes

**Step 1: Run focused downstream verification**

Run:
```bash
cargo nextest run --release -p tensor4all-treetn -p tensor4all-itensorlike -p tensor4all-partitionedtt
```

**Step 2: Fix only the regressions caused by the backend unification**

Do not widen scope beyond backend migration fallout.

**Step 3: Re-run the same focused verification**

Run:
```bash
cargo nextest run --release -p tensor4all-treetn -p tensor4all-itensorlike -p tensor4all-partitionedtt
```

**Step 4: Commit**

```bash
git add -A
git commit -m "fix: adapt downstream tensor network crates to native-only TensorDynLen"
```

### Task 8: Final verification and API dump refresh

**Files:**
- Update generated API docs under `docs/api/`

**Step 1: Refresh API dump**

Run:
```bash
cargo run -p api-dump --release -- . -o docs/api
```

**Step 2: Run formatting**

Run:
```bash
cargo fmt --all
```

**Step 3: Run lint**

Run:
```bash
cargo clippy --workspace
```

**Step 4: Run full workspace tests**

Run:
```bash
cargo nextest run --release --workspace
```

**Step 5: Commit**

```bash
git add docs/api docs/plans
git commit -m "docs: refresh API dump after native backend unification"
```
