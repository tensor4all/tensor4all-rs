# Tenferro Backend Big-Bang Replacement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace tensor4all-rs backend in one shot with tenferro (`tenferro-tensor` + `tenferro-einsum` + `tenferro-linalg` + `tenferro-dyadtensor`) for `storage`, `einsum`, `linalg`, and AD-related scalar/tensor plumbing.

**Architecture:** Keep high-level tensor4all APIs (`TensorDynLen`, `Storage`, `AnyScalar`, factorization APIs) but switch their internals to tenferro-backed execution. Remove legacy mdarray/faer/lapack/libtorch backend code paths in the same PR (no compatibility layer, no fallback path). Add a focused adapter layer inside `tensor4all-tensorbackend` for data layout conversion, integer-label einsum mapping, and AD wrapper conversions.

**Tech Stack:** Rust, tenferro-rs crates (`tenferro-tensor`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-dyadtensor`), tensor4all-core/tensorbackend/simplett, cargo-nextest (`--release`)

---

## Preconditions (Hard Gates)

1. `tenferro-rs` side must provide required APIs in the target revision used by tensor4all:
   - AD-aware contraction/linalg entry points with non-`_ad` user-facing naming policy (internally alias acceptable).
   - Scalar AD ops required by tensor4all (`conj`, `sqrt`, `powf`, `powi`) with complex-safe behavior.
   - Binary scalar AD rules (`mul`, `div`; and preferably `add`, `sub` for closure) must be defined in `chainrules-scalarops` first, then reused from `tenferro-dyadtensor` (no duplicated local rule math).
   - Mixed scalar operator coverage must exist for common combinations used by tensor4all (`f64`/`Complex64` with `AdScalar`/`DynAdValue`) without exposing tape symbols at call sites.
   - AD-preserving complex composition for both typed and dynamic APIs:
     - typed: `complex(re: AdScalar<Real>, im: AdScalar<Real>) -> AdScalar<Complex>`
     - dyn: equivalent constructors/helpers on `DynAdValue` / `DynAdTensor`.
   - Explicit AD metadata drop path (`AdScalar<T> -> T`) for safe scalar extraction.
2. Runtime scope is CPU-only for cutover v1 unless tenferro CUDA/ROCm path is actually implemented (current `tenferro-dyadtensor` runtime wrapper still returns unsupported for CUDA/ROCm ops).
3. One-shot cutover is done on a dedicated branch/worktree and merged only after full workspace tests pass.

---

## Extension Principles (tenferro + tensor4all)

- Upstream-first for generic functionality: if a feature is broadly useful (scalar ops, AD-safe real/imag/complex compose, mixed-type arithmetic), extend `tenferro-rs` first, then consume it from tensor4all.
- KISS: add the smallest composable public API that solves the class of use-cases; avoid one-off knobs for tensor4all internals.
- DRY: keep rule definitions in one place (`chainrules-scalarops` for scalar calculus; dyadtensor for AD container propagation; tensor/einsum/linalg for primal kernels).
- Layering: do not bypass abstractions from downstream crates. tensor4all should call high-level tenferro APIs and should not depend on dyadtensor internals or tape implementation details.
- No ad-hoc fast paths that diverge primal and AD semantics unless they are generalized and tested at the correct layer.

---

## Dependency Workflow (Local Path During Development)

During implementation:
- depend on local sibling checkout (`../tenferro-rs/...`) via `path` dependencies for fast edit-test cycle across both repos.

Before opening final PRs:
- update tensor4all dependency references from local `path` to remote git refs (branch or commit SHA) so CI is reproducible.
- keep tenferro PR and tensor4all PR in sync; after tenferro merge, re-pin tensor4all to merged SHA and rerun full CI.

---

## DiagTensor Placement Decision (Must Decide Before Task 3)

**Recommendation:** `DiagTensor` should **not** be introduced first in `tenferro-dyadtensor`.

Reasoning:
- `tenferro-dyadtensor` is an AD wrapper layer; diagonal storage is a tensor representation concern, not an AD-layer concern.
- If implemented only in dyadtensor, primal-only (`non-AD`) and AD paths diverge, increasing maintenance and behavior mismatch risk.
- Correct layering is:
  - representation in `tenferro-tensor` (or equivalent low-level tensor crate),
  - contraction optimization in `tenferro-einsum`,
  - AD propagation in `tenferro-dyadtensor` by wrapping the same primal representation.

**Big-bang v1 choice:** keep `tensor4all`'s `DiagStorage` API, and lower it to tenferro execution through adapter logic (hyperedge/repeated-label subscripts), without adding a dyadtensor-only diag type.

**Future upstream target (optional v2):** add native diagonal representation to tenferro core tensor/einsum layers, then remove tensor4all-local diagonal lowering code.

---

### Task 1: Freeze Expected Behavior with Golden Tests

**Files:**
- Create: `crates/tensor4all-core/tests/tenferro_backend_golden_contract.rs`
- Create: `crates/tensor4all-core/tests/tenferro_backend_golden_linalg.rs`
- Create: `crates/tensor4all-tensorbackend/tests/tenferro_backend_golden_storage.rs`

**Step 1: Add golden tests for contraction behavior**

Cover:
- dense f64 / dense c64 contract
- diag-including hyperedge contraction
- scalar output (`->[]`)
- mixed real/complex promotion

**Step 2: Add golden tests for linalg behavior**

Cover:
- SVD/QR shape convention and reconstruction
- factorize output rank and index wiring
- existing `rtol` behavior smoke checks

**Step 3: Add golden tests for scalar/storage operations**

Cover:
- `AnyScalar` arithmetic and conversion
- `Storage::scale`, `try_add`, `try_sub`, `inner_product`
- conjugation and complex conversion

**Step 4: Run tests in release mode**

Run: `cargo nextest run --release -p tensor4all-core -p tensor4all-tensorbackend`

Expected: PASS (baseline captured before cutover).

**Step 5: Commit**

```bash
git add crates/tensor4all-core/tests/tenferro_backend_golden_contract.rs crates/tensor4all-core/tests/tenferro_backend_golden_linalg.rs crates/tensor4all-tensorbackend/tests/tenferro_backend_golden_storage.rs
git commit -m "test: add golden coverage before tenferro backend big-bang cutover"
```

---

### Task 2: Replace Workspace Dependencies and Feature Flags

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/tensor4all-tensorbackend/Cargo.toml`
- Modify: `crates/tensor4all-core/Cargo.toml`
- Modify: `crates/tensor4all-simplett/Cargo.toml`

**Step 1: Add tenferro dependencies at workspace level**

Add pinned git dependencies (or fixed local path for temporary local integration) for:
- `tenferro-tensor`
- `tenferro-einsum`
- `tenferro-linalg`
- `tenferro-dyadtensor`
- required tenferro support crates (`tenferro-prims`, `tenferro-algebra`, `tenferro-device`) if directly needed

Development default in this project:
- use `path = "../tenferro-rs/<crate>"` while both repos are under active edit.
- switch to git ref before PR finalization.

**Step 2: Remove legacy backend feature matrix**

Remove:
- `backend-faer`
- `backend-lapack`
- `backend-libtorch`

Add single backend feature:
- `backend-tenferro` (default)

**Step 3: Remove obsolete crate dependency wiring**

Remove direct dependence on:
- `mdarray-einsum` backend route from `tensor4all-tensorbackend`
- `tch`-backed feature path

**Step 4: Compile check**

Run: `cargo check --workspace`

Expected: Fails only on yet-to-be-migrated call sites, not on dependency resolution.

**Step 5: Commit**

```bash
git add Cargo.toml crates/tensor4all-tensorbackend/Cargo.toml crates/tensor4all-core/Cargo.toml crates/tensor4all-simplett/Cargo.toml
git commit -m "build: switch backend dependency model to tenferro-only"
```

---

### Task 3: Rewrite tensor4all-tensorbackend Core Backend Layer

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/src/backend.rs`
- Modify: `crates/tensor4all-tensorbackend/src/einsum.rs`
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Delete: `crates/tensor4all-tensorbackend/src/torch/mod.rs`
- Delete: `crates/tensor4all-tensorbackend/src/torch/storage.rs`
- Create: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`

**Step 1: Add bridge module for data conversion**

Implement:
- row-major aware conversion between tensor4all storage data and `tenferro_tensor::Tensor`
- integer label mapping (`DynId`/internal IDs -> `u32`) for `tenferro_einsum::Subscripts`
- helper for diag materialization policy
  - default: preserve tensor4all `DiagStorage` externally, lower to tenferro einsum inputs internally

**Step 2: Replace einsum implementation**

`einsum_storage` must call tenferro einsum APIs (prefer `einsum_with_subscripts` path to avoid string-label limits).

**Step 3: Replace SVD/QR backend wrappers**

`svd_backend` / `qr_backend` should be tenferro-linalg wrappers; adapt outputs to existing tensor4all expectations.

**Step 4: Remove torch-specific variants and methods**

Remove all `Torch*` enum variants and torch-only autograd dispatch from `Storage`/`AnyScalar`.

**Step 5: Build and run crate tests**

Run:
- `cargo test -p tensor4all-tensorbackend --release`
- `cargo nextest run --release -p tensor4all-tensorbackend`

Expected: PASS; no `backend-libtorch` code remains.

**Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/lib.rs crates/tensor4all-tensorbackend/src/backend.rs crates/tensor4all-tensorbackend/src/einsum.rs crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-tensorbackend/src/any_scalar.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs
git rm crates/tensor4all-tensorbackend/src/torch/mod.rs crates/tensor4all-tensorbackend/src/torch/storage.rs
git commit -m "refactor(tensorbackend): big-bang switch to tenferro backend"
```

---

### Task 4: Rewrite tensor4all-core Contraction Path to tenferro Einsum

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/contract.rs`

**Step 1: Remove direct `mdarray_einsum` dependency from contract path**

Replace `TypedTensor`/`einsum_optimized` flow with:
- adapter-generated `tenferro_tensor::Tensor` operands
- `tenferro_einsum` contraction with integer subscripts

**Step 2: Preserve current diagonal hyperedge semantics**

Maintain current union-find grouping logic, but emit grouped subscripts for tenferro execution.
Do not require a dyadtensor-native diag storage type in v1.

**Step 3: Preserve external index ordering contract**

Ensure output index mapping back to `DynIndex` remains identical to old behavior.

**Step 4: Run focused tests**

Run: `cargo nextest run --release -p tensor4all-core --test contract`

Expected: PASS, including diag-related cases.

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/contract.rs
git commit -m "refactor(core): move multi-tensor contract engine to tenferro einsum"
```

---

### Task 5: Rewrite tensor4all-core / simplett Linalg Integration

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/factorize.rs`

**Step 1: Update call sites to new backend wrappers**

Adapt SVD/QR extraction to tenferro return formats and keep tensor4all output conventions.

**Step 2: Validate truncation behavior still matches tensor4all policy**

Keep `rtol`/`max_rank` semantics unchanged at tensor4all API level.

**Step 3: Verify matrix reconstruction contracts**

For each decomposition path, assert reconstruction error against existing tolerances.

**Step 4: Run tests**

Run:
- `cargo nextest run --release -p tensor4all-core`
- `cargo nextest run --release -p tensor4all-simplett`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/svd.rs crates/tensor4all-core/src/defaults/qr.rs crates/tensor4all-core/src/defaults/factorize.rs crates/tensor4all-simplett/src/mpo/factorize.rs
git commit -m "refactor(core): route linalg paths through tenferro backend"
```

---

### Task 6: Big-Bang AD API Cutover (Replace libtorch-style AD surface)

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs` (only if AD helper methods are exposed here)
- Create: `crates/tensor4all-tensorbackend/tests/tenferro_ad_api.rs`

**Step 1: Introduce tenferro-backed AD scalar/tensor variants**

Replace torch variants with tenferro AD wrappers:
- scalar level via `AdScalar<T>` (or `DynAdValue` when needed)
- tensor level via `AdTensor<T>` (or `DynAdTensor` for dynamic dispatch)
- implement arithmetic operator traits for both typed and dynamic AD scalars:
  - `AdScalar<T>`: `Add/Sub/Mul/Div` for `AdScalar<T> <op> AdScalar<T>`
  - `DynAdValue` (DynAdScalar): `Add/Sub/Mul/Div` for dynamic dispatch with dtype checks
  - mixed scalar forms on both sides (`f64 * DynAdValue`, `DynAdValue * f64`, `Complex64` variants)
  - define and document promotion rules (`f64` + `C64` -> `C64`, etc.)
- include dynamic complex composition entry points:
  - `DynAdValue::complex(re, im)` (or equivalent free function)
  - `DynAdTensor::complex(re, im)` (or equivalent free function)

**Step 2: Define explicit AD-drop conversion policy**

Rules:
- implicit extraction to plain `f64`/`Complex64` from AD-carrying scalar is disallowed or clearly fallible
- explicit drop path required (`into_primal`-style API)
- for dynamic complex composition, add runtime checks:
  - dtype compatibility (`F32+F32->C32`, `F64+F64->C64`)
  - AD mode/tape compatibility for reverse-mode operands (mismatch -> error)
- when converting complex AD value/tensor to real domain, require runtime `is_real` guard:
  - `is_real_exact()` for strict zero-imag checks
  - `is_real_eps(eps)` for tolerance-based checks
  - conversion API must return error if guard fails (no silent imaginary discard)

**Step 3: Re-implement `requires_grad/grad/backward/detach` surface**

Use tenferro AD model and clearly document what is supported in v1 (metadata-only vs full pullback).

**Step 4: Add AD API tests**

Cover:
- scalar conj/sqrt/powf/powi propagation
- `mul/div` (and `add/sub`) AD rule propagation via `chainrules-scalarops`
- mixed operator coverage for dynamic AD scalars (`f64`/`Complex64` lhs/rhs combinations)
- reverse metadata propagation (`node_id`, `tape_id`)
- explicit AD drop conversion behavior

**Step 5: Run tests**

Run: `cargo nextest run --release -p tensor4all-tensorbackend --test tenferro_ad_api`

Expected: PASS.

**Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/any_scalar.rs crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-tensorbackend/tests/tenferro_ad_api.rs
git commit -m "refactor(ad): replace torch-style AD plumbing with tenferro dyadtensor model"
```

---

### Task 7: Enforce PyTorch-Compatible Complex Differentiation Semantics

**Files:**
- Create: `crates/tensor4all-core/tests/complex_autograd_semantics.rs`
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs` (if rule adjustments are required)

**Step 1: Add complex AD semantic tests**

Check against PyTorch-compatible expectations for:
- `x^2`
- `sqrt(x)`
- `conj(x)`

**Step 2: Validate scalar-rule consistency**

Confirm chain-rule composition uses same convention across composed operations.

**Step 3: Run tests**

Run: `cargo nextest run --release -p tensor4all-core --test complex_autograd_semantics`

Expected: PASS.

**Step 4: Commit**

```bash
git add crates/tensor4all-core/tests/complex_autograd_semantics.rs crates/tensor4all-tensorbackend/src/any_scalar.rs
git commit -m "test(ad): enforce pytorch-compatible complex differentiation semantics"
```

---

### Task 8: Delete Obsolete Backend Code and Update Docs

**Files:**
- Delete: `crates/mdarray-einsum/*` (if no longer used anywhere)
- Modify: `README.md`
- Modify: `docs/api/*.md` (regenerated)
- Modify: backend-related docs/comments across affected crates

**Step 1: Remove obsolete code paths and crate memberships**

Remove old backend-only code and workspace references that are now dead.

**Step 2: Regenerate API docs**

Run: `cargo run -p api-dump --release -- . -o docs/api`

**Step 3: Update README/backend description**

Describe tenferro-only backend and AD status.

**Step 4: Run lint and full tests**

Run:
- `cargo fmt --all`
- `cargo clippy --workspace`
- `cargo nextest run --release --workspace`

Expected: all PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove legacy backends and finalize tenferro big-bang migration"
```

---

## Execution Order Summary

1. Golden tests (Task 1)
2. Dependency/feature cutover (Task 2)
3. tensorbackend rewrite (Task 3)
4. core contract rewrite (Task 4)
5. linalg rewrite (Task 5)
6. AD surface rewrite (Task 6)
7. complex gradient semantics (Task 7)
8. dead code removal + verification (Task 8)

## Risk Controls

- Keep golden tests from Task 1 untouched until final merge.
- Treat row-major vs column-major layout mismatch as highest-priority risk.
- If `tenferro-dyadtensor` full pullback/HVP APIs are not ready, explicitly scope v1 AD behavior and fail fast on unsupported operations.
- No partial fallback backend: either all tests pass with tenferro backend, or cutover is rejected.
