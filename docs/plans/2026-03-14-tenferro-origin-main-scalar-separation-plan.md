# Tensor4all Scalar/Tensor API over Tenferro origin/main Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve distinct tensor4all `Scalar` and `TensorDynLen` public semantics on top of tenferro `origin/main`, including immutable AD APIs and dedicated public HVP support.

**Architecture:** Add a new rank-0 `Scalar` wrapper backed by tenferro native tensors, keep `TensorDynLen` as the normal tensor-facing type, and hide tenferro native payload types behind `pub(crate)` helpers. Rework scalar-returning and scalar-taking tensor APIs to use `Scalar`, add a tensor4all-owned `GradTape` and forward tangent surface, and expose HVP only through a dedicated helper that uses internal forward-over-reverse.

**Tech Stack:** Rust, tensor4all-core, tensor4all-tensorbackend, tenferro-rs `origin/main`, anyhow, cargo test / cargo nextest (`--release`)

---

### Task 1: Lock Down Public API Expectations with Failing Tests

**Files:**
- Create: `crates/tensor4all-core/tests/scalar_public_api.rs`
- Modify: `crates/tensor4all-core/tests/tensor_native_ad.rs`
- Test: `crates/tensor4all-core/tests/scalar_public_api.rs`

**Step 1: Write failing public-API tests for the new scalar semantics**

Add tests that assert:

- `TensorDynLen::sum()` returns `AnyScalar`
- `TensorDynLen::only()` returns `AnyScalar`
- `TensorDynLen::inner_product()` returns `AnyScalar`
- `AnyScalar` behaves as a rank-0 scalar value
- `TensorDynLen::scale()` and `axpby()` accept `AnyScalar`

Example test skeleton:

```rust
#[test]
fn sum_returns_rank0_scalar() {
    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::from_dense_f64(vec![i], vec![1.0, 2.0]);
    let sum: AnyScalar = tensor.sum();
    assert_eq!(sum.real(), 3.0);
}
```

**Step 2: Run tests to confirm the new API expectations fail where needed**

Run: `cargo nextest run --release -p tensor4all-core scalar_public_api`
Expected: FAIL because the new `Scalar` wrapper and related APIs do not exist yet.

**Step 3: Extend AD regression tests to target scalar semantics rather than native payload access**

Update `tensor_native_ad.rs` so scalar-result assertions are written against `AnyScalar`/`Scalar`
methods instead of direct native payload assumptions.

**Step 4: Re-run focused tests**

Run: `cargo nextest run --release -p tensor4all-core tensor_native_ad scalar_public_api`
Expected: still FAIL, but now the failures should point at missing scalar/public API changes instead
of ambiguous old expectations.

**Step 5: Commit**

```bash
git add crates/tensor4all-core/tests/scalar_public_api.rs crates/tensor4all-core/tests/tensor_native_ad.rs
git commit -m "test: lock scalar api expectations for tenferro origin main"
```

### Task 2: Introduce `Scalar` and Keep `AnyScalar` as a Compatibility Alias

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Test: `crates/tensor4all-core/tests/scalar_public_api.rs`

**Step 1: Replace the old scalar alias implementation with a real `Scalar` type**

Add:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Scalar {
    native: DynAdTensor,
}

pub type AnyScalar = Scalar;
```

with internal validation that `native.dims().is_empty()`.

**Step 2: Implement minimal scalar constructors and inspectors**

Add:

- `from_real`
- `from_complex`
- `real`
- `imag`
- `abs`
- `is_real`
- `is_complex`
- `is_zero`
- `conj`
- `mode`

Keep native accessors `pub(crate)`.

**Step 3: Restore existing ergonomic conversions**

Implement:

- `From<f64> for Scalar`
- `From<Complex64> for Scalar`
- `TryFrom<Scalar> for f64`
- `From<Scalar> for Complex64`

**Step 4: Re-export `Scalar` alongside `AnyScalar`**

Expose the new public type from tensorbackend so downstream crates can adopt it cleanly.

**Step 5: Run focused tests**

Run: `cargo nextest run --release -p tensor4all-core scalar_public_api`
Expected: fewer failures; scalar construction and inspection should now compile.

**Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/any_scalar.rs crates/tensor4all-tensorbackend/src/lib.rs crates/tensor4all-core/tests/scalar_public_api.rs
git commit -m "feat(tensorbackend): add rank-0 scalar wrapper"
```

### Task 3: Rework `TensorDynLen` Scalar Boundaries and Hide Native Payload Access

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Modify: `docs/api/tensor4all_core.md` after regeneration
- Test: `crates/tensor4all-core/tests/scalar_public_api.rs`

**Step 1: Make native tensor conversion helpers internal**

Change:

- `from_native`
- `as_native`
- `into_native`

to `pub(crate)` or move them to internal helper impl blocks.

**Step 2: Switch scalar-result methods to `Scalar`**

Update signatures and implementations for:

- `sum`
- `only`
- `inner_product`

to produce the new `Scalar` wrapper.

**Step 3: Switch scalar-input methods to `Scalar`**

Update signatures and implementations for:

- `scale`
- `axpby`

to accept `Scalar`/`AnyScalar`.

**Step 4: Preserve storage interop**

Leave `from_storage` and `to_storage` public, because they are tensor4all-owned boundaries.

**Step 5: Run focused tests**

Run: `cargo nextest run --release -p tensor4all-core scalar_public_api tensor_native_ad`
Expected: scalar API tests pass; AD tests may still fail until dedicated AD wrappers land.

**Step 6: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/src/lib.rs crates/tensor4all-core/tests/scalar_public_api.rs crates/tensor4all-core/tests/tensor_native_ad.rs
git commit -m "refactor(core): separate scalar api from tensor api"
```

### Task 4: Add Immutable Reverse-Mode and Forward-Mode Public AD APIs

**Files:**
- Create: `crates/tensor4all-core/src/ad.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `crates/tensor4all-core/tests/tensor_native_ad.rs`

**Step 1: Introduce `GradTape`**

Add a tensor4all-owned wrapper that can create unique reverse tape identities without exposing
tenferro `TapeId` publicly.

**Step 2: Add reverse-mode constructors**

Implement:

- `TensorDynLen::requires_grad(&self, tape: &GradTape) -> Result<Self>`
- `Scalar::requires_grad(&self, tape: &GradTape) -> Result<Self>`

These should return new values and should reject already-forward annotated values.

**Step 3: Add forward-mode constructors and accessors**

Implement:

- `TensorDynLen::with_tangent(&self, tangent: &TensorDynLen) -> Result<Self>`
- `Scalar::with_tangent(&self, tangent: &Scalar) -> Result<Self>`
- `TensorDynLen::tangent(&self) -> Option<TensorDynLen>`
- `Scalar::tangent(&self) -> Option<Scalar>`
- `mode() -> AdMode`

**Step 4: Explicitly reject ordinary public mixed-mode construction**

Return a clear error if public code tries to place both forward and reverse annotations on the same
value outside the dedicated HVP path.

**Step 5: Run AD tests**

Run: `cargo nextest run --release -p tensor4all-core tensor_native_ad`
Expected: PASS for forward-mode preservation and new reverse-mode surface checks.

**Step 6: Commit**

```bash
git add crates/tensor4all-core/src/ad.rs crates/tensor4all-core/src/lib.rs crates/tensor4all-tensorbackend/src/any_scalar.rs crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/tests/tensor_native_ad.rs
git commit -m "feat(core): add immutable ad api for tensors and scalars"
```

### Task 5: Add Dedicated Public HVP API without Exposing General Mixed-Mode

**Files:**
- Modify: `crates/tensor4all-core/src/ad.rs`
- Create: `crates/tensor4all-core/tests/hvp_public_api.rs`
- Test: `crates/tensor4all-core/tests/hvp_public_api.rs`

**Step 1: Write failing HVP API tests**

Add tests for:

- scalar loss HVP with one tensor input
- multi-input HVP ordering
- shape/index mismatch rejection
- non-reverse loss rejection
- non-scalar loss rejection

Example:

```rust
#[test]
fn scalar_loss_hvp_accepts_wrt_direction_pairs() {
    let tape = GradTape::new();
    let x = make_tensor(...).requires_grad(&tape).unwrap();
    let v = make_tensor(...);
    let loss = (&x * &x).sum();
    let hv = loss.hvp_tensors(&[(&x, &v)]).unwrap();
    assert_eq!(hv.len(), 1);
}
```

**Step 2: Implement `Scalar::hvp_tensors`**

Signature:

```rust
pub fn hvp_tensors(&self, wrt: &[(&TensorDynLen, &TensorDynLen)]) -> Result<Vec<TensorDynLen>>
```

Use internal forward-over-reverse wiring only inside this helper.

**Step 3: Keep mixed-mode internal**

Do not add any general public API that lets users manually construct mixed forward+reverse values.

**Step 4: Run focused tests**

Run: `cargo nextest run --release -p tensor4all-core hvp_public_api`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/ad.rs crates/tensor4all-core/tests/hvp_public_api.rs
git commit -m "feat(core): add dedicated public hvp api"
```

### Task 6: Update Downstream Usage and Regenerate API Docs

**Files:**
- Modify: `crates/tensor4all-itensorlike/tests/tensortrain_native_ad.rs`
- Modify: any compile-failing downstream tensor/scalar call sites found by `cargo test`
- Modify: `docs/api/tensor4all_core.md`
- Modify: `docs/api/tensor4all_tensorbackend.md`

**Step 1: Update downstream tests and call sites**

Replace old assumptions about direct native payload access with the new `Scalar`, `GradTape`, and
public tensor API where appropriate.

**Step 2: Regenerate API docs**

Run:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Expected: regenerated docs show public `Scalar`, hidden native accessors, and updated AD/HVP APIs.

**Step 3: Run targeted downstream tests**

Run:

```bash
cargo nextest run --release -p tensor4all-itensorlike tensortrain_native_ad
```

Expected: PASS

**Step 4: Commit**

```bash
git add crates/tensor4all-itensorlike/tests/tensortrain_native_ad.rs docs/api/tensor4all_core.md docs/api/tensor4all_tensorbackend.md
git commit -m "test: update downstream ad coverage for scalar tensor split"
```

### Task 7: Full Verification

**Files:**
- Modify: none expected
- Test: workspace verification commands

**Step 1: Format**

Run: `cargo fmt --all`
Expected: no diff after formatting pass.

**Step 2: Lint**

Run: `cargo clippy --workspace`
Expected: PASS

**Step 3: Run focused workspace tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core
cargo nextest run --release -p tensor4all-itensorlike
```

Expected: PASS

**Step 4: Run full workspace tests if focused suites pass**

Run: `cargo nextest run --release --workspace`
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: align tensor4all scalar api with tenferro tensor-native model"
```
