# Review Refresh: EagerTensor Migration Design v2

Date: 2026-04-13
Reviewer: Codex CLI
Design doc: `docs/design/symbolic-shape-and-tenferro-migration.md`

---

## Resolved Upstream Blockers

The original v2 review identified several upstream tenferro gaps. Those are no
longer blockers if tensor4all-rs re-pins to the tenferro-rs revision merged on
2026-04-13 (or any later descendant):

1. `EagerTensor::tracks_grad()` now exists as a public query.
2. `EagerTensor::clear_grad()` and `EagerContext::clear_grads()` now exist as
   public reset APIs.
3. `EagerTensor::backward()` now uses PyTorch-style accumulation semantics
   instead of implicitly clearing gradients first.
4. `EagerTensor<CpuBackend>` is now Send+Sync-safe via Arc/Mutex internals.

These upstream changes should be treated as required migration baseline, not as
optional follow-up work.

---

## High Priority

### 1. tensor4all-rs should align its public AD names with the final upstream API

The rewritten design should use:

- `enable_grad(self) -> Self`
- `tracks_grad(&self) -> bool`
- `grad(&self) -> Result<Option<TensorDynLen>>`
- `clear_grad(&self) -> Result<()>`
- `backward(&self) -> Result<()>`
- `detach(&self) -> Self`

It should *not* preserve or reintroduce:

- `requires_grad(&self) -> bool`
- `set_requires_grad(...)`
- `zero_grad(...)`
- `backward_with_seed(...)`
- `has_grad(...)`

`has_grad` is especially problematic because it sounds like "a gradient value is
present now," while the upstream API's `tracks_grad` means "this tensor
participates in gradient tracking."

This naming alignment should apply to both `TensorDynLen` and the eventual
core-owned `tensor4all_core::AnyScalar` wrapper. The backend-layer
`tensor4all_tensorbackend::AnyScalar` should remain primal-only and must not be
treated as the public compatibility layer once AD is reintroduced.

### 2. Actual tensor4all-rs code still uses the legacy AD surface and re-exports the wrong AnyScalar

Current uses that the implementation plan must remove or rewrite:

| File | Legacy API usage |
|------|------------------|
| `crates/tensor4all-core/src/lib.rs` | re-exports backend `AnyScalar`; plan should replace this with a core-owned rank-0 `TensorDynLen` wrapper |
| `crates/tensor4all-core/src/defaults/tensordynlen.rs` | `requires_grad`, `set_requires_grad`, `zero_grad`, `backward_with_seed` |
| `crates/tensor4all-tensorbackend/src/any_scalar.rs` | `requires_grad`, `set_requires_grad`, `zero_grad`, `backward_with_seed`, `grad`, `detach` |
| `crates/tensor4all-core/tests/tensor_native_ad.rs` | legacy tensor-level AD API names |
| `crates/tensor4all-treetn/tests/ad_treetn.rs` | `set_requires_grad` expectations on migrated tensors |
| `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs` | `tensor.requires_grad()` assumptions on legacy native tensors |

The old review text mentioned `with_requires_grad(...)`, but the actual
tensor4all-rs wrapper method to remove is `set_requires_grad(...)`.

### 3. The plan must treat accumulation and explicit clearing as first-class behavior

Now that upstream eager AD accumulates gradients by default, tensor4all-rs
should add explicit tests and docs for:

- repeated `backward()` accumulation
- `clear_grad()` resetting one tracked tensor
- `grad()` returning the accumulated gradient
- `tracks_grad()` reporting tracking state, not gradient presence

If these behaviors are not exercised directly, users are likely to infer the
old overwrite-on-backward semantics.

### 4. The public AnyScalar contract should be explicit

The design should document the following instead of leaving scalar behavior
implicit:

- `tensor4all_core::AnyScalar` is a core-owned wrapper over rank-0
  `TensorDynLen`
- named-value arithmetic in docs/tests uses the borrow-based contract
  `&x * &y + &x`
- function APIs take `&AnyScalar`
- `Clone` is a shared-handle clone, not detach or deep copy
- dropping a handle is not gradient clearing, and `close()` should not be a
  value-producing public API

Without this, the migration leaves the most user-visible scalar semantics
underspecified.

### 5. The context wording must stay honest

`TypedTensor` operations use a mutable `CpuBackend`, while eager graphs own
their backend through `EagerContext<CpuBackend>`. The design should therefore
describe these as paired thread-local execution objects, not as one shared
backend instance, unless the implementation can actually prove single-instance
sharing.

---

## Medium Priority

### 6. Memory order risk remains unchanged and still needs an explicit audit

`TypedTensor::from_vec(...)` stores data as-is. The design and plan correctly
need an audited migration rather than a blind rename. The highest-risk call
sites remain:

- `crates/tensor4all-tcicore/src/matrix.rs`
- `crates/tensor4all-simplett/src/einsum_helper.rs`
- `crates/tensor4all-simplett/src/contraction.rs`
- `crates/tensor4all-simplett/src/compression.rs`
- `crates/tensor4all-simplett/src/vidal.rs`
- `crates/tensor4all-simplett/src/mpo/factorize.rs`
- `crates/tensor4all-simplett/src/tensor.rs`

### 7. Verification guidance should grep for the APIs that actually exist today

The migration sweep should include:

```bash
rg -n "ScalarType|MemoryOrder|contiguous\\(|buffer\\(|from_row_major_slice|from_slice\\(|set_default_runtime|RuntimeContext::Cpu|print_and_reset_contract_profile|requires_grad\\(|set_requires_grad|zero_grad\\(|backward_with_seed" crates
```

The earlier grep pattern missed `set_requires_grad`, which is the real wrapper
API still present in tensor4all-rs.

### 8. Send + Sync assertions are now meaningful and should be added

Because upstream eager tensors are now Send+Sync, the plan should add static
assertions for:

- `tenferro::EagerTensor<tenferro::CpuBackend>`
- `tensor4all_core::defaults::TensorDynLen`

This is especially relevant for FFI-facing tensor handles.

---

## API Spot Checks (Verified Against Current tenferro-rs)

| Symbol | Status |
|--------|--------|
| `Tensor::new(shape, data)` | OK |
| `tensor.dtype()` / `DType{F32,F64,C32,C64}` | OK |
| `tensor.as_slice::<T>() -> Option<&[T]>` | OK |
| `tensor.shape() -> &[usize]` | OK |
| `TypedTensor::from_vec(shape, data)` | OK |
| `TypedTensor::svd(&self, ctx: &mut CpuBackend)` | OK |
| `TypedTensor::qr(&self, ctx: &mut CpuBackend)` | OK |
| `CpuContext::with_threads(n)` | OK |
| `EagerContext::with_backend()` -> `Arc<Self>` | OK |
| `EagerTensor::from_tensor_in(tensor, Arc<EagerContext<B>>)` | OK |
| `EagerTensor::requires_grad_in(tensor, ctx)` | OK |
| `EagerTensor::tracks_grad()` | OK |
| `EagerTensor::clear_grad()` | OK |
| `EagerContext::clear_grads()` | OK |
| `EagerTensor::backward()` accumulates | OK |

### Path / signature corrections

| Design assumption | Actual API |
|------------------|-----------|
| `eager_einsum_ad` is at crate root | `tenferro::eager_einsum::eager_einsum_ad` |
| `typed_eager_einsum` takes `&mut CpuBackend` specifically | it accepts `&mut impl TensorBackend` |

---

## Scope Guardrails

The rewritten design correctly keeps the following in scope and they should not
be dropped during implementation:

- `crates/tensor4all-core/src/defaults/svd.rs`
- `crates/tensor4all-core/src/defaults/qr.rs`
- `crates/tensor4all-simplett/src/tensor.rs`
- `crates/tensor4all-simplett/src/types.rs`
- `crates/tensor4all-simplett/src/contraction.rs`
- `crates/tensor4all-tensorbackend/tests/bench_einsum_native.rs`
- `crates/tensor4all-treetn/tests/ad_treetn.rs`
