# Core-First Carrier Hard Cut Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `AdTensor<T>` and `DynAdTensor` from `tenferro-rs` by switching directly to one dynamic edge-aware core carrier, then deleting the remaining legacy tape compatibility island.

**Architecture:** This plan intentionally replaces the slower `AdTensor<T> -> DynAdTensor -> Tensor` widening strategy. Introduce the final dynamic carrier in `tenferro-internal-ad-core` first, make `internal-ad-surface::Tensor` wrap that carrier directly, then migrate `internal-ad-ops` and `internal-ad-linalg` to typed borrows over the dynamic carrier. Only after all reverse-capable ops are edge-native do we delete `legacy_attachment`, `reverse_tape()`, `register_*rule`, `AdTensor<T>`, and `DynAdTensor`.

**Tech Stack:** Rust, `tidu-rs` edge-based reverse graph (`Value`, `Op`), `tenferro-rs` internal crates, `cargo fmt`, `cargo clippy`, `cargo nextest --release`

**Supersedes:** `docs/plans/2026-03-30-adtensor-cutover-refined-plan.md` for implementation order, and narrows the transitional strategy described in `docs/plans/2026-03-30-adtensor-carrier-replacement-design.md`.

---

### Task 1: Unblock the upstream dependency before more tenferro work

**Files:**
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/Cargo.toml`
- Verify: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd`

**Step 1: Wait for or record the merged `tidu-rs` commit**

Source PR:

```text
https://github.com/tensor4all/tidu-rs/pull/10
```

**Step 2: Replace the local path dependency in `tenferro-rs`**

Change:

```toml
tidu = { path = "../../../tidu-rs/.worktrees/edge-based-autograd/crates/tidu" }
```

to:

```toml
tidu = { git = "https://github.com/tensor4all/tidu-rs", rev = "<merged-commit>" }
```

**Step 3: Verify**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo check -p tenferro --tests --release
```

Expected: PASS.

**Step 4: Commit**

```bash
git add /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/Cargo.toml
git commit -m "build: pin tidu edge-based reverse dependency"
```

### Task 2: Introduce the final dynamic core carrier before touching more surface code

**Files:**
- Create: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/dyn_autograd_tensor.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/lib.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tensor.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tensor/reverse.rs`
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tests/core_value_reverse_api.rs`
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tests/dyn_autograd_tensor.rs`

**Step 1: Write the failing core tests**

Cover:
- dynamic primal creation
- dynamic forward dual creation
- reverse leaf creation without `AdTensor<T>`
- reverse output creation from `tidu::Value`
- leaf grad accumulation and `zero_grad`

**Step 2: Add the new carrier**

Create a core-owned type, conceptually:

```rust
pub struct DynAutogradTensor {
    primal: DynTensor,
    tangent: Option<DynTensor>,
    reverse: Option<ReverseState>,
    flags: AutogradFlags,
}
```

**Step 3: Keep temporary compatibility constructors inside core only**

Allow:
- `From<AdTensor<T>> for DynAutogradTensor`
- `TryFrom<DynAutogradTensor> for AdTensor<T>`

Do not expose these conversions above `tenferro-internal-ad-core`.

**Step 4: Add typed borrow helpers**

Add a typed borrow/view API that can provide:
- `primal_typed::<T>()`
- `structured_primal_typed::<T>()`
- `tangent_typed::<T>()`
- `reverse_edge_value_typed::<T>()`

without reconstructing `&AdTensor<T>`.

**Step 5: Verify**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo test -p tenferro-internal-ad-core --release dyn_autograd_tensor
cargo test -p tenferro-internal-ad-core --release reverse_non_leaf_allows_explicit_input_gradient_cache
```

Expected: PASS.

**Step 6: Commit**

```bash
git add \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/dyn_autograd_tensor.rs \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/lib.rs \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tensor.rs \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tensor/reverse.rs \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tests/core_value_reverse_api.rs \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tests/dyn_autograd_tensor.rs
git commit -m "refactor: add dynamic core autograd carrier"
```

### Task 3: Cut the public and surface tensor wrapper directly onto the new carrier

**Files:**
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/mod.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/accessors.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/basics.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/downcast.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/snapshot.rs`
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/tests/integration/public_surface_tests.rs`
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/tests/integration/core_value_surface_regressions.rs`

**Step 1: Change `Tensor` to wrap the new dynamic carrier**

Replace:

```rust
pub struct Tensor(pub(crate) DynAdTensor);
```

with a wrapper over `DynAutogradTensor` (or the final core carrier name).

**Step 2: Rewrite `TypedTensorRef`**

`TypedTensorRef` must borrow from the dynamic carrier and expose:
- `primal()`
- `structured_primal()`
- `tangent()`
- `structured_tangent()`
- `requires_grad()`
- `is_leaf()`
- `reverse_edge_value()`

without yielding `&AdTensor<T>`.

**Step 3: Delete surface helpers that only existed for `DynAdTensor`**

Remove or replace:
- `as_dyn_ad_ref()`
- `From<AdTensor<T>> for Tensor`
- `From<DynAdTensor> for Tensor`

unless a temporary adapter is required strictly inside one module.

**Step 4: Update public tests**

Rewrite integration tests to use:
- `Tensor`
- typed borrow helpers
- public reverse APIs

and stop matching on `Tensor::F64(...)` / `DynAdTensor`.

**Step 5: Verify**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo test -p tenferro --test integration --release public_surface_tests
cargo test -p tenferro --test integration --release core_value_surface_regressions
```

Expected: PASS.

### Task 4: Make ops and linalg crate boundaries dynamic-first, not `AdTensor<T>`-first

**Files:**
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-ops/src/ops/ad/mod.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-ops/src/ops/ad/scalar_eager.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-ops/src/ops/einsum/ad.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-ops/src/ops/reduction/ad.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-linalg/src/ops/linalg/results.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/eager_impl.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/svd_qr_impl.rs`
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-ops/tests/dyn_entrypoints.rs`
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-linalg/tests/eager_dyn_extra.rs`

**Step 1: Introduce one dynamic borrow abstraction at crate boundaries**

Examples:
- `DynAutogradTensorRef<'a>`
- `TypedTensorBorrow<'a, T>`

Use that abstraction for cross-crate entrypoints instead of `&AdTensor<T>` or `DynAdTensorRef`.

**Step 2: Convert result structs to dynamic carriers**

Examples:
- `SvdResult<TensorLike>`-style dynamic result types
- no `type DynSvdResult = SvdResult<DynAdTensor>`

Mixed-dtype outputs must remain natural.

**Step 3: Keep typed kernels private**

Typed functions may still exist after runtime dispatch, but they must not be the canonical cross-crate API.

**Step 4: Verify**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo test -p tenferro-internal-ad-ops --release dyn_entrypoints
cargo test -p tenferro-internal-ad-linalg --release eager_dyn_extra
cargo check -p tenferro --tests --release
```

Expected: PASS.

### Task 5: Make the edge-based path canonical and isolate the legacy tape path

**Files:**
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/registry.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/ops.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tensor.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tensor/reverse.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/autograd_api.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/pullback.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/src/tape/mod.rs`

**Step 1: Delete the `can_use_edge_*` split where edge behavior is already equivalent**

Start with:
- scalar eager ops
- reduction
- einsum

and remove legacy fallback branches one family at a time.

**Step 2: Push old rule registration behind one compatibility module**

If any op still needs old rule registration temporarily, isolate it in one module.
Do not let `register_rule`, `register_closure_rule`, or `register_mixed_rule` remain scattered across ops/linalg/surface.

**Step 3: Remove `legacy_attachment` reads from the public reverse API**

Delete:
- `reverse_tape()`
- `reverse_node_id()`
- surface logic that requires tape identity checks before backward

**Step 4: Verify the old tape island is empty**

Run:

```bash
rg -n "register_(closure_)?rule|register_mixed_rule|legacy_attachment|reverse_tape\\(|reverse_node_id\\(" \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro \
  -g '!target'
```

Expected: zero production hits, or one deliberately named compatibility module that is about to be deleted in Task 6.

**Step 5: Verify**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo test -p tenferro --test integration --release public_surface_tests
cargo test -p tenferro --test integration --release hvp_tests
cargo test -p tenferro --test integration --release dynamic_wrapper_coverage_tests
```

Expected: PASS.

### Task 6: Delete `AdTensor<T>`, `DynAdTensor`, and the old tape surface

**Files:**
- Delete: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/dyn_ad_tensor.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/lib.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/lib.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/src/lib.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/README.md`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/docs/design/autodiff.md`
- Test: all integration and focused internal tests that still name `AdTensor` or `DynAdTensor`

**Step 1: Delete the old types and re-export paths**

Remove:
- `AdTensor<T>`
- `DynAdTensor`
- `DynAdTensorRef`
- any `pub use` that keeps them reachable

**Step 2: Rewrite tests and docs**

Stop naming the removed carrier types in:
- integration tests
- rustdoc examples
- design docs
- README status notes

**Step 3: Re-run the ownership graph query**

Run:

```bash
rg -n "\\bDynAdTensor\\b|\\bAdTensor\\s*<|\\bAdTensor\\b" \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api \
  -g '!target' -g '!Cargo.lock'
```

Expected: only historical design docs mention them, or zero hits if docs are updated too.

**Step 4: Final verification**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
```

Expected: PASS.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove legacy ad tensor carriers"
```

### Task 7: Update downstream integration after the hard cut

**Files:**
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-core/tests/tensor_native_ad.rs`
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-treetn/tests/ad_treetn.rs`
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-29-tidu-tenferro-cutover-design.md`

**Step 1: Replace any remaining assumptions about tape identity or typed carrier access**

Downstream tests must treat `tenferro::Tensor` as the only tensor handle.

**Step 2: Re-run patched downstream suites**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo --config /tmp/tensor4all-local-patches.toml test -p tensor4all-core --test tensor_native_ad --release
cargo --config /tmp/tensor4all-local-patches.toml test -p tensor4all-treetn --test ad_treetn --release
```

Expected: PASS.

**Step 3: Commit**

```bash
git add \
  /home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-core/tests/tensor_native_ad.rs \
  /home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-treetn/tests/ad_treetn.rs \
  /home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-29-tidu-tenferro-cutover-design.md
git commit -m "test: align downstream suites with core-first ad cutover"
```
