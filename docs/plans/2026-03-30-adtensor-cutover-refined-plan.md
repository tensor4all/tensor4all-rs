# AdTensor Cutover Refined Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining public and cross-crate `AdTensor<T>` coupling on the path to the torch-like `Tensor + edge-based reverse metadata` model, while preserving the already-fixed helper-removal regressions.

**Architecture:** Treat the already-landed `ensure_common_reverse_tape(...)` removal as the new baseline. The public tape-helper leak is no longer the blocker. First, shorten and harden crate boundaries so `AdTensor<T>` is owned only by `tenferro-internal-ad-core` and not re-exported casually through higher crates. Then stop at an explicit design gate before attempting the deeper carrier replacement from `AdTensor<T>` to `Tensor + AutogradMeta`, because that second step changes the internal value model rather than only visibility or plumbing.

**Current Findings:**

- proven non-blockers:
  - helper-free reverse joins across independently created leaves now work
  - explicit backward inputs on non-leaf tensors now work
  - the QR regression was a test-input rank issue, not a carrier-model issue
- completed boundary cleanup:
  - `tenferro` now sources `AdTensor` / `NodeId` from `tenferro-internal-ad-core`
  - `tenferro-internal-ad-surface` reduced `AdTensor` / `NodeId` / `AdTensorSnapshot` to crate-private re-exports
  - `tenferro-internal-ad-ops` and `tenferro-internal-ad-linalg` no longer re-export `AdTensor` from their crate roots
- remaining real blocker:
  - `internal-ad-ops`, `internal-ad-linalg`, and `internal-ad-surface` still use `AdTensor<T>` as the typed carrier in their internal AD code
  - `internal-ad-linalg` typed results and eager entrypoints still expose `AdTensor<T>` directly, so deletion now requires carrier replacement rather than more visibility cleanup

The chosen carrier-replacement staging is documented in:

- `/home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-30-adtensor-carrier-replacement-design.md`

**Tech Stack:** Rust, `tidu-rs`, `tenferro-rs`, `tensor4all-rs`, `cargo fmt`, `cargo test --release`, `cargo check --release`

---

### Task 1: Freeze the current passing baseline

**Files:**
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/tests/integration/public_surface_tests.rs`
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tests/core_value_reverse_api.rs`
- Test: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-core/tests/tensor_native_ad.rs`
- Test: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-treetn/tests/ad_treetn.rs`

**Step 1: Re-run the upstream tenferro public regression suite**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo test -p tenferro --test integration --release public_surface_tests
```

Expected: PASS, including the helper-free reverse join tests for `einsum`, `qr`, and `svd`.

**Step 2: Re-run the explicit non-leaf input-gradient regression**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo test -p tenferro-internal-ad-core --release reverse_non_leaf_allows_explicit_input_gradient_cache
```

Expected: PASS.

**Step 3: Re-run the patched downstream sequential regressions**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo --config /tmp/tensor4all-local-patches.toml test -p tensor4all-core --test tensor_native_ad --release reverse_ad_sequential_
```

Expected: PASS.

**Step 4: Re-run the patched TreeTN regressions**

Run:

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo --config /tmp/tensor4all-local-patches.toml test -p tensor4all-treetn --test ad_treetn --release backward_ad_truncate_propagates_gradients_f64
cargo --config /tmp/tensor4all-local-patches.toml test -p tensor4all-treetn --test ad_treetn --release backward_ad_swap_site_indices_propagates_gradients_f64
```

Expected: PASS.

**Step 5: Commit the baseline stabilization if needed**

```bash
git add \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tensor.rs \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tests/core_value_reverse_api.rs \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/tests/integration/public_surface_tests.rs \
  /home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-core/src/tensor_like.rs \
  /home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-core/src/defaults/tensordynlen.rs \
  /home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-treetn/src/treetn/mod.rs \
  /home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-treetn/src/treetn/localupdate.rs
git commit -m "fix: remove public tape helper regressions"
```

### Task 2: Shorten the crate boundary so `tenferro` can stop depending on `AdTensor` through `internal-ad-surface`

**Files:**
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/Cargo.toml`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/src/core/value/mod.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/src/core/mod.rs`
- Verify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/lib.rs`

**Step 1: Add a normal dependency from `tenferro` to `tenferro-internal-ad-core`**

Modify `tenferro/Cargo.toml` so `tenferro` can import `AdTensor` and `NodeId` from the owning crate at normal build time, not only in `dev-dependencies`.

**Step 2: Switch `tenferro/src/core/value/mod.rs` to import `AdTensor` and `NodeId` from `tenferro-internal-ad-core`**

Expected: `tenferro` no longer needs `internal-ad-surface` as a transitive source of truth for the typed carrier.

**Step 3: Keep `AdMode` sourced from the public-facing internal surface**

Reason: `AdMode` is still part of the transitional high-level state API and does not itself force typed-carrier coupling.

**Step 4: Verify the crate graph compiles**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo check -p tenferro --tests --release
```

Expected: PASS.

**Step 5: Commit the dependency-boundary slice**

```bash
git add \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/Cargo.toml \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/src/core/value/mod.rs \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/src/core/mod.rs
git commit -m "refactor: source typed ad state from internal ad core"
```

### Task 3: Remove casual `AdTensor` re-exports from higher internal crates

**Files:**
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/lib.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/mod.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-ops/src/lib.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-linalg/src/lib.rs`

**Step 1: Inventory which of these re-exports are actually consumed**

Run:

```bash
rg -n "tenferro_internal_ad_surface::.*AdTensor|tenferro_internal_ad_ops::.*AdTensor|tenferro_internal_ad_linalg::.*AdTensor" \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api -g '!target'
```

Expected: only direct internal consumers remain.

**Step 2: Remove one redundant re-export at a time**

After each removal, run `cargo check -p tenferro --tests --release`.

**Step 3: Stop immediately when a removal reveals a real cross-crate dependency**

Do not paper over missing imports with new re-exports. Either:
- retarget the consumer to `tenferro-internal-ad-core`, or
- document the dependency as a blocker for the next design slice.

**Step 4: Leave `AdTensor` exported only from the owning internal crate**

Target end state for this task:
- `tenferro-internal-ad-core` owns and exports `AdTensor`
- higher internal crates may use it internally, but do not re-export it as part of their top-level convenience API

**Step 5: Verify**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo check -p tenferro --tests --release
cargo test -p tenferro --test integration --release public_surface_tests
```

Expected: PASS.

### Task 4: Add an explicit design gate for the real carrier replacement

**Files:**
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-29-tenferro-edge-based-reverse-design.md`
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-29-tidu-tenferro-cutover-design.md`
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-30-adtensor-cutover-refined-plan.md`

**Step 1: Record the structural blocker explicitly**

Document that `AdTensor<T>` is still the typed carrier used by:
- `tenferro-internal-ad-ops`
- `tenferro-internal-ad-linalg`
- `tenferro-internal-ad-surface` dynamic wrappers

and therefore cannot be removed by visibility cleanup alone.

**Step 2: Record the proven non-blockers explicitly**

Document that:
- helper-free reverse graph joins already work
- non-leaf explicit backward inputs already work
- the recent QR regression was a test-input issue, not a value-model blocker

**Step 3: Define the next design question precisely**

The next design note must decide one transitional carrier strategy:
- `AdTensor<T>` survives only inside `tenferro-internal-ad-core` while higher crates switch to `Tensor + edge metadata`, or
- `AdTensor<T>` is replaced in place by a new typed internal carrier and then erased from higher crates

Do not start broad code churn before this is chosen.

Status update: this stop condition has already triggered. The transitional
carrier strategy is now chosen in
`docs/plans/2026-03-30-adtensor-carrier-replacement-design.md`:

- Stage A: move `internal-ad-surface::Tensor` to an erased carrier
- Stage B: make higher crate-boundary APIs erased-only
- Stage C: replace the core `AdTensor<T>` carrier
- Stage D: delete transitional shims

The next code batch should start from Stage A rather than directly attempting
carrier deletion.

**Step 4: Commit the refined plan/docs slice**

```bash
git add \
  /home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-29-tenferro-edge-based-reverse-design.md \
  /home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-29-tidu-tenferro-cutover-design.md \
  /home/shinaoka/tensor4all/tensor4all-rs/docs/plans/2026-03-30-adtensor-cutover-refined-plan.md
git commit -m "docs: refine AdTensor cutover plan"
```

### Task 5: Decision checkpoint before deeper code churn

**Step 1: Re-read the remaining `AdTensor<T>` ownership graph**

Run:

```bash
rg -n "\\bAdTensor<|\\bAdTensor\\b" \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal \
  -g '!target'
```

**Step 2: If `AdTensor<T>` still appears as the typed API of `internal-ad-ops` / `internal-ad-linalg`, stop and write the carrier-replacement design note before implementation continues**

Expected: this stop condition is likely to trigger.

**Step 3: Only continue directly into code if the remaining uses are local implementation details inside `tenferro-internal-ad-core`**

That is the minimum acceptable condition for starting actual deletion work.

### Task 6: Final verification for this refined-plan batch

**Step 1: Run formatting**

```bash
cargo fmt --all
```

**Step 2: Run upstream verification**

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo check -p tenferro --tests --release
cargo test -p tenferro --test integration --release public_surface_tests
cargo test -p tenferro-internal-ad-core --release reverse_non_leaf_allows_explicit_input_gradient_cache
```

**Step 3: Run patched downstream verification**

```bash
cd /home/shinaoka/tensor4all/tensor4all-rs
cargo --config /tmp/tensor4all-local-patches.toml test -p tensor4all-core --test tensor_native_ad --release reverse_ad_sequential_
cargo --config /tmp/tensor4all-local-patches.toml test -p tensor4all-treetn --test ad_treetn --release backward_ad_truncate_propagates_gradients_f64
cargo --config /tmp/tensor4all-local-patches.toml test -p tensor4all-treetn --test ad_treetn --release backward_ad_swap_site_indices_propagates_gradients_f64
```

Expected: all commands pass.
