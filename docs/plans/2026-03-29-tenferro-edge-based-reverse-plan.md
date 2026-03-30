# Edge-Based Reverse AD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current single-tape reverse execution path with one edge-based reverse graph model so `tenferro` no longer exposes `ensure_common_reverse_tape(...)`, downstream sequential multi-leaf AD flows work without tape normalization, and `tidu` no longer exposes a separate `expert::Tape` reverse API.

**Architecture:** Unify `tidu` public reverse-mode APIs around edge-based `Value` / `Op` and remove the public tape-based reverse substrate. `tidu` owns edge-based reverse nodes and traversal. `tenferro` stores only public AD header state plus an optional reverse edge handle. Downstream `tensor4all-rs` remains a consumer and removes workaround calls after upstream behavior is fixed.

**Tech Stack:** Rust, `tidu-rs`, `tenferro-rs`, `tensor4all-rs`, `cargo nextest`, `cargo test`

---

### Task 1: Lock the regression and remove new public-helper dependency

**Files:**
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-core/tests/tensor_native_ad.rs`
- Test: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/tests/integration/public_surface_tests.rs`

**Step 1: Add or refine an upstream tenferro regression that combines independently-created reverse leaves without calling any tape-sharing helper**

Expected behavior: the regression fails on the current upstream implementation for the same root reason as the downstream QR/SVD absorption tests.

**Step 2: Remove the temporary downstream call to `NativeTensor::ensure_common_reverse_tape(...)`**

Expected behavior: downstream code no longer depends on the helper even temporarily.

**Step 3: Run the targeted downstream regression**

Run:

```bash
CARGO_TARGET_DIR=/tmp/t4a-target-patched cargo \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-algebra.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-algebra"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-device.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-device"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-einsum.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-einsum"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-prims.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-prims"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-tensor.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-tensor"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-linalg.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-linalg"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-tensor-compute.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-tensor-compute"' \
  test -p tensor4all-core reverse_ad_sequential_ --release
```

Expected: failure still reproduces, but no downstream workaround remains.

### Task 2: Introduce edge-based reverse primitives in tidu and retire public tape-based reverse APIs

**Files:**
- Modify: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/value.rs`
- Modify: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/engine/mod.rs`
- Modify: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/engine/context.rs`
- Modify: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/engine/rule.rs`
- Modify: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/function.rs`
- Modify: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/lib.rs`
- Modify or delete: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/expert.rs`
- Modify or delete: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/engine/tape.rs`
- Modify or delete: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/src/engine/tracked.rs`
- Test: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/tests/op_api_tests.rs`
- Test: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/tests/function_api_tests.rs`
- Test: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd/crates/tidu/tests/value_api_tests.rs`

**Step 1: Add internal reverse-edge types and node ownership model**

Add minimal internal structs for reverse edges, node outputs, and leaf gradient sinks without changing the public `Op` trait.

**Step 2: Teach `Value<V>` to carry an edge-based reverse handle instead of relying on one shared tape identity**

Keep the public `Value` API stable.

**Step 3: Add `GraphTask` build/execution state with linear-time invariants**

Requirements:

- each reachable node is visited once during task construction
- each reachable edge is processed once to compute dependency counts
- hot-path execution uses task-local dense vectors rather than repeated graph scans

**Step 4: Replace the `common_tape(inputs)` rejection path in `Op::apply(...)`**

Expected behavior: mixed previously-independent reverse inputs can participate in one new op call.

**Step 5: Remove public `expert::Tape` / `TrackedValue` reverse entrypoints**

Expected behavior: one public reverse AD model remains.

**Step 6: Add focused tidu tests**

Cover:
- combining two independent reverse leaves in one op
- combining outputs from previously separate computations in a later op
- public docs/examples no longer rely on `expert::Tape`
- no quadratic behavior hidden behind repeated graph normalization

**Step 7: Run targeted tidu verification**

Run:

```bash
cd /home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd
cargo nextest run --release -p tidu
cargo test -p tidu --doc --release
```

Expected: all tidu tests pass.

### Task 3: Rewire tenferro reverse metadata to use edges

**Files:**
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-core/src/tensor.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/basics.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/pullback.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/merge.rs`

**Step 1: Replace tape-specific reverse attachment with edge-based metadata**

Leaf tensors should no longer require `ensure_reverse_leaf_on(&Tape<_>)`.

**Step 2: Preserve user-visible leaf/non-leaf and grad-cache semantics**

Keep `grad`, `retain_grad`, `zero_grad`, and `is_leaf` behavior unchanged.

**Step 3: Internalize or delete tape-sharing helpers**

`ensure_common_reverse_tape_impl(...)` should disappear or become unreachable by public surface code.

**Step 4: Run targeted tenferro verification**

Run:

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo nextest run --release -p tenferro
cargo test -p tenferro --doc --release
```

Expected: all tenferro tests pass.

### Task 4: Remove the public helper and align public tests

**Files:**
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/internal/tenferro-internal-ad-surface/src/core/dynamic/dyn_ad_tensor/basics.rs`
- Modify: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro/tests/integration/public_surface_tests.rs`
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Step 1: Delete `Tensor::ensure_common_reverse_tape(...)` from the public API**

No public AD helper should mention tape identity.

**Step 2: Remove now-dead compatibility code in downstream wrappers**

Keep only API-following compatibility required by the cutover, not tape workarounds.

**Step 3: Run the focused downstream regression**

Use the same patched command from Task 1.

Expected: the sequential QR/SVD regression tests pass without helper calls.

### Task 5: Full verification and cleanup

**Files:**
- Modify: `docs/plans/2026-03-29-tidu-tenferro-cutover-design.md`
- Modify: `docs/plans/2026-03-29-tenferro-edge-based-reverse-design.md`
- Modify: `docs/plans/2026-03-29-tenferro-edge-based-reverse-plan.md`

**Step 1: Update notes to reflect implemented behavior**

Remove transitional language once the helper is gone.

**Step 2: Run formatting**

Run:

```bash
cargo fmt --all
```

**Step 3: Run final targeted verification**

Run:

```bash
cd /home/shinaoka/tensor4all/tidu-rs/.worktrees/edge-based-autograd
cargo nextest run --release -p tidu

cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api
cargo nextest run --release -p tenferro
cargo test -p tenferro --doc --release

cd /home/shinaoka/tensor4all/tensor4all-rs
CARGO_TARGET_DIR=/tmp/t4a-target-patched cargo \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-algebra.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-algebra"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-device.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-device"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-einsum.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-einsum"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-prims.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-prims"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-tensor.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-tensor"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-linalg.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-linalg"' \
  --config 'patch."https://github.com/tensor4all/tenferro-rs".tenferro-tensor-compute.path="/home/shinaoka/tensor4all/tenferro-rs/.worktrees/redesign-tidu-edge-api/tenferro-tensor-compute"' \
  test -p tensor4all-core reverse_ad_sequential_ --release
```

Expected: all commands pass.

### Task 6: Commit in intentional slices

**Step 1: Commit the design docs**

```bash
git add docs/plans/2026-03-29-tenferro-edge-based-reverse-design.md docs/plans/2026-03-29-tenferro-edge-based-reverse-plan.md docs/plans/2026-03-29-tidu-tenferro-cutover-design.md
git commit -m "docs: add edge-based reverse design and plan"
```

**Step 2: Commit `tidu-rs` unified edge-based engine slice**

**Step 3: Commit `tenferro-rs` edge-based reverse metadata slice**

**Step 4: Commit downstream cleanup and regression verification slice**
