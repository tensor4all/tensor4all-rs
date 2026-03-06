# Tenferro Native Slice 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve native `DynAdTensor` payloads inside `TensorDynLen` so scalar reductions and inner products can keep AD metadata while maintaining `treetn`-level behavior.

**Architecture:** Keep the current public `TensorDynLen` API and `Storage` snapshot compatibility, but add an internal structured `DynAdTensor` payload. Route scalar reductions and core algebra through native payloads when available, and fall back to legacy storage paths only for operations that still lack a native bridge.

**Tech Stack:** Rust, tensor4all-core, tensor4all-tensorbackend, tenferro-dyadtensor partial-diagonal API, cargo-nextest (`--release`)

---

### Task 1: Add structured native bridge helpers

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Test: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`

**Step 1: Write failing tests**

Cover:
- structured diag storage round-trip preserves `Diag` storage
- forward-mode native tensor `sum` returns forward-mode `DynAdScalar`
- unary native `permute` and binary native `einsum` helpers preserve primal parity

**Step 2: Run focused tests and verify failure**

Run: `cargo nextest run --release -p tensor4all-tensorbackend tenferro_bridge`

**Step 3: Implement minimal helpers**

Add helpers for:
- structured `Storage <-> DynAdTensor` conversion
- rank-0 / single-element `DynAdTensor -> DynAdScalar` conversion
- native unary/binary einsum wrappers for `permute`, `contract`, `outer_product`, `sum`
- complex promotion helper for real/complex mixed native contraction

**Step 4: Re-run focused tests**

Run: `cargo nextest run --release -p tensor4all-tensorbackend tenferro_bridge`

**Step 5: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/lib.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs
git commit -m "feat(tensorbackend): add structured native tensor bridge helpers"
```

### Task 2: Add native payload to TensorDynLen

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Step 1: Write failing tests**

Cover:
- forward-mode native tensor `sum` preserves tangent
- forward-mode scalar tensor `only` preserves tangent
- forward-mode `inner_product` preserves tangent

**Step 2: Run focused tests and verify failure**

Run: `cargo nextest run --release -p tensor4all-core tensordynlen`

**Step 3: Implement native payload plumbing**

Add:
- internal `Option<DynAdTensor>` payload on `TensorDynLen`
- constructors that seed native payload from simple storage
- internal constructor from native payload with synchronized primal snapshot
- native-aware `sum`, `only`, `conj`, `scale`, `axpby`, `permute`, `contract`, `outer_product`, `inner_product`
- keep existing `storage()` / `materialize_storage()` compatibility by using the primal snapshot

**Step 4: Re-run focused tests**

Run: `cargo nextest run --release -p tensor4all-core tensordynlen`

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs
git commit -m "feat(core): preserve native ad payloads in TensorDynLen"
```

### Task 3: Verify treetn compatibility gate

**Files:**
- Modify only if regressions appear

**Step 1: Run focused verification**

Run:
```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release -p tensor4all-tensorbackend -p tensor4all-core -p tensor4all-treetn
```

**Step 2: Fix regressions if any**

Apply minimal fixes only.

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add native payload-preserving TensorDynLen reductions"
```
