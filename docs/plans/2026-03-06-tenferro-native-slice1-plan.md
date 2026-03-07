# Tenferro Native Slice 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route `TensorDynLen` core tensor algebra through tenferro native execution while preserving `treetn`-level behavior.

**Architecture:** Keep the current `TensorDynLen` public shape/index API, add native conversion helpers at the tensorbackend boundary, and switch a small set of algebra operations to tenferro-backed execution. Leave `Storage`/`TensorData` materialization APIs in place for compatibility.

**Tech Stack:** Rust, tenferro-rs, tensor4all-core, tensor4all-tensorbackend, cargo-nextest (`--release`)

---

### Task 1: Update tenferro dependency pin

**Files:**
- Modify: `Cargo.toml`

**Step 1: Update workspace tenferro revisions to the latest reviewed commit**

Set all `tenferro-*` workspace dependencies to the reviewed `origin/main` commit that contains
issues `#277/#278` fixes.

**Step 2: Run a focused build**

Run: `cargo build -p tensor4all-tensorbackend -p tensor4all-core`

**Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "build: update tenferro dependencies to latest reviewed main"
```

### Task 2: Add native storage conversion helpers

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Add tests in `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`

**Step 1: Write failing tests for storage/native round-trip**

Cover:
- dense f64 -> native -> dense
- dense c64 -> native -> dense
- diag f64 -> native -> dense-compatible output
- diag c64 -> native -> dense-compatible output

**Step 2: Run the focused tests and verify failure**

Run: `cargo nextest run --release -p tensor4all-tensorbackend tenferro_bridge`

**Step 3: Implement minimal conversion helpers**

Add helpers that convert between `Storage` and `DynAdTensor` for the supported scalar types.

**Step 4: Re-run tests**

Run: `cargo nextest run --release -p tensor4all-tensorbackend tenferro_bridge`

**Step 5: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/lib.rs crates/tensor4all-tensorbackend/src/tenferro_bridge.rs
git commit -m "feat(tensorbackend): add native tenferro tensor conversion helpers"
```

### Task 3: Route TensorDynLen scalar algebra through native execution

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Add tests in `crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Step 1: Write failing tests**

Cover:
- `scale` preserves scalar AD metadata
- `axpby` preserves scalar AD metadata
- `inner_product` returns AD-aware scalar through native execution

**Step 2: Run focused tests and verify failure**

Run: `cargo nextest run --release -p tensor4all-core tensordynlen`

**Step 3: Implement native execution path**

Use native conversion helpers and tenferro-backed operations in:
- `sum`
- `only`
- `conj`
- `scale`
- `axpby`
- `inner_product`

**Step 4: Re-run focused tests**

Run: `cargo nextest run --release -p tensor4all-core tensordynlen`

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs
git commit -m "feat(core): route TensorDynLen scalar algebra through tenferro"
```

### Task 4: Route TensorDynLen permutation/contraction through native execution

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/defaults/contract.rs`
- Add focused regression tests in `crates/tensor4all-core/src/defaults/`

**Step 1: Write failing tests**

Cover:
- `permute`
- `contract`
- `outer_product`
- mixed dense/diag contraction parity with existing behavior

**Step 2: Run focused tests and verify failure**

Run: `cargo nextest run --release -p tensor4all-core contract`

**Step 3: Implement native execution path**

Keep index pairing logic in tensor4all, but execute the numeric contraction via tenferro-native tensors.

**Step 4: Re-run focused tests**

Run: `cargo nextest run --release -p tensor4all-core contract`

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/src/defaults/contract.rs
git commit -m "feat(core): route TensorDynLen contraction path through tenferro"
```

### Task 5: Verify treetn compatibility gate

**Files:**
- No code changes required unless regressions appear

**Step 1: Run focused treetn tests**

Run: `cargo nextest run --release -p tensor4all-treetn`

**Step 2: Fix regressions if any**

Apply minimal fixes only.

**Step 3: Run final focused verification**

Run:
```bash
cargo fmt --all
cargo nextest run --release -p tensor4all-tensorbackend -p tensor4all-core -p tensor4all-treetn
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add first native tenferro execution slice for TensorDynLen"
```
