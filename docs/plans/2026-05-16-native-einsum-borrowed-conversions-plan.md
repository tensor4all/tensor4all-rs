# Native Einsum Borrowed Conversions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Avoid deep-cloning native einsum operands when borrowed inputs can be reused directly or only mismatched dtype operands need conversion.

**Architecture:** Keep the public tensorbackend API unchanged. Update `einsum_native_tensors` to build borrowed input references to either original operands or local converted temporaries, then call tenferro's borrowed `eager_einsum`.

**Tech Stack:** Rust, tensor4all-tensorbackend, tenferro `eager_einsum`, `anyhow`, release-mode cargo tests.

---

### Task 1: Add Tests For Borrowed Native Einsum Paths

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge/tests/mod.rs`

**Step 1: Write failing tests**

Add or update tests so same-dtype borrowed einsum records a new borrowed path
and mixed-dtype borrowed einsum records a borrowed-with-conversions path while
returning the promoted dtype and expected values.

**Step 2: Run tests to verify failure**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend tenferro_bridge::tests::einsum_native_tensors_dense_binary_records_borrowed_profile -- --exact
cargo test --release -p tensor4all-tensorbackend tenferro_bridge::tests::einsum_native_tensors_mixed_dtype_records_borrowed_conversion_profile -- --exact
```

Expected: fail to compile or fail assertions because the profile variants and
borrowed conversion path do not exist yet.

### Task 2: Implement Borrowed And Borrowed-With-Conversions Paths

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`

**Step 1: Import borrowed einsum**

Import `eager_einsum` alongside `eager_einsum_owned`.

**Step 2: Add profile variants**

Extend `NativeEinsumPath` with `Borrowed` and `BorrowedWithConversions`.

**Step 3: Implement minimal bridge change**

In `einsum_native_tensors`:

- reject empty operands;
- compute target dtype with `common_dtype`;
- validate rank and collect label ids;
- convert only operands whose dtype differs from the target;
- build references to converted temporaries or original operands;
- call `eager_einsum` with the borrowed references;
- record `Borrowed` when no conversions are needed, otherwise
  `BorrowedWithConversions`.

**Step 4: Run targeted tests**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend tenferro_bridge
```

Expected: all tensorbackend tenferro bridge tests pass.

### Task 3: Verification And Commit

**Files:**
- Verify all modified files.

**Step 1: Format and check**

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
git diff --check
cargo clippy -p tensor4all-tensorbackend --all-targets -- -D warnings
```

Expected: all commands pass.

**Step 2: Commit**

Run:

```bash
git add docs/plans/2026-05-16-native-einsum-borrowed-conversions-design.md docs/plans/2026-05-16-native-einsum-borrowed-conversions-plan.md crates/tensor4all-tensorbackend/src/tenferro_bridge.rs
git commit -m "Avoid native einsum operand clones"
```
