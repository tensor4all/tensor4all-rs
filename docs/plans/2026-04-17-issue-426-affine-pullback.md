# Issue 426 Affine Pullback C API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `t4a_qtransform_affine_pullback_materialize` to the C API with a DRY affine-family implementation and C API coverage.

**Architecture:** Keep the public C API surface explicit, but move affine-family parsing, validation, and `AffineParams` construction into a shared helper inside `tensor4all-capi::quanticstransform`. The forward affine and pullback affine entry points should differ only by the Rust quantics operator builder they call and their docstrings.

**Tech Stack:** Rust, `tensor4all-capi`, `tensor4all-quanticstransform`, C-API unit tests, `cargo fmt`, `cargo nextest`, `cargo clippy`

---

### Task 1: Add the first failing pullback C API test

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`

**Step 1: Write the failing test**

Add a test next to the existing affine materialization coverage that calls the new missing symbol and compares the materialized operator against `affine_pullback_operator(...)`.

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-capi test_affine_pullback_fused_materialization_matches_rust_reference`

Expected: FAIL because the C symbol does not exist yet or returns the wrong behavior.

**Step 3: Commit**

Do not commit yet if the implementation immediately follows.

### Task 2: Add one more failing behavior check

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`

**Step 1: Write the failing test**

Add a 1D dense-matrix comparison for a nontrivial pullback case, preferably a shift-like map, and verify the C API materialization matches the Rust pullback operator.

**Step 2: Run tests to verify they fail**

Run: `cargo nextest run --release -p tensor4all-capi test_affine_pullback`

Expected: FAIL due to missing implementation.

### Task 3: Implement the DRY affine-family helper

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`

**Step 1: Extract shared logic**

Introduce a helper for affine-family materialization that owns:

- fused-layout validation
- rational parsing
- boundary parsing
- `AffineParams::new`
- `t4a_treetn` wrapping

**Step 2: Add the pullback export**

Implement `t4a_qtransform_affine_pullback_materialize` as a thin wrapper over the helper.

**Step 3: Re-point the existing affine export**

Update `t4a_qtransform_affine_materialize` to use the same helper so the implementation stays DRY.

### Task 4: Document boundary semantics

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`

**Step 1: Update rustdoc**

Document the pullback semantics and make clear that `bc[i]` controls source coordinate `i` when `(A * y + b)[i]` leaves the valid interval.

### Task 5: Verify green

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`
- Modify: `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`

**Step 1: Run focused tests**

Run: `cargo nextest run --release -p tensor4all-capi test_affine_pullback`

Expected: PASS

**Step 2: Run crate verification**

Run: `cargo nextest run --release -p tensor4all-capi`

Expected: PASS

**Step 3: Run formatting and lint**

Run: `cargo fmt --all`

Run: `cargo clippy --workspace --all-targets -- -D warnings`

Expected: PASS

### Task 6: Commit

**Step 1: Create a focused commit**

```bash
git add docs/plans/2026-04-17-issue-426-affine-pullback-design.md \
        docs/plans/2026-04-17-issue-426-affine-pullback.md \
        crates/tensor4all-capi/src/quanticstransform.rs \
        crates/tensor4all-capi/src/quanticstransform/tests/mod.rs
git commit -m "feat(capi): expose affine pullback materialization"
```
