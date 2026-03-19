# Shift Open Boundary Full-Cycle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `shift_operator(..., BoundaryCondition::Open)` return the zero operator for negative full-cycle offsets, matching positive full-cycle behavior.

**Architecture:** Add a regression at the public operator level, then fix the single boundary-condition branch in `shift.rs` that currently preserves negative full cycles. Verify the targeted regression plus the crate test suite so the change stays scoped to shift semantics.

**Tech Stack:** Rust, cargo-nextest, tensor4all-quanticstransform integration tests

---

### Task 1: Add the failing regression

**Files:**
- Modify: `crates/tensor4all-quanticstransform/tests/integration_test.rs`
- Test: `crates/tensor4all-quanticstransform/tests/integration_test.rs`

**Step 1: Write the failing test**

Add an integration test that builds `shift_operator(3, offset, BoundaryCondition::Open)` for `offset in [-8, -9, -16]`, contracts the operator to a dense matrix, and asserts every entry is zero.

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-quanticstransform test_shift_open_boundary_negative_full_cycles_zero_operator`

Expected: FAIL because the current implementation leaves negative full-cycle operators non-zero.

### Task 2: Implement the minimal fix

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/shift.rs`

**Step 1: Update Open boundary full-cycle handling**

Change the `nbc != 0` branch so `BoundaryCondition::Open` always scales by zero, regardless of the sign of `nbc`.

**Step 2: Re-run the regression**

Run: `cargo nextest run --release -p tensor4all-quanticstransform test_shift_open_boundary_negative_full_cycles_zero_operator`

Expected: PASS.

### Task 3: Verify and finish

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/shift.rs`
- Modify: `crates/tensor4all-quanticstransform/tests/integration_test.rs`

**Step 1: Run relevant verification**

Run: `cargo nextest run --release -p tensor4all-quanticstransform`

Expected: PASS.

**Step 2: Format**

Run: `cargo fmt --all`

Expected: no diff after formatting.

**Step 3: Commit and open PR**

Run:

```bash
git add docs/plans/2026-03-19-shift-open-boundary-full-cycle.md \
  crates/tensor4all-quanticstransform/src/shift.rs \
  crates/tensor4all-quanticstransform/tests/integration_test.rs
git commit -m "fix(quanticstransform): zero open-boundary full-cycle shifts"
bash scripts/create-pr.sh
```

**Step 4: Monitor checks and merge**

Run: `bash scripts/monitor-pr-checks.sh`

Expected: CI green, then merge or stop only on an external blocker.
