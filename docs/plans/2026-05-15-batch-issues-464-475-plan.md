# Batch Issues 464 and 475 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify issue #464's cached evaluator surface and implement issue #475's constraint-row normalization substrate without changing affine-map semantics.

**Architecture:** Keep the existing affine map API unchanged. Add a dedicated `LinearConstraintRow` API in `tensor4all-quanticstransform` that stores primitive integer coefficients and RHS after rational denominator clearing and positive gcd reduction. Export and document it so future affine/halfspace projector builders can normalize constraints before operator construction.

**Tech Stack:** Rust, `num-rational::Rational64`, `num-integer::Integer`, cargo nextest, rustdoc.

---

### Task 1: Verify #464 Cached Evaluator Surface

**Files:**
- Read: `docs/api/tensor4all_treetn.md`
- Read: `docs/api/tensor4all_capi.md`
- Test: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Test: `crates/tensor4all-capi/src/treetn/tests/mod.rs`

**Step 1: Run focused TreeTN evaluator tests**

Run:

```bash
cargo nextest run --release -p tensor4all-treetn cached_evaluator
```

Expected: PASS. This verifies `TreeTNCachedEvaluator` correctness and validation paths.

**Step 2: Run focused C API evaluator tests**

Run:

```bash
cargo nextest run --release -p tensor4all-capi treetn_evaluator
```

Expected: PASS. This verifies the handle-based C API evaluator surface.

**Step 3: Record result**

If both pass, mark #464 as already implemented by the base branch in the final batch report. Do not add artificial code.

### Task 2: Add Failing Tests for #475 Constraint Normalization

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine/tests/mod.rs`

**Step 1: Write tests**

Add tests for:

- `LinearConstraintRow::from_integers(vec![16], 64)` returns coefficients `[1]` and RHS `4`.
- `LinearConstraintRow::from_rationals(vec![2/3, 4/3], 2)` returns coefficients `[1, 2]` and RHS `3`.
- `LinearConstraintRow::from_integers(vec![-16], -64)` returns coefficients `[-1]` and RHS `-4`.
- `LinearConstraintRow::from_integers(vec![0, 0], 0)` stays `[0, 0]`, `0`.
- `AffineParams::from_integers(vec![16], vec![64], 1, 1).unwrap().to_integer_scaled()` still returns `[16]`, `[64]`, `1`.

**Step 2: Verify RED**

Run:

```bash
cargo test --release -p tensor4all-quanticstransform linear_constraint
```

Expected: FAIL because `LinearConstraintRow` does not exist.

### Task 3: Implement `LinearConstraintRow`

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs`
- Modify: `crates/tensor4all-quanticstransform/src/lib.rs`

**Step 1: Add the type**

Add a public `LinearConstraintRow` with fields:

```rust
pub struct LinearConstraintRow {
    pub coefficients: Vec<i64>,
    pub rhs: i64,
}
```

**Step 2: Add constructors**

Add:

```rust
pub fn from_integers(coefficients: Vec<i64>, rhs: i64) -> Self
pub fn from_rationals(coefficients: Vec<Rational64>, rhs: Rational64) -> Self
```

The rational constructor clears denominators using a row-local LCM, then calls
the integer constructor. The integer constructor divides by a positive gcd of
all coefficients and the RHS. If the row is all zero, it returns the zero row.

**Step 3: Export**

Export `LinearConstraintRow` from `crates/tensor4all-quanticstransform/src/lib.rs`.

**Step 4: Verify GREEN**

Run:

```bash
cargo test --release -p tensor4all-quanticstransform linear_constraint
```

Expected: PASS.

### Task 4: Focused Verification and Formatting

**Files:**
- Modified files from Tasks 2 and 3

**Step 1: Format**

Run:

```bash
cargo fmt --all
```

**Step 2: Run focused tests**

Run:

```bash
cargo nextest run --release -p tensor4all-quanticstransform affine
cargo nextest run --release -p tensor4all-treetn cached_evaluator
cargo nextest run --release -p tensor4all-capi treetn_evaluator
```

Expected: PASS.

**Step 3: Run rustdoc for changed crate if public API changed**

Run:

```bash
cargo test --doc --release -p tensor4all-quanticstransform
```

Expected: PASS.

### Task 5: Commit Locally

**Files:**
- `docs/plans/2026-05-15-batch-issues-464-475-design.md`
- `docs/plans/2026-05-15-batch-issues-464-475-plan.md`
- `crates/tensor4all-quanticstransform/src/affine.rs`
- `crates/tensor4all-quanticstransform/src/affine/tests/mod.rs`
- `crates/tensor4all-quanticstransform/src/lib.rs`

**Step 1: Inspect diff**

Run:

```bash
git diff --stat
git diff -- crates/tensor4all-quanticstransform/src/affine.rs
```

**Step 2: Commit**

Run:

```bash
git add docs/plans/2026-05-15-batch-issues-464-475-design.md \
        docs/plans/2026-05-15-batch-issues-464-475-plan.md \
        crates/tensor4all-quanticstransform/src/affine.rs \
        crates/tensor4all-quanticstransform/src/affine/tests/mod.rs \
        crates/tensor4all-quanticstransform/src/lib.rs
git commit -m "feat(quanticstransform): normalize linear constraint rows"
```
