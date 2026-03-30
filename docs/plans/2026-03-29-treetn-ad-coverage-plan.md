# TreeTN AD Coverage Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a generic real/complex TreeTN AD integration harness and use it to expand representative state-operation coverage without duplicating test logic.

**Architecture:** Keep all new work in test code first. Introduce a small test-only scalar helper trait in `crates/tensor4all-treetn/tests/ad_treetn.rs`, convert the existing real-only tests to generic harnesses, and then add representative operation checks operation-by-operation. If a newly exposed AD bug appears, fix only the minimal production code needed to satisfy the new failing test.

**Tech Stack:** Rust 2021, `tensor4all-core`, `tensor4all-treetn`, `num-complex`, `cargo fmt`, `cargo nextest --release`

---

### Task 1: Add generic scalar test helpers

**Files:**
- Modify: `crates/tensor4all-treetn/tests/ad_treetn.rs`

**Steps:**
1. Add a test-only scalar helper trait for `f64` and `Complex64`.
2. Move fixed fixture generation and dense/scalar comparison helpers behind the trait where needed.
3. Add thin `f64` / `Complex64` wrapper tests for one existing forward operation.
4. Run the focused test target and verify the first generic conversion fails for the expected reason before changing the rest.

### Task 2: Convert existing TreeTN AD coverage to the generic harness

**Files:**
- Modify: `crates/tensor4all-treetn/tests/ad_treetn.rs`

**Steps:**
1. Convert `to_dense` forward/backward tests to the generic harness.
2. Convert `canonicalize` and `truncate` forward tests to the generic harness.
3. Add complex-aware backward losses for the tensor-returning paths.
4. Run the focused AD test target and keep only the minimal changes required to turn it green.

### Task 3: Add representative state-operation AD tests

**Files:**
- Modify: `crates/tensor4all-treetn/tests/ad_treetn.rs`
- Reference: `crates/tensor4all-treetn/tests/ops.rs`
- Reference: `crates/tensor4all-treetn/tests/swap_test.rs`

**Steps:**
1. Add generic forward tests for `add`.
2. Add generic forward tests for `evaluate`.
3. Add generic forward tests for `swap_site_indices`.
4. Add reverse-mode checks only where the operation returns a tensor and the loss can remain explicitly real-valued.
5. Reuse existing non-AD fixtures and expectations where practical instead of inventing new semantics.

### Task 4: Verify representative coverage and stop short of Tier 2

**Files:**
- Modify: `crates/tensor4all-treetn/tests/ad_treetn.rs`
- Reference: `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs`
- Reference: `crates/tensor4all-treetn/src/treetn/fit/tests/mod.rs`

**Steps:**
1. Confirm the current pass does not accidentally sprawl into `apply_linear_operator`, `contract_fit`, or `contract_zipup`.
2. If a failing test points into one of those paths indirectly, either make the minimal fix or stop and split the bug cleanly.
3. Keep the issue scope aligned with state-operation AD coverage first.

### Task 5: Verification

**Files:**
- Verify: `crates/tensor4all-treetn/tests/ad_treetn.rs`

**Steps:**
1. Run `cargo nextest run --release -p tensor4all-treetn --test ad_treetn`.
2. If additional representative tests land outside `ad_treetn.rs`, run those focused targets as well.
3. Run `cargo fmt --all`.
4. Re-run the focused release-mode test target after formatting.
