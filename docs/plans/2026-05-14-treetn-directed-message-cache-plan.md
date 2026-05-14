# TreeTN Directed Message Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace whole-component cached TreeTN evaluation with directed message caching that reuses intermediate subtree contractions.

**Architecture:** Add a generic `TensorLike::select_indices` API, then update `TreeTNCachedEvaluator` to root the tree at the chosen center and compute cached child-to-parent messages in postorder. Keep the public `TreeTNCachedEvaluator` API unchanged.

**Tech Stack:** Rust, `tensor4all-core::TensorLike`, existing `TensorDynLen::select_indices`, `TreeTN`, Criterion benchmarks.

---

### Task 1: Add TensorLike slicing API

**Files:**
- Modify: `crates/tensor4all-core/src/tensor_like.rs`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Steps:**
1. Add a default `fn select_indices(&self, selected_indices: &[Self::Index], positions: &[usize]) -> Result<Self>` to `TensorLike`.
2. Default implementation validates lengths, builds one-hot pairs, and contracts `self` with the one-hot tensor.
3. Override the method in `impl TensorLike for TensorDynLen` by calling the existing inherent `TensorDynLen::select_indices`.
4. Add rustdoc examples with assertions.
5. Verify with `cargo test --doc --release -p tensor4all-core`.

### Task 2: Add failing message-cache behavior test

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Steps:**
1. Extend the test-only stats assertion to expect a directed-message count.
2. Add a 3-node chain test where center `1` has two neighbor components, four unique component assignments, but only four directed messages total including internal reuse. The current implementation should fail because it has no directed-message stats.
3. Run `cargo test --release -p tensor4all-treetn cached_evaluator_reuses_directed_messages --lib` and verify failure.

### Task 3: Implement rooted message planning

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Steps:**
1. Add private `RootedMessagePlan<V>` with `parent`, `children`, and postorder node list for a selected center.
2. Add helpers for collecting subtree nodes from the rooted plan.
3. Keep existing center search and component-cost code unchanged.
4. Add tests for deterministic parent/child layout if needed.

### Task 4: Replace component environment construction

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Steps:**
1. Replace `compute_component_environment` with recursive `compute_message(node, assignment, plan, cache)`.
2. Message keys are exact `ComponentAssignment` values for the rooted subtree at `node`.
3. Slice local tensors with `TensorLike::select_indices`.
4. Contract local slices with child messages using `T::contract`.
5. Store messages in `HashMap<(V, ComponentAssignment), T>` per child node.
6. Update test-only stats for message compute count.
7. Verify cached evaluator tests pass.

### Task 5: Re-run benchmarks and validation

**Commands:**
- `cargo fmt --all -- --check`
- `cargo test --release -p tensor4all-treetn cached_evaluator --lib`
- `cargo test --doc --release -p tensor4all-core`
- `cargo test --doc --release -p tensor4all-treetn`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo bench --no-run -p tensor4all-treetn --bench cached_evaluator`
- short Criterion runs for `treetn_cached_batch_size/treetn_cached` and `treetn_cached_chain_size/treetn_cached`

### Task 6: Push PR update

**Steps:**
1. Commit implementation and docs.
2. Push `codex/batch-issues-2026-05-14`.
3. Comment on PR #501 with updated benchmark results and the root-cause summary.
