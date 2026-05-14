# TreeTN Cached Batch-Dim C API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Batch `TreeTNCachedEvaluator` message contractions over an explicit retained assignment index and make the C API TreeTN evaluator use that cached path.

**Architecture:** Specialize `TreeTNCachedEvaluator` for `TensorDynLen` so it can create `DynIndex` assignment axes and call `ContractionOptions::with_retain_indices`. Keep greedy center selection and compact assignment batches, but compute each node's messages as one stacked tensor per node instead of one tensor per unique assignment. Replace the C API evaluator internals with owned cached-evaluation state and route both reusable and one-shot C evaluation through the cached path.

**Tech Stack:** Rust, `tensor4all-core::TensorDynLen`, retained-index `contract_multi_with_options`, `TreeTN`, `tensor4all-capi`, cbindgen, Criterion.

---

### Task 1: Add failing TensorDynLen stack/gather tests

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Step 1: Write failing helper tests**

Add tests inside the existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn stack_tensors_adds_assignment_axis_in_column_major_order() {
    let batch = DynIndex::new_dyn(2);
    let i = DynIndex::new_dyn(2);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0_f64, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap();

    let stacked = stack_tensors_with_assignment_index(&batch, &[a, b]).unwrap();

    assert_eq!(stacked.indices(), &[batch, i]);
    assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn gather_stacked_tensor_remaps_assignment_axis() {
    let source_batch = DynIndex::new_dyn(3);
    let target_batch = DynIndex::new_dyn(4);
    let i = DynIndex::new_dyn(2);
    let stacked = TensorDynLen::from_dense(
        vec![source_batch.clone(), i.clone()],
        vec![10.0_f64, 20.0, 30.0, 11.0, 21.0, 31.0],
    ).unwrap();

    let gathered =
        gather_stacked_tensor(&stacked, &source_batch, &target_batch, &[2, 0, 2, 1]).unwrap();

    assert_eq!(gathered.indices(), &[target_batch, i]);
    assert_eq!(gathered.to_vec::<f64>().unwrap(), vec![30.0, 10.0, 30.0, 20.0, 31.0, 11.0, 31.0, 21.0]);
}
```

**Step 2: Run tests to verify red**

Run:

```bash
cargo test --release -p tensor4all-treetn stack_tensors_adds_assignment_axis_in_column_major_order --lib
cargo test --release -p tensor4all-treetn gather_stacked_tensor_remaps_assignment_axis --lib
```

Expected: FAIL to compile because the helper functions do not exist.

**Step 3: Implement minimal helpers**

Add private helpers near `slice_tensor`:

- `stack_tensors_with_assignment_index(batch_index: &DynIndex, tensors: &[TensorDynLen]) -> Result<TensorDynLen>`
- `gather_stacked_tensor(stacked: &TensorDynLen, source_batch: &DynIndex, target_batch: &DynIndex, selected_assignments: &[usize]) -> Result<TensorDynLen>`
- `tensor_values_any(tensor: &TensorDynLen) -> Result<Vec<AnyScalar>>`

Both helpers should require the assignment axis to be the first axis in stacked
tensors. Use column-major indexing: with batch first, flat offset is
`batch_value + batch_dim * rest_offset`.

**Step 4: Run tests to verify green**

Run:

```bash
cargo test --release -p tensor4all-treetn stack_tensors_adds_assignment_axis_in_column_major_order --lib
cargo test --release -p tensor4all-treetn gather_stacked_tensor_remaps_assignment_axis --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "test(treetn): cover stacked cached evaluator tensors"
```

### Task 2: Specialize cached evaluator around TensorDynLen

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treetn/src/lib.rs` only if public re-exports need signature updates

**Step 1: Write failing compile-targeted tests**

Update existing rustdoc examples and tests to instantiate
`TreeTNCachedEvaluator<'_, V>` rather than `TreeTNCachedEvaluator<'_, T, V>` if
the type signature changes. Add a test that calls `center()` before and after
evaluation to ensure the public behavior remains.

**Step 2: Run targeted tests**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator --lib
```

Expected: existing generic signatures fail until the implementation is updated.

**Step 3: Refactor type signatures**

Change cached evaluator internals from generic `T: TensorLike` to
`TensorDynLen`/`DynIndex`:

- `EvaluatorLayout<DynIndex, V>`
- `EnvironmentCache<V> = HashMap<V, TensorDynLen>`
- `TreeTNCachedEvaluator<'a, V> { tree: &'a TreeTN<TensorDynLen, V>, ... }`
- `ComponentCostIndex::new` and `from_layout` should accept
  `TreeTN<TensorDynLen, V>`

Keep `GreedyCenterSearch<V>`, `CenterSearchResult<V>`, and
`CachedEvaluatorOptions<V>` generic over node names.

**Step 4: Run targeted tests**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator --lib
cargo test --doc --release -p tensor4all-treetn TreeTNCachedEvaluator
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs crates/tensor4all-treetn/src/lib.rs
git commit -m "refactor(treetn): specialize cached evaluator tensors"
```

### Task 3: Batch directed message construction

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Step 1: Write failing behavior test**

Add a test on `five_node_chain()` that evaluates the same points as
`cached_evaluator_reuses_directed_messages_inside_components`, then asserts a
new test stat such as `batched_message_contract_count` is smaller than
`directed_message_count`.

```rust
assert!(stats.batched_message_contract_count < stats.directed_message_count);
```

**Step 2: Run red test**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator_batches_directed_messages --lib
```

Expected: FAIL to compile because the stat and batched path do not exist.

**Step 3: Implement stacked message tensors**

Replace `EnvironmentCache<V> = HashMap<V, Vec<TensorDynLen>>` with stacked
message storage:

```rust
struct StackedMessage {
    assignment_index: DynIndex,
    tensor: TensorDynLen,
}
type EnvironmentCache<V> = HashMap<V, StackedMessage>;
```

For each non-center node in postorder:

1. Create `assignment_index = DynIndex::new_dyn(assignment_batch.first_points.len())`.
2. Build local slices for each `first_point`.
3. Stack them with `stack_tensors_with_assignment_index`.
4. For each child, gather the child's stacked message onto the node assignment
   index using the child assignment selected at each `first_point`.
5. Contract local stack and gathered child stacks with:

```rust
let retain = [assignment_index.clone()];
let options = ContractionOptions::new(AllowedPairs::All).with_retain_indices(&retain);
contract_multi_with_options(&tensor_refs, options)
```

6. Store the result as that node's `StackedMessage`.

**Step 4: Run green tests**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "perf(treetn): batch cached directed messages"
```

### Task 4: Batch center contraction and scalar extraction

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`

**Step 1: Write failing test**

Add a star-tree test that evaluates four points, checks values against
`TreeTN::evaluate`, and asserts `batched_center_contract_count == 1`.

**Step 2: Run red test**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator_batches_center_contraction --lib
```

Expected: FAIL before the center batched path is implemented.

**Step 3: Implement center stack**

In `contract_center_for_points`:

1. Create a point assignment index `DynIndex::new_dyn(n_points)`.
2. Stack sliced center tensors for each point.
3. For each neighbor environment, gather the neighbor stacked message using
   `component_batches[*].point_to_assignment`.
4. Contract all stacked operands retaining the point index.
5. Convert the resulting rank-1 tensor to `Vec<AnyScalar>`.

**Step 4: Run green tests**

Run:

```bash
cargo test --release -p tensor4all-treetn cached_evaluator --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "perf(treetn): batch cached center contraction"
```

### Task 5: Replace C API evaluator internals with cached state

**Files:**
- Modify: `crates/tensor4all-capi/src/types.rs`
- Modify: `crates/tensor4all-capi/src/treetn.rs`
- Modify: `crates/tensor4all-capi/src/treetn/tests/mod.rs`
- Regenerate: `crates/tensor4all-capi/include/tensor4all_capi.h`

**Step 1: Write failing C API tests**

Update existing evaluator tests so `t4a_treetn_evaluator_evaluate` takes
`*mut t4a_treetn_evaluator`. Add a new test that releases the original TreeTN
after creating an evaluator, then evaluates successfully through the evaluator.
This proves the handle owns enough state.

**Step 2: Run red test**

Run:

```bash
cargo test --release -p tensor4all-capi test_treetn_evaluator_reuses_index_order_for_multiple_batches --lib
```

Expected: FAIL to compile until the C function signature and handle state are updated.

**Step 3: Update handle state**

In `types.rs`, replace `InternalTreeTNEvaluatorHandle` with:

```rust
#[derive(Clone)]
pub(crate) struct InternalTreeTNEvaluatorHandle {
    pub(crate) tree: InternalTreeTN,
    pub(crate) indices: Vec<InternalIndex>,
    pub(crate) center: Option<usize>,
    pub(crate) scalar_kind: t4a_scalar_kind,
}
```

Remove the `InternalTreeTNEvaluator` type alias if unused.

**Step 4: Route C evaluation through TreeTNCachedEvaluator**

In `treetn.rs`:

- `t4a_treetn_evaluator_new` stores `tn.inner().clone()`, collected indices,
  `center: None`, and scalar kind.
- `require_evaluator_mut` accepts `*mut t4a_treetn_evaluator`.
- `t4a_treetn_evaluator_evaluate` takes `*mut t4a_treetn_evaluator`.
- It constructs `TreeTNCachedEvaluator::new(&handle.tree, &handle.indices, CachedEvaluatorOptions { center: handle.center, ..Default::default() })`.
- After evaluation, write `handle.center = evaluator.center().copied()`.
- `t4a_treetn_evaluate` uses the same cached evaluator path for one-shot calls.

**Step 5: Regenerate header**

Run:

```bash
mkdir -p crates/tensor4all-capi/include
cbindgen crates/tensor4all-capi --config crates/tensor4all-capi/cbindgen.toml --output crates/tensor4all-capi/include/tensor4all_capi.h
```

Expected: header signature changes for `t4a_treetn_evaluator_evaluate`.

**Step 6: Run C API tests**

Run:

```bash
cargo test --release -p tensor4all-capi treetn_evaluator --lib
```

Expected: PASS.

**Step 7: Commit**

```bash
git add crates/tensor4all-capi/src/types.rs crates/tensor4all-capi/src/treetn.rs crates/tensor4all-capi/src/treetn/tests/mod.rs crates/tensor4all-capi/include/tensor4all_capi.h
git commit -m "feat(capi): use cached TreeTN evaluator"
```

### Task 6: Benchmark and validate

**Files:**
- Modify: `crates/tensor4all-treetn/benches/cached_evaluator.rs`
- Modify: PR body after benchmark data is collected

**Step 1: Update benchmark labels if needed**

Keep `treetn_cached` as the optimized cached evaluator. Add a temporary
`treetn_uncached` baseline where it is already present. Do not keep a duplicate
old cached implementation unless profiling shows it is necessary.

**Step 2: Run validation**

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release -p tensor4all-treetn cached_evaluator --lib
cargo test --release -p tensor4all-capi treetn_evaluator --lib
cargo test --doc --release --workspace
cargo nextest run --release --workspace
cargo bench --no-run -p tensor4all-treetn --bench cached_evaluator
```

Expected: PASS.

**Step 3: Run short benchmarks**

Run:

```bash
cargo bench -p tensor4all-treetn --bench cached_evaluator -- treetn_cached_bond_dim/{ttcache,treetn_cached} --sample-size 10 --measurement-time 1 --warm-up-time 1
cargo bench -p tensor4all-treetn --bench cached_evaluator -- treetn_cached_chain_size/{ttcache,treetn_cached} --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Expected: report TreeTN/TTCache ratio and whether it decreases as `bond_dim`
grows.

**Step 4: Commit benchmark/docs updates**

```bash
git add crates/tensor4all-treetn/benches/cached_evaluator.rs
git commit -m "bench(treetn): report cached batch-dim scaling"
```

### Task 7: Push and open PR

**Files:**
- No source changes unless PR body is saved to a temp file.

**Step 1: Check branch state**

Run:

```bash
git fetch origin
git merge-base --is-ancestor origin/main HEAD
git status --short --branch
```

Expected: branch contains current `origin/main`, working tree clean.

**Step 2: Push branch**

Run:

```bash
git push -u origin codex/treetn-batch-dim-capi
```

**Step 3: Create PR**

Run:

```bash
gh pr create --base main --head codex/treetn-batch-dim-capi --title "[codex] Batch TreeTN cached evaluator contractions" --body-file /tmp/treetn-batch-dim-capi-pr.md
```

The PR body must mention issue #502, C API signature changes, benchmark data,
and validation commands.
