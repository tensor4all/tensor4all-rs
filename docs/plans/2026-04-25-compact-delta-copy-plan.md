# Compact Delta/Copy Tensor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure `TensorDynLen::diagonal()` and `TensorLike::delta()` preserve compact diagonal/copy structure instead of silently constructing dense tensors.

**Architecture:** Reuse the existing `TensorDynLen::from_diag` and structured axis-class storage model. Change only the `TensorDynLen` implementation of `TensorLike::diagonal`, keep the generic `TensorLike::delta` default, and add representation-level regression tests in core and TreeTN apply.

**Tech Stack:** Rust, `tensor4all-core::TensorDynLen`, `tensor4all_core::StorageKind`, `TensorLike`, `tensor4all-treetn::apply_linear_operator`, `cargo nextest --release`.

---

### Task 1: Add compact diagonal/delta representation tests

**Files:**
- Modify: `crates/tensor4all-core/tests/tensor_diag.rs`

**Step 1: Write failing tests**

Add tests near the existing diagonal/copy tensor tests:

```rust
#[test]
fn tensorlike_diagonal_uses_compact_diagonal_storage() {
    let i = Index::new_dyn(4);
    let o = Index::new_dyn(4);

    let delta = <TensorDynLen as TensorLike>::diagonal(&i, &o).unwrap();

    assert!(delta.is_diag());
    assert_eq!(delta.storage().storage_kind(), StorageKind::Diagonal);
    assert_eq!(delta.storage().payload_dims(), &[4]);
    assert_eq!(delta.storage().axis_classes(), &[0, 0]);
    assert_eq!(
        delta.storage().payload_f64_col_major_vec().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0],
    );
}

#[test]
fn tensorlike_delta_two_pairs_preserves_independent_copy_structure() {
    let i1 = Index::new_dyn(2);
    let o1 = Index::new_dyn(2);
    let i2 = Index::new_dyn(3);
    let o2 = Index::new_dyn(3);

    let delta = <TensorDynLen as TensorLike>::delta(
        &[i1.clone(), i2.clone()],
        &[o1.clone(), o2.clone()],
    )
    .unwrap();

    assert_eq!(delta.dims(), vec![2, 2, 3, 3]);
    assert_eq!(delta.storage().storage_kind(), StorageKind::Structured);
    assert_eq!(delta.storage().payload_dims(), &[2, 3]);
    assert_eq!(delta.storage().axis_classes(), &[0, 0, 1, 1]);

    let expected = TensorDynLen::from_diag(vec![i1, o1], vec![1.0_f64, 1.0])
        .unwrap()
        .outer_product(
            &TensorDynLen::from_diag(vec![i2, o2], vec![1.0_f64, 1.0, 1.0]).unwrap(),
        )
        .unwrap();
    assert!(delta.isapprox(&expected, 1e-12, 0.0));
}
```

**Step 2: Verify RED**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_diag tensorlike_diagonal
cargo nextest run --release -p tensor4all-core --test tensor_diag tensorlike_delta_two_pairs
```

Expected: the first test fails because `diagonal()` currently returns dense storage.

**Step 3: Commit tests if desired**

Do not commit only red tests unless the workflow requires it. They can be committed with the implementation in Task 2.

### Task 2: Make `TensorDynLen::diagonal` compact

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `crates/tensor4all-core/tests/tensor_diag.rs`

**Step 1: Implement minimal change**

Replace the dense identity allocation in the `impl TensorLike for TensorDynLen` method:

```rust
fn diagonal(input_index: &DynIndex, output_index: &DynIndex) -> Result<Self> {
    let dim = input_index.dim();
    if dim != output_index.dim() {
        return Err(anyhow::anyhow!(
            "Dimension mismatch: input index has dim {}, output has dim {}",
            dim,
            output_index.dim(),
        ));
    }

    TensorDynLen::from_diag(
        vec![input_index.clone(), output_index.clone()],
        vec![1.0_f64; dim],
    )
}
```

Do not change `TensorLike::delta()` unless tests show the default implementation loses structure after this change.

**Step 2: Verify GREEN**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_diag tensorlike_diagonal
cargo nextest run --release -p tensor4all-core --test tensor_diag tensorlike_delta_two_pairs
```

Expected: both tests pass.

**Step 3: Broader core verification**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_diag
cargo nextest run --release -p tensor4all-core --test tensor_contraction structured
```

Expected: all selected tests pass.

**Step 4: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/tests/tensor_diag.rs
git commit -m "fix: preserve compact delta tensors"
```

### Task 3: Add TreeTN bridge-delta regression

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs`

**Step 1: Write failing or protective test**

Add a unit test near the local naive apply tests. It should inspect the extended
operator before final apply, because final local contraction may legitimately
change storage representation:

```rust
#[test]
fn naive_apply_noncontiguous_bonded_identity_uses_compact_bridge_delta() {
    let (state, sites) = build_chain_state();
    let operator = build_redundant_bonded_identity_operator(
        &[sites[0].clone(), sites[2].clone()],
        2,
    );

    let extended = extend_operator_to_full_space(&operator, &state).unwrap();
    let middle = extended.mpo.node_index(&sites[1].0).unwrap();
    let middle_tensor = extended.mpo.tensor(middle).unwrap();

    assert_ne!(middle_tensor.storage().storage_kind(), StorageKind::Dense);
    assert!(
        middle_tensor
            .storage()
            .axis_classes()
            .windows(2)
            .any(|window| window[0] == window[1]),
        "bridge tensor should retain repeated axis classes"
    );

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let result_dense = result.to_dense().unwrap();
    let state_dense = state.to_dense().unwrap();
    assert!((&result_dense - &state_dense).maxabs() < 1e-10);
}
```

Add `StorageKind` to the test imports if needed.

**Step 2: Verify test**

Run:

```bash
cargo nextest run --release -p tensor4all-treetn naive_apply_noncontiguous_bonded_identity_uses_compact_bridge_delta
```

Expected: pass after Task 2. If it fails because the bridge is still dense, inspect `compose_operator_along_state_paths` for an accidental dense helper tensor.

**Step 3: Commit**

```bash
git add crates/tensor4all-treetn/src/operator/apply/tests/mod.rs
git commit -m "test: cover compact bridge delta in naive apply"
```

### Task 4: Update docs for local naive and compact structure

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/apply.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/contraction.rs`
- Modify: `docs/book/src/guides/quantics.md`
- Optional: `docs/book/src/guides/qft.md` if wording implies dense/reference behavior.

**Step 1: Search stale wording**

Run:

```bash
rg "ApplyOptions::naive\\(\\)|Contract to full tensor|full tensor, re-decompose|Naive contraction|dense/reference|contract_to_tensor" crates/tensor4all-treetn docs/book/src README.md
```

**Step 2: Update `ApplyOptions::naive()` wording**

In `apply.rs`, make rustdoc say:

- `ApplyOptions::naive()` performs exact local operator/state application.
- It does not full-materialize the state or operator.
- It is exact and local, but can grow bond dimensions as products of state and MPO bonds.
- Truncation should use zipup/fit paths until exact-then-truncate is implemented.

**Step 3: Clarify generic TreeTN `ContractionMethod::Naive`**

In `treetn/contraction.rs`, keep generic `ContractionMethod::Naive` documented as dense/reference behavior if it remains public.

**Step 4: Update mdBook**

Change stale guide text such as:

```markdown
| `ApplyOptions::naive()` | Contract to full tensor, re-decompose | Small systems, debugging, exactness required |
```

to a local exact description:

```markdown
| `ApplyOptions::naive()` | Local exact MPO/state contraction | Exact small-to-moderate applies; bond dimensions may grow as products |
```

**Step 5: Verify docs**

Run:

```bash
cargo test --doc --release -p tensor4all-core -p tensor4all-treetn
```

Expected: doctests pass.

**Step 6: Commit**

```bash
git add crates/tensor4all-treetn/src/operator/apply.rs crates/tensor4all-treetn/src/treetn/contraction.rs docs/book/src/guides/quantics.md docs/book/src/guides/qft.md
git commit -m "docs: clarify local naive apply semantics"
```

Only stage files actually modified.

### Task 5: Final verification

**Files:**
- All touched files.

**Step 1: Formatting**

Run:

```bash
cargo fmt --all
git diff --check
```

Expected: no formatting or whitespace issues.

**Step 2: Targeted tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_diag
cargo nextest run --release -p tensor4all-treetn naive_apply
cargo nextest run --release -p tensor4all-treetn apply_linear_operator
```

Expected: all pass.

**Step 3: Broader crate tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core -p tensor4all-treetn
```

Expected: all pass.

**Step 4: Doctests**

Run:

```bash
cargo test --doc --release -p tensor4all-core -p tensor4all-treetn
```

Expected: all pass.

**Step 5: Review dense paths**

Run:

```bash
rg "to_dense\\(|contract_to_tensor\\(|from_dense\\(|T::delta\\(|TensorDynLen::diagonal|ApplyOptions::naive" crates/tensor4all-core/src crates/tensor4all-treetn/src docs/book/src
```

Expected: no stale production dense delta or naive apply wording remains.

**Step 6: Commit verification-only changes if any**

If formatting or docs changed:

```bash
git add <changed-files>
git commit -m "chore: finalize compact naive apply"
```
