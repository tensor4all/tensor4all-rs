# Structured Select Indices Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add general structured-storage support to `TensorDynLen::select_indices` while preserving compact structure and avoiding full input dense materialization.

**Architecture:** First add internal structured-zero construction for compact zero tensors. Then implement structured selection as payload slicing plus axis-class canonicalization. Keep the API surface private and add representation-level regression tests.

**Tech Stack:** Rust, `tensor4all-core::TensorDynLen`, `tensor4all_tensorbackend::Storage`, `StorageKind`, structured `axis_classes`, `cargo test --release`.

---

### Task 1: Add structured zero tests

**Files:**
- Modify: `crates/tensor4all-core/tests/tensor_diag.rs`

**Step 1: Write failing tests**

Add tests near the structured/diagonal storage tests:

```rust
#[test]
fn structured_zeros_preserves_copy_structure() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(3);

    let tensor = TensorDynLen::structured_zeros::<f64>(
        vec![i, j, k],
        vec![0, 0, 1],
    )
    .unwrap();

    assert_eq!(tensor.storage().storage_kind(), StorageKind::Structured);
    assert_eq!(tensor.storage().payload_dims(), &[2, 3]);
    assert_eq!(tensor.storage().axis_classes(), &[0, 0, 1]);
    assert_eq!(
        tensor.storage().payload_f64_col_major_vec().unwrap(),
        vec![0.0; 6],
    );
    assert!(tensor.to_vec::<f64>().unwrap().iter().all(|&x| x == 0.0));
}

#[test]
fn structured_zeros_rejects_mismatched_shared_class_dims() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let err = TensorDynLen::structured_zeros::<f64>(vec![i, j], vec![0, 0])
        .unwrap_err();

    assert!(err.to_string().contains("same structured axis class"));
}
```

These tests call a new `pub(crate)` helper, so place them in a source-level
`#[cfg(test)]` module if integration tests cannot access the helper. If keeping
the helper private, move equivalent tests into
`crates/tensor4all-core/src/defaults/tensordynlen.rs`.

**Step 2: Verify RED**

Run:

```bash
cargo test -p tensor4all-core --release structured_zeros -- --nocapture
```

Expected: fail because `structured_zeros` does not exist yet.

### Task 2: Implement private structured zero helper

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Step 1: Add helper**

Implement:

```rust
pub(crate) fn structured_zeros<T>(
    indices: Vec<DynIndex>,
    axis_classes: Vec<usize>,
) -> Result<Self>
where
    T: TensorElement + Zero + Clone,
{
    let dims = Self::expected_dims_from_indices(&indices);
    let payload_dims = Self::payload_dims_from_axis_classes(&dims, &axis_classes)?;
    let payload_len = payload_dims.iter().product::<usize>();
    let strides = tensor4all_tensorbackend::col_major_strides(&payload_dims);
    let storage = Storage::new_structured(
        vec![T::zero(); payload_len],
        payload_dims,
        strides,
        axis_classes,
    )?;
    Self::from_storage(indices, Arc::new(storage))
}
```

If `col_major_strides` is not public, either add a small local checked helper in
`tensordynlen.rs` or add a crate-visible backend helper with focused tests.

**Step 2: Add dimension helper**

Implement a private helper:

```rust
fn payload_dims_from_axis_classes(
    logical_dims: &[usize],
    axis_classes: &[usize],
) -> Result<Vec<usize>>
```

Rules:

- `axis_classes.len() == logical_dims.len()`;
- classes must be canonical first-appearance form;
- all logical axes in the same class have the same dimension;
- return payload dimensions in class order.

**Step 3: Verify GREEN**

Run:

```bash
cargo test -p tensor4all-core --release structured_zeros -- --nocapture
```

Expected: structured-zero tests pass.

### Task 3: Add structured select tests

**Files:**
- Modify: `crates/tensor4all-core/tests/tensor_diag.rs`

**Step 1: Write failing tests**

Add tests for general structured selection. Use `Storage::new_structured` and
`TensorDynLen::from_structured_storage` to construct the input:

```rust
#[test]
fn structured_select_preserves_repeated_kept_axis_class() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let l = Index::new_dyn(5);
    let storage = Arc::new(
        Storage::new_structured(
            (0..30).map(|x| x as f64).collect(),
            vec![2, 3, 5],
            vec![1, 2, 6],
            vec![0, 1, 0, 2],
        )
        .unwrap(),
    );
    let tensor = TensorDynLen::from_structured_storage(
        vec![i.clone(), j.clone(), k.clone(), l.clone()],
        storage,
    )
    .unwrap();

    let selected = tensor.select_indices(&[j], &[1]).unwrap();

    assert_eq!(selected.indices(), &[i, k, l]);
    assert_eq!(selected.storage().storage_kind(), StorageKind::Structured);
    assert_eq!(selected.storage().payload_dims(), &[2, 5]);
    assert_eq!(selected.storage().axis_classes(), &[0, 0, 1]);
}

#[test]
fn structured_select_conflicting_fixed_class_returns_compact_zero() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(3);
    let storage = Arc::new(
        Storage::new_structured(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![2, 3],
            vec![1, 2],
            vec![0, 0, 1],
        )
        .unwrap(),
    );
    let tensor = TensorDynLen::from_structured_storage(
        vec![i.clone(), j.clone(), k.clone()],
        storage,
    )
    .unwrap();

    let selected = tensor.select_indices(&[i, j], &[0, 1]).unwrap();

    assert_eq!(selected.indices(), &[k]);
    assert_eq!(selected.storage().payload_dims(), &[3]);
    assert_eq!(selected.storage().payload_f64_col_major_vec().unwrap(), vec![0.0; 3]);
    assert!(selected.to_vec::<f64>().unwrap().iter().all(|&x| x == 0.0));
}
```

**Step 2: Verify RED**

Run:

```bash
cargo test -p tensor4all-core --release structured_select -- --nocapture
```

Expected: fail because `select_indices` still rejects general structured storage.

### Task 4: Implement structured select metadata planning

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Step 1: Add a plan struct**

Add private struct:

```rust
struct StructuredSelectPlan {
    kept_indices: Vec<DynIndex>,
    new_axis_classes: Vec<usize>,
    old_class_for_new_payload_axis: Vec<usize>,
    fixed_old_class_positions: Vec<Option<usize>>,
    conflict: bool,
}
```

Adjust fields as needed, but keep the plan explicit and testable.

**Step 2: Add planner**

Implement a private helper that takes selected axes/positions and storage
metadata, then:

- groups selected positions by old class;
- detects conflicts;
- canonicalizes kept old classes into new axis classes;
- records mapping from new payload axes to old payload classes.

**Step 3: Unit-test planner if useful**

If planner complexity grows, test it in a `#[cfg(test)]` module in
`tensordynlen.rs`.

### Task 5: Implement compact payload selection

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Step 1: Add typed payload helper**

Implement:

```rust
fn select_structured_payload<T>(
    payload: Vec<T>,
    old_payload_dims: &[usize],
    old_payload_strides: &[isize],
    new_payload_dims: &[usize],
    plan: &StructuredSelectPlan,
) -> Result<Vec<T>>
where
    T: TensorElement + Copy + Zero;
```

Iterate over `product(new_payload_dims)`, decode each column-major coordinate,
map it to an old payload coordinate, fill fixed classes, compute old offset, and
copy the value.

**Step 2: Build output storage**

For f64 and Complex64 storage:

- get payload via existing payload accessors;
- call `select_structured_payload`;
- build `Storage::new_structured(data, new_payload_dims, new_strides, new_axis_classes)`;
- wrap with `TensorDynLen::from_structured_storage`.

**Step 3: Handle conflict**

If the planner reports conflict, return:

```rust
TensorDynLen::structured_zeros::<T>(kept_indices, new_axis_classes)
```

using the scalar type of the input storage.

### Task 6: Replace structured rejection in `select_indices`

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Step 1: Route storage kinds**

Make `select_indices` dispatch:

- `Dense`: existing dynamic-slice path;
- `Diagonal`: either keep the current specialized path or replace it with the
  general structured path;
- `Structured`: new structured path.

Prefer using the general structured path for diagonal too if it stays simple and
keeps tests green. Otherwise keep diagonal special-case for clarity.

**Step 2: Update rustdoc**

Update `select_indices` docs to state that structured layouts are preserved when
possible and that full input dense materialization is avoided.

### Task 7: Verification

**Files:**
- Test: `crates/tensor4all-core/tests/tensor_diag.rs`
- Test: `crates/tensor4all-core/src/defaults/tensordynlen.rs`

**Step 1: Run focused tests**

```bash
cargo test -p tensor4all-core --release structured_zeros -- --nocapture
cargo test -p tensor4all-core --release structured_select -- --nocapture
cargo test -p tensor4all-core --release diag_tensor_select -- --nocapture
```

Expected: all focused tests pass.

**Step 2: Run core diag test file**

```bash
cargo test -p tensor4all-core --release --test tensor_diag
```

Expected: all tests pass.

**Step 3: Format**

```bash
cargo fmt --all -- --check
```

Expected: no diff.

**Step 4: Optional wider check**

```bash
cargo test -p tensor4all-core --release
```

Expected: all core tests pass.

