# General Structured Contraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement structured contraction and contraction AD so dense, diagonal, and general structured tensors keep compact storage and sound output axis classes.

**Architecture:** Add a metadata planner that computes union-find roots over operand payload classes, then execute contraction over compact payload tensors. Refactor `TensorDynLen` AD so tracked leaves own compact payload `EagerTensor`s with layout metadata; contraction AD runs on compact payloads and returns structured gradients with the same layout as the primal.

**Tech Stack:** Rust, `tensor4all-core`, `tensor4all-tensorbackend::Storage`, tenferro `EagerTensor`, public tenferro eager einsum for payload execution, `anyhow`, release-mode cargo tests.

---

## Prerequisites

- Read `README.md`.
- Read `docs/plans/2026-04-22-issue-434-structured-tensor-design.md`.
- Read `docs/plans/2026-04-23-general-structured-contraction-design.md`.
- Run targeted tests in release mode.
- Keep docs and code comments in English.

## Task 1: Add Metadata Planner Tests

**Files:**
- Create: `crates/tensor4all-core/src/defaults/structured_contraction.rs`
- Modify: `crates/tensor4all-core/src/defaults/mod.rs`

**Step 1: Write failing unit tests**

Create a `#[cfg(test)]` module in `structured_contraction.rs` with tests for:

```rust
#[test]
fn plans_diag_diag_partial_as_diag_output() {
    let operands = vec![
        OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
        OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
    ];
    let spec = StructuredContractionSpec {
        input_labels: vec![vec![0, 1], vec![1, 2]],
        output_labels: vec![0, 2],
        retained_labels: Default::default(),
    };

    let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
    assert_eq!(plan.output_axis_classes, vec![0, 0]);
    assert_eq!(plan.output_payload_roots.len(), 1);
}

#[test]
fn plans_general_structured_output_classes() {
    let operands = vec![
        OperandLayout::new(vec![2, 2, 3], vec![0, 0, 1]).unwrap(),
        OperandLayout::new(vec![3, 5, 5], vec![0, 1, 1]).unwrap(),
    ];
    let spec = StructuredContractionSpec {
        input_labels: vec![vec![0, 1, 2], vec![2, 3, 4]],
        output_labels: vec![0, 1, 3, 4],
        retained_labels: Default::default(),
    };

    let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
    assert_eq!(plan.output_axis_classes, vec![0, 0, 1, 1]);
    assert_eq!(plan.output_payload_dims, vec![2, 5]);
}

#[test]
fn detects_payload_repeated_labels() {
    let operands = vec![
        OperandLayout::new(vec![3, 3], vec![0, 1]).unwrap(),
        OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
    ];
    let spec = StructuredContractionSpec {
        input_labels: vec![vec![0, 1], vec![0, 1]],
        output_labels: vec![],
        retained_labels: Default::default(),
    };

    let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
    assert!(plan.operand_plans[0].has_repeated_roots());
}
```

**Step 2: Run tests to verify failure**

```bash
cargo test --release -p tensor4all-core structured_contraction::tests -- --nocapture
```

Expected: compile failure because the planner types do not exist.

**Step 3: Implement planner data types**

Add private types:

```rust
pub(crate) struct OperandLayout {
    pub(crate) logical_dims: Vec<usize>,
    pub(crate) axis_classes: Vec<usize>,
}

pub(crate) struct StructuredContractionSpec {
    pub(crate) input_labels: Vec<Vec<usize>>,
    pub(crate) output_labels: Vec<usize>,
    pub(crate) retained_labels: HashSet<usize>,
}

pub(crate) struct StructuredContractionPlan {
    pub(crate) operand_plans: Vec<OperandPayloadPlan>,
    pub(crate) output_axis_classes: Vec<usize>,
    pub(crate) output_payload_roots: Vec<usize>,
    pub(crate) output_payload_dims: Vec<usize>,
}
```

Implement the union-find algorithm from the design doc. Validate rank and
dimension mismatches with `anyhow::Result`.

**Step 4: Run tests**

```bash
cargo test --release -p tensor4all-core structured_contraction::tests -- --nocapture
```

Expected: PASS.

## Task 2: Add Compact Payload Normalization

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/structured_contraction.rs`
- Test: same file

**Step 1: Write failing tests for repeated payload roots**

Add a test that normalizes `roots = [0, 0]` for a 2x2 payload and verifies the
payload becomes rank 1 with diagonal values.

```rust
#[test]
fn normalizes_repeated_payload_roots_by_extracting_diagonal() {
    let payload = NativeTensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let (normalized, roots) =
        normalize_payload_for_roots(&payload, &[0, 0]).unwrap();

    assert_eq!(normalized.shape(), &[2]);
    assert_eq!(normalized.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    assert_eq!(roots, vec![0]);
}
```

**Step 2: Run test**

```bash
cargo test --release -p tensor4all-core normalizes_repeated_payload_roots_by_extracting_diagonal
```

Expected: FAIL.

**Step 3: Implement normalization**

Implement the old tenferro algorithm:

- find the first duplicate root pair;
- build a unary repeated-label einsum such as `aa->a`;
- call public `einsum_native_tensors`;
- remove the duplicate root;
- repeat until roots are unique.

Do not call strict binary/GEMM lowering directly.

**Step 4: Run tests**

```bash
cargo test --release -p tensor4all-core structured_contraction::tests -- --nocapture
```

Expected: PASS.

## Task 3: Add Storage Payload Native Helpers

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/structured_contraction.rs`
- Modify if needed: `crates/tensor4all-tensorbackend/src/storage.rs`
- Test: `crates/tensor4all-core/src/defaults/structured_contraction.rs`

**Step 1: Write failing tests**

Test that a `Storage::new_structured` payload can be converted to a compact
`NativeTensor` without logical dense materialization, and rebuilt as structured
storage after payload einsum.

**Step 2: Implement helpers**

Add private helpers:

```rust
fn storage_payload_native(storage: &Storage) -> Result<NativeTensor>;

fn storage_from_payload_native(
    payload: NativeTensor,
    output_payload_dims: &[usize],
    output_axis_classes: Vec<usize>,
) -> Result<Storage>;
```

Use `payload_f64_col_major_vec`, `payload_c64_col_major_vec`,
`Storage::new_structured`, and compact column-major strides.

**Step 3: Run tests**

```bash
cargo test --release -p tensor4all-core structured_contraction::tests -- --nocapture
```

Expected: PASS.

## Task 4: Refactor TensorDynLen AD To Track Compact Payloads

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `crates/tensor4all-core/tests/tensor_native_ad.rs`
- Test: `crates/tensor4all-core/tests/tensor_diag.rs`

**Step 1: Write failing structured AD tests**

Add tests that currently fail because `enable_grad()` uses `as_native()`:

```rust
#[test]
fn general_structured_grad_preserves_input_axis_classes() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let storage = Storage::new_structured(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        vec![0, 1, 0],
    )
    .map(std::sync::Arc::new)
    .unwrap();
    let x = TensorDynLen::from_storage(vec![i.clone(), j.clone(), k.clone()], storage)
        .unwrap()
        .enable_grad();
    let ones = TensorDynLen::from_dense(
        vec![i, j, k],
        vec![1.0; 12],
    )
    .unwrap();

    let loss = contract_multi(&[&x, &ones], AllowedPairs::All).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.storage().axis_classes(), &[0, 1, 0]);
    assert_eq!(grad.storage().storage_kind(), StorageKind::Structured);
    assert_eq!(grad.storage().payload_f64_col_major_vec().unwrap(), vec![1.0; 6]);
}
```

Also add a diagonal test for a tracked partial contraction:

```rust
#[test]
fn tracked_diag_partial_contraction_preserves_diag_result_and_grad() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![2.0, 3.0, 5.0])
        .unwrap()
        .enable_grad();
    let b = diag_tensor_dyn_len(vec![j, k.clone()], vec![7.0, 11.0, 13.0]).unwrap();

    let c = a.contract(&b);
    assert_eq!(c.storage().storage_kind(), StorageKind::Diagonal);

    let ones = diag_tensor_dyn_len(vec![i, k], vec![1.0, 1.0, 1.0]).unwrap();
    let loss = contract_multi(&[&c, &ones], AllowedPairs::All).unwrap();
    loss.backward().unwrap();

    let grad = a.grad().unwrap().unwrap();
    assert_eq!(grad.storage().storage_kind(), StorageKind::Diagonal);
    assert_eq!(grad.storage().payload_f64_col_major_vec().unwrap(), vec![7.0, 11.0, 13.0]);
}
```

**Step 2: Run tests to verify failure**

```bash
cargo test --release -p tensor4all-core --test tensor_native_ad general_structured_grad_preserves_input_axis_classes
cargo test --release -p tensor4all-core --test tensor_diag tracked_diag_partial_contraction_preserves_diag_result_and_grad
```

Expected: at least the general structured test fails because gradients are
materialized as dense storage.

**Step 3: Add structured AD value**

Replace the dense-only AD cache with a compact payload AD value. Keep a dense
materialization cache for non-AD value operations.

```rust
struct StructuredAdValue {
    payload: EagerTensor<CpuBackend>,
    payload_dims: Vec<usize>,
    axis_classes: Vec<usize>,
}
```

`enable_grad()` must:

- build a compact payload native tensor from `self.storage`;
- call `EagerTensor::requires_grad_in(payload_native, default_eager_ctx())`;
- store layout metadata next to that payload leaf;
- not call `as_native()`.

`grad()` must:

- read `ad.payload.grad()`;
- rebuild `Storage` from the gradient payload and original layout;
- return `TensorDynLen::from_storage(self.indices.clone(), Arc::new(storage))`.

`tracks_grad()`, `clear_grad()`, `backward()`, and `detach()` must use the
structured AD value when present.

**Step 4: Run AD tests**

```bash
cargo test --release -p tensor4all-core --test tensor_native_ad
cargo test --release -p tensor4all-core --test tensor_diag tracked_diag_partial_contraction_preserves_diag_result_and_grad
```

Expected: PASS after later contraction integration; intermediate compile/test
failures are expected until Task 5 is complete.

## Task 5: Integrate Compact Structured Contraction

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/defaults/contract.rs`
- Modify: `crates/tensor4all-core/src/defaults/structured_contraction.rs`
- Test: `crates/tensor4all-core/tests/tensor_diag.rs`
- Test: `crates/tensor4all-core/src/defaults/contract/tests/mod.rs`

**Step 1: Add failing value tests**

Add tests for:

- pairwise diagonal/diagonal partial contraction returns diagonal storage;
- `contract_multi` diagonal/diagonal partial contraction returns diagonal storage;
- general structured contraction returns `StorageKind::Structured` with expected
  `axis_classes = [0, 0, 1, 1]`;
- dense materialization of the structured result matches the old dense path.

**Step 2: Build compact payload specs**

For pairwise contraction, map `prepare_contraction` / `prepare_contraction_pairs`
into `StructuredContractionSpec`.

For `contract_multi`, map the existing internal ids into labels:

- output labels are existing `output`;
- retained labels come from future options; initially empty;
- input labels are `ixs`.

**Step 3: Execute compact payload contraction**

For each operand:

- get compact payload native tensor from storage or structured AD value;
- normalize repeated roots;
- collect normalized root labels.

Call:

- `eager_einsum_ad` when any operand tracks grad;
- `einsum_native_tensors` when no operand tracks grad.

Wrap the output payload into `Storage::new_structured` using the planned output
payload dims and axis classes. For AD, also store the output compact
`EagerTensor` in the result's structured AD value.

**Step 4: Remove dense structured contraction fallback from tracked paths**

`contract`, `tensordot`, `outer_product`, and `contract_multi_impl` should not
call `as_native()` merely to execute a structured contraction when a compact
plan is available.

Dense materialization is allowed only for explicitly unsupported operations or
for readback APIs.

**Step 5: Run tests**

```bash
cargo test --release -p tensor4all-core --test tensor_diag
cargo test --release -p tensor4all-core defaults::contract::tests
cargo test --release -p tensor4all-core --test tensor_native_ad
```

Expected: PASS.

## Task 6: Add Retained-Index Option Surface

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/contract.rs`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: public exports as needed in `crates/tensor4all-core/src/lib.rs`
- Test: `crates/tensor4all-core/src/defaults/contract/tests/mod.rs`

**Step 1: Add API tests**

Add tests for issue #436:

- `contract_multi_with_options` keeps a shared batch index;
- retained labels are not reduced;
- retained labels still union structured classes;
- default options preserve current behavior.

**Step 2: Add option types**

Add:

```rust
pub struct ContractOptions {
    pub retained_indices: Vec<DynIndex>,
}

pub struct ContractMultiOptions {
    pub retained_indices: Vec<DynIndex>,
}
```

Add `contract_with_options`, `tensordot_with_options` if needed, and
`contract_multi_with_options`. Existing APIs delegate with empty retained sets.

**Step 3: Wire retained labels into planner**

Map retained `DynIndex` ids to internal contraction labels and pass them into
`StructuredContractionSpec.retained_labels`.

**Step 4: Run tests**

```bash
cargo test --release -p tensor4all-core defaults::contract::tests
cargo test --release -p tensor4all-core --test tensor_diag
```

Expected: PASS.

## Task 7: Full Verification

**Files:**
- All touched files

**Step 1: Format**

```bash
cargo fmt --all
```

Expected: no formatting diff after running.

**Step 2: Run focused suite**

```bash
cargo nextest run --release -p tensor4all-core --test tensor_diag
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
cargo nextest run --release -p tensor4all-core defaults::contract::tests
```

Expected: all pass.

**Step 3: Run crate suite**

```bash
cargo nextest run --release -p tensor4all-core
cargo clippy -p tensor4all-core --all-targets
git diff --check
```

Expected: all pass.

**Step 4: Update docs if public API changed**

If retained-index options are public, update rustdoc examples and any stale
README/user-guide claims. Then run:

```bash
cargo test --doc --release -p tensor4all-core
```

Expected: PASS.

