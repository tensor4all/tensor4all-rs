# Issue 440 Local Naive Apply Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `ApplyOptions::naive()` full-dense apply with local exact MPO/state application.

**Architecture:** Add generic product-link and local axis-fusion support in `tensor4all-core`, then use it in a dedicated `apply_linear_operator` naive path. The new path contracts one state tensor with one MPO tensor per node, fuses each pair of state/MPO edge bonds into one product bond, and constructs the result TreeTN without full-network dense materialization.

**Tech Stack:** Rust, `tensor4all-core::TensorDynLen`, `TensorLike`, `IndexLike`, `tensor4all-treetn::TreeTN`, `cargo nextest --release`.

---

### Task 1: Add `IndexLike::product_link`

**Files:**
- Modify: `crates/tensor4all-core/src/index_like.rs`
- Modify: `crates/tensor4all-core/src/defaults/index.rs`
- Test: `crates/tensor4all-core/src/defaults/index/tests/mod.rs`

**Step 1: Write failing tests**

Add tests for product link construction:

```rust
#[test]
fn test_product_link_uses_product_dimension_and_fresh_id() {
    let a = DynIndex::new_link(2).unwrap();
    let b = DynIndex::new_link(3).unwrap();

    let product = DynIndex::product_link(&[a.clone(), b.clone()]).unwrap();

    assert_eq!(product.dim(), 6);
    assert!(product.tags().has_tag("Link"));
    assert_ne!(product.id(), a.id());
    assert_ne!(product.id(), b.id());
}

#[test]
fn test_product_link_rejects_empty_input() {
    let result = DynIndex::product_link(&[]);
    assert!(result.is_err());
}
```

If there is no existing `TagSetLike` import in the test module, add it.

**Step 2: Run tests to verify failure**

Run:

```bash
cargo nextest run --release -p tensor4all-core test_product_link
```

Expected: compile failure because `product_link` does not exist.

**Step 3: Implement trait API**

In `IndexLike`, add:

```rust
fn product_link(indices: &[Self]) -> anyhow::Result<Self>
where
    Self: Sized;
```

Update the local `TestIndex` implementation inside `index_like.rs` tests:

```rust
fn product_link(indices: &[Self]) -> anyhow::Result<Self> {
    anyhow::ensure!(!indices.is_empty(), "product_link requires at least one index");
    let dim = indices.iter().try_fold(1usize, |acc, idx| {
        acc.checked_mul(idx.dim)
            .ok_or_else(|| anyhow::anyhow!("product link dimension overflow"))
    })?;
    Ok(TestIndex { id: 9999, dim })
}
```

**Step 4: Implement `DynIndex::product_link`**

In `impl IndexLike for DynIndex`:

```rust
fn product_link(indices: &[Self]) -> anyhow::Result<Self> {
    anyhow::ensure!(!indices.is_empty(), "product_link requires at least one index");
    let dim = indices.iter().try_fold(1usize, |acc, idx| {
        acc.checked_mul(idx.dim())
            .ok_or_else(|| anyhow::anyhow!("product link dimension overflow"))
    })?;
    DynIndex::new_bond(dim)
}
```

**Step 5: Verify**

Run:

```bash
cargo nextest run --release -p tensor4all-core test_product_link
```

Expected: product link tests pass.

**Step 6: Commit**

```bash
git add crates/tensor4all-core/src/index_like.rs crates/tensor4all-core/src/defaults/index.rs crates/tensor4all-core/src/defaults/index/tests/mod.rs
git commit -m "feat: add product link index construction"
```

### Task 2: Add `TensorDynLen::fuse_indices`

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `crates/tensor4all-core/tests/tensor_unfuse.rs`

**Step 1: Write failing tests**

Add tests next to existing `unfuse_index` tests:

```rust
#[test]
fn fuse_indices_column_major_roundtrips_unfuse_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = tensor
        .fuse_indices(&[i.clone(), j.clone()], fused.clone(), LinearizationOrder::ColumnMajor)
        .unwrap();
    assert_eq!(fused_tensor.dims(), vec![6, 2]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[i, j], LinearizationOrder::ColumnMajor)
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0));
}

#[test]
fn fuse_indices_supports_non_adjacent_axes() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(4).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = tensor
        .fuse_indices(&[i.clone(), k.clone()], fused.clone(), LinearizationOrder::ColumnMajor)
        .unwrap();
    assert_eq!(fused_tensor.indices(), &[fused.clone(), j.clone()]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[i, k], LinearizationOrder::ColumnMajor)
        .unwrap()
        .permuteinds(&tensor.indices())
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0));
}
```

Also add error tests for missing index and dimension mismatch.

**Step 2: Run tests to verify failure**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_unfuse fuse_indices
```

Expected: compile failure because `fuse_indices` does not exist.

**Step 3: Implement inherent method**

Add `TensorDynLen::fuse_indices` near `unfuse_index`.

Algorithm:

1. Validate `old_indices` is non-empty.
2. Find each old index axis by ID.
3. Validate no duplicate old index IDs.
4. Validate product of old dimensions equals `new_index.dim()`.
5. Build output indices by removing all old axes and inserting `new_index` at the earliest old-axis position.
6. For each old linear element, decode column-major coordinates, encode the selected old coordinates into one fused coordinate using `encode_linear_with_order`, and write the value into the new dense buffer.

Add helper:

```rust
fn encode_linear_with_order(
    indices: &[usize],
    dims: &[usize],
    order: LinearizationOrder,
) -> Result<usize>
```

It must be the inverse of existing `decode_linear_with_order`.

**Step 4: Verify**

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_unfuse fuse_indices
```

Expected: new fuse tests pass.

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/tests/tensor_unfuse.rs
git commit -m "feat: add local tensor index fusion"
```

### Task 3: Add `TensorLike::fuse_indices`

**Files:**
- Modify: `crates/tensor4all-core/src/tensor_like.rs`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-core/src/block_tensor.rs`
- Modify: `crates/tensor4all-itensorlike/src/tensortrain.rs`

**Step 1: Add trait method**

In `TensorLike`, add after `permuteinds`:

```rust
fn fuse_indices(
    &self,
    old_indices: &[<Self as TensorIndex>::Index],
    new_index: <Self as TensorIndex>::Index,
    order: LinearizationOrder,
) -> Result<Self>;
```

**Step 2: Implement for `TensorDynLen`**

Delegate to the inherent method:

```rust
fn fuse_indices(
    &self,
    old_indices: &[DynIndex],
    new_index: DynIndex,
    order: LinearizationOrder,
) -> Result<Self> {
    TensorDynLen::fuse_indices(self, old_indices, new_index, order)
}
```

**Step 3: Implement for `BlockTensor<T>`**

Delegate to each block:

```rust
fn fuse_indices(
    &self,
    old_indices: &[Self::Index],
    new_index: Self::Index,
    order: LinearizationOrder,
) -> Result<Self> {
    let blocks: Result<Vec<T>> = self
        .blocks
        .iter()
        .map(|block| block.fuse_indices(old_indices, new_index.clone(), order))
        .collect();
    Ok(Self {
        blocks: blocks?,
        shape: self.shape,
    })
}
```

**Step 4: Implement unsupported for `TensorTrain`**

```rust
fn fuse_indices(
    &self,
    _old_indices: &[Self::Index],
    _new_index: Self::Index,
    _order: LinearizationOrder,
) -> anyhow::Result<Self> {
    anyhow::bail!("TensorTrain does not support TensorLike::fuse_indices")
}
```

**Step 5: Verify compile**

Run:

```bash
cargo check --workspace
```

Expected: workspace compiles.

**Step 6: Commit**

```bash
git add crates/tensor4all-core/src/tensor_like.rs crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/src/block_tensor.rs crates/tensor4all-itensorlike/src/tensortrain.rs
git commit -m "feat: expose tensor index fusion through TensorLike"
```

### Task 4: Implement local exact naive apply

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/apply.rs`
- Test: `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs`

**Step 1: Write small correctness tests**

Add a rank-2 two-site state and bonded identity operator test:

```rust
#[test]
fn naive_apply_identity_matches_dense_small_case() {
    let (state, sites) = build_rank_two_two_site_state();
    let operator = build_bonded_identity_operator(&sites);

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    let diff = result.to_dense().unwrap().sub(&state.to_dense().unwrap()).unwrap();
    assert!(diff.maxabs() < 1e-12);
}
```

Add a nontrivial one-site or two-site operator test if an existing helper can build one cheaply.

**Step 2: Write long no-dense regression test**

Add helpers for a long chain state and identity chain MPO with 10-15 binary sites. The test must not call `to_dense()` or `maxabs()` on a TreeTN.

```rust
#[test]
fn naive_apply_long_identity_chain_stays_local() {
    let (state, sites) = build_long_chain_state(12);
    let operator = build_chain_identity_operator(&sites);

    let result = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();

    assert_eq!(result.node_count(), state.node_count());
    assert_eq!(result.edge_count(), state.edge_count());

    for (a, b) in state.site_index_network().edges() {
        let state_edge = state.edge_between(&a, &b).unwrap();
        let result_edge = result.edge_between(&a, &b).unwrap();
        let state_dim = state.bond_index(state_edge).unwrap().dim();
        let result_dim = result.bond_index(result_edge).unwrap().dim();
        assert!(result_dim <= state_dim);
    }

    assert_sampled_evaluations_match(&state, &result);
}
```

For identity MPO link dimension 1, the product bond bound is exactly the state
bond dimension. Sample evaluation should use `all_site_indices()` and
`evaluate_at()` at a small fixed set of multi-indices.

**Step 3: Run tests to verify failure**

Run:

```bash
cargo nextest run --release -p tensor4all-treetn naive_apply
```

Expected: small tests may pass through current dense path, but long regression
should fail by exhausting memory/time or by an explicit timeout if added. If the
long test is too risky before implementation, mark it `#[ignore]` until Step 6,
then unignore before commit.

**Step 4: Implement helper dispatch**

In `apply_linear_operator`, replace generic dispatch for `ContractionMethod::Naive`:

```rust
let contracted = if options.method == ContractionMethod::Naive {
    apply_linear_operator_naive_local(&full_operator, &transformed_state, center)?
} else {
    contract(&transformed_state, full_operator.mpo(), center, contraction_options)?
};
```

**Step 5: Implement `apply_linear_operator_naive_local`**

Private helper outline:

```rust
fn apply_linear_operator_naive_local<T, V>(
    operator: &LinearOperator<T, V>,
    transformed_state: &TreeTN<T, V>,
    center: &V,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let state = transformed_state.sim_internal_inds();
    let mpo = operator.mpo().sim_internal_inds();
    anyhow::ensure!(state.same_topology(&mpo), "naive apply requires matching topologies");

    let mut tensors_by_node: HashMap<V, T> = HashMap::new();
    let mut node_names = state.node_names();
    node_names.sort();

    for node in &node_names {
        let state_tensor = state.tensor(state.node_index(node).unwrap()).unwrap();
        let mpo_tensor = mpo.tensor(mpo.node_index(node).unwrap()).unwrap();
        let local = T::contract(&[state_tensor, mpo_tensor], AllowedPairs::All)?;
        tensors_by_node.insert(node.clone(), local);
    }

    for edge in state.graph.graph().edge_indices() {
        let (a_idx, b_idx) = state.graph.graph().edge_endpoints(edge).unwrap();
        let a = state.graph.node_name(a_idx).unwrap().clone();
        let b = state.graph.node_name(b_idx).unwrap().clone();
        let state_bond = state.bond_index(edge).unwrap().clone();
        let mpo_edge = mpo.edge_between(&a, &b).unwrap();
        let mpo_bond = mpo.bond_index(mpo_edge).unwrap().clone();
        let fused = T::Index::product_link(&[state_bond.clone(), mpo_bond.clone()])?;

        for node in [&a, &b] {
            let tensor = tensors_by_node.remove(node).unwrap();
            let fused_tensor = tensor.fuse_indices(
                &[state_bond.clone(), mpo_bond.clone()],
                fused.clone(),
                LinearizationOrder::ColumnMajor,
            )?;
            tensors_by_node.insert(node.clone(), fused_tensor);
        }
    }

    let tensors = node_names
        .iter()
        .map(|node| tensors_by_node.remove(node).unwrap())
        .collect();
    let mut result = TreeTN::from_tensors(tensors, node_names)?;
    result.set_canonical_region(std::iter::once(center.clone()))?;
    Ok(result)
}
```

If direct access to private graph fields is awkward, add a small private
iterator helper inside the same module or use existing public methods such as
`node_names`, `edge_between`, and `site_index_network().edges()`.

**Step 6: Verify tests**

Run:

```bash
cargo nextest run --release -p tensor4all-treetn naive_apply
```

Expected: new apply tests pass.

**Step 7: Commit**

```bash
git add crates/tensor4all-treetn/src/operator/apply.rs crates/tensor4all-treetn/src/operator/apply/tests/mod.rs
git commit -m "fix: make naive operator apply local exact"
```

### Task 5: Remove dense naive apply dispatch and update docs

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/apply.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/contraction.rs`
- Search docs/tests for stale claims.

**Step 1: Update rustdoc**

Change `ApplyOptions::naive()` docs to say local exact apply. Do not describe it
as full dense reference behavior.

Clarify `ContractionMethod::Naive` as dense/reference if it remains for generic
TreeTN contraction tests. If possible, rename or make it private in a separate
follow-up; do not expand the public dense path in this task.

**Step 2: Search for stale docs**

Run:

```bash
rg "Naive|naive|contract_to_tensor|dense reference|full tensor" crates/tensor4all-treetn docs README.md
```

Update only claims directly affected by this change.

**Step 3: Verify docs compile for touched crates**

Run:

```bash
cargo test --doc --release -p tensor4all-core -p tensor4all-treetn
```

Expected: doctests pass.

**Step 4: Commit**

```bash
git add crates/tensor4all-treetn/src/operator/apply.rs crates/tensor4all-treetn/src/treetn/contraction.rs docs README.md
git commit -m "docs: clarify naive apply semantics"
```

Only stage docs files that were actually modified.

### Task 6: Final verification

**Files:**
- All touched files.

**Step 1: Format**

Run:

```bash
cargo fmt --all
```

Expected: no unexpected large unrelated diffs.

**Step 2: Run targeted release tests**

Run:

```bash
cargo nextest run --release -p tensor4all-core fuse_indices product_link
cargo nextest run --release -p tensor4all-treetn naive_apply
```

Expected: all targeted tests pass.

**Step 3: Run broader checks if time allows**

Run:

```bash
cargo clippy --workspace
cargo nextest run --release -p tensor4all-core -p tensor4all-treetn
```

Expected: no clippy errors and both crates pass.

**Step 4: Inspect full diff**

Run:

```bash
git status --short
git diff --stat HEAD
```

Expected: only intended files are modified. Do not include existing unrelated
`AGENTS.md` or `docs/test-reports/` changes unless the user explicitly asks.
