# split_to chain topology fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `split_tensor_for_targets` to produce correct chain topology by including inherited bond indices in `left_inds`; add topology validation to `split_to`.

**Architecture:** In `split_tensor_for_targets`, precompute the set of all site index IDs. In each QR peeling step, extend `left_inds` with indices that are not site indices (inherited bonds). Add defense-in-depth validation in `split_to`.

**Tech Stack:** Rust, tensor4all-core (TensorLike, IndexLike, FactorizeOptions, Canonical)

---

### Task 1: Fix `split_tensor_for_targets` — include inherited bond indices in `left_inds`

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform.rs:568-597`

- [ ] **Step 1: Precompute `all_site_ids` and extend `left_inds` with inherited bonds**

After `partition` is fully built (after the `if let Some(boundary_indices)` block, currently at line 568 `}`), add `all_site_ids` and `original_index_ids` computation:

```rust
        }

        // Collect all site index IDs to distinguish inherited bond indices
        let all_site_ids: HashSet<_> = partition.values().flatten().cloned().collect();

        // Record original tensor index IDs to distinguish inherited bonds
        // (created by QR during the split) from original current-tree bonds.
        let original_index_ids: HashSet<_> = tensor
            .external_indices()
            .iter()
            .map(|idx| idx.id().clone())
            .collect();

        // Sort target names for deterministic processing
```

Then in the loop body, replace `let left_inds: Vec<_> = ...` block with:

```rust
            // Find site indices for this target
            let mut left_inds: Vec<_> = remaining_tensor
                .external_indices()
                .iter()
                .filter(|idx| site_ids_for_target.contains(idx.id()))
                .cloned()
                .collect();

            // Include inherited bond indices created by previous QR steps.
            // Exclude original current-tree bonds — they are routed via the
            // boundary_indices mechanism when target edges are present.
            left_inds.extend(
                remaining_tensor
                    .external_indices()
                    .iter()
                    .filter(|idx| {
                        let id = idx.id();
                        !all_site_ids.contains(id) && !original_index_ids.contains(id)
                    })
                    .cloned(),
            );
```

- [ ] **Step 2: Verify compilation**

```bash
cargo build -p tensor4all-treetn
```

Expected: compiles without errors.

- [ ] **Step 3: Run existing split tests to check for regressions**

```bash
cargo test -p tensor4all-treetn -- transform
```

Expected: all transform tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/transform.rs
git commit -m "fix(treetn): include inherited bond indices in split_tensor_for_targets left_inds"
```

---

### Task 2: Add topology validation in `split_to`

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform.rs:428-429`

- [ ] **Step 1: Insert topology check after `from_tensors`**

After the `from_tensors` call (current line 428), insert validation:

```rust
        let result = TreeTN::<T, TargetV>::from_tensors(tensors, names)
            .context("split_to: failed to build result TreeTN")?;

        // Validate the result topology matches the target
        if !result
            .site_index_network()
            .share_equivalent_site_index_network(target)
        {
            return Err(anyhow::anyhow!(
                "split_to: result topology does not match target: \
                 expected edges {:?}, got {:?}",
                target
                    .edges()
                    .map(|(a, b)| (a.clone(), b.clone()))
                    .collect::<Vec<_>>(),
                result
                    .site_index_network()
                    .edges()
                    .map(|(a, b)| (a.clone(), b.clone()))
                    .collect::<Vec<_>>(),
            ));
        }

        // Step 5: Phase 2 - Optional truncation sweep
```

Note: `SiteIndexNetwork::edges` returns `impl Iterator<Item = (NodeName, NodeName)>` where `NodeName` is `&V`. The `.map(|(a, b)| (a.clone(), b.clone()))` is needed to satisfy `Clone + Debug` for `Vec`.

- [ ] **Step 2: Update Step comment numbering**

Renumber the subsequent Step comment from `// Step 5: Phase 2 - Optional truncation sweep` to `// Step 6: Phase 2 - Optional truncation sweep`.

- [ ] **Step 3: Verify compilation**

```bash
cargo build -p tensor4all-treetn
```

Expected: compiles.

- [ ] **Step 4: Run transform tests**

```bash
cargo test -p tensor4all-treetn -- transform
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/transform.rs
git commit -m "feat(treetn): add topology validation in split_to"
```

---

### Task 3: Add topology assertion to existing split test

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform/tests/mod.rs:263`

- [ ] **Step 1: Add topology assertion**

After `assert_eq!(split.node_count(), 3);` (line 263), add:

```rust
    assert_eq!(split.node_count(), 3);
    assert!(
        split
            .site_index_network()
            .share_equivalent_site_index_network(&split_target),
        "split_to must preserve the requested chain topology, not silently produce a star/tree"
    );
```

- [ ] **Step 2: Run the test to confirm it passes with the fix**

```bash
cargo test -p tensor4all-treetn test_split_to_with_actual_splitting
```

Expected: test passes.

- [ ] **Step 3: Run all transform tests**

```bash
cargo test -p tensor4all-treetn -- transform
```

Expected: all pass.

- [ ] **Step 4: Run full test suite**

```bash
cargo test -p tensor4all-treetn
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/transform/tests/mod.rs
git commit -m "test(treetn): add topology assertion to test_split_to_with_actual_splitting"
```
