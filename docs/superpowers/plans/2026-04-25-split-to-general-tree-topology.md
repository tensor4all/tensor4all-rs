# split_to general tree topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Generalize `split_tensor_for_targets` from chain-only to arbitrary tree topologies using post-order tree decomposition, and add tests for branching shapes.

**Architecture:** Replace sequential QR (sorted target order) with post-order QR decomposition using `petgraph::visit::DfsPostOrder` via `NodeNameNetwork::post_order_dfs`. Each current node's fragment sub-topology is extracted from the target `SiteIndexNetwork`. The algorithm mirrors `factorize_tensor_to_treetn` in `decompose.rs`.

**Tech Stack:** Rust, petgraph (via NodeNameNetwork/SiteIndexNetwork), tensor4all-core (TensorLike, FactorizeOptions)

---

### Task 1: Extract fragment sub-topology helper

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform.rs`

Goal: Given a current node name and the full target `SiteIndexNetwork`, build a sub-`SiteIndexNetwork`
containing only the fragments whose `current` maps to that node. This reuses existing `edges()`,
`site_space()`, and `node_names()` methods.

**Signature:**
```rust
fn sub_target_for_current_node<'a, T, CurrentV, TargetV>(
    target: &'a SiteIndexNetwork<TargetV, T::Index>,
    site_to_target: &HashMap<<T::Index as IndexLike>::Id, TargetV>,
    site_to_current: &HashMap<<T::Index as IndexLike>::Id, CurrentV>,
    current_node_name: &CurrentV,
) -> Result<SiteIndexNetwork<TargetV, T::Index>>
```

- [ ] **Step 1:** Collect fragment nodes for the current node. Iterate `target.node_names()`, filter those whose sites all map to `current_node_name` via `site_to_current`.
- [ ] **Step 2:** Clone each qualifying node with its site space into a new `SiteIndexNetwork`.
- [ ] **Step 3:** Add edges from `target.edges()` where BOTH endpoints are in the sub-target.
- [ ] **Step 4:** Return the sub-network.
- [ ] **Step 5:** Unit test: given a 3-node target (A, B spanning 2 currents, C), extract sub-target for each current. Verify node count and edges.

---

### Task 2: Rewrite `split_tensor_for_targets` using post-order tree decomposition

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform.rs` (lines ~529–628)

Goal: Replace sequential QR loop with post-order decomposition borrowed from
`factorize_tensor_to_treetn_with_root_impl` (decompose.rs:146–364), adapted to
work on fragments within a single tensor.

**Key references to reuse:**
- `NodeNameNetwork::post_order_dfs` (petgraph `DfsPostOrder`) — traversal order
- `TensorLike::factorize` — QR factorization
- `index_ops::common_inds` — identify the bond between left and right tensors

**New signature:**
```rust
fn split_tensor_for_targets<TargetV>(
    &self,
    tensor: &T,
    site_to_target: &HashMap<<T::Index as IndexLike>::Id, TargetV>,
    boundary_indices: Option<&HashMap<TargetV, HashSet<<T::Index as IndexLike>::Id>>>,
    fragment_target: &SiteIndexNetwork<TargetV, T::Index>,  // NEW: sub-topology
) -> Result<Vec<(TargetV, T)>>
```

**Algorithm (replaces lines ~583–628):**

1. If `fragment_target.node_count() <= 1` → identity (return tensor as-is)
2. Choose a root:
   - If `boundary_indices` exists, pick the target with the most boundary indices
   - Otherwise pick the target with highest degree (most edges) in the sub-topology
   - Fallback: `target_names.sort()` first element
3. Compute post-order via `fragment_target.post_order_dfs(&root)`
   (delegates to `NodeNameNetwork::post_order_dfs` → petgraph `DfsPostOrder`)
4. Build `children_by_parent` map:
   - Initialize empty `HashMap<TargetV, Vec<TargetV>>`
   - For each edge `(a, b)` in `fragment_target.edges()`: determine parent-child via traversal order (the one later in post-order is the child, or use the BFS tree from root). Use petgraph's `graph.neighbors()` on the internal graph.
5. Process nodes in post-order (leaves first, skip root, root gets remaining tensor):
   - `left_inds` = fragment's site indices + bonds to already-processed children (stored in `child_bonds: HashMap<TargetV, T::Index>`)
   - If `boundary_indices` has specified bonds for this target, include them in `left_inds` too
   - QR factorization
   - Extract the new parent-bond via `common_inds(&left, &right)[0]`
   - Store left as fragment tensor, right becomes new `remaining_tensor`
   - Record parent-bond for this node in `child_bonds`
6. Root fragment gets the final `remaining_tensor`

- [ ] **Step 1:** Implement the new body (steps 1-6 above).
- [ ] **Step 2:** Update the calling code in `split_to` (around line ~404) to pass `fragment_target`.
- [ ] **Step 3:** Remove the old `all_site_ids` / `original_index_ids` logic (no longer needed — post-order naturally handles bond routing).
- [ ] **Step 4:** Run all existing `transform` tests. All 19 must pass.
- [ ] **Step 5:** Run `test_restructure_to_split_then_fuse_mixed_case`. Must pass.
- [ ] **Step 6:** Commit.

---

### Task 3: Add branching topology tests for `split_to`

**Files:**
- Create: `crates/tensor4all-treetn/src/treetn/transform/tests/mod.rs` (append ~100 lines)

Goal: Verify `split_to` produces correct topology for non-chain tree shapes.

**Test helpers needed:**
- A single fused node containing 4 site indices (s_A, s_B, s_C, s_D)

**Tests:**

- [ ] **Step 1: Y-shape (star with 3 leaves)**

```
Target: A--B  B--C  B--D   (A,C,D are leaves, B is center)
Fused tensor: [s_A, s_B, s_C, s_D]
Split result must have edges: [(A,B), (B,C), (B,D)]
```

Assert: `node_count() == 4`, `share_equivalent_site_index_network`, contraction matches.

- [ ] **Step 2: 4-node chain (keeps working as before)**

```
Target: A--B  B--C  C--D
Fused tensor: [s_A, s_B, s_C, s_D]
Split result must have edges: [(A,B), (B,C), (C,D)]
```

- [ ] **Step 3: Two branches from center (3 nodes)**

```
Target: A--B  B--C
Fused tensor: [s_A, s_B, s_C]
```

(Covered by existing `test_split_to_with_actual_splitting`, just re-verify)

- [ ] **Step 4: 2-node split (regression)**

```
Target: A--B
Fused tensor: [s_A, s_B]
```

(Covered by existing tests)

- [ ] **Step 5: Split across original bond with Y-shape target**

```
Current: 2 nodes [x0,x1]-[y0,y1]
Target: X(x0)--Y(x1,y0)--Z(sub)  and Y--W(y1)
```

Verifies boundary_indices routing works with branching.

- [ ] **Step 6: Run all transform tests. All must pass.**

---

### Task 4: Clean up and final verification

- [ ] **Step 1:** Remove `all_site_ids` and `original_index_ids` from `split_tensor_for_targets` if still present.
- [ ] **Step 2:** Run full `tensor4all-treetn` test suite:
  ```bash
  cargo test -p tensor4all-treetn
  ```
  All must pass.
- [ ] **Step 3:** Run `cargo fmt --all --check`.
- [ ] **Step 4:** Commit.
