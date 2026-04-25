# Design: `split_to` chain topology fix

**Date**: 2026-04-25
**Issue**: [#78](https://github.com/tensor4all/Tensor4all.jl/issues/78)

## Problem

`TreeTN::split_to` produces a star topology instead of the requested chain
topology when splitting a single fused node into 3+ target nodes.

**Actual** (3-site fused, target chain `1--2--3`):
```
edges: [(1,3), (2,3)]  ← star
```

**Expected**:
```
edges: [(1,2), (2,3)]  ← chain
```

## Root Cause

`split_tensor_for_targets` (`transform.rs:529`) uses sequential QR to peel off
each target's site indices. When constructing `left_inds` for each QR step, it
only includes the site indices of the current target, neglecting inherited bond
indices from previous QR steps. These bonds stay on the "remaining" tensor and
accumulate on the final target node, creating a star topology.

## Fix

In `split_tensor_for_targets`, include **QR-created inherited bond** indices
in `left_inds` for each QR step. These are indices on the remaining tensor that
were not present in the original tensor (i.e., created by previous QR steps).
Original current-tree bonds (present before any QR) are excluded — they are
routed via the `boundary_indices` mechanism when target edges are present,
and should be left on the remaining tensor otherwise. This ensures each
inherited bond is forwarded to the correct intermediate target, not
accumulated on the final target.

### Code change

`crates/tensor4all-treetn/src/treetn/transform.rs`:

1. Precompute the set of all site index IDs and original tensor index IDs:
   ```rust
   let all_site_ids: HashSet<_> = partition.values().flatten().cloned().collect();
   let original_index_ids: HashSet<_> = tensor
       .external_indices()
       .iter()
       .map(|idx| idx.id().clone())
       .collect();
   ```

2. In the loop, extend `left_inds` with QR-created inherited bond indices only:
   ```rust
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

### Validation (defense in depth)

`split_to` must validate the result topology matches the target before
returning. Add after `TreeTN::from_tensors(...)`:

```rust
if !result.site_index_network().share_equivalent_site_index_network(target) {
    return Err(anyhow::anyhow!(
        "split_to: result topology does not match target: \
         expected edges {:?}, got {:?}",
        target.edges().collect::<Vec<_>>(),
        result.site_index_network().edges().collect::<Vec<_>>(),
    ));
}
```

## Test

Add topology assertion to `test_split_to_with_actual_splitting`:

```rust
assert!(
    split.site_index_network().share_equivalent_site_index_network(&split_target),
    "split_to must preserve the requested chain topology"
);
```

## Scope

- `crates/tensor4all-treetn/src/treetn/transform.rs`: fix + validation (~10 lines)
- `crates/tensor4all-treetn/src/treetn/transform/tests/mod.rs`: test assertion (~5 lines)
