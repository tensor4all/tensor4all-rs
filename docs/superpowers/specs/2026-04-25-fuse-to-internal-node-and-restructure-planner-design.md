# Design: `fuse_to` internal node handling and `restructure_to` planner connectivity fix

**Date**: 2026-04-25
**Issues**: #451 (fuse_to), #452 (restructure_to) branching topology tests exposed bugs

## Problem

Tests on Y-shape branching topologies revealed two bugs sharing the same root cause:
the code operating on `SiteIndexNetwork` (site-only subgraph) where the full tree
graph is needed.

### Bug 1: `fuse_to` cannot handle internal nodes

`fuse_to` rejects tree networks that contain internal nodes — nodes whose tensors
have only bond (link) indices and no physical site indices. For example:

```
  A(site_a)
     |
  B(no site)   ← internal node: has only bonds ab, bc, bd
    / \
C(site_c) D(site_d)
```

`fuse_to` maps site-index IDs → current node names, but B has no site indices
and never appears in any target group. The validation at
`transform.rs:122-131` then rejects B with `"missing target for current node"`.

`contract_node_group` (which performs the actual tensor merging) also depends
on `SiteIndexNetwork` for connectivity checks and edge-canonicailzation ordering.
Internal nodes are invisible to both, so even if the validation were relaxed,
the contraction would fail connectivity checks.

### Bug 2: `restructure_to` planner blind to internal nodes

`build_plan` (`restructure/mod.rs:842`) uses `SiteIndexNetwork` for the
`target_nodes_span_connected_currents` connectivity check. When site-bearing nodes
are connected only through an internal node, the site-only graph shows them as
disconnected, causing `build_plan` to fall through to the `"planner placeholder
only"` error.

## Root Cause

Both bugs stem from `SiteIndexNetwork` being an incomplete graph representation.
It only tracks nodes with registered site indices, omitting internal nodes that
contain only bond indices. Operations that need the full connectivity of the
tree tensor network must use the full graph (`NamedGraph`) instead.

## Design

### Fix 1: `contract_node_group` — use full graph

`contract_node_group` (`transform.rs:167`) currently uses `self.site_index_network`
for two operations:

1. **Connectivity check** (line 200): `self.site_index_network.is_connected_subset(&node_indices)`
2. **Edge ordering** (lines 211-213): `self.site_index_network.edges_to_canonicalize(None, root_idx)`

Both must switch to the full `NamedGraph`:

- Connectivity: compute BFS from a seed node in the group on the full graph,
  verify all group members are reachable within the induced subgraph.
- Edge ordering: run BFS from the root on the full graph to build parent
  relationships, then order edges in post-order (leaves before parents).

The method already uses `self.graph.node_index(name)` for NodeIndex resolution
(line 176), so the transition to `self.graph` for all structural queries is
natural.

### Fix 2: `fuse_to` — skip internal nodes in validation, expand groups

In `fuse_to` (`transform.rs:61`):

1. **Step 3 validation** (lines 122-131): when checking that all current nodes
   are accounted for in `current_to_target`, skip nodes that have no site indices.
   These internal nodes will be implicitly included when their containing target
   group is expanded.

2. **Pre-contraction group expansion** (before step 4): for each target group's
   current node set, compute the Steiner tree in the full graph spanning those
   nodes. Include all Steiner tree nodes (including internal connector nodes)
   in the set passed to `contract_node_group`.

The Steiner tree expansion ensures that `contract_node_group` receives a
connected set that properly covers the subgraph.

### Fix 3: `restructure_to` planner — full graph connectivity

`build_plan` (`restructure/mod.rs:842`) currently builds its plan using only
`SiteIndexNetwork`:

```rust
fn build_plan<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
) -> Result<RestructurePlan<...>>
```

Change `build_plan` to accept the full TreeTN graph (or a `NodeNameNetwork`
built from it) for connectivity checks. Specifically:

- `target_nodes_span_connected_currents` (`restructure/mod.rs:230`):
  currently calls `current.is_connected_subset(&current_nodes)` on the
  `SiteIndexNetwork`. Replace with connectivity check on the full graph.

The full graph access requires threading an additional parameter through
`build_plan` and its helpers.

### Affected files

| File | Change |
|------|--------|
| `transform.rs` | `contract_node_group`: use `self.graph` for connectivity + edge ordering |
| `transform.rs` | `fuse_to`: skip internal nodes in validation; expand groups via Steiner tree |
| `restructure/mod.rs` | `build_plan`: accept full graph, pass to connectivity helpers |
| `restructure/mod.rs` | `target_nodes_span_connected_currents`: use full graph connectivity |

### Non-goals

- Does not add new public API methods
- Does not change `contract_node_group`'s signature
- Does not modify the `SiteIndexNetwork` data structure itself
- Does not add site index requirements to internal nodes

### Testing

The Y-shape branching tests added for #451/#452 exercise these paths. Specific
cases to add:

1. `fuse_to` on a Y-shape with an internal center node (no site index on B)
2. `restructure_to` on a Y-shape with an internal center node
3. `fuse_to` on a deeper tree with multiple internal connector nodes

All should produce correct contracted tensors matching dense reference.
