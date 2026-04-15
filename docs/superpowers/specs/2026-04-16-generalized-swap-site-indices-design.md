# Generalized `swap_site_indices` for Non-Chain Tree TNs

**Date:** 2026-04-16
**Issue:** tensor4all/tensor4all-rs#253
**Status:** Design

## Problem

`swap_site_indices` works for chain (linear) tree TNs but fails on branching
topologies (Y-shape, star, binary tree). The root cause is the **site-count
preservation** invariant: each node must retain exactly the same number of site
indices after every two-site update. On a chain this is natural, but on a
Y-shape the center node C has 0 site indices, so an index traveling from L0 to
L1 cannot transit through C.

## Design

Drop site-count preservation. At each two-site update on edge (A, B), assign
every site index to the A-side or B-side based solely on its target location
(determined by `SubtreeOracle`). Indices not in `target_assignment` stay on
their current side.

### Pre-Computed Swap Schedule

The entire multi-sweep execution is deterministic given:
- tree topology (fixed)
- current site assignment
- target site assignment
- sweep order (Euler tour from a chosen root)

Therefore, the full schedule---including per-sweep branch pruning and
canonical-center transport---can be computed **before any tensor operations**.
This gives a testable, inspectable plan.

#### Canonical Center Transport

When Euler-tour steps are pruned, the canonical center may end up far from the
next swap edge. Each swap step requires the center to be at `node_a` or
`node_b`. If it is not, we must **transport** the center along the unique tree
path before the swap.

Transport uses `sweep_edge(src, dst)`:
1. QR-factorize tensor at `src` (no truncation) -> Q stays at src, R absorbed
   into dst
2. `src` becomes isometry, `dst` becomes the new center
3. Repeat along each edge of the path

Transport is exact (no truncation, no approximation error). Only swap steps
may truncate (via `SwapOptions`).

#### Data Structures

```rust
/// A single two-site update step in the schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledSwapStep<V, Id> {
    /// Path to transport canonical center before the swap.
    /// Empty if center is already at node_a or node_b.
    /// Entries: [current_center, ..., target_node] where target_node
    /// is one of {node_a, node_b}. Transport calls sweep_edge on each
    /// consecutive pair (exact QR, no truncation).
    pub transport_path: Vec<V>,
    /// The two nodes forming the edge.
    pub node_a: V,
    pub node_b: V,
    /// Site index IDs that should be on node_a's side after this step.
    pub a_side_sites: HashSet<Id>,
    /// Site index IDs that should be on node_b's side after this step.
    pub b_side_sites: HashSet<Id>,
}

/// Pre-computed, fully expanded swap schedule across all sweeps.
///
/// Constructed from topology + current/target assignments + root choice
/// using only graph operations (no tensor math). Edges where no site
/// index needs to cross are pruned, and entire sweeps are omitted once
/// all movements are complete.
#[derive(Debug, Clone)]
pub struct SwapSchedule<V, Id> {
    /// The root used for canonicalization and Euler tour.
    pub root: V,
    /// Flattened sequence of steps (pruned, ordered by execution).
    pub steps: Vec<ScheduledSwapStep<V, Id>>,
}
```

After each swap step, the canonical center is always at `node_b`
(`Canonical::Left` factorization makes `node_a` an isometry and places
the norm on `node_b`).

#### Construction Algorithm

```
fn build(topology, current_assignment, target_assignment, root):
    oracle = SubtreeOracle::new(topology, root)
    base_sweep = LocalUpdateSweepPlan::new(topology, root, nsite=2)
    diameter = tree_diameter(topology)

    // Simulate site index positions through sweeps
    position = copy(current_assignment)
    center = root   // canonical center after initial canonicalization
    steps = []

    for pass in 0..diameter:
        if all targets satisfied(position, target_assignment):
            break

        any_moved_this_pass = false

        for (node_a, node_b) in base_sweep.edges():
            // Collect site indices currently on A or B
            sites_on_a = { id | position[id] == node_a }
            sites_on_b = { id | position[id] == node_b }
            all_sites = sites_on_a | sites_on_b

            if all_sites.is_empty():
                continue

            // Decide L/R assignment
            a_side = {}
            b_side = {}
            any_crossing = false

            for id in all_sites:
                if id in target_assignment:
                    target = target_assignment[id]
                    if oracle.is_target_on_a_side(node_a, node_b, target):
                        a_side.insert(id)
                        if id in sites_on_b: any_crossing = true
                    else:
                        b_side.insert(id)
                        if id in sites_on_a: any_crossing = true
                else:
                    // No target: stay on current side
                    if id in sites_on_a: a_side.insert(id)
                    else:                b_side.insert(id)

            if !any_crossing:
                continue

            // Compute transport path (center must be at node_a or node_b)
            transport_path = if center == node_a || center == node_b {
                vec![]
            } else {
                tree_path(topology, center, node_a)
                // Always transport to node_a (consistent with Euler tour:
                // node_a is "from" direction, node_b is "to" direction)
            }

            steps.push(ScheduledSwapStep {
                transport_path,
                node_a, node_b,
                a_side_sites: a_side,
                b_side_sites: b_side,
            })

            // After swap: center = node_b (Canonical::Left)
            center = node_b

            // Update simulated positions
            for id in a_side: position[id] = node_a
            for id in b_side: position[id] = node_b

            any_moved_this_pass = true

        if !any_moved_this_pass:
            break

    return SwapSchedule { root, steps }
```

**Key properties:**
- `max_passes` = tree diameter (sufficient for any index to traverse the full tree)
- Steps where no site index crosses the edge are omitted (pruned)
- Transport paths bridge the gap when pruning skips Euler-tour steps
- Entire sweeps are omitted once all movements complete
- Construction is pure graph computation: O(passes x edges x moving_sites)

#### Convergence

Each full Euler-tour sweep moves every in-transit site index at least one edge
closer to its target (when the sweep visits the edge in the correct direction).
The worst case is a site index traveling the full diameter, requiring O(diameter)
sweeps. Total schedule size is O(diameter x active_edges).

### Execution

```
fn execute(treetn, schedule, swap_options):
    // 1. Canonicalize to the schedule root
    treetn.canonicalize_mut(iter::once(schedule.root), default_options)

    // 2. Build factorize options for swap steps (may truncate)
    swap_factopts = FactorizeOptions::svd().with_canonical(Left)
    if swap_options.max_rank: swap_factopts.with_max_rank(...)
    if swap_options.rtol:     swap_factopts.with_rtol(...)

    // 3. Transport factorize options (exact QR, never truncate)
    transport_factopts = FactorizeOptions::svd().with_canonical(Left)

    // 4. Execute each step
    for step in schedule.steps:
        // Transport canonical center to swap edge
        for i in 0..step.transport_path.len() - 1:
            treetn.sweep_edge(
                transport_path[i], transport_path[i+1],
                transport_factopts, "swap_transport")

        // Perform the two-site swap
        treetn.swap_on_edge(
            step.node_a, step.node_b,
            step.a_side_sites, step.b_side_sites,
            swap_factopts)

        // Center is now at node_b
        treetn.set_canonical_region([step.node_b])
```

**Why always canonicalize to root first**: The schedule assumes the center
starts at `root`. If the network is already canonicalized at a different node,
the transport paths would be wrong. Always re-canonicalizing to `root` is
cheap (one QR sweep) and ensures correctness.

### Modified `swap_on_edge`

The current `swap_on_edge` computes L/R assignment internally using a 6-level
priority system with site-count preservation. The new version takes explicit
`a_side_sites` and `b_side_sites` parameters:

```rust
pub(crate) fn swap_on_edge(
    &mut self,
    node_a_idx: NodeIndex,
    node_b_idx: NodeIndex,
    a_side_sites: &HashSet<Id>,   // NEW: explicit assignment
    b_side_sites: &HashSet<Id>,   // NEW: explicit assignment
    factorize_options: &FactorizeOptions,
) -> Result<()>
```

**Algorithm (simplified from current):**
1. Get bond between A and B; collect structural bonds of A and B
2. Contract tensors A and B into AB
3. `left_inds` = structural bonds of A + indices whose ID is in `a_side_sites`
4. Factorize AB by `left_inds` with given options (Canonical::Left)
5. Replace tensors and bond; `site_index_network` updates automatically via
   `replace_tensor`

The 6-level priority logic and `.take(target_a_site_count)` are removed
entirely.

**Canonical form correctness**: After factorize with `Canonical::Left`, A is
an isometry and B holds the norm. The bond A-B points toward B via
`set_edge_ortho_towards`. Bonds from A's other neighbors are unaffected
(their tensors did not change, and they were already isometric). This holds
regardless of whether the center was at A or B before the swap.

### Modified `swap_site_indices`

```rust
pub fn swap_site_indices(
    &mut self,
    target_assignment: &HashMap<Id, V>,
    options: &SwapOptions,
) -> Result<()> {
    // 1. Build schedule (pure graph computation, no tensor ops)
    let root = self.node_names().into_iter().min()?;
    let schedule = SwapSchedule::build(
        self.site_index_network().topology(),
        &current_site_assignment(self),
        target_assignment,
        &root,
    )?;

    // 2. Always canonicalize to schedule root
    self.canonicalize_mut(iter::once(root), default_options)?;

    // 3. Execute schedule (transport + swap for each step)
    schedule.execute(self, options)?;

    Ok(())
}
```

## Component Reuse

| Component | Status |
|---|---|
| `SubtreeOracle` | Reuse as-is (O(1) side queries) |
| `LocalUpdateSweepPlan` | Reuse for base sweep order generation |
| `SwapOptions` | Reuse as-is |
| `current_site_assignment()` | Reuse as-is |
| `sweep_edge()` | Reuse as-is for canonical center transport |
| `replace_tensor()` | Reuse as-is (auto-updates `site_index_network`) |
| `replace_edge_bond()` | Reuse as-is |
| `swap_on_edge()` | Simplify: remove priority logic, accept explicit L/R sets |
| `SwapPlan` (old) | Remove: replaced by `SwapSchedule` |
| `SwapStep` (old) | Remove: replaced by `ScheduledSwapStep` |

## Testing Strategy

### Schedule construction (unit tests, no tensor ops)

1. **Chain 3-node**: swap endpoints. Verify 2 steps (one sweep), correct L/R
   assignments, middle node as transit. Transport paths are empty (full Euler
   tour has no gaps for a chain).
2. **Y-shape**: s0: L0->L1, s1: L1->L0. Verify 3 steps, transport paths
   `[L0,C]` and `[L1,C]`, edge (C,L2) never appears.
3. **Star 4-leaf**: center + 4 leaves, rotate site indices. Verify transport
   paths are correct between non-adjacent leaves.
4. **Binary tree**: deeper topology, verify O(depth) sweeps and transport
   path lengths.
5. **No-op**: target == current. Verify empty schedule.
6. **Partial assignment**: only some indices targeted. Verify unassigned
   indices stay in place and don't generate steps.
7. **Multi-site node**: node starts with 2+ site indices, some move, some
   stay. Verify correct a_side/b_side split.
8. **Transit node 0->N->0**: internal node has 0 site indices, temporarily
   holds N during transit, returns to 0. Verify schedule reflects this.

### Execution (integration tests)

9. **Re-enable `test_swap_y_shape` and `test_swap_y_shape_c64`** (currently
   `#[ignore]`).
10. **Chain regression**: existing chain tests must still pass unchanged.
11. **Tensor correctness**: contract full network before and after swap,
    verify the dense tensor is the same (up to truncation tolerance).
12. **Canonical form check**: after execution, verify the network is in
    valid canonical form (`is_canonicalized() == true`).

## Worked Example: Y-Shape

```
Topology:       L0 -- C -- L1
                      |
                      L2

Current:  s0 @ L0, s1 @ L1, s2 @ L2
Target:   s0 -> L1, s1 -> L0

Root = C (minimum node name)
Euler tour edges: [(C,L0), (L0,C), (C,L1), (L1,C), (C,L2), (L2,C)]
Initial canonical center: C
```

### Schedule construction (simulation)

Notation: edge (A, B) means node_a=A, node_b=B. "A-side" = component
containing A after removing the edge.

**Pass 1:**

| Edge (A, B) | Sites on edge | Decision | Emit? | Positions after |
|-------------|---------------|----------|-------|-----------------|
| (C, L0) | s0@L0 | s0 target=L1, L1 on A-side(C) -> s0 to A(C). Crossing. | Yes | s0@C |
| (L0, C) | s0@C | s0 on B(C), target on B-side(C). No crossing. | Skip | s0@C |
| (C, L1) | s0@C, s1@L1 | s0 target=L1 -> B(L1). s1 target=L0 -> A(C). Both cross. | Yes | s0@L1, s1@C |
| (L1, C) | s1@C | s1 on B(C), target on B-side(C). No crossing. | Skip | s1@C |
| (C, L2) | s1@C, s2@L2 | s1 on A(C), target on A-side. s2 stays. No crossing. | Skip | unchanged |
| (L2, C) | s1@C | s1 on B(C), target on B-side(C). No crossing. | Skip | unchanged |

**Pass 2:**

| Edge (A, B) | Sites on edge | Decision | Emit? | Positions after |
|-------------|---------------|----------|-------|-----------------|
| (C, L0) | s1@C | s1 target=L0 -> B(L0). Crossing. | Yes | s1@L0 |

All targets satisfied. Stop.

### Center tracking and transport paths

| Step | Center before | Swap edge | Transport needed? | Transport path | Center after |
|------|--------------|-----------|-------------------|----------------|-------------|
| 1 | C | (C, L0) | No (C = node_a) | [] | L0 |
| 2 | L0 | (C, L1) | Yes (L0 not in {C, L1}) | [L0, C] | L1 |
| 3 | L1 | (C, L0) | Yes (L1 not in {C, L0}) | [L1, C] | L0 |

### Final schedule

```
steps = [
  { transport: [],       swap: (C, L0), a:{s0}, b:{}   },  // center: C -> L0
  { transport: [L0, C],  swap: (C, L1), a:{s1}, b:{s0} },  // center: L0 -> C -> L1
  { transport: [L1, C],  swap: (C, L0), a:{},   b:{s1} },  // center: L1 -> C -> L0
]
```

3 swap steps, 2 transport steps (2 QR ops total). Edge (C,L2) never appears.

### Execution trace

1. **Canonicalize** to root C (center = C)
2. **Step 1**: no transport. Swap (C, L0): contract C,L0 -> factorize. s0 moves
   to C. Center -> L0.
3. **Step 2**: transport L0->C (one `sweep_edge`). Swap (C, L1): contract
   C,L1 -> factorize. s0 to L1, s1 to C. Center -> L1.
4. **Step 3**: transport L1->C (one `sweep_edge`). Swap (C, L0): contract
   C,L0 -> factorize. s1 to L0. Center -> L0.
5. **Done**: s0@L1, s1@L0, s2@L2. Network is in valid canonical form with
   center at L0.
