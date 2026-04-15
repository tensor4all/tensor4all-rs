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

Therefore, the full schedule---including per-sweep branch pruning---can be
computed **before any tensor operations**. This gives a testable, inspectable
plan.

#### Data Structures

```rust
/// A single two-site update step in the schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledSwapStep<V, Id> {
    /// The two nodes forming the edge (node_a, node_b).
    pub node_a: V,
    pub node_b: V,
    /// Site index IDs that should be on node_a's side after this step.
    pub a_side_sites: HashSet<Id>,
    /// Site index IDs that should be on node_b's side after this step.
    pub b_side_sites: HashSet<Id>,
    /// Canonical center after this step (= node_b, same as LocalUpdateStep).
    pub new_center: V,
}

/// Pre-computed, fully expanded swap schedule across all sweeps.
///
/// Constructed from topology + current/target assignments using only graph
/// operations (no tensor math). Edges where no site index needs to cross
/// are pruned, and entire sweeps are omitted once all movements are complete.
#[derive(Debug, Clone)]
pub struct SwapSchedule<V, Id> {
    /// Flattened sequence of steps (pruned, ordered by execution).
    pub steps: Vec<ScheduledSwapStep<V, Id>>,
}
```

#### Construction Algorithm

```
fn build_swap_schedule(topology, current_assignment, target_assignment, root):
    oracle = SubtreeOracle::new(topology, root)
    base_sweep = LocalUpdateSweepPlan::new(topology, root, nsite=2)

    // Simulate site index positions through sweeps
    position = copy(current_assignment)  // mutable: tracks where each index is NOW
    steps = []

    for pass in 0..max_passes:
        if all targets satisfied(position, target_assignment):
            break

        any_moved_this_pass = false

        for (node_a, node_b) in base_sweep.edges():
            // Collect site indices currently on A or B
            sites_on_a = { id | position[id] == node_a }
            sites_on_b = { id | position[id] == node_b }
            all_sites = sites_on_a | sites_on_b

            if all_sites.is_empty():
                continue  // no site indices on this edge, skip

            // Decide L/R assignment for each site index
            a_side = {}
            b_side = {}
            any_crossing = false

            for id in all_sites:
                if id in target_assignment:
                    target = target_assignment[id]
                    if oracle.is_target_on_a_side(node_a, node_b, target):
                        a_side.insert(id)
                        if id in sites_on_b:
                            any_crossing = true
                    else:
                        b_side.insert(id)
                        if id in sites_on_a:
                            any_crossing = true
                else:
                    // No target: stay where you are
                    if id in sites_on_a:
                        a_side.insert(id)
                    else:
                        b_side.insert(id)

            if !any_crossing:
                continue  // no movement needed on this edge, skip

            steps.push(ScheduledSwapStep { node_a, node_b, a_side, b_side })

            // Update simulated positions
            for id in a_side:
                position[id] = node_a
            for id in b_side:
                position[id] = node_b

            any_moved_this_pass = true

        if !any_moved_this_pass:
            break  // no progress, converged or stuck

    return SwapSchedule { steps }
```

**Key properties:**
- `max_passes` = tree diameter (sufficient for any index to traverse the full tree)
- Steps where no site index crosses the edge are omitted (pruned)
- Entire sweeps are omitted once all movements complete
- Construction is pure graph computation: O(passes x edges x moving_sites)

#### Convergence

Each full Euler-tour sweep moves every in-transit site index at least one edge
closer to its target (when the sweep visits the edge in the correct direction).
The worst case is a site index traveling the full diameter, requiring O(diameter)
sweeps. Total schedule size is O(diameter x active_edges).

### Execution

Execution follows the pre-computed schedule step by step:

```
fn execute_swap_schedule(treetn, schedule, factorize_options):
    for step in schedule.steps:
        swap_on_edge(treetn, step.node_a, step.node_b,
                     step.a_side_sites, step.b_side_sites,
                     factorize_options)
```

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
4. Factorize AB by `left_inds` with given options
5. Replace tensors and bond; update `site_index_network` (automatic via `replace_tensor`)

The 6-level priority logic and `.take(target_a_site_count)` are removed entirely.

### Modified `swap_site_indices`

```rust
pub fn swap_site_indices(
    &mut self,
    target_assignment: &HashMap<Id, V>,
    options: &SwapOptions,
) -> Result<()> {
    // 1. Build schedule (pure graph computation)
    let schedule = SwapSchedule::build(
        self.site_index_network().topology(),
        &current_site_assignment(self),
        target_assignment,
    )?;

    // 2. Canonicalize if needed
    if !self.is_canonicalized() { ... }

    // 3. Execute schedule
    for step in &schedule.steps {
        self.swap_on_edge(a_idx, b_idx,
            &step.a_side_sites, &step.b_side_sites,
            &factorize_options)?;
        self.set_canonical_region([step.new_center])?;
    }
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
| `replace_tensor()` | Reuse as-is (auto-updates `site_index_network`) |
| `replace_edge_bond()` | Reuse as-is |
| `swap_on_edge()` | Simplify: remove priority logic, accept explicit L/R sets |
| `SwapPlan` (old) | Remove: replaced by `SwapSchedule` |
| `SwapStep` (old) | Remove: replaced by `ScheduledSwapStep` |

## Testing Strategy

### Schedule construction (unit tests, no tensor ops)

1. **Chain**: 3-node chain, swap endpoints. Verify step count = 2 (one sweep),
   correct L/R assignments, middle node acts as transit.
2. **Y-shape**: s0: L0->L1, s1: L1->L0. Verify 3 steps across 2 sweeps,
   edge (C,L2) never appears.
3. **Star**: center + 4 leaves, rotate site indices. Verify O(2) sweeps.
4. **Binary tree**: deeper topology, verify O(depth) sweeps.
5. **No-op**: target == current. Verify empty schedule.
6. **Partial assignment**: only some indices have targets. Verify unassigned
   indices stay in place.

### Execution (integration tests)

7. **Re-enable `test_swap_y_shape` and `test_swap_y_shape_c64`** (currently ignored).
8. **Chain regression**: existing chain tests must still pass.
9. **Tensor correctness**: contract full network before and after swap,
   verify the tensor is the same (up to truncation tolerance).

## Worked Example: Y-Shape

```
Topology:       L0 -- C -- L1
                      |
                      L2

Current:  s0 @ L0, s1 @ L1, s2 @ L2
Target:   s0 -> L1, s1 -> L0

Root = C (minimum node name)
Euler tour edges: [(C,L0), (L0,C), (C,L1), (L1,C), (C,L2), (L2,C)]
```

**Schedule construction (simulation):**

Notation: edge (A, B) means node_a=A, node_b=B. "A-side" = component
containing A after removing the edge.

**Pass 1:**

| Edge (A, B) | Sites on edge | Decision | Emit? | Positions after |
|-------------|---------------|----------|-------|-----------------|
| (C, L0) | s0@L0 | s0 target=L1, L1 on A-side(C) -> s0 to A(C). Crossing. | Yes: a:{s0}, b:{} | s0@C |
| (L0, C) | s0@C | s0 target=L1, L1 on B-side(C) -> s0 to B(C). Already on C, no crossing. | Skip | s0@C |
| (C, L1) | s0@C, s1@L1 | s0 target=L1 -> B-side(L1). s1 target=L0 -> A-side(C). Both cross. | Yes: a:{s1}, b:{s0} | s0@L1, s1@C |
| (L1, C) | s1@C | s1 target=L0, L0 on B-side(C) -> s1 to B(C). Already on C, no crossing. | Skip | s1@C |
| (C, L2) | s1@C, s2@L2 | s1 target=L0 -> A-side(C), already on C. s2 no target -> stays on L2. No crossing. | Skip | unchanged |
| (L2, C) | s1@C | Same analysis, no crossing. | Skip | unchanged |

**Pass 2:**

| Edge (A, B) | Sites on edge | Decision | Emit? | Positions after |
|-------------|---------------|----------|-------|-----------------|
| (C, L0) | s1@C | s1 target=L0 -> B-side(L0). Crossing. | Yes: a:{}, b:{s1} | s1@L0 |

All targets satisfied (s0@L1, s1@L0). Stop.

**Final schedule: 3 steps.** Edge (C,L2) never appears.

```
steps = [
  { (C, L0), a:{s0}, b:{},   center: L0 },
  { (C, L1), a:{s1}, b:{s0}, center: L1 },
  { (C, L0), a:{},   b:{s1}, center: L0 },
]
```
