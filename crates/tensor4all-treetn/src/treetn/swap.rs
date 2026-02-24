//! Site index swap: reorder which node holds which site index.
//!
//! Implements swapping site indices between adjacent nodes along the tree
//! so that the network reaches a target assignment (index id → node name).

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};
use petgraph::stable_graph::NodeIndex;

use tensor4all_core::{IndexLike, TensorLike};

use crate::node_name_network::NodeNameNetwork;

use super::TreeTN;

// ============================================================================
// SwapOptions
// ============================================================================

/// Options for site index swap (truncation during SVD).
///
/// When `max_rank` or `rtol` are set, the swap may introduce approximation error
/// by truncating bond dimension. Default (both `None`) allows rank growth to preserve
/// the tensor exactly.
#[derive(Debug, Clone, Default)]
pub struct SwapOptions {
    /// Maximum bond dimension after each SVD (None = no limit).
    pub max_rank: Option<usize>,
    /// Relative tolerance for singular value truncation (None = no truncation).
    pub rtol: Option<f64>,
}

// ============================================================================
// SwapStep
// ============================================================================

/// A single swap step on one edge: which index id moves in which direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapStep<V, I>
where
    I: IndexLike,
{
    /// Edge (node1, node2); canonical key is (min, max).
    pub edge: (V, V),
    /// Index id moving from the first node (min) to the second (max). None if no move.
    pub index_to_second: Option<I::Id>,
    /// Index id moving from the second node (max) to the first (min). None if no move.
    pub index_to_first: Option<I::Id>,
}

// ============================================================================
// SwapPlan
// ============================================================================

/// Swap plan: sequence of swap steps per edge to go from current to target assignment.
///
/// Used for validation and optional debugging/visualization. The actual sweep
/// uses dynamic per-step decision based on current vs target assignment.
#[derive(Debug, Clone)]
pub struct SwapPlan<V, I>
where
    I: IndexLike,
{
    /// Per-edge swap steps. Key is normalized edge (min(node_a, node_b), max(...)).
    swaps: HashMap<(V, V), Vec<SwapStep<V, I>>>,
}

fn normalize_edge<V: Ord>(a: V, b: V) -> (V, V) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

impl<V, I> SwapPlan<V, I>
where
    V: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    I: IndexLike,
    I::Id: Clone + Hash + Eq,
{
    /// Build a swap plan from current and target assignment.
    ///
    /// - `current_assignment`: every site index id in the network → its current node.
    /// - `target_assignment`: partial map; only indices to move need to be present.
    ///   Indices not in `target_assignment` stay in place.
    ///
    /// Validation:
    /// - Every key in `target_assignment` must exist in `current_assignment`.
    /// - Every target node must exist in `topology`.
    pub fn new(
        current_assignment: &HashMap<I::Id, V>,
        target_assignment: &HashMap<I::Id, V>,
        topology: &NodeNameNetwork<V>,
    ) -> Result<Self> {
        let mut swaps: HashMap<(V, V), Vec<SwapStep<V, I>>> = HashMap::new();

        for (index_id, target_node) in target_assignment {
            let current_node = current_assignment.get(index_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "target_assignment contains index id {:?} which is not in the network",
                    index_id
                )
            })?;

            if !topology.has_node(target_node) {
                return Err(anyhow::anyhow!(
                    "target node {:?} for index {:?} is not in the topology",
                    target_node,
                    index_id
                ))
                .context("SwapPlan::new: target node must exist");
            }

            if current_node == target_node {
                continue;
            }

            let from_idx = topology.node_index(current_node).ok_or_else(|| {
                anyhow::anyhow!("current node {:?} not found in topology", current_node)
            })?;
            let to_idx = topology.node_index(target_node).ok_or_else(|| {
                anyhow::anyhow!("target node {:?} not found in topology", target_node)
            })?;

            let path = topology.path_between(from_idx, to_idx).ok_or_else(|| {
                anyhow::anyhow!("no path between {:?} and {:?}", current_node, target_node)
            })?;

            let path_names: Vec<V> = path
                .iter()
                .filter_map(|ni| topology.node_name(*ni).cloned())
                .collect();

            for i in 0..path_names.len().saturating_sub(1) {
                let a = path_names[i].clone();
                let b = path_names[i + 1].clone();
                let (first, second) = normalize_edge(a.clone(), b.clone());
                let edge_key = (first.clone(), second.clone());

                let step = if a < b {
                    SwapStep {
                        edge: (a, b),
                        index_to_second: Some(index_id.clone()),
                        index_to_first: None,
                    }
                } else {
                    SwapStep {
                        edge: (b.clone(), a.clone()),
                        index_to_second: None,
                        index_to_first: Some(index_id.clone()),
                    }
                };

                swaps.entry(edge_key).or_default().push(step);
            }
        }

        Ok(Self { swaps })
    }

    /// Returns true if there is at least one swap on the given edge.
    pub fn has_swaps_at(&self, edge: &(V, V)) -> bool {
        let key = normalize_edge(edge.0.clone(), edge.1.clone());
        self.swaps.get(&key).is_some_and(|v| !v.is_empty())
    }

    /// Returns the set of edges that have at least one swap.
    pub fn edges_with_swaps(&self) -> HashSet<(V, V)> {
        self.swaps
            .iter()
            .filter(|(_, steps)| !steps.is_empty())
            .map(|(k, _)| k.clone())
            .collect()
    }
}

// ============================================================================
// Helpers: current assignment
// ============================================================================

/// Build index id → node name from a TreeTN (all site indices).
pub(crate) fn current_site_assignment<T, V>(
    treetn: &TreeTN<T, V>,
) -> HashMap<<T::Index as IndexLike>::Id, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut out: HashMap<<T::Index as IndexLike>::Id, V> = HashMap::new();
    for node_name in treetn.node_names() {
        if let Some(site_space) = treetn.site_space(&node_name) {
            for idx in site_space {
                out.insert(idx.id().to_owned(), node_name.clone());
            }
        }
    }
    out
}

// ============================================================================
// SubtreeOracle
// ============================================================================

/// Pre-computed DFS timestamps enabling O(1) "which side of edge?" queries.
///
/// For an edge (A, B): `is_target_on_a_side(A, B, target)` returns true iff
/// `target` is in the component containing A when the edge is removed.
pub(crate) struct SubtreeOracle<V> {
    in_time: HashMap<V, usize>,
    out_time: HashMap<V, usize>,
}

impl<V> SubtreeOracle<V>
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    /// Build from a tree topology rooted at `root`.
    /// DFS entry/exit timestamps are computed iteratively.
    pub(crate) fn new(topology: &NodeNameNetwork<V>, root: &V) -> Result<Self> {
        let root_idx = topology
            .node_index(root)
            .ok_or_else(|| anyhow::anyhow!("SubtreeOracle: root {:?} not in topology", root))?;

        let mut in_time: HashMap<V, usize> = HashMap::new();
        let mut out_time: HashMap<V, usize> = HashMap::new();
        let mut timer = 0usize;

        // Stack: (node_idx, parent_idx, is_exit)
        let mut stack: Vec<(NodeIndex, Option<NodeIndex>, bool)> = vec![(root_idx, None, false)];

        while let Some((node_idx, parent_idx, is_exit)) = stack.pop() {
            let name = topology
                .node_name(node_idx)
                .ok_or_else(|| anyhow::anyhow!("SubtreeOracle: node name not found"))?
                .clone();
            if is_exit {
                out_time.insert(name, timer);
                timer += 1;
            } else {
                in_time.insert(name, timer);
                timer += 1;
                // Push exit marker (processed after all children)
                stack.push((node_idx, parent_idx, true));
                // Push children (all neighbors except parent)
                let graph = topology.graph();
                for neighbor in graph.neighbors(node_idx) {
                    if Some(neighbor) != parent_idx {
                        stack.push((neighbor, Some(node_idx), false));
                    }
                }
            }
        }

        Ok(Self { in_time, out_time })
    }

    /// Returns `true` iff `target` is on the A-side of edge (A, B).
    ///
    /// A-side = the connected component containing A after the (A,B) edge is removed.
    pub(crate) fn is_target_on_a_side(&self, node_a: &V, node_b: &V, target: &V) -> bool {
        if target == node_a {
            return true;
        }
        if target == node_b {
            return false;
        }
        let in_a = match self.in_time.get(node_a) {
            Some(&t) => t,
            None => return false,
        };
        let out_a = match self.out_time.get(node_a) {
            Some(&t) => t,
            None => return false,
        };
        let in_b = match self.in_time.get(node_b) {
            Some(&t) => t,
            None => return false,
        };
        let out_b = match self.out_time.get(node_b) {
            Some(&t) => t,
            None => return false,
        };
        let in_t = match self.in_time.get(target) {
            Some(&t) => t,
            None => return false,
        };
        let out_t = match self.out_time.get(target) {
            Some(&t) => t,
            None => return false,
        };

        // Is A the ancestor of B in the DFS tree? Then A-side = NOT in subtree(B).
        if in_a <= in_b && out_b <= out_a {
            !(in_b <= in_t && out_t <= out_b)
        } else {
            // B is ancestor of A; A-side = subtree(A).
            in_a <= in_t && out_t <= out_a
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tensor4all_core::DynIndex;

    fn build_chain_topology() -> NodeNameNetwork<String> {
        let mut net = NodeNameNetwork::new();
        net.add_node("A".to_string()).unwrap();
        net.add_node("B".to_string()).unwrap();
        net.add_node("C".to_string()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net
    }

    fn build_chain_topology_with_d() -> NodeNameNetwork<String> {
        let mut net = NodeNameNetwork::new();
        net.add_node("A".to_string()).unwrap();
        net.add_node("B".to_string()).unwrap();
        net.add_node("C".to_string()).unwrap();
        net.add_node("D".to_string()).unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();
        net
    }

    #[test]
    fn test_subtree_oracle_chain() {
        // Chain A-B-C, rooted at A.
        // DFS from A: in_A=0, in_B=1, in_C=2, out_C=3, out_B=4, out_A=5
        let topo = build_chain_topology();
        let oracle = SubtreeOracle::new(&topo, &"A".to_string()).unwrap();

        // Edge (A, B): A-side = {A}, B-side = {B, C}
        assert!(oracle.is_target_on_a_side(&"A".to_string(), &"B".to_string(), &"A".to_string()));
        assert!(!oracle.is_target_on_a_side(
            &"A".to_string(),
            &"B".to_string(),
            &"B".to_string()
        ));
        assert!(!oracle.is_target_on_a_side(
            &"A".to_string(),
            &"B".to_string(),
            &"C".to_string()
        ));

        // Edge (B, C): B-side = {A, B}, C-side = {C}
        assert!(oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"A".to_string()));
        assert!(oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"B".to_string()));
        assert!(!oracle.is_target_on_a_side(
            &"B".to_string(),
            &"C".to_string(),
            &"C".to_string()
        ));

        // Reversed edge (B, A): B-side = {B, C}, A-side = {A}
        assert!(!oracle.is_target_on_a_side(
            &"B".to_string(),
            &"A".to_string(),
            &"A".to_string()
        ));
        assert!(oracle.is_target_on_a_side(&"B".to_string(), &"A".to_string(), &"B".to_string()));
        assert!(oracle.is_target_on_a_side(&"B".to_string(), &"A".to_string(), &"C".to_string()));
    }

    #[test]
    fn test_subtree_oracle_longer_chain() {
        // Chain A-B-C-D, rooted at B.
        let topo = build_chain_topology_with_d();
        let oracle = SubtreeOracle::new(&topo, &"B".to_string()).unwrap();

        // Edge (B, C): B-side = {A, B}, C-side = {C, D}
        assert!(oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"A".to_string()));
        assert!(!oracle.is_target_on_a_side(
            &"B".to_string(),
            &"C".to_string(),
            &"D".to_string()
        ));
    }

    #[test]
    fn test_subtree_oracle_unknown_root() {
        let topo = build_chain_topology();
        let res = SubtreeOracle::new(&topo, &"Z".to_string());
        assert!(res.is_err());
    }

    #[test]
    fn test_swap_plan_has_swaps_at() {
        let topo = build_chain_topology();
        let ix = DynIndex::new_dyn(2);
        let mut current = HashMap::new();
        current.insert(ix.id().to_owned(), "A".to_string());
        let mut target = HashMap::new();
        target.insert(ix.id().to_owned(), "B".to_string());

        let plan = SwapPlan::<String, DynIndex>::new(&current, &target, &topo).unwrap();
        assert!(plan.has_swaps_at(&("A".to_string(), "B".to_string())));
        assert!(!plan.has_swaps_at(&("B".to_string(), "C".to_string())));
    }

    #[test]
    fn test_swap_plan_no_move() {
        let topo = build_chain_topology();
        let mut current = HashMap::new();
        let ix = DynIndex::new_dyn(2);
        let iy = DynIndex::new_dyn(2);
        current.insert(ix.id().to_owned(), "A".to_string());
        current.insert(iy.id().to_owned(), "B".to_string());
        let target = HashMap::new();

        let plan = SwapPlan::<String, DynIndex>::new(&current, &target, &topo).unwrap();
        assert!(plan.edges_with_swaps().is_empty());
    }

    #[test]
    fn test_swap_plan_two_node_swap() {
        let topo = build_chain_topology();
        let mut current = HashMap::new();
        let ix = DynIndex::new_dyn(2);
        let iy = DynIndex::new_dyn(2);
        current.insert(ix.id().to_owned(), "A".to_string());
        current.insert(iy.id().to_owned(), "B".to_string());
        let mut target = HashMap::new();
        target.insert(ix.id().to_owned(), "B".to_string());
        target.insert(iy.id().to_owned(), "A".to_string());

        let plan = SwapPlan::<String, DynIndex>::new(&current, &target, &topo).unwrap();
        let edges = plan.edges_with_swaps();
        assert_eq!(edges.len(), 1);
        assert!(edges.contains(&("A".to_string(), "B".to_string())));
    }

    #[test]
    fn test_swap_plan_invalid_target_node() {
        let topo = build_chain_topology();
        let ix = DynIndex::new_dyn(2);
        let mut current = HashMap::new();
        current.insert(ix.id().to_owned(), "A".to_string());
        let mut target = HashMap::new();
        target.insert(ix.id().to_owned(), "Z".to_string()); // Z not in topology

        let res = SwapPlan::<String, DynIndex>::new(&current, &target, &topo);
        assert!(res.is_err());
    }

    #[test]
    fn test_swap_plan_unknown_index_in_target() {
        let topo = build_chain_topology();
        let ix = DynIndex::new_dyn(2);
        let iy = DynIndex::new_dyn(2);
        let mut current = HashMap::new();
        current.insert(ix.id().to_owned(), "A".to_string());
        let mut target = HashMap::new();
        target.insert(iy.id().to_owned(), "B".to_string()); // iy not in current (different id)

        let res = SwapPlan::<String, DynIndex>::new(&current, &target, &topo);
        assert!(res.is_err());
    }
}
