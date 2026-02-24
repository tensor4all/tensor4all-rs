//! Site index swap: reorder which node holds which site index.
//!
//! Implements swapping site indices between adjacent nodes along the tree
//! so that the network reaches a target assignment (index id → node name).

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};
use petgraph::stable_graph::NodeIndex;

use tensor4all_core::{AllowedPairs, Canonical, FactorizeOptions, IndexLike, TensorLike};

use crate::node_name_network::NodeNameNetwork;

use super::localupdate::{LocalUpdateStep, LocalUpdater};
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
// Helpers: current assignment, is_target_on_a_side
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

/// True iff `target_node` is on the A-side of the edge (A, B) in the tree.
/// A-side = A and all nodes reachable from A without going through B.
fn is_target_on_a_side<V>(
    topology: &NodeNameNetwork<V>,
    node_a: &V,
    node_b: &V,
    target_node: &V,
) -> bool
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    if target_node == node_a {
        return true;
    }
    if target_node == node_b {
        return false;
    }
    let a_idx = match topology.node_index(node_a) {
        Some(i) => i,
        None => return false,
    };
    let b_idx = match topology.node_index(node_b) {
        Some(i) => i,
        None => return false,
    };
    let t_idx = match topology.node_index(target_node) {
        Some(i) => i,
        None => return false,
    };
    let path: Vec<petgraph::stable_graph::NodeIndex> = match topology.path_between(a_idx, t_idx) {
        Some(p) => p,
        None => return false,
    };
    // Path from A to T: path[0]=A, path[1]=first neighbor. If that neighbor is B, T is on B-side.
    !(path.len() >= 2 && path[1] == b_idx)
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

// ============================================================================
// SwapUpdater
// ============================================================================

/// LocalUpdater that reorders site indices toward a target assignment.
pub struct SwapUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Target: index id → node name. Partial; indices not present stay in place.
    pub target_assignment: HashMap<<T::Index as IndexLike>::Id, V>,
    /// Maximum bond dimension after each SVD (None = no limit).
    pub max_rank: Option<usize>,
    /// Relative tolerance for singular value truncation (None = no truncation).
    pub rtol: Option<f64>,
}

impl<T, V> SwapUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a swap updater with the given target assignment and options.
    pub fn new(
        target_assignment: HashMap<<T::Index as IndexLike>::Id, V>,
        options: &SwapOptions,
    ) -> Self {
        Self {
            target_assignment,
            max_rank: options.max_rank,
            rtol: options.rtol,
        }
    }
}

impl<T, V> LocalUpdater<T, V> for SwapUpdater<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    fn update(
        &mut self,
        mut subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTN<T, V>> {
        if step.nodes.len() != 2 {
            return Err(anyhow::anyhow!(
                "SwapUpdater requires exactly 2 nodes, got {}",
                step.nodes.len()
            ));
        }

        let node_a = &step.nodes[0];
        let node_b = &step.nodes[1];
        let topology = full_treetn.site_index_network().topology();

        let idx_a = subtree
            .node_index(node_a)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node_a))?;
        let idx_b = subtree
            .node_index(node_b)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node_b))?;

        let edge_ab = subtree
            .edge_between(node_a, node_b)
            .ok_or_else(|| anyhow::anyhow!("No edge between {:?} and {:?}", node_a, node_b))?;
        let bond_ab = subtree
            .bond_index(edge_ab)
            .ok_or_else(|| anyhow::anyhow!("Bond index not found"))?
            .clone();

        let tensor_a = subtree.tensor(idx_a).unwrap();
        let tensor_b = subtree.tensor(idx_b).unwrap();

        let site_at_a: Vec<T::Index> = tensor_a
            .external_indices()
            .iter()
            .filter(|idx| idx.id() != bond_ab.id())
            .cloned()
            .collect();
        let site_at_b: Vec<T::Index> = tensor_b
            .external_indices()
            .iter()
            .filter(|idx| idx.id() != bond_ab.id())
            .cloned()
            .collect();

        // Which index ids should go to the left factor (A'). Do not include bond_ab.
        let mut left_id_set: HashSet<<T::Index as IndexLike>::Id> = HashSet::new();
        for idx in &site_at_a {
            let target = self.target_assignment.get(idx.id());
            let stay_on_a = target.is_none_or(|t| is_target_on_a_side(topology, node_a, node_b, t));
            if stay_on_a {
                left_id_set.insert(idx.id().to_owned());
            }
        }
        for idx in &site_at_b {
            let target = self.target_assignment.get(idx.id());
            let move_to_a =
                target.is_some_and(|t| is_target_on_a_side(topology, node_a, node_b, t));
            if move_to_a {
                left_id_set.insert(idx.id().to_owned());
            }
        }

        let tensor_ab = T::contract(&[tensor_a, tensor_b], AllowedPairs::All)
            .context("SwapUpdater: contract A and B")?;

        // Build left_inds from tensor_ab's indices so factorize/unfold finds them (same refs/clones).
        let ab_indices = tensor_ab.external_indices();
        let mut left_inds: Vec<T::Index> = ab_indices
            .iter()
            .filter(|idx| left_id_set.contains(idx.id()))
            .cloned()
            .collect();

        // SVD requires 0 < left_len < rank. If left_inds is empty or all we use a 1-1 split.
        let all_len = ab_indices.len();
        let swap_result = if left_inds.is_empty() {
            left_inds = vec![ab_indices[0].clone()];
            true
        } else if left_inds.len() == all_len {
            left_inds = vec![ab_indices[0].clone()];
            false
        } else {
            false
        };

        let mut options = FactorizeOptions::svd().with_canonical(Canonical::Left);
        if let Some(mr) = self.max_rank {
            options = options.with_max_rank(mr);
        }
        if let Some(rtol) = self.rtol {
            options = options.with_rtol(rtol);
        }

        let factorize_result = tensor_ab
            .factorize(&left_inds, &options)
            .map_err(|e| anyhow::anyhow!("SwapUpdater: factorize failed: {}", e))?;

        let (mut new_tensor_a, mut new_tensor_b) = if swap_result {
            (factorize_result.right, factorize_result.left)
        } else {
            (factorize_result.left, factorize_result.right)
        };
        let new_bond = factorize_result.bond_index;

        // In the full graph, node_a/node_b may have other edges (to nodes not in this step).
        // replace_subtree will put these tensors back into the full graph, which requires
        // every connection index to be present. Attach other bonds via outer product with ones.
        let full_idx_a = full_treetn
            .node_index(node_a)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not in full treetn", node_a))?;
        let full_idx_b = full_treetn
            .node_index(node_b)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not in full treetn", node_b))?;
        let other_bonds_a: Vec<T::Index> = full_treetn
            .edges_for_node(full_idx_a)
            .iter()
            .filter_map(|(edge_idx, _)| full_treetn.bond_index(*edge_idx).cloned())
            .filter(|b| b.id() != bond_ab.id())
            .collect();
        let other_bonds_b: Vec<T::Index> = full_treetn
            .edges_for_node(full_idx_b)
            .iter()
            .filter_map(|(edge_idx, _)| full_treetn.bond_index(*edge_idx).cloned())
            .filter(|b| b.id() != bond_ab.id())
            .collect();
        // Factorize may have kept some other bonds on one side; only attach bonds not already present.
        for other in &other_bonds_a {
            let has = new_tensor_a
                .external_indices()
                .iter()
                .any(|i| i.id() == other.id());
            if !has {
                new_tensor_a = new_tensor_a
                    .outer_product(&T::ones(std::slice::from_ref(other))?)
                    .context("SwapUpdater: outer_product ones for node_a other bond")?;
            }
        }
        for other in &other_bonds_b {
            let has = new_tensor_b
                .external_indices()
                .iter()
                .any(|i| i.id() == other.id());
            if !has {
                new_tensor_b = new_tensor_b
                    .outer_product(&T::ones(std::slice::from_ref(other))?)
                    .context("SwapUpdater: outer_product ones for node_b other bond")?;
            }
        }

        subtree.replace_edge_bond(edge_ab, new_bond.clone())?;
        subtree.replace_tensor(idx_a, new_tensor_a)?;
        subtree.replace_tensor(idx_b, new_tensor_b)?;

        subtree.set_ortho_towards(&new_bond, Some(step.new_center.clone()));
        subtree.set_canonical_region([step.new_center.clone()])?;

        Ok(subtree)
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
