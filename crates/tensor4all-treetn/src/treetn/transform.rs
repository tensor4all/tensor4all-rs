//! Structural transformation operations for TreeTN.
//!
//! This module provides methods to transform a TreeTN's structure:
//! - [`fuse_to`](TreeTN::fuse_to): Merge adjacent nodes to match a target structure
//! - [`split_to`](TreeTN::split_to): Split nodes to match a target structure

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};
use petgraph::stable_graph::NodeIndex;

use tensor4all_core::index::{DynId, Index, NoSymmSpace, Symmetry};
use tensor4all_core::{IndexLike, factorize, Canonical, FactorizeAlg, FactorizeOptions, TensorDynLen};

use super::TreeTN;
use crate::options::SplitOptions;
use crate::site_index_network::SiteIndexNetwork;

impl<I, V> TreeTN<I, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    Symm: Clone + Symmetry + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Fuse (merge) adjacent nodes to match the target structure.
    ///
    /// This operation contracts adjacent nodes that should be merged according to
    /// the target `SiteIndexNetwork`. The target structure must be a "coarsening"
    /// of the current structure: each target node should contain the site indices
    /// of one or more adjacent current nodes.
    ///
    /// # Algorithm
    ///
    /// 1. Compare current structure with target structure
    /// 2. Map each current node to its target node (by matching site indices)
    /// 3. For each group of current nodes mapping to the same target node:
    ///    - Contract all nodes in the group into a single node
    /// 4. Build the new TreeTN with the fused structure
    ///
    /// # Arguments
    /// * `target` - The target `SiteIndexNetwork` defining the desired structure
    ///
    /// # Returns
    /// A new TreeTN with the fused structure, or an error if:
    /// - The target structure is incompatible with the current structure
    /// - Nodes to be fused are not connected
    ///
    /// # Properties
    /// - **Bond dimension**: Unchanged (pure contraction, no truncation)
    /// - **Commutative**: Non-overlapping groups can be merged in any order
    ///
    /// # Example
    /// ```text
    /// Before: x1_1---x2_1---x1_2---x2_2---x1_3---x2_3  (6 nodes)
    /// After:  {x1_1,x2_1}---{x1_2,x2_2}---{x1_3,x2_3}  (3 nodes)
    /// ```
    pub fn fuse_to<TargetV>(
        &self,
        target: &SiteIndexNetwork<TargetV, I>,
    ) -> Result<TreeTN<I, TargetV>>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // Step 1: Build a mapping from site index ID to current node name
        let mut site_to_current_node: HashMap<I::Id, V> = HashMap::new();
        for current_node_name in self.node_names() {
            if let Some(site_space) = self.site_space(&current_node_name) {
                for site_idx in site_space {
                    site_to_current_node.insert(site_idx.id.clone(), current_node_name.clone());
                }
            }
        }

        // Step 2: For each target node, find which current nodes should be merged
        // Map: target node name -> set of current node names
        let mut target_to_current: HashMap<TargetV, HashSet<V>> = HashMap::new();

        for target_node_name in target.node_names() {
            let target_site_space = target.site_space(target_node_name).ok_or_else(|| {
                anyhow::anyhow!("Target node {:?} has no site space", target_node_name)
            })?;

            let mut current_nodes_for_target: HashSet<V> = HashSet::new();
            for target_site_idx in target_site_space {
                if let Some(current_node) = site_to_current_node.get(&target_site_idx.id) {
                    current_nodes_for_target.insert(current_node.clone());
                }
            }

            if current_nodes_for_target.is_empty() {
                return Err(anyhow::anyhow!(
                    "Target node {:?} has site indices not found in current TreeTN",
                    target_node_name
                ))
                .context("fuse_to: incompatible target structure");
            }

            target_to_current.insert(target_node_name.clone(), current_nodes_for_target);
        }

        // Step 3: Validate that every current node maps to exactly one target node
        let mut current_to_target: HashMap<V, TargetV> = HashMap::new();
        for (target_name, current_nodes) in &target_to_current {
            for current_node in current_nodes {
                if let Some(existing_target) = current_to_target.get(current_node) {
                    return Err(anyhow::anyhow!(
                        "Current node {:?} maps to multiple target nodes: {:?} and {:?}",
                        current_node,
                        existing_target,
                        target_name
                    ))
                    .context("fuse_to: ambiguous mapping");
                }
                current_to_target.insert(current_node.clone(), target_name.clone());
            }
        }

        // Check all current nodes are accounted for
        for current_name in self.node_names() {
            if !current_to_target.contains_key(&current_name) {
                return Err(anyhow::anyhow!(
                    "Current node {:?} has no corresponding target node",
                    current_name
                ))
                .context("fuse_to: missing target for current node");
            }
        }

        // Step 4: For each target node, contract all its current nodes into one tensor
        let mut result_tensors: HashMap<TargetV, TensorDynLen<I::Id, I::Symm>> = HashMap::new();

        for (target_name, current_nodes) in &target_to_current {
            let contracted = self.contract_node_group(current_nodes).with_context(|| {
                format!(
                    "fuse_to: failed to contract nodes for target {:?}",
                    target_name
                )
            })?;
            result_tensors.insert(target_name.clone(), contracted);
        }

        // Step 5: Build the new TreeTN
        // Sort target node names for deterministic ordering
        let mut target_names: Vec<TargetV> = target.node_names().into_iter().cloned().collect();
        target_names.sort();

        let tensors: Vec<TensorDynLen<I::Id, I::Symm>> = target_names
            .iter()
            .map(|name| result_tensors.remove(name).unwrap())
            .collect();

        let result = TreeTN::<Id, Symm, TargetV>::from_tensors(tensors, target_names)
            .context("fuse_to: failed to build result TreeTN")?;

        Ok(result)
    }

    /// Contract a group of nodes into a single tensor.
    ///
    /// The nodes must form a connected subtree in the current TreeTN.
    /// Contracts all internal bonds (bonds between nodes in the group),
    /// keeping external bonds and site indices.
    fn contract_node_group(&self, nodes: &HashSet<V>) -> Result<TensorDynLen<I::Id, I::Symm>>
    where
        V: Ord,
    {
        if nodes.is_empty() {
            return Err(anyhow::anyhow!("Cannot contract empty node group"));
        }

        // Convert node names to NodeIndex
        let node_indices: HashSet<NodeIndex> = nodes
            .iter()
            .filter_map(|name| self.graph.node_index(name))
            .collect();

        if node_indices.len() != nodes.len() {
            return Err(anyhow::anyhow!(
                "Some nodes not found in graph: expected {} nodes, found {}",
                nodes.len(),
                node_indices.len()
            ));
        }

        // Single node case: just clone the tensor
        if nodes.len() == 1 {
            let node_name = nodes.iter().next().unwrap();
            let node_idx = self.graph.node_index(node_name).unwrap();
            return self
                .tensor(node_idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name));
        }

        // Validate connectivity
        if !self.site_index_network.is_connected_subset(&node_indices) {
            return Err(anyhow::anyhow!(
                "Nodes to contract do not form a connected subtree"
            ));
        }

        // Pick a root (smallest node name for determinism)
        let root_name = nodes.iter().min().unwrap();
        let root_idx = self.graph.node_index(root_name).unwrap();

        // Get edges within the group, ordered from leaves to root
        let edges = self
            .site_index_network
            .edges_to_canonicalize(None, root_idx);

        // Filter to only edges within our group
        let internal_edges: Vec<(NodeIndex, NodeIndex)> = edges
            .iter()
            .filter(|(from, to)| node_indices.contains(from) && node_indices.contains(to))
            .cloned()
            .collect();

        // Initialize with cloned tensors
        let mut tensors: HashMap<NodeIndex, TensorDynLen<I::Id, I::Symm>> = node_indices
            .iter()
            .filter_map(|&idx| self.tensor(idx).cloned().map(|t| (idx, t)))
            .collect();

        // Contract along each internal edge (from leaves to root)
        for (from, to) in internal_edges {
            let from_tensor = tensors
                .remove(&from)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", from))?;
            let to_tensor = tensors
                .remove(&to)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", to))?;

            // Find the bond index for this edge
            let edge = self
                .graph
                .graph()
                .find_edge(from, to)
                .or_else(|| self.graph.graph().find_edge(to, from))
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?}", from, to))?;

            let bond_idx = self
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))?
                .clone();

            // Contract
            let contracted = to_tensor
                .tensordot(&from_tensor, &[(bond_idx.clone(), bond_idx)])
                .context("Failed to contract tensors")?;

            tensors.insert(to, contracted);
        }

        // The root tensor is the result
        tensors
            .remove(&root_idx)
            .ok_or_else(|| anyhow::anyhow!("Contraction produced no result at root"))
    }

    /// Split nodes to match the target structure.
    ///
    /// This operation splits nodes that contain site indices belonging to multiple
    /// target nodes. The target structure must be a "refinement" of the current
    /// structure: each current node's site indices should map to one or more
    /// target nodes.
    ///
    /// # Algorithm (Two-Phase Approach)
    ///
    /// **Phase 1: Exact factorization (no truncation)**
    /// 1. Build mapping: site index ID -> target node name
    /// 2. For each current node, check if its site indices map to multiple target nodes
    /// 3. If so, split the node using QR factorization
    /// 4. Repeat until all nodes match the target structure
    ///
    /// **Phase 2: Truncation sweep (optional)**
    /// If `options.final_sweep` is true, perform a truncation sweep to optimize
    /// bond dimensions globally.
    ///
    /// # Arguments
    /// * `target` - The target `SiteIndexNetwork` defining the desired structure
    /// * `options` - Options controlling truncation and final sweep
    ///
    /// # Returns
    /// A new TreeTN with the split structure, or an error if:
    /// - The target structure is incompatible with the current structure
    /// - Factorization fails
    ///
    /// # Properties
    /// - **Bond dimension**: May increase during split, controlled by truncation
    /// - **Exact (Phase 1)**: Without truncation, represents the same tensor
    ///
    /// # Example
    /// ```text
    /// Before: {x1_1,x2_1}---{x1_2,x2_2}---{x1_3,x2_3}  (3 nodes, fused)
    /// After:  x1_1---x2_1---x1_2---x2_2---x1_3---x2_3  (6 nodes, interleaved)
    /// ```
    pub fn split_to<TargetV>(
        &self,
        target: &SiteIndexNetwork<TargetV, I>,
        options: &SplitOptions,
    ) -> Result<TreeTN<I, TargetV>>
    where
        I::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync,
        I::Symm: Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync,
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // Step 1: Build mapping from site index ID to target node name
        let mut site_to_target: HashMap<I::Id, TargetV> = HashMap::new();
        for target_node_name in target.node_names() {
            if let Some(site_space) = target.site_space(target_node_name) {
                for site_idx in site_space {
                    site_to_target.insert(site_idx.id.clone(), target_node_name.clone());
                }
            }
        }

        // Step 2: For each current node, determine which target nodes it maps to
        // and validate the mapping
        let mut current_to_targets: HashMap<V, HashSet<TargetV>> = HashMap::new();
        for current_node_name in self.node_names() {
            if let Some(site_space) = self.site_space(&current_node_name) {
                let mut targets_for_node: HashSet<TargetV> = HashSet::new();
                for site_idx in site_space {
                    if let Some(target_name) = site_to_target.get(&site_idx.id) {
                        targets_for_node.insert(target_name.clone());
                    } else {
                        return Err(anyhow::anyhow!(
                            "Site index {:?} in current node {:?} has no corresponding target node",
                            site_idx.id,
                            current_node_name
                        ))
                        .context("split_to: incompatible target structure");
                    }
                }
                current_to_targets.insert(current_node_name.clone(), targets_for_node);
            }
        }

        // Step 3: Phase 1 - Split all nodes that need splitting
        // Collect all resulting tensors with their target node names
        let mut result_tensors: Vec<(TargetV, TensorDynLen<I::Id, I::Symm>)> = Vec::new();

        for current_node_name in self.node_names() {
            let node_idx = self
                .node_index(&current_node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", current_node_name))?;
            let tensor = self
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for {:?}", current_node_name))?;

            let targets_for_node = current_to_targets.get(&current_node_name).ok_or_else(|| {
                anyhow::anyhow!("No target mapping for node {:?}", current_node_name)
            })?;

            if targets_for_node.len() == 1 {
                // No split needed - just relabel
                let target_name = targets_for_node.iter().next().unwrap().clone();
                result_tensors.push((target_name, tensor.clone()));
            } else {
                // Need to split this node
                let split_tensors = self
                    .split_tensor_for_targets(tensor, &site_to_target)
                    .with_context(|| {
                        format!("split_to: failed to split node {:?}", current_node_name)
                    })?;
                result_tensors.extend(split_tensors);
            }
        }

        // Step 4: Build the result TreeTN with target node names
        // Sort by target name for deterministic ordering
        result_tensors.sort_by(|(a, _), (b, _)| a.cmp(b));

        let names: Vec<TargetV> = result_tensors.iter().map(|(name, _)| name.clone()).collect();
        let tensors: Vec<TensorDynLen<I::Id, I::Symm>> =
            result_tensors.into_iter().map(|(_, t)| t).collect();

        let result = TreeTN::<Id, Symm, TargetV>::from_tensors(tensors, names)
            .context("split_to: failed to build result TreeTN")?;

        // Step 5: Phase 2 - Optional truncation sweep
        if options.final_sweep {
            // Find a center node for truncation
            let center = result.node_names().into_iter().min().ok_or_else(|| {
                anyhow::anyhow!("split_to: no nodes in result for truncation sweep")
            })?;

            let truncation_options = crate::TruncationOptions {
                form: options.form,
                rtol: options.rtol,
                max_rank: options.max_rank,
            };

            return result
                .truncate([center], truncation_options)
                .context("split_to: truncation sweep failed");
        }

        Ok(result)
    }

    /// Split a tensor into multiple tensors, one for each target node.
    ///
    /// This uses QR factorization to iteratively separate site indices
    /// belonging to different target nodes.
    ///
    /// Returns a vector of (target_name, tensor) pairs.
    fn split_tensor_for_targets<TargetV>(
        &self,
        tensor: &TensorDynLen<I::Id, I::Symm>,
        site_to_target: &HashMap<I::Id, TargetV>,
    ) -> Result<Vec<(TargetV, TensorDynLen<I::Id, I::Symm>)>>
    where
        I::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync,
        I::Symm: Clone + Symmetry + From<NoSymmSpace> + std::fmt::Debug + Send + Sync,
        TargetV: Clone + Hash + Eq + Ord + std::fmt::Debug,
    {
        // Group tensor's site indices by their target node
        let mut partition: HashMap<TargetV, HashSet<I::Id>> = HashMap::new();
        for idx in &tensor.indices {
            if let Some(target_name) = site_to_target.get(&idx.id) {
                partition
                    .entry(target_name.clone())
                    .or_default()
                    .insert(idx.id.clone());
            }
            // Note: bond indices (not in site_to_target) are handled by factorize
        }

        // Sort target names for deterministic processing
        let mut target_names: Vec<TargetV> = partition.keys().cloned().collect();
        target_names.sort();

        if target_names.len() <= 1 {
            // Should not happen if called correctly, but handle gracefully
            let target_name = target_names
                .first()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No site indices found in tensor"))?;
            return Ok(vec![(target_name, tensor.clone())]);
        }

        // Split iteratively: separate first target's indices, then next, etc.
        let mut remaining_tensor = tensor.clone();
        let mut result: Vec<(TargetV, TensorDynLen<I::Id, I::Symm>)> = Vec::new();

        // Process all but the last target (the last one gets the remaining tensor)
        for target_name in target_names.iter().take(target_names.len() - 1) {
            let site_ids_for_target = partition.get(target_name).unwrap();

            // Find the actual Index objects for these site IDs
            let left_inds: Vec<_> = remaining_tensor
                .indices
                .iter()
                .filter(|idx| site_ids_for_target.contains(&idx.id))
                .cloned()
                .collect();

            if left_inds.is_empty() {
                continue;
            }

            // Factorize: separate these site indices
            let factorize_options = FactorizeOptions {
                alg: FactorizeAlg::QR,
                canonical: Canonical::Left,
                rtol: None,
                max_rank: None,
            };

            let factorize_result = factorize(&remaining_tensor, &left_inds, &factorize_options)
                .map_err(|e| anyhow::anyhow!("Factorization failed: {:?}", e))?;

            // Left tensor gets the separated indices
            result.push((target_name.clone(), factorize_result.left));

            // Right tensor becomes the remaining tensor for next iteration
            remaining_tensor = factorize_result.right;
        }

        // The last target gets the remaining tensor
        let last_target = target_names.last().unwrap().clone();
        result.push((last_target, remaining_tensor));

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{random_treetn_f64, LinkSpace};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use tensor4all_core::index::{DefaultIndex as Index, DynId, NoSymmSpace};

    // =========================================================================
    // Test helper: generic fuse_to test for any topology
    // =========================================================================

    /// Test fuse_to with identity transformation (same structure).
    /// The result should equal the original tensor.
    fn test_fuse_to_identity_generic(
        site_network: &SiteIndexNetwork<String, Index<DynId, NoSymmSpace>>,
        link_space: LinkSpace<String>,
        seed: u64,
        rtol: f64,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tn = random_treetn_f64(&mut rng, site_network, link_space);

        // Identity transformation: use the same site_index_network as target
        let fused = tn.fuse_to(site_network).unwrap();

        // Verify structure is preserved
        assert_eq!(fused.node_count(), tn.node_count());
        assert_eq!(fused.edge_count(), tn.edge_count());

        // Verify tensor is unchanged
        let original_tensor = tn.contract_to_tensor().unwrap();
        let fused_tensor = fused.contract_to_tensor().unwrap();
        let distance = original_tensor.distance(&fused_tensor);
        assert!(
            distance < rtol,
            "Identity fuse should not change tensor: distance = {}",
            distance
        );
    }

    /// Test fuse_to that merges all nodes into one.
    /// The result should be a single tensor equal to full contraction.
    fn test_fuse_to_all_into_one_generic(
        site_network: &SiteIndexNetwork<String, Index<DynId, NoSymmSpace>>,
        link_space: LinkSpace<String>,
        seed: u64,
        rtol: f64,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tn = random_treetn_f64(&mut rng, site_network, link_space);

        // Collect all site indices from all nodes
        let mut all_site_indices: HashSet<Index<DynId, NoSymmSpace>> = HashSet::new();
        for node_name in site_network.node_names() {
            if let Some(site_space) = site_network.site_space(node_name) {
                all_site_indices.extend(site_space.iter().cloned());
            }
        }

        // Create target with single node containing all site indices
        let mut target = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        target
            .add_node("ALL".to_string(), all_site_indices)
            .unwrap();

        // Fuse
        let fused = tn.fuse_to(&target).unwrap();

        // Verify single node
        assert_eq!(fused.node_count(), 1);
        assert_eq!(fused.edge_count(), 0);

        // Verify tensor is unchanged
        let original_tensor = tn.contract_to_tensor().unwrap();
        let fused_tensor = fused.contract_to_tensor().unwrap();
        let distance = original_tensor.distance(&fused_tensor);
        assert!(
            distance < rtol,
            "Full fuse should produce same tensor: distance = {}",
            distance
        );
    }

    /// Test fuse_to that merges adjacent pairs.
    /// For a chain A-B-C-D, merge to AB-CD.
    fn test_fuse_to_pairwise_generic(
        site_network: &SiteIndexNetwork<String, Index<DynId, NoSymmSpace>>,
        target_network: &SiteIndexNetwork<String, Index<DynId, NoSymmSpace>>,
        link_space: LinkSpace<String>,
        seed: u64,
        rtol: f64,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tn = random_treetn_f64(&mut rng, site_network, link_space);

        // Fuse
        let fused = tn.fuse_to(target_network).unwrap();

        // Verify structure
        assert_eq!(fused.node_count(), target_network.node_names().len());

        // Verify tensor is unchanged
        let original_tensor = tn.contract_to_tensor().unwrap();
        let fused_tensor = fused.contract_to_tensor().unwrap();
        let distance = original_tensor.distance(&fused_tensor);
        assert!(
            distance < rtol,
            "Pairwise fuse should produce same tensor: distance = {}",
            distance
        );
    }

    // =========================================================================
    // Topology builders
    // =========================================================================

    /// Create a 4-node chain: A -- B -- C -- D
    fn create_4node_chain() -> SiteIndexNetwork<String, Index<DynId, NoSymmSpace>> {
        let mut net = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        let site_a = Index::new_dyn(2);
        let site_b = Index::new_dyn(2);
        let site_c = Index::new_dyn(2);
        let site_d = Index::new_dyn(2);
        net.add_node("A".to_string(), HashSet::from([site_a]))
            .unwrap();
        net.add_node("B".to_string(), HashSet::from([site_b]))
            .unwrap();
        net.add_node("C".to_string(), HashSet::from([site_c]))
            .unwrap();
        net.add_node("D".to_string(), HashSet::from([site_d]))
            .unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();
        net
    }

    /// Create a star topology: B is center, connected to A, C, D
    ///     A
    ///     |
    /// C - B - D
    fn create_star() -> SiteIndexNetwork<String, Index<DynId, NoSymmSpace>> {
        let mut net = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        let site_a = Index::new_dyn(2);
        let site_b = Index::new_dyn(2);
        let site_c = Index::new_dyn(2);
        let site_d = Index::new_dyn(2);
        net.add_node("A".to_string(), HashSet::from([site_a]))
            .unwrap();
        net.add_node("B".to_string(), HashSet::from([site_b]))
            .unwrap();
        net.add_node("C".to_string(), HashSet::from([site_c]))
            .unwrap();
        net.add_node("D".to_string(), HashSet::from([site_d]))
            .unwrap();
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
        net.add_edge(&"B".to_string(), &"D".to_string()).unwrap();
        net
    }

    /// Create a Y-shape topology: A -- B -- C, B -- D
    ///     A
    ///     |
    ///     B -- C
    ///     |
    ///     D
    fn create_y_shape() -> SiteIndexNetwork<String, Index<DynId, NoSymmSpace>> {
        create_star() // Same as star for this topology
    }

    // =========================================================================
    // Tests: 4-node chain
    // =========================================================================

    #[test]
    fn test_fuse_to_identity_4node_chain() {
        let net = create_4node_chain();
        test_fuse_to_identity_generic(&net, LinkSpace::uniform(3), 42, 1e-10);
    }

    #[test]
    fn test_fuse_to_all_into_one_4node_chain() {
        let net = create_4node_chain();
        test_fuse_to_all_into_one_generic(&net, LinkSpace::uniform(3), 42, 1e-10);
    }

    #[test]
    fn test_fuse_to_pairwise_4node_chain() {
        let net = create_4node_chain();

        // Get site indices
        let site_a = net
            .site_space(&"A".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_b = net
            .site_space(&"B".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_c = net
            .site_space(&"C".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_d = net
            .site_space(&"D".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();

        // Create target: A+B -> AB, C+D -> CD
        let mut target = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        target
            .add_node("AB".to_string(), HashSet::from([site_a, site_b]))
            .unwrap();
        target
            .add_node("CD".to_string(), HashSet::from([site_c, site_d]))
            .unwrap();
        target
            .add_edge(&"AB".to_string(), &"CD".to_string())
            .unwrap();

        test_fuse_to_pairwise_generic(&net, &target, LinkSpace::uniform(3), 42, 1e-10);
    }

    // =========================================================================
    // Tests: Star topology
    // =========================================================================

    #[test]
    fn test_fuse_to_identity_star() {
        let net = create_star();
        test_fuse_to_identity_generic(&net, LinkSpace::uniform(3), 42, 1e-10);
    }

    #[test]
    fn test_fuse_to_all_into_one_star() {
        let net = create_star();
        test_fuse_to_all_into_one_generic(&net, LinkSpace::uniform(3), 42, 1e-10);
    }

    #[test]
    fn test_fuse_to_star_merge_leaves() {
        let net = create_star();

        // Get site indices
        let site_a = net
            .site_space(&"A".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_b = net
            .site_space(&"B".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_c = net
            .site_space(&"C".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_d = net
            .site_space(&"D".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();

        // Merge center B with leaf A, keep C and D separate
        // Target: AB (center+one leaf), C, D
        let mut target = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        target
            .add_node("AB".to_string(), HashSet::from([site_a, site_b]))
            .unwrap();
        target
            .add_node("C".to_string(), HashSet::from([site_c]))
            .unwrap();
        target
            .add_node("D".to_string(), HashSet::from([site_d]))
            .unwrap();
        target
            .add_edge(&"AB".to_string(), &"C".to_string())
            .unwrap();
        target
            .add_edge(&"AB".to_string(), &"D".to_string())
            .unwrap();

        test_fuse_to_pairwise_generic(&net, &target, LinkSpace::uniform(3), 42, 1e-10);
    }

    // =========================================================================
    // Tests: Y-shape topology
    // =========================================================================

    #[test]
    fn test_fuse_to_identity_y_shape() {
        let net = create_y_shape();
        test_fuse_to_identity_generic(&net, LinkSpace::uniform(3), 42, 1e-10);
    }

    #[test]
    fn test_fuse_to_all_into_one_y_shape() {
        let net = create_y_shape();
        test_fuse_to_all_into_one_generic(&net, LinkSpace::uniform(3), 42, 1e-10);
    }

    // =========================================================================
    // Error tests
    // =========================================================================

    #[test]
    fn test_fuse_to_incompatible_error() {
        let net = create_4node_chain();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let tn = random_treetn_f64(&mut rng, &net, LinkSpace::uniform(3));

        // Get site indices for A and D only (skip B, C)
        let site_a = net
            .site_space(&"A".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_d = net
            .site_space(&"D".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();

        // Create target that tries to fuse non-adjacent nodes A and D (skipping B, C)
        // This should fail because B and C have no target node
        let mut target = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        target
            .add_node("AD".to_string(), HashSet::from([site_a, site_d]))
            .unwrap();

        let result = tn.fuse_to(&target);
        assert!(result.is_err());
    }

    // =========================================================================
    // split_to tests
    // =========================================================================

    /// Test split_to with identity transformation (same structure).
    /// The result should equal the original tensor.
    #[test]
    fn test_split_to_identity() {
        let net = create_4node_chain();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let tn = random_treetn_f64(&mut rng, &net, LinkSpace::uniform(3));

        // Identity transformation
        let options = SplitOptions::default();
        let split = tn.split_to(&net, &options).unwrap();

        // Verify structure is preserved
        assert_eq!(split.node_count(), tn.node_count());
        assert_eq!(split.edge_count(), tn.edge_count());

        // Verify tensor is unchanged
        let original_tensor = tn.contract_to_tensor().unwrap();
        let split_tensor = split.contract_to_tensor().unwrap();
        let distance = original_tensor.distance(&split_tensor);
        assert!(
            distance < 1e-10,
            "Identity split should not change tensor: distance = {}",
            distance
        );
    }

    /// Test split_to: split a 2-node fused chain into 4 nodes.
    /// Before: AB -- CD (2 nodes, each with 2 site indices)
    /// After: A -- B -- C -- D (4 nodes)
    #[test]
    fn test_split_to_fused_to_chain() {
        // Create target (fine) structure: 4-node chain
        let target = create_4node_chain();

        // Get site indices from target
        let site_a = target
            .site_space(&"A".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_b = target
            .site_space(&"B".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_c = target
            .site_space(&"C".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_d = target
            .site_space(&"D".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();

        // Create source (coarse) structure: AB -- CD
        let mut source = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        source
            .add_node("AB".to_string(), HashSet::from([site_a.clone(), site_b.clone()]))
            .unwrap();
        source
            .add_node("CD".to_string(), HashSet::from([site_c.clone(), site_d.clone()]))
            .unwrap();
        source
            .add_edge(&"AB".to_string(), &"CD".to_string())
            .unwrap();

        // Create random TreeTN with source structure
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let tn = random_treetn_f64(&mut rng, &source, LinkSpace::uniform(3));

        // Split to target structure
        let options = SplitOptions::default();
        let split = tn.split_to(&target, &options).unwrap();

        // Verify structure
        assert_eq!(split.node_count(), 4, "Should have 4 nodes after split");
        assert_eq!(split.edge_count(), 3, "Chain should have 3 edges");

        // Verify tensor is unchanged (exact factorization)
        let original_tensor = tn.contract_to_tensor().unwrap();
        let split_tensor = split.contract_to_tensor().unwrap();
        let distance = original_tensor.distance(&split_tensor);
        assert!(
            distance < 1e-10,
            "Split should preserve tensor: distance = {}",
            distance
        );
    }

    /// Test fuse_to followed by split_to (roundtrip).
    /// A -- B -- C -- D -> AB -- CD -> A -- B -- C -- D
    #[test]
    fn test_fuse_then_split_roundtrip() {
        let fine = create_4node_chain();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let original = random_treetn_f64(&mut rng, &fine, LinkSpace::uniform(3));

        // Get site indices
        let site_a = fine
            .site_space(&"A".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_b = fine
            .site_space(&"B".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_c = fine
            .site_space(&"C".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_d = fine
            .site_space(&"D".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();

        // Create coarse structure
        let mut coarse = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        coarse
            .add_node("AB".to_string(), HashSet::from([site_a, site_b]))
            .unwrap();
        coarse
            .add_node("CD".to_string(), HashSet::from([site_c, site_d]))
            .unwrap();
        coarse
            .add_edge(&"AB".to_string(), &"CD".to_string())
            .unwrap();

        // Fuse: fine -> coarse
        let fused = original.fuse_to(&coarse).unwrap();
        assert_eq!(fused.node_count(), 2);

        // Split: coarse -> fine
        let options = SplitOptions::default();
        let restored = fused.split_to(&fine, &options).unwrap();
        assert_eq!(restored.node_count(), 4);

        // Verify tensor is preserved through roundtrip
        let original_tensor = original.contract_to_tensor().unwrap();
        let restored_tensor = restored.contract_to_tensor().unwrap();
        let distance = original_tensor.distance(&restored_tensor);
        assert!(
            distance < 1e-10,
            "Roundtrip should preserve tensor: distance = {}",
            distance
        );
    }

    /// Test split_to with final_sweep option.
    #[test]
    fn test_split_to_with_truncation() {
        // Create target (fine) structure
        let target = create_4node_chain();

        // Get site indices from target
        let site_a = target
            .site_space(&"A".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_b = target
            .site_space(&"B".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_c = target
            .site_space(&"C".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let site_d = target
            .site_space(&"D".to_string())
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();

        // Create source structure
        let mut source = SiteIndexNetwork::<String, Index<DynId, NoSymmSpace>>::new();
        source
            .add_node("AB".to_string(), HashSet::from([site_a.clone(), site_b.clone()]))
            .unwrap();
        source
            .add_node("CD".to_string(), HashSet::from([site_c.clone(), site_d.clone()]))
            .unwrap();
        source
            .add_edge(&"AB".to_string(), &"CD".to_string())
            .unwrap();

        // Create random TreeTN
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let tn = random_treetn_f64(&mut rng, &source, LinkSpace::uniform(5));

        // Split with truncation
        let options = SplitOptions::default()
            .with_final_sweep(true)
            .with_max_rank(3);
        let split = tn.split_to(&target, &options).unwrap();

        // Verify structure
        assert_eq!(split.node_count(), 4);

        // Tensor should be approximately preserved (with some truncation error)
        let original_tensor = tn.contract_to_tensor().unwrap();
        let split_tensor = split.contract_to_tensor().unwrap();
        let distance = original_tensor.distance(&split_tensor);
        // With truncation, we expect some error but it should be reasonable
        assert!(
            distance < 1.0,
            "Truncated split should approximately preserve tensor: distance = {}",
            distance
        );
    }
}
