//! Structural transformation operations for TreeTN.
//!
//! This module provides methods to transform a TreeTN's structure:
//! - [`fuse_to`](TreeTN::fuse_to): Merge adjacent nodes to match a target structure
//! - [`split_to`](TreeTN::split_to): Split nodes to match a target structure

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};
use petgraph::stable_graph::NodeIndex;

use tensor4all_core::{
    AllowedPairs, Canonical, FactorizeAlg, FactorizeOptions, IndexLike, TensorLike,
};

use super::TreeTN;
use crate::options::SplitOptions;
use crate::site_index_network::SiteIndexNetwork;

type SplitBoundaryIndices<V, TargetV, Id> = HashMap<V, HashMap<TargetV, HashSet<Id>>>;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
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
        target: &SiteIndexNetwork<TargetV, T::Index>,
    ) -> Result<TreeTN<T, TargetV>>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // Step 1: Build a mapping from site index ID to current node name
        let mut site_to_current_node: HashMap<<T::Index as IndexLike>::Id, V> = HashMap::new();
        for current_node_name in self.node_names() {
            if let Some(site_space) = self.site_space(&current_node_name) {
                for site_idx in site_space {
                    site_to_current_node.insert(site_idx.id().clone(), current_node_name.clone());
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
                if let Some(current_node) = site_to_current_node.get(target_site_idx.id()) {
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
        let mut result_tensors: HashMap<TargetV, T> = HashMap::new();

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

        let tensors: Vec<T> = target_names
            .iter()
            .map(|name| result_tensors.remove(name).unwrap())
            .collect();

        let result = TreeTN::<T, TargetV>::from_tensors(tensors, target_names)
            .context("fuse_to: failed to build result TreeTN")?;

        Ok(result)
    }

    /// Contract a group of nodes into a single tensor.
    ///
    /// The nodes must form a connected subtree in the current TreeTN.
    /// Contracts all internal bonds (bonds between nodes in the group),
    /// keeping external bonds and site indices.
    fn contract_node_group(&self, nodes: &HashSet<V>) -> Result<T>
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
        let mut tensors: HashMap<NodeIndex, T> = node_indices
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

            // Contract using TensorLike::contract
            // (bond indices are auto-detected via is_contractable)
            let contracted = T::contract(&[&to_tensor, &from_tensor], AllowedPairs::All)
                .map_err(|e| anyhow::anyhow!("Failed to contract tensors: {}", e))?;

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
        target: &SiteIndexNetwork<TargetV, T::Index>,
        options: &SplitOptions,
    ) -> Result<TreeTN<T, TargetV>>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // Step 1: Build mapping from site index ID to target node name
        let mut site_to_target: HashMap<<T::Index as IndexLike>::Id, TargetV> = HashMap::new();
        for target_node_name in target.node_names() {
            if let Some(site_space) = target.site_space(target_node_name) {
                for site_idx in site_space {
                    site_to_target.insert(site_idx.id().clone(), target_node_name.clone());
                }
            }
        }

        let mut site_to_current: HashMap<<T::Index as IndexLike>::Id, V> = HashMap::new();
        for current_node_name in self.node_names() {
            let site_space = self.site_space(&current_node_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "split_to: current node {:?} has no registered site space",
                    current_node_name
                )
            })?;
            for site_idx in site_space {
                site_to_current.insert(site_idx.id().clone(), current_node_name.clone());
            }
        }

        let mut target_to_current: HashMap<TargetV, V> = HashMap::new();
        for target_node_name in target.node_names() {
            let site_space = target.site_space(target_node_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "split_to: target node {:?} has no registered site space",
                    target_node_name
                )
            })?;
            let current_names: HashSet<_> = site_space
                .iter()
                .map(|site_idx| {
                    site_to_current.get(site_idx.id()).cloned().ok_or_else(|| {
                        anyhow::anyhow!(
                            "split_to: target site index {:?} is missing from the current network",
                            site_idx.id()
                        )
                    })
                })
                .collect::<Result<_>>()?;
            if current_names.len() != 1 {
                return Err(anyhow::anyhow!(
                    "split_to: target node {:?} spans multiple current nodes; use restructure_to for mixed regrouping",
                    target_node_name
                ));
            }
            let current_name = current_names.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!(
                    "split_to: target node {:?} has no current owner",
                    target_node_name
                )
            })?;
            target_to_current.insert(target_node_name.clone(), current_name);
        }

        let boundary_indices = if target.edge_count() == 0 {
            HashMap::new()
        } else {
            self.split_boundary_indices_by_target(target, &target_to_current)
                .context("split_to: assign current bonds to target boundary fragments")?
        };

        // Step 2: For each current node, determine which target nodes it maps to
        // and validate the mapping
        let mut current_to_targets: HashMap<V, HashSet<TargetV>> = HashMap::new();
        for current_node_name in self.node_names() {
            if let Some(site_space) = self.site_space(&current_node_name) {
                let mut targets_for_node: HashSet<TargetV> = HashSet::new();
                for site_idx in site_space {
                    if let Some(target_name) = site_to_target.get(site_idx.id()) {
                        targets_for_node.insert(target_name.clone());
                    } else {
                        return Err(anyhow::anyhow!(
                            "Site index {:?} in current node {:?} has no corresponding target node",
                            site_idx.id(),
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
        let mut result_tensors: Vec<(TargetV, T)> = Vec::new();

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
                    .split_tensor_for_targets(
                        tensor,
                        &site_to_target,
                        boundary_indices.get(&current_node_name),
                    )
                    .with_context(|| {
                        format!("split_to: failed to split node {:?}", current_node_name)
                    })?;
                result_tensors.extend(split_tensors);
            }
        }

        // Step 4: Build the result TreeTN with target node names
        // Sort by target name for deterministic ordering
        result_tensors.sort_by(|(a, _), (b, _)| a.cmp(b));

        let names: Vec<TargetV> = result_tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let tensors: Vec<T> = result_tensors.into_iter().map(|(_, t)| t).collect();

        let result = TreeTN::<T, TargetV>::from_tensors(tensors, names)
            .context("split_to: failed to build result TreeTN")?;

        // Step 5: Phase 2 - Optional truncation sweep
        if options.final_sweep {
            // Find a center node for truncation
            let center = result.node_names().into_iter().min().ok_or_else(|| {
                anyhow::anyhow!("split_to: no nodes in result for truncation sweep")
            })?;

            let truncation_options = crate::TruncationOptions {
                form: options.form,
                truncation: options.truncation,
            };

            return result
                .truncate([center], truncation_options)
                .context("split_to: truncation sweep failed");
        }

        Ok(result)
    }

    fn split_boundary_indices_by_target<TargetV>(
        &self,
        target: &SiteIndexNetwork<TargetV, T::Index>,
        target_to_current: &HashMap<TargetV, V>,
    ) -> Result<SplitBoundaryIndices<V, TargetV, <T::Index as IndexLike>::Id>>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        let mut boundary_indices: SplitBoundaryIndices<V, TargetV, <T::Index as IndexLike>::Id> =
            HashMap::new();

        for edge in self.graph.graph().edge_indices() {
            let (node_a, node_b) = self.graph.graph().edge_endpoints(edge).ok_or_else(|| {
                anyhow::anyhow!("split_to: current edge {:?} has missing endpoints", edge)
            })?;
            let current_a = self.graph.node_name(node_a).cloned().ok_or_else(|| {
                anyhow::anyhow!("split_to: current edge {:?} has missing source node", edge)
            })?;
            let current_b = self.graph.node_name(node_b).cloned().ok_or_else(|| {
                anyhow::anyhow!("split_to: current edge {:?} has missing target node", edge)
            })?;
            let bond_index = self.graph.graph().edge_weight(edge).ok_or_else(|| {
                anyhow::anyhow!("split_to: current edge {:?} has no bond index", edge)
            })?;

            let mut crossing_edges = Vec::new();
            for (target_a, target_b) in target.edges() {
                let owner_a = target_to_current.get(&target_a).ok_or_else(|| {
                    anyhow::anyhow!(
                        "split_to: target edge references node {:?} without a current owner",
                        target_a
                    )
                })?;
                let owner_b = target_to_current.get(&target_b).ok_or_else(|| {
                    anyhow::anyhow!(
                        "split_to: target edge references node {:?} without a current owner",
                        target_b
                    )
                })?;

                if owner_a == &current_a && owner_b == &current_b {
                    crossing_edges.push((target_a, target_b));
                } else if owner_a == &current_b && owner_b == &current_a {
                    crossing_edges.push((target_b, target_a));
                }
            }

            let [(target_for_a, target_for_b)] = crossing_edges.as_slice() else {
                return Err(anyhow::anyhow!(
                    "split_to: expected exactly one target edge crossing current edge {:?}-{:?}, found {}",
                    current_a,
                    current_b,
                    crossing_edges.len()
                ));
            };

            boundary_indices
                .entry(current_a)
                .or_default()
                .entry(target_for_a.clone())
                .or_default()
                .insert(bond_index.id().clone());
            boundary_indices
                .entry(current_b)
                .or_default()
                .entry(target_for_b.clone())
                .or_default()
                .insert(bond_index.id().clone());
        }

        Ok(boundary_indices)
    }

    /// Split a tensor into multiple tensors, one for each target node.
    ///
    /// This uses QR factorization to iteratively separate site indices
    /// belonging to different target nodes.
    ///
    /// Returns a vector of (target_name, tensor) pairs.
    fn split_tensor_for_targets<TargetV>(
        &self,
        tensor: &T,
        site_to_target: &HashMap<<T::Index as IndexLike>::Id, TargetV>,
        boundary_indices: Option<&HashMap<TargetV, HashSet<<T::Index as IndexLike>::Id>>>,
    ) -> Result<Vec<(TargetV, T)>>
    where
        TargetV: Clone + Hash + Eq + Ord + std::fmt::Debug,
    {
        // Group tensor's site indices by their target node
        let mut partition: HashMap<TargetV, HashSet<<T::Index as IndexLike>::Id>> = HashMap::new();
        for idx in tensor.external_indices() {
            if let Some(target_name) = site_to_target.get(idx.id()) {
                partition
                    .entry(target_name.clone())
                    .or_default()
                    .insert(idx.id().clone());
            }
            // Note: bond indices (not in site_to_target) are handled by factorize
        }
        if let Some(boundary_indices) = boundary_indices {
            let tensor_index_ids: HashSet<_> = tensor
                .external_indices()
                .iter()
                .map(|idx| idx.id().clone())
                .collect();
            for (target_name, index_ids) in boundary_indices {
                let target_partition = partition.entry(target_name.clone()).or_default();
                for index_id in index_ids {
                    if !tensor_index_ids.contains(index_id) {
                        return Err(anyhow::anyhow!(
                            "split_to: boundary index {:?} is not present in the tensor for target {:?}",
                            index_id,
                            target_name
                        ));
                    }
                    target_partition.insert(index_id.clone());
                }
            }
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
        let mut result: Vec<(TargetV, T)> = Vec::new();

        // Process all but the last target (the last one gets the remaining tensor)
        for target_name in target_names.iter().take(target_names.len() - 1) {
            let site_ids_for_target = partition.get(target_name).unwrap();

            // Find the actual Index objects for these site IDs
            let left_inds: Vec<_> = remaining_tensor
                .external_indices()
                .iter()
                .filter(|idx| site_ids_for_target.contains(idx.id()))
                .cloned()
                .collect();

            if left_inds.is_empty() {
                continue;
            }

            // Factorize: separate these site indices
            let factorize_options = FactorizeOptions {
                alg: FactorizeAlg::QR,
                canonical: Canonical::Left,
                max_rank: None,
                svd_policy: None,
                qr_rtol: None,
            };

            let factorize_result = remaining_tensor
                .factorize(&left_inds, &factorize_options)
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

// Tests are disabled until random module is refactored
#[cfg(test)]
mod tests;
