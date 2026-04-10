//! Addition operations for TreeTN using direct-sum (block) construction.
//!
//! This module provides helper functions and types for adding two TreeTNs:
//! - [`MergedBondInfo`]: Information about merged bond indices
//! - [`compute_merged_bond_indices`]: Compute merged bond index information from two networks

use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{bail, Result};

use tensor4all_core::{AnyScalar, IndexLike, TensorIndex, TensorLike};

use super::TreeTN;

/// Information about a merged bond index for direct-sum addition.
///
/// When adding two TreeTNs, each bond index in the result has dimension
/// `dim_a + dim_b`, where `dim_a` and `dim_b` are the original bond dimensions.
#[derive(Debug, Clone)]
pub struct MergedBondInfo<I>
where
    I: IndexLike,
{
    /// Bond dimension from the first TreeTN
    pub dim_a: usize,
    /// Bond dimension from the second TreeTN
    pub dim_b: usize,
    /// The new merged bond index (with dimension dim_a + dim_b)
    pub merged_index: I,
}

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn sorted_site_space(site_space: &HashSet<T::Index>) -> Vec<T::Index>
    where
        T::Index: Clone,
        <T::Index as IndexLike>::Id: Ord,
    {
        let mut indices: Vec<_> = site_space.iter().cloned().collect();
        indices.sort_by(|left, right| {
            left.dim()
                .cmp(&right.dim())
                .then_with(|| left.id().cmp(right.id()))
        });
        indices
    }

    /// Reindex this TreeTN's site space to match a template network.
    ///
    /// The topology must match, and each corresponding node must carry the same
    /// number of site indices with the same dimensions. Site indices are paired
    /// node-by-node after sorting by `(dim, id)` for deterministic matching.
    ///
    /// # Arguments
    /// * `template` - Reference TreeTN whose site index IDs should be adopted
    ///
    /// # Returns
    /// A new TreeTN with the same tensor data as `self`, but site index IDs
    /// rewritten to match `template`.
    ///
    /// # Errors
    /// Returns an error if the two networks have different topologies or
    /// incompatible site-space dimensions on any node.
    ///
    /// # Examples
    /// ```ignore
    /// let aligned = state_b.reindex_site_space_like(&state_a).unwrap();
    /// assert!(aligned.share_equivalent_site_index_network(&state_a));
    /// ```
    pub fn reindex_site_space_like(&self, template: &Self) -> Result<Self>
    where
        V: Ord,
        T::Index: Clone,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        if !self.same_topology(template) {
            bail!("reindex_site_space_like: networks have incompatible topologies");
        }

        let mut old_indices = Vec::new();
        let mut new_indices = Vec::new();

        for node_name in self.node_names() {
            let self_site_space = self
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("site space not found for node {:?}", node_name))?;
            let template_site_space = template.site_space(&node_name).ok_or_else(|| {
                anyhow::anyhow!("template site space not found for node {:?}", node_name)
            })?;

            if self_site_space.len() != template_site_space.len() {
                bail!(
                    "reindex_site_space_like: node {:?} has {} site indices in self but {} in template",
                    node_name,
                    self_site_space.len(),
                    template_site_space.len()
                );
            }

            let self_sorted = Self::sorted_site_space(self_site_space);
            let template_sorted = Self::sorted_site_space(template_site_space);

            for (old_index, new_index) in self_sorted.iter().zip(template_sorted.iter()) {
                if old_index.dim() != new_index.dim() {
                    bail!(
                        "reindex_site_space_like: node {:?} site dimension mismatch {} != {}",
                        node_name,
                        old_index.dim(),
                        new_index.dim()
                    );
                }
                old_indices.push(old_index.clone());
                new_indices.push(new_index.clone());
            }
        }

        self.replaceinds(&old_indices, &new_indices)
    }

    /// Add two TreeTNs after aligning the second operand's site index IDs to the first.
    ///
    /// This is useful when two states share the same topology and site dimensions
    /// but were constructed with different site index IDs.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to align and add
    ///
    /// # Returns
    /// The direct-sum addition result with site IDs matching `self`.
    ///
    /// # Examples
    /// ```ignore
    /// let sum = state_a.add_aligned(&state_b).unwrap();
    /// assert!(sum.share_equivalent_site_index_network(&state_a));
    /// ```
    pub fn add_aligned(&self, other: &Self) -> Result<Self>
    where
        V: Ord,
        T::Index: Clone,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        let other_aligned = other.reindex_site_space_like(self)?;
        self.add(&other_aligned)
    }

    /// Compute merged bond indices for direct-sum addition.
    ///
    /// For each edge in the network, compute the merged bond information
    /// containing dimensions from both networks and a new merged index.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to compute merged bonds with
    ///
    /// # Returns
    /// A HashMap mapping edge keys (node_name_pair in canonical order) to MergedBondInfo.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Networks have incompatible topologies
    /// - Bond indices cannot be found
    #[allow(clippy::type_complexity)]
    pub fn compute_merged_bond_indices(
        &self,
        other: &Self,
    ) -> Result<HashMap<(V, V), MergedBondInfo<T::Index>>>
    where
        V: Ord,
    {
        let mut result = HashMap::new();

        for edge in self.graph.graph().edge_indices() {
            let (src, tgt) = self
                .graph
                .graph()
                .edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge has no endpoints"))?;

            let bond_index_a = self
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found in self"))?;
            let dim_a = bond_index_a.dim();

            let src_name = self
                .graph
                .node_name(src)
                .ok_or_else(|| anyhow::anyhow!("Source node name not found"))?
                .clone();
            let tgt_name = self
                .graph
                .node_name(tgt)
                .ok_or_else(|| anyhow::anyhow!("Target node name not found"))?
                .clone();

            // Find corresponding edge in other
            let src_idx_other = other
                .graph
                .node_index(&src_name)
                .ok_or_else(|| anyhow::anyhow!("Source node not found in other"))?;
            let tgt_idx_other = other
                .graph
                .node_index(&tgt_name)
                .ok_or_else(|| anyhow::anyhow!("Target node not found in other"))?;

            // Find edge between these nodes in other
            let edge_other = other
                .graph
                .graph()
                .edges_connecting(src_idx_other, tgt_idx_other)
                .next()
                .or_else(|| {
                    other
                        .graph
                        .graph()
                        .edges_connecting(tgt_idx_other, src_idx_other)
                        .next()
                })
                .ok_or_else(|| anyhow::anyhow!("Edge not found in other"))?;

            let bond_index_b = other
                .bond_index(edge_other.id())
                .ok_or_else(|| anyhow::anyhow!("Bond index not found in other"))?;
            let dim_b = bond_index_b.dim();

            // Create merged bond index using direct_sum on dummy tensors
            // For now, we just store dimensions; the actual merged index will be
            // created during the direct sum operation using TensorLike::direct_sum
            //
            // Note: We need a way to create a new index with dim_a + dim_b.
            // This requires the TensorLike implementation to handle index creation.
            // For now, we clone one of the existing indices as a placeholder.
            // The actual merging happens in the direct_sum operation.
            let merged_index = bond_index_a.clone();

            // Store in canonical order (smaller name first)
            let key = if src_name < tgt_name {
                (src_name, tgt_name)
            } else {
                (tgt_name, src_name)
            };

            result.insert(
                key,
                MergedBondInfo {
                    dim_a,
                    dim_b,
                    merged_index,
                },
            );
        }

        Ok(result)
    }

    /// Add two TreeTNs using direct-sum construction.
    ///
    /// This creates a new TreeTN where each tensor is the direct sum of the
    /// corresponding tensors from self and other, with bond dimensions merged.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to add
    ///
    /// # Returns
    /// A new TreeTN representing the sum.
    ///
    /// # Errors
    /// Returns an error if the networks have incompatible structures.
    pub fn add(&self, other: &Self) -> Result<Self>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        // Verify same topology
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!(
                "Cannot add TreeTNs with different topologies"
            ));
        }

        // Track merged indices for each edge.
        // Key: (smaller_node_name, larger_node_name) for canonical ordering
        // Value: the merged bond index to use for this edge
        let mut edge_merged_indices: HashMap<(V, V), T::Index> = HashMap::new();

        // For each node, compute the direct sum of tensors
        let mut result_tensors: Vec<T> = Vec::new();
        let mut result_node_names: Vec<V> = Vec::new();

        for node_name in self.node_names() {
            let self_idx = self.node_index(&node_name).unwrap();
            let other_idx = other.node_index(&node_name).unwrap();

            let tensor_a = self.tensor(self_idx).unwrap();
            let tensor_b = other.tensor(other_idx).unwrap();

            // Find bond index pairs for this node and track neighbors
            let mut bond_pairs: Vec<(T::Index, T::Index)> = Vec::new();
            let mut neighbors_for_edges: Vec<V> = Vec::new();

            for neighbor in self.site_index_network().neighbors(&node_name) {
                // Get bond index from self
                let self_edge = self.edge_between(&node_name, &neighbor).unwrap();
                let self_bond = self.bond_index(self_edge).unwrap();

                // Get bond index from other
                let other_edge = other.edge_between(&node_name, &neighbor).unwrap();
                let other_bond = other.bond_index(other_edge).unwrap();

                bond_pairs.push((self_bond.clone(), other_bond.clone()));
                neighbors_for_edges.push(neighbor);
            }

            // For nodes with no bonds (single-node network), use element-wise addition
            // instead of direct_sum (which requires at least one index pair).
            if bond_pairs.is_empty() {
                let sum_tensor =
                    tensor_a.axpby(AnyScalar::new_real(1.0), tensor_b, AnyScalar::new_real(1.0))?;
                result_tensors.push(sum_tensor);
                result_node_names.push(node_name);
                continue;
            }

            // Compute direct sum
            let direct_sum_result = tensor_a.direct_sum(tensor_b, &bond_pairs)?;
            let mut result_tensor = direct_sum_result.tensor;

            // For each edge, ensure we use consistent merged indices:
            // - If we've already seen this edge, replace the auto-generated index with the stored one
            // - If this is the first time seeing this edge, store the auto-generated index
            for (i, neighbor) in neighbors_for_edges.iter().enumerate() {
                // Create canonical edge key (smaller name first)
                let edge_key = if node_name < *neighbor {
                    (node_name.clone(), neighbor.clone())
                } else {
                    (neighbor.clone(), node_name.clone())
                };

                let new_index = &direct_sum_result.new_indices[i];

                if let Some(stored_index) = edge_merged_indices.get(&edge_key) {
                    // Edge already processed - replace auto-generated index with stored one
                    result_tensor = result_tensor.replaceind(new_index, stored_index)?;
                } else {
                    // First time seeing this edge - store the auto-generated index
                    edge_merged_indices.insert(edge_key, new_index.clone());
                }
            }

            result_tensors.push(result_tensor);
            result_node_names.push(node_name);
        }

        // Build result TreeTN
        TreeTN::from_tensors(result_tensors, result_node_names)
    }
}

#[cfg(test)]
mod tests;
