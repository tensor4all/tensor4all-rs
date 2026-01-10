//! Addition operations for TreeTN using direct-sum (block) construction.
//!
//! This module provides helper functions and types for adding two TreeTNs:
//! - [`MergedBondInfo`]: Information about merged bond indices
//! - [`compute_merged_bond_indices`]: Compute merged bond index information from two networks

use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::{IndexLike, TensorLike};

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
        <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        // Verify same topology
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!(
                "Cannot add TreeTNs with different topologies"
            ));
        }

        // For each node, compute the direct sum of tensors
        let mut result_tensors: Vec<T> = Vec::new();
        let mut result_node_names: Vec<V> = Vec::new();

        for node_name in self.node_names() {
            let self_idx = self.node_index(&node_name).unwrap();
            let other_idx = other.node_index(&node_name).unwrap();

            let tensor_a = self.tensor(self_idx).unwrap();
            let tensor_b = other.tensor(other_idx).unwrap();

            // Find bond index pairs for this node
            let mut bond_pairs: Vec<(T::Index, T::Index)> = Vec::new();

            for neighbor in self.site_index_network().neighbors(&node_name) {
                // Get bond index from self
                let self_edge = self.edge_between(&node_name, &neighbor).unwrap();
                let self_bond = self.bond_index(self_edge).unwrap();

                // Get bond index from other
                let other_edge = other.edge_between(&node_name, &neighbor).unwrap();
                let other_bond = other.bond_index(other_edge).unwrap();

                bond_pairs.push((self_bond.clone(), other_bond.clone()));
            }

            // Compute direct sum
            let direct_sum_result = tensor_a.direct_sum(tensor_b, &bond_pairs)?;
            result_tensors.push(direct_sum_result.tensor);
            result_node_names.push(node_name);
        }

        // Build result TreeTN
        TreeTN::from_tensors(result_tensors, result_node_names)
    }
}
