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
mod tests {
    use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};

    use crate::treetn::TreeTN;

    /// Create two TreeTNs with the same topology for testing addition.
    /// Both are 2-node chains: A -- bond -- B
    fn make_two_matching_treetns() -> (
        TreeTN<TensorDynLen, String>,
        TreeTN<TensorDynLen, String>,
        DynIndex, // s0
        DynIndex, // s1
    ) {
        let s0 = DynIndex::new_dyn(2);
        let bond_a = DynIndex::new_dyn(3);
        let s1 = DynIndex::new_dyn(2);

        let t0_a = TensorDynLen::from_dense_f64(
            vec![s0.clone(), bond_a.clone()],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let t1_a = TensorDynLen::from_dense_f64(
            vec![bond_a.clone(), s1.clone()],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );

        let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0_a, t1_a],
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap();

        let bond_b = DynIndex::new_dyn(2);

        let t0_b = TensorDynLen::from_dense_f64(
            vec![s0.clone(), bond_b.clone()],
            vec![0.0, 1.0, 1.0, 0.0],
        );
        let t1_b = TensorDynLen::from_dense_f64(
            vec![bond_b.clone(), s1.clone()],
            vec![1.0, 0.0, 0.0, 1.0],
        );

        let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0_b, t1_b],
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap();

        (tn_a, tn_b, s0, s1)
    }

    #[test]
    fn test_compute_merged_bond_indices() {
        let (tn_a, tn_b, _s0, _s1) = make_two_matching_treetns();

        let merged = tn_a.compute_merged_bond_indices(&tn_b).unwrap();
        assert_eq!(merged.len(), 1);

        // The key should be ("A", "B") in canonical order
        let key = ("A".to_string(), "B".to_string());
        let info = merged.get(&key).unwrap();
        assert_eq!(info.dim_a, 3); // tn_a bond dim
        assert_eq!(info.dim_b, 2); // tn_b bond dim
    }

    #[test]
    fn test_add_basic() {
        let (tn_a, tn_b, s0, s1) = make_two_matching_treetns();

        let result = tn_a.add(&tn_b).unwrap();

        // Result should have same number of nodes
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 1);

        // Result should have the same site indices
        let ext_ids: Vec<_> = result.external_indices().iter().map(|i| *i.id()).collect();
        assert_eq!(ext_ids.len(), 2);
        assert!(ext_ids.contains(s0.id()));
        assert!(ext_ids.contains(s1.id()));

        // Bond dimension should be sum of original dimensions
        let edge = result.graph.graph().edge_indices().next().unwrap();
        let merged_bond = result.bond_index(edge).unwrap();
        assert_eq!(merged_bond.dim(), 3 + 2); // dim_a + dim_b
    }

    #[test]
    fn test_add_verifies_with_contraction() {
        let (tn_a, tn_b, _s0, _s1) = make_two_matching_treetns();

        let result = tn_a.add(&tn_b).unwrap();

        // Contract all three and verify: contract(result) ≈ contract(tn_a) + contract(tn_b)
        let tensor_a = tn_a.contract_to_tensor().unwrap();
        let tensor_b = tn_b.contract_to_tensor().unwrap();
        let tensor_sum = result.contract_to_tensor().unwrap();

        // Get dense data for comparison
        let data_a = tensor_a.to_vec_f64().unwrap();
        let data_b = tensor_b.to_vec_f64().unwrap();
        let data_sum = tensor_sum.to_vec_f64().unwrap();

        // Verify element-wise: contract(result) ≈ contract(tn_a) + contract(tn_b)
        assert_eq!(data_sum.len(), data_a.len());
        assert_eq!(data_sum.len(), data_b.len());
        for i in 0..data_a.len() {
            assert!(
                (data_sum[i] - (data_a[i] + data_b[i])).abs() < 1e-10,
                "Element {} mismatch: {} != {} + {}",
                i,
                data_sum[i],
                data_a[i],
                data_b[i]
            );
        }
    }

    #[test]
    fn test_add_topology_mismatch() {
        let s0 = DynIndex::new_dyn(2);
        let bond = DynIndex::new_dyn(3);
        let s1 = DynIndex::new_dyn(2);

        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone(), bond.clone()], vec![1.0; 6]);
        let t1 = TensorDynLen::from_dense_f64(vec![bond.clone(), s1.clone()], vec![1.0; 6]);
        let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0, t1],
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap();

        // Single-node TN (different topology)
        let t_single = TensorDynLen::from_dense_f64(vec![s0.clone()], vec![1.0, 2.0]);
        let tn_b =
            TreeTN::<TensorDynLen, String>::from_tensors(vec![t_single], vec!["X".to_string()])
                .unwrap();

        let result = tn_a.add(&tn_b);
        assert!(result.is_err());
    }
}
