//! Local update operations for TreeTN.
//!
//! This module provides APIs for:
//! - Extracting a sub-tree from a TreeTN (creating a new TreeTN object)
//! - Replacing a sub-tree with another TreeTN of the same appearance
//!
//! These operations are fundamental for local update algorithms in tensor networks.

use std::collections::HashSet;
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all::index::Symmetry;

use super::TreeTN;

// ============================================================================
// Sub-tree extraction
// ============================================================================

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Extract a sub-tree from this TreeTN.
    ///
    /// Creates a new TreeTN containing only the specified nodes and their
    /// connecting edges. Tensors are cloned into the new TreeTN.
    ///
    /// # Arguments
    /// * `node_names` - The names of nodes to include in the sub-tree
    ///
    /// # Returns
    /// A new TreeTN containing the specified sub-tree, or an error if:
    /// - Any specified node doesn't exist
    /// - The specified nodes don't form a connected subtree
    ///
    /// # Notes
    /// - Bond indices between included nodes are preserved
    /// - Bond indices to excluded nodes become external (site) indices in the sub-tree
    /// - ortho_towards directions are copied for edges within the sub-tree
    /// - canonical_center is intersected with the extracted nodes
    pub fn extract_subtree(&self, node_names: &[V]) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug,
        V: Ord,
    {
        if node_names.is_empty() {
            return Err(anyhow::anyhow!("Cannot extract empty subtree"));
        }

        // Validate all nodes exist
        for name in node_names {
            if self.graph.node_index(name).is_none() {
                return Err(anyhow::anyhow!("Node {:?} does not exist", name))
                    .context("extract_subtree: invalid node name");
            }
        }

        // Check connectivity: the specified nodes must form a connected subtree
        let node_indices: HashSet<_> = node_names
            .iter()
            .filter_map(|n| self.graph.node_index(n))
            .collect();

        if !self.site_index_network.is_connected_subset(&node_indices) {
            return Err(anyhow::anyhow!(
                "Specified nodes do not form a connected subtree"
            ))
            .context("extract_subtree: nodes must be connected");
        }

        let node_name_set: HashSet<V> = node_names.iter().cloned().collect();

        // Create new TreeTN with extracted tensors
        let mut subtree = TreeTN::<Id, Symm, V>::new();

        // Step 1: Add all nodes with their tensors
        for name in node_names {
            let node_idx = self.graph.node_index(name).unwrap();
            let tensor = self
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", name))?
                .clone();

            subtree
                .add_tensor(name.clone(), tensor)
                .context("extract_subtree: failed to add tensor")?;
        }

        // Step 2: Add edges between nodes in the subtree
        // Track which edges we've already added to avoid duplicates
        let mut added_edges: HashSet<(V, V)> = HashSet::new();

        for name in node_names {
            let neighbors: Vec<V> = self.site_index_network.neighbors(name).collect();

            for neighbor in neighbors {
                // Only add edge if neighbor is also in the subtree
                if !node_name_set.contains(&neighbor) {
                    continue;
                }

                // Avoid adding the same edge twice (undirected)
                let edge_key = if *name < neighbor {
                    (name.clone(), neighbor.clone())
                } else {
                    (neighbor.clone(), name.clone())
                };

                if added_edges.contains(&edge_key) {
                    continue;
                }
                added_edges.insert(edge_key);

                // Get bond index from original TreeTN
                let orig_edge = self
                    .edge_between(name, &neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?}", name, neighbor))?;

                let bond_index = self
                    .bond_index(orig_edge)
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found"))?
                    .clone();

                // Get node indices in new subtree
                let subtree_node_a = subtree
                    .graph
                    .node_index(name)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", name))?;
                let subtree_node_b = subtree
                    .graph
                    .node_index(&neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", neighbor))?;

                // Connect in subtree
                subtree
                    .connect(subtree_node_a, &bond_index, subtree_node_b, &bond_index)
                    .context("extract_subtree: failed to connect nodes")?;

                // Copy ortho_towards if it exists (keyed by bond index ID)
                if let Some(ortho_dir) = self.ortho_towards.get(&bond_index.id) {
                    // Only copy if the direction node is in the subtree
                    if node_name_set.contains(ortho_dir) {
                        subtree.ortho_towards.insert(bond_index.id.clone(), ortho_dir.clone());
                    }
                }
            }
        }

        // Step 3: Set canonical_center to intersection with extracted nodes
        let new_center: HashSet<V> = self
            .canonical_center
            .intersection(&node_name_set)
            .cloned()
            .collect();
        subtree.canonical_center = new_center;

        // Copy canonical_form if any center nodes were included
        if !subtree.canonical_center.is_empty() {
            subtree.canonical_form = self.canonical_form;
        }

        Ok(subtree)
    }
}

// ============================================================================
// Sub-tree replacement
// ============================================================================

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Replace a sub-tree with another TreeTN of the same appearance.
    ///
    /// This method replaces the tensors and ortho_towards directions for a subset
    /// of nodes with those from another TreeTN. The replacement TreeTN must have
    /// the same "appearance" as the sub-tree being replaced.
    ///
    /// # Arguments
    /// * `node_names` - The names of nodes to replace
    /// * `replacement` - The TreeTN to use as replacement
    ///
    /// # Returns
    /// `Ok(())` if the replacement succeeds, or an error if:
    /// - Any specified node doesn't exist
    /// - The replacement doesn't have the same appearance as the extracted sub-tree
    /// - Tensor replacement fails
    ///
    /// # Notes
    /// - The replacement TreeTN must have the same nodes, edges, site indices, and ortho_towards
    /// - Bond dimensions may differ (this is the typical use case for truncation)
    /// - The original TreeTN is modified in-place
    pub fn replace_subtree(&mut self, node_names: &[V], replacement: &Self) -> Result<()>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug,
        Symm: Clone + Symmetry + PartialEq + std::fmt::Debug,
        V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        if node_names.is_empty() {
            return Ok(()); // Nothing to replace
        }

        // Extract current subtree for comparison
        let current_subtree = self.extract_subtree(node_names)?;

        // Verify that replacement has the same appearance
        if !current_subtree.same_appearance(replacement) {
            return Err(anyhow::anyhow!(
                "Replacement TreeTN does not have the same appearance as the current subtree"
            ))
            .context("replace_subtree: appearance mismatch");
        }

        // Replace tensors
        for name in node_names {
            let self_node_idx = self
                .graph
                .node_index(name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", name))?;
            let replacement_node_idx = replacement
                .graph
                .node_index(name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in replacement", name))?;

            let new_tensor = replacement
                .tensor(replacement_node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?} in replacement", name))?
                .clone();

            self.replace_tensor(self_node_idx, new_tensor)
                .with_context(|| format!("replace_subtree: failed to replace tensor at node {:?}", name))?;
        }

        // Replace ortho_towards for edges within the subtree
        let node_name_set: HashSet<V> = node_names.iter().cloned().collect();
        let mut processed_edges: HashSet<(V, V)> = HashSet::new();

        for name in node_names {
            let neighbors: Vec<V> = self.site_index_network.neighbors(name).collect();

            for neighbor in neighbors {
                if !node_name_set.contains(&neighbor) {
                    continue;
                }

                let edge_key = if *name < neighbor {
                    (name.clone(), neighbor.clone())
                } else {
                    (neighbor.clone(), name.clone())
                };

                if processed_edges.contains(&edge_key) {
                    continue;
                }
                processed_edges.insert(edge_key);

                // Get bond index ID from self edge
                let self_edge = self
                    .edge_between(name, &neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Edge not found"))?;
                let bond_id = self
                    .bond_index(self_edge)
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found"))?
                    .id
                    .clone();

                // Copy ortho_towards from replacement (keyed by bond index ID)
                match replacement.ortho_towards.get(&bond_id) {
                    Some(dir) => {
                        self.ortho_towards.insert(bond_id, dir.clone());
                    }
                    None => {
                        self.ortho_towards.remove(&bond_id);
                    }
                }
            }
        }

        // Update canonical_center: remove old nodes, add from replacement
        for name in node_names {
            self.canonical_center.remove(name);
        }
        for name in &replacement.canonical_center {
            if node_name_set.contains(name) {
                self.canonical_center.insert(name.clone());
            }
        }

        // Update canonical_form if replacement has one
        if replacement.canonical_form.is_some() {
            self.canonical_form = replacement.canonical_form;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tensor4all::index::{DefaultIndex as Index, DynId};
    use tensor4all::storage::{DenseStorageF64, Storage};
    use tensor4all::{NoSymmSpace, TensorDynLen};

    /// Create a 4-node Y-shape TreeTN:
    ///     A
    ///     |
    ///     B
    ///    / \
    ///   C   D
    fn create_y_shape_treetn() -> (
        TreeTN<DynId, NoSymmSpace, String>,
        Index<DynId>,
        Index<DynId>,
        Index<DynId>,
        Index<DynId>,
    ) {
        let mut tn = TreeTN::<DynId, NoSymmSpace, String>::new();

        let site_a = Index::new_dyn(2);
        let site_c = Index::new_dyn(2);
        let site_d = Index::new_dyn(2);
        let bond_ab = Index::new_dyn(3);
        let bond_bc = Index::new_dyn(3);
        let bond_bd = Index::new_dyn(3);

        // Tensor A: [site_a, bond_ab]
        let tensor_a = TensorDynLen::new(
            vec![site_a.clone(), bond_ab.clone()],
            vec![2, 3],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6]))),
        );
        tn.add_tensor("A".to_string(), tensor_a).unwrap();

        // Tensor B: [bond_ab, bond_bc, bond_bd]
        let tensor_b = TensorDynLen::new(
            vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
            vec![3, 3, 3],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 27]))),
        );
        tn.add_tensor("B".to_string(), tensor_b).unwrap();

        // Tensor C: [bond_bc, site_c]
        let tensor_c = TensorDynLen::new(
            vec![bond_bc.clone(), site_c.clone()],
            vec![3, 2],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6]))),
        );
        tn.add_tensor("C".to_string(), tensor_c).unwrap();

        // Tensor D: [bond_bd, site_d]
        let tensor_d = TensorDynLen::new(
            vec![bond_bd.clone(), site_d.clone()],
            vec![3, 2],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6]))),
        );
        tn.add_tensor("D".to_string(), tensor_d).unwrap();

        // Connect
        let n_a = tn.node_index(&"A".to_string()).unwrap();
        let n_b = tn.node_index(&"B".to_string()).unwrap();
        let n_c = tn.node_index(&"C".to_string()).unwrap();
        let n_d = tn.node_index(&"D".to_string()).unwrap();

        tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
        tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
        tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();

        (tn, site_a, site_c, site_d, bond_ab)
    }

    #[test]
    fn test_extract_subtree_single_node() {
        let (tn, site_a, _, _, _) = create_y_shape_treetn();

        // Extract just node A
        let subtree = tn.extract_subtree(&["A".to_string()]).unwrap();

        assert_eq!(subtree.node_count(), 1);
        assert_eq!(subtree.edge_count(), 0);

        // Should have site_a as external index plus bond_ab (which becomes external)
        let n_a = subtree.node_index(&"A".to_string()).unwrap();
        let tensor_a = subtree.tensor(n_a).unwrap();
        assert_eq!(tensor_a.indices.len(), 2);
    }

    #[test]
    fn test_extract_subtree_two_nodes() {
        let (tn, _, _, _, _) = create_y_shape_treetn();

        // Extract A-B subtree
        let subtree = tn
            .extract_subtree(&["A".to_string(), "B".to_string()])
            .unwrap();

        assert_eq!(subtree.node_count(), 2);
        assert_eq!(subtree.edge_count(), 1);

        // Verify connectivity
        let n_a = subtree.node_index(&"A".to_string()).unwrap();
        let n_b = subtree.node_index(&"B".to_string()).unwrap();
        assert!(subtree.edge_between(&"A".to_string(), &"B".to_string()).is_some());
    }

    #[test]
    fn test_extract_subtree_disconnected_fails() {
        let (tn, _, _, _, _) = create_y_shape_treetn();

        // Try to extract A and C (not connected)
        let result = tn.extract_subtree(&["A".to_string(), "C".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_subtree_preserves_consistency() {
        let (tn, _, _, _, _) = create_y_shape_treetn();

        // Extract B-C-D subtree
        let subtree = tn
            .extract_subtree(&["B".to_string(), "C".to_string(), "D".to_string()])
            .unwrap();

        // Verify consistency
        subtree.verify_internal_consistency().unwrap();
    }

    #[test]
    fn test_replace_subtree_same_appearance() {
        let (mut tn, _, _, _, _) = create_y_shape_treetn();

        // Extract subtree, modify tensor data (but keep same structure), replace
        let subtree = tn.extract_subtree(&["C".to_string()]).unwrap();

        // Replace with itself (should work)
        tn.replace_subtree(&["C".to_string()], &subtree).unwrap();

        // Verify consistency
        tn.verify_internal_consistency().unwrap();
    }

    #[test]
    fn test_replace_subtree_two_nodes() {
        let (mut tn, _, _, _, _) = create_y_shape_treetn();

        // Extract C-D subtree (through B)... wait, C and D are not connected.
        // Let's use B-C subtree instead.
        let subtree = tn
            .extract_subtree(&["B".to_string(), "C".to_string()])
            .unwrap();

        // Replace with itself
        tn.replace_subtree(&["B".to_string(), "C".to_string()], &subtree)
            .unwrap();

        // Verify consistency
        tn.verify_internal_consistency().unwrap();
    }
}
