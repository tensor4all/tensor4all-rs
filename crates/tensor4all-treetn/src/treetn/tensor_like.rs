//! TensorIndex and TensorLike implementations for TreeTN.
//!
//! **Design Decision**: TreeTN implements TensorIndex but NOT TensorLike.
//!
//! ## TensorIndex (Implemented)
//!
//! TreeTN implements TensorIndex because index operations are well-defined:
//! - `external_indices()`: Returns all site (physical) indices
//! - `replaceind()` / `replaceinds()`: Replace indices in tensors and metadata
//!
//! ## TensorLike (NOT Implemented)
//!
//! TreeTN does NOT implement TensorLike because:
//! 1. **Unclear semantics**: What would `tensordot` between two TreeTNs mean?
//! 2. **Hidden costs**: Full contraction has exponential cost
//! 3. **Separation of concerns**: Dense tensors and TNs are fundamentally different
//!
//! ## Alternative API
//!
//! TreeTN provides its own methods instead:
//! - `site_indices()`: Returns physical indices (not bonds)
//! - `contract_to_tensor()`: Explicit method for full contraction (exponential cost)
//! - `contract_nodes()`: Graph operations for node contraction

use std::hash::Hash;

use anyhow::Result;
use tensor4all_core::{IndexLike, TensorIndex, TensorLike};

use super::TreeTN;

// ============================================================================
// TensorIndex implementation for TreeTN
// ============================================================================

impl<T, V> TensorIndex for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    type Index = T::Index;

    /// Return all external (site/physical) indices from all nodes.
    ///
    /// This collects all site indices from `site_index_network`.
    /// Bond indices are NOT included (they are internal to the network).
    fn external_indices(&self) -> Vec<Self::Index> {
        let mut result = Vec::new();
        for node_name in self.node_names() {
            if let Some(site_space) = self.site_space(&node_name) {
                result.extend(site_space.iter().cloned());
            }
        }
        result
    }

    fn num_external_indices(&self) -> usize {
        self.node_names()
            .iter()
            .filter_map(|name| self.site_space(name))
            .map(|space| space.len())
            .sum()
    }

    /// Replace an index in this TreeTN.
    ///
    /// Looks up the index location (site or link) and replaces it in:
    /// - The tensor containing it
    /// - The appropriate index network (site_index_network or link_index_network)
    ///
    /// Note: `replace_tensor` automatically updates the `site_index_network` based on
    /// the new tensor's indices, so we don't need to manually call `replace_site_index`.
    fn replaceind(&self, old_index: &Self::Index, new_index: &Self::Index) -> Result<Self> {
        let mut result = self.clone();

        // Check if it's a site index
        if let Some(node_name) = self.site_index_network.find_node_by_index(old_index) {
            let node_idx = result
                .node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))?;

            // Replace in tensor - this also updates site_index_network via replace_tensor
            let tensor = result
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))?;
            let old_in_tensor = tensor
                .external_indices()
                .iter()
                .find(|idx| idx.id() == old_index.id())
                .ok_or_else(|| {
                    anyhow::anyhow!("Index not found in tensor at node {:?}", node_name)
                })?
                .clone();
            let new_tensor = tensor.replaceind(&old_in_tensor, new_index)?;
            result.replace_tensor(node_idx, new_tensor)?;

            // Keep ortho_towards consistent (if present)
            if let Some(dir) = result.ortho_towards.remove(old_index) {
                result.ortho_towards.insert(new_index.clone(), dir);
            }

            return Ok(result);
        }

        // Check if it's a link index
        if let Some(edge) = self.link_index_network.find_edge(old_index) {
            let (node_a, node_b) = result
                .graph
                .graph()
                .edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge {:?} not found", edge))?;

            // IMPORTANT: Update edge weight FIRST so replace_tensor validation matches.
            *result
                .bond_index_mut(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found"))? = new_index.clone();

            // Replace in both endpoint tensors - this also updates site_index_network
            for node in [node_a, node_b] {
                let tensor = result
                    .tensor(node)
                    .ok_or_else(|| anyhow::anyhow!("Tensor not found"))?;
                let old_in_tensor = tensor
                    .external_indices()
                    .iter()
                    .find(|idx| idx.id() == old_index.id())
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found in endpoint tensor"))?
                    .clone();
                let new_tensor = tensor.replaceind(&old_in_tensor, new_index)?;
                result.replace_tensor(node, new_tensor)?;
            }

            // Keep ortho_towards consistent (if present)
            if let Some(dir) = result.ortho_towards.remove(old_index) {
                result.ortho_towards.insert(new_index.clone(), dir);
            }

            // Replace in link_index_network
            result
                .link_index_network
                .replace_index(old_index, new_index, edge)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            return Ok(result);
        }

        Err(anyhow::anyhow!(
            "Index {:?} not found in TreeTN",
            old_index.id()
        ))
    }

    /// Replace multiple indices in this TreeTN.
    fn replaceinds(
        &self,
        old_indices: &[Self::Index],
        new_indices: &[Self::Index],
    ) -> Result<Self> {
        if old_indices.len() != new_indices.len() {
            return Err(anyhow::anyhow!(
                "Length mismatch: {} old indices, {} new indices",
                old_indices.len(),
                new_indices.len()
            ));
        }

        let mut result = self.clone();
        for (old, new) in old_indices.iter().zip(new_indices.iter()) {
            result = result.replaceind(old, new)?;
        }
        Ok(result)
    }
}
