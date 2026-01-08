//! TensorLike implementation for TreeTN.
//!
//! This module provides the `TensorLike` trait implementation for `TreeTN`,
//! allowing it to be used in contexts that require a generic tensor interface.

use std::hash::Hash;

use tensor4all_core::index::{Index, Symmetry};

use super::TreeTN;

// ============================================================================
// TensorLike implementation for TreeTN
// ============================================================================

impl<Id, Symm, V> tensor4all_core::TensorLike for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    Symm: Clone + Symmetry + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    type Id = Id;
    type Symm = Symm;
    type Tags = tensor4all_core::DefaultTagSet;

    fn external_indices(&self) -> Vec<Index<Self::Id, Self::Symm, Self::Tags>> {
        // Collect all site indices from the site_index_network.
        //
        // For deterministic ordering (as required by the trait):
        // 1. Sort nodes by name (V: Ord)
        // 2. Within each node, sort indices by id (Id: Ord)
        // 3. Flatten into a single Vec

        // Get all node names and sort them
        let mut node_names: Vec<_> = self
            .site_index_network
            .node_names()
            .into_iter()
            .cloned()
            .collect();
        node_names.sort();

        let mut result = Vec::new();

        for node_name in node_names {
            if let Some(site_space) = self.site_index_network.site_space(&node_name) {
                // Collect and sort indices by id
                let mut indices: Vec<_> = site_space
                    .iter()
                    .map(|idx| {
                        Index::new_with_tags(
                            idx.id.clone(),
                            idx.symm.clone(),
                            tensor4all_core::DefaultTagSet::default(),
                        )
                    })
                    .collect();
                indices.sort_by(|a, b| a.id.cmp(&b.id));
                result.extend(indices);
            }
        }

        result
    }

    fn num_external_indices(&self) -> usize {
        // Sum up all site indices across all nodes
        self.site_index_network
            .node_names()
            .iter()
            .filter_map(|name| self.site_index_network.site_space(name))
            .map(|site_space| site_space.len())
            .sum()
    }

    fn to_tensor(&self) -> anyhow::Result<tensor4all_core::TensorDynLen<Self::Id, Self::Symm>> {
        // Use the existing contract_to_tensor method
        self.contract_to_tensor()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // Use the default implementation of tensordot which calls to_tensor
}
