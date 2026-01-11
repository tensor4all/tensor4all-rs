//! Operator trait implementation for TreeTN.
//!
//! This module implements the Operator trait for TreeTN, allowing TreeTNs
//! to be used with the operator composition infrastructure.

use std::collections::HashSet;
use std::hash::Hash;

use tensor4all_core::{IndexLike, TensorLike};

use crate::operator::Operator;
use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;

impl<T, V> Operator<T, V> for TreeTN<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn site_indices(&self) -> HashSet<T::Index> {
        // Collect all site indices from all nodes
        let mut result = HashSet::new();
        for node_name in self.site_index_network.node_names() {
            if let Some(site_space) = self.site_index_network.site_space(node_name) {
                result.extend(site_space.iter().cloned());
            }
        }
        result
    }

    fn site_index_network(&self) -> &SiteIndexNetwork<V, T::Index> {
        &self.site_index_network
    }

    fn node_names(&self) -> HashSet<V> {
        self.site_index_network
            .node_names()
            .into_iter()
            .cloned()
            .collect()
    }
}
