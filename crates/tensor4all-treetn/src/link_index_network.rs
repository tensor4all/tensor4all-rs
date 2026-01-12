//! Link Index Network for bond/link index management.
//!
//! Provides efficient reverse lookup from index ID to edge.
//! This complements `SiteIndexNetwork` (which handles site/physical indices)
//! by handling link/bond indices between nodes.

use petgraph::stable_graph::EdgeIndex;
use std::collections::HashMap;
use std::fmt::Debug;
use tensor4all_core::IndexLike;

/// Link Index Network: manages bond/link indices with reverse lookup.
///
/// Provides O(1) lookup from index ID to the edge containing that index.
/// This is essential for efficient `replaceind` operations on TreeTN.
///
/// # Type Parameters
/// - `I`: Index type (must implement `IndexLike`)
#[derive(Debug, Clone)]
pub struct LinkIndexNetwork<I>
where
    I: IndexLike,
{
    /// Reverse lookup: index ID → edge containing this index
    index_to_edge: HashMap<I::Id, EdgeIndex>,
}

impl<I> Default for LinkIndexNetwork<I>
where
    I: IndexLike,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<I> LinkIndexNetwork<I>
where
    I: IndexLike,
{
    /// Create a new empty LinkIndexNetwork.
    pub fn new() -> Self {
        Self {
            index_to_edge: HashMap::new(),
        }
    }

    /// Create with initial capacity.
    pub fn with_capacity(edges: usize) -> Self {
        Self {
            index_to_edge: HashMap::with_capacity(edges),
        }
    }

    /// Register a link index for an edge.
    ///
    /// # Arguments
    /// * `edge` - The edge index
    /// * `index` - The link index on this edge
    pub fn insert(&mut self, edge: EdgeIndex, index: &I) {
        self.index_to_edge.insert(index.id().clone(), edge);
    }

    /// Remove a link index registration.
    ///
    /// # Arguments
    /// * `index` - The link index to remove
    ///
    /// # Returns
    /// The edge that was associated with this index, if any.
    pub fn remove(&mut self, index: &I) -> Option<EdgeIndex> {
        self.index_to_edge.remove(index.id())
    }

    /// Find the edge containing a given index.
    ///
    /// # Arguments
    /// * `index` - The index to look up
    ///
    /// # Returns
    /// The edge containing this index, or None if not found.
    pub fn find_edge(&self, index: &I) -> Option<EdgeIndex> {
        self.index_to_edge.get(index.id()).copied()
    }

    /// Find the edge containing an index by ID.
    pub fn find_edge_by_id(&self, id: &I::Id) -> Option<EdgeIndex> {
        self.index_to_edge.get(id).copied()
    }

    /// Check if an index is registered.
    pub fn contains(&self, index: &I) -> bool {
        self.index_to_edge.contains_key(index.id())
    }

    /// Check if an index ID is registered.
    pub fn contains_id(&self, id: &I::Id) -> bool {
        self.index_to_edge.contains_key(id)
    }

    /// Update the index for an edge (e.g., after SVD creates new bond index).
    ///
    /// # Arguments
    /// * `old_index` - The old index to replace
    /// * `new_index` - The new index
    /// * `edge` - The edge (for validation)
    ///
    /// # Returns
    /// Ok if successful, Err if old_index was not registered or edge mismatch.
    pub fn replace_index(&mut self, old_index: &I, new_index: &I, edge: EdgeIndex) -> Result<(), String> {
        match self.index_to_edge.remove(old_index.id()) {
            Some(old_edge) => {
                if old_edge != edge {
                    // Restore and return error
                    self.index_to_edge.insert(old_index.id().clone(), old_edge);
                    return Err(format!(
                        "Edge mismatch: old_index was on edge {:?}, not {:?}",
                        old_edge, edge
                    ));
                }
                self.index_to_edge.insert(new_index.id().clone(), edge);
                Ok(())
            }
            None => Err(format!("Index {:?} not found in link_index_network", old_index.id())),
        }
    }

    /// Number of registered link indices.
    pub fn len(&self) -> usize {
        self.index_to_edge.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.index_to_edge.is_empty()
    }

    /// Clear all registrations.
    pub fn clear(&mut self) {
        self.index_to_edge.clear();
    }

    /// Iterate over all (index_id, edge) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&I::Id, &EdgeIndex)> {
        self.index_to_edge.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::DynIndex;

    #[test]
    fn test_basic_operations() {
        let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

        let idx1 = DynIndex::new_dyn(4);
        let idx2 = DynIndex::new_dyn(4);
        let edge1 = EdgeIndex::new(0);
        let edge2 = EdgeIndex::new(1);

        network.insert(edge1, &idx1);
        network.insert(edge2, &idx2);

        assert!(network.contains(&idx1));
        assert!(network.contains(&idx2));
        assert_eq!(network.find_edge(&idx1), Some(edge1));
        assert_eq!(network.find_edge(&idx2), Some(edge2));
        assert_eq!(network.len(), 2);
    }

    #[test]
    fn test_replace_index() {
        let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

        let old_idx = DynIndex::new_dyn(4);
        let new_idx = DynIndex::new_dyn(4);
        let edge = EdgeIndex::new(0);

        network.insert(edge, &old_idx);
        assert!(network.contains(&old_idx));

        network.replace_index(&old_idx, &new_idx, edge).unwrap();

        assert!(!network.contains(&old_idx));
        assert!(network.contains(&new_idx));
        assert_eq!(network.find_edge(&new_idx), Some(edge));
    }

    #[test]
    fn test_remove() {
        let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

        let idx = DynIndex::new_dyn(4);
        let edge = EdgeIndex::new(0);

        network.insert(edge, &idx);
        assert_eq!(network.remove(&idx), Some(edge));
        assert!(!network.contains(&idx));
    }
}
