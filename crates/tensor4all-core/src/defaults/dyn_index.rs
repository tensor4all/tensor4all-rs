//! DynIndex: Default index type implementation.
//!
//! This module provides the default `DynIndex` type alias and its `IndexLike` implementation.
//! `DynIndex = Index<DynId, NoSymmSpace, TagSet>` is the recommended index type for most use cases.

use crate::index::{DynId, Index, NoSymmSpace, Symmetry, TagSet};
use crate::index_like::IndexLike;
use anyhow::Result;

/// Type alias for the default index type with IndexLike bound.
///
/// `DynIndex` uses:
/// - `DynId`: Dynamic identity (UUID-based unique identifier)
/// - `NoSymmSpace`: No symmetry (trivial symmetry space)
/// - `TagSet`: Default tag set for metadata
///
/// This is the recommended index type for most tensor network applications.
pub type DynIndex = Index<DynId, NoSymmSpace, TagSet>;

impl IndexLike for DynIndex {
    type Id = DynId;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn dim(&self) -> usize {
        self.symm.total_dim()
    }

    fn new_bond(dim: usize) -> Result<Self> {
        Index::new_link(dim).map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_like_basic() {
        let i: DynIndex = Index::new_dyn(5);

        // Test IndexLike methods
        assert_eq!(i.dim(), 5);

        // Test id() method
        let id = i.id();
        assert_eq!(*id, i.id);
    }

    #[test]
    fn test_index_like_id_methods() {
        let i1: DynIndex = Index::new_dyn(5);
        let i2 = i1.clone();
        let i3: DynIndex = Index::new_dyn(5);

        // same_id should return true for clones
        assert!(i1.same_id(&i2));
        // same_id should return false for different indices
        assert!(!i1.same_id(&i3));

        // has_id should match by ID
        assert!(i1.has_id(i1.id()));
        assert!(i1.has_id(i2.id()));
        assert!(!i1.has_id(i3.id()));
    }

    #[test]
    fn test_index_like_equality() {
        let i1: DynIndex = Index::new_dyn(5);
        let i2 = i1.clone();
        let i3: DynIndex = Index::new_dyn(5);

        // Same index (cloned) should be equal
        assert_eq!(i1, i2);
        // Different index (new ID) should not be equal
        assert_ne!(i1, i3);
    }

    #[test]
    fn test_index_like_in_hashset() {
        use std::collections::HashSet;

        let i1: DynIndex = Index::new_dyn(5);
        let i2 = i1.clone();
        let i3: DynIndex = Index::new_dyn(5);

        let mut set = HashSet::new();
        set.insert(i1.clone());

        // Clone of same index should be found
        assert!(set.contains(&i2));
        // Different index should not be found
        assert!(!set.contains(&i3));
    }

    #[test]
    fn test_new_bond() {
        let bond: DynIndex = DynIndex::new_bond(10).unwrap();
        assert_eq!(bond.dim(), 10);

        // Each new_bond creates a unique index
        let bond2: DynIndex = DynIndex::new_bond(10).unwrap();
        assert_ne!(bond, bond2);
    }

    fn _assert_index_like_bounds<I: IndexLike>() {}

    #[test]
    fn test_index_satisfies_index_like() {
        // Compile-time check that DynIndex implements IndexLike
        _assert_index_like_bounds::<DynIndex>();
    }
}
