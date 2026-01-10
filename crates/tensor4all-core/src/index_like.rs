//! IndexLike trait for abstracting index types.
//!
//! This trait allows algorithms to be generic over different index types
//! without needing to know about the internal ID representation.

use anyhow::Result;
use std::fmt::Debug;
use std::hash::Hash;

/// Trait for index-like types that can be used in tensor operations.
///
/// This trait abstracts away the identity mechanism of indices, allowing algorithms
/// to work with any index type that provides equality, hashing, and dimension access.
///
/// # Design Principles
///
/// - **`Id` as associated type**: Lightweight identifier (conjugate-independent)
/// - **`Eq` by ID**: Two indices are equal iff their IDs match
/// - **`dim()`**: Returns the dimension of the index
///
/// # Key Properties
///
/// - **`Eq`**: Defines when two indices are considered "the same leg" (for contraction pairing)
/// - **`Hash`**: Enables efficient lookup in `HashMap<I, ...>` / `HashSet<I>`
/// - **`Clone`**: Indices are small value types, freely copyable
///
/// # Invariant
///
/// Two `Index` values are equal (via `Eq`) iff they refer to the same logical leg.
/// Symmetry metadata, direction flags, prime levels, tags, etc. may be carried by the index
/// but **must not affect `Eq` / `Hash`** for ITensors-like semantics (where indices match by ID only).
///
/// # Example
///
/// ```ignore
/// fn contract_common<I: IndexLike>(a: &Tensor<I>, b: &Tensor<I>) -> Tensor<I> {
///     // Algorithm doesn't need to know about Id, Symm, Tags
///     // It only needs indices to be comparable and have dimensions
///     let common: Vec<_> = a.indices().iter()
///         .filter(|idx| b.indices().contains(idx))
///         .cloned()
///         .collect();
///     // ...
/// }
/// ```
pub trait IndexLike: Clone + Eq + Hash + Debug + Send + Sync + 'static {
    /// Lightweight identifier type (conjugate-independent).
    ///
    /// Used as edge labels in TreeTN graphs for efficient matching.
    type Id: Clone + Eq + Hash + Debug + Send + Sync;

    /// Get the identifier of this index.
    ///
    /// The ID is used for matching indices during contraction.
    /// Two indices with the same ID are considered the same logical leg.
    fn id(&self) -> &Self::Id;

    /// Get the total dimension (state-space dimension) of the index.
    fn dim(&self) -> usize;

    /// Check if this index has the same ID as another.
    ///
    /// Default implementation compares IDs directly.
    fn same_id(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    /// Check if this index has the given ID.
    ///
    /// Default implementation compares with the given ID.
    fn has_id(&self, id: &Self::Id) -> bool {
        self.id() == id
    }

    /// Create a new bond index with a fresh identity and the specified dimension.
    ///
    /// This is used by factorization operations (SVD, QR) to create new internal
    /// bond indices connecting the factors.
    ///
    /// # Arguments
    /// * `dim` - The dimension of the new index
    ///
    /// # Returns
    /// A new index with a unique identity and the specified dimension.
    fn new_bond(dim: usize) -> Result<Self>
    where
        Self: Sized;
}

// ============================================================================
// Implementation for the default index type: Index<DynId, NoSymmSpace, TagSet>
// ============================================================================

use crate::index::{DynId, Index, NoSymmSpace, Symmetry, TagSet};

/// Type alias for the default index type with IndexLike bound.
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
