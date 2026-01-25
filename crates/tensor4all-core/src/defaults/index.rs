//! Index types for tensor network operations.
//!
//! This module provides the default index types:
//!
//! - [`DynId`]: Runtime identity (UUID-based unique identifier)
//! - [`TagSet`]: Tag set for metadata (Arc-wrapped for cheap cloning)
//! - [`Index`]: Generic index type parameterized by Id and Tags
//! - [`DynIndex`]: Default index type (`Index<DynId, TagSet>`)
//!
//! The `DynIndex` type implements the [`IndexLike`] trait.
//!
//! **Note**: Symmetry (quantum numbers) is not included in the default implementation.
//! For QSpace-compatible indices with non-Abelian symmetries, use a separate concrete type
//! that implements `IndexLike` directly.

use crate::index_like::IndexLike;
use crate::tagset::{DefaultTagSet as InlineTagSet, TagSetError, TagSetIterator, TagSetLike};
use anyhow::Result;
use rand::Rng;
use std::cell::RefCell;
use std::sync::Arc;

/// Runtime ID for ITensors-like dynamic identity.
///
/// Uses UInt128 for extremely low collision probability (see design.md for analysis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DynId(pub u128);

/// Tag set wrapper using `Arc` for efficient cloning.
///
/// This wraps the underlying tag storage in an `Arc` for cheap cloning (reference count increment only).
/// Tags are immutable and shared across indices with the same tag set.
///
/// # Size comparison
/// - Inline storage: 168 bytes (Copy)
/// - `TagSet` (Arc): 8 bytes (Clone only)
///
/// # Example
/// ```
/// use tensor4all_core::index::TagSet;
///
/// let tags = TagSet::from_str("Site,Link").unwrap();
/// assert!(tags.has_tag("Site"));
/// assert!(tags.has_tag("Link"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TagSet(Arc<InlineTagSet>);

impl TagSet {
    /// Create an empty tag set.
    pub fn new() -> Self {
        Self(Arc::new(InlineTagSet::new()))
    }

    /// Create a tag set from a comma-separated string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, TagSetError> {
        Ok(Self(Arc::new(InlineTagSet::from_str(s)?)))
    }

    /// Create a tag set from a slice of tag strings.
    ///
    /// Returns an error if any tag contains a comma (reserved as separator in `from_str`).
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::index::TagSet;
    ///
    /// let tags = TagSet::from_tags(&["Site", "Link"]).unwrap();
    /// assert!(tags.has_tag("Site"));
    /// assert!(tags.has_tag("Link"));
    /// assert_eq!(tags.len(), 2);
    ///
    /// // Comma in tag is an error
    /// assert!(TagSet::from_tags(&["Site,Link"]).is_err());
    /// ```
    pub fn from_tags(tags: &[&str]) -> Result<Self, TagSetError> {
        let mut inner = InlineTagSet::new();
        for tag in tags {
            if tag.contains(',') {
                return Err(TagSetError::TagContainsComma {
                    tag: (*tag).to_string(),
                });
            }
            inner.add_tag(tag)?;
        }
        Ok(Self(Arc::new(inner)))
    }

    /// Check if a tag is present.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.0.has_tag(tag)
    }

    /// Get the number of tags.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the tag set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the inner Arc for advanced use.
    pub fn inner(&self) -> &Arc<InlineTagSet> {
        &self.0
    }
}

impl std::ops::Deref for TagSet {
    type Target = InlineTagSet;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TagSetLike for TagSet {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn capacity(&self) -> usize {
        self.0.capacity()
    }

    fn get(&self, index: usize) -> Option<String> {
        TagSetLike::get(&*self.0, index)
    }

    fn iter(&self) -> TagSetIterator<'_> {
        TagSetLike::iter(&*self.0)
    }

    fn has_tag(&self, tag: &str) -> bool {
        self.0.has_tag(tag)
    }

    fn add_tag(&mut self, tag: &str) -> Result<(), TagSetError> {
        // Arc is immutable, so we need to clone and replace
        let mut inner = *self.0;
        inner.add_tag(tag)?;
        self.0 = Arc::new(inner);
        Ok(())
    }

    fn remove_tag(&mut self, tag: &str) -> bool {
        // Arc is immutable, so we need to clone and replace
        let mut inner = *self.0;
        let removed = inner.remove_tag(tag);
        if removed {
            self.0 = Arc::new(inner);
        }
        removed
    }
}

/// Index with generic identity type `Id` and tag type `Tags`.
///
/// - `Id = DynId` for ITensors-like runtime identity
/// - `Id = ZST marker type` for compile-time-known identity
/// - `Tags = TagSet` for tags (default, Arc-wrapped for cheap cloning)
///
/// **Note**: This default implementation does not include symmetry (quantum numbers).
/// For QSpace-compatible indices with non-Abelian symmetries, use a separate concrete type.
///
/// # Memory Layout
/// With default types (`DynId`, `TagSet`):
/// - Size: 24 bytes (16 + 8)
/// - Tags are shared via `Arc`, so cloning is cheap (reference count increment only)
///
/// **Equality**: Two `Index` values are considered equal if and only if their `id` fields match.
/// Tags are not used for equality comparison.
///
/// # Example
/// ```
/// use tensor4all_core::index::{Index, DynId, TagSet};
///
/// // Create shared tags once
/// let site_tags = TagSet::from_str("Site").unwrap();
///
/// // Share the same tags across many indices (cheap clone)
/// let i1 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
/// let i2 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
/// // i1.tags and i2.tags point to the same Arc
/// ```
#[derive(Debug, Clone)]
pub struct Index<Id, Tags = TagSet> {
    /// The unique identifier for this index.
    pub id: Id,
    /// The dimension (size) of this index.
    pub dim: usize,
    /// The tag set associated with this index.
    pub tags: Tags,
}

impl<Id, Tags> Index<Id, Tags>
where
    Tags: Default,
{
    /// Create a new index with the given identity and dimension.
    pub fn new(id: Id, dim: usize) -> Self {
        Self {
            id,
            dim,
            tags: Tags::default(),
        }
    }

    /// Create a new index with the given identity, dimension, and tags.
    pub fn new_with_tags(id: Id, dim: usize, tags: Tags) -> Self {
        Self { id, dim, tags }
    }

    /// Get the dimension (size) of the index.
    pub fn size(&self) -> usize {
        self.dim
    }

    /// Get a reference to the tags.
    pub fn tags(&self) -> &Tags {
        &self.tags
    }
}

impl<Id, Tags> Index<Id, Tags>
where
    Tags: Default,
{
    /// Create a new index from dimension (convenience constructor).
    pub fn new_with_size(id: Id, size: usize) -> Self {
        Self {
            id,
            dim: size,
            tags: Tags::default(),
        }
    }

    /// Create a new index from dimension and tags.
    pub fn new_with_size_and_tags(id: Id, size: usize, tags: Tags) -> Self {
        Self {
            id,
            dim: size,
            tags,
        }
    }
}

// Constructors for Index with TagSet (default)
impl Index<DynId, TagSet> {
    /// Create a new index with a generated dynamic ID and no tags.
    pub fn new_dyn(size: usize) -> Self {
        Self {
            id: DynId(generate_id()),
            dim: size,
            tags: TagSet::new(),
        }
    }

    /// Create a new index with a generated dynamic ID and shared tags.
    ///
    /// This is the most efficient way to create many indices with the same tags.
    /// The `Arc` is cloned (reference count increment only), not the underlying data.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::index::{Index, DynId, TagSet};
    ///
    /// let site_tags = TagSet::from_str("Site").unwrap();
    /// let i1 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
    /// let i2 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
    /// ```
    pub fn new_dyn_with_tags(size: usize, tags: TagSet) -> Self {
        Self {
            id: DynId(generate_id()),
            dim: size,
            tags,
        }
    }

    /// Create a new index with a generated dynamic ID and a single tag.
    ///
    /// This creates a new `TagSet` with the given tag.
    /// For sharing the same tag across many indices, create the `TagSet`
    /// once and use `new_dyn_with_tags` instead.
    pub fn new_dyn_with_tag(size: usize, tag: &str) -> Result<Self, TagSetError> {
        Ok(Self {
            id: DynId(generate_id()),
            dim: size,
            tags: TagSet::from_str(tag)?,
        })
    }

    /// Create a new bond index with "Link" tag (for SVD, QR, etc.).
    ///
    /// This is a convenience method for creating bond indices commonly used in tensor
    /// decompositions like SVD and QR factorization.
    pub fn new_link(size: usize) -> Result<Self, TagSetError> {
        Self::new_dyn_with_tag(size, "Link")
    }
}

// Equality and Hash implementations: only compare by `id`
impl<Id: PartialEq, Tags> PartialEq for Index<Id, Tags> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<Id: Eq, Tags> Eq for Index<Id, Tags> {}

impl<Id: std::hash::Hash, Tags> std::hash::Hash for Index<Id, Tags> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// Copy implementation: Index is Copy when Id and Tags are both Copy
impl<Id: Copy, Tags: Copy> Copy for Index<Id, Tags> {}

thread_local! {
    /// Thread-local random number generator for ID generation.
    ///
    /// Each thread has its own RNG, similar to ITensors.jl's task-local RNG.
    /// This provides thread-safe ID generation without global synchronization.
    static ID_RNG: RefCell<rand::rngs::ThreadRng> = RefCell::new(rand::thread_rng());
}

/// Generate a unique random ID for dynamic indices (thread-safe).
///
/// Uses thread-local random number generator to generate UInt128 IDs,
/// providing extremely low collision probability (see design.md for analysis).
pub(crate) fn generate_id() -> u128 {
    ID_RNG.with(|rng| rng.borrow_mut().gen())
}

/// Default Index type alias (same as `Index<Id>` with default tags).
///
/// This is provided for convenience and compatibility.
pub type DefaultIndex<Id> = Index<Id, TagSet>;

/// Type alias for backwards compatibility.
pub type DefaultTagSet = TagSet;

// ============================================================================
// DynIndex: Default index type with IndexLike implementation
// ============================================================================

/// Type alias for the default index type with IndexLike bound.
///
/// `DynIndex` uses:
/// - `DynId`: Dynamic identity (UUID-based unique identifier)
/// - `TagSet`: Default tag set for metadata
///
/// This is the recommended index type for most tensor network applications.
/// It does not include symmetry (quantum numbers); for QSpace-compatible indices,
/// use a separate concrete type that implements `IndexLike` directly.
pub type DynIndex = Index<DynId, TagSet>;

impl IndexLike for DynIndex {
    type Id = DynId;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn conj_state(&self) -> crate::ConjState {
        // Default indices are undirected (ITensors.jl-like behavior)
        crate::ConjState::Undirected
    }

    fn conj(&self) -> Self {
        // For undirected indices, conj() is a no-op
        self.clone()
    }

    fn sim(&self) -> Self {
        Index {
            id: DynId(generate_id()),
            dim: self.dim,
            tags: self.tags.clone(),
        }
    }

    fn create_dummy_link_pair() -> (Self, Self) {
        let id = DynId(generate_id());
        let idx1 = Index {
            id,
            dim: 1,
            tags: TagSet::default(),
        };
        let idx2 = Index {
            id,
            dim: 1,
            tags: TagSet::default(),
        };
        (idx1, idx2)
    }
}

impl DynIndex {
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
    pub fn new_bond(dim: usize) -> Result<Self> {
        Index::new_link(dim).map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::thread;

    #[test]
    fn test_id_generation() {
        let id1 = generate_id();
        let id2 = generate_id();
        let id3 = generate_id();

        // IDs should be unique (random generation, not sequential)
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);

        // IDs should be non-zero (very high probability with u128)
        assert_ne!(id1, 0);
        assert_ne!(id2, 0);
        assert_ne!(id3, 0);
    }

    #[test]
    fn test_thread_local_rng_different_seeds() {
        const NUM_THREADS: usize = 4;
        const IDS_PER_THREAD: usize = 100;

        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|_| {
                thread::spawn(|| {
                    (0..IDS_PER_THREAD)
                        .map(|_| generate_id())
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let mut all_ids = HashSet::new();
        for handle in handles {
            let thread_ids = handle.join().unwrap();
            all_ids.extend(thread_ids);
        }

        assert_eq!(
            all_ids.len(),
            NUM_THREADS * IDS_PER_THREAD,
            "All IDs should be unique across threads"
        );
    }

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

    #[test]
    fn test_sim() {
        let tags = TagSet::from_str("Site,x=1").unwrap();
        let i1 = Index::<DynId>::new_dyn_with_tags(5, tags);

        // Create a similar index
        let i2 = i1.sim();

        // Different ID (not equal)
        assert_ne!(i1, i2);
        assert!(!i1.same_id(&i2));

        // Same dimension
        assert_eq!(i1.dim(), i2.dim());

        // Same tags
        assert_eq!(i1.tags, i2.tags);
        assert!(i2.tags.has_tag("Site"));
        assert!(i2.tags.has_tag("x=1"));
    }

    fn _assert_index_like_bounds<I: IndexLike>() {}

    #[test]
    fn test_index_satisfies_index_like() {
        // Compile-time check that DynIndex implements IndexLike
        _assert_index_like_bounds::<DynIndex>();
    }

    #[test]
    fn test_conj_state_undirected() {
        let i: DynIndex = Index::new_dyn(5);
        assert_eq!(i.conj_state(), crate::ConjState::Undirected);
    }

    #[test]
    fn test_conj_undirected_noop() {
        let i: DynIndex = Index::new_dyn(5);
        let i_conj = i.conj();
        // For undirected indices, conj() should be a no-op
        assert_eq!(i, i_conj);
        assert_eq!(i.conj_state(), i_conj.conj_state());
    }

    #[test]
    fn test_is_contractable_undirected() {
        let i1: DynIndex = Index::new_dyn(5);
        let i2 = i1.clone();
        let i3: DynIndex = Index::new_dyn(5);

        // Same index (clone) should be contractable
        assert!(i1.is_contractable(&i2));
        // Different index (different ID) should not be contractable
        assert!(!i1.is_contractable(&i3));
    }

    #[test]
    fn test_is_contractable_same_id_dim() {
        let i1: DynIndex = Index::new_dyn(5);
        let i2 = i1.clone();
        let i3: DynIndex = Index::new_dyn(3);

        // Same ID and dim should be contractable for undirected
        assert!(i1.is_contractable(&i2));
        // Different dim should not be contractable even if same ID
        // (but in practice, different IDs are used)
        assert!(!i1.is_contractable(&i3));
    }
}
