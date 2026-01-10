//! Index types for tensor network operations.
//!
//! This module provides the default index types:
//!
//! - [`DynId`]: Runtime identity (UUID-based unique identifier)
//! - [`NoSymmSpace`]: No symmetry (trivial symmetry space)
//! - [`TagSet`]: Tag set for metadata (Arc-wrapped for cheap cloning)
//! - [`Index`]: Generic index type parameterized by Id, Symm, Tags
//! - [`DynIndex`]: Default index type (`Index<DynId, NoSymmSpace, TagSet>`)
//!
//! The `DynIndex` type implements the [`IndexLike`] trait.

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

/// Trait for symmetry information (quantum number space).
///
/// This corresponds to the `T` parameter in ITensors.jl's `Index{T}`,
/// where `T = Int` for no symmetry and `T = QNBlocks` for quantum numbers.
pub trait Symmetry: Clone + PartialEq + Eq + std::hash::Hash {
    /// Return the total dimension of the space.
    ///
    /// For no symmetry, this is just the dimension.
    /// For quantum number spaces, this is the sum of all block dimensions.
    fn total_dim(&self) -> usize;
}

/// No symmetry space (corresponds to ITensors.jl's `Index{Int}`).
///
/// This represents a simple index with no quantum number structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NoSymmSpace {
    dim: usize,
}

impl NoSymmSpace {
    /// Create a new no-symmetry space with the given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Get the dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Symmetry for NoSymmSpace {
    fn total_dim(&self) -> usize {
        self.dim
    }
}

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
        let mut inner = (*self.0).clone();
        inner.add_tag(tag)?;
        self.0 = Arc::new(inner);
        Ok(())
    }

    fn remove_tag(&mut self, tag: &str) -> bool {
        // Arc is immutable, so we need to clone and replace
        let mut inner = (*self.0).clone();
        let removed = inner.remove_tag(tag);
        if removed {
            self.0 = Arc::new(inner);
        }
        removed
    }
}

/// Index with generic identity type `Id`, symmetry type `Symm`, and tag type `Tags`.
///
/// - `Id = DynId` for ITensors-like runtime identity
/// - `Id = ZST marker type` for compile-time-known identity
/// - `Symm = NoSymmSpace` for no symmetry (default, corresponds to `Index{Int}` in ITensors.jl)
/// - `Symm = QNSpace` (future) for quantum number spaces (corresponds to `Index{QNBlocks}` in ITensors.jl)
/// - `Tags = TagSet` for tags (default, Arc-wrapped for cheap cloning)
///
/// # Memory Layout
/// With default types (`DynId`, `NoSymmSpace`, `TagSet`):
/// - Size: 32 bytes (16 + 8 + 8)
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
pub struct Index<Id, Symm = NoSymmSpace, Tags = TagSet> {
    pub id: Id,
    pub symm: Symm,
    pub tags: Tags,
}

impl<Id, Symm: Symmetry, Tags> Index<Id, Symm, Tags>
where
    Tags: Default,
{
    /// Create a new index with the given identity and symmetry.
    pub fn new(id: Id, symm: Symm) -> Self {
        Self {
            id,
            symm,
            tags: Tags::default(),
        }
    }

    /// Create a new index with the given identity, symmetry, and tags.
    pub fn new_with_tags(id: Id, symm: Symm, tags: Tags) -> Self {
        Self { id, symm, tags }
    }

    /// Get the total dimension (size) of the index.
    ///
    /// This is computed from the symmetry information.
    pub fn size(&self) -> usize {
        self.symm.total_dim()
    }

    /// Get a reference to the tags.
    pub fn tags(&self) -> &Tags {
        &self.tags
    }
}

impl<Id, Tags> Index<Id, NoSymmSpace, Tags>
where
    Tags: Default,
{
    /// Create a new index with no symmetry from dimension.
    ///
    /// This is a convenience constructor for the common case of no symmetry.
    pub fn new_with_size(id: Id, size: usize) -> Self {
        Self {
            id,
            symm: NoSymmSpace::new(size),
            tags: Tags::default(),
        }
    }

    /// Create a new index with no symmetry from dimension and tags.
    pub fn new_with_size_and_tags(id: Id, size: usize, tags: Tags) -> Self {
        Self {
            id,
            symm: NoSymmSpace::new(size),
            tags,
        }
    }
}

// Constructors for Index with TagSet (default)
impl Index<DynId, NoSymmSpace, TagSet> {
    /// Create a new index with a generated dynamic ID and no tags.
    pub fn new_dyn(size: usize) -> Self {
        Self {
            id: DynId(generate_id()),
            symm: NoSymmSpace::new(size),
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
            symm: NoSymmSpace::new(size),
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
            symm: NoSymmSpace::new(size),
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
impl<Id: PartialEq, Symm, Tags> PartialEq for Index<Id, Symm, Tags> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<Id: Eq, Symm, Tags> Eq for Index<Id, Symm, Tags> {}

impl<Id: std::hash::Hash, Symm, Tags> std::hash::Hash for Index<Id, Symm, Tags> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// Copy implementation: Index is Copy when Id, Symm, and Tags are all Copy
impl<Id: Copy, Symm: Copy, Tags: Copy> Copy for Index<Id, Symm, Tags> {}

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

/// Default Index type alias (same as `Index<Id, Symm>` with default tags).
///
/// This is provided for convenience and compatibility.
pub type DefaultIndex<Id, Symm = NoSymmSpace> = Index<Id, Symm, TagSet>;

/// Type alias for backwards compatibility.
pub type DefaultTagSet = TagSet;

// ============================================================================
// DynIndex: Default index type with IndexLike implementation
// ============================================================================

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

    fn sim(&self) -> Self {
        Index {
            id: DynId(generate_id()),
            symm: self.symm.clone(),
            tags: self.tags.clone(),
        }
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

        // Same symm
        assert_eq!(i1.symm, i2.symm);
    }

    fn _assert_index_like_bounds<I: IndexLike>() {}

    #[test]
    fn test_index_satisfies_index_like() {
        // Compile-time check that DynIndex implements IndexLike
        _assert_index_like_bounds::<DynIndex>();
    }
}
