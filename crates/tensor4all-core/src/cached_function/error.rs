//! Error types for cache key operations.
//!
//! [`CacheKeyError`] is returned when constructing a [`CachedFunction`](crate::CachedFunction)
//! fails due to an index space that exceeds the key type's capacity.

use thiserror::Error;

/// Errors that can occur during cache key computation.
#[derive(Debug, Error)]
pub enum CacheKeyError {
    /// The index space requires more bits than the key type supports.
    #[error(
        "Cache key overflow: {total_bits} bits required, but {key_type} supports only \
         {max_bits} bits. Use CachedFunction::with_key_type::<LargerType>() to specify \
         a larger key type."
    )]
    Overflow {
        /// Total bits required by the index space.
        total_bits: u32,
        /// Maximum bits the key type supports.
        max_bits: u32,
        /// Name of the key type.
        key_type: &'static str,
    },

    /// Tensor has wrong number of dimensions for batch evaluation.
    #[error("Expected 2D tensor for batch evaluation, got {ndim}D")]
    InvalidTensorDim {
        /// Actual number of dimensions.
        ndim: usize,
    },

    /// An evaluation index has the wrong rank.
    #[error("Invalid index rank: expected {expected}, got {got}")]
    InvalidIndexLength {
        /// Number of configured local dimensions.
        expected: usize,
        /// Number of coordinates supplied by the caller.
        got: usize,
    },

    /// An evaluation coordinate is outside its local dimension.
    #[error("Index coordinate {index} at axis {axis} is out of range for dimension {dim}")]
    IndexOutOfBounds {
        /// Axis containing the invalid coordinate.
        axis: usize,
        /// Invalid coordinate.
        index: usize,
        /// Valid exclusive upper bound.
        dim: usize,
    },

    /// A batch callback returned a result vector with the wrong length.
    #[error("Batch callback returned {got} values for {expected} indices")]
    BatchResultLength {
        /// Number of cache misses requested.
        expected: usize,
        /// Number of values returned by the callback.
        got: usize,
    },

    /// The cache's key width cannot encode one side of a site split.
    #[error(
        "a {key_type} cache has no side-key support: only u64 and u128 caches build side keys"
    )]
    SideKeyUnsupported {
        /// Name of the cache's selected key type.
        key_type: &'static str,
    },

    /// The cache's index space has an empty site, so it holds no point at all.
    #[error("a cache with local dimension 0 at site {site} has no points and no side keys")]
    SideKeyDegenerate {
        /// Site whose local dimension is zero.
        site: usize,
    },

    /// A side's site list names a site outside the cache's index space.
    #[error("side site {site} is out of range for {n_sites} sites")]
    SideSiteOutOfBounds {
        /// Invalid site.
        site: usize,
        /// Number of sites in the cache's index space.
        n_sites: usize,
    },

    /// A side's site list names the same site twice.
    #[error("side names site {site} more than once")]
    RepeatedSideSite {
        /// Repeated site.
        site: usize,
    },

    /// Side keys were built by a different cache.
    #[error("side keys belong to a different cache")]
    SideKeyOwner,

    /// Side keys were built before the cache's generation advanced.
    #[error("side keys are stale: the cache was cleared after they were built")]
    SideKeyStale,

    /// The two sides of a keyed request cover the same site.
    #[error("the left and right sides of a keyed request both cover site {site}")]
    OverlappingSideSites {
        /// Site covered by both sides.
        site: usize,
    },

    /// A keyed request addresses a fragment outside its side.
    #[error("{side} fragment {index} is out of range for {len} fragments")]
    SideKeyPosition {
        /// Side named by the request: `"left"` or `"right"`.
        side: &'static str,
        /// Index addressed by the request.
        index: usize,
        /// Number of fragments on that side.
        len: usize,
    },
}
