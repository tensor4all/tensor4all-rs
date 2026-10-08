//! Caller-driven mixed-radix cache for multi-index targets.
//!
//! [`MultiIndexCache`] is the fallible, callback-free counterpart of
//! [`CachedFunction`](super::CachedFunction). It encodes multi-indices as
//! mixed-radix flat integers with the same machinery, but it stores no
//! evaluation callback: the caller looks values up, evaluates the misses, and
//! inserts the successful results.
//!
//! Use it when the target cannot be a
//! [`CachedFunction`](super::CachedFunction) callback, for example because it
//! returns `Result`, borrows a value that is neither `Send` nor `Sync` (such as
//! a language binding's interpreter handle), or must never have a failed
//! evaluation cached.

use super::error;
use super::{CacheBackend, IndexInt};
use crate::ColMajorArrayRef;

/// Failure while evaluating a batch through [`MultiIndexCache`].
///
/// Preserves invalid-index and callback diagnostics. Failed batches insert no
/// new values, so a caller can correct the input or retry the callback.
///
/// # Examples
/// ```
/// use tensor4all_core::{ColMajorArrayRef, MultiIndexCache};
/// let mut cache = MultiIndexCache::<f64>::new(&[2])?;
/// let error = cache.evaluate_batched(ColMajorArrayRef::new(&[0], &[1, 1])?,
///     |_| anyhow::bail!("unavailable")) .unwrap_err();
/// assert_eq!(error.to_string(), "unavailable");
/// assert!(cache.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CachedBatchError {
    /// The batch rank, site count, coordinates, or output count is invalid.
    #[error(transparent)]
    Index(#[from] error::CacheKeyError),
    /// A batch view cannot represent the supplied flat storage.
    #[error(transparent)]
    Shape(#[from] crate::col_major_array::ColMajorArrayError),
    /// The target callback failed; its complete source chain is retained.
    #[error("{0}")]
    Evaluation(#[source] anyhow::Error),
}

/// Logical payload bytes a [`MultiIndexCache`] retains by default.
///
/// The limit bounds the sum of one key and one value per entry. When a run
/// needs to keep more distinct points than that, its memoization degrades to
/// re-evaluation rather than unbounded memory growth.
pub const DEFAULT_RETAINED_BYTE_LIMIT: usize = 256 * 1024 * 1024;

/// A persistent cache that maps multi-indices to values through mixed-radix
/// flat-integer keys, without owning an evaluation callback.
///
/// The cache is owned by its caller and lives as long as the caller keeps it.
/// The value type must be `Send + Sync`, and the backend is lock-protected, but
/// the cache stores no evaluation callback, so the target itself may be
/// borrowed, fallible, or thread-affine. Only successful evaluations are
/// stored: an evaluation that fails is never turned into a cached value.
///
/// Retained storage is bounded by
/// [`DEFAULT_RETAINED_BYTE_LIMIT`] logical payload bytes by default
/// ([`Self::with_retained_byte_limit`] changes it, [`Self::set_retained_byte_limit`]
/// at run time, [`Self::clear`] drops every entry). Once the limit is reached,
/// further insertions are skipped instead of evicting entries: the value is
/// simply not cached, the caller re-evaluates that point if it asks again, and
/// [`Self::dropped_inserts`] counts the skipped insertions. The accounting in
/// [`Self::retained_bytes`] is the cache's own logical payload estimate and
/// excludes allocator overhead.
///
/// Key width is selected automatically from the index space (up to 1024 bits,
/// as in [`CachedFunction`](super::CachedFunction)); an index space that needs
/// more bits is rejected by [`MultiIndexCache::new`] with
/// [`error::CacheKeyError::Overflow`]. Keys are the cache's only payload, so
/// [`MultiIndexCache::retained_bytes`] reports a logical estimate of the
/// retained key and value storage.
///
/// # Examples
///
/// ```
/// use tensor4all_core::MultiIndexCache;
///
/// // Two sites with local dimensions 3 and 4.
/// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[3, 4]).unwrap();
///
/// // The caller drives evaluation: look up, evaluate misses, insert results.
/// let mut evaluations = 0;
/// let evaluate = |idx: &[usize]| (idx[0] * 4 + idx[1]) as f64;
/// let mut value = |cache: &mut MultiIndexCache<f64>, idx: &[usize]| {
///     if let Some(v) = cache.get(idx).unwrap() {
///         return v;
///     }
///     let v = evaluate(idx);
///     evaluations += 1;
///     cache.insert(idx, v).unwrap();
///     v
/// };
///
/// assert_eq!(value(&mut cache, &[1, 2]), 6.0);
/// assert_eq!(value(&mut cache, &[1, 2]), 6.0); // cache hit, no re-evaluation
/// assert_eq!(evaluations, 1);
/// assert_eq!(cache.hits(), 1);
/// assert_eq!(cache.misses(), 1);
/// assert_eq!(cache.len(), 1);
/// assert_eq!(cache.local_dims(), &[3, 4]);
/// ```
pub struct MultiIndexCache<V, I = usize>
where
    I: IndexInt,
    V: Clone + Send + Sync + 'static,
{
    backend: CacheBackend<V>,
    local_dims: Vec<usize>,
    key_bytes: usize,
    hits: usize,
    misses: usize,
    retained_byte_limit: usize,
    dropped_inserts: usize,
    _phantom: std::marker::PhantomData<I>,
}

impl<V, I> std::fmt::Debug for MultiIndexCache<V, I>
where
    I: IndexInt,
    V: Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiIndexCache")
            .field("local_dims", &self.local_dims)
            .field("len", &self.backend.len())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .field("retained_byte_limit", &self.retained_byte_limit)
            .field("dropped_inserts", &self.dropped_inserts)
            .field("key_type", &self.backend.key_type_name())
            .finish()
    }
}

impl<V, I> MultiIndexCache<V, I>
where
    I: IndexInt,
    V: Clone + Send + Sync + 'static,
{
    /// Create an empty cache for an index space with the given local dimensions.
    ///
    /// The key width is selected automatically: `u64`, `u128`, then the
    /// extended integers up to 1024 bits.
    ///
    /// # Errors
    ///
    /// Returns [`error::CacheKeyError::Overflow`] when the index space needs more
    /// than 1024 bits, which no built-in key type can represent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// assert!(cache.is_empty());
    /// assert_eq!(cache.key_type(), "u64");
    ///
    /// // 1025 binary sites exceed the widest built-in key.
    /// let too_wide: Result<MultiIndexCache<f64>, _> = MultiIndexCache::new(&vec![2; 1025]);
    /// assert!(too_wide.is_err());
    /// ```
    pub fn new(local_dims: &[usize]) -> Result<Self, error::CacheKeyError> {
        Self::with_retained_byte_limit(local_dims, DEFAULT_RETAINED_BYTE_LIMIT)
    }

    /// Create an empty cache with an explicit logical payload limit.
    ///
    /// Use this to cap the retained key and value storage more tightly than
    /// [`DEFAULT_RETAINED_BYTE_LIMIT`]. Insertions that would exceed the limit
    /// are skipped and counted by [`Self::dropped_inserts`].
    ///
    /// # Errors
    ///
    /// Returns [`error::CacheKeyError::Overflow`] when the index space needs more
    /// than 1024 bits, which no built-in key type can represent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// // Room for exactly one u64 key plus one f64 value.
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::with_retained_byte_limit(&[4], 16).unwrap();
    /// assert_eq!(cache.retained_byte_limit(), 16);
    /// cache.insert(&[0], 1.0).unwrap();
    /// cache.insert(&[1], 2.0).unwrap(); // skipped: the limit is reached
    /// assert_eq!(cache.dropped_inserts(), 1);
    /// assert_eq!(cache.get(&[0]).unwrap(), Some(1.0));
    /// assert_eq!(cache.get(&[1]).unwrap(), None);
    /// ```
    pub fn with_retained_byte_limit(
        local_dims: &[usize],
        retained_byte_limit: usize,
    ) -> Result<Self, error::CacheKeyError> {
        let backend = CacheBackend::Auto(super::InnerCache::<V>::new(local_dims)?);
        let key_bytes = backend.key_bytes();
        Ok(Self {
            backend,
            local_dims: local_dims.to_vec(),
            key_bytes,
            hits: 0,
            misses: 0,
            retained_byte_limit,
            dropped_inserts: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Look a multi-index up and count the lookup.
    ///
    /// Returns `Some(value)` and increments [`Self::hits`] on a hit, or `None`
    /// and increments [`Self::misses`] on a miss. The caller evaluates a miss
    /// and stores it with [`Self::insert`].
    ///
    /// # Errors
    ///
    /// Returns [`error::CacheKeyError::InvalidIndexLength`] when the index rank
    /// differs from the configured dimensions, and
    /// [`error::CacheKeyError::IndexOutOfBounds`] when a coordinate is outside
    /// its local dimension. Coordinates are validated before they are encoded,
    /// so an invalid index can never alias a valid key.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// assert_eq!(cache.get(&[0, 1]).unwrap(), None);
    /// cache.insert(&[0, 1], 5.0).unwrap();
    /// assert_eq!(cache.get(&[0, 1]).unwrap(), Some(5.0));
    /// assert!(cache.get(&[0, 2]).is_err());
    /// assert!(cache.get(&[0]).is_err());
    /// ```
    pub fn get(&mut self, idx: &[I]) -> Result<Option<V>, error::CacheKeyError> {
        self.validate(idx)?;
        match self.backend.get(idx) {
            Some(value) => {
                self.hits += 1;
                Ok(Some(value))
            }
            None => {
                self.misses += 1;
                Ok(None)
            }
        }
    }

    /// Evaluate only distinct cache misses and return values in request order.
    ///
    /// `indices` is a column-major `(local_dims.len(), n_points)` view. The
    /// fallible `evaluate` callback receives the distinct missing points in
    /// first-occurrence order, in one batch. It may borrow mutable or
    /// thread-affine state; no `Send`/`Sync` bound is imposed on the callback.
    /// Empty and all-hit batches do not invoke it. Duplicate misses are
    /// evaluated once even when the retained-byte limit is zero.
    ///
    /// Inserts occur only after the entire callback succeeds with one value
    /// per distinct miss. Failed batches retain no new entries, but lookup
    /// counters still count requests inspected before failure. Hits/misses
    /// count persistent lookups, so duplicates within a cold batch are misses.
    /// Scratch consists of one flat miss buffer, integer keys and result slots,
    /// bounded by the input batch; it is released after this call.
    ///
    /// # Errors
    /// Returns [`CachedBatchError`] for invalid batch shape or coordinates,
    /// callback failure, or a callback result with the wrong length.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, MultiIndexCache};
    /// let mut cache = MultiIndexCache::<f64>::new(&[4])?;
    /// let mut evaluated = 0;
    /// let values = cache.evaluate_batched(
    ///     ColMajorArrayRef::new(&[2, 1, 2], &[1, 3])?, |misses| {
    ///         evaluated += misses.shape()[1];
    ///         Ok(misses.data().iter().map(|&i| 10.0 * i as f64).collect())
    ///     })?;
    /// assert_eq!(values, vec![20.0, 10.0, 20.0]);
    /// assert_eq!(evaluated, 2);
    /// assert_eq!(cache.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn evaluate_batched<F>(
        &mut self,
        indices: ColMajorArrayRef<'_, I>,
        evaluate: F,
    ) -> Result<Vec<V>, CachedBatchError>
    where
        F: FnOnce(ColMajorArrayRef<'_, I>) -> anyhow::Result<Vec<V>>,
    {
        if indices.ndim() != 2 {
            return Err(error::CacheKeyError::InvalidTensorDim {
                ndim: indices.ndim(),
            }
            .into());
        }
        let n_sites = indices.shape()[0];
        let n_points = indices.shape()[1];
        if n_sites != self.local_dims.len() {
            return Err(error::CacheKeyError::InvalidIndexLength {
                expected: self.local_dims.len(),
                got: n_sites,
            }
            .into());
        }
        let pending =
            MultiIndexCache::<usize, I>::with_retained_byte_limit(&self.local_dims, usize::MAX)?;
        let mut miss_data = Vec::new();
        let mut slots = Vec::with_capacity(n_points);
        for position in 0..n_points {
            // INVARIANT: The checked 2D view contains exactly n_sites*n_points
            // elements, including empty columns for a zero-site scalar batch.
            let point = &indices.data()[position * n_sites..(position + 1) * n_sites];
            if let Some(value) = self.get(point)? {
                slots.push(Ok(value));
            } else {
                let slot = match pending.backend.get(point) {
                    Some(slot) => slot,
                    None => {
                        let slot = pending.len();
                        pending.backend.insert(point, slot);
                        miss_data.extend_from_slice(point);
                        slot
                    }
                };
                slots.push(Err(slot));
            }
        }
        let n_misses = pending.len();
        let values = if n_misses == 0 {
            Vec::new()
        } else {
            let shape = [n_sites, n_misses];
            let values = evaluate(ColMajorArrayRef::new(&miss_data, &shape)?)
                .map_err(CachedBatchError::Evaluation)?;
            if values.len() != n_misses {
                return Err(error::CacheKeyError::BatchResultLength {
                    expected: n_misses,
                    got: values.len(),
                }
                .into());
            }
            for (position, value) in values.iter().enumerate() {
                let point = &miss_data[position * n_sites..(position + 1) * n_sites];
                self.insert(point, value.clone())?;
            }
            values
        };
        Ok(slots
            .into_iter()
            .map(|slot| match slot {
                Ok(value) => value,
                Err(position) => values[position].clone(),
            })
            .collect())
    }

    /// Store one successfully evaluated value.
    ///
    /// Insertion does not change the hit and miss counters; those count
    /// lookups. Inserting an index that is already present overwrites the
    /// value.
    ///
    /// # Errors
    ///
    /// Returns [`error::CacheKeyError::InvalidIndexLength`] when the index rank
    /// differs from the configured dimensions, and
    /// [`error::CacheKeyError::IndexOutOfBounds`] when a coordinate is outside
    /// its local dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// cache.insert(&[1, 1], 1.0).unwrap();
    /// cache.insert(&[1, 1], 2.0).unwrap(); // overwrite
    /// assert_eq!(cache.get(&[1, 1]).unwrap(), Some(2.0));
    /// assert_eq!(cache.len(), 1);
    /// ```
    pub fn insert(&mut self, idx: &[I], value: V) -> Result<(), error::CacheKeyError> {
        self.validate(idx)?;
        if !self.backend.contains(idx) {
            let entry_bytes = self.key_bytes + std::mem::size_of::<V>();
            if self.retained_bytes().saturating_add(entry_bytes) > self.retained_byte_limit {
                self.dropped_inserts += 1;
                return Ok(());
            }
        }
        self.backend.insert(idx, value);
        Ok(())
    }

    /// Test whether a multi-index is cached without changing the counters.
    ///
    /// Use this to deduplicate repeated points inside one batch before
    /// evaluating it.
    ///
    /// # Errors
    ///
    /// Returns [`error::CacheKeyError::InvalidIndexLength`] when the index rank
    /// differs from the configured dimensions, and
    /// [`error::CacheKeyError::IndexOutOfBounds`] when a coordinate is outside
    /// its local dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// assert!(!cache.is_cached(&[0, 0]).unwrap());
    /// cache.insert(&[0, 0], 1.0).unwrap();
    /// assert!(cache.is_cached(&[0, 0]).unwrap());
    /// assert_eq!(cache.hits(), 0);
    /// ```
    pub fn is_cached(&self, idx: &[I]) -> Result<bool, error::CacheKeyError> {
        self.validate(idx)?;
        Ok(self.backend.contains(idx))
    }

    /// Number of retained entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// assert!(cache.is_empty());
    /// cache.insert(&[0, 0], 1.0).unwrap();
    /// assert_eq!(cache.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.backend.len()
    }

    /// Whether the cache holds no entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// assert!(cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.backend.len() == 0
    }

    /// Drop every retained entry; counters and dimensions are preserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// cache.insert(&[0], 1.0).unwrap();
    /// cache.clear();
    /// assert_eq!(cache.len(), 0);
    /// assert_eq!(cache.get(&[0]).unwrap(), None);
    /// ```
    pub fn clear(&mut self) {
        self.backend.clear();
    }

    /// Number of lookups that found a cached value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// cache.get(&[0]).unwrap();
    /// cache.insert(&[0], 1.0).unwrap();
    /// cache.get(&[0]).unwrap();
    /// assert_eq!(cache.misses(), 1);
    /// assert_eq!(cache.hits(), 1);
    /// assert_eq!(cache.hit_ratio(), 0.5);
    /// ```
    pub fn hits(&self) -> usize {
        self.hits
    }

    /// Number of lookups that found no cached value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// cache.get(&[1]).unwrap();
    /// assert_eq!(cache.misses(), 1);
    /// ```
    pub fn misses(&self) -> usize {
        self.misses
    }

    /// Fraction of lookups served from the cache, or `0.0` before any lookup.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// assert_eq!(cache.hit_ratio(), 0.0);
    /// ```
    pub fn hit_ratio(&self) -> f64 {
        let lookups = self.hits + self.misses;
        if lookups == 0 {
            0.0
        } else {
            self.hits as f64 / lookups as f64
        }
    }

    /// Logical estimate of the retained payload: one key and one value per entry.
    ///
    /// This is the cache's own accounting, not process memory. It excludes
    /// allocator overhead and any heap payload reachable from a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// assert_eq!(cache.retained_bytes(), 0);
    /// cache.insert(&[0, 0], 1.0).unwrap();
    /// // one u64 key plus one f64 value
    /// assert_eq!(cache.retained_bytes(), 8 + std::mem::size_of::<f64>());
    /// ```
    pub fn retained_bytes(&self) -> usize {
        self.backend.len() * (self.key_bytes + std::mem::size_of::<V>())
    }

    /// The configured local dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[3, 4, 5]).unwrap();
    /// assert_eq!(cache.local_dims(), &[3, 4, 5]);
    /// ```
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Name of the selected key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let narrow: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// assert_eq!(narrow.key_type(), "u64");
    ///
    /// // 65 binary sites need 65 bits.
    /// let wide: MultiIndexCache<f64> = MultiIndexCache::new(&vec![2; 65]).unwrap();
    /// assert_eq!(wide.key_type(), "u128");
    /// ```
    pub fn key_type(&self) -> &'static str {
        self.backend.key_type_name()
    }

    /// Logical payload limit in bytes that this cache enforces.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// assert_eq!(cache.retained_byte_limit(), tensor4all_core::DEFAULT_RETAINED_BYTE_LIMIT);
    /// ```
    pub fn retained_byte_limit(&self) -> usize {
        self.retained_byte_limit
    }

    /// Change the logical payload limit at run time.
    ///
    /// Shrinking the limit below [`Self::retained_bytes`] does not evict
    /// entries; it only prevents further insertions until the cache falls below
    /// the limit again.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// cache.insert(&[0], 1.0).unwrap();
    /// cache.set_retained_byte_limit(0);
    /// cache.insert(&[1], 2.0).unwrap(); // skipped
    /// assert_eq!(cache.len(), 1);
    /// assert_eq!(cache.dropped_inserts(), 1);
    /// ```
    pub fn set_retained_byte_limit(&mut self, limit: usize) {
        self.retained_byte_limit = limit;
    }

    /// Number of insertions skipped because the cache was at its payload limit.
    ///
    /// A skipped insertion only means the point will be evaluated again if it is
    /// requested later; it never makes a lookup return a wrong value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::with_retained_byte_limit(&[2], 0).unwrap();
    /// cache.insert(&[0], 1.0).unwrap();
    /// assert_eq!(cache.dropped_inserts(), 1);
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn dropped_inserts(&self) -> usize {
        self.dropped_inserts
    }

    fn validate(&self, idx: &[I]) -> Result<(), error::CacheKeyError> {
        if idx.len() != self.local_dims.len() {
            return Err(error::CacheKeyError::InvalidIndexLength {
                expected: self.local_dims.len(),
                got: idx.len(),
            });
        }
        for (axis, (&value, &dim)) in idx.iter().zip(&self.local_dims).enumerate() {
            let value = value.to_usize();
            if value >= dim {
                return Err(error::CacheKeyError::IndexOutOfBounds {
                    axis,
                    index: value,
                    dim,
                });
            }
        }
        Ok(())
    }
}
