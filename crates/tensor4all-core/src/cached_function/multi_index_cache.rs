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

/// A persistent cache that maps multi-indices to values through mixed-radix
/// flat-integer keys, without owning an evaluation callback.
///
/// The cache is owned by its caller and lives as long as the caller keeps it.
/// It is not `Sync`; keep it in the single-threaded driver that owns the
/// evaluations. Unlike [`CachedFunction`](super::CachedFunction), neither the
/// value nor the cache requires a `Send + Sync` callback, and only successful
/// evaluations are stored: an evaluation that fails is never turned into a
/// cached value.
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
        let backend = CacheBackend::Auto(super::InnerCache::<V>::new(local_dims)?);
        let key_bytes = backend.key_bytes();
        Ok(Self {
            backend,
            local_dims: local_dims.to_vec(),
            key_bytes,
            hits: 0,
            misses: 0,
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
