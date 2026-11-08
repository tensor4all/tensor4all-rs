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
use super::memo_table::MemoTable;
use super::{cache_key::CacheKey, flat_index, IndexInt, InnerCache};
use crate::ColMajorArrayRef;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicU64, Ordering};

/// Marker for the left side of an additive site split.
///
/// Side keys are side-tagged at the type level, so a left side cannot stand in
/// where a right side is expected; see [`SideKeys`] and
/// [`MultiIndexCache::evaluate_keyed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeftSide;

/// Marker for the right side of an additive site split.
///
/// The counterpart of [`LeftSide`]; the tag exists only in the type, so
/// swapping the two sides of a keyed request is a compile error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RightSide;

/// One side of an additive site split, encoded with the cache's own keys.
///
/// Built by [`MultiIndexCache::left_keys`] or [`MultiIndexCache::right_keys`],
/// never from a bare integer: the cache owns key construction, so a key always
/// belongs to a validated site list of the cache that made it. The `Side`
/// parameter keeps the two sides apart, and the keys stay bound to the cache
/// instance and to its generation, so [`MultiIndexCache::evaluate_keyed`]
/// rejects keys of another cache or of a cleared cache.
///
/// # Examples
///
/// ```
/// use tensor4all_core::MultiIndexCache;
///
/// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[3, 4]).unwrap();
/// let left = cache.left_keys(&[0], &[vec![2usize]]).unwrap();
/// assert_eq!(left.len(), 1);
/// assert!(!left.is_empty());
/// assert_eq!(left.sites(), &[0]);
/// ```
#[derive(Clone, Debug)]
pub struct SideKeys<Side> {
    cache: u64,
    generation: u64,
    sites: Vec<usize>,
    values: SideKeyValues,
    _side: std::marker::PhantomData<Side>,
}

impl<Side> SideKeys<Side> {
    /// Number of fragments on this side, one per key.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// let side = cache.right_keys(&[1], &[vec![0usize], vec![1]]).unwrap();
    /// assert_eq!(side.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        match &self.values {
            SideKeyValues::U64(values) => values.len(),
            SideKeyValues::U128(values) => values.len(),
        }
    }

    /// Whether this side carries no fragment.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// assert!(cache.left_keys(&[0], &[]).unwrap().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sites this side covers, in fragment order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 3, 4]).unwrap();
    /// let side = cache.left_keys(&[2, 0], &[vec![3usize, 1]]).unwrap();
    /// assert_eq!(side.sites(), &[2, 0]);
    /// ```
    pub fn sites(&self) -> &[usize] {
        &self.sites
    }
}

/// Side keys of one side, as the cache stores them.
#[derive(Clone, Debug)]
pub(crate) enum SideKeyValues {
    U64(Vec<u64>),
    U128(Vec<u128>),
}

/// The integers a side key can be, so one keyed batch serves both widths.
trait SideWord: Copy + Eq + std::hash::Hash {
    /// The cache key of an entry: the two sides of an additive split add.
    fn compose(self, right: Self) -> Self;
}

impl SideWord for u64 {
    fn compose(self, right: Self) -> Self {
        self.wrapping_add(right)
    }
}

impl SideWord for u128 {
    fn compose(self, right: Self) -> Self {
        self.wrapping_add(right)
    }
}

/// Source of the ids that bind side keys to their cache.
static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

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

/// Bytes of hash table a [`MultiIndexCache`] may occupy by default.
///
/// The limit bounds the tables themselves (buckets, entries and control bytes),
/// not a per-entry estimate. When a run needs to keep more distinct points than
/// that, the oldest entries are evicted and re-evaluated if asked again, rather
/// than memory growing without bound.
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
/// Retained storage is bounded by [`DEFAULT_RETAINED_BYTE_LIMIT`] bytes of hash
/// table by default ([`Self::with_retained_byte_limit`] changes it,
/// [`Self::set_retained_byte_limit`] at run time, [`Self::clear`] drops every
/// entry). The entries live in two generations of one table. When the current
/// generation cannot grow within the limit, the older generation is evicted and
/// the current one takes its place, so the most recent insertions always stay
/// and an evicted point is simply evaluated again if asked for;
/// [`Self::evicted_entries`] counts them. [`Self::retained_bytes`] is the size of
/// both tables from their capacities: buckets × (key + value + 1 control byte),
/// plus one group of control bytes. While a table grows, the outgoing table
/// (half the new one's buckets) is alive next to the new one, so the transient
/// peak is at most 1.5 × the limit + 8 bytes (half a control group); a limit of
/// 152 bytes peaks at 236 for `f64` values. Allocator overhead and heap payload
/// owned by values are excluded.
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
    backend: InnerCache<V>,
    local_dims: Vec<usize>,
    /// Process-unique identity, so side keys of one cache cannot key another.
    id: u64,
    /// Advanced by [`Self::clear`]; side keys of an earlier generation are stale.
    generation: u64,
    hits: usize,
    misses: usize,
    retained_byte_limit: usize,
    dropped_inserts: usize,
    evicted_entries: usize,
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
            .field("evicted_entries", &self.evicted_entries)
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

    /// Create an empty cache with an explicit limit on its hash-table bytes.
    ///
    /// At the limit the oldest generation of entries is evicted
    /// ([`Self::evicted_entries`]) and the new entry is stored; an insertion is
    /// skipped ([`Self::dropped_inserts`]) only when the limit is below the
    /// smallest table that holds one entry.
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
    /// // Room for one table of 4 buckets (3 entries) of u64 keys and f64 values.
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::with_retained_byte_limit(&[8], 100).unwrap();
    /// assert_eq!(cache.retained_byte_limit(), 100);
    /// for i in 0..8 {
    ///     cache.insert(&[i], i as f64).unwrap();
    /// }
    /// // The oldest entries were evicted; the latest is still answered.
    /// assert!(cache.retained_bytes() <= 100);
    /// assert_eq!(cache.evicted_entries() + cache.len(), 8);
    /// assert_eq!(cache.get(&[7]).unwrap(), Some(7.0));
    /// assert_eq!(cache.get(&[0]).unwrap(), None);
    /// ```
    pub fn with_retained_byte_limit(
        local_dims: &[usize],
        retained_byte_limit: usize,
    ) -> Result<Self, error::CacheKeyError> {
        let backend = InnerCache::<V>::new(local_dims)?;
        Ok(Self {
            backend,
            local_dims: local_dims.to_vec(),
            id: NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            hits: 0,
            misses: 0,
            retained_byte_limit,
            dropped_inserts: 0,
            evicted_entries: 0,
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
        if n_sites != self.local_dims.len() {
            return Err(error::CacheKeyError::InvalidIndexLength {
                expected: self.local_dims.len(),
                got: n_sites,
            }
            .into());
        }
        macro_rules! evaluate_with {
            ($cache:expr, $coeffs:expr) => {
                evaluate_encoded_batch(
                    $cache
                        .get_mut()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    $coeffs,
                    &self.local_dims,
                    self.retained_byte_limit,
                    &mut self.hits,
                    &mut self.misses,
                    &mut self.dropped_inserts,
                    &mut self.evicted_entries,
                    indices,
                    evaluate,
                )
            };
        }
        match &mut self.backend {
            InnerCache::U64 { cache, coeffs } => evaluate_with!(cache, coeffs),
            InnerCache::U128 { cache, coeffs } => evaluate_with!(cache, coeffs),
            InnerCache::U256 { cache, coeffs } => evaluate_with!(cache, coeffs),
            InnerCache::U512 { cache, coeffs } => evaluate_with!(cache, coeffs),
            InnerCache::U1024 { cache, coeffs } => evaluate_with!(cache, coeffs),
        }
    }

    /// Whether this cache can build the side keys [`Self::evaluate_keyed`]
    /// consumes.
    ///
    /// Only `u64` and `u128` caches can: the wider key types have no side-key
    /// support, so [`Self::left_keys`] and [`Self::right_keys`] return
    /// [`error::CacheKeyError::SideKeyUnsupported`] for them.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let narrow: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// assert!(narrow.supports_side_keys());
    ///
    /// // 129 binary sites need 129 bits, which selects a 256-bit key.
    /// let wide: MultiIndexCache<f64> = MultiIndexCache::new(&vec![2; 129]).unwrap();
    /// assert!(!wide.supports_side_keys());
    /// ```
    pub fn supports_side_keys(&self) -> bool {
        self.backend.supports_side_keys()
    }

    /// Build the cache-owned keys of one side of an additive site split.
    ///
    /// `sites` lists the sites this side covers, in the order the fragments
    /// carry them; `fragments` holds one multi-index fragment per key. Every
    /// site must be distinct and in range, every fragment must have
    /// `sites.len()` coordinates, and every coordinate must be in its site's
    /// local dimension.
    ///
    /// The keys use the cache's own mixed-radix coefficients, and the flat key
    /// of a full point is linear in its coordinates, so an entry's key is the
    /// sum of its two sides' keys. A caller that splits the sites into two
    /// disjoint sets therefore reads a matrix entry with
    /// [`Self::evaluate_keyed`] instead of assembling the point. The result is
    /// tagged [`LeftSide`], so it cannot stand in where a [`RightSide`] is
    /// expected.
    ///
    /// # Errors
    ///
    /// Returns [`error::CacheKeyError::SideKeyUnsupported`] for a cache whose
    /// key width has no side-key support,
    /// [`error::CacheKeyError::SideKeyDegenerate`] when the index space has a
    /// zero local dimension (its later coefficients are all zero, and it holds
    /// no valid point), [`error::CacheKeyError::SideSiteOutOfBounds`] or
    /// [`error::CacheKeyError::RepeatedSideSite`] for an invalid site list,
    /// [`error::CacheKeyError::InvalidIndexLength`] when a fragment does not
    /// match the site list, and [`error::CacheKeyError::IndexOutOfBounds`] for a
    /// coordinate outside its site's local dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// // Sites 0 and 1 with local dimensions 3 and 4.
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[3, 4]).unwrap();
    /// let left = cache
    ///     .left_keys(&[0], &[vec![0usize], vec![1], vec![2]])
    ///     .unwrap();
    /// let right = cache
    ///     .right_keys(&[1], &[vec![0usize], vec![1], vec![2], vec![3]])
    ///     .unwrap();
    ///
    /// // Entry (2, 3) and entry (0, 3); the last request repeats an entry.
    /// let mut evaluated = 0;
    /// let values = cache
    ///     .evaluate_keyed(&left, &right, &[(2, 3), (0, 3), (2, 3)], |misses| {
    ///         evaluated += misses.len();
    ///         Ok(misses.iter().map(|&(i, j)| (i * 4 + j) as f64).collect())
    ///     })
    ///     .unwrap();
    /// assert_eq!(values, vec![11.0, 3.0, 11.0]);
    /// assert_eq!(evaluated, 2, "the repeated entry is evaluated once");
    /// assert_eq!(cache.misses(), 3, "every request is a lookup");
    ///
    /// // The second call is served from the cache.
    /// let values = cache
    ///     .evaluate_keyed(&left, &right, &[(2, 3)], |_| Ok(Vec::new()))
    ///     .unwrap();
    /// assert_eq!(values, vec![11.0]);
    /// assert_eq!(cache.hits(), 1);
    /// ```
    pub fn left_keys(
        &self,
        sites: &[usize],
        fragments: &[Vec<I>],
    ) -> Result<SideKeys<LeftSide>, error::CacheKeyError> {
        self.side_keys(sites, fragments)
    }

    /// Build the cache-owned keys of the other side of an additive site split.
    ///
    /// The mirror image of [`Self::left_keys`] with the [`RightSide`] tag; the
    /// two sides must be disjoint when they are used together. See
    /// [`Self::left_keys`] for the validation and the errors.
    ///
    /// # Errors
    ///
    /// The same conditions as [`Self::left_keys`]:
    /// [`error::CacheKeyError::SideKeyUnsupported`],
    /// [`error::CacheKeyError::SideKeyDegenerate`],
    /// [`error::CacheKeyError::SideSiteOutOfBounds`],
    /// [`error::CacheKeyError::RepeatedSideSite`],
    /// [`error::CacheKeyError::InvalidIndexLength`], and
    /// [`error::CacheKeyError::IndexOutOfBounds`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 3, 4]).unwrap();
    /// let right = cache.right_keys(&[2, 0], &[vec![3usize, 1]]).unwrap();
    /// assert_eq!(right.len(), 1);
    /// assert_eq!(right.sites(), &[2, 0]);
    /// ```
    pub fn right_keys(
        &self,
        sites: &[usize],
        fragments: &[Vec<I>],
    ) -> Result<SideKeys<RightSide>, error::CacheKeyError> {
        self.side_keys(sites, fragments)
    }

    /// Evaluate the entries of a keyed request list and return values in
    /// request order.
    ///
    /// `requests` pairs one [`LeftSide`] fragment index with one [`RightSide`]
    /// fragment index, in any order; a repeated entry is evaluated once. The
    /// `evaluate` callback receives the distinct missing entries, in
    /// first-occurrence order, as fragment indices, and returns one value per
    /// entry. It may borrow mutable or thread-affine state; no `Send`/`Sync`
    /// bound is imposed on the callback.
    ///
    /// The keys are authoritative, not a lookup hint: they come from this
    /// cache, they are side-tagged, and the two sides are disjoint, so two
    /// distinct points cannot share a key (repeated fragments produce repeated
    /// entries, and a repeated entry is evaluated once). No point is
    /// re-validated on a hit, and the insert stores the value under the composed
    /// key. A site that neither side covers contributes zero to the composed
    /// key, so sides that leave sites out address the point that is zero there.
    /// A cache with a zero local dimension holds no valid point and builds no
    /// side keys.
    ///
    /// Hits and misses count lookups, so a duplicate within a cold batch is a
    /// miss. Inserts occur only after the whole callback succeeds; a failed
    /// callback retains no new entries but keeps the lookups already counted.
    /// The retained-byte limit and [`Self::dropped_inserts`] apply as in
    /// [`Self::insert`].
    ///
    /// A left key used as a right key is a compile error:
    ///
    /// ```compile_fail
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// let left = cache.left_keys(&[0], &[vec![1usize]]).unwrap();
    /// let right = cache.right_keys(&[1], &[vec![1usize]]).unwrap();
    /// // Swapped sides: `SideKeys<LeftSide>` is not a `SideKeys<RightSide>`.
    /// let _ = cache.evaluate_keyed(&right, &left, &[(0, 0)], |_| Ok(Vec::new()));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CachedBatchError`] for keys of another cache
    /// ([`error::CacheKeyError::SideKeyOwner`]), stale keys of a cleared cache
    /// ([`error::CacheKeyError::SideKeyStale`]), overlapping sides
    /// ([`error::CacheKeyError::OverlappingSideSites`]), a request index outside
    /// its side ([`error::CacheKeyError::SideKeyPosition`]), a callback failure,
    /// or a callback result with the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CacheKeyError, CachedBatchError, MultiIndexCache};
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// let left = cache.left_keys(&[0], &[vec![0usize], vec![1]]).unwrap();
    /// let right = cache.right_keys(&[1], &[vec![0usize], vec![1]]).unwrap();
    /// let requests = [(0, 0), (1, 0), (0, 1), (1, 1)];
    /// let first = cache
    ///     .evaluate_keyed(&left, &right, &requests, |misses| {
    ///         Ok(misses.iter().map(|&(i, j)| (2 * i + j) as f64).collect())
    ///     })
    ///     .unwrap();
    /// assert_eq!(first, vec![0.0, 2.0, 1.0, 3.0]);
    /// assert_eq!(cache.len(), 4);
    ///
    /// // A side that also covers site 1 breaks key composition.
    /// let overlapping = cache.left_keys(&[0, 1], &[vec![0usize, 0]]).unwrap();
    /// let error = cache
    ///     .evaluate_keyed(&overlapping, &right, &[(0, 0)], |_| Ok(Vec::new()))
    ///     .unwrap_err();
    /// assert!(matches!(
    ///     error,
    ///     CachedBatchError::Index(CacheKeyError::OverlappingSideSites { site: 1 })
    /// ));
    /// ```
    pub fn evaluate_keyed<F>(
        &mut self,
        left: &SideKeys<LeftSide>,
        right: &SideKeys<RightSide>,
        requests: &[(usize, usize)],
        evaluate: F,
    ) -> Result<Vec<V>, CachedBatchError>
    where
        F: FnOnce(&[(usize, usize)]) -> anyhow::Result<Vec<V>>,
    {
        self.check_side_pair(left, right)?;
        macro_rules! keyed_with {
            ($cache:expr, $left:expr, $right:expr) => {
                evaluate_keyed_requests(
                    $cache
                        .get_mut()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    $left,
                    $right,
                    requests,
                    self.retained_byte_limit,
                    &mut self.hits,
                    &mut self.misses,
                    &mut self.dropped_inserts,
                    &mut self.evicted_entries,
                    evaluate,
                )
            };
        }
        match (&left.values, &right.values, &mut self.backend) {
            (
                SideKeyValues::U64(left),
                SideKeyValues::U64(right),
                InnerCache::U64 { cache, .. },
            ) => {
                keyed_with!(cache, left, right)
            }
            (
                SideKeyValues::U128(left),
                SideKeyValues::U128(right),
                InnerCache::U128 { cache, .. },
            ) => keyed_with!(cache, left, right),
            // INVARIANT: both sides were built by this cache (checked above), so
            // their widths are this cache's width and always agree.
            _ => Err(error::CacheKeyError::SideKeyOwner.into()),
        }
    }

    fn side_keys<Side>(
        &self,
        sites: &[usize],
        fragments: &[Vec<I>],
    ) -> Result<SideKeys<Side>, error::CacheKeyError> {
        // A zero local dimension makes every coefficient after it zero, so
        // distinct fragments would share a key; and no point of such a space is
        // valid, so no key of it can be authoritative.
        if let Some(site) = self.local_dims.iter().position(|&dim| dim == 0) {
            return Err(error::CacheKeyError::SideKeyDegenerate { site });
        }
        validate_side_sites(&self.local_dims, sites)?;
        for fragment in fragments {
            if fragment.len() != sites.len() {
                return Err(error::CacheKeyError::InvalidIndexLength {
                    expected: sites.len(),
                    got: fragment.len(),
                });
            }
            for (&site, &value) in sites.iter().zip(fragment.iter()) {
                let value = value.to_usize();
                let dim = self.local_dims[site];
                if value >= dim {
                    return Err(error::CacheKeyError::IndexOutOfBounds {
                        axis: site,
                        index: value,
                        dim,
                    });
                }
            }
        }
        let values = self.backend.side_key_values(sites, fragments).ok_or(
            error::CacheKeyError::SideKeyUnsupported {
                key_type: self.backend.key_type_name(),
            },
        )?;
        Ok(SideKeys {
            cache: self.id,
            generation: self.generation,
            sites: sites.to_vec(),
            values,
            _side: std::marker::PhantomData,
        })
    }

    /// The keys are authoritative only if they are this cache's, of its current
    /// generation, and of two disjoint sides.
    ///
    /// One pass over each site list: a site count is bounded only by the key
    /// width, and a scan per left site would cost `n_left * n_right`.
    fn check_side_pair(
        &self,
        left: &SideKeys<LeftSide>,
        right: &SideKeys<RightSide>,
    ) -> Result<(), error::CacheKeyError> {
        if left.cache != self.id || right.cache != self.id {
            return Err(error::CacheKeyError::SideKeyOwner);
        }
        if left.generation != self.generation || right.generation != self.generation {
            return Err(error::CacheKeyError::SideKeyStale);
        }
        let mut covered = vec![false; self.local_dims.len()];
        for &site in &left.sites {
            covered[site] = true;
        }
        for &site in &right.sites {
            if covered[site] {
                return Err(error::CacheKeyError::OverlappingSideSites { site });
            }
        }
        Ok(())
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
        let inserted = self
            .backend
            .insert_bounded(idx, value, self.retained_byte_limit);
        self.evicted_entries += inserted.evicted;
        if !inserted.stored {
            self.dropped_inserts += 1;
        }
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
    /// The cache's generation advances, so side keys built before this call are
    /// stale and [`Self::evaluate_keyed`] rejects them.
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
        self.generation += 1;
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

    /// Bytes of the cache's hash tables: buckets × (key + value + 1 control
    /// byte) plus one group of control bytes, for both generations.
    ///
    /// Computed from the tables' capacities, so it never undercounts the
    /// buckets they hold. It excludes allocator overhead and any heap payload
    /// reachable from a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    /// assert_eq!(cache.retained_bytes(), 0);
    /// cache.insert(&[0, 0], 1.0).unwrap();
    /// // the first table has 4 buckets of a u64 key, an f64 value and a control byte
    /// assert_eq!(cache.retained_bytes(), 4 * (8 + 8 + 1) + 16);
    /// ```
    pub fn retained_bytes(&self) -> usize {
        self.backend.bytes()
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

    /// Limit in bytes of hash table that this cache enforces.
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

    /// Change the table byte limit at run time.
    ///
    /// Shrinking the limit below [`Self::retained_bytes`] evicts nothing at once;
    /// the next new entry evicts generations until the tables fit.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    /// cache.insert(&[0], 1.0).unwrap();
    /// cache.set_retained_byte_limit(0);
    /// assert_eq!(cache.len(), 1); // nothing evicted yet
    /// cache.insert(&[1], 2.0).unwrap(); // evicts [0]; no table fits in 0 bytes
    /// assert_eq!((cache.len(), cache.evicted_entries(), cache.dropped_inserts()), (0, 1, 1));
    /// ```
    pub fn set_retained_byte_limit(&mut self, limit: usize) {
        self.retained_byte_limit = limit;
    }

    /// Number of insertions skipped because the limit is below the smallest
    /// table that holds one entry.
    ///
    /// A skipped insertion only means the point will be evaluated again if it is
    /// requested later; it never makes a lookup return a wrong value. A cache at
    /// its limit evicts instead (see [`Self::evicted_entries`]).
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

    /// Number of entries evicted to keep the tables within the limit.
    ///
    /// A saturated cache evicts its oldest generation of entries each time the
    /// current one fills up; an evicted point is evaluated again if requested.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MultiIndexCache;
    ///
    /// let mut cache: MultiIndexCache<f64> = MultiIndexCache::with_retained_byte_limit(&[64], 1024).unwrap();
    /// for i in 0..64 {
    ///     cache.insert(&[i], i as f64).unwrap();
    /// }
    /// assert!(cache.evicted_entries() > 0);
    /// assert_eq!(cache.evicted_entries() + cache.len(), 64);
    /// assert_eq!(cache.get(&[63]).unwrap(), Some(63.0)); // the latest stays
    /// ```
    pub fn evicted_entries(&self) -> usize {
        self.evicted_entries
    }

    fn validate(&self, idx: &[I]) -> Result<(), error::CacheKeyError> {
        validate_index(&self.local_dims, idx)
    }
}

fn validate_index<I: IndexInt>(
    local_dims: &[usize],
    idx: &[I],
) -> Result<(), error::CacheKeyError> {
    if idx.len() != local_dims.len() {
        return Err(error::CacheKeyError::InvalidIndexLength {
            expected: local_dims.len(),
            got: idx.len(),
        });
    }
    for (axis, (&value, &dim)) in idx.iter().zip(local_dims).enumerate() {
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

// INVARIANT: These are disjoint fields of one cache plus its batch and callback;
// grouping the borrowed fields would add an otherwise unused state type.
#[allow(clippy::too_many_arguments)]
fn evaluate_encoded_batch<K, V, I, F>(
    cache: &mut MemoTable<K, V>,
    coeffs: &[K],
    local_dims: &[usize],
    retained_byte_limit: usize,
    hits: &mut usize,
    misses: &mut usize,
    dropped_inserts: &mut usize,
    evicted_entries: &mut usize,
    indices: ColMajorArrayRef<'_, I>,
    evaluate: F,
) -> Result<Vec<V>, CachedBatchError>
where
    K: CacheKey,
    V: Clone,
    I: IndexInt,
    F: FnOnce(ColMajorArrayRef<'_, I>) -> anyhow::Result<Vec<V>>,
{
    let n_sites = local_dims.len();
    let n_points = indices.shape()[1];
    let mut pending = FxHashMap::default();
    let mut miss_keys = Vec::new();
    let mut miss_data = Vec::new();
    let mut slots = Vec::with_capacity(n_points);
    for position in 0..n_points {
        // INVARIANT: The checked 2D view contains exactly n_sites*n_points
        // elements, including empty columns for a zero-site scalar batch.
        let point = &indices.data()[position * n_sites..(position + 1) * n_sites];
        validate_index(local_dims, point)?;
        let key = flat_index(point, coeffs);
        if let Some(value) = cache.get(&key) {
            *hits += 1;
            slots.push(Ok(value.clone()));
        } else {
            *misses += 1;
            let slot = match pending.entry(key.clone()) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let slot = miss_keys.len();
                    entry.insert(slot);
                    miss_keys.push(key);
                    miss_data.extend_from_slice(point);
                    slot
                }
            };
            slots.push(Err(slot));
        }
    }
    let n_misses = miss_keys.len();
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
        for (key, value) in miss_keys.into_iter().zip(&values) {
            let inserted = cache.insert(key, value.clone(), retained_byte_limit);
            *evicted_entries += inserted.evicted;
            if !inserted.stored {
                *dropped_inserts += 1;
            }
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

fn validate_side_sites(local_dims: &[usize], sites: &[usize]) -> Result<(), error::CacheKeyError> {
    let n_sites = local_dims.len();
    let mut seen = vec![false; n_sites];
    for &site in sites {
        if site >= n_sites {
            return Err(error::CacheKeyError::SideSiteOutOfBounds { site, n_sites });
        }
        if seen[site] {
            return Err(error::CacheKeyError::RepeatedSideSite { site });
        }
        seen[site] = true;
    }
    Ok(())
}

/// Serve one keyed request list from one keyed map.
///
/// The mirror image of `evaluate_encoded_batch` for requests whose keys the
/// cache composed from two sides: the keys are authoritative, so nothing is
/// re-validated on a hit, and the caller sees only the distinct missing entries.
#[allow(clippy::too_many_arguments)]
fn evaluate_keyed_requests<K, V, F>(
    cache: &mut MemoTable<K, V>,
    left: &[K],
    right: &[K],
    requests: &[(usize, usize)],
    retained_byte_limit: usize,
    hits: &mut usize,
    misses: &mut usize,
    dropped_inserts: &mut usize,
    evicted_entries: &mut usize,
    evaluate: F,
) -> Result<Vec<V>, CachedBatchError>
where
    K: SideWord,
    V: Clone,
    F: FnOnce(&[(usize, usize)]) -> anyhow::Result<Vec<V>>,
{
    for &(left_index, right_index) in requests {
        if left_index >= left.len() {
            return Err(error::CacheKeyError::SideKeyPosition {
                side: "left",
                index: left_index,
                len: left.len(),
            }
            .into());
        }
        if right_index >= right.len() {
            return Err(error::CacheKeyError::SideKeyPosition {
                side: "right",
                index: right_index,
                len: right.len(),
            }
            .into());
        }
    }
    let mut pending = FxHashMap::default();
    let mut miss_keys = Vec::new();
    let mut miss_entries = Vec::new();
    let mut slots = Vec::with_capacity(requests.len());
    for &(left_index, right_index) in requests {
        let key = left[left_index].compose(right[right_index]);
        if let Some(value) = cache.get(&key) {
            *hits += 1;
            slots.push(Ok(value.clone()));
        } else {
            *misses += 1;
            let slot = match pending.entry(key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let slot = miss_keys.len();
                    entry.insert(slot);
                    miss_keys.push(key);
                    miss_entries.push((left_index, right_index));
                    slot
                }
            };
            slots.push(Err(slot));
        }
    }
    let n_misses = miss_keys.len();
    let values = if n_misses == 0 {
        Vec::new()
    } else {
        let values = evaluate(&miss_entries).map_err(CachedBatchError::Evaluation)?;
        if values.len() != n_misses {
            return Err(error::CacheKeyError::BatchResultLength {
                expected: n_misses,
                got: values.len(),
            }
            .into());
        }
        for (key, value) in miss_keys.into_iter().zip(&values) {
            let inserted = cache.insert(key, value.clone(), retained_byte_limit);
            *evicted_entries += inserted.evicted;
            if !inserted.stored {
                *dropped_inserts += 1;
            }
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

#[cfg(test)]
mod tests;
