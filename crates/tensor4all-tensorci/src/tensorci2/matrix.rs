//! Matrix-valued chain TCI2.
//!
//! [`crossinterpolate2_matrix`] interpolates a function `x -> F(x)` whose value
//! is a `rows x cols` matrix. The matrix indices become the two boundary sites
//! of an ordinary chain: the internal problem has local dimensions
//! `[rows, d_1, .., d_L, cols]` and `f(i, x, j) = F(x)[i, j]`.
//! The solver therefore sees scalar callbacks only; pivot kernels, random
//! number use, convergence and normalization are those of
//! [`crossinterpolate2`](super::crossinterpolate2).
//!
//! The caller's callback is evaluated once per physical point `x` and returns
//! the whole matrix. A fit-owned [`MultiIndexCache`] keyed by `x` serves every
//! later request for an entry of the same point, so a point is not sampled
//! again for another matrix entry while it stays retained.
//!
//! # When to use
//!
//! The joint fit targets genuine matrix-valued functions whose entries share
//! structure, such as `F(x) = sum_k a_k(x) U_k V_k^T` with a few modes `k`; the
//! shared middle cores then carry the `a_k(x)` once for all entries. For
//! unrelated entries (independent scalar functions broadcast over `(i, j)`),
//! separate scalar [`crossinterpolate2`](super::crossinterpolate2) fits, ideally
//! sharing one whole-matrix cache, can be cheaper and smaller. Rule of thumb:
//! the joint fit wins when its bond dimension grows more slowly than
//! `sqrt(rows * cols)` times that of a single entry.
//!
//! # Cache memory
//!
//! The cache uses the default retained-byte limit
//! ([`tensor4all_core::DEFAULT_RETAINED_BYTE_LIMIT`], 256 MiB), and that limit
//! bounds the hash table **and** the matrix payload: every cached point is
//! accounted as `rows * cols * size_of::<T>()` bytes of matrix data plus the
//! reference-count header of its shared allocation (two words, padded when
//! `align_of::<T>()` exceeds the word size), see [`MatrixSamplingStats::payload_bytes`]. When
//! the limit is reached the oldest generation of points is evicted, and a point
//! requested again is passed to the callback again. At-most-once evaluation
//! per point therefore holds only while the point stays retained; a fit whose
//! distinct points exceed the limit re-evaluates evicted ones. The limit is not
//! caller-tunable yet, see [`crossinterpolate2_matrix`].

use super::TciLinalg;
use super::{crossinterpolate2_with_rng_linalg, seeded_rng, TCI2OptimizationResult, TCI2Options};
use crate::error::{Result, TCIError};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::Arc;
use tensor4all_core::{MatrixLuciScalar, MultiIndex, MultiIndexCache, Scalar};
use tensor4all_simplett::{SimpleTensorTrain, TTScalar, Tensor3Ops};
use tensor4all_tensorbackend::Matrix;

/// Shape of the matrix returned by the callback at every point.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::MatrixShape;
///
/// let shape = MatrixShape { rows: 2, cols: 3 };
/// assert_eq!(shape.rows * shape.cols, 6);
/// assert_ne!(shape, MatrixShape { rows: 3, cols: 2 });
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    /// Number of matrix rows; the dimension of the leading boundary site.
    pub rows: usize,
    /// Number of matrix columns; the dimension of the trailing boundary site.
    pub cols: usize,
}

/// Source-work counters of one matrix-valued fit.
///
/// All counters are taken at the end of the fit. They count the work the
/// solver asked for, not wall-clock time.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::MatrixSamplingStats;
///
/// let stats = MatrixSamplingStats::default();
/// assert_eq!(stats.source_points, 0);
/// assert_eq!(stats.payload_bytes, 0);
/// assert_eq!(stats.retained_bytes, stats.table_bytes + stats.payload_bytes);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixSamplingStats {
    /// Points passed to the caller's callback (one whole matrix each).
    pub source_points: usize,
    /// Scalar entries `(i, x, j)` the solver requested.
    pub entry_requests: usize,
    /// Entry requests answered without a new callback evaluation, either from
    /// the cache or from another request for the same point in the same batch.
    pub cache_hits: usize,
    /// Total bytes retained by the cache at the end of the fit
    /// ([`MultiIndexCache::retained_bytes`]): hash tables plus matrix payloads,
    /// equal to `table_bytes + payload_bytes`. The cache runs with the default
    /// retained-byte limit ([`tensor4all_core::DEFAULT_RETAINED_BYTE_LIMIT`]),
    /// which bounds this value; oldest points are evicted, and so possibly
    /// evaluated again, when it is reached.
    pub retained_bytes: usize,
    /// Hash-table part of [`Self::retained_bytes`] (buckets, keys, the handles
    /// to the payloads and control bytes).
    pub table_bytes: usize,
    /// Payload part of [`Self::retained_bytes`]: `cached points * (rows * cols *
    /// size_of::<T>() + 2 * size_of::<usize>())`, the matrix data of each
    /// retained point plus the reference-count header of its shared allocation.
    /// Allocator overhead is not counted.
    pub payload_bytes: usize,
}

/// Result of [`crossinterpolate2_matrix`].
///
/// Wraps the unchanged scalar result on the internal chain
/// `[rows, d_1, .., d_L, cols]` together with the matrix shape and the
/// source-work counters. The scalar tensor train keeps its invariants (unit
/// boundary bonds, `L + 2` cores); [`MatrixTCI2Result::evaluate`] reads out a
/// whole matrix at once.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::AbstractTensorTrain;
/// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options, TCI2Termination};
///
/// // F(x)[i, j] = (1 + i + 2 j) * (1 + x0 + x1), a non-symmetric 2 x 3 matrix.
/// let source = |points: &[Vec<usize>], out: &mut [f64]| {
///     for (p, x) in points.iter().enumerate() {
///         for j in 0..3 {
///             for i in 0..2 {
///                 out[p * 6 + i + 2 * j] = ((1 + i + 2 * j) * (1 + x[0] + x[1])) as f64;
///             }
///         }
///     }
/// };
/// let fit = crossinterpolate2_matrix::<f64, _>(
///     source,
///     MatrixShape { rows: 2, cols: 3 },
///     vec![3, 3],
///     vec![vec![1, 1, 1, 1]],
///     TCI2Options { tolerance: 1e-12, seed: Some(7), ..TCI2Options::default() },
/// )
/// .unwrap();
/// assert_eq!(fit.result.termination, TCI2Termination::Converged);
/// assert_eq!(fit.shape(), MatrixShape { rows: 2, cols: 3 });
///
/// let m = fit.evaluate(&[2, 1]).unwrap();
/// assert_eq!((m.nrows(), m.ncols()), (2, 3));
/// for j in 0..3 {
///     for i in 0..2 {
///         assert!((m[[i, j]] - ((1 + i + 2 * j) * 4) as f64).abs() < 1e-9);
///     }
/// }
/// assert_eq!(fit.to_tensor_train().unwrap().len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct MatrixTCI2Result<T: Scalar + TTScalar> {
    /// The scalar result on the internal chain `[rows, d_1, .., d_L, cols]`.
    pub result: TCI2OptimizationResult<T>,
    shape: MatrixShape,
    stats: MatrixSamplingStats,
}

impl<T: Scalar + TTScalar + Default + MatrixLuciScalar> MatrixTCI2Result<T> {
    /// Matrix shape of the fitted function.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options};
    ///
    /// let fit = crossinterpolate2_matrix::<f64, _>(
    ///     |points: &[Vec<usize>], out: &mut [f64]| {
    ///         for (p, x) in points.iter().enumerate() {
    ///             out[p * 2] = 1.0 + x[0] as f64;
    ///             out[p * 2 + 1] = 2.0;
    ///         }
    ///     },
    ///     MatrixShape { rows: 2, cols: 1 },
    ///     vec![3],
    ///     vec![],
    ///     TCI2Options { seed: Some(1), ..TCI2Options::default() },
    /// )
    /// .unwrap();
    /// assert_eq!(fit.shape(), MatrixShape { rows: 2, cols: 1 });
    /// ```
    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// Source-work counters of the fit.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options};
    ///
    /// let fit = crossinterpolate2_matrix::<f64, _>(
    ///     |points: &[Vec<usize>], out: &mut [f64]| {
    ///         for (p, x) in points.iter().enumerate() {
    ///             out[p * 2] = 1.0 + x[0] as f64;
    ///             out[p * 2 + 1] = 2.0;
    ///         }
    ///     },
    ///     MatrixShape { rows: 2, cols: 1 },
    ///     vec![3],
    ///     vec![],
    ///     TCI2Options { seed: Some(1), ..TCI2Options::default() },
    /// )
    /// .unwrap();
    /// let stats = fit.sampling_stats();
    /// assert!(stats.source_points >= 1);
    /// assert!(stats.source_points <= 3); // only three distinct points exist
    /// let per_point = 2 * std::mem::size_of::<f64>() + 2 * std::mem::size_of::<usize>();
    /// assert_eq!(stats.payload_bytes % per_point, 0);
    /// assert_eq!(stats.retained_bytes, stats.table_bytes + stats.payload_bytes);
    /// ```
    pub fn sampling_stats(&self) -> &MatrixSamplingStats {
        &self.stats
    }

    /// Converts the fit to the scalar tensor train on `[rows, d_1, .., d_L, cols]`.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError::SimpleTensorTrain`] when the site tensors do not
    /// form a valid tensor train.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::AbstractTensorTrain;
    /// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options};
    ///
    /// let fit = crossinterpolate2_matrix::<f64, _>(
    ///     |points: &[Vec<usize>], out: &mut [f64]| {
    ///         for (p, x) in points.iter().enumerate() {
    ///             out[p * 2] = 1.0 + x[0] as f64;
    ///             out[p * 2 + 1] = 2.0;
    ///         }
    ///     },
    ///     MatrixShape { rows: 2, cols: 1 },
    ///     vec![3],
    ///     vec![],
    ///     TCI2Options { seed: Some(1), ..TCI2Options::default() },
    /// )
    /// .unwrap();
    /// let tt = fit.to_tensor_train().unwrap();
    /// assert_eq!(tt.len(), 3);
    /// // Entry (i = 1, x = 2, j = 0) of the matrix function is 2.
    /// assert!((tt.evaluate(&[1, 2, 0]).unwrap() - 2.0).abs() < 1e-9);
    /// ```
    pub fn to_tensor_train(&self) -> Result<SimpleTensorTrain<T>> {
        self.result.tci.to_tensor_train()
    }

    /// Evaluates the interpolated `rows x cols` matrix at the physical point `x`.
    ///
    /// One whole-matrix pass: the row core slab (`rows x r_1`), the middle
    /// cores sliced at `x`, and the column core slab (`r_L x cols`) are
    /// multiplied once. The result is column-major.
    ///
    /// The products are plain column-major loops, deliberately not the session's
    /// `mat_mul`: readout is not a hot path yet, and routing it through the
    /// session would make its cost depend on a backend that this step does not
    /// otherwise touch. This is a choice, not an oversight; when readout becomes
    /// hot it should move to the session.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError::DimensionMismatch`] when `x.len()` differs from the
    /// number of physical sites `L`, [`TCIError::IndexOutOfBounds`] when a
    /// coordinate is outside its local dimension, and
    /// [`TCIError::InvalidOperation`] when the site tensors are unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options, TCIError};
    ///
    /// let fit = crossinterpolate2_matrix::<f64, _>(
    ///     |points: &[Vec<usize>], out: &mut [f64]| {
    ///         for (p, x) in points.iter().enumerate() {
    ///             out[p * 2] = 1.0 + x[0] as f64;
    ///             out[p * 2 + 1] = 2.0;
    ///         }
    ///     },
    ///     MatrixShape { rows: 2, cols: 1 },
    ///     vec![3],
    ///     vec![],
    ///     TCI2Options { seed: Some(1), ..TCI2Options::default() },
    /// )
    /// .unwrap();
    /// let m = fit.evaluate(&[2]).unwrap();
    /// assert!((m[[0, 0]] - 3.0).abs() < 1e-9);
    /// assert!((m[[1, 0]] - 2.0).abs() < 1e-9);
    /// assert!(matches!(fit.evaluate(&[3]), Err(TCIError::IndexOutOfBounds { .. })));
    /// assert!(matches!(fit.evaluate(&[0, 0]), Err(TCIError::DimensionMismatch { .. })));
    /// ```
    pub fn evaluate(&self, x: &[usize]) -> Result<Matrix<T>> {
        let tci = &self.result.tci;
        let local_dims = tci.local_dims();
        let n_physical = local_dims.len().saturating_sub(2);
        if x.len() != n_physical {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "expected {n_physical} physical coordinates, got {}",
                    x.len()
                ),
            });
        }
        for (k, (&xk, &dk)) in x.iter().zip(&local_dims[1..=n_physical]).enumerate() {
            if xk >= dk {
                return Err(TCIError::IndexOutOfBounds {
                    message: format!("coordinate {k} is {xk}, local dimension is {dk}"),
                });
            }
        }
        if !tci.is_site_tensors_available() {
            return Err(TCIError::InvalidOperation {
                message: "site tensors are not available".to_string(),
            });
        }
        let MatrixShape { rows, cols } = self.shape;

        let first = tci.site_tensor(0);
        let mut width = first.right_dim();
        let mut current = vec![T::zero(); rows * width];
        for a in 0..width {
            for i in 0..rows {
                current[i + rows * a] = *first.get3(0, i, a);
            }
        }
        for (k, &xk) in x.iter().enumerate() {
            let core = tci.site_tensor(k + 1);
            let slab = core.slice_site(xk);
            current = col_major_gemm(&current, rows, width, &slab, core.right_dim());
            width = core.right_dim();
        }
        let last = tci.site_tensor(n_physical + 1);
        let mut tail = vec![T::zero(); width * cols];
        for j in 0..cols {
            for a in 0..width {
                tail[a + width * j] = *last.get3(a, j, 0);
            }
        }
        let out = col_major_gemm(&current, rows, width, &tail, cols);
        Ok(Matrix::from_col_major_vec(rows, cols, out))
    }
}

/// Column-major `(m x k) * (k x n)` product in plain loops.
fn col_major_gemm<T: Scalar>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::zero(); m * n];
    for j in 0..n {
        for l in 0..k {
            let blj = b[l + k * j];
            for i in 0..m {
                out[i + m * j] = out[i + m * j] + a[i + m * l] * blj;
            }
        }
    }
    out
}

/// Caller callback plus the fit-owned cache and counters.
struct SourceState<T, F>
where
    T: Scalar,
{
    source: F,
    cache: MultiIndexCache<Arc<[T]>>,
    rows: usize,
    block_len: usize,
    max_points_per_call: usize,
    stats: MatrixSamplingStats,
    scratch: Scratch<T>,
}

/// Buffers reused by every [`SourceState::entries`] call, so a request costs no
/// fresh vectors beyond the cached payloads themselves. They are cleared at the
/// start of each call and hold no state between calls.
struct Scratch<T> {
    /// Slot (distinct point) of each request.
    request_slots: Vec<usize>,
    /// One entry per distinct point of the call.
    slots: Vec<PointSlot<T>>,
    /// Pooled coordinate vectors of the missing points; only the first
    /// `n_misses` of the current call are meaningful.
    miss_points: Vec<MultiIndex>,
    /// Payloads of the missing points, in first-occurrence order.
    miss_blocks: Vec<Arc<[T]>>,
    /// Output buffer handed to the callback.
    out: Vec<T>,
}

impl<T> Scratch<T> {
    fn new() -> Self {
        Self {
            request_slots: Vec::new(),
            slots: Vec::new(),
            miss_points: Vec::new(),
            miss_blocks: Vec::new(),
            out: Vec::new(),
        }
    }

    /// Records `x` as the next missing point, reusing a pooled vector.
    fn push_miss(&mut self, n_misses: usize, x: &[usize]) {
        match self.miss_points.get_mut(n_misses) {
            Some(point) => {
                point.clear();
                point.extend_from_slice(x);
            }
            None => self.miss_points.push(x.to_vec()),
        }
    }
}

/// How a distinct point of one request batch is obtained.
enum PointSlot<T> {
    Ready(Arc<[T]>),
    Missing(usize),
}

impl<T, F> SourceState<T, F>
where
    T: Scalar,
    F: Fn(&[MultiIndex], &mut [T]),
{
    fn new(source: F, shape: MatrixShape, physical_dims: &[usize]) -> Result<Self> {
        let block_len = checked_block_len::<T>(shape)?;
        let cache = MultiIndexCache::new(physical_dims)
            .map_err(|error| TCIError::InvalidConfiguration {
                message: format!("cannot key the matrix cache: {error}"),
            })?
            .with_value_payload_bytes(point_payload_bytes::<T>(block_len));
        let max_points_per_call = (isize::MAX as usize / std::mem::size_of::<T>().max(1))
            .checked_div(block_len)
            .unwrap_or(1)
            .max(1);
        Ok(Self {
            source,
            cache,
            rows: shape.rows,
            block_len,
            max_points_per_call,
            stats: MatrixSamplingStats::default(),
            scratch: Scratch::new(),
        })
    }

    /// Answers scalar requests `(x, i, j)` in request order.
    ///
    /// Distinct points are resolved once per call: cached points are read from
    /// the cache, the remaining ones are passed to the callback in
    /// first-occurrence order in consecutive chunks of at most
    /// `max_points_per_call` points (a deterministic split), and the results are
    /// inserted into the cache.
    fn entries(&mut self, requests: &[(&[usize], usize, usize)]) -> Vec<T> {
        let scratch = &mut self.scratch;
        scratch.request_slots.clear();
        scratch.slots.clear();
        scratch.miss_blocks.clear();
        let mut n_misses = 0;
        // The solver only requests in-range points, so a cache key error cannot
        // occur; it would be treated as a miss.
        let mut resolve =
            |scratch: &mut Scratch<T>, cache: &mut MultiIndexCache<Arc<[T]>>, x: &[usize]| {
                let slot = scratch.slots.len();
                match cache.get(x).ok().flatten() {
                    Some(block) => scratch.slots.push(PointSlot::Ready(block)),
                    None => {
                        scratch.push_miss(n_misses, x);
                        scratch.slots.push(PointSlot::Missing(n_misses));
                        n_misses += 1;
                    }
                }
                slot
            };
        if let [(x, _, _)] = requests {
            // A single request has nothing to deduplicate.
            let slot = resolve(scratch, &mut self.cache, x);
            scratch.request_slots.push(slot);
        } else {
            let mut slot_of: FxHashMap<&[usize], usize> =
                FxHashMap::with_capacity_and_hasher(requests.len(), Default::default());
            for &(x, _, _) in requests {
                let slot = *slot_of
                    .entry(x)
                    .or_insert_with(|| resolve(scratch, &mut self.cache, x));
                scratch.request_slots.push(slot);
            }
        }

        let mut start = 0;
        while start < n_misses {
            let end = start.saturating_add(self.max_points_per_call).min(n_misses);
            let chunk = &scratch.miss_points[start..end];
            scratch.out.clear();
            scratch
                .out
                .resize(chunk.len() * self.block_len, T::default());
            (self.source)(chunk, &mut scratch.out);
            self.stats.source_points += chunk.len();
            for (point, block) in chunk.iter().zip(scratch.out.chunks_exact(self.block_len)) {
                let block: Arc<[T]> = block.into();
                // An insert fails only for an invalid key, which cannot
                // happen for a point the callback was just asked about.
                let _ = self.cache.insert(point, block.clone());
                scratch.miss_blocks.push(block);
            }
            start = end;
        }

        let mut values = Vec::with_capacity(requests.len());
        for (&(_, i, j), &slot) in requests.iter().zip(&scratch.request_slots) {
            let block = match &scratch.slots[slot] {
                PointSlot::Ready(block) => block,
                PointSlot::Missing(m) => &scratch.miss_blocks[*m],
            };
            values.push(block[i + self.rows * j]);
        }
        self.stats.entry_requests += requests.len();
        self.stats.cache_hits += requests.len() - n_misses;
        values
    }

    fn finished_stats(&self) -> MatrixSamplingStats {
        let retained_bytes = self.cache.retained_bytes();
        let payload_bytes = self
            .cache
            .len()
            .saturating_mul(self.cache.value_payload_bytes());
        MatrixSamplingStats {
            retained_bytes,
            table_bytes: retained_bytes.saturating_sub(payload_bytes),
            payload_bytes,
            ..self.stats
        }
    }
}

/// Bytes one cached point adds to the retained-byte accounting: the size of the
/// `Arc<[T]>` allocation holding its `block_len` matrix entries, computed from
/// the `repr(C)` layout of `ArcInner` (strong and weak counts, then the slice,
/// padded to the alignment of `T` and to the size of the whole allocation), so
/// it stays exact for over-aligned `T`. Allocator overhead is excluded.
/// Saturating, so a block whose byte count overflows is simply never retained.
fn point_payload_bytes<T>(block_len: usize) -> usize {
    use std::alloc::Layout;
    Layout::array::<T>(block_len)
        .ok()
        .and_then(|data| Layout::new::<[usize; 2]>().extend(data).ok())
        .map(|(layout, _)| layout.pad_to_align().size())
        .unwrap_or(usize::MAX)
}

/// `rows * cols`, rejecting zero and sizes whose byte count cannot be allocated.
fn checked_block_len<T>(shape: MatrixShape) -> Result<usize> {
    if shape.rows == 0 || shape.cols == 0 {
        return Err(TCIError::InvalidConfiguration {
            message: format!(
                "matrix shape {} x {} must have positive rows and cols",
                shape.rows, shape.cols
            ),
        });
    }
    shape
        .rows
        .checked_mul(shape.cols)
        .filter(|len| {
            len.checked_mul(std::mem::size_of::<T>().max(1))
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or_else(|| TCIError::DimensionMismatch {
            message: format!(
                "matrix shape {} x {} overflows the addressable size",
                shape.rows, shape.cols
            ),
        })
}

fn validate_matrix_problem(
    shape: MatrixShape,
    local_dims: &[usize],
    initial_pivots: &[MultiIndex],
) -> Result<()> {
    if local_dims.is_empty() {
        return Err(TCIError::DimensionMismatch {
            message: "local_dims must contain at least one physical site".to_string(),
        });
    }
    if let Some(site) = local_dims.iter().position(|&d| d == 0) {
        return Err(TCIError::InvalidConfiguration {
            message: format!("local dimension {site} is zero"),
        });
    }
    let internal_dims: Vec<usize> = std::iter::once(shape.rows)
        .chain(local_dims.iter().copied())
        .chain(std::iter::once(shape.cols))
        .collect();
    for (n, pivot) in initial_pivots.iter().enumerate() {
        if pivot.len() != internal_dims.len() {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "initial pivot {n} has length {}, expected {}",
                    pivot.len(),
                    internal_dims.len()
                ),
            });
        }
        if let Some(site) = pivot.iter().zip(&internal_dims).position(|(p, d)| p >= d) {
            return Err(TCIError::IndexOutOfBounds {
                message: format!("initial pivot {n} is out of range at site {site}"),
            });
        }
    }
    Ok(())
}

/// Interpolates a matrix-valued function of `L` physical sites with TCI2.
///
/// `source(points, out)` receives a batch of distinct physical points
/// (`points[p]` has length `L`) and must write the `rows x cols` matrix of
/// point `p` into `out[p * rows * cols ..][.. rows * cols]` in column-major
/// order: entry `(i, j)` of point `p` is `out[p * rows * cols + i + rows * j]`.
/// The library allocates `out` with length `points.len() * rows * cols` and
/// fills it with `T::default()`; the callback must overwrite every entry, since
/// unwritten entries cannot be detected. The callback is infallible, like the
/// scalar entry points, and is never asked for a point that is still cached.
///
/// The points that one solver request needs and that are not cached are passed
/// to the callback deterministically: in first-occurrence order, in consecutive
/// chunks of at most `isize::MAX / (size_of::<T>() * rows * cols)` points.
/// **The callback may therefore be called more than once for one logical
/// batch** (once per chunk); a request costs a single call unless that bound is
/// exceeded, and the split never changes the values or the fit. A callback must
/// not assume that one call sees all the points of a solver step.
///
/// The fit-owned cache uses the default retained-byte limit
/// ([`tensor4all_core::DEFAULT_RETAINED_BYTE_LIMIT`], 256 MiB), which bounds
/// the hash tables **and** the matrix payloads: each cached point is accounted
/// as `rows * cols * size_of::<T>() + 2 * size_of::<usize>()` bytes of payload
/// (the matrix data and the reference-count header of its shared allocation),
/// see [`MatrixSamplingStats::retained_bytes`]. When the limit is reached the
/// oldest generation of points is evicted and a point asked for again is
/// evaluated again: "not asked for a point that is still cached" is the
/// guarantee, and at-most-once evaluation holds only while a point stays
/// retained. A matrix too large for the limit is never retained.
///
/// The limit is not caller-tunable yet: [`TCI2Options`] is shared with the
/// scalar solver and has no cache field, and an extra option or entry point is
/// a public-API decision that is left open. Memory that matters is therefore
/// bounded by the 256 MiB default, and only heap payload that the accounting
/// above declares is counted (the matrix data of each point, not allocator
/// overhead). A fit that needs more distinct points than fit re-evaluates
/// evicted ones, which costs callback calls, never correctness.
///
/// Internally the problem is the chain `[rows, d_1, .., d_L, cols]` with
/// `f(i, x, j) = F(x)[i, j]`, solved by the unchanged scalar TCI2 machinery.
/// Only entries the solver requests enter the amplitude scale, so `tolerance`
/// and `normalize_error` mean what they mean for [`crossinterpolate2`](super::crossinterpolate2)
/// on that endpoint problem, and the fit reproduces its pivots, ranks and
/// errors exactly.
///
/// # When to use
///
/// Use this for a genuine matrix-valued function whose entries share structure,
/// for example `F(x) = sum_k a_k(x) U_k V_k^T` (a Green's function driven by a
/// few modes), where one set of shared middle cores carries the `a_k(x)` for
/// all entries. It is not a way to compress `rows * cols` unrelated scalar
/// functions merely broadcast over a grid of `(i, j)`: nothing is shared then,
/// and the two boundary sites must carry `rows * cols` independent pieces.
///
/// Rule of thumb: the joint fit stores and reads out less than separate fits
/// when its bond dimension grows more slowly than `sqrt(rows * cols)` times the
/// bond dimension of a single entry. For unrelated entries, run separate scalar
/// [`crossinterpolate2`](super::crossinterpolate2) fits on `x` (each
/// callback computing the whole matrix and keeping one entry, ideally behind one
/// shared whole-matrix cache); they can be cheaper and smaller.
///
/// # Arguments
///
/// * `source` - batched matrix callback described above.
/// * `shape` - matrix shape; both extents must be positive.
/// * `local_dims` - dimensions `d_1..d_L` of the physical sites only; `L >= 1`
///   and every `d_k >= 1`.
/// * `initial_pivots` - full indices `[i, x_1, .., x_L, j]` of length `L + 2`;
///   empty means the all-zero index.
/// * `options` - the unchanged scalar [`TCI2Options`].
///
/// # Errors
///
/// Returns [`TCIError::InvalidConfiguration`] for a zero extent in `shape` or
/// `local_dims`, invalid options, or physical dimensions the cache cannot key;
/// [`TCIError::DimensionMismatch`] for `L == 0`, a pivot of the wrong length,
/// or a matrix size that overflows; [`TCIError::IndexOutOfBounds`] for an
/// out-of-range pivot; and [`TCIError::InvalidPivot`] when all initial pivots
/// evaluate to zero.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::{crossinterpolate2_matrix, MatrixShape, TCI2Options, TCIError};
///
/// let source = |points: &[Vec<usize>], out: &mut [f64]| {
///     for (p, x) in points.iter().enumerate() {
///         // 2 x 3 matrix, entry (i, j) = (1 + i + 2 j) * (1 + x0)
///         for j in 0..3 {
///             for i in 0..2 {
///                 out[p * 6 + i + 2 * j] = ((1 + i + 2 * j) * (1 + x[0])) as f64;
///             }
///         }
///     }
/// };
/// let fit = crossinterpolate2_matrix::<f64, _>(
///     source,
///     MatrixShape { rows: 2, cols: 3 },
///     vec![4],
///     vec![],
///     TCI2Options { seed: Some(3), ..TCI2Options::default() },
/// )
/// .unwrap();
/// let m = fit.evaluate(&[3]).unwrap();
/// assert!((m[[1, 2]] - (6.0 * 4.0)).abs() < 1e-9);
///
/// let bad = crossinterpolate2_matrix::<f64, _>(
///     |_: &[Vec<usize>], _: &mut [f64]| {},
///     MatrixShape { rows: 0, cols: 3 },
///     vec![4],
///     vec![],
///     TCI2Options::default(),
/// );
/// assert!(matches!(bad, Err(TCIError::InvalidConfiguration { .. })));
/// ```
pub fn crossinterpolate2_matrix<T, F>(
    source: F,
    shape: MatrixShape,
    local_dims: Vec<usize>,
    initial_pivots: Vec<MultiIndex>,
    options: TCI2Options,
) -> Result<MatrixTCI2Result<T>>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar + Send + Sync + 'static,
    F: Fn(&[MultiIndex], &mut [T]),
{
    let mut rng = seeded_rng(options.seed);
    crossinterpolate2_matrix_with_rng(source, shape, local_dims, initial_pivots, options, &mut rng)
}

/// Matrix-valued TCI2 on a caller-owned random stream.
///
/// Same as [`crossinterpolate2_matrix`], but consumes `rng` directly for the
/// global pivot search, like
/// [`crossinterpolate2_with_rng`](super::crossinterpolate2_with_rng).
///
/// The callback contract, the deterministic batch split (the callback may be
/// called more than once for one logical batch) and the cache memory note of
/// [`crossinterpolate2_matrix`] apply unchanged: the cache uses the default
/// retained-byte limit ([`tensor4all_core::DEFAULT_RETAINED_BYTE_LIMIT`]), which
/// bounds hash tables and matrix payloads together
/// ([`MatrixSamplingStats::retained_bytes`]); evicted points are evaluated
/// again. As there, the limit is not configurable yet (no cache field in the
/// shared [`TCI2Options`], and no separate entry point).
///
/// # Errors
///
/// Returns [`TCIError::InvalidConfiguration`] for a zero extent in `shape` or
/// `local_dims`, invalid options, or physical dimensions the cache cannot key;
/// [`TCIError::DimensionMismatch`] for `L == 0`, a pivot of the wrong length,
/// or a matrix size that overflows; [`TCIError::IndexOutOfBounds`] for an
/// out-of-range pivot; and [`TCIError::InvalidPivot`] when all initial pivots
/// evaluate to zero.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_tensorci::{crossinterpolate2_matrix_with_rng, MatrixShape, TCI2Options};
///
/// let mut rng = ChaCha8Rng::seed_from_u64(5);
/// let fit = crossinterpolate2_matrix_with_rng::<f64, _, _>(
///     |points: &[Vec<usize>], out: &mut [f64]| {
///         for (p, x) in points.iter().enumerate() {
///             out[p * 2] = 1.0 + x[0] as f64;
///             out[p * 2 + 1] = 2.0;
///         }
///     },
///     MatrixShape { rows: 2, cols: 1 },
///     vec![3],
///     vec![],
///     TCI2Options::default(),
///     &mut rng,
/// )
/// .unwrap();
/// let m = fit.evaluate(&[1]).unwrap();
/// assert!((m[[0, 0]] - 2.0).abs() < 1e-9);
/// assert!((m[[1, 0]] - 2.0).abs() < 1e-9);
/// ```
pub fn crossinterpolate2_matrix_with_rng<T, F, R>(
    source: F,
    shape: MatrixShape,
    local_dims: Vec<usize>,
    initial_pivots: Vec<MultiIndex>,
    options: TCI2Options,
    rng: &mut R,
) -> Result<MatrixTCI2Result<T>>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar + Send + Sync + 'static,
    F: Fn(&[MultiIndex], &mut [T]),
    R: rand::Rng + ?Sized,
{
    validate_matrix_problem(shape, &local_dims, &initial_pivots)?;
    let n_physical = local_dims.len();
    let state = RefCell::new(SourceState::new(source, shape, &local_dims)?);

    let single = |index: &MultiIndex| -> T {
        let request = (&index[1..=n_physical], index[0], index[n_physical + 1]);
        state.borrow_mut().entries(&[request])[0]
    };
    let batched = |indices: &[MultiIndex]| -> Vec<T> {
        let requests: Vec<_> = indices
            .iter()
            .map(|index| (&index[1..=n_physical], index[0], index[n_physical + 1]))
            .collect();
        state.borrow_mut().entries(&requests)
    };

    let internal_dims: Vec<usize> = std::iter::once(shape.rows)
        .chain(local_dims)
        .chain(std::iter::once(shape.cols))
        .collect();
    let result = crossinterpolate2_with_rng_linalg(
        single,
        Some(batched),
        internal_dims,
        initial_pivots,
        options,
        rng,
        &mut TciLinalg::Global,
    )?;
    let stats = state.borrow().finished_stats();
    Ok(MatrixTCI2Result {
        result,
        shape,
        stats,
    })
}

#[cfg(test)]
mod tests;
