//! QuanticsTensorCI2 and interpolation functions.
//!
//! Every interpolation entry point hands the target function a
//! [`QuanticsBatch`](crate::QuanticsBatch): one call per batch of grid points,
//! in column-major `(n_dims, n_points)` order. The deprecated point-wise entry
//! points are thin wrappers over the batched ones.

use rand::Rng as _;
use rand::SeedableRng;
use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{anyhow, Result};
use quanticsgrids::{DiscretizedGrid, InherentDiscreteGrid};
use tensor4all_core::MultiIndexCache;
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetci::materialize::to_treetn;
use tensor4all_treetci::{
    optimize_with_proposer, DefaultProposer, GlobalIndexBatch, TreeTCI2, TreeTciGraph,
};
use tensor4all_treetn::treetn_to_tensor_train as bridge_treetn_to_tensor_train;

use crate::batch::{pointwise_coordinate_batch, pointwise_index_batch, QuanticsBatch};
use crate::error::{QuanticsTCIError, Result as QtciResult};
use crate::options::QtciOptions;

/// Build the memo cache of one run from the grid's local dimensions.
///
/// A quantics index space wider than the widest built-in cache key (1024 bits)
/// cannot be memoized; such a run is rejected instead of silently losing cache
/// coverage.
fn new_memo_cache<V>(local_dims: &[usize]) -> QtciResult<MultiIndexCache<V>>
where
    V: Clone + Send + Sync + 'static,
{
    MultiIndexCache::new(local_dims).map_err(|error| QuanticsTCIError::InvalidConfiguration {
        message: format!("quantics index space cannot be memoized: {error}"),
    })
}

/// Read the evaluation accounting out of a run's memo cache.
fn memo_cache_stats<V>(cache: &MemoCache<V>) -> CacheStats
where
    V: Clone + Send + Sync + 'static,
{
    let cache = cache.borrow();
    CacheStats {
        num_evals: cache.len(),
        num_cache_hits: cache.hits(),
        num_cache_misses: cache.misses(),
        dropped_inserts: cache.dropped_inserts(),
    }
}

/// The stream a seeded quantics run uses.
///
/// Entropy is drawn only when the run actually samples random initial pivots;
/// with `n_random_init_pivot == 0` the run is deterministic and a fixed seed
/// keeps it so without touching the OS.
pub(crate) fn seeded_quantics_stream(options: &QtciOptions) -> rand_chacha::ChaCha8Rng {
    match (options.n_random_init_pivot > 0, options.rng_seed) {
        (true, None) => rand_chacha::ChaCha8Rng::from_os_rng(),
        (_, seed) => rand_chacha::ChaCha8Rng::seed_from_u64(seed.unwrap_or(0)),
    }
}

/// Convert the caller's initial pivots to quantics indices and append random
/// starting pivots.
fn prepare_pivots(
    initial_pivots: Option<Vec<Vec<usize>>>,
    local_dims: &[usize],
    convert: impl Fn(&[usize]) -> Result<Vec<usize>>,
    n_random: usize,
    rng: &mut dyn rand::RngCore,
) -> Result<Vec<Vec<usize>>> {
    let mut pivots = match initial_pivots {
        Some(pivots) if !pivots.is_empty() => pivots
            .iter()
            .map(|pivot| convert(pivot))
            .collect::<Result<Vec<_>>>()?,
        _ => vec![vec![0; local_dims.len()]],
    };
    for _ in 0..n_random {
        pivots.push(
            local_dims
                .iter()
                .map(|&dim| rng.random_range(0..dim))
                .collect(),
        );
    }
    Ok(pivots)
}

/// The memoization cache a quantics run shares with its target evaluator.
///
/// The cache lives for one run and is dropped with it. It encodes quantics
/// multi-indices as mixed-radix flat integers, so a lookup never hashes an owned
/// key vector, and it stores no callback: [`site_evaluator`] looks points up,
/// evaluates only the misses, and inserts the successful results.
type MemoCache<V> = Rc<RefCell<MultiIndexCache<V>>>;

/// Wrap a per-point conversion and a batched target function into the
/// quantics-site batched evaluator TreeTCI expects.
///
/// A point already in `cache` is served from the cache without converting it or
/// calling the target. The remaining points are converted, evaluated in one
/// batched call, and inserted; a failed evaluation returns before anything is
/// inserted, so a failure never becomes a cached value. Repeated points inside
/// one batch are converted and evaluated once.
fn site_evaluator<T, V, C, F>(
    convert: C,
    evaluate: F,
    cache: MemoCache<V>,
) -> impl Fn(GlobalIndexBatch<'_>) -> Result<Vec<V>>
where
    T: Copy,
    V: Clone + Send + Sync + 'static,
    C: Fn(&[usize]) -> Result<Vec<T>>,
    F: Fn(QuanticsBatch<'_, T>) -> Result<Vec<V>>,
{
    move |batch: GlobalIndexBatch<'_>| {
        let n_points = batch.n_points();
        let n_sites = batch.n_sites();
        let mut cache = cache.borrow_mut();
        let mut results: Vec<Option<V>> = Vec::with_capacity(n_points);
        let mut miss_positions: Vec<usize> = Vec::new();
        let mut miss_slots: Vec<usize> = Vec::new();
        let mut miss_points: Vec<&[usize]> = Vec::new();
        let mut miss_slot_of: std::collections::HashMap<&[usize], usize> =
            std::collections::HashMap::new();

        // Read the whole batch into one flat buffer first: a per-point vector
        // would allocate once per requested point, which dominates the lookup
        // cost for cheap targets.
        let mut flat_indices: Vec<usize> = Vec::with_capacity(n_points * n_sites);
        for point in 0..n_points {
            for site in 0..n_sites {
                flat_indices.push(batch.get(site, point).ok_or_else(|| {
                    anyhow!(
                        "invalid batch index: site {site}, point {point}, batch shape {n_sites}x{n_points}"
                    )
                })?);
            }
        }

        for point in 0..n_points {
            let quantics = &flat_indices[point * n_sites..(point + 1) * n_sites];
            match cache.get(quantics)? {
                Some(value) => results.push(Some(value)),
                None => {
                    results.push(None);
                    miss_positions.push(point);
                    let slot = match miss_slot_of.get(quantics) {
                        Some(&slot) => slot,
                        None => {
                            let slot = miss_points.len();
                            miss_points.push(quantics);
                            miss_slot_of.insert(quantics, slot);
                            slot
                        }
                    };
                    miss_slots.push(slot);
                }
            }
        }

        if !miss_points.is_empty() {
            let mut flat: Vec<T> = Vec::new();
            let mut n_dims = 0usize;
            for quantics in &miss_points {
                let values = convert(quantics)?;
                if n_dims == 0 {
                    n_dims = values.len();
                } else if values.len() != n_dims {
                    return Err(anyhow!(
                        "inconsistent point dimension: expected {n_dims}, got {}",
                        values.len()
                    ));
                }
                flat.extend_from_slice(&values);
            }
            let values = evaluate(QuanticsBatch::new(&flat, n_dims, miss_points.len())?)?;
            if values.len() != miss_points.len() {
                return Err(anyhow!(
                    "target function returned {} values for {} evaluated points",
                    values.len(),
                    miss_points.len()
                ));
            }
            for (quantics, value) in miss_points.iter().zip(values.iter()) {
                cache.insert(quantics, value.clone())?;
            }
            drop(miss_slot_of);
            for (position, slot) in miss_positions.iter().zip(miss_slots.iter()) {
                results[*position] = Some(values[*slot].clone());
            }
        }

        results
            .into_iter()
            .collect::<Option<Vec<V>>>()
            .ok_or_else(|| anyhow!("internal error: a requested point has no value"))
    }
}

/// Run TreeTCI on a quantics-site batched evaluator and materialize the result.
#[allow(clippy::type_complexity)]
fn run_treetci_batch<V, F>(
    local_dims: Vec<usize>,
    evaluate: F,
    cache: MemoCache<V>,
    pivots: Vec<Vec<usize>>,
    options: &QtciOptions,
) -> QtciResult<(TreeTCI2<V>, SimpleTensorTrain<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<V>>,
{
    let n_sites = local_dims.len();
    let graph = TreeTciGraph::linear_chain(n_sites)?;
    let mut tci = TreeTCI2::<V>::new(local_dims, graph)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value from the initial pivots.
    let flat: Vec<usize> = pivots
        .iter()
        .flat_map(|pivot| pivot.iter().copied())
        .collect();
    let init_batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = evaluate(init_batch)?;
    tci.max_sample_value = init_vals
        .iter()
        .map(|value| <V as tensor4all_core::Scalar>::abs_val(*value))
        .fold(0.0f64, f64::max);
    if tci.max_sample_value <= 0.0 {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "initial pivots must not all evaluate to zero".to_string(),
        });
    }

    let tree_opts = options.to_treetci_options();
    let (ranks, errors) =
        optimize_with_proposer(&mut tci, &evaluate, &tree_opts, &DefaultProposer)?;
    let treetn = to_treetn(&tci, &evaluate, Some(0))?;

    // Convert TreeTN → SimpleTensorTrain<V> via the sanctioned bridge
    let tt: SimpleTensorTrain<V> =
        bridge_treetn_to_tensor_train(treetn).map_err(|error| QuanticsTCIError::Operation {
            source: anyhow::Error::new(error)
                .context("TreeTN to SimpleTensorTrain conversion failed"),
        })?;

    // Drop the evaluator so it releases its cache handle; the caller reads the
    // counters from its own handle.
    drop(evaluate);
    drop(cache);

    Ok((tci, tt, ranks, errors))
}

/// TCI result wrapped with grid information.
///
/// Combines a [`SimpleTensorTrain`] approximation with grid metadata so you
/// can [`evaluate`](Self::evaluate) at grid indices, compute
/// [`sum`](Self::sum) and [`integral`](Self::integral), and access the
/// underlying [`tensor_train`](Self::tensor_train) for further
/// manipulation.
///
/// Created by [`quanticscrossinterpolate`], [`quanticscrossinterpolate_discrete`],
/// or [`quanticscrossinterpolate_from_arrays`].
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     QtciOptions,
///     pointwise_coordinate_batch,
///     pointwise_index_batch,
///     quanticscrossinterpolate_discrete_batch,
/// };
///
/// // Interpolate f(i) = i on a grid of size 8 (0-indexed)
/// let f = |idx: &[usize]| idx[0] as f64;
/// let (qtci, _ranks, _errors) =
///     quanticscrossinterpolate_discrete_batch::<f64, _>(
///         &[8], pointwise_index_batch(f), None, QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
///     ).unwrap();
///
/// // Evaluate at grid point 4
/// let val = qtci.evaluate(&[4]).unwrap();
/// assert!((val - 4.0).abs() < 1e-8);
///
/// // Sum over all grid points: 0 + 1 + ... + 7 = 28
/// let sum = qtci.sum().unwrap();
/// assert!((sum - 28.0).abs() < 1e-6);
///
/// // rank() gives the maximum bond dimension
/// assert!(qtci.rank() >= 1);
///
/// // link_dims() gives bond dimensions between sites
/// assert!(!qtci.link_dims().is_empty());
/// ```
#[derive(Clone)]
pub struct QuanticsTensorCI2<V: TTScalar> {
    /// Underlying tensor train
    tt: SimpleTensorTrain<V>,
    /// TreeTCI2 state (pivot sets, graph, etc.)
    tci_state: TreeTCI2<V>,
    /// Grid for coordinate conversion (DiscretizedGrid)
    discretized_grid: Option<DiscretizedGrid>,
    /// Grid for coordinate conversion (InherentDiscreteGrid)
    inherent_grid: Option<InherentDiscreteGrid>,
    /// Target-evaluation accounting of the run that produced this result
    cache_stats: CacheStats,
}

/// Target-evaluation accounting for one quantics interpolation run.
///
/// The run memoizes target evaluations on the quantics multi-index, so the
/// target is called only for points it has not seen. The counters separate the
/// quantities an optimization needs: how many requests were cached, how many
/// were not, how many distinct points the target evaluated, and how many
/// evaluations could not be retained because the memo cache reached its
/// logical payload limit.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::CacheStats;
///
/// let stats = CacheStats {
///     num_evals: 3,
///     num_cache_hits: 5,
///     num_cache_misses: 3,
///     dropped_inserts: 0,
/// };
/// assert!((stats.hit_ratio() - 5.0 / 8.0).abs() < 1e-12);
/// assert_eq!(CacheStats::default().hit_ratio(), 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Distinct points the target function evaluated.
    pub num_evals: usize,
    /// Point requests served from the cache.
    pub num_cache_hits: usize,
    /// Point requests that were not cached and had to be evaluated.
    pub num_cache_misses: usize,
    /// Successful evaluations that were not retained because the memo cache was
    /// at its logical payload limit. Such a point is evaluated again if it is
    /// requested later; no returned value changes.
    pub dropped_inserts: usize,
}

impl CacheStats {
    /// Fraction of point requests served from the cache, or `0.0` when the run
    /// requested no points.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::CacheStats;
    ///
    /// let stats = CacheStats {
    ///     num_evals: 1,
    ///     num_cache_hits: 1,
    ///     num_cache_misses: 1,
    ///     dropped_inserts: 0,
    /// };
    /// assert!((stats.hit_ratio() - 0.5).abs() < 1e-12);
    /// ```
    pub fn hit_ratio(&self) -> f64 {
        let requests = self.num_cache_hits + self.num_cache_misses;
        if requests == 0 {
            0.0
        } else {
            self.num_cache_hits as f64 / requests as f64
        }
    }
}

impl<V> QuanticsTensorCI2<V>
where
    V: TTScalar + Default + Clone,
{
    /// Create a new QuanticsTensorCI2 from a SimpleTensorTrain, TreeTCI2 state, and discretized grid.
    pub fn from_discretized(
        tt: SimpleTensorTrain<V>,
        tci_state: TreeTCI2<V>,
        grid: DiscretizedGrid,
        cache_stats: CacheStats,
    ) -> Self {
        Self {
            tt,
            tci_state,
            discretized_grid: Some(grid),
            inherent_grid: None,
            cache_stats,
        }
    }

    /// Create a new QuanticsTensorCI2 from a SimpleTensorTrain, TreeTCI2 state, and inherent discrete grid.
    pub fn from_inherent(
        tt: SimpleTensorTrain<V>,
        tci_state: TreeTCI2<V>,
        grid: InherentDiscreteGrid,
        cache_stats: CacheStats,
    ) -> Self {
        Self {
            tt,
            tci_state,
            discretized_grid: None,
            inherent_grid: Some(grid),
            cache_stats,
        }
    }

    /// Get the discretized grid (if available).
    pub fn discretized_grid(&self) -> Option<&DiscretizedGrid> {
        self.discretized_grid.as_ref()
    }

    /// Get the inherent discrete grid (if available).
    pub fn inherent_grid(&self) -> Option<&InherentDiscreteGrid> {
        self.inherent_grid.as_ref()
    }

    /// Get the bond dimension (maximum rank).
    pub fn rank(&self) -> usize {
        self.tt.rank()
    }

    /// Get link dimensions.
    pub fn link_dims(&self) -> Vec<usize> {
        self.tt.link_dims()
    }

    /// Convert grid indices to quantics indices.
    fn grididx_to_quantics(&self, indices: &[usize]) -> Result<Vec<usize>> {
        if let Some(grid) = &self.discretized_grid {
            grid.grididx_to_quantics(indices)
                .map_err(|e| anyhow!("Grid index conversion error: {}", e))
        } else if let Some(grid) = &self.inherent_grid {
            grid.grididx_to_quantics(indices)
                .map_err(|e| anyhow!("Grid index conversion error: {}", e))
        } else {
            Err(anyhow!("No grid available"))
        }
    }

    /// Evaluate at grid indices.
    ///
    /// # Arguments
    /// * `indices` - Grid indices (0-indexed). For a grid of size N,
    ///
    ///   valid indices are `0..N`.
    ///
    /// # Returns
    /// The interpolated value at the specified grid point.
    ///
    /// # Errors
    ///
    /// Returns an error when the grid coordinate conversion fails (an
    /// /// grid shape mismatch failure) or the evaluation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     QtciOptions,
    ///     pointwise_coordinate_batch,
    ///     pointwise_index_batch,
    ///     quanticscrossinterpolate_discrete_batch,
    /// };
    ///
    /// let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
    /// let (qtci, _, _) = quanticscrossinterpolate_discrete_batch::<f64, _>(
    ///     &[4, 4], pointwise_index_batch(f), None, QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
    /// ).unwrap();
    ///
    /// // Indices are 0-indexed: f(1, 2) = 1 + 2 = 3
    /// let val = qtci.evaluate(&[1, 2]).unwrap();
    /// assert!((val - 3.0).abs() < 1e-8);
    /// ```
    pub fn evaluate(&self, indices: &[usize]) -> QtciResult<V> {
        let quantics = self.grididx_to_quantics(indices)?;
        self.tt
            .evaluate(&quantics)
            .map_err(|e| anyhow!("Evaluation error: {e}"))
            .map_err(QuanticsTCIError::from)
    }

    /// Factorized sum over all grid points.
    ///
    /// Computes the sum efficiently using the tensor train structure,
    /// without visiting every grid point individually.
    ///
    /// # Errors
    ///
    /// This method is infallible and always returns `Ok`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     QtciOptions,
    ///     pointwise_coordinate_batch,
    ///     pointwise_index_batch,
    ///     quanticscrossinterpolate_discrete_batch,
    /// };
    ///
    /// // f(i) = 1 on a grid of size 8 => sum = 8
    /// let f = |_idx: &[usize]| 1.0_f64;
    /// let (qtci, _, _) = quanticscrossinterpolate_discrete_batch::<f64, _>(
    ///     &[8], pointwise_index_batch(f), None, QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
    /// ).unwrap();
    ///
    /// let sum = qtci.sum().unwrap();
    /// assert!((sum - 8.0).abs() < 1e-8);
    /// ```
    pub fn sum(&self) -> QtciResult<V> {
        Ok(self.tt.sum())
    }

    /// Integral over the continuous domain (left Riemann sum).
    ///
    /// Computes `sum(f(x_i)) * product(step_sizes)`, a left Riemann sum
    /// with O(h) convergence where h is the grid spacing. The result
    /// depends on the `include_endpoint` setting of the [`DiscretizedGrid`].
    ///
    /// For inherent discrete grids (created via
    /// [`quanticscrossinterpolate_discrete`]), there is no continuous
    /// domain, so this returns the plain [`sum`](Self::sum).
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying summation reports a failure (a
    /// [`QuanticsTCIError::Operation`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     DiscretizedGrid,
    ///     QtciOptions,
    ///     pointwise_coordinate_batch,
    ///     pointwise_index_batch,
    ///     quanticscrossinterpolate_batch,
    /// };
    ///
    /// // Integrate f(x) = 1 over [0, 1) with 16 points => integral = 1.0
    /// let grid = DiscretizedGrid::builder(&[4])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    /// let f = |_: &[f64]| 1.0_f64;
    /// let (qtci, _, _) = quanticscrossinterpolate_batch::<f64, _>(
    ///     &grid, pointwise_coordinate_batch(f), None, QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
    /// ).unwrap();
    ///
    /// let integral = qtci.integral().unwrap();
    /// assert!((integral - 1.0).abs() < 1e-8);
    /// ```
    pub fn integral(&self) -> QtciResult<V>
    where
        V: std::ops::Mul<f64, Output = V>,
    {
        let sum_val = self.sum()?;
        if let Some(grid) = &self.discretized_grid {
            let step_product: f64 = grid.grid_step().iter().product();
            Ok(sum_val * step_product)
        } else {
            // For inherent discrete grids, just return the sum
            Ok(sum_val)
        }
    }

    /// Get the underlying [`SimpleTensorTrain`].
    ///
    /// Returns a clone of the tensor train. Use this to pass the result
    /// to other tensor-train operations (contraction, SVD compression, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     AbstractTensorTrain,
    ///     QtciOptions,
    ///     pointwise_coordinate_batch,
    ///     pointwise_index_batch,
    ///     quanticscrossinterpolate_discrete_batch,
    /// };
    ///
    /// let f = |idx: &[usize]| idx[0] as f64;
    /// let (qtci, _, _) = quanticscrossinterpolate_discrete_batch::<f64, _>(
    ///     &[4], pointwise_index_batch(f), None, QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
    /// ).unwrap();
    ///
    /// let tt = qtci.tensor_train();
    /// assert!(tt.rank() >= 1);
    /// assert!(tt.len() > 0);
    /// ```
    pub fn tensor_train(&self) -> SimpleTensorTrain<V> {
        self.tt.clone()
    }

    /// Access the TreeTCI2 state.
    pub fn tci(&self) -> &TreeTCI2<V> {
        &self.tci_state
    }

    /// Target-evaluation accounting of the run that produced this result.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::quanticscrossinterpolate_discrete_batch;
    /// # use tensor4all_quanticstci::{QtciOptions, QuanticsBatch};
    /// # let f = |batch: QuanticsBatch<'_, usize>| -> anyhow::Result<Vec<f64>> {
    /// #     Ok((0..batch.n_points()).map(|p| batch.get(0, p).unwrap() as f64).collect())
    /// # };
    /// # let (qtci, _, _) = quanticscrossinterpolate_discrete_batch::<f64, _>(
    /// #     &[4], f, None, QtciOptions::default()).unwrap();
    /// let stats = qtci.cache_stats();
    /// assert!(stats.num_evals > 0);
    /// assert!(stats.num_cache_hits + stats.num_cache_misses >= stats.num_evals);
    /// ```
    pub fn cache_stats(&self) -> CacheStats {
        self.cache_stats
    }

    /// Distinct points the target function evaluated.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensor4all_quanticstci::{quanticscrossinterpolate_discrete_batch, QtciOptions, QuanticsBatch};
    /// # let f = |batch: QuanticsBatch<'_, usize>| -> anyhow::Result<Vec<f64>> {
    /// #     Ok((0..batch.n_points()).map(|p| batch.get(0, p).unwrap() as f64).collect())
    /// # };
    /// # let (qtci, _, _) = quanticscrossinterpolate_discrete_batch::<f64, _>(
    /// #     &[4], f, None, QtciOptions::default()).unwrap();
    /// assert!(qtci.num_evals() > 0);
    /// ```
    pub fn num_evals(&self) -> usize {
        self.cache_stats.num_evals
    }

    /// Point requests served from the cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensor4all_quanticstci::{quanticscrossinterpolate_discrete_batch, QtciOptions, QuanticsBatch};
    /// # let f = |batch: QuanticsBatch<'_, usize>| -> anyhow::Result<Vec<f64>> {
    /// #     Ok((0..batch.n_points()).map(|p| batch.get(0, p).unwrap() as f64).collect())
    /// # };
    /// # let (qtci, _, _) = quanticscrossinterpolate_discrete_batch::<f64, _>(
    /// #     &[4], f, None, QtciOptions::default()).unwrap();
    /// assert_eq!(qtci.num_cache_hits(), qtci.cache_stats().num_cache_hits);
    /// ```
    pub fn num_cache_hits(&self) -> usize {
        self.cache_stats.num_cache_hits
    }

    /// Fraction of point requests served from the cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensor4all_quanticstci::{quanticscrossinterpolate_discrete_batch, QtciOptions, QuanticsBatch};
    /// # let f = |batch: QuanticsBatch<'_, usize>| -> anyhow::Result<Vec<f64>> {
    /// #     Ok((0..batch.n_points()).map(|p| batch.get(0, p).unwrap() as f64).collect())
    /// # };
    /// # let (qtci, _, _) = quanticscrossinterpolate_discrete_batch::<f64, _>(
    /// #     &[4], f, None, QtciOptions::default()).unwrap();
    /// assert!((0.0..=1.0).contains(&qtci.cache_hit_ratio()));
    /// ```
    pub fn cache_hit_ratio(&self) -> f64 {
        self.cache_stats.hit_ratio()
    }
}

/// Interpolate a function with an explicit Grid, evaluating `f` in batches, on a
/// caller-owned random stream.
///
/// `f` receives a [`QuanticsBatch`] of original coordinates and must return one
/// value per requested point, in the same order. The batch is column-major
/// `(n_dims, n_points)`, so point `p` occupies `batch.point(p)`. Use this entry
/// point — not the deprecated point-wise [`quanticscrossinterpolate`] — for
/// vectorized targets and language bindings.
///
/// Every random draw — the random initial pivots and the global searches of the
/// underlying tree TCI — comes from `rng`, and [`QtciOptions::rng_seed`] is
/// ignored.
///
/// # Arguments
/// * `grid` - Discretized grid describing the function domain
/// * `f` - Batched function to interpolate
/// * `initial_pivots` - Initial pivot grid indices (optional)
/// * `options` - TCI options; `rng_seed` is ignored
/// * `rng` - Caller-owned random stream
///
/// # Returns
/// Tuple of ([`QuanticsTensorCI2`], ranks per sweep, errors per sweep)
///
/// # Errors
///
/// Returns [`QuanticsTCIError::InvalidConfiguration`] when the grid or options
/// are invalid. Returns [`QuanticsTCIError::Operation`] when an initial pivot
/// conversion fails, `f` returns a value count that does not equal the number
/// of requested points, or the underlying interpolation fails.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng as _;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_quanticstci::{
///     DiscretizedGrid,
///     QtciOptions,
///     QuanticsBatch,
///     pointwise_coordinate_batch,
///     pointwise_index_batch,
///     quanticscrossinterpolate_batch_with_rng,
/// };
///
/// let grid = DiscretizedGrid::builder(&[4])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[std::f64::consts::PI])
///     .build()
///     .unwrap();
///
/// let f = |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
///     Ok((0..batch.n_points())
///         .map(|point| batch.get(0, point).unwrap().sin())
///         .collect())
/// };
/// let mut rng = ChaCha8Rng::seed_from_u64(0);
/// let (qtci, _ranks, errors) =
///     quanticscrossinterpolate_batch_with_rng::<f64, _, _>(&grid, f, None, QtciOptions::default(), &mut rng).unwrap();
///
/// assert!(*errors.last().unwrap() < 1e-6);
/// assert!(qtci.sum().unwrap() > 0.0); // sin(x) > 0 on (0, pi)
/// ```
pub fn quanticscrossinterpolate_batch_with_rng<V, F, R>(
    grid: &DiscretizedGrid,
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
    rng: &mut R,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
    R: rand::Rng + ?Sized,
{
    let local_dims = grid.local_dimensions();
    let cache: MemoCache<V> = Rc::new(RefCell::new(new_memo_cache(&local_dims)?));
    let grid_for_evaluation = grid.clone();
    let evaluate = site_evaluator(
        move |quantics: &[usize]| {
            grid_for_evaluation
                .quantics_to_origcoord(quantics)
                .map_err(|error| anyhow!("failed to convert quantics index {quantics:?}: {error}"))
        },
        f,
        cache.clone(),
    );
    // Erase the caller's RNG type once so the run below is instantiated once
    // per scalar type instead of once per (scalar, RNG) pair.
    let mut stream: &mut R = rng;
    let pivots = prepare_pivots(
        initial_pivots,
        &local_dims,
        |pivot| {
            grid.grididx_to_quantics(pivot)
                .map_err(|error| anyhow!("initial pivot {pivot:?} conversion failed: {error}"))
        },
        options.n_random_init_pivot,
        &mut stream,
    )?;

    let (tci, tt, ranks, errors) =
        run_treetci_batch(local_dims, evaluate, cache.clone(), pivots, &options)?;
    let cache_stats = memo_cache_stats(&cache);
    Ok((
        QuanticsTensorCI2::from_discretized(tt, tci, grid.clone(), cache_stats),
        ranks,
        errors,
    ))
}

/// Interpolate a quantics function with an explicitly named deterministic seed.
///
/// Builds a `ChaCha8Rng` from [`QtciOptions::rng_seed`] (OS entropy when unset,
/// drawn once per run) and delegates to
/// [`quanticscrossinterpolate_batch_with_rng`], which ignores the option because
/// it consumes the caller's stream directly.
///
/// # Errors
/// Returns [`QuanticsTCIError::InvalidConfiguration`] when the grid or options
/// are invalid. Returns [`QuanticsTCIError::Operation`] when an initial pivot
/// conversion fails, `f` returns a value count that does not equal the number
/// of requested points, or the underlying interpolation fails.
pub fn quanticscrossinterpolate_batch<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
{
    let mut rng = seeded_quantics_stream(&options);
    quanticscrossinterpolate_batch_with_rng(grid, f, initial_pivots, options, &mut rng)
}

/// Interpolate a function with an explicit Grid, one point at a time.
///
/// Deprecated: `f` is called once per grid point, which is the wrong boundary
/// for vectorized functions and language bindings. Use
/// [`quanticscrossinterpolate_batch`] instead, optionally with
/// [`pointwise_coordinate_batch`](crate::pointwise_coordinate_batch) to keep a
/// point-wise closure.
///
/// # Errors
///
/// Returns the same errors as [`quanticscrossinterpolate_batch`].
#[deprecated(
    note = "calls `f` once per point; use `quanticscrossinterpolate_batch` (optionally with `pointwise_coordinate_batch`) instead"
)]
pub fn quanticscrossinterpolate<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(&[f64]) -> V + 'static,
{
    quanticscrossinterpolate_batch(grid, pointwise_coordinate_batch(f), initial_pivots, options)
}

/// Interpolate from explicit grid point arrays, evaluating `f` in batches.
///
/// Convenience wrapper around [`quanticscrossinterpolate_batch_with_rng`] and
/// [`quanticscrossinterpolate_discrete_batch_with_rng`] that evaluates `f` at the
/// exact coordinates supplied in `xvals`; `f` always receives original
/// coordinates, batched as in [`quanticscrossinterpolate_batch`]. Every random
/// draw comes from `rng` and [`QtciOptions::rng_seed`] is ignored.
///
/// # Arguments
/// * `xvals` - Strictly increasing, finite coordinate arrays. All dimensions must have
///   the **same** number of points and each must be a power of 2.
/// * `f` - Batched function to interpolate, one value per requested point
/// * `initial_pivots` - Initial pivot grid indices (0-indexed, optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of ([`QuanticsTensorCI2`], ranks per sweep, errors per sweep)
///
/// # Errors
///
/// Returns [`QuanticsTCIError::InvalidConfiguration`] when `xvals` is empty,
/// contains an empty, non-finite, or non-increasing dimension, or when the
/// options are invalid. Returns [`QuanticsTCIError::Operation`] when an initial
/// pivot conversion fails, `f` returns a value count that does not equal the
/// number of requested points, or the underlying interpolation fails.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng as _;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_quanticstci::{
///     QtciOptions,
///     QuanticsBatch,
///     pointwise_coordinate_batch,
///     pointwise_index_batch,
///     quanticscrossinterpolate_from_arrays_batch_with_rng,
/// };
///
/// // 4 points in [0, 3]
/// let xvals = vec![vec![0.0, 1.0, 2.0, 3.0]];
/// let f = |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
///     Ok((0..batch.n_points())
///         .map(|point| batch.get(0, point).unwrap().powi(2))
///         .collect())
/// };
/// let mut rng = ChaCha8Rng::seed_from_u64(0);
/// let (qtci, _, _) =
///     quanticscrossinterpolate_from_arrays_batch_with_rng::<f64, _, _>(&xvals, f, None, QtciOptions::default(), &mut rng)
///         .unwrap();
///
/// // Grid index 2 maps to x = 2.0, so f = 4.0
/// let val = qtci.evaluate(&[2]).unwrap();
/// assert!((val - 4.0).abs() < 1e-8);
/// ```
pub fn quanticscrossinterpolate_from_arrays_batch_with_rng<V, F, R>(
    xvals: &[Vec<f64>],
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
    rng: &mut R,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
    R: rand::Rng + ?Sized,
{
    if xvals.is_empty() {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "xvals must not be empty".to_string(),
        });
    }
    if xvals.iter().any(|x| x.is_empty()) {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "xvals must not contain empty dimensions".to_string(),
        });
    }

    for (dimension, values) in xvals.iter().enumerate() {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(QuanticsTCIError::InvalidConfiguration {
                message: format!("xvals[{dimension}] must contain only finite values"),
            });
        }
        if values.windows(2).any(|window| window[0] >= window[1]) {
            return Err(QuanticsTCIError::InvalidConfiguration {
                message: format!(
                    "xvals[{dimension}] must be strictly increasing without duplicates"
                ),
            });
        }
    }

    let sizes = xvals.iter().map(Vec::len).collect::<Vec<_>>();
    let dimensions: Vec<f64> = sizes.iter().map(|&size| (size as f64).log2()).collect();
    if !dimensions
        .windows(2)
        .all(|window| (window[0] - window[1]).abs() < 1e-10)
    {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message:
                "this method only supports grids with equal number of points in each direction"
                    .to_string(),
        });
    }
    if !dimensions
        .iter()
        .all(|&dimension| (dimension - dimension.round()).abs() < 1e-10)
    {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "this method only supports grid sizes that are powers of 2".to_string(),
        });
    }

    let is_uniform = xvals.iter().all(|values| {
        let Some(first_window) = values.windows(2).next() else {
            return true;
        };
        let step = first_window[1] - first_window[0];
        values
            .windows(2)
            .all(|window| (window[1] - window[0] - step).abs() <= 1e-12)
    });
    if is_uniform {
        let rs = dimensions
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let lower = xvals
            .iter()
            .map(|values| {
                values
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("xvals must not be empty"))
            })
            .collect::<Result<Vec<_>>>()?;
        let upper = xvals
            .iter()
            .map(|values| {
                values
                    .last()
                    .copied()
                    .ok_or_else(|| anyhow!("xvals must not be empty"))
            })
            .collect::<Result<Vec<_>>>()?;
        let grid = DiscretizedGrid::builder(&rs)
            .with_lower_bound(&lower)
            .with_upper_bound(&upper)
            .with_unfolding_scheme(options.unfolding_scheme)
            .include_endpoint(true)
            .build()
            .map_err(|error| anyhow!("Failed to build grid: {error}"))?;
        return quanticscrossinterpolate_batch_with_rng(&grid, f, initial_pivots, options, rng);
    }

    // Non-uniform coordinates: map grid indices to the supplied coordinates.
    let coordinates = xvals.to_vec();
    let mapped = move |batch: QuanticsBatch<'_, usize>| -> Result<Vec<V>> {
        let n_dims = batch.n_dims();
        let n_points = batch.n_points();
        let mut coords = Vec::with_capacity(n_dims * n_points);
        for point in 0..n_points {
            for dimension in 0..n_dims {
                let index = batch.get(dimension, point).ok_or_else(|| {
                    anyhow!("invalid batch index: dimension {dimension}, point {point}")
                })?;
                let value = coordinates
                    .get(dimension)
                    .and_then(|values| values.get(index))
                    .ok_or_else(|| {
                        anyhow!("grid index {index} is out of range for xvals[{dimension}]")
                    })?;
                coords.push(*value);
            }
        }
        f(QuanticsBatch::new(&coords, n_dims, n_points)?)
    };

    quanticscrossinterpolate_discrete_batch_with_rng(&sizes, mapped, initial_pivots, options, rng)
}

/// The same entry point with an explicitly named deterministic seed.
///
/// Builds a `ChaCha8Rng` from [`QtciOptions::rng_seed`] (OS entropy when unset,
/// drawn once per run) and delegates to
/// [`quanticscrossinterpolate_from_arrays_batch_with_rng`], which ignores the option because it consumes the
/// caller's stream directly.
///
/// # Errors
/// Returns [`QuanticsTCIError::InvalidConfiguration`] when `xvals` is empty,
/// contains an empty, non-finite, or non-increasing dimension, or when the
/// options are invalid. Returns [`QuanticsTCIError::Operation`] when an initial
/// pivot conversion fails, `f` returns a value count that does not equal the
/// number of requested points, or the underlying interpolation fails.
pub fn quanticscrossinterpolate_from_arrays_batch<V, F>(
    xvals: &[Vec<f64>],
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
{
    let mut rng = seeded_quantics_stream(&options);
    quanticscrossinterpolate_from_arrays_batch_with_rng(xvals, f, initial_pivots, options, &mut rng)
}

/// Interpolate from explicit grid point arrays, one point at a time.
///
/// Deprecated: `f` is called once per grid point. Use
/// [`quanticscrossinterpolate_from_arrays_batch`] instead, optionally with
/// [`pointwise_coordinate_batch`](crate::pointwise_coordinate_batch) to keep a
/// point-wise closure.
///
/// # Errors
///
/// Returns the same errors as [`quanticscrossinterpolate_from_arrays_batch`].
#[deprecated(
    note = "calls `f` once per point; use `quanticscrossinterpolate_from_arrays_batch` (optionally with `pointwise_coordinate_batch`) instead"
)]
pub fn quanticscrossinterpolate_from_arrays<V, F>(
    xvals: &[Vec<f64>],
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(&[f64]) -> V + 'static,
{
    quanticscrossinterpolate_from_arrays_batch(
        xvals,
        pointwise_coordinate_batch(f),
        initial_pivots,
        options,
    )
}

/// Interpolate a function defined on a discrete integer grid, in batches.
///
/// Use this when your function is naturally indexed by integers (e.g.,
/// lattice models, combinatorial functions). Grid indices are **0-indexed**:
/// the first grid point is `[0, 0, ...]`, and the last is
/// `[size[0] - 1, size[1] - 1, ...]`.
///
/// `f` receives a [`QuanticsBatch<usize>`] of grid indices and must return one
/// value per requested point, in the same order. The batch is column-major
/// `(n_dims, n_points)`, so point `p` occupies `batch.point(p)`.
///
/// For functions on continuous domains, use
/// [`quanticscrossinterpolate_batch_with_rng`] with a [`DiscretizedGrid`]
/// instead. Every random draw comes from `rng` and [`QtciOptions::rng_seed`] is
/// ignored.
///
/// # Arguments
/// * `size` - Grid size in each dimension. All dimensions must have the **same**
///   number of points and each must be a power of 2 (e.g., `&[16, 16]`).
/// * `f` - Batched function to interpolate, taking **0-indexed** grid indices
/// * `initial_pivots` - Initial pivot grid indices (0-indexed, optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of ([`QuanticsTensorCI2`], ranks per sweep, errors per sweep)
///
/// # Errors
///
/// Returns [`QuanticsTCIError::InvalidConfiguration`] when the grid size or
/// options are invalid. Returns [`QuanticsTCIError::Operation`] when an initial
/// pivot conversion fails, `f` returns a value count that does not equal the
/// number of requested points, or the underlying interpolation fails.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng as _;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_quanticstci::{
///     QtciOptions,
///     QuanticsBatch,
///     pointwise_coordinate_batch,
///     pointwise_index_batch,
///     quanticscrossinterpolate_discrete_batch_with_rng,
/// };
///
/// let f = |batch: QuanticsBatch<'_, usize>| -> anyhow::Result<Vec<f64>> {
///     Ok((0..batch.n_points())
///         .map(|point| (batch.get(0, point).unwrap() * 10 + batch.get(1, point).unwrap()) as f64)
///         .collect())
/// };
/// let mut rng = ChaCha8Rng::seed_from_u64(0);
/// let (qtci, _ranks, _errors) =
///     quanticscrossinterpolate_discrete_batch_with_rng::<f64, _, _>(
///         &[16, 16],
///         f,
///         None,
///         QtciOptions::default(),
///         &mut rng,
///     )
///     .unwrap();
///
/// // Evaluate: f(2, 4) = 2 * 10 + 4 = 24
/// let val = qtci.evaluate(&[2, 4]).unwrap();
/// assert!((val - 24.0).abs() < 1e-8);
/// ```
pub fn quanticscrossinterpolate_discrete_batch_with_rng<V, F, R>(
    size: &[usize],
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
    rng: &mut R,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(QuanticsBatch<'_, usize>) -> Result<Vec<V>>,
    R: rand::Rng + ?Sized,
{
    if size.is_empty() {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "this method requires at least one grid dimension, got an empty size"
                .to_string(),
        });
    }
    // Validate sizes are powers of 2
    let dimensions: Vec<f64> = size.iter().map(|&s| (s as f64).log2()).collect();

    if !dimensions.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message:
                "this method only supports grids with equal number of points in each direction"
                    .to_string(),
        });
    }

    if !dimensions.iter().all(|&d| (d - d.round()).abs() < 1e-10) {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "this method only supports grid sizes that are powers of 2".to_string(),
        });
    }

    let r = dimensions[0] as usize;
    let n = size.len();

    // Build inherent discrete grid - rs is the number of bits per variable
    let rs: Vec<usize> = vec![r; n];
    let grid = InherentDiscreteGrid::builder(&rs)
        .with_unfolding_scheme(options.unfolding_scheme)
        .build()
        .map_err(|e| anyhow!("Failed to build grid: {}", e))?;

    let local_dims = grid.local_dimensions();
    let cache: MemoCache<V> = Rc::new(RefCell::new(new_memo_cache(&local_dims)?));
    let grid_for_evaluation = grid.clone();
    let evaluate = site_evaluator(
        move |quantics: &[usize]| {
            grid_for_evaluation
                .quantics_to_grididx(quantics)
                .map_err(|error| anyhow!("failed to convert quantics index {quantics:?}: {error}"))
        },
        f,
        cache.clone(),
    );
    // Erase the caller's RNG type once so the run below is instantiated once
    // per scalar type instead of once per (scalar, RNG) pair.
    let mut stream: &mut R = rng;
    let pivots = prepare_pivots(
        initial_pivots,
        &local_dims,
        |pivot| {
            grid.grididx_to_quantics(pivot)
                .map_err(|error| anyhow!("initial pivot {pivot:?} conversion failed: {error}"))
        },
        options.n_random_init_pivot,
        &mut stream,
    )?;

    let (tci, tt, ranks, errors) =
        run_treetci_batch(local_dims, evaluate, cache.clone(), pivots, &options)?;
    let cache_stats = memo_cache_stats(&cache);
    Ok((
        QuanticsTensorCI2::from_inherent(tt, tci, grid, cache_stats),
        ranks,
        errors,
    ))
}

/// The same entry point with an explicitly named deterministic seed.
///
/// Builds a `ChaCha8Rng` from [`QtciOptions::rng_seed`] (OS entropy when unset,
/// drawn once per run) and delegates to
/// [`quanticscrossinterpolate_discrete_batch_with_rng`], which ignores the option because it consumes the
/// caller's stream directly.
///
/// # Errors
/// Returns [`QuanticsTCIError::InvalidConfiguration`] when the grid size or
/// options are invalid. Returns [`QuanticsTCIError::Operation`] when an initial
/// pivot conversion fails, `f` returns a value count that does not equal the
/// number of requested points, or the underlying interpolation fails.
pub fn quanticscrossinterpolate_discrete_batch<V, F>(
    size: &[usize],
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(QuanticsBatch<'_, usize>) -> Result<Vec<V>>,
{
    let mut rng = seeded_quantics_stream(&options);
    quanticscrossinterpolate_discrete_batch_with_rng(size, f, initial_pivots, options, &mut rng)
}

/// Interpolate a function defined on a discrete integer grid, one point at a time.
///
/// Deprecated: `f` is called once per grid point. Use
/// [`quanticscrossinterpolate_discrete_batch`] instead, optionally with
/// [`pointwise_index_batch`](crate::pointwise_index_batch) to keep a point-wise
/// closure.
///
/// # Errors
///
/// Returns the same errors as [`quanticscrossinterpolate_discrete_batch`].
#[deprecated(
    note = "calls `f` once per point; use `quanticscrossinterpolate_discrete_batch` (optionally with `pointwise_index_batch`) instead"
)]
pub fn quanticscrossinterpolate_discrete<V, F>(
    size: &[usize],
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(&[usize]) -> V + 'static,
{
    quanticscrossinterpolate_discrete_batch(size, pointwise_index_batch(f), initial_pivots, options)
}

#[cfg(test)]
mod tests;
