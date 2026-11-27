//! Batched (vector/tensor-valued) Quantics TCI interpolation.
//!
//! This module provides [`quanticscrossinterpolate_multicomponent`], which
//! interpolates vector- or tensor-valued functions by interpolating each output
//! component independently and combining the results into a single
//! [`SimpleTensorTrain`] with an additional component site.
//!
//! The component runs share one memo of the target's answers, keyed by the
//! quantics multi-index and living for one call: the callback returns all
//! components of a point, so a point one component evaluated is served to the
//! others without another callback call. Each component's fit is unchanged by
//! this (#1463); only the callback's traffic is.

use anyhow::{anyhow, Result};
use quanticsgrids::DiscretizedGrid;
use tensor4all_core::{TensorElement, DEFAULT_RETAINED_BYTE_LIMIT};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar};
use tensor4all_simplett::{Tensor3, Tensor3Ops};
use tensor4all_tensorbackend::FullPivLuScalar;

use crate::batch::QuanticsBatch;
use crate::error::{QuanticsTCIError, Result as QtciResult};
use crate::options::QtciOptions;
use crate::quantics_tci::SharedSampling;

/// Retained-byte limit of the memo the component runs of one multicomponent call
/// share (tables and payload together): the default 256 MiB of
/// [`MultiIndexCache`](tensor4all_core::MultiIndexCache). Not caller-tunable.
pub(crate) const MULTICOMPONENT_SHARED_MEMO_BYTES: usize = DEFAULT_RETAINED_BYTE_LIMIT;

/// Interpolate a multi-component function with an explicitly named seed.
///
/// Builds a `ChaCha8Rng` from `QtciOptions::rng_seed` (OS entropy when unset)
/// and delegates to [`quanticscrossinterpolate_multicomponent_with_rng`], so one
/// stream drives every component. Components share their sampled points, which
/// makes the callback's contract stricter than for a single component:
///
/// * `f` must be a pure function of the point **and independent of batching and
///   call order**: the shared memo sends one batch for several components, so a
///   callback that branches on `n_points` or on how often it was called behaves
///   differently from before;
/// * the memo retains at most 256 MiB (`MULTICOMPONENT_SHARED_MEMO_BYTES`, the
///   default retained-byte limit of the cache) and the limit is not tunable.
///
/// See [`quanticscrossinterpolate_multicomponent_with_rng`] for the details,
/// including the number of values `f` evaluates.
///
/// # Errors
/// Returns [`QuanticsTCIError::InvalidConfiguration`] when `output_dims` is
/// empty or has a zero factor, or when the grid or options are invalid.
/// Returns [`QuanticsTCIError::Operation`] when `f` returns a value count that
/// does not equal `n_points * product(output_dims)`, or when the underlying
/// component interpolation fails.
pub fn quanticscrossinterpolate_multicomponent<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    output_dims: &[usize],
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2Batched<V>, Vec<usize>, Vec<f64>)>
where
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
    V: TTScalar
        + Default
        + Clone
        + 'static
        + TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
{
    let mut rng = super::quantics_tci::seeded_quantics_stream(&options);
    quanticscrossinterpolate_multicomponent_with_rng(
        grid,
        f,
        output_dims,
        initial_pivots,
        options,
        &mut rng,
    )
}

/// Result of batched (vector/tensor-valued) Quantics TCI interpolation.
///
/// Wraps a [`SimpleTensorTrain`] where the last site is a component index,
/// plus the output shape and grid information.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     quanticscrossinterpolate_multicomponent, AbstractTensorTrain, DiscretizedGrid,
///     QtciOptions, QuanticsBatch,
/// };
///
/// let grid = DiscretizedGrid::builder(&[2])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
///     &grid,
///     |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
///         let mut values = Vec::with_capacity(2 * batch.n_points());
///         for point in 0..batch.n_points() {
///             let x = batch.get(0, point).unwrap();
///             values.extend([x + 1.0, 2.0 * x + 1.0]);
///         }
///         Ok(values)
///     },
///     &[2],
///     None,
///     QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
/// ).unwrap();
///
/// assert_eq!(result.output_dims(), &[2]);
/// assert_eq!(result.tensor_train().len(), 3); // 2 grid sites + 1 component site
/// ```
#[derive(Clone)]
pub struct QuanticsTensorCI2Batched<V: TTScalar> {
    /// Combined tensor train with component index as the last site.
    tt: SimpleTensorTrain<V>,
    /// Shape of the output (e.g., [3] for 3-vector, [2, 2] for 2x2 matrix).
    output_dims: Vec<usize>,
    /// Grid for coordinate conversion.
    grid: DiscretizedGrid,
}

impl<V> QuanticsTensorCI2Batched<V>
where
    V: TTScalar + Default + Clone,
{
    /// Get the combined tensor train.
    ///
    /// The last site of the tensor train is the component index with
    /// dimension equal to the product of `output_dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_multicomponent, AbstractTensorTrain, DiscretizedGrid,
    ///     QtciOptions, QuanticsBatch,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
    ///     &grid,
    ///     |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
    ///         let mut values = Vec::with_capacity(2 * batch.n_points());
    ///         for point in 0..batch.n_points() {
    ///             let x = batch.get(0, point).unwrap();
    ///             values.extend([x + 1.0, x * x + 1.0]);
    ///         }
    ///         Ok(values)
    ///     },
    ///     &[2],
    ///     None,
    ///     QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
    /// ).unwrap();
    ///
    /// let tt = result.tensor_train();
    /// assert_eq!(tt.len(), 3); // 2 grid sites + 1 component site
    /// ```
    pub fn tensor_train(&self) -> &SimpleTensorTrain<V> {
        &self.tt
    }

    /// Get the output dimensions (shape of the function output).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_multicomponent, QtciOptions, DiscretizedGrid, QuanticsBatch,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
    ///     &grid,
    ///     |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
    ///         let mut values = Vec::with_capacity(2 * batch.n_points());
    ///         for point in 0..batch.n_points() {
    ///             let x = batch.get(0, point).unwrap();
    ///             values.extend([x + 1.0, x * x + 1.0]);
    ///         }
    ///         Ok(values)
    ///     },
    ///     &[2],
    ///     None,
    ///     QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
    /// ).unwrap();
    ///
    /// assert_eq!(result.output_dims(), &[2]);
    /// ```
    pub fn output_dims(&self) -> &[usize] {
        &self.output_dims
    }

    /// Get the discretized grid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_multicomponent, QtciOptions, DiscretizedGrid, QuanticsBatch,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_multicomponent::<f64, _>(
    ///     &grid,
    ///     |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
    ///         Ok((0..batch.n_points())
    ///             .map(|point| batch.get(0, point).unwrap() + 1.0)
    ///             .collect())
    ///     },
    ///     &[1],
    ///     None,
    ///     QtciOptions { rng_seed: Some(0), ..QtciOptions::default() },
    /// ).unwrap();
    ///
    /// assert!(result.grid().grid_step().len() > 0);
    /// ```
    pub fn grid(&self) -> &DiscretizedGrid {
        &self.grid
    }
}

/// Interpolate a vector/tensor-valued function, evaluating `f` in batches, on a
/// caller-owned random stream.
///
/// Each output component is interpolated with
/// [`quanticscrossinterpolate_batch_with_rng`](crate::quanticscrossinterpolate_batch_with_rng) on the *same* stream, in
/// component order, and the per-component tensor trains are
/// combined into a single [`SimpleTensorTrain`] with an additional component
/// site at the end.
///
/// # Shared sampling
///
/// The component runs share one memo, created for this call and dropped with
/// it, keyed by the quantics multi-index (never by float coordinates). The
/// callback answers all components of a point, so the memo keeps all
/// `n_components` values per point; a run asks `f` only for the distinct points
/// no earlier run (or earlier request of its own) evaluated, in one batch per
/// request, and a request that is all hits does not call `f` at all. The
/// first component's calls are the same as without sharing; later components
/// send fewer batches, possibly none, and never a point already evaluated.
///
/// Tensor trains, ranks, errors and the final position of `rng` are identical
/// to running the components one after the other with separate memos: only the
/// callback's traffic changes.
///
/// **Cost.** Every call of `f` evaluates all `n_components` values of each
/// requested point. Without sharing, each component run `i` requests the set
/// `S_i` of points it samples and discards the other components' values, so `f`
/// evaluates `n_components * Σ|S_i|` values. With the memo it evaluates
/// `n_components * |∪ S_i|` values (while nothing is evicted).
///
/// **Precondition:** `f` must be a pure function of the point **and independent
/// of batching and call order**. A value computed for one component is reused
/// for the others and the memo sends one batch for several components, so a
/// stateful or noisy callback (one that answers the same point differently on
/// different calls), or one that branches on `n_points` or on how often it has
/// been called, is called less often than before, with different batches, and
/// can give different fits.
///
/// The memo retains at most 256 MiB (`MULTICOMPONENT_SHARED_MEMO_BYTES`, the
/// default retained-byte limit of the cache) of table and payload bytes together; the payload is
/// `n_components * size_of::<V>()` plus the allocation header of each point's
/// value block. The limit is not tunable by the caller. When it is reached the
/// oldest entries are evicted and an evicted point is simply evaluated again if
/// requested later, so evictions change the number of callback calls but never
/// a value. For large `n_components` (about 64 `f64` components per point, a
/// few hundred bytes each) the limit holds a few hundred thousand points.
/// Counters of the memo are internal and not part of the return value.
///
/// A grid whose quantics key is wider than 1024 bits is rejected before any
/// callback call.
///
/// # Arguments
///
/// * `grid` - Discretized grid describing the function domain
/// * `f` - Batched function to interpolate. It receives original coordinates as
///   a [`QuanticsBatch`] and must return `n_points * n_components` values, where
///   `n_components = product(output_dims)`: the components of each requested
///   point consecutively, in point order.
/// * `output_dims` - Shape of the function output (e.g., `&[3]` for 3-vector,
///
///   `&[2, 2]` for 2x2 matrix)
/// * `initial_pivots` - Initial pivot grid indices (0-indexed, optional)
/// * `options` - TCI options; `rng_seed` is ignored
/// * `rng` - Caller-owned random stream. One stream drives every component, so
///   the whole run is reproducible from the stream's position
///
/// # Returns
///
/// Tuple of ([`QuanticsTensorCI2Batched`], max_bond_dims_across_components, max_errors_across_components)
///
/// # Errors
///
/// Returns [`QuanticsTCIError::InvalidConfiguration`] when `output_dims` is
/// empty or has a zero factor, or when the grid or options are invalid.
/// Returns [`QuanticsTCIError::Operation`] when `f` returns a value count that
/// does not equal `n_points * product(output_dims)`, or when the underlying
/// component interpolation fails. An error in any component, also one raised
/// after sampling (for example initial pivots that all evaluate to zero for one
/// component), aborts all components and no partial result is returned. The
/// error variant is the component run's own, and its message is prefixed with
/// `component <index> of <n_components>` to name the failing component (the
/// original error stays in the source chain). A wrong value count or an error
/// returned by `f` inserts nothing into the memo.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng as _;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_quanticstci::{
///     quanticscrossinterpolate_multicomponent_with_rng, AbstractTensorTrain, DiscretizedGrid,
///     QtciOptions, QuanticsBatch,
/// };
///
/// let grid = DiscretizedGrid::builder(&[2])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let f = |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
///     let mut values = Vec::with_capacity(2 * batch.n_points());
///     for point in 0..batch.n_points() {
///         let x = batch.get(0, point).unwrap();
///         values.extend([x + 1.0, 2.0 * x + 1.0]);
///     }
///     Ok(values)
/// };
///
/// let mut rng = ChaCha8Rng::seed_from_u64(0);
/// let (result, ranks, errors) = quanticscrossinterpolate_multicomponent_with_rng::<f64, _, _>(
///     &grid,
///     f,
///     &[2],
///     None,
///     QtciOptions::default().with_tolerance(1e-8),
///     &mut rng,
/// ).unwrap();
///
/// assert_eq!(result.tensor_train().len(), 3); // 2 grid sites + 1 component site
/// assert_eq!(result.output_dims(), &[2]);
/// assert!(!ranks.is_empty());
/// assert!(!errors.is_empty());
/// ```
///
/// The callback is asked for each point at most once, however many components
/// the target has:
///
/// ```
/// use std::cell::RefCell;
/// use std::collections::BTreeSet;
///
/// use rand::SeedableRng as _;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_quanticstci::{
///     quanticscrossinterpolate_multicomponent_with_rng, DiscretizedGrid, QtciOptions,
///     QuanticsBatch,
/// };
///
/// let grid = DiscretizedGrid::builder(&[4])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let requested = RefCell::new(Vec::new());
/// let f = |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
///     let mut values = Vec::new();
///     for point in 0..batch.n_points() {
///         let x = batch.get(0, point).unwrap();
///         requested.borrow_mut().push(x.to_bits());
///         values.extend([x + 1.0, x * x + 1.0, 2.0 * x + 1.0]);
///     }
///     Ok(values)
/// };
///
/// let mut rng = ChaCha8Rng::seed_from_u64(0);
/// quanticscrossinterpolate_multicomponent_with_rng::<f64, _, _>(
///     &grid,
///     f,
///     &[3],
///     None,
///     QtciOptions::default().with_tolerance(1e-8),
///     &mut rng,
/// ).unwrap();
///
/// let requested = requested.into_inner();
/// let distinct: BTreeSet<u64> = requested.iter().copied().collect();
/// assert_eq!(requested.len(), distinct.len()); // no point requested twice
/// assert!(distinct.len() <= 16); // 2^4 grid points
/// ```
pub fn quanticscrossinterpolate_multicomponent_with_rng<V, F, R>(
    grid: &DiscretizedGrid,
    f: F,
    output_dims: &[usize],
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
    rng: &mut R,
) -> QtciResult<(QuanticsTensorCI2Batched<V>, Vec<usize>, Vec<f64>)>
where
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
    V: TTScalar
        + Default
        + Clone
        + 'static
        + TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    R: rand::Rng + ?Sized,
{
    let (result, ranks, errors, _sampling) = multicomponent_with_sampling(
        grid,
        f,
        output_dims,
        initial_pivots,
        options,
        rng,
        MULTICOMPONENT_SHARED_MEMO_BYTES,
    )?;
    Ok((result, ranks, errors))
}

/// Result, aggregated ranks, aggregated errors and the shared memo of one run.
type MulticomponentRun<V> = (
    QuanticsTensorCI2Batched<V>,
    Vec<usize>,
    Vec<f64>,
    SharedSampling<V>,
);

/// The multicomponent run with an explicit memo limit, returning the shared
/// memo so tests can read its source-call and byte accounting.
///
/// The public entry point calls this with the default limit and drops the memo.
fn multicomponent_with_sampling<V, F, R>(
    grid: &DiscretizedGrid,
    f: F,
    output_dims: &[usize],
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
    rng: &mut R,
    retained_byte_limit: usize,
) -> QtciResult<MulticomponentRun<V>>
where
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
    V: TTScalar
        + Default
        + Clone
        + 'static
        + TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    R: rand::Rng + ?Sized,
{
    // Validate output_dims
    if output_dims.is_empty() {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "output_dims must not be empty".to_string(),
        });
    }
    let n_components = output_dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| QuanticsTCIError::InvalidConfiguration {
                message: format!("product of output_dims overflowed usize: {output_dims:?}"),
            })
    })?;
    if n_components == 0 {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: format!("product of output_dims must be positive, got 0 from {output_dims:?}"),
        });
    }

    // One memo for the whole call: the callback answers every component of a
    // point at once, so a point any component run evaluated is served to the
    // later runs without calling `f` again. Built before the first callback
    // call, so a grid whose quantics key is too wide is rejected up front.
    let sampling =
        SharedSampling::<V>::new(&grid.local_dimensions(), n_components, retained_byte_limit)?;

    let mut component_tts: Vec<SimpleTensorTrain<V>> = Vec::with_capacity(n_components);
    let mut all_ranks: Vec<Vec<usize>> = Vec::with_capacity(n_components);
    let mut all_errors: Vec<Vec<f64>> = Vec::with_capacity(n_components);

    for component in 0..n_components {
        // One stream serves every component, so the second component continues
        // the caller's sequence instead of restarting it.
        // Erase the caller's RNG type once so every component instantiates the
        // batch pipeline a single time.
        let mut stream: &mut R = rng;
        let (tt, ranks, errors) = sampling.run_component(
            grid,
            &f,
            component,
            initial_pivots.clone(),
            &options,
            &mut stream,
        )?;

        component_tts.push(tt);
        all_ranks.push(ranks);
        all_errors.push(errors);
    }

    // Combine component TTs into a single TT with a component selector site.
    let combined_tt = combine_component_tts(&component_tts)?;

    // Compute aggregate ranks and errors (element-wise max across components).
    let max_len = all_ranks.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut max_bond_dims = vec![0usize; max_len];
    for ranks in &all_ranks {
        for (i, &r) in ranks.iter().enumerate() {
            max_bond_dims[i] = max_bond_dims[i].max(r);
        }
    }

    let max_err_len = all_errors.iter().map(|e| e.len()).max().unwrap_or(0);
    let mut max_errors = vec![0.0f64; max_err_len];
    for errors in &all_errors {
        for (i, &e) in errors.iter().enumerate() {
            max_errors[i] = max_errors[i].max(e);
        }
    }

    let result = QuanticsTensorCI2Batched {
        tt: combined_tt,
        output_dims: output_dims.to_vec(),
        grid: grid.clone(),
    };

    Ok((result, max_bond_dims, max_errors, sampling))
}

/// Interpolate a vector/tensor-valued function, one point at a time.
///
/// Deprecated: `f` is called once per grid point. Use
/// [`quanticscrossinterpolate_multicomponent`] instead, optionally with
/// [`pointwise_components_batch`](crate::pointwise_components_batch) to keep a
/// point-wise closure.
///
/// # Errors
///
/// Returns the same errors as [`quanticscrossinterpolate_multicomponent`].
#[deprecated(
    note = "calls `f` once per point; use `quanticscrossinterpolate_multicomponent` (optionally with `pointwise_components_batch`) instead"
)]
pub fn quanticscrossinterpolate_batched<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    output_dims: &[usize],
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2Batched<V>, Vec<usize>, Vec<f64>)>
where
    F: Fn(&[f64]) -> Vec<V> + 'static,
    V: TTScalar
        + Default
        + Clone
        + 'static
        + TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
{
    quanticscrossinterpolate_multicomponent(
        grid,
        crate::batch::pointwise_components_batch(f),
        output_dims,
        initial_pivots,
        options,
    )
}

/// Combine per-component tensor trains into a single TT with a component
/// selector as the final site.
///
/// The combination strategy ensures SimpleTensorTrain validity:
/// - **First site** (`left_dim = 1` for all components): the component tensors
///
///   are concatenated along the right bond, giving shape `(1, site_dim, sum_right)`.
/// - **Middle sites**: block-diagonal in both bond dimensions.
/// - **Last grid site** (`right_dim = 1` for all components): block-diagonal in
///
///   left bond, concatenated right bond gives `(sum_left, site_dim, n_components)`.
///
/// A final selector site of shape `(total_right_bond, n_components, 1)` is
/// appended so that fixing the component index selects the corresponding
/// component's value.
fn combine_component_tts<V>(component_tts: &[SimpleTensorTrain<V>]) -> Result<SimpleTensorTrain<V>>
where
    V: TTScalar + Default + Clone,
{
    let n_components = component_tts.len();
    if n_components == 0 {
        return Err(anyhow!("no component tensor trains to combine"));
    }

    let n_sites = component_tts[0].len();
    if n_sites == 0 {
        return Err(anyhow!(
            "component tensor trains must have at least one site"
        ));
    }

    // Verify all components have the same number of sites and site dimensions.
    for (c, tt) in component_tts.iter().enumerate() {
        if tt.len() != n_sites {
            return Err(anyhow!(
                "component {} has {} sites, expected {}",
                c,
                tt.len(),
                n_sites
            ));
        }
        for s in 0..n_sites {
            if tt.site_tensor(s).site_dim() != component_tts[0].site_tensor(s).site_dim() {
                return Err(anyhow!(
                    "component {} site {} has site_dim {}, expected {}",
                    c,
                    s,
                    tt.site_tensor(s).site_dim(),
                    component_tts[0].site_tensor(s).site_dim()
                ));
            }
        }
    }

    let mut combined_tensors: Vec<Tensor3<V>> = Vec::with_capacity(n_sites + 1);

    for s in 0..n_sites {
        let site_dim = component_tts[0].site_tensor(s).site_dim();

        let total_right: usize = component_tts
            .iter()
            .map(|tt| tt.site_tensor(s).right_dim())
            .sum();

        if s == 0 {
            // First site: all components have left_dim = 1.
            // Concatenate along right bond: shape (1, site_dim, total_right).
            let mut combined = tensor4all_simplett::tensor3_zeros::<V>(1, site_dim, total_right)?;

            let mut right_offset = 0;
            for tt in component_tts.iter() {
                let t = tt.site_tensor(s);
                let rd = t.right_dim();

                for sd in 0..site_dim {
                    for r in 0..rd {
                        combined.set3(0, sd, right_offset + r, *t.get3(0, sd, r));
                    }
                }

                right_offset += rd;
            }

            combined_tensors.push(combined);
        } else {
            // Middle and last grid sites: block-diagonal in both bond dims.
            let total_left: usize = component_tts
                .iter()
                .map(|tt| tt.site_tensor(s).left_dim())
                .sum();

            let mut combined =
                tensor4all_simplett::tensor3_zeros::<V>(total_left, site_dim, total_right)?;

            let mut left_offset = 0;
            let mut right_offset = 0;
            for tt in component_tts.iter() {
                let t = tt.site_tensor(s);
                let ld = t.left_dim();
                let rd = t.right_dim();

                for l in 0..ld {
                    for sd in 0..site_dim {
                        for r in 0..rd {
                            combined.set3(left_offset + l, sd, right_offset + r, *t.get3(l, sd, r));
                        }
                    }
                }

                left_offset += ld;
                right_offset += rd;
            }

            combined_tensors.push(combined);
        }
    }

    // Build the component selector site.
    // Shape: (total_right_of_last_grid_site, n_components, 1)
    let total_right: usize = component_tts
        .iter()
        .map(|tt| tt.site_tensor(n_sites - 1).right_dim())
        .sum();

    let mut selector = tensor4all_simplett::tensor3_zeros::<V>(total_right, n_components, 1)?;

    let mut offset = 0;
    for (c, tt) in component_tts.iter().enumerate() {
        let rd = tt.site_tensor(n_sites - 1).right_dim();
        for i in 0..rd {
            selector.set3(offset + i, c, 0, V::one());
        }
        offset += rd;
    }

    combined_tensors.push(selector);

    SimpleTensorTrain::new(combined_tensors)
        .map_err(|e| anyhow!("Failed to build combined TT: {}", e))
}

#[cfg(test)]
mod tests;
