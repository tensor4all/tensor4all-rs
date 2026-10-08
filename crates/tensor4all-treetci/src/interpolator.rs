//! TreeTCI implementation of the tree interpolation engine contract.
//!
//! [`TreeTciInterpolator`] implements
//! [`TreeInterpolator`](tensor4all_treetn::interpolation::TreeInterpolator)
//! on top of [`TreeTCI2`] and [`optimize_with_proposer`].

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;

use anyhow::{ensure, Result};
use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, IndexLike,
};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationOutcome, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};
use tensor4all_treetn::TreeTN;

use crate::error::Result as TreeTciResult;
use crate::globalpivot::ScalarParts;
use crate::materialize::to_named_treetn;
use crate::{
    column_2d, ncols_2d, optimize_with_proposer, DefaultProposer, GlobalIndexBatch, MultiIndex,
    TreeTCI2, TreeTciEdge, TreeTciError, TreeTciGraph, TreeTciOptions, TreeTciTermination,
};

/// Tree interpolation engine backed by TreeTCI.
///
/// Implements
/// [`TreeInterpolator`](tensor4all_treetn::interpolation::TreeInterpolator)
/// for the scalar types supported by [`crossinterpolate2`](crate::crossinterpolate2).
/// Use it when code is written against the engine-independent contract in
/// [`tensor4all_treetn::interpolation`]; use [`crossinterpolate2`](crate::crossinterpolate2)
/// for a direct TreeTCI run on a [`TreeTciGraph`].
///
/// Each node of the problem becomes one TreeTCI vertex whose local dimension
/// is the product of the node's active site dimensions (fused column-major in
/// the node's site order), or one for a node without active sites. The
/// optimization uses [`DefaultProposer`] and the engine's
/// [`TreeTciOptions`], except that the problem supplies
///
/// - `tolerance` (set to the problem's absolute tolerance),
/// - `max_bond_dim` (the problem's cap),
/// - `normalize_error` (always `false`: the tolerance is absolute), and
/// - `seed` (the problem's seed, the only source of randomness).
///
/// The remaining options (`max_iter`, the global pivot search settings, and
/// `evaluation_cache_bytes`) come from the engine. The optional target memo
/// covers optimization; named network materialization calls the evaluator
/// separately. A single-node problem is evaluated exactly on its full
/// index set, which costs the product of its site dimensions.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::interpolation::{
///     InterpolationProblem, InterpolationTermination, TreeInterpolator,
/// };
/// use tensor4all_treetn::NodeNameNetwork;
///
/// // Node "x" carries two sites (fused into one vertex), node "y" one site.
/// let (x0, x1, y) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(3));
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node("x".to_string())?;
/// topology.add_node("y".to_string())?;
/// topology.add_edge(&"x".to_string(), &"y".to_string())?;
/// let node_sites = BTreeMap::from([
///     ("x".to_string(), vec![x0.clone(), x1.clone()]),
///     ("y".to_string(), vec![y.clone()]),
/// ]);
/// let pivots = ColMajorArray::new(vec![0, 0, 0], vec![3, 1])?;
/// let problem = InterpolationProblem::new(topology, node_sites, pivots, 1e-12, None, 1)?;
///
/// // f(x0, x1, y) = (1 + x0 + 2 x1) * (1 + y), rank one across the edge.
/// let f = |p: &[usize]| ((1 + p[0] + 2 * p[1]) * (1 + p[2])) as f64;
/// let evaluate = |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
///     Ok(batch.data().chunks(batch.shape()[0]).map(f).collect())
/// };
/// let outcome = TreeTciInterpolator::default().interpolate(&problem, evaluate)?;
///
/// assert_eq!(outcome.termination, InterpolationTermination::Converged);
/// assert!(outcome.error_estimate <= 1e-12);
/// assert_eq!(outcome.max_sample_magnitude, 12.0);
/// let value = outcome.network.evaluate_point(&[x0, x1, y], &[1, 1, 2])?;
/// assert!((value.real() - 12.0).abs() < 1e-10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct TreeTciInterpolator {
    options: TreeTciOptions,
}

impl TreeTciInterpolator {
    /// Minimum `max_iter`: convergence needs three sweeps of history.
    pub const MIN_MAX_ITER: usize = 3;

    /// Build an engine from TreeTCI options.
    ///
    /// `tolerance`, `max_bond_dim`, `normalize_error`, and `seed` are replaced
    /// by the problem on every run (see the type documentation); the other
    /// fields configure the run.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTciError::InvalidConfiguration`] when the options fail
    /// the checks of [`optimize_with_proposer`] or when `max_iter` is below
    /// [`Self::MIN_MAX_ITER`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTciError, TreeTciInterpolator, TreeTciOptions};
    ///
    /// let engine = TreeTciInterpolator::new(TreeTciOptions { max_iter: 8, ..Default::default() })?;
    /// assert_eq!(engine.options().max_iter, 8);
    ///
    /// let error = TreeTciInterpolator::new(TreeTciOptions { max_iter: 2, ..Default::default() })
    ///     .unwrap_err();
    /// assert!(matches!(error, TreeTciError::InvalidConfiguration { .. }));
    /// # Ok::<(), TreeTciError>(())
    /// ```
    pub fn new(options: TreeTciOptions) -> TreeTciResult<Self> {
        options.validate()?;
        if options.max_iter < Self::MIN_MAX_ITER {
            return Err(TreeTciError::InvalidConfiguration {
                message: format!(
                    "max_iter must be at least {} for the interpolation engine, got {}",
                    Self::MIN_MAX_ITER,
                    options.max_iter
                ),
            });
        }
        Ok(Self { options })
    }

    /// Borrow the engine's options as given to [`Self::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTciInterpolator, TreeTciOptions};
    ///
    /// let engine = TreeTciInterpolator::default();
    /// assert_eq!(engine.options().max_iter, TreeTciOptions::default().max_iter);
    /// ```
    pub fn options(&self) -> &TreeTciOptions {
        &self.options
    }

    /// The options of one run: the engine's options with the problem's
    /// tolerance, cap, and seed, and `normalize_error` disabled.
    fn run_options(
        &self,
        absolute_tolerance: f64,
        cap: Option<NonZeroUsize>,
        seed: u64,
    ) -> TreeTciOptions {
        TreeTciOptions {
            tolerance: absolute_tolerance,
            max_bond_dim: cap.map(NonZeroUsize::get),
            normalize_error: false,
            seed: Some(seed),
            ..self.options.clone()
        }
    }
}

impl Default for TreeTciInterpolator {
    /// An engine with [`TreeTciOptions::default`], which satisfies the
    /// `max_iter` requirement.
    fn default() -> Self {
        Self {
            options: TreeTciOptions::default(),
        }
    }
}

impl<T> TreeInterpolator<T> for TreeTciInterpolator
where
    T: FullPivLuScalar
        + CommonScalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_core::TensorElement
        + ScalarParts,
{
    /// Run TreeTCI on `problem`.
    ///
    /// # Errors
    ///
    /// - [`InterpolationError::InvalidProblem`] when the product of a node's
    ///   site dimensions (or the full index set of a single-node problem)
    ///   overflows `usize`, or a single-node point list cannot be reserved.
    /// - [`InterpolationError::Evaluator`] when `evaluate` returns an error or
    ///   a number of values other than the number of points, at any stage, or
    ///   a non-finite scalar component or magnitude in any batch.
    /// - [`InterpolationError::AllSamplesZero`] when every initial pivot
    ///   evaluates to exactly zero.
    /// - [`InterpolationError::Engine`] when TreeTCI fails for any other
    ///   reason (graph construction, pivot bookkeeping, a singular solve, or
    ///   materialization).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
    /// use tensor4all_treetci::TreeTciInterpolator;
    /// use tensor4all_treetn::interpolation::{
    ///     InterpolationError, InterpolationProblem, TreeInterpolator,
    /// };
    /// use tensor4all_treetn::NodeNameNetwork;
    ///
    /// let mut topology = NodeNameNetwork::new();
    /// topology.add_node(0usize)?;
    /// topology.add_node(1usize)?;
    /// topology.add_edge(&0, &1)?;
    /// let node_sites = BTreeMap::from([
    ///     (0usize, vec![DynIndex::new_dyn(2)]),
    ///     (1, vec![DynIndex::new_dyn(2)]),
    /// ]);
    /// let pivots = ColMajorArray::new(vec![0, 0], vec![2, 1])?;
    /// let problem = InterpolationProblem::new(topology, node_sites, pivots, 1e-10, None, 0)?;
    ///
    /// let failing = |_: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
    ///     anyhow::bail!("out of domain")
    /// };
    /// let error = TreeTciInterpolator::default().interpolate(&problem, failing).unwrap_err();
    /// assert!(matches!(error, InterpolationError::Evaluator { .. }));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
    {
        let layout = VertexLayout::new(problem)?;

        // Zero-patch rule, applied before any topology-specific path.
        let initial = problem.initial_pivots();
        let initial_values =
            call_evaluator(&evaluate, initial.data(), layout.n_sites()).map_err(classify_anyhow)?;
        let max_initial = initial_sample_scale(&initial_values)?;

        if layout.vertex_count() == 1 {
            return interpolate_single_node(&layout, &evaluate);
        }

        let options = self.run_options(
            problem.absolute_tolerance(),
            problem.max_bond_dim(),
            problem.seed(),
        );
        let vertex_evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
            let sites = layout.vertex_batch_to_sites(batch)?;
            call_evaluator(&evaluate, &sites, layout.n_sites())
        };

        let graph = TreeTciGraph::new(layout.vertex_count(), &layout.edges).map_err(engine)?;
        let mut state = TreeTCI2::<T>::new(layout.local_dims.clone(), graph).map_err(engine)?;
        let vertex_pivots = layout
            .site_columns_to_vertices(initial)
            .map_err(|source| InterpolationError::Engine { source })?;
        state.add_global_pivots(&vertex_pivots).map_err(engine)?;
        state.max_sample_value = max_initial;

        let report =
            optimize_with_proposer(&mut state, vertex_evaluate, &options, &DefaultProposer)
                .map_err(classify)?;
        let network = to_named_treetn(
            &state,
            vertex_evaluate,
            None,
            &layout.node_names,
            &layout.vertex_sites,
        )
        .map_err(classify)?;
        let pivots = joined_pivots(&state, &layout)
            .map_err(|source| InterpolationError::Engine { source })?;

        let (Some(&final_rank), Some(&error_estimate)) =
            (report.ranks.last(), report.errors.last())
        else {
            return Err(InterpolationError::Engine {
                source: anyhow::anyhow!("TreeTCI finished without running a sweep"),
            });
        };
        Ok(InterpolationOutcome {
            network,
            termination: map_termination(report.termination, final_rank, problem.max_bond_dim()),
            error_estimate,
            max_sample_magnitude: state.max_sample_value,
            pivots: Some(pivots),
        })
    }
}

/// Map a TreeTCI stop reason to the contract's verdict. `Converged` requires
/// the final maximum bond dimension to be strictly below the cap; a criterion
/// met at the cap is `BondCapReached`.
fn map_termination(
    reason: TreeTciTermination,
    final_rank: usize,
    cap: Option<NonZeroUsize>,
) -> InterpolationTermination {
    match reason {
        // INVARIANT: `optimize_with_proposer` checks the saturation stop before
        // the convergence criterion, so a `Converged` run with a cap ends
        // strictly below it (see `TreeTciTermination::Converged`). This guard
        // is defence in depth: it keeps the contract's precedence even if the
        // loop order changes.
        TreeTciTermination::Converged if cap.is_some_and(|cap| final_rank >= cap.get()) => {
            InterpolationTermination::BondCapReached
        }
        TreeTciTermination::Converged => InterpolationTermination::Converged,
        TreeTciTermination::MaxBondDimension => InterpolationTermination::BondCapReached,
        TreeTciTermination::MaxIterations => InterpolationTermination::IterationLimit,
    }
}

/// Private marker wrapping every failure attributable to the caller's
/// evaluator, so it can be told apart from TreeTCI's own failures after
/// TreeTCI has wrapped it into [`TreeTciError`].
#[derive(Debug, thiserror::Error)]
#[error("batch evaluator failed: {source}")]
struct EvaluatorFailure {
    #[source]
    source: anyhow::Error,
}

/// Call the caller's evaluator on a column-major `[n_sites, n_points]` batch
/// and check that it returns one finite value with finite magnitude per
/// point. Evaluator failures are wrapped in [`EvaluatorFailure`].
fn call_evaluator<T, F>(evaluate: &F, data: &[usize], n_sites: usize) -> Result<Vec<T>>
where
    T: CommonScalar,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
{
    // INVARIANT: `n_sites >= 1` because `InterpolationProblem::new` requires
    // at least one active site, so the division is defined.
    let n_points = data.len() / n_sites;
    let shape = [n_sites, n_points];
    let batch = ColMajorArrayRef::new(data, &shape)?;
    let values = evaluate(batch).map_err(|source| EvaluatorFailure { source })?;
    if values.len() != n_points {
        return Err(EvaluatorFailure {
            source: anyhow::anyhow!(
                "evaluator returned {} values for {n_points} points",
                values.len()
            ),
        }
        .into());
    }
    if let Some(point) = values.iter().position(|&value| {
        // Multiplication by zero detects non-finite components even when a
        // scalar's magnitude implementation masks a NaN component.
        (value * T::from_f64(0.0)).abs_val() != 0.0 || !value.abs_val().is_finite()
    }) {
        return Err(EvaluatorFailure {
            source: anyhow::anyhow!(
                "evaluator returned a non-finite value or magnitude at batch point {point}, \
                 coordinates {:?}",
                &data[point * n_sites..(point + 1) * n_sites]
            ),
        }
        .into());
    }
    Ok(values)
}

fn has_evaluator_failure(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if error.downcast_ref::<EvaluatorFailure>().is_some() {
            return true;
        }
        current = error.source();
    }
    false
}

/// Classify a TreeTCI failure: `Evaluator` when the error chain contains the
/// evaluator marker, `Engine` otherwise.
fn classify(error: TreeTciError) -> InterpolationError {
    if !has_evaluator_failure(&error) {
        return engine(error);
    }
    match error {
        TreeTciError::Operation { source } => classify_anyhow(source),
        other => InterpolationError::Evaluator {
            source: anyhow::Error::new(other),
        },
    }
}

/// Classify an `anyhow` failure like [`classify`]. When the marker is the
/// error itself (possibly under context), the caller's original error becomes
/// the `Evaluator` source.
fn classify_anyhow(error: anyhow::Error) -> InterpolationError {
    if !has_evaluator_failure(error.as_ref()) {
        return InterpolationError::Engine { source: error };
    }
    match error.downcast::<EvaluatorFailure>() {
        Ok(failure) => InterpolationError::Evaluator {
            source: failure.source,
        },
        Err(error) => InterpolationError::Evaluator { source: error },
    }
}

fn engine(error: TreeTciError) -> InterpolationError {
    InterpolationError::Engine {
        source: anyhow::Error::new(error),
    }
}

/// Apply the zero-patch rule to the initial samples and return their largest
/// magnitude. Values have been checked by `call_evaluator`;
/// `AllSamplesZero` requires every sample to be exactly zero.
fn initial_sample_scale<T: CommonScalar>(values: &[T]) -> Result<f64, InterpolationError> {
    let magnitudes: Vec<f64> = values
        .iter()
        .map(|value| CommonScalar::abs_val(*value))
        .collect();
    if magnitudes.iter().all(|&magnitude| magnitude == 0.0) {
        return Err(InterpolationError::AllSamplesZero);
    }
    Ok(magnitudes.into_iter().fold(0.0_f64, f64::max))
}

fn max_magnitude<T: CommonScalar>(values: &[T]) -> f64 {
    values
        .iter()
        .map(|value| CommonScalar::abs_val(*value))
        .fold(0.0_f64, f64::max)
}

/// Mapping between the problem's site order and TreeTCI vertex coordinates.
///
/// Vertex `v` is the `v`-th node in ascending name order. Its sites occupy the
/// contiguous rows `site_offsets[v]..site_offsets[v + 1]` of the site order,
/// and its coordinate is `s_0 + d_0 * (s_1 + d_1 * (...))` over those sites.
struct VertexLayout<V> {
    node_names: Vec<V>,
    vertex_sites: Vec<Vec<DynIndex>>,
    site_offsets: Vec<usize>,
    site_dims: Vec<usize>,
    local_dims: Vec<usize>,
    edges: Vec<TreeTciEdge>,
    /// The vertex of every site row when every vertex has at most one site
    /// (`None` when some vertex fuses several sites). The rows then follow
    /// the vertices in order, and a site coordinate equals the coordinate of
    /// its vertex, whose local dimension is the site dimension.
    site_vertices: Option<Vec<usize>>,
}

impl<V> VertexLayout<V>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    fn new(problem: &InterpolationProblem<V>) -> Result<Self, InterpolationError> {
        let node_names: Vec<V> = problem.node_sites().keys().cloned().collect();
        let vertex_sites: Vec<Vec<DynIndex>> = problem.node_sites().values().cloned().collect();
        let site_dims: Vec<usize> = problem.site_order().iter().map(|site| site.dim()).collect();

        let mut site_offsets = Vec::with_capacity(vertex_sites.len() + 1);
        let mut local_dims = Vec::with_capacity(vertex_sites.len());
        let mut offset = 0usize;
        for (name, sites) in node_names.iter().zip(&vertex_sites) {
            site_offsets.push(offset);
            // INVARIANT: the node site lists partition the site order held in
            // memory, so the running offset cannot overflow.
            offset += sites.len();
            let local_dim = sites
                .iter()
                .try_fold(1usize, |product, site| product.checked_mul(site.dim()))
                .ok_or_else(|| InterpolationError::InvalidProblem {
                    message: format!(
                        "the product of the site dimensions of node {name:?} overflows usize"
                    ),
                })?;
            local_dims.push(local_dim);
        }
        site_offsets.push(offset);

        let topology = problem.topology();
        let vertex_of = |node| -> Result<usize, InterpolationError> {
            let name = topology
                .node_name(node)
                .ok_or_else(|| InterpolationError::Engine {
                    source: anyhow::anyhow!("topology edge refers to an unknown node"),
                })?;
            node_names
                .binary_search(name)
                .map_err(|_| InterpolationError::Engine {
                    source: anyhow::anyhow!("topology node {name:?} has no vertex"),
                })
        };
        let graph = topology.graph();
        let mut edges = Vec::with_capacity(graph.edge_count());
        for edge in graph.edge_indices() {
            let (a, b) = graph
                .edge_endpoints(edge)
                .ok_or_else(|| InterpolationError::Engine {
                    source: anyhow::anyhow!("topology edge {edge:?} has no endpoints"),
                })?;
            edges.push(TreeTciEdge::new(vertex_of(a)?, vertex_of(b)?));
        }

        let site_vertices = vertex_sites.iter().all(|sites| sites.len() <= 1).then(|| {
            (0..vertex_sites.len())
                .filter(|&vertex| site_offsets[vertex] < site_offsets[vertex + 1])
                .collect()
        });

        Ok(Self {
            node_names,
            vertex_sites,
            site_offsets,
            site_dims,
            local_dims,
            edges,
            site_vertices,
        })
    }

    fn vertex_count(&self) -> usize {
        self.node_names.len()
    }

    fn n_sites(&self) -> usize {
        self.site_dims.len()
    }

    /// Split each vertex coordinate of a TreeTCI batch into site-order rows,
    /// as a `[n_sites, n_points]` batch.
    ///
    /// Every vertex coordinate is checked against its local dimension. When
    /// every vertex has at most one site (`site_vertices`), the site rows are
    /// the vertex rows without the site-free vertices: the batch itself is
    /// returned without a copy when no vertex is site-free, and otherwise its
    /// site rows are gathered without the column-major split. A vertex that
    /// fuses several sites takes the general path through
    /// [`Self::split_point`].
    fn vertex_batch_to_sites<'b>(&self, batch: GlobalIndexBatch<'b>) -> Result<Cow<'b, [usize]>> {
        let n_vertices = self.vertex_count();
        ensure!(
            batch.n_sites() == n_vertices,
            "TreeTCI batch has {} vertices, expected {}",
            batch.n_sites(),
            n_vertices
        );
        let len = self
            .n_sites()
            .checked_mul(batch.n_points())
            .ok_or_else(|| anyhow::anyhow!("site batch size overflowed usize"))?;
        let Some(site_vertices) = &self.site_vertices else {
            let mut sites = vec![0usize; len];
            for (vertices, out) in batch
                .data()
                .chunks_exact(n_vertices)
                .zip(sites.chunks_exact_mut(self.n_sites()))
            {
                self.split_point(vertices, out)?;
            }
            return Ok(Cow::Owned(sites));
        };
        if site_vertices.len() == n_vertices {
            for vertices in batch.data().chunks_exact(n_vertices) {
                self.check_vertex_point(vertices)?;
            }
            return Ok(Cow::Borrowed(batch.data()));
        }
        let mut sites = Vec::with_capacity(len);
        for vertices in batch.data().chunks_exact(n_vertices) {
            self.check_vertex_point(vertices)?;
            sites.extend(site_vertices.iter().map(|&vertex| vertices[vertex]));
        }
        Ok(Cow::Owned(sites))
    }

    /// Check every coordinate of one vertex point against its local
    /// dimension.
    fn check_vertex_point(&self, vertices: &[usize]) -> Result<()> {
        for (vertex, (&coordinate, &local_dim)) in vertices.iter().zip(&self.local_dims).enumerate()
        {
            ensure!(
                coordinate < local_dim,
                "vertex {vertex} coordinate {coordinate} is out of range for local dimension \
                 {local_dim}"
            );
        }
        Ok(())
    }

    /// Write the site-order coordinates of one vertex point into `out`.
    fn split_point(&self, vertices: &[usize], out: &mut [usize]) -> Result<()> {
        ensure!(
            vertices.len() == self.vertex_count() && out.len() == self.n_sites(),
            "point has {} vertices and {} site slots, expected {} and {}",
            vertices.len(),
            out.len(),
            self.vertex_count(),
            self.n_sites()
        );
        self.check_vertex_point(vertices)?;
        for (vertex, &coordinate) in vertices.iter().enumerate() {
            let mut rest = coordinate;
            let rows = self.site_offsets[vertex]..self.site_offsets[vertex + 1];
            for (slot, &dim) in out[rows.clone()].iter_mut().zip(&self.site_dims[rows]) {
                *slot = rest % dim;
                rest /= dim;
            }
        }
        Ok(())
    }

    /// Fuse each site-order column into one vertex point.
    fn site_columns_to_vertices(&self, columns: &ColMajorArray<usize>) -> Result<Vec<MultiIndex>> {
        let n_columns = ncols_2d(columns)?;
        let mut points = Vec::with_capacity(n_columns);
        for column in 0..n_columns {
            let sites = column_2d(columns, column)?;
            ensure!(
                sites.len() == self.n_sites(),
                "pivot column has {} rows, expected {}",
                sites.len(),
                self.n_sites()
            );
            let point = (0..self.vertex_count())
                .map(|vertex| {
                    let rows = self.site_offsets[vertex]..self.site_offsets[vertex + 1];
                    // INVARIANT: every site coordinate is below its dimension
                    // (checked by `InterpolationProblem::new`), so the fused
                    // value stays below the vertex's local dimension, whose
                    // product was checked in `VertexLayout::new`.
                    rows.rev().fold(0usize, |fused, row| {
                        fused * self.site_dims[row] + sites[row]
                    })
                })
                .collect();
            points.push(point);
        }
        Ok(points)
    }
}

/// Join the two pivot sets of every edge column by column into full-domain
/// points, convert them to site order, and deduplicate them in first-seen
/// order.
fn joined_pivots<T, V>(
    state: &TreeTCI2<T>,
    layout: &VertexLayout<V>,
) -> Result<ColMajorArray<usize>>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let mut seen = HashSet::new();
    let mut data = Vec::new();
    let mut n_pivots = 0usize;
    let mut vertices = vec![0usize; layout.vertex_count()];
    for edge in state.graph.edges() {
        let (left_key, right_key) = state.graph.subregion_vertices(edge)?;
        let left = state
            .ijset
            .get(&left_key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {left_key:?}"))?;
        let right = state
            .ijset
            .get(&right_key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {right_key:?}"))?;
        let (n_left, n_right) = (ncols_2d(left)?, ncols_2d(right)?);
        ensure!(
            n_left == n_right,
            "pivot counts disagree across edge {edge:?}: left {n_left}, right {n_right}"
        );
        for column in 0..n_left {
            for (key, set) in [(&left_key, left), (&right_key, right)] {
                for (&vertex, &value) in key.as_slice().iter().zip(column_2d(set, column)?) {
                    vertices[vertex] = value;
                }
            }
            let mut sites = vec![0usize; layout.n_sites()];
            layout.split_point(&vertices, &mut sites)?;
            if seen.insert(sites.clone()) {
                data.extend_from_slice(&sites);
                n_pivots += 1;
            }
        }
    }
    Ok(ColMajorArray::new(data, vec![layout.n_sites(), n_pivots])?)
}

/// Exact path for a single-node problem: evaluate every point of the node's
/// sites and store the values as the node's tensor.
fn interpolate_single_node<T, V, F>(
    layout: &VertexLayout<V>,
    evaluate: &F,
) -> Result<InterpolationOutcome<V>, InterpolationError>
where
    T: tensor4all_core::TensorElement + CommonScalar,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
{
    let n_sites = layout.n_sites();
    let n_points = layout.local_dims[0];
    let len = n_points
        .checked_mul(n_sites)
        .ok_or_else(|| InterpolationError::InvalidProblem {
            message: format!(
                "the full index set of the single node ({n_points} points of {n_sites} sites) \
                 overflows usize"
            ),
        })?;
    let mut points = Vec::new();
    points
        .try_reserve_exact(len)
        .map_err(|error| InterpolationError::InvalidProblem {
            message: format!("could not reserve the single-node point list: {error}"),
        })?;
    let mut out = vec![0usize; n_sites];
    for point in 0..n_points {
        layout
            .split_point(&[point], &mut out)
            .map_err(|source| InterpolationError::Engine { source })?;
        points.extend_from_slice(&out);
    }
    let values = call_evaluator(evaluate, &points, n_sites).map_err(classify_anyhow)?;
    let max_sample_magnitude = max_magnitude(&values);
    let tensor =
        IdxTensor::from_dense(layout.vertex_sites[0].clone(), values).map_err(|error| {
            InterpolationError::Engine {
                source: error.into(),
            }
        })?;
    let network = TreeTN::from_tensors(vec![tensor], vec![layout.node_names[0].clone()]).map_err(
        |error| InterpolationError::Engine {
            source: error.into(),
        },
    )?;
    Ok(InterpolationOutcome {
        network,
        termination: InterpolationTermination::Converged,
        error_estimate: 0.0,
        max_sample_magnitude,
        pivots: None,
    })
}

#[cfg(test)]
mod tests;
