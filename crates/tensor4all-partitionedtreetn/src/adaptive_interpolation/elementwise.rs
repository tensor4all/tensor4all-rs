//! Adaptive patched element-wise operations (milestone M6c).
//!
//! [`patched_elementwise`] computes `g = f(x_1, ..., x_k)` for partitions
//! `x_i` on one site space through the patch driver of
//! [`patched_interpolate`](super::patched_interpolate): the same waves,
//! measurement, acceptance, splitting, and report, under the same L2
//! contract, with the target `f(x_1(x), ..., x_k(x))` evaluated pointwise
//! from the input networks. What differs is the engine: an
//! [`ElementwiseInterpolator`] receives the input networks restricted to the
//! patch instead of sampling a black box. The design is recorded in
//! `docs/design/tree-patching-elementwise.md`.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Mutex;

use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, TensorElement,
};
use tensor4all_treetn::interpolation::{
    ElementwiseInterpolator, ElementwiseOperator, EvalContext, InterpolationError,
    InterpolationOutcome, InterpolationProblem, TreeInterpolator,
};
use tensor4all_treetn::{
    CachedEvaluatorOptions, EvaluationHint, NodeNameNetwork, TreeTN, TreeTNCachedEvaluator,
};

use super::embed::{restrict_fixed_sites, zeros_like};
use super::engine::PatchEngine;
use super::fan_out::run_on_lanes;
use super::layout::SiteLayout;
use super::{
    invalid, run_serial, PatchedInterpolationError, PatchedInterpolationOptions,
    PatchedInterpolationResult, Plan,
};
use crate::partitioned_tree_tn::PatchLocator;
use crate::subdomain_tree_tn::{ensure_same_dtype, ensure_same_tree_structure};
use crate::{PartitionedTreeTN, Projector, SubDomainTreeTN};
use hataori_tenferro::TenferroLanes;

/// Where a region lies against one input's patches.
enum Region<'a, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Inside this patch.
    Inside(&'a SubDomainTreeTN<V>),
    /// Outside every patch: the input is zero on the region.
    Outside,
    /// Across several patches, or partly inside one.
    Partial,
}

/// One input partition with its patches in canonical order.
struct Input<'a, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    patches: Vec<&'a SubDomainTreeTN<V>>,
}

impl<'a, V> Input<'a, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    fn region(&self, region: &Projector) -> Region<'a, V> {
        let mut overlapping = self
            .patches
            .iter()
            .filter(|patch| patch.projector().is_compatible_with(region));
        match (overlapping.next(), overlapping.next()) {
            (None, _) => Region::Outside,
            (Some(patch), None) if region.is_subset_of(patch.projector()) => Region::Inside(patch),
            _ => Region::Partial,
        }
    }
}

/// The element-wise engine of one run: the kernel, the operator, the inputs,
/// and the black-box engine of the residual estimator.
pub(super) struct Elementwise<'a, E, Op, R, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    engine: &'a E,
    operator: &'a Op,
    residual: &'a R,
    inputs: Vec<Input<'a, V>>,
}

impl<T, V, E, Op, R> PatchEngine<T, V> for Elementwise<'_, E, Op, R, V>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: ElementwiseInterpolator<T, Op>,
    R: TreeInterpolator<T>,
{
    type Residual = R;

    fn residual_engine(&self) -> &R {
        self.residual
    }

    fn runs_on(&self, projector: &Projector) -> Result<bool, PatchedInterpolationError> {
        Ok(self
            .inputs
            .iter()
            .all(|input| !matches!(input.region(projector), Region::Partial)))
    }

    fn run<F>(
        &self,
        problem: &InterpolationProblem<V>,
        layout: &SiteLayout<V>,
        fixed: &[Option<usize>],
        projector: &Projector,
        ctx: &mut EvalContext<'_>,
        _evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>>,
    {
        let engine_error = |source: crate::PartitionedTreeTNError| InterpolationError::Engine {
            source: anyhow::Error::new(source),
        };
        // An input outside the region restricts to an exact zero through any of
        // its patches (the mask zeroes the selected coordinates); an empty
        // input takes the zero network of the first restricted one.
        let mut restricted: Vec<Option<TreeTN<IdxTensor, V>>> =
            Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            let source = match input.region(projector) {
                Region::Inside(patch) => Some(patch),
                Region::Outside => input.patches.first().copied(),
                Region::Partial => {
                    return Err(InterpolationError::InvalidProblem {
                        message: format!(
                            "the region {projector:?} spans several patches of an input"
                        ),
                    })
                }
            };
            restricted.push(
                source
                    .map(|patch| {
                        restrict_fixed_sites::<T, V>(patch.data(), layout, fixed, ctx.session())
                    })
                    .transpose()
                    .map_err(engine_error)?,
            );
        }
        let template = restricted.iter().flatten().next().cloned().ok_or_else(|| {
            InterpolationError::InvalidProblem {
                message: "every input of the element-wise operation is empty".to_string(),
            }
        })?;
        let inputs = restricted
            .into_iter()
            .map(|network| match network {
                Some(network) => Ok(network),
                None => zeros_like::<T, V>(&template),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(engine_error)?;
        self.engine
            .interpolate_elementwise(problem, &inputs, self.operator, ctx)
    }
}

/// Idle cached evaluators per (input, patch).
type EvaluatorPool<'a, V> = HashMap<(usize, usize), Vec<TreeTNCachedEvaluator<'a, V>>>;

/// The target `f(x_1, ..., x_k)` at full-domain points, evaluated from the
/// input patches on the run's session.
pub(super) struct Target<'a, T, V, Op>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    inputs: Vec<PatchLocator<'a, V>>,
    sites: Vec<DynIndex>,
    operator: &'a Op,
    /// Idle evaluators per (input, patch), built on first use. An evaluator
    /// is taken out while it runs and put back afterwards, so lanes never
    /// share one and the lock is held only to take or return it.
    evaluators: Mutex<EvaluatorPool<'a, V>>,
    scalar: std::marker::PhantomData<T>,
}

impl<'a, T, V, Op> Target<'a, T, V, Op>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    Op: ElementwiseOperator<T>,
{
    fn pool(&self) -> anyhow::Result<std::sync::MutexGuard<'_, EvaluatorPool<'a, V>>> {
        self.evaluators
            .lock()
            .map_err(|_| anyhow::anyhow!("an evaluator pool lock was poisoned by a panic"))
    }

    pub(super) fn evaluate(
        &self,
        batch: ColMajorArrayRef<'_, usize>,
        ctx: &mut EvalContext<'_>,
    ) -> anyhow::Result<Vec<T>> {
        let &[rows, n_points] = batch.shape() else {
            anyhow::bail!("an element-wise batch must be two-dimensional");
        };
        let k = self.inputs.len();
        let data = batch.data();
        let zero = T::from_f64(0.0);
        let mut columns = vec![zero; k * n_points];
        let mut owned: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut points = Vec::new();
        for (input_id, locator) in self.inputs.iter().enumerate() {
            owned.clear();
            for point in 0..n_points {
                if let Some(patch) = locator.owner(&data[point * rows..(point + 1) * rows]) {
                    owned.entry(patch).or_default().push(point);
                }
            }
            for (&patch_id, members) in &owned {
                points.clear();
                for &point in members {
                    points.extend_from_slice(&data[point * rows..(point + 1) * rows]);
                }
                let key = (input_id, patch_id);
                let idle = self.pool()?.get_mut(&key).and_then(Vec::pop);
                let mut evaluator = match idle {
                    Some(evaluator) => evaluator,
                    None => {
                        let network = locator.patch(patch_id).data();
                        let options = CachedEvaluatorOptions {
                            center: network.node_names().into_iter().min(),
                            ..CachedEvaluatorOptions::default()
                        };
                        TreeTNCachedEvaluator::new(network, &self.sites, options)?
                    }
                };
                let shape = [rows, members.len()];
                let values = evaluator.evaluate_batched_typed_on_session::<T>(
                    ctx.session(),
                    ColMajorArrayRef::new(&points, &shape)?,
                    EvaluationHint::default(),
                )?;
                self.pool()?.entry(key).or_default().push(evaluator);
                for (&point, value) in members.iter().zip(values) {
                    columns[input_id + k * point] = value;
                }
            }
        }
        let shape = [k, n_points];
        let mut output = vec![zero; n_points];
        self.operator
            .apply(ColMajorArrayRef::new(&columns, &shape)?, &mut output)?;
        Ok(output)
    }
}

/// Adaptively compute `f(x_1, ..., x_k)` of partitions on one site space as a
/// partition, under the L2 contract of
/// [`patched_interpolate`](super::patched_interpolate).
///
/// The run is the patch driver of `patched_interpolate` with an element-wise
/// engine. Patches are processed in waves from the whole domain:
///
/// 1. A patch that does not lie inside one patch of every input (or outside
///    all of an input's patches) is split at the next site of
///    [`PatchedInterpolationOptions::patch_order`] without any evaluation
///    (a *structural* split), so the waves descend until every input is a
///    single network, or zero, on each patch. A patch with at most one active
///    site is evaluated exactly instead, whatever the inputs look like there.
/// 2. Otherwise the inputs are restricted to the patch's active sites and
///    `engine` interpolates `operator` applied to them. An input without a
///    patch on the region enters as an exact zero network, so a map with
///    `f(0, ...) != 0` (for example `1 / (a - x)`) is represented there.
/// 3. The result is measured against `f(x_1(x), ..., x_k(x))`, evaluated
///    pointwise from the input patches, and accepted, rerun, split, or
///    retained exactly as in `patched_interpolate` (a patch whose engine rank
///    reaches the cap splits, and so does one that fails its measurement).
///    A region where every sampled value is zero, such as a product where an
///    input has no patch, is measured as a zero patch without an engine run.
///
/// The error contract concerns `f` of the given input networks: how the
/// inputs' own errors propagate through `f` is not part of it.
///
/// Without an element-wise engine, the same contract is met by
/// `patched_interpolate` with a black-box evaluator that evaluates every
/// input with [`PartitionedTreeTN::evaluate_batched`] and applies `f`. That
/// route uses none of the inputs' structure, so it costs one pointwise input
/// evaluation per sample, but it needs no kernel; it is the reference for the
/// kernels.
///
/// # Arguments
///
/// * `engine` - The element-wise kernel, for example
///   [`PointwiseElementwise`](tensor4all_treetn::interpolation::PointwiseElementwise)
///   around a black-box engine, or a dedicated kernel such as
///   `tensor4all_treeaci::TreeAciInterpolator` (tree ACI, any `f`).
/// * `residual_engine` - The black-box engine of the
///   [`L2Estimator::ResidualTt`](super::L2Estimator::ResidualTt) estimator,
///   which interpolates a measured residual; for example
///   `tensor4all_treetci::TreeTciInterpolator`. Unused under
///   [`L2Estimator::UniformMonteCarlo`](super::L2Estimator::UniformMonteCarlo).
/// * `inputs` - At least one partition. Every patch of every input must have
///   the same topology, the same site indices (identity and dimension), and
///   the same dtype, which is `T`; rename equivalent sites with
///   [`TreeTN::replace_indices`](tensor4all_treetn::TreeTN::replace_indices)
///   beforehand. An empty partition is the zero function. At least one input
///   must have a patch.
/// * `operator` - The pointwise map; its batch rows follow `inputs`.
/// * `ctx` - The session every stage runs on.
/// * `options` - As for `patched_interpolate`. Under `ErrorNorm::L2` the
///   reference norm is that of the result `f(x_1, ..., x_k)`. A non-empty
///   `patch_order` must list every site that an input projector fixes. List
///   those sites first: the structural splits descend through `patch_order`
///   in order, so every site listed before them is split as well, and an
///   empty `patch_order` (every site in the derived order) can split much
///   deeper than the inputs require.
///
/// # Returns
///
/// The partition and the report of `patched_interpolate`.
///
/// # Errors
///
/// Returns [`PatchedInterpolationError::InvalidInput`] for an empty input
/// list, inputs without any patch, or a `patch_order` that omits a site an
/// input projector fixes; [`PatchedInterpolationError::Partition`] for inputs
/// whose patches differ in topology, sites, or dtype; and every error of
/// `patched_interpolate`, an operator failure being an evaluator error.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_elementwise, PatchedInterpolationOptions,
/// };
/// use tensor4all_partitionedtreetn::{
///     ErrorNorm, ErrorTolerance, L2Reference, PartitionedTreeTN, Projector, SubDomainTreeTN,
/// };
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::interpolation::{EvalContext, Parallelism, PointwiseElementwise, Product};
/// use tensor4all_treetn::TreeTN;
///
/// // x(a, b) = 1 + a + b on two nodes, stored as two patches split on a;
/// // y(a, b) = 2 stored as one patch on a = 1 only.
/// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
/// let tree = |values: Vec<f64>| -> Result<TreeTN<IdxTensor, usize>, Box<dyn std::error::Error>> {
///     let bond = DynIndex::new_dyn(1);
///     Ok(TreeTN::from_tensors(
///         vec![
///             IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0, 1.0])?,
///             IdxTensor::from_dense(vec![bond, b.clone()], values)?,
///         ],
///         vec![0usize, 1],
///     )?)
/// };
/// let sum = tree(vec![1.0, 2.0, 3.0])?; // 1 + b, constant in a ...
/// let x = PartitionedTreeTN::from_subdomains(vec![
///     SubDomainTreeTN::new(sum.clone(), Projector::from_pairs([(a.clone(), 0)])?)?,
///     SubDomainTreeTN::new(tree(vec![2.0, 3.0, 4.0])?, Projector::from_pairs([(a.clone(), 1)])?)?,
/// ])?; // ... so x = 1 + a + b
/// let y = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(
///     tree(vec![2.0, 2.0, 2.0])?,
///     Projector::from_pairs([(a.clone(), 1)])?,
/// )?)?;
///
/// let options = PatchedInterpolationOptions::new(4)
///     .with_error_norm(ErrorNorm::l2(L2Reference::Given(10.0)))
///     .with_tolerance(ErrorTolerance { rtol: 1e-10, atol: 0.0 })
///     .with_patch_order(vec![a.clone(), b.clone()]);
/// let mut backend = CpuBackend::with_threads(1)?;
/// let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
/// let result = patched_elementwise::<f64, _, _, _, _>(
///     &PointwiseElementwise(TreeTciInterpolator::default()),
///     &TreeTciInterpolator::default(),
///     &[&x, &y],
///     &Product,
///     &mut ctx,
///     &options,
/// )?;
/// // y is zero on a = 0, so x * y is a zero patch there and 2 (2 + b) on a = 1.
/// assert_eq!(result.report.zero_patches.len(), 1);
/// let dense = result.partition.to_treetn()?.to_dense()?;
/// let expected = IdxTensor::from_dense(
///     vec![a.clone(), b.clone()],
///     vec![0.0, 4.0, 0.0, 6.0, 0.0, 8.0],
/// )?;
/// assert!(dense.sub(&expected)?.maxabs()? < 1e-10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn patched_elementwise<T, V, E, Op, R>(
    engine: &E,
    residual_engine: &R,
    inputs: &[&PartitionedTreeTN<V>],
    operator: &Op,
    ctx: &mut EvalContext<'_>,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: ElementwiseInterpolator<T, Op> + Sync,
    Op: ElementwiseOperator<T>,
    R: TreeInterpolator<T> + Sync,
{
    let Prepared {
        engine,
        target,
        topology,
        node_sites,
        pivots,
    } = prepare(engine, residual_engine, inputs, operator, options)?;
    let plan = Plan::validated(&engine, topology, node_sites, pivots, options)?;
    run_serial(ctx, &plan, |batch, ctx| target.evaluate(batch, ctx))
}

/// The lanes counterpart of [`patched_elementwise`]: the same validation,
/// patches, and report, with each wave of patches spread over the lanes of a
/// caller-owned [`TenferroLanes`], as in
/// [`patched_interpolate_on_lanes`](super::patched_interpolate_on_lanes).
///
/// Every patch runs on the engine of the lane that pulled it, including the
/// restriction of the inputs, the kernel, and the pointwise evaluation of the
/// target. Cached input evaluators are pooled across lanes: a lane takes one
/// out while it evaluates, so no evaluator is shared between threads at
/// once. The result does not depend on the number of lanes, as for the
/// interpolation entries.
///
/// # Arguments
///
/// As for [`patched_elementwise`], with `lanes` in place of the context; the
/// engines and the operator are shared by the lanes, so they must be `Sync`.
///
/// # Errors
///
/// As for [`patched_elementwise`], plus
/// [`PatchedInterpolationError::FanOut`] when the lane executor fails.
///
/// # Examples
///
/// ```
/// use hataori_tenferro::TenferroLanes;
/// use tenferro_cpu::CpuEngine;
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_elementwise_on_lanes, PatchedInterpolationOptions,
/// };
/// use tensor4all_partitionedtreetn::{
///     ErrorNorm, ErrorTolerance, L2Reference, PartitionedTreeTN, Projector, SubDomainTreeTN,
/// };
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::interpolation::{PointwiseElementwise, Product};
/// use tensor4all_treetn::TreeTN;
///
/// // x(a, b) = 1 + b, stored as two patches split on a, squared on two lanes.
/// let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
/// let bond = DynIndex::new_dyn(1);
/// let tree = TreeTN::from_tensors(
///     vec![
///         IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0, 1.0])?,
///         IdxTensor::from_dense(vec![bond, b.clone()], vec![1.0, 2.0, 3.0])?,
///     ],
///     vec![0usize, 1],
/// )?;
/// let x = PartitionedTreeTN::from_subdomains(vec![
///     SubDomainTreeTN::new(tree.clone(), Projector::from_pairs([(a.clone(), 0)])?)?,
///     SubDomainTreeTN::new(tree, Projector::from_pairs([(a.clone(), 1)])?)?,
/// ])?;
/// let options = PatchedInterpolationOptions::new(4)
///     .with_error_norm(ErrorNorm::l2(L2Reference::Given(10.0)))
///     .with_tolerance(ErrorTolerance { rtol: 1e-10, atol: 0.0 })
///     .with_patch_order(vec![a.clone(), b.clone()]);
/// let mut lanes = TenferroLanes::new(vec![CpuEngine::serial(), CpuEngine::serial()])?;
/// let result = patched_elementwise_on_lanes::<f64, _, _, _, _>(
///     &mut lanes,
///     &PointwiseElementwise(TreeTciInterpolator::default()),
///     &TreeTciInterpolator::default(),
///     &[&x, &x],
///     &Product,
///     &options,
/// )?;
/// assert_eq!(result.partition.len(), 2);
/// let dense = result.partition.to_treetn()?.to_dense()?;
/// let expected = IdxTensor::from_dense(
///     vec![a.clone(), b.clone()],
///     vec![1.0, 1.0, 4.0, 4.0, 9.0, 9.0],
/// )?;
/// assert!(dense.sub(&expected)?.maxabs()? < 1e-10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn patched_elementwise_on_lanes<T, V, E, Op, R>(
    lanes: &mut TenferroLanes,
    engine: &E,
    residual_engine: &R,
    inputs: &[&PartitionedTreeTN<V>],
    operator: &Op,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement + Send,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: ElementwiseInterpolator<T, Op> + Sync,
    Op: ElementwiseOperator<T> + Sync,
    R: TreeInterpolator<T> + Sync,
{
    let Prepared {
        engine,
        target,
        topology,
        node_sites,
        pivots,
    } = prepare(engine, residual_engine, inputs, operator, options)?;
    let plan = Plan::validated(&engine, topology, node_sites, pivots, options)?;
    run_on_lanes(lanes, &plan, |batch, ctx| target.evaluate(batch, ctx))
}

/// What every entry of `patched_elementwise` builds from its inputs before
/// the plan: the engine, the pointwise target, and the problem's layout.
pub(super) struct Prepared<'a, T, V, E, Op, R>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    pub(super) engine: Elementwise<'a, E, Op, R, V>,
    pub(super) target: Target<'a, T, V, Op>,
    pub(super) topology: NodeNameNetwork<V>,
    pub(super) node_sites: BTreeMap<V, Vec<DynIndex>>,
    pub(super) pivots: ColMajorArray<usize>,
}

/// Validate the inputs of `patched_elementwise` and build its parts.
///
/// # Errors
///
/// As documented on [`patched_elementwise`] for the inputs.
pub(super) fn prepare<'a, T, V, E, Op, R>(
    engine: &'a E,
    residual_engine: &'a R,
    inputs: &[&'a PartitionedTreeTN<V>],
    operator: &'a Op,
    options: &PatchedInterpolationOptions,
) -> Result<Prepared<'a, T, V, E, Op, R>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    if inputs.is_empty() {
        return Err(invalid(
            "patched_elementwise needs at least one input partition",
        ));
    }
    let ordered: Vec<Vec<&SubDomainTreeTN<V>>> =
        inputs.iter().map(|input| input.ordered_patches()).collect();
    let template = *ordered.iter().flatten().next().ok_or_else(|| {
        invalid(
            "every input partition of patched_elementwise is empty; at least one patch is \
             needed to know the site space",
        )
    })?;
    let dtype = template.scalar_kind()?;
    for patch in ordered.iter().flatten() {
        ensure_same_tree_structure(template.data(), patch.data())?;
        ensure_same_dtype(dtype, patch.scalar_kind()?)?;
    }
    if !options.patch_order.is_empty() {
        for patch in ordered.iter().flatten() {
            if let Some(index) = patch
                .projector()
                .projected_indices()
                .find(|index| !options.patch_order.contains(index))
            {
                return Err(invalid(format!(
                    "the input projector {:?} fixes {index:?}, which patch_order does not list; \
                     a structural split needs every such site in patch_order",
                    patch.projector()
                )));
            }
        }
    }

    let network = template.data();
    let topology: NodeNameNetwork<V> = network.site_index_network().topology().clone();
    let mut node_sites = BTreeMap::new();
    for node in network.node_names() {
        let sites = network.node_site_indices(&node).unwrap_or_default();
        node_sites.insert(node, sites);
    }
    let sites = InterpolationProblem::derive_site_order(&node_sites);
    let positions: HashMap<&DynIndex, usize> = sites
        .iter()
        .enumerate()
        .map(|(row, site)| (site, row))
        .collect();
    let locators = inputs
        .iter()
        .map(|input| PatchLocator::new(input, &positions))
        .collect::<Result<Vec<_>, _>>()?;
    let target = Target {
        inputs: locators,
        sites: sites.clone(),
        operator,
        evaluators: Mutex::new(HashMap::new()),
        scalar: std::marker::PhantomData::<T>,
    };
    let engine = Elementwise {
        engine,
        operator,
        residual: residual_engine,
        inputs: ordered
            .into_iter()
            .map(|patches| Input { patches })
            .collect(),
    };
    let pivots = ColMajorArray::new(Vec::new(), vec![sites.len(), 0]).map_err(super::internal)?;
    Ok(Prepared {
        engine,
        target,
        topology,
        node_sites,
        pivots,
    })
}

#[cfg(test)]
mod tests;
