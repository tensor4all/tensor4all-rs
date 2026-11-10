//! Element-wise interpolation: a pointwise map of known input networks.
//!
//! [`TreeInterpolator`] samples a black-box function. An element-wise
//! operation `g(x) = f(x_1(x), ..., x_k(x))` knows more: its inputs are tree
//! networks on the problem's sites, and a dedicated kernel (TreeACI for any
//! `f`, tree RSI for products) can use their structure. An
//! [`ElementwiseInterpolator`] receives the input networks and the operator
//! instead of an evaluator. The operator is a type parameter, so an engine
//! that supports only some operators (tree RSI implements only [`Product`])
//! rejects the others at compile time.
//!
//! [`PointwiseElementwise`] turns any [`TreeInterpolator`] into an
//! [`ElementwiseInterpolator`] by evaluating the inputs pointwise; it needs
//! no kernel of its own.

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{ColMajorArrayRef, CommonScalar, IdxTensor, TensorElement};

use super::{
    EvalContext, InterpolationError, InterpolationOutcome, InterpolationProblem, TreeInterpolator,
};
use crate::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

/// A pointwise map `f(v_1, ..., v_k)` applied to a batch of points.
///
/// `inputs` is a column-major `[k, n_points]` array: column `p` holds the
/// `k` input values at point `p`. The implementation writes one value per
/// point into `output`, whose length is `n_points`. A closure with the same
/// signature implements the trait.
///
/// # Examples
///
/// ```
/// use tensor4all_core::ColMajorArrayRef;
/// use tensor4all_treetn::interpolation::{ElementwiseOperator, Product};
///
/// let values = [2.0_f64, 3.0, 4.0, 5.0]; // two inputs at two points
/// let batch = ColMajorArrayRef::new(&values, &[2, 2])?;
/// let mut product = [0.0; 2];
/// Product.apply(batch, &mut product)?;
/// assert_eq!(product, [6.0, 20.0]);
///
/// let sum = |inputs: ColMajorArrayRef<'_, f64>, output: &mut [f64]| -> anyhow::Result<()> {
///     for (point, value) in output.iter_mut().enumerate() {
///         *value = inputs.get(&[0, point]).unwrap() + inputs.get(&[1, point]).unwrap();
///     }
///     Ok(())
/// };
/// let mut total = [0.0; 2];
/// sum.apply(batch, &mut total)?;
/// assert_eq!(total, [5.0, 9.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait ElementwiseOperator<T> {
    /// Apply the map to every column of `inputs`.
    ///
    /// # Errors
    ///
    /// Returns the operator's own error; the engine reports it as an
    /// evaluator failure.
    fn apply(&self, inputs: ColMajorArrayRef<'_, T>, output: &mut [T]) -> anyhow::Result<()>;
}

impl<T, F> ElementwiseOperator<T> for F
where
    F: Fn(ColMajorArrayRef<'_, T>, &mut [T]) -> anyhow::Result<()>,
{
    fn apply(&self, inputs: ColMajorArrayRef<'_, T>, output: &mut [T]) -> anyhow::Result<()> {
        self(inputs, output)
    }
}

/// The product of all inputs, `f(v_1, ..., v_k) = v_1 * ... * v_k`; the
/// Hadamard product for `k = 2`.
///
/// A marker type rather than a closure, so that an engine can implement
/// [`ElementwiseInterpolator`] for products only.
///
/// # Examples
///
/// ```
/// use tensor4all_core::ColMajorArrayRef;
/// use tensor4all_treetn::interpolation::{ElementwiseOperator, Product};
///
/// let values = [2.0_f64, 3.0, 4.0]; // three inputs at one point
/// let mut output = [0.0];
/// Product.apply(ColMajorArrayRef::new(&values, &[3, 1])?, &mut output)?;
/// assert_eq!(output, [24.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Product;

impl<T> ElementwiseOperator<T> for Product
where
    T: Copy + std::ops::Mul<Output = T>,
{
    fn apply(&self, inputs: ColMajorArrayRef<'_, T>, output: &mut [T]) -> anyhow::Result<()> {
        let &[k, n_points] = inputs.shape() else {
            anyhow::bail!("Product: inputs must be a [k, n_points] array");
        };
        if k == 0 {
            anyhow::bail!("Product: at least one input is required");
        }
        if output.len() != n_points {
            anyhow::bail!(
                "Product: output has {} entries for {n_points} points",
                output.len()
            );
        }
        for (column, value) in inputs.data().chunks(k).zip(output.iter_mut()) {
            *value = column[1..].iter().fold(column[0], |acc, &v| acc * v);
        }
        Ok(())
    }
}

/// An engine that interpolates `f(x_1, ..., x_k)` from the input networks.
///
/// `inputs` carry exactly the problem's topology and active sites (the same
/// node names and the same sites per node as `problem.node_sites()`); the
/// returned outcome follows the [`TreeInterpolator`] contract for the same
/// problem, so code that runs engines treats both kinds alike.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, DynIndex, IdxTensor};
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treetn::interpolation::{
///     ElementwiseInterpolator, EvalContext, InterpolationOutcome, InterpolationProblem,
///     InterpolationTermination, Parallelism, Product,
/// };
/// use tensor4all_treetn::{NodeNameNetwork, TreeTN};
///
/// // An engine for products only: it returns the first input unchanged,
/// // which is the product when the other input is all ones.
/// struct FirstInput;
/// impl ElementwiseInterpolator<f64, Product> for FirstInput {
///     fn interpolate_elementwise<V>(
///         &self,
///         _problem: &InterpolationProblem<V>,
///         inputs: &[TreeTN<IdxTensor, V>],
///         _operator: &Product,
///         _ctx: &mut EvalContext<'_>,
///     ) -> Result<InterpolationOutcome<V>, tensor4all_treetn::interpolation::InterpolationError>
///     where
///         V: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
///     {
///         Ok(InterpolationOutcome {
///             network: inputs[0].clone(),
///             termination: InterpolationTermination::Converged,
///             error_estimate: 0.0,
///             max_sample_magnitude: 0.0,
///             pivots: None,
///         })
///     }
/// }
///
/// let site = DynIndex::new_dyn(2);
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let problem = InterpolationProblem::new(
///     topology,
///     BTreeMap::from([(0usize, vec![site.clone()])]),
///     ColMajorArray::new(vec![0], vec![1, 1])?,
///     1e-12,
///     None,
///     0,
/// )?;
/// let input = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 3.0])?],
///     vec![0usize],
/// )?;
/// let mut backend = CpuBackend::with_threads(1)?;
/// let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
/// let outcome = FirstInput.interpolate_elementwise(&problem, &[input], &Product, &mut ctx)?;
/// assert_eq!(outcome.network.to_dense()?.to_vec::<f64>()?, vec![2.0, 3.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait ElementwiseInterpolator<T, Op> {
    /// Interpolate `operator` applied to `inputs` for `problem`.
    ///
    /// # Errors
    ///
    /// Returns [`InterpolationError`] as a [`TreeInterpolator`] does; an
    /// operator failure is an [`InterpolationError::Evaluator`].
    fn interpolate_elementwise<V>(
        &self,
        problem: &InterpolationProblem<V>,
        inputs: &[TreeTN<IdxTensor, V>],
        operator: &Op,
        ctx: &mut EvalContext<'_>,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync;
}

/// Any [`TreeInterpolator`] as an [`ElementwiseInterpolator`] for any
/// operator: the wrapped engine samples `f` at its points, and each sample
/// evaluates every input network at that point on the run's session.
///
/// It uses no structure of the inputs beyond pointwise evaluation, so a
/// dedicated kernel can be cheaper; it is the generic fallback and a
/// reference for kernels.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use std::fmt::Debug;
/// use std::hash::Hash;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
/// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
/// use tensor4all_treetn::interpolation::{
///     ElementwiseInterpolator, EvalContext, InterpolationError, InterpolationOutcome,
///     InterpolationProblem, InterpolationTermination, Parallelism, PointwiseElementwise,
///     Product, TreeInterpolator,
/// };
/// use tensor4all_treetn::{NodeNameNetwork, TreeTN};
///
/// // A black-box engine for one node with one site: it evaluates every point.
/// struct Exhaustive;
/// impl TreeInterpolator<f64> for Exhaustive {
///     fn interpolate<V, F>(
///         &self,
///         problem: &InterpolationProblem<V>,
///         ctx: &mut EvalContext<'_>,
///         mut evaluate: F,
///     ) -> Result<InterpolationOutcome<V>, InterpolationError>
///     where
///         V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
///         F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<f64>>,
///     {
///         let site = problem.site_order()[0].clone();
///         let points: Vec<usize> = (0..site.dim).collect();
///         let shape = [1, site.dim];
///         let batch = ColMajorArrayRef::new(&points, &shape).unwrap();
///         let values = evaluate(batch, ctx).map_err(|source| InterpolationError::Evaluator { source })?;
///         let node = problem.node_sites().keys().next().unwrap().clone();
///         let tensor = IdxTensor::from_dense(vec![site], values)
///             .map_err(|e| InterpolationError::Engine { source: e.into() })?;
///         Ok(InterpolationOutcome {
///             network: TreeTN::from_tensors(vec![tensor], vec![node])
///                 .map_err(|e| InterpolationError::Engine { source: e.into() })?,
///             termination: InterpolationTermination::Converged,
///             error_estimate: 0.0,
///             max_sample_magnitude: 0.0,
///             pivots: None,
///         })
///     }
/// }
///
/// let site = DynIndex::new_dyn(3);
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node(0usize)?;
/// let problem = InterpolationProblem::new(
///     topology,
///     BTreeMap::from([(0usize, vec![site.clone()])]),
///     ColMajorArray::new(vec![0], vec![1, 1])?,
///     1e-12,
///     None,
///     0,
/// )?;
/// let input = |values: Vec<f64>| -> Result<TreeTN<IdxTensor, usize>, Box<dyn std::error::Error>> {
///     Ok(TreeTN::from_tensors(
///         vec![IdxTensor::from_dense(vec![site.clone()], values)?],
///         vec![0usize],
///     )?)
/// };
/// let inputs = [input(vec![1.0, 2.0, 3.0])?, input(vec![4.0, 5.0, 6.0])?];
/// let mut backend = CpuBackend::with_threads(1)?;
/// let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
/// let outcome = PointwiseElementwise(Exhaustive)
///     .interpolate_elementwise(&problem, &inputs, &Product, &mut ctx)?;
/// assert_eq!(outcome.network.to_dense()?.to_vec::<f64>()?, vec![4.0, 10.0, 18.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct PointwiseElementwise<E>(pub E);

impl<T, Op, E> ElementwiseInterpolator<T, Op> for PointwiseElementwise<E>
where
    T: CommonScalar + TensorElement,
    Op: ElementwiseOperator<T>,
    E: TreeInterpolator<T>,
{
    fn interpolate_elementwise<V>(
        &self,
        problem: &InterpolationProblem<V>,
        inputs: &[TreeTN<IdxTensor, V>],
        operator: &Op,
        ctx: &mut EvalContext<'_>,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    {
        if inputs.is_empty() {
            return Err(InterpolationError::InvalidProblem {
                message: "an element-wise interpolation needs at least one input".to_string(),
            });
        }
        let sites = problem.site_order();
        let mut evaluators = inputs
            .iter()
            .map(|network| {
                let options = CachedEvaluatorOptions {
                    center: network.node_names().into_iter().min(),
                    ..CachedEvaluatorOptions::default()
                };
                TreeTNCachedEvaluator::new(network, sites, options)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| InterpolationError::Engine {
                source: anyhow::Error::new(source),
            })?;
        let k = inputs.len();
        self.0.interpolate(problem, ctx, |batch, ctx| {
            let n_points = batch.shape().get(1).copied().unwrap_or(0);
            let mut columns = vec![T::default(); k * n_points];
            for (input, evaluator) in evaluators.iter_mut().enumerate() {
                let values = evaluator.evaluate_batched_typed_on_session::<T>(
                    ctx.session(),
                    batch,
                    EvaluationHint::default(),
                )?;
                for (point, value) in values.into_iter().enumerate() {
                    columns[input + k * point] = value;
                }
            }
            let shape = [k, n_points];
            let mut output = vec![T::default(); n_points];
            operator.apply(ColMajorArrayRef::new(&columns, &shape)?, &mut output)?;
            Ok(output)
        })
    }
}

#[cfg(test)]
mod tests;
