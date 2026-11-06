//! Final stored-rank reports over the existing contraction dispatch.

use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;

use rand::Rng;
use tensor4all_core::{IdxTensor, IndexLike, TensorLike};
use tensor4all_tensorbackend::ExecutionContext;

use super::{ContractionMethod, ContractionOptions, TreeTN, TreeTNOperationError};

/// The stored bond dimension of one output edge, oriented toward the report root.
///
/// This is an entry of [`ContractionReport`], not a Schmidt-rank or error estimate.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::contraction::EdgeRank;
/// let edge = EdgeRank { child: "leaf", parent: "hub", rank: 2 };
/// assert_eq!((edge.child, edge.parent, edge.rank), ("leaf", "hub", 2));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeRank<V> {
    /// Endpoint farther from the report root.
    pub child: V,
    /// Endpoint closer to the report root.
    pub parent: V,
    /// Dimension of the final stored link; equality with an abort threshold saturates.
    pub rank: usize,
}

/// Final output-link dimensions in leaves-to-root sweep order.
///
/// Unlike truncation diagnostics, this report describes only the stored network;
/// it gives no discarded-weight, accuracy, or mathematical-rank guarantee.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::contraction::{ContractionReport, EdgeRank};
/// let report = ContractionReport {
///     edge_ranks: vec![EdgeRank { child: 1, parent: 0, rank: 3 }],
/// };
/// assert_eq!(report.edge_ranks, vec![EdgeRank { child: 1, parent: 0, rank: 3 }]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionReport<V> {
    /// Every output edge in [`TreeTN::edges_to_canonicalize_by_names`] order.
    ///
    /// The root is the requested center if it survives, otherwise the unique
    /// canonical-region node, otherwise the smallest output node name. An
    /// edgeless result has an empty list.
    pub edge_ranks: Vec<EdgeRank<V>>,
}

/// A completed contraction with its report, or its first threshold-reaching edge.
///
/// [`contract_with_outcome`] currently performs the full contraction before
/// classification. `Saturated` discards the computed network and does not imply
/// that work was aborted early or that a truncation error occurred.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{TreeTN, contraction::{
///     contract_with_outcome, ContractionOptions, ContractionOutcome,
/// }};
/// let i = DynIndex::new_dyn(2);
/// let a = TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![i.clone()], vec![1., 2.])?], vec![0])?;
/// let b = TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![i], vec![3., 4.])?], vec![0])?;
/// let outcome = contract_with_outcome(&a, &b, &0, ContractionOptions::zipup(), None)?;
/// let ContractionOutcome::Completed { network, report } = outcome else { unreachable!() };
/// assert!(report.edge_ranks.is_empty());
/// assert!((network.to_dense()?.to_vec::<f64>()?[0] - 11.).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
// Keep the completed network by value, matching `contract` without an extra heap allocation.
#[allow(clippy::large_enum_variant)]
pub enum ContractionOutcome<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// The stored network has no edge at or above the threshold, or no threshold was supplied.
    Completed {
        /// The unmodified result of the selected contraction method.
        network: TreeTN<T, V>,
        /// Stored output-link dimensions in deterministic report order.
        report: ContractionReport<V>,
    },
    /// The first report edge whose stored rank is at least the threshold.
    Saturated {
        /// Its endpoints and final stored dimension; no partial network is returned.
        edge: EdgeRank<V>,
    },
}

/// Contract two trees completely, then report their stored output ranks or saturation.
///
/// # Arguments
///
/// * `a`, `b` - Operands accepted by [`super::contract`], including its context and topology rules.
/// * `center` - Contraction center and preferred report root.
/// * `options` - The ordinary contraction options, unchanged. In particular,
///   Naive requires an explicit dense-reference limit; randomized methods keep
///   their current seed fields and named RNG.
/// * `abort_at_rank` - `None` returns `Completed`; `Some(k)` returns `Saturated`
///   on the first report edge with stored rank `>= k`. It is independent of
///   `options.max_bond_dim`. An edgeless result always completes.
///
/// # Returns
///
/// A [`ContractionOutcome`]. All numerical work completes before rank detection,
/// even when saturation is reported; this version does not save contraction work.
/// No payload readback, canonicalization, or additional truncation is performed.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] for the same option, topology, context,
/// factorization or backend failures as [`super::contract`], preserving their
/// sources, or when the returned network's edge metadata cannot form a report.
///
/// # Examples
///
/// The result is `|00> + 2 |11>`, whose stored bond has dimension two.
///
/// ```
/// use std::num::NonZeroUsize;
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{TreeTN, contraction::{
///     contract_with_outcome, ContractionOptions, ContractionOutcome, EdgeRank,
/// }};
/// let k = DynIndex::new_dyn(2);
/// let x = DynIndex::new_dyn(2);
/// let y = DynIndex::new_dyn(2);
/// let a_bond = DynIndex::new_dyn(2);
/// let b_bond = DynIndex::new_dyn(1);
/// let a = TreeTN::from_tensors(vec![
///     IdxTensor::from_dense(vec![k.clone(), x, a_bond.clone()], vec![1., 0., 0., 0., 0., 0., 0., 2.])?,
///     IdxTensor::from_dense(vec![a_bond, y], vec![1., 0., 0., 1.])?,
/// ], vec![0, 1])?;
/// let b = TreeTN::from_tensors(vec![
///     IdxTensor::from_dense(vec![k, b_bond.clone()], vec![1., 1.])?,
///     IdxTensor::from_dense(vec![b_bond], vec![1.])?,
/// ], vec![0, 1])?;
/// let outcome = contract_with_outcome(&a, &b, &1, ContractionOptions::zipup(), NonZeroUsize::new(2))?;
/// let ContractionOutcome::Saturated { edge } = outcome else { unreachable!() };
/// assert_eq!(edge, EdgeRank { child: 0, parent: 1, rank: 2 });
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_with_outcome<T, V>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    center: &V,
    options: ContractionOptions,
    abort_at_rank: Option<NonZeroUsize>,
) -> Result<ContractionOutcome<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    finish(
        super::contract(a, b, center, options)?,
        center,
        abort_at_rank,
    )
}

/// Contract completely on a caller-owned context, then classify the final stored ranks.
///
/// # Arguments
///
/// * `a`, `b`, `center`, `options` - The inputs and method/options of [`super::contract_in`].
/// * `abort_at_rank` - Independent stored-rank threshold; equality saturates,
///   `None` always completes, and an edgeless result always completes.
/// * `context` - The execution context used by every numerical stage, validated
///   by the existing context entry before any contraction or RNG consumption.
///
/// # Returns
///
/// The same report/classification as [`contract_with_outcome`], without changing
/// the context or AD semantics of the underlying method. Detection is post-hoc.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] for all errors of [`super::contract_in`]
/// (including foreign contexts and unsupported storage), with preserved sources,
/// or invalid output-edge metadata.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};
/// use tensor4all_treetn::{TreeTN, contraction::{
///     contract_with_outcome_in, ContractionOptions, ContractionOutcome,
/// }};
/// let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?)));
/// let i = DynIndex::new_dyn(2);
/// let a = TreeTN::from_tensors(vec![IdxTensor::from_dense_in(&context, vec![i.clone()], vec![1., 2.])?], vec![0])?;
/// let b = TreeTN::from_tensors(vec![IdxTensor::from_dense_in(&context, vec![i], vec![3., 4.])?], vec![0])?;
/// let outcome = contract_with_outcome_in(&a, &b, &0, ContractionOptions::zipup(), None, &context)?;
/// let ContractionOutcome::Completed { network, .. } = outcome else { unreachable!() };
/// assert!((network.to_dense()?.to_vec::<f64>()?[0] - 11.).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_with_outcome_in<V>(
    a: &TreeTN<IdxTensor, V>,
    b: &TreeTN<IdxTensor, V>,
    center: &V,
    options: ContractionOptions,
    abort_at_rank: Option<NonZeroUsize>,
    context: &ExecutionContext,
) -> Result<ContractionOutcome<IdxTensor, V>, TreeTNOperationError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    finish(
        super::contract_in(a, b, center, options, context)?,
        center,
        abort_at_rank,
    )
}

/// Contract completely with a caller-owned RNG, then classify final stored ranks.
///
/// # Arguments
///
/// * `a`, `b`, `center` - Contraction operands and center.
/// * `options` - SRC options or RSVD Zipup/Fit options accepted by
///   [`super::contract_src_with_rng`] or [`super::contract_with_rng`]. Seed fields
///   are ignored; deterministic factorization is rejected as in those entries.
/// * `abort_at_rank` - Stored-rank threshold; equality saturates. `None` and
///   edgeless outputs complete normally.
/// * `rng` - Consumed directly through the existing dispatch, even when the
///   result saturates; no private stream is created.
///
/// # Returns
///
/// A post-hoc outcome with the same result and RNG consumption as the matching
/// caller-RNG entry. Saturation discards the completed network.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] for the existing caller-RNG entry's invalid
/// options, unsupported method/storage, topology, context or numerical failures,
/// with preserved sources, or invalid output-edge metadata.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{DynIndex, FactorizeAlg, IdxTensor};
/// use tensor4all_treetn::{TreeTN, contraction::{
///     contract_with_outcome_with_rng, ContractionOptions, ContractionOutcome,
/// }};
/// let i = DynIndex::new_dyn(2);
/// let a = TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![i.clone()], vec![1., 2.])?], vec![0])?;
/// let b = TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![i], vec![3., 4.])?], vec![0])?;
/// let options = ContractionOptions::zipup().with_factorize_alg(FactorizeAlg::RSVD).with_max_bond_dim(1);
/// let mut rng = ChaCha8Rng::seed_from_u64(7);
/// let outcome = contract_with_outcome_with_rng(&a, &b, &0, options, None, &mut rng)?;
/// let ContractionOutcome::Completed { network, .. } = outcome else { unreachable!() };
/// assert!((network.to_dense()?.to_vec::<f64>()?[0] - 11.).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_with_outcome_with_rng<T, V, R>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    center: &V,
    options: ContractionOptions,
    abort_at_rank: Option<NonZeroUsize>,
    rng: &mut R,
) -> Result<ContractionOutcome<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
    R: Rng + ?Sized,
{
    let network = if options.method == ContractionMethod::Src {
        super::contract_src_with_rng(a, b, center, options, rng)?
    } else {
        super::contract_with_rng(a, b, center, options, rng)?
    };
    finish(network, center, abort_at_rank)
}

/// Run caller-RNG contraction completely on `context`, then classify final ranks.
///
/// # Arguments
///
/// * `a`, `b`, `center`, `options` - Inputs and random method as in
///   [`contract_with_outcome_with_rng`]; SRC and RSVD Zipup/Fit are supported.
/// * `abort_at_rank` - Independent stored-rank threshold; `None` completes,
///   equality saturates, and edgeless outputs complete.
/// * `rng` - Caller-owned stream consumed by the complete numerical execution.
/// * `context` - Execution owner validated by the selected existing entry.
///
/// # Returns
///
/// A post-hoc outcome preserving the corresponding caller-RNG/context entry's
/// numerical result, context/AD rules and RNG consumption, without readback.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] for errors of [`super::contract_with_rng_in`]
/// or [`super::contract_src_with_rng_in`], including invalid options and foreign
/// contexts, with preserved sources, or invalid output-edge metadata.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{DynIndex, FactorizeAlg, IdxTensor};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};
/// use tensor4all_treetn::{TreeTN, contraction::{
///     contract_with_outcome_with_rng_in, ContractionOptions, ContractionOutcome,
/// }};
/// let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?)));
/// let i = DynIndex::new_dyn(2);
/// let a = TreeTN::from_tensors(vec![IdxTensor::from_dense_in(&context, vec![i.clone()], vec![1., 2.])?], vec![0])?;
/// let b = TreeTN::from_tensors(vec![IdxTensor::from_dense_in(&context, vec![i], vec![3., 4.])?], vec![0])?;
/// let options = ContractionOptions::zipup().with_factorize_alg(FactorizeAlg::RSVD).with_max_bond_dim(1);
/// let mut rng = ChaCha8Rng::seed_from_u64(7);
/// let outcome = contract_with_outcome_with_rng_in(&a, &b, &0, options, None, &mut rng, &context)?;
/// let ContractionOutcome::Completed { network, .. } = outcome else { unreachable!() };
/// assert!((network.to_dense()?.to_vec::<f64>()?[0] - 11.).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_with_outcome_with_rng_in<V, R>(
    a: &TreeTN<IdxTensor, V>,
    b: &TreeTN<IdxTensor, V>,
    center: &V,
    options: ContractionOptions,
    abort_at_rank: Option<NonZeroUsize>,
    rng: &mut R,
    context: &ExecutionContext,
) -> Result<ContractionOutcome<IdxTensor, V>, TreeTNOperationError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
    R: Rng + ?Sized,
{
    let network = if options.method == ContractionMethod::Src {
        super::contract_src_with_rng_in(a, b, center, options, rng, context)?
    } else {
        super::contract_with_rng_in(a, b, center, options, rng, context)?
    };
    finish(network, center, abort_at_rank)
}

fn finish<T, V>(
    network: TreeTN<T, V>,
    center: &V,
    threshold: Option<NonZeroUsize>,
) -> Result<ContractionOutcome<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut edge_ranks = Vec::with_capacity(network.edge_count());
    if network.edge_count() != 0 {
        let root = if network.node_index(center).is_some() {
            center.clone()
        } else if network.canonical_region().len() == 1 {
            network
                .canonical_region()
                .iter()
                .next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("contraction outcome: missing canonical root"))?
        } else {
            network
                .node_names()
                .into_iter()
                .min()
                .ok_or_else(|| anyhow::anyhow!("contraction outcome: missing output root"))?
        };
        let edges = network
            .edges_to_canonicalize_by_names(&root)
            .ok_or_else(|| anyhow::anyhow!("contraction outcome: output root is absent"))?;
        for (child, parent) in edges {
            let rank = network
                .edge_between(&child, &parent)
                .and_then(|edge| network.bond_index(edge))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "contraction outcome: missing output bond between {child:?} and {parent:?}"
                    )
                })?
                .dim();
            let edge = EdgeRank {
                child,
                parent,
                rank,
            };
            if threshold.is_some_and(|threshold| rank >= threshold.get()) {
                return Ok(ContractionOutcome::Saturated { edge });
            }
            edge_ranks.push(edge);
        }
    }
    Ok(ContractionOutcome::Completed {
        network,
        report: ContractionReport { edge_ranks },
    })
}

#[cfg(test)]
mod tests;
