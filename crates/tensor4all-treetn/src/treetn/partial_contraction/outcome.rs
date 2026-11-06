//! Rank outcomes for the existing partial-contraction adapter.
//!
//! This adapter's correctness argument is independent of the patching paper;
//! see `docs/design/treetn-contraction-outcome.md`. The existing local copy
//! construction changes site legs only, and restoration only renames those
//! legs, so retained graph edges keep their endpoints and dimensions. Ordered
//! outputs and mismatched topologies retain their complete existing pipelines.

use std::num::NonZeroUsize;

use super::*;
use crate::contraction::{contract_with_outcome, report_contraction_outcome, ContractionOutcome};

/// Partially contract two trees, reporting final stored ranks or saturation.
///
/// # Arguments
///
/// * `a`, `b` - Operands accepted by [`partial_contract`], with the same context
///   and scalar-kind requirements; this is its compatibility entry, not a new
///   caller-owned-context or caller-RNG API.
/// * `spec` - Contracted and diagonal site pairs, and optional final output order,
///   with the same validation and index semantics as [`partial_contract`].
/// * `center` - Contraction center and preferred report root.
/// * `options` - Unchanged contraction, seed, truncation and dense/reference options.
/// * `abort_at_rank` - Independent positive stored-rank threshold. Equality
///   saturates; `None` always completes and edgeless outputs always complete.
///
/// # Returns
///
/// [`ContractionOutcome::Completed`] contains the ordinary partial-contraction
/// result with its original left-hand diagonal indices restored, plus its rank
/// report. For equal input topologies without `output_order`, the selected
/// method uses [`contract_with_outcome`], including general-tree SVD zip-up
/// cancellation. No auxiliary graph nodes are introduced by the copy tensors.
/// Other methods, ordered outputs and mismatched topologies classify only
/// after completion. Their reports use the final network's edge order and
/// include every retained edge, including dimension-one structural links.
/// Saturation returns only the edge, never a partial network or error estimate.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] for the same invalid options, site-pair
/// specification, alignment, context, factorization, restoration, output-order
/// or explicitly bounded dense/reference failures as [`partial_contract`].
/// Options and pair validation precede cancellation. Output ordering and
/// mismatched-topology fallback always finish before classification, so their
/// errors cannot be hidden by a rank shortcut. Errors from work skipped after
/// an early stop, including subsequent index restoration, are not evaluated.
/// Sources of executed contraction errors are preserved.
///
/// # Examples
///
/// Diagonal pairing represents a Hadamard product and keeps the left index.
/// A one-node result has no edges, so even a threshold of one completes.
///
/// ```
/// use std::num::NonZeroUsize;
/// use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};
/// use tensor4all_treetn::{partial_contract_with_outcome, PartialContractionSpec, TreeTN};
/// use tensor4all_treetn::contraction::{ContractionOptions, ContractionOutcome};
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![i.clone()], vec![2., 3.])?], vec![0])?;
/// let b = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![j.clone()], vec![5., 7.])?], vec![0])?;
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![], diagonal_pairs: vec![(i.clone(), j)], output_order: None,
/// };
/// let outcome = partial_contract_with_outcome(
///     &a, &b, &spec, &0, ContractionOptions::zipup(), NonZeroUsize::new(1))?;
/// let ContractionOutcome::Completed { network, report } = outcome else { unreachable!() };
/// let dense = network.to_dense()?;
/// assert_eq!(dense.external_indices(), vec![i]);
/// assert_eq!(dense.to_vec::<f64>()?, vec![10., 21.]);
/// assert!(report.edge_ranks.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn partial_contract_with_outcome<V>(
    a: &TreeTN<IdxTensor, V>,
    b: &TreeTN<IdxTensor, V>,
    spec: &PartialContractionSpec<DynIndex>,
    center: &V,
    options: ContractionOptions,
    abort_at_rank: Option<NonZeroUsize>,
) -> std::result::Result<ContractionOutcome<IdxTensor, V>, TreeTNOperationError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
{
    // Validate cheap configuration before topology inspection; preparation and
    // the ordinary fallback retain their existing boundary validation too.
    crate::contraction::validate_contract_options(&options)?;
    if spec.output_order.is_some() || !a.same_topology(b) {
        return report_contraction_outcome(
            partial_contract(a, b, spec, center, options)?,
            center,
            abort_at_rank,
        );
    }

    let (mut left, mut right, restore_from, restore_to) =
        prepare_partial_operands(a, b, spec, &options)?;
    align_contract_pair_site_nodes(&left, &mut right, &spec.contract_pairs)?;
    connect_nodewise_outer_product_spectators(&mut left, &mut right)?;
    let outcome = contract_with_outcome(&left, &right, center, options, abort_at_rank)
        .context("partial_contract: contraction failed")?;
    match outcome {
        ContractionOutcome::Completed { network, report } => {
            let network = finish_partial_result(network, &restore_from, &restore_to, None)?;
            Ok(ContractionOutcome::Completed { network, report })
        }
        ContractionOutcome::Saturated { edge } => Ok(ContractionOutcome::Saturated { edge }),
    }
}
