use std::{collections::HashSet, fmt::Debug, hash::Hash};

use tensor4all_core::SvdTruncationPolicy;
use tensor4all_treetn::TruncationOptions;

use super::{
    finite, invalid, ReconstructedTreeTN, ReconstructionOptions, ReconstructionReport,
    ReconstructionTarget, ReconstructionTolerance, Region,
};
use crate::projector::canonical_index_cmp;
use crate::{
    DynIndex, PartitionedTreeTNError, PatchSplitStrategy, Projector, Result, SubDomainTreeTN,
};

struct Reduced<V: Clone + Hash + Eq + Send + Sync + Debug> {
    terms: Vec<SubDomainTreeTN<V>>,
    error: f64,
    merges: usize,
}

struct WorkingRegion<V: Clone + Hash + Eq + Send + Sync + Debug> {
    projector: Projector,
    source: Vec<SubDomainTreeTN<V>>,
    reduced: Reduced<V>,
    allowance: f64,
}

/// Reconstruct an orthogonal target with a fixed global L2 error allowance.
///
/// `target` owns the original data and reference norm; `center` is an existing
/// node for local compression and residual norms. `tolerance` sets
/// `delta = max(atol, rtol * ||target||)`. `options` controls only representation
/// choices. No full dense tensor or initial global direct sum is constructed.
///
/// Pairwise sums are retained only when compression reduces the sum of the
/// operand ranks. Over-goal regions split only if the largest child term rank
/// improves. Candidate children are formed from original region inputs, not
/// from a truncated parent. Failed probes never consume error budget.
/// [`PatchSplitStrategy::Sequential`] considers only the next unprojected
/// nontrivial index in `patch_order`; a rejected split never skips that index.
/// [`PatchSplitStrategy::ExactParameterGain`] selects the best permitted split
/// by logical parameter count, breaking ties in `patch_order` order.
///
/// Every accepted approximation is checked by the norm of its explicit local
/// difference network. Errors are added within a region and combined by
/// `hypot` across disjoint regions. The numerical bound excludes floating-point
/// roundoff; see [`ReconstructionReport`]. The original target, inputs, and
/// caller's options are unchanged. Zero tolerance retains exact terms without
/// SVD compression. Soft rank/search limits never force extra approximation.
///
/// # Errors
/// Returns `InvalidOptions` for invalid tolerances, zero rank/search limits,
/// duplicate or absent split indices; `SiteIndexMismatch` for aliased dimensions;
/// `InvalidCenter` for an absent center on a nonempty target;
/// `NonFiniteAdaptiveValue` for non-finite norms/budgets;
/// `LogicalParameterCountOverflow` for checked counts; and typed TreeTN/backend
/// errors from local products, addition, masking, compression, or residuals.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{DynIndex, IdxTensor, TreeTN, SubDomainTreeTN,
///     PartitionedTreeTN, Projector, reconstruction::*};
/// let site = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?;
/// let tree = TreeTN::from_tensors(vec![tensor], vec![0usize])?;
/// let partition = PartitionedTreeTN::from_subdomains(vec![
///     SubDomainTreeTN::new(tree.clone(), Projector::from_pairs([(site.clone(), 0)])?)?,
///     SubDomainTreeTN::new(tree, Projector::from_pairs([(site, 1)])?)?,
/// ])?;
/// let target = ReconstructionTarget::from_partition(&partition)?;
/// let output = reconstruct(&target, &0, ReconstructionTolerance { rtol: 1e-8, atol: 0.0 },
///     &ReconstructionOptions::default())?;
/// assert!((output.report().reference_norm - 5.0).abs() < 1e-12);
/// assert!(output.report().error_bound <= output.report().absolute_tolerance);
/// let dense = output.into_partition()?.to_treetn()?.to_dense()?;
/// assert_eq!(dense.to_vec::<f64>()?, vec![3.0, 4.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn reconstruct<V>(
    target: &ReconstructionTarget<V>,
    center: &V,
    tolerance: ReconstructionTolerance,
    options: &ReconstructionOptions,
) -> Result<ReconstructedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let indices = validate(target, center, tolerance, options)?;
    let allowance = finite(
        tolerance
            .atol
            .max(finite(tolerance.rtol * target.reference_norm())?),
    )?;
    let originals = target.materialize_terms(center)?;
    let projector = Projector::new();
    let reduced = reduce(&originals, &projector, center, allowance)?;
    let mut stack = vec![WorkingRegion {
        projector,
        source: originals,
        reduced,
        allowance,
    }];
    let mut regions = Vec::new();
    let mut error = 0.0_f64;
    let mut merges = 0usize;
    let mut splits = 0usize;
    let mut live_regions = 1usize;

    while let Some(region) = stack.pop() {
        let rank = max_rank(&region.reduced.terms);
        let over_goal = options.target_bond_dim.is_some_and(|goal| rank > goal);
        let mut best: Option<(usize, Vec<WorkingRegion<V>>)> = None;
        if over_goal {
            let candidate_limit = match options.split_strategy {
                PatchSplitStrategy::Sequential => 1,
                PatchSplitStrategy::ExactParameterGain => usize::MAX,
            };
            // Limit before feasibility/gain checks so Sequential never bypasses
            // an unprofitable or unaffordable prefix bit to try a later bit.
            for index in indices
                .iter()
                .filter(|index| !region.projector.is_projected_at(index) && index.dim > 1)
                .take(candidate_limit)
            {
                let Some(prospective) = live_regions.checked_add(index.dim - 1) else {
                    continue;
                };
                if prospective > options.max_regions {
                    continue;
                }
                // INVARIANT: only one best candidate and the current candidate
                // are retained; fanout is bounded by max_regions before allocation.
                let mut children = Vec::with_capacity(index.dim);
                let mut child_rank = 0;
                let mut child_count = 0usize;
                // Conservative l1 allocation also satisfies the l2 allowance
                // for disjoint children and avoids a roundoff-boundary overspend.
                let child_allowance = region.allowance / index.dim as f64;
                for coordinate in 0..index.dim {
                    let mut projector = region.projector.clone();
                    projector.insert(index.clone(), coordinate)?;
                    let mut source = Vec::with_capacity(region.source.len());
                    for original in &region.source {
                        if let Some(child) = original.project(&projector)? {
                            source.push(child);
                        }
                    }
                    let reduced = reduce(&source, &projector, center, child_allowance)?;
                    child_rank = child_rank.max(max_rank(&reduced.terms));
                    child_count = checked_add(child_count, parameter_count(&reduced.terms)?)?;
                    children.push(WorkingRegion {
                        projector,
                        source,
                        reduced,
                        allowance: child_allowance,
                    });
                }
                if child_rank < rank && best.as_ref().is_none_or(|(count, _)| child_count < *count)
                {
                    best = Some((child_count, children));
                }
            }
        }
        if let Some((_, children)) = best {
            live_regions = checked_add(live_regions, children.len() - 1)?;
            splits = checked_add(splits, 1)?;
            stack.extend(children.into_iter().rev());
        } else {
            error = finite(error.hypot(region.reduced.error))?;
            merges = checked_add(merges, region.reduced.merges)?;
            if !region.reduced.terms.is_empty() {
                regions.push(Region {
                    projector: region.projector,
                    terms: region.reduced.terms,
                });
            }
        }
    }
    let term_count = regions
        .iter()
        .try_fold(0, |n, region| checked_add(n, region.terms.len()))?;
    let max_bond_dim = regions
        .iter()
        .map(|r| max_rank(&r.terms))
        .max()
        .unwrap_or(0);
    if error > allowance {
        return Err(invalid(
            "measured reconstruction error exceeds the global allowance",
        ));
    }
    let report = ReconstructionReport {
        reference_norm: target.reference_norm(),
        absolute_tolerance: allowance,
        error_bound: error,
        region_count: regions.len(),
        term_count,
        max_bond_dim,
        split_count: splits,
        merge_count: merges,
    };
    Ok(ReconstructedTreeTN { regions, report })
}

fn validate<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    target: &ReconstructionTarget<V>,
    center: &V,
    tolerance: ReconstructionTolerance,
    options: &ReconstructionOptions,
) -> Result<Vec<DynIndex>> {
    if !tolerance.rtol.is_finite()
        || tolerance.rtol < 0.0
        || !tolerance.atol.is_finite()
        || tolerance.atol < 0.0
    {
        return Err(invalid("rtol and atol must be finite and nonnegative"));
    }
    if options.target_bond_dim == Some(0) || options.max_regions == 0 {
        return Err(invalid("target_bond_dim and max_regions must be positive"));
    }
    let mut all = Vec::new();
    if let Some(network) = &target.network {
        if network.node_index(center).is_none() {
            return Err(PartitionedTreeTNError::InvalidCenter);
        }
        for node in network.node_names() {
            if let Some(indices) = network.site_space(node) {
                all.extend(indices.iter().cloned());
            }
        }
    }
    all.sort_by(canonical_index_cmp);
    let mut seen = HashSet::new();
    for requested in &options.patch_order {
        if !seen.insert(requested) {
            return Err(invalid(
                "patch_order must not contain duplicate full indices",
            ));
        }
        let canonical = all
            .iter()
            .find(|index| *index == requested)
            .ok_or_else(|| invalid("patch_order must belong to the target site space"))?;
        if canonical.dim != requested.dim {
            return Err(PartitionedTreeTNError::SiteIndexMismatch);
        }
    }
    Ok(if options.patch_order.is_empty() {
        all
    } else {
        options.patch_order.clone()
    })
}

fn reduce<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    originals: &[SubDomainTreeTN<V>],
    region: &Projector,
    center: &V,
    allowance: f64,
) -> Result<Reduced<V>> {
    // At most M initial compressions and M-1 successful pair compressions.
    // Each measured local residual gets the same globally derived allowance.
    let slots = originals
        .len()
        .checked_mul(2)
        .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)?;
    let share = if slots == 0 {
        0.0
    } else {
        allowance / slots as f64
    };
    let mut current = Vec::with_capacity(originals.len());
    let mut error = 0.0;
    let mut ordered: Vec<_> = originals.iter().collect();
    ordered.sort_by(|a, b| {
        b.projector()
            .len()
            .cmp(&a.projector().len())
            .then_with(|| a.projector().canonical_cmp(b.projector()))
    });
    for source in ordered {
        // INVARIANT: source has already been restricted to this region. Forget
        // the finer input support only in metadata; the data remains masked.
        let source =
            SubDomainTreeTN::from_masked_data(source.data().clone(), region.clone(), None)?;
        let (term, residual) = compress(&source, center, share)?;
        error = finite(error + residual)?;
        if let Some(term) = term {
            current.push(term);
        }
    }
    let mut terminal = Vec::new();
    let mut merges = 0;
    while current.len() > 1 {
        let mut next = Vec::with_capacity(current.len().div_ceil(2));
        let mut iter = current.into_iter();
        while let Some(left) = iter.next() {
            let Some(right) = iter.next() else {
                next.push(left);
                break;
            };
            let original_rank = checked_add(left.max_bond_dim(), right.max_bond_dim())?;
            let sum = left.add(&right)?;
            let (candidate, residual) = compress(&sum, center, share)?;
            if candidate
                .as_ref()
                .is_none_or(|term| term.max_bond_dim() < original_rank)
            {
                error = finite(error + residual)?;
                merges = checked_add(merges, 1)?;
                if let Some(term) = candidate {
                    next.push(term);
                }
            } else {
                terminal.push(left);
                terminal.push(right);
            }
        }
        current = next;
    }
    terminal.extend(current);
    Ok(Reduced {
        terms: terminal,
        error,
        merges,
    })
}

fn compress<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    source: &SubDomainTreeTN<V>,
    center: &V,
    allowance: f64,
) -> Result<(Option<SubDomainTreeTN<V>>, f64)> {
    let norm = finite(source.norm()?)?;
    if norm <= allowance {
        return Ok((None, norm));
    }
    if allowance == 0.0 || source.max_bond_dim() == 1 {
        return Ok((Some(source.clone()), 0.0));
    }
    let edges = source.node_count() - 1;
    let local = allowance / (2.0 * edges as f64);
    let cutoff = local * local;
    if !cutoff.is_finite() {
        return Ok((Some(source.clone()), 0.0));
    }
    let policy = SvdTruncationPolicy::new(cutoff)
        .with_absolute()
        .with_squared_values()
        .with_discarded_tail_sum();
    let mut candidate = source.clone();
    candidate.truncate(center, TruncationOptions::default().with_svd_policy(policy))?;
    // Restore exact support zeros even if factorization introduced roundoff.
    let candidate = SubDomainTreeTN::new(candidate.into_data(), source.projector().clone())?;
    let mut difference = source.data().axpby(1.0, candidate.data(), -1.0)?;
    let residual = finite(difference.norm()?)?;
    if residual <= allowance {
        Ok((Some(candidate), residual))
    } else {
        Ok((Some(source.clone()), 0.0))
    }
}

fn max_rank<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    terms: &[SubDomainTreeTN<V>],
) -> usize {
    terms
        .iter()
        .map(SubDomainTreeTN::max_bond_dim)
        .max()
        .unwrap_or(0)
}

fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)
}

fn parameter_count<V: Clone + Hash + Eq + Ord + Send + Sync + Debug>(
    terms: &[SubDomainTreeTN<V>],
) -> Result<usize> {
    terms.iter().try_fold(0, |total, term| {
        checked_add(total, crate::patching::logical_parameter_count(term)?)
    })
}
