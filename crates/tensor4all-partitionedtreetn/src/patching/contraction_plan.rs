//! Selection-preserving, bounded refinement of adaptive contraction tasks.
//!
//! This independently derived projector algebra extends the compatible-pair
//! sum of Grosso et al., Sec. 4.2 (<https://arxiv.org/abs/2602.22372>) to the
//! repository's selected-output semantics. The grid and its limit are software
//! policy, not a theorem or an accuracy guarantee from that paper.

use super::*;
use crate::partitioned_tree_tn::{bounded_value_grid, fixed_projector_indices};
use tensor4all_core::DynIndex;

type ContractGroup<V> = Vec<(SubDomainTreeTN<V>, SubDomainTreeTN<V>)>;
type ProjectionCache<V> = HashMap<(Projector, Projector), SubDomainTreeTN<V>>;

/// Select original tasks first, then refine overlapping outputs without adding
/// contributions from an unselected original output key.
pub(super) fn plan_groups<V>(
    left: &PartitionedTreeTN<V>,
    right: &PartitionedTreeTN<V>,
    center: &V,
    wanted: Option<&HashSet<Projector>>,
    limit: usize,
) -> Result<Vec<ContractGroup<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut plan = Vec::new();
    let mut outputs = Vec::new();
    let mut seen = HashSet::new();
    left.for_each_planned_pair(right, center, wanted, |(left, right, output)| {
        if seen.insert(output.clone()) {
            outputs.push(output.clone());
        }
        plan.push((left, right, output));
        Ok(())
    })?;

    // Accepted disjoint layouts keep their old grouping, tensor schedule and
    // split candidates, even if they contain more than `limit` output keys.
    // Only overlap resolution invokes the automatic refinement policy.
    let grid = if Projector::are_disjoint(&outputs) {
        None
    } else {
        Some(bounded_value_grid(
            &fixed_projector_indices(&outputs),
            limit,
        )?)
    };
    let sites = if grid.is_some() {
        plan.first()
            .map(|(a, b, _)| -> Result<_> {
                Ok((
                    a.data().all_site_indices()?.0,
                    b.data().all_site_indices()?.0,
                ))
            })
            .transpose()?
    } else {
        None
    };

    let mut left_cache = ProjectionCache::new();
    let mut right_cache = ProjectionCache::new();
    let mut groups = Vec::new();
    let mut positions = HashMap::new();
    // Outer iteration is the canonical original pair order. In particular,
    // summing a refined cell never changes to hash or old-group-major order.
    for (left, right, output) in plan {
        if let Some((grid, (left_sites, right_sites))) = grid.as_ref().zip(sites.as_ref()) {
            for atom in grid.iter().filter(|atom| atom.is_subset_of(&output)) {
                let a = project_cached(left, atom, left_sites, &mut left_cache)?;
                let b = project_cached(right, atom, right_sites, &mut right_cache)?;
                push_group(&mut groups, &mut positions, atom.clone(), (a, b));
            }
        } else {
            push_group(
                &mut groups,
                &mut positions,
                output,
                (left.clone(), right.clone()),
            );
        }
    }
    Ok(groups)
}

fn push_group<V>(
    groups: &mut Vec<ContractGroup<V>>,
    positions: &mut HashMap<Projector, usize>,
    output: Projector,
    pair: (SubDomainTreeTN<V>, SubDomainTreeTN<V>),
) where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let position = *positions.entry(output).or_insert(groups.len());
    if position == groups.len() {
        groups.push(Vec::new());
    }
    // INVARIANT: insertion records the old length immediately before a push;
    // every other position names an already initialized group.
    groups[position].push(pair);
}

fn project_cached<V>(
    patch: &SubDomainTreeTN<V>,
    output: &Projector,
    sites: &[DynIndex],
    cache: &mut ProjectionCache<V>,
) -> Result<SubDomainTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // Output axes may belong to only one operand. Keep original internal
    // restrictions and mask only the output coordinates that this side owns.
    let restriction = output.filter_indices(sites);
    let key = (patch.projector().clone(), restriction.clone());
    if let Some(projected) = cache.get(&key) {
        return Ok(projected.clone());
    }
    // INVARIANT: the partition has one patch per original projector. Each
    // output atom lies inside this pair's original output region, so projection
    // cannot conflict. Exact masking adds no truncation or norm threshold.
    let projected = patch
        .project(&restriction)?
        .ok_or(PartitionedTreeTNError::ProjectorConflict)?;
    cache.insert(key, projected.clone());
    Ok(projected)
}

#[cfg(test)]
mod tests;
