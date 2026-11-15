//! Bounded spectator geometry, prepared before any tensor operation.

use super::*;
use crate::partitioned_tree_tn::{bounded_value_grid, fixed_projector_indices};
use crate::projector::index::ProjectorIndex;
use std::ops::ControlFlow;

/// Form disjoint spectator cells and validate selected coverage in each cell.
pub(super) fn prepare<'a>(
    projectors: impl Iterator<Item = &'a Projector>,
    geometry: &mut ScheduleGeometry,
    coverage: CoverageContract,
    max_work_items: usize,
    max_terms: usize,
) -> Result<usize> {
    let projectors: Vec<_> = projectors.collect();
    if projectors.is_empty() {
        validate_leaf_geometry(projectors.into_iter(), geometry, coverage)?;
        return Ok(0);
    }
    let source_spectators: Vec<_> = projectors
        .iter()
        .map(|projector| geometry.spectator_part(projector))
        .collect();
    let references: Vec<_> = source_spectators.iter().collect();
    let sources = ProjectorIndex::new(&references);
    let mut cells = source_spectators.clone();
    cells.sort_by(Projector::canonical_cmp);
    cells.dedup();
    if !Projector::are_disjoint(&cells) {
        let common = cells.iter().skip(1).fold(cells[0].clone(), |common, cell| {
            common.common_restriction(cell)
        });
        let varying: Vec<_> = fixed_projector_indices(&cells)
            .into_iter()
            .filter(|index| !common.is_projected_at(index))
            .collect();
        // Check the Cartesian metadata grid before enumerating it. The tensor
        // images are not split until both the grid and leaf counts are accepted.
        cells = bounded_value_grid(&varying, max_work_items.min(max_terms))?
            .into_iter()
            .filter_map(|cell| cell.intersection(&common))
            .filter(|cell| {
                sources
                    .for_each_overlapping(cell, |_| ControlFlow::Break(()))
                    .is_break()
            })
            .collect();
        cells.sort_by(Projector::canonical_cmp);
    }

    let mut count = 0usize;
    let mut source_groups = vec![Vec::new(); projectors.len()];
    // Query only spectator metadata: indexing selected prefixes too would make
    // cell queries traverse irrelevant selected branches. Memberships are bounded
    // by the refined leaf budget and reused when images are initialized, avoiding
    // a source-by-cell Cartesian scan at either stage.
    for (group, cell) in cells.iter().enumerate() {
        let members = sources.overlapping(cell);
        let leaves = validate_leaf_geometry(
            members.iter().map(|&member| projectors[member]),
            geometry,
            coverage,
        )?;
        count = checked_add(count, leaves.len())?;
        for (limit, maximum) in [("max_work_items", max_work_items), ("max_terms", max_terms)] {
            if count > maximum {
                return Err(PartitionedTreeTNError::ResourceLimit {
                    operation: "merge-refine spectator refinement",
                    limit,
                    value: count,
                });
            }
        }
        for member in members {
            source_groups[member].push(group);
        }
    }
    geometry.spectators = cells;
    geometry.source_groups = source_groups;
    Ok(count)
}

#[cfg(test)]
mod tests;
