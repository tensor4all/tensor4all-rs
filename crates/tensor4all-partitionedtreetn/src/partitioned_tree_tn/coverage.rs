//! Coverage metadata and set algebra for [`PartitionedTreeTN`](super::PartitionedTreeTN).
//!
//! A partition's optional coverage is a canonicalized list of [`Projector`]s
//! describing the region on which the partition is defined:
//! - inside the coverage, an absent patch is an exact zero;
//! - outside the coverage, the value is unknown (uncomputed);
//! - `None` means the partition is defined everywhere.

use std::collections::{HashMap, HashSet};

use tensor4all_core::DynIndex;

use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::Projector;

/// Canonicalize a list of coverage projectors: deterministic order, no duplicates.
pub(crate) fn canonicalize_coverage(mut coverage: Vec<Projector>) -> Vec<Projector> {
    coverage.sort_by(Projector::canonical_cmp);
    coverage.dedup();
    coverage
}

/// Return the first canonically ordered sub-projector of `target` not covered
/// by the union of `coverage`, or `None` when `target` is completely covered.
pub(crate) fn first_uncovered_in(target: &Projector, coverage: &[Projector]) -> Option<Projector> {
    let mut pending = vec![target.clone()];
    for cov in coverage {
        let mut next = Vec::new();
        for piece in pending {
            match piece.split_by(cov) {
                Some(cells) => next.extend(cells.into_iter().skip(1)),
                None => next.push(piece),
            }
        }
        pending = next;
        if pending.is_empty() {
            return None;
        }
    }
    pending.sort_by(Projector::canonical_cmp);
    pending.into_iter().next()
}

/// Return whether `target` is completely covered by the union of `coverage`.
pub(crate) fn projector_covered_by(target: &Projector, coverage: &[Projector]) -> bool {
    first_uncovered_in(target, coverage).is_none()
}

/// Decompose the union of `coverage` into pairwise-disjoint projectors in
/// canonical order.
pub(crate) fn disjoint_cover(coverage: &[Projector]) -> Vec<Projector> {
    let mut cells: Vec<Projector> = Vec::new();
    for cov in coverage {
        let mut pending = vec![cov.clone()];
        for cell in &cells {
            pending = pending
                .into_iter()
                .flat_map(|piece| match piece.split_by(cell) {
                    Some(pieces) => pieces.into_iter().skip(1).collect(),
                    None => vec![piece],
                })
                .collect();
        }
        cells.extend(pending);
    }
    cells.sort_by(Projector::canonical_cmp);
    cells
}

/// Ensure `target` lies completely inside `coverage` (`None` covers everything).
pub(crate) fn ensure_covered(target: &Projector, coverage: Option<&[Projector]>) -> Result<()> {
    let Some(cov) = coverage else {
        return Ok(());
    };
    if let Some(offending) = first_uncovered_in(target, cov) {
        return Err(PartitionedTreeTNError::OutsideCoverage {
            projector: offending,
        });
    }
    Ok(())
}

/// Intersect two optional coverages.
///
/// `None` represents the whole domain, so `None ∩ Some(c) = Some(c)` and
/// `None ∩ None = None`.
pub(crate) fn intersect_coverages(
    left: Option<&[Projector]>,
    right: Option<&[Projector]>,
) -> Option<Vec<Projector>> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(canonicalize_coverage(l.to_vec())),
        (None, Some(r)) => Some(canonicalize_coverage(r.to_vec())),
        (Some(l), Some(r)) => {
            let mut out = Vec::new();
            for p in l {
                for q in r {
                    if let Some(inter) = p.intersection(q) {
                        out.push(inter);
                    }
                }
            }
            Some(canonicalize_coverage(out))
        }
    }
}

/// Refine an optional coverage onto a layout by intersecting with `layout`.
pub(crate) fn refine_coverage(
    coverage: Option<&[Projector]>,
    layout: &[Projector],
) -> Option<Vec<Projector>> {
    let cov = coverage?;
    intersect_coverages(Some(cov), Some(layout))
}

/// Derive the output coverage of contracting two partitions with optional
/// coverages, given the contraction's uncontracted `output_sites`.
///
/// Conservative rule: if either operand's coverage contains a projector that
/// constrains any contracted index (an index absent from `output_sites`),
/// returns [`PartitionedTreeTNError::OutsideCoverage`] naming the first such
/// projector in canonical order. Otherwise every coverage projector lives on
/// `output_sites`, and the output coverage is their intersection.
pub(crate) fn derive_contraction_coverage(
    left: Option<&[Projector]>,
    right: Option<&[Projector]>,
    output_sites: &[DynIndex],
) -> Result<Option<Vec<Projector>>> {
    let output_set: HashSet<&DynIndex> = output_sites.iter().collect();
    for cov in [left, right].into_iter().flatten() {
        for projector in cov {
            if projector
                .projected_indices()
                .any(|index| !output_set.contains(index))
            {
                return Err(PartitionedTreeTNError::OutsideCoverage {
                    projector: projector.clone(),
                });
            }
        }
    }
    Ok(intersect_coverages(left, right))
}

/// Validate a coverage list against a partition's site space and stored patches.
pub(crate) fn validate_coverage(
    coverage: &[Projector],
    site_indices: Option<&[DynIndex]>,
    stored_projectors: impl Iterator<Item = impl std::borrow::Borrow<Projector>>,
) -> Result<()> {
    if let Some(sites) = site_indices {
        let dims: HashMap<&DynIndex, usize> = sites.iter().map(|idx| (idx, idx.dim)).collect();
        for projector in coverage {
            for (index, &value) in projector.iter() {
                let Some(&dim) = dims.get(index) else {
                    return Err(PartitionedTreeTNError::ProjectorIndexNotFound {
                        index: index.clone(),
                    });
                };
                if dim != index.dim {
                    return Err(PartitionedTreeTNError::SiteIndexMismatch);
                }
                if value >= dim {
                    return Err(PartitionedTreeTNError::ProjectorCoordinateOutOfBounds {
                        index: index.clone(),
                        value,
                        dim,
                    });
                }
            }
        }
    }

    for stored in stored_projectors {
        ensure_covered(stored.borrow(), Some(coverage))?;
    }
    Ok(())
}
