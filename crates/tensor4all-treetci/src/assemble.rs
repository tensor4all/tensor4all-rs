use crate::{OwnedGlobalIndexBatch, SubtreeKey};
use anyhow::{ensure, Result};

/// Multi-index type used by TreeTCI.
pub type MultiIndex = Vec<usize>;

/// Assemble one global site-order point from subtree-local assignments and central site values.
pub fn assemble_global_point(
    n_sites: usize,
    subtree_assignments: &[(&SubtreeKey, &MultiIndex)],
    central_assignments: &[(usize, usize)],
) -> Result<MultiIndex> {
    let mut point = vec![usize::MAX; n_sites];

    for &(key, values) in subtree_assignments {
        ensure!(
            key.as_slice().len() == values.len(),
            "subtree key of length {} cannot be filled from multi-index of length {}",
            key.as_slice().len(),
            values.len()
        );
        for (&site, &value) in key.as_slice().iter().zip(values.iter()) {
            ensure!(
                site < n_sites,
                "site {} is out of bounds for {} sites",
                site,
                n_sites
            );
            ensure!(
                point[site] == usize::MAX,
                "site {} was assigned more than once",
                site
            );
            point[site] = value;
        }
    }

    for &(site, value) in central_assignments {
        ensure!(
            site < n_sites,
            "site {} is out of bounds for {} sites",
            site,
            n_sites
        );
        ensure!(
            point[site] == usize::MAX,
            "site {} was assigned more than once",
            site
        );
        point[site] = value;
    }

    ensure!(
        point.iter().all(|&value| value != usize::MAX),
        "global point assembly left some sites unassigned"
    );
    Ok(point)
}

/// Pack global points into column-major `(n_sites, n_points)` storage.
pub fn assemble_points_column_major(points: &[MultiIndex]) -> Result<OwnedGlobalIndexBatch> {
    let n_points = points.len();
    let n_sites = points.first().map_or(0, Vec::len);
    ensure!(n_sites > 0, "at least one point with one site is required");
    ensure!(n_points > 0, "at least one point is required");
    ensure!(
        points.iter().all(|point| point.len() == n_sites),
        "all points must have the same site count"
    );

    let mut data = Vec::with_capacity(n_sites * n_points);
    for point in points {
        data.extend_from_slice(point);
    }
    OwnedGlobalIndexBatch::new(data, n_sites, n_points)
}

#[cfg(test)]
mod tests;
