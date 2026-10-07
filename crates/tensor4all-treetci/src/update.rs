use crate::{
    assemble::MultiIndex,
    batch::{evaluate_points_chunked, GlobalIndexBatch, EVALUATION_CHUNK_POINTS},
    PivotCandidateProposer, TreeTCI2, TreeTciEdge,
};
use anyhow::{ensure, Result};
use tensor4all_core::{
    matrix_luci_factors_from_matrix, MatrixLuciFactors, MatrixLuciScalar as Scalar, RrLUOptions,
};
use tensor4all_core::{ColMajorArray, CommonScalar};
use tensor4all_tensorbackend::Matrix;

#[cfg(test)]
use crate::DefaultProposer;

/// Update one edge bipartition using a batch evaluator and a pivot-candidate proposer.
///
/// Evaluates the function at candidate pivot points for one edge bipartition,
/// then selects pivots via LU decomposition. The selected pivots are stored
/// back into `state.ijset`.
///
/// This is a low-level building block; prefer [`optimize_default`](crate::optimize_default)
/// or [`crossinterpolate2`](crate::crossinterpolate2) for typical usage.
pub(crate) fn update_edge<T, F, P, R>(
    state: &mut TreeTCI2<T>,
    edge: TreeTciEdge,
    evaluate: F,
    options: &RrLUOptions,
    proposer: &P,
    rng: &mut R,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
    R: rand::Rng + ?Sized,
{
    let (left_key, right_key) = state.graph.subregion_vertices(edge)?;
    let (left_candidates, right_candidates) = proposer.candidates_with_rng(state, edge, rng)?;
    if left_candidates.is_empty() || right_candidates.is_empty() {
        return Err(anyhow::anyhow!(
            "proposer returned empty candidate list for edge {edge:?}",
        ));
    }
    let values = evaluate_candidate_matrix(
        state.local_dims.len(),
        &left_key,
        &left_candidates,
        &right_key,
        &right_candidates,
        &state.local_dims,
        &evaluate,
        EVALUATION_CHUNK_POINTS,
    )?;

    for value in &values {
        state.max_sample_value = state.max_sample_value.max(CommonScalar::abs_val(*value));
    }

    // `evaluate_candidate_matrix` already returns column-major data with
    // `left_candidates.len()` rows, so this hands over the buffer as-is.
    let matrix = Matrix::from_col_major_vec(left_candidates.len(), right_candidates.len(), values);
    let selection = matrix_luci_factors_from_matrix(&matrix, Some(options.clone()))?;

    // The LU can select zero pivots when the sampled submatrix is numerically
    // zero (the function underflows in a subdomain). Keep at least one index
    // so the stored pivot sets never become empty and the state stays
    // materializable; the resulting rank-1 zero core is the correct
    // approximation of a zero subdomain. Candidate lists are validated
    // non-empty above.
    let row_indices = if selection.row_indices.is_empty() {
        vec![0]
    } else {
        selection.row_indices.clone()
    };
    let col_indices = if selection.col_indices.is_empty() {
        vec![0]
    } else {
        selection.col_indices.clone()
    };

    // Build ColMajorArray from selected pivot indices
    let n_left_sites = left_key.as_slice().len();
    let n_right_sites = right_key.as_slice().len();

    let left_len = n_left_sites
        .checked_mul(row_indices.len())
        .ok_or_else(|| anyhow::anyhow!("selected left pivot data size overflowed usize"))?;
    let mut left_data = Vec::with_capacity(left_len);
    for &row in &row_indices {
        let candidate = left_candidates
            .get(row)
            .ok_or_else(|| anyhow::anyhow!("selected left pivot row {row} is out of bounds"))?;
        left_data.extend_from_slice(candidate);
    }
    let left_arr = ColMajorArray::new(left_data, vec![n_left_sites, row_indices.len()])?;
    state.ijset.insert(left_key.clone(), left_arr);

    let right_len = n_right_sites
        .checked_mul(col_indices.len())
        .ok_or_else(|| anyhow::anyhow!("selected right pivot data size overflowed usize"))?;
    let mut right_data = Vec::with_capacity(right_len);
    for &col in &col_indices {
        let candidate = right_candidates
            .get(col)
            .ok_or_else(|| anyhow::anyhow!("selected right pivot column {col} is out of bounds"))?;
        right_data.extend_from_slice(candidate);
    }
    let right_arr = ColMajorArray::new(right_data, vec![n_right_sites, col_indices.len()])?;
    state.ijset.insert(right_key.clone(), right_arr);

    let last_error = selection.pivot_errors.last().copied().unwrap_or(0.0);
    state.update_bond_error(edge, last_error);
    state.update_pivot_errors(&selection.pivot_errors);

    Ok(selection)
}

/// Update one edge using the default proposer.
///
/// Convenience wrapper around [`update_edge`] that uses [`DefaultProposer`].
#[cfg(test)]
pub(crate) fn update_edge_default<T, F>(
    state: &mut TreeTCI2<T>,
    edge: TreeTciEdge,
    evaluate: F,
    options: &RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    update_edge(state, edge, evaluate, options, &DefaultProposer, &mut rng)
}

/// Evaluate the function on the full `I x J` candidate matrix for one edge.
///
/// Returns the values in column-major order with `left_candidates.len()` rows.
///
/// The global points are written straight into a contiguous batch buffer.
/// Assembling them as individual `Vec<usize>` points instead and packing
/// those into a batch costs one heap allocation and one extra full copy per
/// matrix *entry*, and re-validating per point checks the
/// bipartition `n_left * n_right` times over -- at a branching vertex that is
/// O(10^7) allocations for a single edge update. The bipartition is a property
/// of the two subtree keys, so it is checked once up front instead.
///
/// The points are evaluated in calls of at most `chunk_points` points through
/// one reused buffer, so memory stays bounded by `n_sites * chunk_points`
/// rather than `n_sites * n_left * n_right` (gigabytes at a branching vertex).
/// The values, and their order, do not depend on `chunk_points`.
#[allow(clippy::too_many_arguments)]
fn evaluate_candidate_matrix<T, F>(
    n_sites: usize,
    left_key: &crate::SubtreeKey,
    left_candidates: &[MultiIndex],
    right_key: &crate::SubtreeKey,
    right_candidates: &[MultiIndex],
    local_dims: &[usize],
    evaluate: &F,
    chunk_points: usize,
) -> Result<Vec<T>>
where
    T: Scalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    let left_sites = left_key.as_slice();
    let right_sites = right_key.as_slice();

    let mut assigned = vec![false; n_sites];
    for &site in left_sites.iter().chain(right_sites.iter()) {
        ensure!(
            site < n_sites,
            "site {} is out of bounds for {} sites",
            site,
            n_sites
        );
        ensure!(!assigned[site], "site {} was assigned more than once", site);
        assigned[site] = true;
    }
    ensure!(
        assigned.iter().all(|&seen| seen),
        "global point assembly left some sites unassigned"
    );

    // Every candidate must match its own side's subtree key length, and every
    // coordinate must lie within the site's local dimension. The two sides are
    // validated separately (rather than inferring the side from the candidate
    // length): a malformed candidate whose length coincides with the *other*
    // side's key would otherwise pass validation and then silently leave part
    // of its point at zero in the fill loop below. Coordinates are validated
    // at this public boundary so a caller-supplied proposer cannot submit
    // out-of-domain indices.
    for (side, (candidates, sites)) in [
        ("left", (left_candidates, left_sites)),
        ("right", (right_candidates, right_sites)),
    ] {
        for candidate in candidates {
            ensure!(
                candidate.len() == sites.len(),
                "subtree key of length {} cannot be filled from {side} multi-index of length {}",
                sites.len(),
                candidate.len()
            );
            for (&site, &value) in sites.iter().zip(candidate.iter()) {
                ensure!(
                    value < local_dims[site],
                    "{side} candidate value {value} out of range for site {site} with local dimension {}",
                    local_dims[site]
                );
            }
        }
    }

    let n_left = left_candidates.len();
    let n_right = right_candidates.len();
    let n_points = n_left.checked_mul(n_right).ok_or_else(|| {
        anyhow::anyhow!("candidate matrix shape {n_left} x {n_right} overflows usize")
    })?;

    // Column-major (n_sites, n_points): left candidates vary fastest.
    let mut left_index = 0;
    let mut right_index = 0;
    evaluate_points_chunked(
        n_sites,
        n_points,
        chunk_points,
        |point| {
            let left = &left_candidates[left_index];
            let right = &right_candidates[right_index];
            for (&site, &value) in left_sites.iter().zip(left.iter()) {
                point[site] = value;
            }
            for (&site, &value) in right_sites.iter().zip(right.iter()) {
                point[site] = value;
            }
            left_index += 1;
            if left_index == n_left {
                left_index = 0;
                right_index += 1;
            }
        },
        evaluate,
        "candidate-matrix entries",
    )
}

#[cfg(test)]
mod tests;
