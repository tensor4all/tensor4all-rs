use crate::{
    assemble::{assemble_points_column_major, MultiIndex},
    assemble_global_point,
    batch::GlobalIndexBatch,
    DefaultProposer, PivotCandidateProposer, TreeTCI2, TreeTciEdge,
};
use anyhow::{ensure, Result};
use tensor4all_core::ColMajorArray;
use tensor4all_tcicore::{
    DenseLuKernel, DenseMatrixSource, MatrixLuciScalar as Scalar, PivotKernel, PivotKernelOptions,
    PivotSelectionCore,
};

/// Update one edge bipartition using a batch evaluator and a pivot-candidate proposer.
///
/// Evaluates the function at candidate pivot points for one edge bipartition,
/// then selects pivots via LU decomposition. The selected pivots are stored
/// back into `state.ijset`.
///
/// This is a low-level building block; prefer [`optimize_default`](crate::optimize_default)
/// or [`crossinterpolate2`](crate::crossinterpolate2) for typical usage.
pub fn update_edge<T, F, P>(
    state: &mut TreeTCI2<T>,
    edge: TreeTciEdge,
    evaluate: F,
    options: &PivotKernelOptions,
    proposer: &P,
) -> Result<PivotSelectionCore>
where
    T: Scalar,
    DenseLuKernel: PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    let (left_key, right_key) = state.graph.subregion_vertices(edge)?;
    let (left_candidates, right_candidates) = proposer.candidates(state, edge)?;
    let values = evaluate_candidate_matrix(
        state.local_dims.len(),
        &left_key,
        &left_candidates,
        &right_key,
        &right_candidates,
        evaluate,
    )?;

    for value in &values {
        state.max_sample_value = state.max_sample_value.max(value.abs_val());
    }

    let source = DenseMatrixSource::from_column_major(
        &values,
        left_candidates.len(),
        right_candidates.len(),
    );
    let selection = DenseLuKernel.factorize(&source, options)?;

    // Build ColMajorArray from selected pivot indices
    let n_left_sites = left_key.as_slice().len();
    let n_right_sites = right_key.as_slice().len();

    let left_data: Vec<usize> = selection
        .row_indices
        .iter()
        .flat_map(|&row| left_candidates[row].iter().copied())
        .collect();
    let left_arr = ColMajorArray::new(left_data, vec![n_left_sites, selection.row_indices.len()])?;
    state.ijset.insert(left_key.clone(), left_arr);

    let right_data: Vec<usize> = selection
        .col_indices
        .iter()
        .flat_map(|&col| right_candidates[col].iter().copied())
        .collect();
    let right_arr =
        ColMajorArray::new(right_data, vec![n_right_sites, selection.col_indices.len()])?;
    state.ijset.insert(right_key.clone(), right_arr);

    let last_error = selection.pivot_errors.last().copied().unwrap_or(0.0);
    state.update_bond_error(edge, last_error);
    state.update_pivot_errors(&selection.pivot_errors);

    Ok(selection)
}

/// Update one edge using the default proposer.
///
/// Convenience wrapper around [`update_edge`] that uses [`DefaultProposer`].
pub fn update_edge_default<T, F>(
    state: &mut TreeTCI2<T>,
    edge: TreeTciEdge,
    evaluate: F,
    options: &PivotKernelOptions,
) -> Result<PivotSelectionCore>
where
    T: Scalar,
    DenseLuKernel: PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    update_edge(state, edge, evaluate, options, &DefaultProposer)
}

fn evaluate_candidate_matrix<T, F>(
    n_sites: usize,
    left_key: &crate::SubtreeKey,
    left_candidates: &[MultiIndex],
    right_key: &crate::SubtreeKey,
    right_candidates: &[MultiIndex],
    evaluate: F,
) -> Result<Vec<T>>
where
    T: Scalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    let mut points = Vec::with_capacity(left_candidates.len() * right_candidates.len());
    for right in right_candidates {
        for left in left_candidates {
            points.push(assemble_global_point(
                n_sites,
                &[(left_key, left), (right_key, right)],
                &[],
            )?);
        }
    }
    let batch = assemble_points_column_major(&points)?;
    let values = evaluate(batch.as_view())?;
    ensure!(
        values.len() == left_candidates.len() * right_candidates.len(),
        "batch evaluator returned {} values for {} candidate-matrix entries",
        values.len(),
        left_candidates.len() * right_candidates.len()
    );
    Ok(values)
}

#[cfg(test)]
mod tests;
