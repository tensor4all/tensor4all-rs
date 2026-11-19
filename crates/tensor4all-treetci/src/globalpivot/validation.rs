//! Historical-pool and current-pivot validation for one convergence confirmation.

use super::confirmation::{retained_pivot_edges, RetainedPivotEdge};
use super::{read_batch_errors, ScalarParts, SessionReadout};
use crate::error::Result as TreeTciResult;
use crate::evaluation::BatchExecution;
use crate::{batch::EVALUATION_CHUNK_POINTS, GlobalIndexBatch, MultiIndex, TreeTCI2};
use anyhow::Result;
use std::collections::HashSet;
use tensor4all_core::index_key::FlatIndexer;
use tensor4all_core::{AnyScalar, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::{TreeTN, TreeTNCachedEvaluator};

/// Largest validation residual and borrowed pivot metadata reused by discovery.
pub(super) struct SampleValidation<'a> {
    pub(super) max_error: f64,
    pub(super) point: MultiIndex,
    pub(super) retained: Vec<RetainedPivotEdge<'a>>,
}

// INVARIANT: one validation stage borrows the search's execution, approximation,
// cache, readout and scratch owners; it creates no second evaluator or session.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_samples<'a, T, R>(
    state: &'a TreeTCI2<T>,
    execution: &mut dyn BatchExecution<T>,
    cache: &mut TreeTNCachedEvaluator<'_, usize>,
    treetn: &TreeTN<IdxTensor, usize>,
    site_indices: &[DynIndex],
    flat: &mut Vec<usize>,
    readout: &mut R,
    readout_on_session: SessionReadout<T>,
) -> TreeTciResult<SampleValidation<'a>>
where
    T: FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts + Copy,
    R: FnMut(
        &mut TreeTNCachedEvaluator<'_, usize>,
        &TreeTN<IdxTensor, usize>,
        &[DynIndex],
        ColMajorArrayRef<'_, usize>,
        Option<usize>,
    ) -> Result<Vec<AnyScalar>>,
{
    let n_sites = state.local_dims.len();
    // Replay the bounded sample pool, including its protected residual witness.
    // Oracle traffic cannot evict the old witness before this replay. No oracle
    // requests occur inside the loop, so its slot order remains unchanged.
    execution.begin_validation();
    let known_count = execution.validation_len();
    let mut expected = Vec::new();
    let mut known_error = 0.0_f64;
    let mut known_point = Vec::new();
    for offset in (0..known_count).step_by(EVALUATION_CHUNK_POINTS) {
        let count =
            execution.validation_batch(offset, EVALUATION_CHUNK_POINTS, flat, &mut expected);
        let shape = [n_sites, count];
        let points = ColMajorArrayRef::new(flat, &shape).map_err(anyhow::Error::new)?;
        let errors = read_batch_errors(
            execution,
            cache,
            treetn,
            site_indices,
            points,
            None,
            &expected,
            readout,
            readout_on_session,
        )?;
        execution.observe_validation_residuals(
            GlobalIndexBatch::new(flat, n_sites, count)?,
            &expected,
            &errors,
        )?;
        update_max_sample(flat, n_sites, &errors, &mut known_error, &mut known_point);
    }
    let validation_edges = retained_pivot_edges(state)?;
    {
        let retained = &validation_edges;
        // Current pivot pairs must be checked even when history is tiny, zero,
        // or has evicted their old values. Pairs already replayed from the pool
        // need no second read. Batch missing distinct points through the same
        // execution owner and cached approximation as every other search read.
        let indexer = FlatIndexer::try_new(&state.local_dims).map_err(anyhow::Error::new)?;
        let mut seen = HashSet::new();
        let mut point = vec![0; n_sites];
        flat.clear();
        for anchor in retained {
            for pair in (0..anchor.rank).rev() {
                anchor.write_point(pair, &mut point)?;
                if !seen.insert(indexer.encode(&point).map_err(anyhow::Error::new)?)
                    || execution.validation_contains(&point)?
                {
                    continue;
                }
                flat.extend_from_slice(&point);
                if flat.len() / n_sites == EVALUATION_CHUNK_POINTS {
                    validate_retained_batch(
                        execution,
                        cache,
                        treetn,
                        site_indices,
                        flat,
                        &mut known_error,
                        &mut known_point,
                        readout,
                        readout_on_session,
                    )?;
                    flat.clear();
                }
            }
        }
        if !flat.is_empty() {
            validate_retained_batch(
                execution,
                cache,
                treetn,
                site_indices,
                flat,
                &mut known_error,
                &mut known_point,
                readout,
                readout_on_session,
            )?;
        }
    }
    Ok(SampleValidation {
        max_error: known_error,
        point: known_point,
        retained: validation_edges,
    })
}

/// Update a search's strongest observed residual using a bounded point batch.
fn update_max_sample(
    flat: &[usize],
    n_sites: usize,
    errors: &[f64],
    max_error: &mut f64,
    point: &mut MultiIndex,
) {
    for (candidate, &error) in flat.chunks_exact(n_sites).zip(errors) {
        if error > *max_error {
            *max_error = error;
            point.clear();
            point.extend_from_slice(candidate);
        }
    }
}

// INVARIANT: one current-pivot batch borrows the same search-scoped
// execution/cache/readout owners as historical replay and coordinate walks.
#[allow(clippy::too_many_arguments)]
fn validate_retained_batch<T, R>(
    execution: &mut dyn BatchExecution<T>,
    cache: &mut TreeTNCachedEvaluator<'_, usize>,
    treetn: &TreeTN<IdxTensor, usize>,
    site_indices: &[DynIndex],
    flat: &[usize],
    max_error: &mut f64,
    point: &mut MultiIndex,
    readout: &mut R,
    readout_on_session: SessionReadout<T>,
) -> Result<()>
where
    T: FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts + Copy,
    R: FnMut(
        &mut TreeTNCachedEvaluator<'_, usize>,
        &TreeTN<IdxTensor, usize>,
        &[DynIndex],
        ColMajorArrayRef<'_, usize>,
        Option<usize>,
    ) -> Result<Vec<AnyScalar>>,
{
    let n_sites = site_indices.len();
    let count = flat.len() / n_sites;
    let values = execution.call(GlobalIndexBatch::new(flat, n_sites, count)?)?;
    anyhow::ensure!(
        values.len() == count,
        "target returned {} values for {count} retained pivots",
        values.len()
    );
    let errors = read_batch_errors(
        execution,
        cache,
        treetn,
        site_indices,
        ColMajorArrayRef::new(flat, &[n_sites, count])?,
        None,
        &values,
        readout,
        readout_on_session,
    )?;
    execution.observe_validation_residuals(
        GlobalIndexBatch::new(flat, n_sites, count)?,
        &values,
        &errors,
    )?;
    update_max_sample(flat, n_sites, &errors, max_error, point);
    Ok(())
}

#[cfg(test)]
mod tests;
