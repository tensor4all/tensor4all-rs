//! Opt-in convergence confirmation, sharing the ordinary search's readout/walk.
use super::*;
use std::collections::HashSet;
use tensor4all_core::index_key::FlatIndexer;

/// Borrowed pivot tables and topology for one edge confirmation.
#[derive(Debug)]
pub(super) struct RetainedPivotEdge<'a> {
    pub(super) left_sites: SubtreeKey,
    pub(super) right_sites: SubtreeKey,
    pub(super) left: &'a tensor4all_core::ColMajorArray<usize>,
    pub(super) right: &'a tensor4all_core::ColMajorArray<usize>,
    pub(super) rank: usize,
    pub(super) unit_endpoint: bool,
}

impl RetainedPivotEdge<'_> {
    /// Assemble one paired pivot into the search's reusable full-site scratch.
    pub(super) fn write_point(&self, pair: usize, point: &mut [usize]) -> TreeTciResult<()> {
        for (&site, &coordinate) in self
            .left_sites
            .as_slice()
            .iter()
            .zip(crate::column_2d(self.left, pair)?)
        {
            point[site] = coordinate;
        }
        for (&site, &coordinate) in self
            .right_sites
            .as_slice()
            .iter()
            .zip(crate::column_2d(self.right, pair)?)
        {
            point[site] = coordinate;
        }
        Ok(())
    }
}

/// Retained pivot tables at every edge, including ordinary physical cuts.
///
/// Only topology metadata is owned here. Full-site points are assembled into
/// reusable scratch, so visiting all retained pairs does not copy the pivot tables.
pub(super) fn retained_pivot_edges<T>(
    state: &TreeTCI2<T>,
) -> TreeTciResult<Vec<RetainedPivotEdge<'_>>> {
    let mut retained = Vec::new();
    for edge in state.graph.edges() {
        let (left_sites, right_sites) = state.graph.subregion_vertices(edge)?;
        let left = state
            .pivot_set(&left_sites)
            .ok_or_else(|| anyhow::anyhow!("missing retained left pivots"))?;
        let right = state
            .pivot_set(&right_sites)
            .ok_or_else(|| anyhow::anyhow!("missing retained right pivots"))?;
        let rank = crate::ncols_2d(left)?;
        if rank == 0 || crate::ncols_2d(right)? != rank {
            return Err(
                anyhow::anyhow!("retained pivot sets must have equal nonzero ranks").into(),
            );
        }
        retained.push(RetainedPivotEdge {
            left_sites,
            right_sites,
            left,
            right,
            rank,
            unit_endpoint: state.local_dims[edge.u()] == 1 || state.local_dims[edge.v()] == 1,
        });
    }
    Ok(retained)
}

// INVARIANT: this is one confirmation boundary's state, execution and search limits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn confirm_convergence_erased<T>(
    state: &TreeTCI2<T>,
    execution: &mut dyn BatchExecution<T>,
    nsearch: usize,
    max_nglobal_pivot: usize,
    abs_tol: f64,
    rng: &mut dyn RngCore,
) -> TreeTciResult<GlobalPivotSearchResult>
where
    T: FullPivLuScalar + Scalar + tensor4all_core::TensorElement + ScalarParts,
{
    let n_sites = state.local_dims.len();
    let mut readout = cached_walk_readout;
    let readout_on_session = cached_walk_readout_on_session::<T>;
    let treetn = to_treetn_chunked(state, execution, None, EVALUATION_CHUNK_POINTS)?;
    let mut site_indices = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let node = treetn
            .node_index(&site)
            .ok_or_else(|| anyhow::anyhow!("materialized tree missing site {site}"))?;
        let tensor = treetn
            .tensor(node)
            .ok_or_else(|| anyhow::anyhow!("materialized tree missing tensor for site {site}"))?;
        // `to_treetn` always stores the site index first.
        site_indices.push(tensor.indices()[0].clone());
    }

    // Cache ownership is one search, across every start/site/sweep. Scratch
    // is reused for bounded validation batches and individual site scans.
    // A prepared branch slice is a re-layout of one branching site tensor, so
    // retaining all of them costs at most the materialized tree once more. A
    // byte cap below one slice (the default 32 MiB is a cubic bond of ~160)
    // makes every readout rebuild the slice, so the cap is sized from the tree
    // instead of removed: see `branch_slice_cache_max_bytes`.
    let branch_slice_cache_cap = branch_slice_cache_max_bytes::<T>(&treetn);
    let mut cache = TreeTNCachedEvaluator::new(
        &treetn,
        &site_indices,
        CachedEvaluatorOptions {
            branch_slice_cache_max_bytes: branch_slice_cache_cap,
            ..CachedEvaluatorOptions::default()
        },
    )
    .map_err(anyhow::Error::from)?;
    let max_points = state
        .local_dims
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .saturating_sub(1)
        .max(1);
    let flat_capacity = n_sites
        .checked_mul(max_points)
        .ok_or_else(|| anyhow::anyhow!("global-pivot flat batch size overflowed usize"))?;
    let mut flat = Vec::with_capacity(flat_capacity);

    let validation::SampleValidation {
        max_error: known_error,
        point: known_point,
        retained: validation_edges,
    } = validation::validate_samples(
        state,
        execution,
        &mut cache,
        &treetn,
        &site_indices,
        &mut flat,
        &mut readout,
        readout_on_session,
    )?;
    let mut max_error = known_error;
    let mut best = Vec::new();
    if known_error > 0.0 && known_error >= abs_tol {
        best.push((known_error, known_point));
    }
    if best.is_empty() && nsearch > 0 {
        let retained = validation_edges;
        let indexer = FlatIndexer::try_new(&state.local_dims).map_err(anyhow::Error::new)?;
        // The same global pivot can occur on many cuts. Its deterministic
        // walk needs to run only once on this unchanged approximation. Compact
        // keys avoid copying full points; at most sum(bond ranks) are retained.
        let mut walked = HashSet::new();
        let mut support_start = vec![0; n_sites];
        for anchor in &retained {
            for pair in (0..anchor.rank).rev() {
                anchor.write_point(pair, &mut support_start)?;
                if !walked.insert(indexer.encode(&support_start).map_err(anyhow::Error::new)?) {
                    continue;
                }
                let (error, point) = walk_from_start(
                    state,
                    execution,
                    &mut cache,
                    &treetn,
                    &site_indices,
                    n_sites,
                    &mut flat,
                    &support_start,
                    abs_tol,
                    &mut readout,
                    readout_on_session,
                )?;
                max_error = max_error.max(error);
                if error > 0.0 && error >= abs_tol {
                    best.push((error, point));
                }
            }
        }
        if best.is_empty() {
            // Give each unit cut its own restart budget: sparse support on
            // one branch must not starve confirmation on another branch.
            for anchor in retained.iter().filter(|edge| edge.unit_endpoint) {
                for restart in 0..nsearch {
                    for varying in [&anchor.left_sites, &anchor.right_sites] {
                        anchor.write_point(
                            anchor.rank - 1 - restart % anchor.rank,
                            &mut support_start,
                        )?;
                        // Restart each side separately. Holding the complement
                        // intact preserves sparse factors on either side of the
                        // cut, instead of always randomizing a sparse branch away.
                        for &site in varying.as_slice() {
                            support_start[site] = rng.random_range(0..state.local_dims[site]);
                        }
                        let (error, point) = walk_from_start(
                            state,
                            execution,
                            &mut cache,
                            &treetn,
                            &site_indices,
                            n_sites,
                            &mut flat,
                            &support_start,
                            abs_tol,
                            &mut readout,
                            readout_on_session,
                        )?;
                        max_error = max_error.max(error);
                        if error > 0.0 && error >= abs_tol {
                            best.push((error, point));
                        }
                    }
                }
            }
        }
    }

    best.sort_by(|(a, _), (b, _)| b.total_cmp(a));
    // Protect a newly discovered confirmation witness without another walk.
    if let Some((error, point)) = best.first() {
        let values = execution.call(GlobalIndexBatch::new(point, n_sites, 1)?)?;
        if values.len() != 1 {
            return Err(
                anyhow::anyhow!("confirmation witness evaluator returned wrong length").into(),
            );
        }
        execution.observe_validation_residuals(
            GlobalIndexBatch::new(point, n_sites, 1)?,
            &values,
            &[*error],
        )?;
    }
    let mut pivots = Vec::new();
    for (_, point) in best {
        if pivots.len() >= max_nglobal_pivot {
            break;
        }
        if !pivots.contains(&point) {
            pivots.push(point);
        }
    }
    Ok(GlobalPivotSearchResult { pivots, max_error })
}

#[cfg(test)]
mod tests;
