use crate::{
    materialize::{to_treetn, FullPivLuScalar},
    optimize_with_proposer, GlobalIndexBatch, MultiIndex, PivotCandidateProposer, SimpleTreeTci,
    TreeTciGraph, TreeTciOptions,
};
use anyhow::{ensure, Result};
use tensor4all_treetn::TreeTN;

/// High-level TreeTCI return type:
/// `(treetn, ranks_per_iter, normalized_errors_per_iter)`.
pub type TreeTciRunResult = (
    TreeTN<tensor4all_core::TensorDynLen, usize>,
    Vec<usize>,
    Vec<f64>,
);

/// Cross interpolate a function on a tree graph and return a `TreeTN`.
pub fn crossinterpolate_tree<T, F, B>(
    point_eval: F,
    batch_eval: Option<B>,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
) -> Result<TreeTciRunResult>
where
    T: FullPivLuScalar,
    matrixluci::DenseFaerLuKernel: matrixluci::PivotKernel<T>,
    F: Fn(&[usize]) -> T,
    B: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    crossinterpolate_tree_with_proposer(
        point_eval,
        batch_eval,
        local_dims,
        graph,
        initial_pivots,
        options,
        center_site,
        &crate::DefaultProposer,
    )
}

/// Cross interpolate a function on a tree graph with a caller-supplied pivot
/// candidate proposer and return a `TreeTN`.
#[allow(clippy::too_many_arguments)]
pub fn crossinterpolate_tree_with_proposer<T, F, B, P>(
    point_eval: F,
    batch_eval: Option<B>,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
) -> Result<TreeTciRunResult>
where
    T: FullPivLuScalar,
    matrixluci::DenseFaerLuKernel: matrixluci::PivotKernel<T>,
    F: Fn(&[usize]) -> T,
    B: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    ensure!(
        local_dims.len() == graph.n_sites(),
        "local_dims length {} must match graph site count {}",
        local_dims.len(),
        graph.n_sites()
    );

    let pivots = if initial_pivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        initial_pivots
    };

    let mut tci = SimpleTreeTci::<T>::new(local_dims, graph)?;
    tci.add_global_pivots(&pivots)?;

    for pivot in &pivots {
        tci.max_sample_value = tci.max_sample_value.max(point_eval(pivot).abs_val());
    }
    ensure!(
        tci.max_sample_value > 0.0,
        "initial pivots must not all evaluate to zero"
    );

    let (ranks, errors) = match &batch_eval {
        Some(batch_eval) => optimize_with_proposer(&mut tci, batch_eval, &options, proposer)?,
        None => optimize_with_proposer(
            &mut tci,
            |batch| fallback_batch_eval(batch, &point_eval),
            &options,
            proposer,
        )?,
    };

    let treetn = match &batch_eval {
        Some(batch_eval) => to_treetn(&tci, batch_eval, center_site)?,
        None => to_treetn(
            &tci,
            |batch| fallback_batch_eval(batch, &point_eval),
            center_site,
        )?,
    };

    Ok((treetn, ranks, errors))
}

fn fallback_batch_eval<T, F>(batch: GlobalIndexBatch<'_>, point_eval: &F) -> Result<Vec<T>>
where
    F: Fn(&[usize]) -> T,
{
    let mut values = Vec::with_capacity(batch.n_points());
    let mut point = vec![0usize; batch.n_sites()];
    for point_idx in 0..batch.n_points() {
        for (site, value) in point.iter_mut().enumerate().take(batch.n_sites()) {
            *value = batch.get(site, point_idx).ok_or_else(|| {
                anyhow::anyhow!("missing batch entry at site {} point {}", site, point_idx)
            })?;
        }
        values.push(point_eval(&point));
    }
    Ok(values)
}
