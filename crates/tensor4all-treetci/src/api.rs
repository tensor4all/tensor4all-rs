use crate::{
    materialize::{to_treetn, FullPivLuScalar},
    optimize_with_proposer, GlobalIndexBatch, MultiIndex, PivotCandidateProposer, TreeTCI2,
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
///
/// This is the unified entry point for tree tensor cross interpolation.
/// The `evaluate` closure receives batches of multi-indices and must return
/// one scalar per point.
///
/// The `proposer` controls how pivot candidates are generated.
#[allow(clippy::too_many_arguments)]
pub fn crossinterpolate2<T, F, P>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
) -> Result<TreeTciRunResult>
where
    T: FullPivLuScalar,
    tensor4all_tcicore::DenseFaerLuKernel: tensor4all_tcicore::PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
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

    let mut tci = TreeTCI2::<T>::new(local_dims, graph)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value via batch evaluate
    let n_sites = tci.local_dims.len();
    let flat: Vec<usize> = pivots.iter().flat_map(|p| p.iter().copied()).collect();
    let batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = evaluate(batch)?;
    tci.max_sample_value = init_vals
        .iter()
        .map(|v| T::abs_val(*v))
        .fold(0.0f64, f64::max);
    ensure!(
        tci.max_sample_value > 0.0,
        "initial pivots must not all evaluate to zero"
    );

    let (ranks, errors) = optimize_with_proposer(&mut tci, &evaluate, &options, proposer)?;
    let treetn = to_treetn(&tci, &evaluate, center_site)?;

    Ok((treetn, ranks, errors))
}
