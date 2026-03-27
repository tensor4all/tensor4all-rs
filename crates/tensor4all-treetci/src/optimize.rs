use crate::{
    update::update_edge, AllEdges, EdgeVisitor, GlobalIndexBatch, PivotCandidateProposer,
    SimpleTreeTci,
};
use anyhow::{ensure, Result};
use tensor4all_tcicore::{
    DenseFaerLuKernel, MatrixLuciScalar as Scalar, PivotKernel, PivotKernelOptions,
};

/// MVP optimization options for TreeTCI.
#[derive(Clone, Debug)]
pub struct TreeTciOptions {
    /// Relative stopping tolerance on the normalized bond error.
    pub tolerance: f64,
    /// Maximum number of edge-order iterations.
    pub max_iter: usize,
    /// Maximum bond dimension retained by the tcicore LUCI pivot substrate.
    pub max_bond_dim: usize,
    /// Whether to normalize by the maximum observed sample magnitude.
    pub normalize_error: bool,
}

impl Default for TreeTciOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 20,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        }
    }
}

/// Optimize a TreeTCI state with the MVP strategy choices:
/// `AllEdges` visitation and `DefaultProposer`.
pub fn optimize_default<T, F>(
    state: &mut SimpleTreeTci<T>,
    batch_eval: F,
    options: &TreeTciOptions,
) -> Result<(Vec<usize>, Vec<f64>)>
where
    T: Scalar,
    DenseFaerLuKernel: PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    optimize_with_proposer(state, batch_eval, options, &crate::DefaultProposer)
}

/// Optimize a TreeTCI state with `AllEdges` visitation and a caller-supplied
/// pivot candidate proposer.
pub fn optimize_with_proposer<T, F, P>(
    state: &mut SimpleTreeTci<T>,
    batch_eval: F,
    options: &TreeTciOptions,
    proposer: &P,
) -> Result<(Vec<usize>, Vec<f64>)>
where
    T: Scalar,
    DenseFaerLuKernel: PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    ensure!(
        options.max_iter > 0,
        "TreeTCI optimization requires max_iter > 0"
    );
    ensure!(
        options.max_bond_dim > 0,
        "TreeTCI optimization requires max_bond_dim > 0"
    );

    let mut ranks = Vec::new();
    let mut errors = Vec::new();
    let visitor = AllEdges;
    const INNER_EDGE_PASSES: usize = 2;

    for _iter in 0..options.max_iter {
        for _pass in 0..INNER_EDGE_PASSES {
            let error_scale = if options.normalize_error && state.max_sample_value > 0.0 {
                state.max_sample_value
            } else {
                1.0
            };
            let kernel_options = PivotKernelOptions {
                rel_tol: 1e-14,
                abs_tol: options.tolerance * error_scale,
                max_rank: options.max_bond_dim,
                left_orthogonal: true,
            };

            state.ijset_history.push(state.ijset.clone());
            state.flush_pivot_errors();

            for edge in visitor.visit_order(state) {
                update_edge(state, edge, &batch_eval, &kernel_options, proposer)?;
            }
        }

        ranks.push(state.max_rank());
        let normalized_error = if options.normalize_error && state.max_sample_value > 0.0 {
            state.max_bond_error() / state.max_sample_value
        } else {
            state.max_bond_error()
        };
        errors.push(normalized_error);
    }

    Ok((ranks, errors))
}

#[cfg(test)]
mod tests;
