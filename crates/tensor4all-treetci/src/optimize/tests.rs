use super::{optimize_default, TreeTciOptions};
use crate::test_support::assert_scalar_close;
use crate::{GlobalIndexBatch, SimpleTreeTci, TreeTciEdge, TreeTciGraph};
use anyhow::Result;

fn two_site_graph() -> TreeTciGraph {
    TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap()
}

#[test]
fn optimize_default_converges_on_two_site_identity() {
    let mut tci = SimpleTreeTci::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let (ranks, errors) = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        },
    )
    .unwrap();

    assert_eq!(ranks.last().copied(), Some(2));
    assert_scalar_close(
        errors.last().copied().unwrap_or(f64::NAN),
        0.0,
        tci.max_sample_value,
        1e-12,
    );
    assert_eq!(tci.max_rank(), 2);
}

#[test]
fn optimize_default_runs_all_iterations_like_upstream_tree_tci() {
    let mut tci = SimpleTreeTci::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let (ranks, errors) = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        },
    )
    .unwrap();

    assert_eq!(ranks.len(), 4);
    assert_eq!(errors.len(), 4);
}
