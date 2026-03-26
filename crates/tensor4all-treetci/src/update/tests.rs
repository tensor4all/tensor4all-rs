use super::update_edge_default;
use crate::{GlobalIndexBatch, SimpleTreeTci, TreeTciEdge, TreeTciGraph};
use anyhow::Result;
use matrixluci::PivotKernelOptions;

fn two_site_graph() -> TreeTciGraph {
    TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap()
}

#[test]
fn update_edge_selects_identity_pivots_on_two_site_tree() {
    let mut tci = SimpleTreeTci::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();
    tci.flush_pivot_errors();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let selection = update_edge_default(
        &mut tci,
        TreeTciEdge::new(0, 1),
        batch_eval,
        &PivotKernelOptions::no_truncation(),
    )
    .unwrap();

    assert_eq!(selection.rank, 2);
    assert_eq!(
        tci.ijset[&crate::SubtreeKey::new(vec![0])],
        vec![vec![0], vec![1]]
    );
    assert_eq!(
        tci.ijset[&crate::SubtreeKey::new(vec![1])],
        vec![vec![0], vec![1]]
    );
    assert_eq!(tci.max_sample_value, 1.0);
    assert_eq!(tci.max_bond_error(), 0.0);
    assert_eq!(tci.pivot_errors.last().copied(), Some(0.0));
}

#[test]
fn update_edge_rejects_bad_batch_length() {
    let mut tci = SimpleTreeTci::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let bad_eval = |_batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![1.0]) };
    let result = update_edge_default(
        &mut tci,
        TreeTciEdge::new(0, 1),
        bad_eval,
        &PivotKernelOptions::no_truncation(),
    );

    assert!(result.is_err());
}
