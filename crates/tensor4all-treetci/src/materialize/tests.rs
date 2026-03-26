use super::{cartesian_entries, to_treetn};
use crate::{
    optimize_default, GlobalIndexBatch, SimpleTreeTci, SubtreeKey, TreeTciEdge, TreeTciGraph,
    TreeTciOptions,
};
use anyhow::Result;
use std::collections::HashMap as StdHashMap;
use std::collections::HashMap;

fn two_site_graph() -> TreeTciGraph {
    TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap()
}

#[test]
fn to_treetn_preserves_two_site_identity_evaluations() {
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

    optimize_default(
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

    let tn = to_treetn(&tci, batch_eval, Some(0)).unwrap();

    let eval = |i: usize, j: usize| -> f64 {
        tn.evaluate(&HashMap::from([(0usize, vec![i]), (1usize, vec![j])]))
            .unwrap()
            .real()
    };

    assert_eq!(eval(0, 0), 1.0);
    assert_eq!(eval(0, 1), 0.0);
    assert_eq!(eval(1, 0), 0.0);
    assert_eq!(eval(1, 1), 1.0);
}

#[test]
fn cartesian_entries_matches_julia_product_order() {
    let key_a = SubtreeKey::new(vec![0]);
    let key_b = SubtreeKey::new(vec![1]);
    let ijset = StdHashMap::from([
        (key_a.clone(), vec![vec![0], vec![1]]),
        (key_b.clone(), vec![vec![10], vec![11], vec![12]]),
    ]);

    let combos = cartesian_entries(&ijset, &[key_a, key_b]).unwrap();

    assert_eq!(
        combos,
        vec![
            vec![vec![0], vec![10]],
            vec![vec![1], vec![10]],
            vec![vec![0], vec![11]],
            vec![vec![1], vec![11]],
            vec![vec![0], vec![12]],
            vec![vec![1], vec![12]],
        ]
    );
}
