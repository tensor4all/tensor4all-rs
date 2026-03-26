use tensor4all_treetci::{
    crossinterpolate_tree, crossinterpolate_tree_with_proposer, SimpleProposer, TreeTciEdge,
    TreeTciGraph, TreeTciOptions,
};

fn sample_graph() -> TreeTciGraph {
    TreeTciGraph::new(
        7,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
            TreeTciEdge::new(3, 4),
            TreeTciEdge::new(4, 5),
            TreeTciEdge::new(4, 6),
        ],
    )
    .unwrap()
}

#[test]
fn simple_tree_parity_matches_reference_points() {
    let f = |idx: &[usize]| {
        let norm_sq = idx.iter().map(|&x| (x + 1) * (x + 1)).sum::<usize>() as f64;
        1.0 / (1.0 + norm_sq)
    };

    let (tn, _ranks, _errors) = crossinterpolate_tree_with_proposer(
        f,
        None::<fn(tensor4all_treetci::GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>>>,
        vec![2; 7],
        sample_graph(),
        vec![vec![0; 7]],
        TreeTciOptions {
            tolerance: 1e-8,
            max_iter: 10,
            max_bond_dim: 5,
            normalize_error: true,
        },
        Some(0),
        &SimpleProposer::seeded(0),
    )
    .unwrap();

    let eval = |point: [usize; 7]| -> f64 {
        tn.evaluate(&std::collections::HashMap::from([
            (0usize, vec![point[0]]),
            (1usize, vec![point[1]]),
            (2usize, vec![point[2]]),
            (3usize, vec![point[3]]),
            (4usize, vec![point[4]]),
            (5usize, vec![point[5]]),
            (6usize, vec![point[6]]),
        ]))
        .unwrap()
        .real()
    };

    for point in [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ] {
        let expected = f(&point);
        let got = eval(point);
        assert!(
            (got - expected).abs() < 1e-8,
            "point {:?}: got {}, expected {}",
            point,
            got,
            expected
        );
    }
}

#[test]
fn simple_tree_product_function_is_exact_on_branching_tree() {
    let f = |idx: &[usize]| idx.iter().fold(1.0, |acc, &x| acc * (x as f64 + 1.0));

    let (tn, _ranks, _errors) = crossinterpolate_tree(
        f,
        None::<fn(tensor4all_treetci::GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>>>,
        vec![2; 7],
        sample_graph(),
        vec![vec![0; 7]],
        TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 8,
            max_bond_dim: 2,
            normalize_error: true,
        },
        Some(0),
    )
    .unwrap();

    let eval = |point: [usize; 7]| -> f64 {
        tn.evaluate(&std::collections::HashMap::from([
            (0usize, vec![point[0]]),
            (1usize, vec![point[1]]),
            (2usize, vec![point[2]]),
            (3usize, vec![point[3]]),
            (4usize, vec![point[4]]),
            (5usize, vec![point[5]]),
            (6usize, vec![point[6]]),
        ]))
        .unwrap()
        .real()
    };

    for point in [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ] {
        let expected = f(&point);
        let got = eval(point);
        assert_eq!(got, expected, "point {:?}", point);
    }
}
