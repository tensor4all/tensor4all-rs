//! Integration tests for `crossinterpolate2` input validation and error paths.

use tensor4all_treetci::{crossinterpolate2, DefaultProposer, TreeTciGraph, TreeTciOptions};

#[test]
fn crossinterpolate2_propagates_initial_callback_error() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let error = crossinterpolate2::<f64, _, _>(
        |_| Err(anyhow::anyhow!("callback failed")),
        vec![2, 2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("callback failed"));
}

#[test]
fn crossinterpolate2_accepts_empty_initial_pivots() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let result = crossinterpolate2::<f64, _, _>(
        |batch| Ok(vec![1.0; batch.n_points()]),
        vec![2, 2],
        graph,
        vec![],
        TreeTciOptions {
            max_iter: 1,
            max_bond_dim: Some(2),
            ..TreeTciOptions::default()
        },
        None,
        &DefaultProposer,
    );
    assert!(
        result.is_ok(),
        "empty initial pivots should be accepted: {:?}",
        result.as_ref().err()
    );
}

#[test]
fn crossinterpolate2_rejects_local_dimension_mismatch() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let error = crossinterpolate2::<f64, _, _>(
        |_| Ok(vec![1.0]),
        vec![2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("local_dims length"));
}

#[test]
fn crossinterpolate2_rejects_initial_callback_length_mismatch() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let error = crossinterpolate2::<f64, _, _>(
        |_| Ok(Vec::new()),
        vec![2, 2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("initial evaluator returned"));
}

#[test]
fn crossinterpolate2_rejects_all_zero_initial_pivots() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let error = crossinterpolate2::<f64, _, _>(
        |_| Ok(vec![0.0]),
        vec![2, 2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("must not all evaluate to zero"));
}
