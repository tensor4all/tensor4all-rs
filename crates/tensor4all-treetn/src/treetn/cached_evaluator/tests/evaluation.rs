//! End-to-end evaluation against `TreeTN::evaluate`, plus input validation and
//! error paths.

use super::*;

/// [AI Supplied] Error and fallback coverage for the cached evaluator.
/// Shape and coordinate failures must return contextual errors, while a
/// degree-four hub (outside the raw-kernel degree limit) must use the
/// generic route and still match the dense oracle.
#[test]
fn cached_evaluator_error_paths_and_unsupported_raw_dispatch_are_typed() {
    let (tree, indices) = two_node_tree();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

    let wrong_rank = ColMajorArrayRef::new(&[0usize, 0, 0], &[1usize, 3usize, 1usize]).unwrap();
    let error = evaluator.evaluate_batched(wrong_rank).unwrap_err();
    assert!(
        error.to_string().contains("2D"),
        "unexpected error: {error}"
    );

    let wrong_rows = ColMajorArrayRef::new(&[0usize, 0], &[1usize, 2usize]).unwrap();
    let error = evaluator.evaluate_batched(wrong_rows).unwrap_err();
    assert!(
        error.to_string().contains("row count"),
        "unexpected error: {error}"
    );

    let out_of_range = ColMajorArrayRef::new(&[0usize, 2], &[2usize, 1usize]).unwrap();
    let error = evaluator.evaluate_batched(out_of_range).unwrap_err();
    assert!(
        error.to_string().contains("out of range"),
        "unexpected error: {error}"
    );

    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let mixed = TreeTN::<_, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f32, 0.0, 0.0, 1.0])
                .unwrap(),
            IdxTensor::from_dense(vec![bond, s1.clone()], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let mixed_points = ColMajorArrayRef::new(&[0usize, 0], &[2usize, 1usize]).unwrap();
    let mixed_indices = vec![s0.clone(), s1.clone()];
    let mut mixed_evaluator =
        TreeTNCachedEvaluator::new(&mixed, &mixed_indices, CachedEvaluatorOptions::default())
            .unwrap();
    let mixed_expected = mixed.evaluate(&mixed_indices, mixed_points);
    let mixed_expected = mixed_expected.unwrap();
    let mixed_cold = mixed_evaluator.evaluate_batched(mixed_points).unwrap();
    let mixed_warm = mixed_evaluator.evaluate_batched(mixed_points).unwrap();
    assert_typed_results::<f64>(&mixed_cold, &mixed_expected);
    assert_typed_results::<f64>(&mixed_warm, &mixed_expected);
    for center in [0, 1] {
        for _ in 0..2 {
            let actual = mixed_evaluator
                .evaluate_batched_with_hint(mixed_points, EvaluationHint::around(center))
                .unwrap();
            assert_typed_results::<f64>(&actual, &mixed_expected);
        }
        assert!(mixed_evaluator
            .try_contract_cached_edge_cut(&center, mixed_points)
            .unwrap()
            .is_none());
    }

    let (degree_four, degree_four_indices) = four_arm_star_tree();
    let values = [
        0usize, 0, 0, 0, 0, // point 0
        1, 0, 1, 0, 1, // point 1
        0, 1, 0, 1, 0, // point 2
    ];
    let points = ColMajorArrayRef::new(&values, &[5usize, 3usize]).unwrap();
    let expected = degree_four.evaluate(&degree_four_indices, points).unwrap();
    let mut fallback = TreeTNCachedEvaluator::new(
        &degree_four,
        &degree_four_indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let actual = fallback.evaluate_batched(points).unwrap();
    assert_typed_results::<f64>(&actual, &expected);
    let hub = fallback
        .build_environment_cache(&1, points, &mut CachedEvaluationRoute::global())
        .unwrap()
        .1
        .remove(&0)
        .unwrap();
    // The values above already match the tree's own evaluation; the hub's
    // three rooted children now reach that answer on the raw path.
    assert!(hub.raw_values.is_some());
    assert!(hub.tensor.is_none());
}

fn two_node_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();
    let t1 = IdxTensor::from_dense(vec![bond, s1.clone()], vec![0.5_f64, 1.5, 2.5, 3.5]).unwrap();

    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
    (tree, vec![s0, s1])
}

/// The whole public evaluation must match the tree's own contraction on a
/// high-coordination star, with the centre both at a leaf (which
/// exercises the arbitrary-degree *message* kernel) and at the hub (which
/// exercises the arbitrary-degree *centre* contraction).
#[test]
fn cached_evaluator_matches_tree_evaluate_on_high_coordination_stars() {
    for arms in [4usize, 5, 6] {
        let (tree, indices) = unequal_star_tree::<f64>(arms);
        let values = star_points(&indices, 5);
        let shape = [indices.len(), 5];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();

        for center in [1usize, 0] {
            let mut evaluator = TreeTNCachedEvaluator::new(
                &tree,
                &indices,
                CachedEvaluatorOptions {
                    center: Some(center),
                    ..Default::default()
                },
            )
            .unwrap();
            let cold = evaluator.evaluate_batched(points).unwrap();
            let warm = evaluator.evaluate_batched(points).unwrap();
            assert_scalars_close(&cold, &expected);
            assert_scalars_close(&warm, &expected);
        }
    }
}

#[test]
fn cached_evaluator_matches_tree_evaluate_on_two_node_chain() {
    let (tree, indices) = two_node_tree();
    let values = vec![0, 0, 1, 0, 0, 1, 1, 1];
    let shape = [2, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let expected = tree.evaluate(&indices, points).unwrap();
    let options = CachedEvaluatorOptions {
        center: Some(0),
        ..CachedEvaluatorOptions::default()
    };
    let mut evaluator = TreeTNCachedEvaluator::new(&tree, &indices, options).unwrap();
    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
    assert_eq!(evaluator.center(), Some(&0));
}

#[test]
fn cached_evaluator_rejects_wrong_value_row_count() {
    let (tree, indices) = two_node_tree();
    let values = vec![0, 1, 1];
    let shape = [1, 3];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

    let err = evaluator.evaluate_batched(points).unwrap_err();
    assert!(err.to_string().contains("row count"));
}

#[test]
fn cached_evaluator_rejects_out_of_range_site_value() {
    let (tree, indices) = two_node_tree();
    let values = vec![0, 2];
    let shape = [2, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

    let err = evaluator.evaluate_batched(points).unwrap_err();
    assert!(err.to_string().contains("out of range"));
}

#[test]
fn cached_evaluator_handles_repeated_points_without_changing_order() {
    let (tree, indices) = two_node_tree();
    let values = vec![0, 0, 1, 1, 0, 0, 1, 1];
    let shape = [2, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
    assert_eq!(actual[0].real(), actual[2].real());
    assert_eq!(actual[1].real(), actual[3].real());
}

#[test]
fn cached_evaluator_matches_tree_evaluate_on_star_tree() {
    let (tree, indices) = star_tree();
    let values = vec![
        0, 0, 0, 0, //
        1, 0, 1, 0, //
        0, 1, 0, 1, //
        1, 1, 1, 1,
    ];
    let shape = [4, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();

    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
}

/// `cached_evaluator_matches_tree_evaluate_on_star_tree` fixes `center`
/// to the hub itself, so the hub's combination goes through the
/// separate one-shot centre contraction, not `get_or_compute_node_message`
/// -- it never exercises `try_compute_branch_message_raw`. This test
/// fixes `center` to a leaf instead, so the hub (rooted toward that
/// leaf) has two children and its message must go through the new
/// branch dispatch, verified end-to-end through the public
/// `evaluate_batched` API rather than by manually driving the message
/// plan (unlike `raw_branch_message_matches_generic_contraction`).
#[test]
fn cached_evaluator_matches_tree_evaluate_on_star_tree_with_fixed_leaf_center() {
    let (tree, indices) = star_tree();
    let values = vec![
        0, 0, 0, 0, //
        1, 1, 0, 1, //
        0, 1, 1, 0, //
        1, 0, 1, 1,
    ];
    let shape = [4, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();

    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
}

#[test]
fn cached_evaluator_batches_center_contraction() {
    let (tree, indices) = star_tree();
    let values = vec![
        0, 0, 0, 0, //
        1, 0, 1, 0, //
        0, 1, 0, 1, //
        1, 1, 1, 1,
    ];
    let shape = [4, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();

    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 1);
}
