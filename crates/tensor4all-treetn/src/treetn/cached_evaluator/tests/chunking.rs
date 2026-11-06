//! Batch chunking: chunk-limit validation, automatic chunk sizing, and branch
//! hints under chunked evaluation.

use super::*;

#[test]
fn chunk_limit_validation_scalar_batches_and_late_errors() {
    let tree = TreeTN::from_tensors(vec![IdxTensor::scalar(2.0).unwrap()], vec![0_usize]).unwrap();
    let plan = CachedEvaluatorPlan::new(&tree, &[]).unwrap();
    let invalid = CachedEvaluatorOptions {
        max_batch_points: Some(0),
        ..Default::default()
    };
    assert!(TreeTNCachedEvaluator::new(&tree, &[], invalid.clone()).is_err());
    assert!(TreeTNCachedEvaluator::with_plan(&tree, &plan, invalid).is_err());
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &[], CachedEvaluatorOptions::default()).unwrap();
    let actual = evaluator
        .evaluate_batched_typed::<f64>(
            ColMajorArrayRef::new(&[], &[0, 33]).unwrap(),
            EvaluationHint::default(),
        )
        .unwrap();
    assert_eq!(actual, vec![2.0; 33]);
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 1);
    let (tree, indices) = typed_three_node_chain::<f64>();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let mut values = vec![0_usize; 3 * 257];
    values[3 * 256] = indices[0].dim(); // invalid value in the last internal chunk
    let points = ColMajorArrayRef::new(&values, &[3, 257]).unwrap();
    assert!(evaluator.evaluate_batched(points).is_err());
    values[3 * 256] = 1;
    let points = ColMajorArrayRef::new(&values, &[3, 257]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let actual = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(0))
        .unwrap();
    assert_typed_results::<f64>(&actual, &expected);
    assert!(evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(99))
        .is_err());
}

fn assert_chunked_branch_hints<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_unequal_y_tree::<T>();
    let values = (0..33)
        .flat_map(|p| [p % 2, (p / 2) % 2, (p / 4) % 2, (p / 8) % 2])
        .collect::<Vec<_>>();
    let points = ColMajorArrayRef::new(&values, &[4, 33]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    for limit in [1_usize, 7, 16, usize::MAX] {
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                max_batch_points: Some(limit),
                message_cache_max_bytes: 0,
                branch_slice_cache_max_bytes: 0,
                ..Default::default()
            },
        )
        .unwrap();
        for hint in [
            EvaluationHint::default(),
            EvaluationHint::around(0),
            EvaluationHint::around(1),
        ] {
            let actual = evaluator.evaluate_batched(points).unwrap();
            assert_typed_results::<T>(&actual, &expected);
            let actual = evaluator.evaluate_batched_typed::<T>(points, hint).unwrap();
            let actual = actual
                .into_iter()
                .map(CachedEvaluatorTestScalar::into_any_scalar)
                .collect::<Vec<AnyScalar>>();
            assert_typed_results::<T>(&actual, &expected);
            assert_eq!(
                evaluator.stats_for_test().batched_center_contract_count,
                33_usize.div_ceil(limit)
            );
        }
    }
}

#[test]
fn chunked_branch_hints_and_disabled_caches_preserve_all_scalar_kinds() {
    assert_chunked_branch_hints::<f32>();
    assert_chunked_branch_hints::<f64>();
    assert_chunked_branch_hints::<Complex32>();
    assert_chunked_branch_hints::<Complex64>();
}

#[test]
fn automatic_chunk_limit_shrinks_for_large_local_tensors() {
    let indices = vec![DynIndex::new_dyn(128), DynIndex::new_dyn(128)];
    let tensor = IdxTensor::from_dense(indices.clone(), vec![2.0_f64; 128 * 128]).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0_usize]).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let limit =
        (32 * 1024 * 1024 / (128 * 128 * std::mem::size_of::<CachedScalar>())).clamp(1, 256);
    assert_eq!(evaluator.batch_point_limit, limit);
    assert!(limit < 256);
    let values = vec![0_usize; 2 * (limit + 1)];
    let shape = [2, limit + 1];
    let actual = evaluator
        .evaluate_batched_typed::<f64>(
            ColMajorArrayRef::new(&values, &shape).unwrap(),
            EvaluationHint::default(),
        )
        .unwrap();
    assert_eq!(actual, vec![2.0; limit + 1]);
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 2);
}

fn assert_default_batch_chunking<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_three_node_chain::<T>();
    for n_points in [0_usize, 1, 255, 256, 257, 512, 513] {
        let values = (0..n_points)
            .flat_map(|point| [point % 2, (point / 2) % 2, point % 3])
            .collect::<Vec<_>>();
        let shape = [3, n_points];
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
        assert_typed_results::<T>(&actual, &expected);
        assert_eq!(
            evaluator.stats_for_test().batched_center_contract_count,
            n_points.div_ceil(256)
        );
    }
}

#[test]
fn default_batch_chunking_bounds_center_work_and_preserves_results() {
    assert_default_batch_chunking::<f32>();
    assert_default_batch_chunking::<f64>();
    assert_default_batch_chunking::<Complex32>();
    assert_default_batch_chunking::<Complex64>();
}
