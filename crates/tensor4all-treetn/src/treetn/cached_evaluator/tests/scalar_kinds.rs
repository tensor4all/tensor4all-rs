//! Scalar-kind coverage: the typed `f32`/`f64`/`Complex32`/`Complex64` routes agree
//! with `TreeTN::evaluate` and with the `AnyScalar` wrapper.

use super::*;

fn assert_four_scalar_kind_chain<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_three_node_chain::<T>();
    let values = [
        0usize, 0, 0, // point 0
        1, 0, 1, // point 1
        0, 1, 1, // point 2
        1, 1, 0, // point 3
        1, 0, 1, // duplicate point 1
    ];
    let points = ColMajorArrayRef::new(&values, &[3, 5]).unwrap();
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

    let cold = evaluator.evaluate_batched(points).unwrap();
    let warm = evaluator.evaluate_batched(points).unwrap();

    assert_typed_results::<T>(&cold, &expected);
    assert_typed_results::<T>(&warm, &expected);
    assert!(
        evaluator.stats_for_test().message_cache_hits > 0,
        "warm four-scalar evaluation must reuse message cache"
    );
}

/// [AI Supplied] Regression matrix for all supported cached-evaluator
/// scalar kinds. The ordinary evaluator is the dense-result oracle; the
/// debug dtype assertion ensures a cache round trip does not silently
/// promote 32-bit payloads.
#[test]
fn cached_evaluator_four_scalar_kinds_match_tree_evaluate() {
    assert_four_scalar_kind_chain::<f32>();
    assert_four_scalar_kind_chain::<f64>();
    assert_four_scalar_kind_chain::<Complex32>();
    assert_four_scalar_kind_chain::<Complex64>();
}

fn rewrap_typed<T: CachedEvaluatorTestScalar>(values: &[T]) -> Vec<AnyScalar> {
    values
        .iter()
        .copied()
        .map(CachedEvaluatorTestScalar::into_any_scalar)
        .collect()
}

fn assert_exactly_equal_scalars(actual: &[AnyScalar], expected: &[AnyScalar]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(
            (actual.real(), actual.imag()),
            (expected.real(), expected.imag()),
            "typed result differs from the AnyScalar wrapper: {actual:?} vs {expected:?}"
        );
        assert_eq!(
            format!("{actual:?}"),
            format!("{expected:?}"),
            "typed result lost the wrapper's dtype"
        );
    }
}

/// The typed batch API must return exactly the values the `AnyScalar`
/// wrapper returns, for the same batch order, duplicates, hint, cache
/// state, evaluated-point accounting, and error paths.
fn assert_typed_matches_wrapper<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_three_node_chain::<T>();
    // Point 4 duplicates point 1 and the batch is deliberately unsorted.
    let values = [
        0usize, 0, 0, // point 0
        1, 0, 1, // point 1
        0, 1, 1, // point 2
        1, 1, 0, // point 3
        1, 0, 1, // point 4 duplicates point 1
    ];
    let points = ColMajorArrayRef::new(&values, &[3, 5]).unwrap();
    let oracle = tree.evaluate(&indices, points).unwrap();

    let options = CachedEvaluatorOptions {
        center: Some(1),
        ..Default::default()
    };
    let mut wrapper = TreeTNCachedEvaluator::new(&tree, &indices, options.clone()).unwrap();
    let mut typed = TreeTNCachedEvaluator::new(&tree, &indices, options).unwrap();

    let wrapper_cold = wrapper.evaluate_batched(points).unwrap();
    let typed_cold = typed
        .evaluate_batched_typed::<T>(points, EvaluationHint::default())
        .unwrap();
    assert_typed_results::<T>(&wrapper_cold, &oracle);
    assert_exactly_equal_scalars(&rewrap_typed(&typed_cold), &wrapper_cold);
    assert_eq!(typed.stats_for_test(), wrapper.stats_for_test());

    let wrapper_warm = wrapper.evaluate_batched(points).unwrap();
    let typed_warm = typed
        .evaluate_batched_typed::<T>(points, EvaluationHint::default())
        .unwrap();
    assert_exactly_equal_scalars(&rewrap_typed(&typed_warm), &wrapper_warm);
    assert_eq!(typed.stats_for_test(), wrapper.stats_for_test());
    assert!(
        typed.stats_for_test().message_cache_hits > 0,
        "warm typed batch must reuse the message cache"
    );

    let wrapper_hinted = wrapper
        .evaluate_batched_with_hint(points, EvaluationHint::around(1))
        .unwrap();
    let typed_hinted = typed
        .evaluate_batched_typed::<T>(points, EvaluationHint::around(1))
        .unwrap();
    assert_exactly_equal_scalars(&rewrap_typed(&typed_hinted), &wrapper_hinted);
    assert_eq!(typed.stats_for_test(), wrapper.stats_for_test());

    // Batch order and duplicate points survive both routes.
    assert_eq!(typed_cold[1], typed_cold[4]);
    assert_eq!(typed_hinted[1], typed_hinted[4]);
    assert_eq!(typed_cold, typed_hinted);

    // An empty batch is empty on both routes and resets the accounting.
    let empty_values: [usize; 0] = [];
    let empty = ColMajorArrayRef::new(&empty_values, &[3, 0]).unwrap();
    assert!(wrapper.evaluate_batched(empty).unwrap().is_empty());
    assert!(typed
        .evaluate_batched_typed::<T>(empty, EvaluationHint::default())
        .unwrap()
        .is_empty());
    assert_eq!(typed.stats_for_test(), wrapper.stats_for_test());

    // A shape mismatch is rejected on both routes.
    let wrong_shape = ColMajorArrayRef::new(&values, &[5, 3]).unwrap();
    assert!(wrapper.evaluate_batched(wrong_shape).is_err());
    assert!(typed
        .evaluate_batched_typed::<T>(wrong_shape, EvaluationHint::default())
        .is_err());

    // A hinted centre that is not a node is rejected on both routes.
    assert!(wrapper
        .evaluate_batched_with_hint(points, EvaluationHint::around(99))
        .is_err());
    assert!(typed
        .evaluate_batched_typed::<T>(points, EvaluationHint::around(99))
        .is_err());

    // Complex payloads must not be silently narrowed into a real request,
    // while a real payload widens into a complex request. This is exactly
    // the dynamic wrapper's own dtype contract.
    let complex_request =
        typed.evaluate_batched_typed::<Complex64>(points, EvaluationHint::default());
    let complex = complex_request.unwrap();
    for (typed_value, wrapper_value) in complex.iter().zip(&wrapper_cold) {
        assert_eq!(typed_value.re, wrapper_value.real());
        assert_eq!(typed_value.im, wrapper_value.imag());
    }
    let real_request = typed.evaluate_batched_typed::<f64>(points, EvaluationHint::default());
    if T::IS_COMPLEX {
        let message = real_request.unwrap_err().to_string();
        assert!(
            message.contains("f64"),
            "a complex tree must reject an f64 request by name: {message}"
        );
    } else {
        let real = real_request.unwrap();
        for (typed_value, wrapper_value) in real.iter().zip(&wrapper_cold) {
            assert_eq!(*typed_value, wrapper_value.real());
        }
    }
}

/// [AI Supplied] #709 differential gate: the typed batch API and the
/// `AnyScalar` compatibility wrapper must agree exactly for every
/// supported scalar kind, cache state, hint, and error path.
#[test]
fn evaluate_batched_typed_matches_any_scalar_wrapper_for_all_scalar_kinds() {
    assert_typed_matches_wrapper::<f32>();
    assert_typed_matches_wrapper::<f64>();
    assert_typed_matches_wrapper::<Complex32>();
    assert_typed_matches_wrapper::<Complex64>();
}

fn assert_reordered_duplicate_and_partial_hit<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_three_node_chain::<T>();
    let initial = ColMajorArrayRef::new(&[0usize, 0, 0, 1, 0, 1], &[3usize, 2usize]).unwrap();
    let reordered =
        ColMajorArrayRef::new(&[1usize, 0, 1, 0, 0, 0, 1, 0, 1], &[3usize, 3usize]).unwrap();
    let partial =
        ColMajorArrayRef::new(&[1usize, 0, 1, 2, 1, 2, 0, 0, 0], &[3usize, 3usize]).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();

    let initial_expected = tree.evaluate(&indices, initial).unwrap();
    let initial_actual = evaluator.evaluate_batched(initial).unwrap();
    assert_typed_results::<T>(&initial_actual, &initial_expected);

    let reordered_expected = tree.evaluate(&indices, reordered).unwrap();
    let reordered_actual = evaluator.evaluate_batched(reordered).unwrap();
    assert_typed_results::<T>(&reordered_actual, &reordered_expected);
    assert!(
        evaluator.stats_for_test().message_cache_hits > 0,
        "reordered duplicate batch must reuse cached columns"
    );
    assert_eq!(reordered_actual[0], reordered_actual[2]);

    let partial_expected = tree.evaluate(&indices, partial).unwrap();
    let partial_actual = evaluator.evaluate_batched(partial).unwrap();
    assert_typed_results::<T>(&partial_actual, &partial_expected);
    let stats = evaluator.stats_for_test();
    assert!(
        stats.message_cache_hits > 0,
        "partial batch must retain hits"
    );
    assert!(
        stats.message_cache_misses > 0,
        "partial batch must compute misses"
    );
}

/// [AI Supplied] Cache-key metamorphic coverage: reordering and duplicating
/// point columns must preserve output order, while a mixed hit/miss batch
/// must agree with a fresh ordinary evaluation for every scalar kind.
#[test]
fn cached_evaluator_reordered_duplicate_and_partial_hit_batches_match() {
    assert_reordered_duplicate_and_partial_hit::<f32>();
    assert_reordered_duplicate_and_partial_hit::<f64>();
    assert_reordered_duplicate_and_partial_hit::<Complex32>();
    assert_reordered_duplicate_and_partial_hit::<Complex64>();
}

fn assert_unequal_y_tree_matches<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_unequal_y_tree::<T>();
    let values = [
        0usize, 0, 0, 0, // all-zero assignment
        1, 1, 1, 1, // all-one assignment
        0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0,
    ];
    let points = ColMajorArrayRef::new(&values, &[4usize, 5usize]).unwrap();
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

    let cold = evaluator.evaluate_batched(points).unwrap();
    let warm = evaluator.evaluate_batched(points).unwrap();
    assert_typed_results::<T>(&cold, &expected);
    assert_typed_results::<T>(&warm, &expected);
    let hub_message = evaluator
        .build_environment_cache(&1, points)
        .unwrap()
        .1
        .remove(&0)
        .unwrap();
    if matches!(T::DEBUG_DTYPE, "f32" | "c32") {
        assert!(hub_message.tensor.is_some());
        assert!(hub_message.raw_values.is_none());
    }
}

/// [AI Supplied] Topology differential coverage for a Y-comb with unequal
/// incident bond dimensions. The leaf-centered root forces the hub's
/// two-child message path, while the ordinary evaluator remains the dense
/// oracle for cold and warm calls.
#[test]
fn cached_evaluator_path_y_comb_and_unequal_bond_layouts_match() {
    assert_unequal_y_tree_matches::<f32>();
    assert_unequal_y_tree_matches::<f64>();
    assert_unequal_y_tree_matches::<Complex32>();
    assert_unequal_y_tree_matches::<Complex64>();
}

#[test]
fn tensor_from_cached_values_preserves_each_scalar_storage_kind() {
    let real32_index = DynIndex::new_dyn(2);
    let real32 = tensor_from_cached_values(
        vec![real32_index],
        vec![CachedScalar::F32(1.0), CachedScalar::F32(2.0)],
    )
    .unwrap();
    assert_eq!(real32.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);

    let real_index = DynIndex::new_dyn(2);
    let real = tensor_from_cached_values(
        vec![real_index],
        vec![CachedScalar::F64(1.0), CachedScalar::F64(2.0)],
    )
    .unwrap();
    assert_eq!(real.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);

    let complex32_index = DynIndex::new_dyn(2);
    let complex32 = tensor_from_cached_values(
        vec![complex32_index],
        vec![
            CachedScalar::C32(Complex32::new(1.0, -2.0)),
            CachedScalar::C32(Complex32::new(3.0, -4.0)),
        ],
    )
    .unwrap();
    assert_eq!(
        complex32.to_vec::<Complex32>().unwrap(),
        vec![Complex32::new(1.0, -2.0), Complex32::new(3.0, -4.0)]
    );

    let complex_index = DynIndex::new_dyn(2);
    let complex = tensor_from_cached_values(
        vec![complex_index],
        vec![
            CachedScalar::C64(Complex64::new(1.0, -2.0)),
            CachedScalar::C64(Complex64::new(3.0, -4.0)),
        ],
    )
    .unwrap();
    assert_eq!(
        complex.to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, -4.0)]
    );

    let mixed_index = DynIndex::new_dyn(2);
    let mixed = tensor_from_cached_values(
        vec![mixed_index],
        vec![
            CachedScalar::F64(5.0),
            CachedScalar::C64(Complex64::new(6.0, 7.0)),
        ],
    )
    .unwrap();
    let mixed_values = tensor_values_any(&mixed).unwrap();
    assert_eq!(mixed_values[0].real(), 5.0);
    assert_eq!(mixed_values[0].imag(), 0.0);
    assert_eq!(mixed_values[1].real(), 6.0);
    assert_eq!(mixed_values[1].imag(), 7.0);
}

/// [AI Supplied] Correctness regression for the raw-path dtype dispatch.
#[test]
fn cached_evaluator_preserves_32_bit_scalar_dtypes() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let values = [0usize, 0, 1, 1];
    let shape = [2usize, 2usize];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let real_tree = TreeTN::<_, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f32, 2.0, 3.0, 4.0])
                .unwrap(),
            IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vec![5.0_f32, 6.0, 7.0, 8.0])
                .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let real_baseline = real_tree
        .evaluate(&[s0.clone(), s1.clone()], points)
        .unwrap();
    assert_eq!(real_baseline[0].as_f64(), Some(23.0));
    assert_eq!(real_baseline[1].as_f64(), Some(46.0));
    let mut real_evaluator = TreeTNCachedEvaluator::new(
        &real_tree,
        &[s0.clone(), s1.clone()],
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let real_result = real_evaluator.evaluate_batched(points);
    let real_cached_result = real_evaluator.evaluate_batched(points);

    let c = |re| Complex32::new(re, 0.0);
    let complex_tree = TreeTN::<_, usize>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0.clone(), bond.clone()],
                vec![c(1.0), c(2.0), c(3.0), c(4.0)],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bond, s1.clone()], vec![c(5.0), c(6.0), c(7.0), c(8.0)])
                .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();
    let complex_baseline = complex_tree
        .evaluate(&[s0.clone(), s1.clone()], points)
        .unwrap();
    assert_eq!(
        complex_baseline[0].as_c64(),
        Some(Complex64::new(23.0, 0.0))
    );
    assert_eq!(
        complex_baseline[1].as_c64(),
        Some(Complex64::new(46.0, 0.0))
    );
    let mut complex_evaluator = TreeTNCachedEvaluator::new(
        &complex_tree,
        &[s0, s1],
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let complex_result = complex_evaluator.evaluate_batched(points);
    let complex_cached_result = complex_evaluator.evaluate_batched(points);

    assert!(
        real_result.is_ok()
            && real_cached_result.is_ok()
            && complex_result.is_ok()
            && complex_cached_result.is_ok(),
        "32-bit cached evaluations must succeed cold and warm: f32={real_result:?}/{real_cached_result:?}, c32={complex_result:?}/{complex_cached_result:?}"
    );
    let real_result = real_result.unwrap();
    let real_cached_result = real_cached_result.unwrap();
    let complex_result = complex_result.unwrap();
    let complex_cached_result = complex_cached_result.unwrap();
    assert_eq!(real_result[0].as_f64(), Some(23.0));
    assert_eq!(real_result[1].as_f64(), Some(46.0));
    assert_eq!(real_cached_result[0].as_f64(), Some(23.0));
    assert_eq!(real_cached_result[1].as_f64(), Some(46.0));
    assert_eq!(complex_result[0].as_c64(), Some(Complex64::new(23.0, 0.0)));
    assert_eq!(complex_result[1].as_c64(), Some(Complex64::new(46.0, 0.0)));
    assert_eq!(
        complex_cached_result[0].as_c64(),
        Some(Complex64::new(23.0, 0.0))
    );
    assert_eq!(
        complex_cached_result[1].as_c64(),
        Some(Complex64::new(46.0, 0.0))
    );
}
