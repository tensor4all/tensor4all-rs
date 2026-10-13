//! Grouped chain and branch contraction kernels, prepared branch slices, and
//! assignment-axis stacking.

use super::*;

#[test]
fn grouped_chain_contraction_matches_scalar_reference_for_real_values() {
    let raw = vec![
        1.0, 2.0, 3.0, 4.0, // physical value 0
        5.0, 6.0, 7.0, 8.0, // physical value 1
    ];
    let physical_values = [0usize, 1, 0, 1];
    let child_columns = [
        0.5, 1.5, // point 0
        2.0, 3.0, // point 1
        4.0, 5.0, // point 2
        6.0, 7.0, // point 3
    ];

    let spec = ChainContractionSpec {
        strides: [1, 2, 4],
        physical_axis: 0,
        parent_axis: 1,
        child_axis: 2,
        parent_dim: 2,
        child_dim: 2,
    };
    let actual = TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child_columns,
    )
    .unwrap();

    let expected = vec![
        8.0, 12.0, // physical 0, child column [0.5, 1.5]
        22.0, 32.0, // physical 1, child column [2.0, 3.0]
        29.0, 47.0, // physical 0, child column [4.0, 5.0]
        54.0, 80.0, // physical 1, child column [6.0, 7.0]
    ];
    assert_eq!(actual, expected);
}

#[test]
fn grouped_chain_contraction_matches_scalar_reference_for_complex_values() {
    let raw = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, -1.0),
        Complex64::new(3.0, 1.5),
        Complex64::new(4.0, -2.0),
        Complex64::new(5.0, 2.5),
        Complex64::new(6.0, -3.0),
        Complex64::new(7.0, 3.5),
        Complex64::new(8.0, -4.0),
    ];
    let physical_values = [0usize, 1, 0, 1];
    let child_columns = [
        Complex64::new(0.5, -0.5),
        Complex64::new(1.5, 0.25),
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -0.75),
        Complex64::new(4.0, -1.5),
        Complex64::new(5.0, 0.5),
        Complex64::new(6.0, 2.0),
        Complex64::new(7.0, -1.25),
    ];

    let spec = ChainContractionSpec {
        strides: [1, 2, 4],
        physical_axis: 0,
        parent_axis: 1,
        child_axis: 2,
        parent_dim: 2,
        child_dim: 2,
    };
    let actual = TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child_columns,
    )
    .unwrap();

    let expected = scalar_grouped_chain_reference(spec, &raw, &physical_values, &child_columns);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).norm() < 1.0e-10);
    }
}

#[test]
fn grouped_branch_contraction_matches_direct_reference_for_real_values() {
    // 2x2x2x2 tensor (physical=2, parent=2, child1=2, child2=2), axis
    // order [physical, parent, child1, child2] so strides = [1, 2, 4, 8].
    let raw: Vec<f64> = (0..16).map(|v| v as f64 + 1.0).collect();
    let spec = BranchContractionSpec {
        strides: [1, 2, 4, 8],
        physical_axis: 0,
        parent_axis: 1,
        child_axis_1: 2,
        child_axis_2: 3,
        parent_dim: 2,
        child_dim_1: 2,
        child_dim_2: 2,
    };
    // 3 points (below the BLAS-group threshold, so this always exercises
    // scalar_branch_message_contraction): point 0 at physical=0, points
    // 1-2 at physical=1.
    let physical_values = [0usize, 1, 1];
    let child1_columns = [1.0, 0.5, 2.0, 1.0, 0.25, 3.0];
    let child2_columns = [1.0, 1.0, 0.5, 2.0, 1.5, 0.5];

    let actual = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();

    let mut expected = vec![0.0f64; 3 * 2];
    for (p, &v) in physical_values.iter().enumerate() {
        for parent in 0..2 {
            let mut sum = 0.0;
            for c1 in 0..2 {
                for c2 in 0..2 {
                    let flat = v + 2 * parent + 4 * c1 + 8 * c2;
                    sum += raw[flat] * child1_columns[p * 2 + c1] * child2_columns[p * 2 + c2];
                }
            }
            expected[p * 2 + parent] = sum;
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn grouped_chain_contraction_large_real_groups_match_scalar_reference() {
    let parent_dim = 64;
    let child_dim = 64;
    let point_count = 16;
    let spec = ChainContractionSpec {
        strides: [1, 2, 2 * parent_dim],
        physical_axis: 0,
        parent_axis: 1,
        child_axis: 2,
        parent_dim,
        child_dim,
    };
    let raw: Vec<f64> = (0..2 * parent_dim * child_dim)
        .map(|value| (value % 19) as f64 - 9.0)
        .collect();
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
    let child_columns: Vec<f64> = (0..point_count * child_dim)
        .map(|value| (value % 13) as f64 - 6.0)
        .collect();

    let actual = TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child_columns,
    )
    .unwrap();
    let expected = scalar_grouped_chain_reference(spec, &raw, &physical_values, &child_columns);

    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-8));
}

#[test]
fn grouped_chain_contraction_large_complex_groups_match_scalar_reference() {
    let parent_dim = 64;
    let child_dim = 64;
    let point_count = 16;
    let spec = ChainContractionSpec {
        strides: [1, 2, 2 * parent_dim],
        physical_axis: 0,
        parent_axis: 1,
        child_axis: 2,
        parent_dim,
        child_dim,
    };
    let raw: Vec<Complex64> = (0..2 * parent_dim * child_dim)
        .map(|value| Complex64::new((value % 19) as f64 - 9.0, (value % 11) as f64 - 5.0))
        .collect();
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
    let child_columns: Vec<Complex64> = (0..point_count * child_dim)
        .map(|value| Complex64::new((value % 13) as f64 - 6.0, (value % 7) as f64 - 3.0))
        .collect();

    let actual = TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child_columns,
    )
    .unwrap();
    let expected = scalar_grouped_chain_reference(spec, &raw, &physical_values, &child_columns);

    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-8));
}

#[test]
fn grouped_branch_contraction_large_real_groups_match_scalar_reference() {
    let parent_dim = 8;
    let child_dim_1 = 8;
    let child_dim_2 = 8;
    let point_count = 16;
    let spec = BranchContractionSpec {
        strides: [1, 2, 2 * parent_dim, 2 * parent_dim * child_dim_1],
        physical_axis: 0,
        parent_axis: 1,
        child_axis_1: 2,
        child_axis_2: 3,
        parent_dim,
        child_dim_1,
        child_dim_2,
    };
    let raw: Vec<f64> = (0..2 * parent_dim * child_dim_1 * child_dim_2)
        .map(|value| (value % 19) as f64 - 9.0)
        .collect();
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
    let child1_columns: Vec<f64> = (0..point_count * child_dim_1)
        .map(|value| (value % 13) as f64 - 6.0)
        .collect();
    let child2_columns: Vec<f64> = (0..point_count * child_dim_2)
        .map(|value| (value % 11) as f64 - 5.0)
        .collect();

    let actual = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();
    let expected = scalar_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();

    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6));
}

#[cfg(feature = "diagnostics")]
#[test]
fn grouped_branch_contraction_counts_one_blas_dispatch_per_physical_group() {
    let parent_dim = 8;
    let child_dim_1 = 8;
    let child_dim_2 = 8;
    let point_count = 16;
    let spec = BranchContractionSpec {
        strides: [1, 2, 2 * parent_dim, 2 * parent_dim * child_dim_1],
        physical_axis: 0,
        parent_axis: 1,
        child_axis_1: 2,
        child_axis_2: 3,
        parent_dim,
        child_dim_1,
        child_dim_2,
    };
    let raw: Vec<f64> = (0..2 * parent_dim * child_dim_1 * child_dim_2)
        .map(|value| value as f64)
        .collect();
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
    let child1_columns = vec![1.0_f64; point_count * child_dim_1];
    let child2_columns = vec![1.0_f64; point_count * child_dim_2];

    diagnostics::reset();
    TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();

    assert_eq!(
        diagnostics::kernel_snapshot().matmul_calls,
        2,
        "each physical-value group dispatches one matrix multiplication"
    );

    let chain_parent_dim = 64;
    let chain_child_dim = 64;
    let chain_spec = ChainContractionSpec {
        strides: [1, 2, 2 * chain_parent_dim],
        physical_axis: 0,
        parent_axis: 1,
        child_axis: 2,
        parent_dim: chain_parent_dim,
        child_dim: chain_child_dim,
    };
    let chain_raw = vec![1.0_f64; 2 * chain_parent_dim * chain_child_dim];
    let chain_child_columns = vec![1.0_f64; point_count * chain_child_dim];

    diagnostics::reset();
    TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
        chain_spec,
        &chain_raw,
        &physical_values,
        &chain_child_columns,
    )
    .unwrap();

    assert_eq!(
        diagnostics::kernel_snapshot().matmul_calls,
        2,
        "each physical-value group dispatches one chain matrix multiplication"
    );
}

/// Same shape as `grouped_branch_contraction_large_real_groups_match_scalar_reference`,
/// but with `parent_axis` mapped to `raw`'s stride-1 (fastest) position
/// instead of `physical_axis` -- the contiguous-read fast path added
/// alongside the loop-invariant-hoisting fix (both for issue #671) only
/// runs when `strides[parent_axis] == 1`, and no existing test happened
/// to exercise that case (every other fixture puts `physical_axis` at
/// stride 1).
#[test]
fn grouped_branch_contraction_matches_scalar_reference_when_parent_axis_is_contiguous() {
    let parent_dim = 8;
    let child_dim_1 = 8;
    let child_dim_2 = 8;
    let physical_dim = 2;
    let point_count = 16;
    let spec = BranchContractionSpec {
        strides: [
            1,
            parent_dim,
            parent_dim * physical_dim,
            parent_dim * physical_dim * child_dim_1,
        ],
        parent_axis: 0,
        physical_axis: 1,
        child_axis_1: 2,
        child_axis_2: 3,
        parent_dim,
        child_dim_1,
        child_dim_2,
    };
    let raw: Vec<f64> = (0..parent_dim * physical_dim * child_dim_1 * child_dim_2)
        .map(|value| (value % 19) as f64 - 9.0)
        .collect();
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
    let child1_columns: Vec<f64> = (0..point_count * child_dim_1)
        .map(|value| (value % 13) as f64 - 6.0)
        .collect();
    let child2_columns: Vec<f64> = (0..point_count * child_dim_2)
        .map(|value| (value % 11) as f64 - 5.0)
        .collect();

    let actual = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();
    let expected = scalar_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();

    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6));
}

/// Same shape again, but with `child_axis_2` mapped to `raw`'s stride-1
/// position -- the second contiguous-read fast path (issue #671), added
/// after a fast-axis census on a real R=10 run found `child_axis_2` the
/// single most common fast axis among branch BLAS calls (51% of calls,
/// vs. 35% for `parent_axis` and 0% for `child_axis_1`, which is why
/// `child_axis_1` gets no dedicated fast path here). Exercises the
/// scatter-write side (`child_axis_2` is contiguous to read but not
/// `left`'s contiguous write axis -- `parent` is), which the
/// `parent_axis`-contiguous test above does not.
#[test]
fn grouped_branch_contraction_matches_scalar_reference_when_child_axis_2_is_contiguous() {
    let parent_dim = 8;
    let child_dim_1 = 8;
    let child_dim_2 = 8;
    let physical_dim = 2;
    let point_count = 16;
    let spec = BranchContractionSpec {
        strides: [
            1,
            child_dim_2,
            child_dim_2 * physical_dim,
            child_dim_2 * physical_dim * parent_dim,
        ],
        child_axis_2: 0,
        physical_axis: 1,
        parent_axis: 2,
        child_axis_1: 3,
        parent_dim,
        child_dim_1,
        child_dim_2,
    };
    let raw: Vec<f64> = (0..child_dim_2 * physical_dim * parent_dim * child_dim_1)
        .map(|value| (value % 19) as f64 - 9.0)
        .collect();
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
    let child1_columns: Vec<f64> = (0..point_count * child_dim_1)
        .map(|value| (value % 13) as f64 - 6.0)
        .collect();
    let child2_columns: Vec<f64> = (0..point_count * child_dim_2)
        .map(|value| (value % 11) as f64 - 5.0)
        .collect();

    let actual = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();
    let expected = scalar_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();

    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6));
}

#[test]
fn grouped_branch_contraction_large_complex_groups_match_scalar_reference() {
    let parent_dim = 8;
    let child_dim_1 = 8;
    let child_dim_2 = 8;
    let point_count = 16;
    let spec = BranchContractionSpec {
        strides: [1, 2, 2 * parent_dim, 2 * parent_dim * child_dim_1],
        physical_axis: 0,
        parent_axis: 1,
        child_axis_1: 2,
        child_axis_2: 3,
        parent_dim,
        child_dim_1,
        child_dim_2,
    };
    let raw: Vec<Complex64> = (0..2 * parent_dim * child_dim_1 * child_dim_2)
        .map(|value| Complex64::new((value % 19) as f64 - 9.0, (value % 11) as f64 - 5.0))
        .collect();
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
    let child1_columns: Vec<Complex64> = (0..point_count * child_dim_1)
        .map(|value| Complex64::new((value % 13) as f64 - 6.0, (value % 7) as f64 - 3.0))
        .collect();
    let child2_columns: Vec<Complex64> = (0..point_count * child_dim_2)
        .map(|value| Complex64::new((value % 9) as f64 - 4.0, (value % 5) as f64 - 2.0))
        .collect();

    let actual = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();
    let expected = scalar_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();

    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-6));
}

#[test]
fn grouped_branch_contiguous_read_fast_paths_match_complex_reference() {
    let parent_dim = 8;
    let child_dim_1 = 8;
    let child_dim_2 = 8;
    let physical_dim = 2;
    let point_count = 16;
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
    let child1_columns: Vec<Complex64> = (0..point_count * child_dim_1)
        .map(|value| Complex64::new((value % 13) as f64 - 6.0, (value % 7) as f64 - 3.0))
        .collect();
    let child2_columns: Vec<Complex64> = (0..point_count * child_dim_2)
        .map(|value| Complex64::new((value % 11) as f64 - 5.0, (value % 9) as f64 - 4.0))
        .collect();

    let check = |spec, raw: Vec<Complex64>| {
        let actual = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction(
            spec,
            &raw,
            &physical_values,
            &child1_columns,
            &child2_columns,
        )
        .unwrap();
        let expected = scalar_branch_message_contraction(
            spec,
            &raw,
            &physical_values,
            &child1_columns,
            &child2_columns,
        )
        .unwrap();

        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-6));
    };

    // The parent-axis contiguous read path.
    check(
        BranchContractionSpec {
            strides: [
                1,
                parent_dim,
                parent_dim * physical_dim,
                parent_dim * physical_dim * child_dim_1,
            ],
            parent_axis: 0,
            physical_axis: 1,
            child_axis_1: 2,
            child_axis_2: 3,
            parent_dim,
            child_dim_1,
            child_dim_2,
        },
        (0..parent_dim * physical_dim * child_dim_1 * child_dim_2)
            .map(|value| Complex64::new((value % 19) as f64 - 9.0, (value % 11) as f64 - 5.0))
            .collect(),
    );

    // The child-2 contiguous read and scatter-write path.
    check(
        BranchContractionSpec {
            strides: [
                1,
                child_dim_2,
                child_dim_2 * physical_dim,
                child_dim_2 * physical_dim * parent_dim,
            ],
            child_axis_2: 0,
            physical_axis: 1,
            parent_axis: 2,
            child_axis_1: 3,
            parent_dim,
            child_dim_1,
            child_dim_2,
        },
        (0..child_dim_2 * physical_dim * parent_dim * child_dim_1)
            .map(|value| Complex64::new((value % 23) as f64 - 11.0, (value % 7) as f64 - 3.0))
            .collect(),
    );
}

#[test]
fn prepared_branch_slice_cache_reuses_hits_and_returns_over_budget_fallbacks() {
    let spec = BranchContractionSpec {
        strides: [1, 2, 16, 128],
        physical_axis: 0,
        parent_axis: 1,
        child_axis_1: 2,
        child_axis_2: 3,
        parent_dim: 8,
        child_dim_1: 8,
        child_dim_2: 8,
    };
    let raw: Vec<f64> = (0..2 * 8 * 8 * 8).map(|value| value as f64).collect();
    let physical_values = vec![0usize; 16];
    let child1_columns = vec![1.0_f64; 16 * 8];
    let child2_columns = vec![1.0_f64; 16 * 8];
    let slice_bytes = 8 * 8 * 8 * std::mem::size_of::<f64>();
    let mut cache = PreparedBranchSliceCache::new(slice_bytes);

    let first = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction_cached(
        &mut cache,
        &0,
        &1,
        ScalarKind::F64,
        BranchMessageBatch {
            spec,
            raw: &raw,
            physical_values: &physical_values,
            child1_columns: &child1_columns,
            child2_columns: &child2_columns,
        },
    )
    .unwrap();
    let second = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction_cached(
        &mut cache,
        &0,
        &1,
        ScalarKind::F64,
        BranchMessageBatch {
            spec,
            raw: &vec![999.0; raw.len()],
            physical_values: &physical_values,
            child1_columns: &child1_columns,
            child2_columns: &child2_columns,
        },
    )
    .unwrap();
    assert_eq!(first, second, "a cache hit must reuse the immutable slice");
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.retained_bytes(), slice_bytes);

    let physical_values_with_new_slice = vec![1usize; 16];
    let third = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction_cached(
        &mut cache,
        &0,
        &1,
        ScalarKind::F64,
        BranchMessageBatch {
            spec,
            raw: &raw,
            physical_values: &physical_values_with_new_slice,
            child1_columns: &child1_columns,
            child2_columns: &child2_columns,
        },
    )
    .unwrap();
    let expected_third = scalar_branch_message_contraction(
        spec,
        &raw,
        &physical_values_with_new_slice,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();
    assert_eq!(third, expected_third);
    assert_eq!(cache.len(), 1, "an over-budget slice must not be retained");
    assert_eq!(cache.retained_bytes(), slice_bytes);

    let mut zero_budget_cache = PreparedBranchSliceCache::new(0);
    let zero_budget = TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction_cached(
        &mut zero_budget_cache,
        &0,
        &1,
        ScalarKind::F64,
        BranchMessageBatch {
            spec,
            raw: &raw,
            physical_values: &physical_values,
            child1_columns: &child1_columns,
            child2_columns: &child2_columns,
        },
    )
    .unwrap();
    assert_eq!(
        zero_budget,
        scalar_branch_message_contraction(
            spec,
            &raw,
            &physical_values,
            &child1_columns,
            &child2_columns,
        )
        .unwrap()
    );
    assert_eq!(zero_budget_cache.len(), 0);
    assert_eq!(zero_budget_cache.retained_bytes(), 0);
}

#[test]
fn prepared_branch_slices_match_scalar_for_all_real_axis_orders_and_unequal_bonds() {
    let point_count = 256;
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 3).collect();
    let child1_columns: Vec<f64> = (0..point_count * 2)
        .map(|value| (value % 17) as f64 - 8.0)
        .collect();
    let child2_columns: Vec<f64> = (0..point_count * 4)
        .map(|value| (value % 23) as f64 - 11.0)
        .collect();

    for physical_axis in 0..4 {
        for parent_axis in 0..4 {
            if parent_axis == physical_axis {
                continue;
            }
            for child_axis_1 in 0..4 {
                if child_axis_1 == physical_axis || child_axis_1 == parent_axis {
                    continue;
                }
                let child_axis_2 = (0..4)
                    .find(|axis| {
                        *axis != physical_axis && *axis != parent_axis && *axis != child_axis_1
                    })
                    .unwrap();
                let mut dims = [0usize; 4];
                dims[physical_axis] = 3;
                dims[parent_axis] = 3;
                dims[child_axis_1] = 2;
                dims[child_axis_2] = 4;
                let mut strides = [1usize; 4];
                for axis in 1..4 {
                    strides[axis] = strides[axis - 1] * dims[axis - 1];
                }
                let spec = BranchContractionSpec {
                    strides,
                    physical_axis,
                    parent_axis,
                    child_axis_1,
                    child_axis_2,
                    parent_dim: 3,
                    child_dim_1: 2,
                    child_dim_2: 4,
                };
                let raw: Vec<f64> = (0..dims.iter().product())
                    .map(|value| (value % 29) as f64 - 14.0)
                    .collect();
                let mut cache = PreparedBranchSliceCache::new(usize::MAX);
                let actual =
                    TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction_cached(
                        &mut cache,
                        &0,
                        &1,
                        ScalarKind::F64,
                        BranchMessageBatch {
                            spec,
                            raw: &raw,
                            physical_values: &physical_values,
                            child1_columns: &child1_columns,
                            child2_columns: &child2_columns,
                        },
                    )
                    .unwrap();
                let expected = scalar_branch_message_contraction(
                    spec,
                    &raw,
                    &physical_values,
                    &child1_columns,
                    &child2_columns,
                )
                .unwrap();
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| (actual - expected).abs() < 1.0e-8));
                let expected_bytes = 3 * 3 * 2 * 4 * std::mem::size_of::<f64>();
                assert_eq!(cache.retained_bytes(), expected_bytes);
            }
        }
    }
}

#[test]
fn prepared_branch_slices_match_scalar_for_all_complex_axis_orders_and_unequal_bonds() {
    let point_count = 256;
    let physical_values: Vec<usize> = (0..point_count).map(|point| point % 3).collect();
    let child1_columns: Vec<Complex64> = (0..point_count * 2)
        .map(|value| Complex64::new((value % 17) as f64 - 8.0, (value % 7) as f64 - 3.0))
        .collect();
    let child2_columns: Vec<Complex64> = (0..point_count * 4)
        .map(|value| Complex64::new((value % 23) as f64 - 11.0, (value % 5) as f64 - 2.0))
        .collect();

    for physical_axis in 0..4 {
        for parent_axis in 0..4 {
            if parent_axis == physical_axis {
                continue;
            }
            for child_axis_1 in 0..4 {
                if child_axis_1 == physical_axis || child_axis_1 == parent_axis {
                    continue;
                }
                let child_axis_2 = (0..4)
                    .find(|axis| {
                        *axis != physical_axis && *axis != parent_axis && *axis != child_axis_1
                    })
                    .unwrap();
                let mut dims = [0usize; 4];
                dims[physical_axis] = 3;
                dims[parent_axis] = 3;
                dims[child_axis_1] = 2;
                dims[child_axis_2] = 4;
                let mut strides = [1usize; 4];
                for axis in 1..4 {
                    strides[axis] = strides[axis - 1] * dims[axis - 1];
                }
                let spec = BranchContractionSpec {
                    strides,
                    physical_axis,
                    parent_axis,
                    child_axis_1,
                    child_axis_2,
                    parent_dim: 3,
                    child_dim_1: 2,
                    child_dim_2: 4,
                };
                let raw: Vec<Complex64> = (0..dims.iter().product())
                    .map(|value| {
                        Complex64::new((value % 29) as f64 - 14.0, (value % 11) as f64 - 5.0)
                    })
                    .collect();
                let mut cache = PreparedBranchSliceCache::new(usize::MAX);
                let actual =
                    TreeTNCachedEvaluator::<usize>::grouped_branch_message_contraction_cached(
                        &mut cache,
                        &0,
                        &1,
                        ScalarKind::C64,
                        BranchMessageBatch {
                            spec,
                            raw: &raw,
                            physical_values: &physical_values,
                            child1_columns: &child1_columns,
                            child2_columns: &child2_columns,
                        },
                    )
                    .unwrap();
                let expected = scalar_branch_message_contraction(
                    spec,
                    &raw,
                    &physical_values,
                    &child1_columns,
                    &child2_columns,
                )
                .unwrap();
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-8));
                let expected_bytes = 3 * 3 * 2 * 4 * std::mem::size_of::<Complex64>();
                assert_eq!(cache.retained_bytes(), expected_bytes);
            }
        }
    }
}

fn scalar_grouped_chain_reference<
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
>(
    spec: ChainContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child_columns: &[T],
) -> Vec<T> {
    let ChainContractionSpec {
        strides,
        physical_axis,
        parent_axis,
        child_axis,
        parent_dim,
        child_dim,
    } = spec;
    let mut output = vec![T::default(); parent_dim * physical_values.len()];
    for (point, &physical_value) in physical_values.iter().enumerate() {
        for parent_value in 0..parent_dim {
            let mut sum = T::default();
            for child_value in 0..child_dim {
                let mut axis_values = [0usize; 3];
                axis_values[physical_axis] = physical_value;
                axis_values[parent_axis] = parent_value;
                axis_values[child_axis] = child_value;
                let flat = axis_values[0] * strides[0]
                    + axis_values[1] * strides[1]
                    + axis_values[2] * strides[2];
                sum += raw[flat] * child_columns[point * child_dim + child_value];
            }
            output[point * parent_dim + parent_value] = sum;
        }
    }
    output
}

#[test]
fn evaluator_reuses_unequal_branch_slices_and_zero_budget_falls_back() {
    let (tree, indices) = unequal_branch_tree();
    let values1 = unequal_branch_values(0);
    let points1 = ColMajorArrayRef::new(&values1, &[4, 64]).unwrap();
    let expected1 = tree.evaluate(&indices, points1).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            max_batch_points: Some(usize::MAX),
            ..Default::default()
        },
    )
    .unwrap();
    let actual1 = evaluator.evaluate_batched(points1).unwrap();
    assert_scalars_close(&actual1, &expected1);
    assert_eq!(evaluator.prepared_branch_slices_f64.len(), 3);
    assert_eq!(
        evaluator.prepared_branch_slices_f64.retained_bytes(),
        3 * 8 * 6 * 4 * std::mem::size_of::<f64>()
    );

    let values2 = unequal_branch_values(1);
    let points2 = ColMajorArrayRef::new(&values2, &[4, 64]).unwrap();
    let expected2 = tree.evaluate(&indices, points2).unwrap();
    let actual2 = evaluator.evaluate_batched(points2).unwrap();
    assert_scalars_close(&actual2, &expected2);
    assert_eq!(evaluator.prepared_branch_slices_f64.len(), 3);

    let mut zero_budget = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            max_batch_points: Some(usize::MAX),
            branch_slice_cache_max_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let actual_zero = zero_budget.evaluate_batched(points1).unwrap();
    assert_scalars_close(&actual_zero, &expected1);
    assert_eq!(zero_budget.prepared_branch_slices_f64.len(), 0);
    assert_eq!(zero_budget.prepared_branch_slices_f64.retained_bytes(), 0);
}

#[test]
fn stack_tensors_adds_trailing_assignment_axis_in_column_major_order() {
    let batch = DynIndex::new_dyn(2);
    let i = DynIndex::new_dyn(2);
    let a = IdxTensor::from_dense(vec![i.clone()], vec![1.0_f64, 2.0]).unwrap();
    let b = IdxTensor::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap();

    let stacked = stack_tensors_with_assignment_index(&batch, &[a, b]).unwrap();

    assert_eq!(stacked.indices(), &[i, batch]);
    assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn gather_stacked_tensor_remaps_trailing_assignment_axis() {
    let source_batch = DynIndex::new_dyn(3);
    let target_batch = DynIndex::new_dyn(4);
    let i = DynIndex::new_dyn(2);
    let stacked = IdxTensor::from_dense(
        vec![i.clone(), source_batch.clone()],
        vec![10.0_f64, 11.0, 20.0, 21.0, 30.0, 31.0],
    )
    .unwrap();

    let gathered =
        gather_stacked_tensor(&stacked, &source_batch, &target_batch, &[2, 0, 2, 1]).unwrap();

    assert_eq!(gathered.indices(), &[i, target_batch]);
    assert_eq!(
        gathered.to_vec::<f64>().unwrap(),
        vec![30.0, 31.0, 10.0, 11.0, 30.0, 31.0, 20.0, 21.0]
    );
}

fn unequal_branch_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let physical = DynIndex::new_dyn(3);
    let parent_site = DynIndex::new_dyn(2);
    let child1_site = DynIndex::new_dyn(4);
    let child2_site = DynIndex::new_dyn(4);
    let parent_bond = DynIndex::new_dyn(8);
    let child1_bond = DynIndex::new_dyn(4);
    let child2_bond = DynIndex::new_dyn(6);

    // Axis order is [child1, physical, child2, parent], deliberately
    // non-canonical. The three incident bonds have unequal dimensions.
    let center_data: Vec<f64> = (0..3 * 4 * 6 * 8)
        .map(|value| (value % 31) as f64 / 17.0 - 0.8)
        .collect();
    let center = IdxTensor::from_dense(
        vec![
            child1_bond.clone(),
            physical.clone(),
            child2_bond.clone(),
            parent_bond.clone(),
        ],
        center_data,
    )
    .unwrap();
    let parent = IdxTensor::from_dense(
        vec![parent_bond, parent_site.clone()],
        (0..16)
            .map(|value| (value % 7) as f64 / 5.0 - 0.4)
            .collect(),
    )
    .unwrap();
    let child1 = IdxTensor::from_dense(
        vec![child1_bond, child1_site.clone()],
        (0..16)
            .map(|value| (value % 5) as f64 / 3.0 - 0.5)
            .collect(),
    )
    .unwrap();
    let child2 = IdxTensor::from_dense(
        vec![child2_bond, child2_site.clone()],
        (0..24)
            .map(|value| (value % 11) as f64 / 7.0 - 0.7)
            .collect(),
    )
    .unwrap();
    let tree =
        TreeTN::from_tensors(vec![center, parent, child1, child2], vec![0, 1, 2, 3]).unwrap();
    (tree, vec![physical, parent_site, child1_site, child2_site])
}

fn unequal_branch_values(offset: usize) -> Vec<usize> {
    let point_count = 64;
    let mut values = vec![0usize; 4 * point_count];
    for point in 0..point_count {
        values[4 * point] = (point + offset) % 3;
        values[4 * point + 1] = (point / 2 + offset) % 2;
        values[4 * point + 2] = (point / 3 + offset) % 4;
        values[4 * point + 3] = (point / 5 + offset) % 4;
    }
    values
}
