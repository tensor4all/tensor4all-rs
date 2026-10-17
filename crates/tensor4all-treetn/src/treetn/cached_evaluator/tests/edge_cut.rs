//! Warm edge-cut evaluation, including borrowed cached cuts and their cache
//! accounting.

use super::*;

/// [AI Supplied] A fully cached top-level message must be returned before
/// its descendant messages are even consulted. The two calls in the
/// five-site chain therefore touch only the two directed messages needed
/// by the warm edge cut, not the complete rooted postorder.
#[test]
fn warm_edge_cut_skips_descendant_reconstruction() {
    let (tree, indices) = five_node_chain();
    let values = vec![0usize; 5 * 4];
    let points = ColMajorArrayRef::new(&values, &[5usize, 4usize]).unwrap();
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

    let cold = evaluator.evaluate_batched_with_hint(points, EvaluationHint::around(0));
    assert_scalars_close(&cold.unwrap(), &expected);
    let warm = evaluator.evaluate_batched_with_hint(points, EvaluationHint::around(0));
    assert_scalars_close(&warm.unwrap(), &expected);

    let stats = evaluator.stats_for_test();
    assert_eq!(
        stats.batched_message_contract_count, 2,
        "warm edge cut should visit only the two cut-directed messages: {stats:?}"
    );
    assert_eq!(
        stats.message_cache_misses, 0,
        "the warm edge cut must not contract a descendant cache miss: {stats:?}"
    );
}

/// [AI Supplied] #671: a warm internal center needs exactly the two
/// directed messages on its selected cut, even when it has other arms.
#[test]
fn warm_internal_edge_cut_skips_unused_center_components() {
    let (tree, indices) = five_node_chain();
    let values = vec![0usize; 5 * 4];
    let points = ColMajorArrayRef::new(&values, &[5, 4]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    for _ in 0..2 {
        let actual = evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(2))
            .unwrap();
        assert_scalars_close(&actual, &expected);
    }
    let stats = evaluator.stats_for_test();
    assert_eq!(stats.message_cache_misses, 0);
    assert_eq!(
        stats.batched_message_contract_count, 2,
        "a fully cached cut must not assemble the other center components: {stats:?}"
    );
}

/// [AI Supplied] #671: already retained columns may take arithmetic
/// proportional to the bond, but must not be copied into transient heaps.
/// The byte measurement itself lives with the counting allocator in
/// `tests/allocation.rs`, because a process-wide `#[global_allocator]` must
/// not be shared with unrelated unit tests; this unit test keeps the cache
/// accounting the allocator cannot observe from outside the crate.
#[test]
fn warm_edge_cut_has_no_cache_misses_at_any_bond_dimension() {
    for bond in [2usize, 32] {
        let (tree, indices) =
            typed_tree_from_edges::<f64>(&[(0, 1), (1, 2), (2, 3), (3, 4)], &[bond; 4]);
        let values = (0..64)
            .flat_map(|point| (0..5).map(move |site| (point >> site) & 1))
            .collect::<Vec<_>>();
        let points = ColMajorArrayRef::new(&values, &[5, 64]).unwrap();
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
        let cold = evaluator
            .evaluate_batched_typed::<f64>(points, EvaluationHint::around(2))
            .unwrap();
        let warm = evaluator
            .evaluate_batched_typed::<f64>(points, EvaluationHint::around(2))
            .unwrap();
        assert_eq!(warm, cold);
        assert_eq!(evaluator.stats_for_test().message_cache_misses, 0);
    }
}

/// [AI Supplied] Borrowing preserves the legacy ordered edge-cut values
/// at degree 2/3/4, for every dtype and duplicate/reordered assignments.
#[test]
fn borrowed_cached_cut_matches_owned_assembly_for_all_scalar_kinds() {
    fn check<T: CachedEvaluatorTestScalar>() {
        for arms in 2..=4 {
            let (tree, indices) = unequal_star_tree::<T>(arms);
            let base = star_points(&indices, 12);
            let values = base
                .chunks(indices.len())
                .rev()
                .cycle()
                .take(32)
                .flat_map(|point| point.iter().copied())
                .collect::<Vec<_>>();
            let shape = [indices.len(), 32];
            let points = ColMajorArrayRef::new(&values, &shape).unwrap();
            let mut evaluator =
                TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default())
                    .unwrap();
            evaluator
                .evaluate_batched_with_hint(points, EvaluationHint::around(0))
                .unwrap();
            let (batches, environment) = evaluator.build_environment_cache(&0, points).unwrap();
            let expected = evaluator
                .contract_edge_cut_or_center(&0, points, &batches, &environment)
                .unwrap()
                .into_iter()
                .map(CachedScalar::into_any)
                .collect::<Vec<_>>();
            let retained = evaluator
                .message_caches
                .iter()
                .map(|(edge, cache)| {
                    (
                        *edge,
                        (
                            cache.key_count(),
                            cache.logical_payload_bytes(),
                            cache.owned_retained_bytes_estimate(),
                        ),
                    )
                })
                .collect::<HashMap<_, _>>();
            let actual = evaluator
                .evaluate_batched_with_hint(points, EvaluationHint::around(0))
                .unwrap();
            assert_eq!(actual, expected);
            let after = evaluator
                .message_caches
                .iter()
                .map(|(edge, cache)| {
                    (
                        *edge,
                        (
                            cache.key_count(),
                            cache.logical_payload_bytes(),
                            cache.owned_retained_bytes_estimate(),
                        ),
                    )
                })
                .collect::<HashMap<_, _>>();
            assert_eq!(after, retained);
            assert_eq!(evaluator.stats_for_test().batched_message_contract_count, 2);
            assert_eq!(evaluator.stats_for_test().message_cache_misses, 0);
        }
    }
    check::<f32>();
    check::<f64>();
    check::<Complex32>();
    check::<Complex64>();
}

/// [AI Supplied] An abandoned all-hit probe must not count any hits;
/// left-miss and right-miss batches match the original admission path.
#[test]
fn borrowed_cut_partial_hits_keep_legacy_cache_accounting() {
    let (tree, indices) = five_node_chain();
    for changed_site in [0, 4] {
        let mut candidate =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
        let mut reference =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
        let zeros = [0usize; 5];
        let cold = ColMajorArrayRef::new(&zeros, &[5, 1]).unwrap();
        for evaluator in [&mut candidate, &mut reference] {
            evaluator
                .evaluate_batched_with_hint(cold, EvaluationHint::around(2))
                .unwrap();
        }
        let mut values = zeros;
        values[changed_site] = 1;
        let points = ColMajorArrayRef::new(&values, &[5, 1]).unwrap();
        let actual = candidate
            .evaluate_batched_with_hint(points, EvaluationHint::around(2))
            .unwrap();
        let (batches, environment) = reference.build_environment_cache(&2, points).unwrap();
        let expected = reference
            .contract_edge_cut_or_center(&2, points, &batches, &environment)
            .unwrap()
            .into_iter()
            .map(CachedScalar::into_any)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(
            candidate.stats_for_test().message_cache_hits,
            reference.stats_for_test().message_cache_hits
        );
        assert_eq!(
            candidate.stats_for_test().message_cache_misses,
            reference.stats_for_test().message_cache_misses
        );
        for (edge, cache) in &candidate.message_caches {
            let expected = &reference.message_caches[edge];
            assert_eq!(
                (
                    cache.hits,
                    cache.misses,
                    cache.key_count(),
                    cache.logical_payload_bytes()
                ),
                (
                    expected.hits,
                    expected.misses,
                    expected.key_count(),
                    expected.logical_payload_bytes()
                )
            );
        }
    }
}

/// [AI Supplied] Packed columns reject malformed metadata rather than
/// panicking or silently reading a different scalar representation.
#[test]
fn borrowed_cut_checks_corrupt_cached_columns_before_recording_hits() {
    let (tree, indices) = five_node_chain();
    let zeros = [0usize; 5];
    let points = ColMajorArrayRef::new(&zeros, &[5, 1]).unwrap();
    for corruption in 0..5 {
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
        evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(2))
            .unwrap();
        let edge = (1, 2);
        let before = evaluator.message_caches[&edge].hits;
        match corruption {
            0 => {
                evaluator.parent_bond_indices.remove(&edge);
            }
            1 => {
                evaluator.message_caches.get_mut(&edge).unwrap().bond_dim += 1;
            }
            2 => {
                evaluator
                    .message_caches
                    .get_mut(&edge)
                    .unwrap()
                    .columns
                    .truncate(1);
            }
            3 => {
                evaluator.message_caches.get_mut(&edge).unwrap().columns[1] =
                    CachedScalar::F32(1.0);
            }
            _ => {
                evaluator
                    .message_caches
                    .get_mut(&edge)
                    .unwrap()
                    .positions
                    .values_mut()
                    .for_each(|position| *position = usize::MAX);
            }
        }
        assert!(evaluator.try_contract_cached_edge_cut(&2, points).is_err());
        assert_eq!(evaluator.message_caches[&edge].hits, before);
    }
    // Corrupt only metadata, without allocating an impossible payload:
    // work-count overflow must fail before result assembly or hit commit.
    let repeated_zeros = [0usize; 10];
    let repeated = ColMajorArrayRef::new(&repeated_zeros, &[5, 2]).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    evaluator
        .evaluate_batched_with_hint(repeated, EvaluationHint::around(2))
        .unwrap();
    let cut_edges = [(1, 2), (2, 1)];
    let hits_before = cut_edges.map(|edge| evaluator.message_caches[&edge].hits);
    evaluator
        .parent_bond_indices
        .insert((1, 2), DynIndex::new_dyn(usize::MAX));
    for edge in cut_edges {
        evaluator.message_caches.get_mut(&edge).unwrap().bond_dim = usize::MAX;
    }
    let error = evaluator
        .try_contract_cached_edge_cut(&2, repeated)
        .unwrap_err();
    assert!(error.to_string().contains("work count overflows"));
    assert_eq!(
        cut_edges.map(|edge| evaluator.message_caches[&edge].hits),
        hits_before
    );

    let cache = PackedMessageCache::<u32, f64>::new(2, 64);
    assert!(cache.checked_column(usize::MAX).is_err());
    assert!(column_dot(&[1.0f64], &[2.0, 3.0], |value| Some(*value)).is_err());
    let unit_bond_cache = PackedMessageCache::<u32, f64>::new(1, 64);
    assert!(unit_bond_cache.checked_column(usize::MAX).is_err());
    assert!(cached_column_dot(
        ScalarKind::F64,
        &[CachedScalar::F64(1.0)],
        &[CachedScalar::F32(2.0)]
    )
    .is_err());

    assert!(cached_column_dot(
        ScalarKind::F32,
        &[CachedScalar::F64(1.0)],
        &[CachedScalar::F64(2.0)]
    )
    .is_err());
    assert!(cached_column_dot(
        ScalarKind::C32,
        &[CachedScalar::F32(1.0)],
        &[CachedScalar::F32(2.0)]
    )
    .is_err());
    assert!(cached_column_dot(
        ScalarKind::C64,
        &[CachedScalar::F64(1.0)],
        &[CachedScalar::F64(2.0)]
    )
    .is_err());
}

/// [AI Supplied] Full-width component keys and invalid coordinates
/// remain checked when a warm cut bypasses recursive assignment building.
#[test]
fn borrowed_cut_handles_wide_keys_and_rejects_invalid_warm_coordinates() {
    let edges = (0..259).map(|node| (node, node + 1)).collect::<Vec<_>>();
    let (tree, indices) = typed_tree_from_edges::<f64>(&edges, &[1; 259]);
    let values = (0..4)
        .flat_map(|point| {
            (0..260).map(move |site| match point {
                0 | 3 => 0,
                1 => 1,
                _ => site % 2,
            })
        })
        .collect::<Vec<_>>();
    let points = ColMajorArrayRef::new(&values, &[260, 4]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    for _ in 0..2 {
        let actual = evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(130))
            .unwrap();
        assert_typed_results::<f64>(&actual, &expected);
    }
    assert_eq!(evaluator.stats_for_test().batched_message_contract_count, 2);
    assert_eq!(evaluator.stats_for_test().message_cache_misses, 0);
    for edge in [(129, 130), (130, 129)] {
        assert!(evaluator.message_caches[&edge]
            .positions
            .keys()
            .any(|key| key.owned_heap_bytes() > 0));
    }
    let mut invalid = values;
    invalid[259] = 2;
    let points = ColMajorArrayRef::new(&invalid, &[260, 4]).unwrap();
    assert!(evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(130))
        .is_err());
}

#[cfg(feature = "diagnostics")]
#[test]
fn borrowed_cut_records_only_its_two_cache_hits_without_message_kernels() {
    let (tree, indices) = five_node_chain();
    let values = [0usize; 20];
    let points = ColMajorArrayRef::new(&values, &[5, 4]).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let expected = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(2))
        .unwrap();
    diagnostics::reset();
    let actual = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(2))
        .unwrap();
    assert_eq!(actual, expected);
    let records = diagnostics::snapshot();
    assert_eq!(
        records
            .iter()
            .map(|record| record.guard_cache_hits)
            .sum::<u64>(),
        2
    );
    assert_eq!(
        records
            .iter()
            .map(|record| record.guard_cache_misses)
            .sum::<u64>(),
        0
    );
    assert_eq!(
        records
            .iter()
            .map(|record| record.guard_kernel.matmul_calls)
            .sum::<u64>(),
        0
    );
    assert!(records
        .iter()
        .any(|record| record.node == "tree:1" && record.coordination_number == 2));
    assert!(records
        .iter()
        .any(|record| record.node == "tree:2" && record.coordination_number == 2));
    assert!(!records.iter().any(|record| record.node == "tree:3"));
}

/// [AI Supplied] The final edge-cut assembly is a dot product over the
/// selected bond, so its deterministic work count is exactly one visit per
/// requested point and bond coordinate.
#[test]
fn warm_edge_cut_assembly_scales_with_edge_bond() {
    let (tree, indices) = five_node_chain();
    let values = vec![0usize; 5 * 7];
    let points = ColMajorArrayRef::new(&values, &[5usize, 7usize]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            ..Default::default()
        },
    )
    .unwrap();

    let cold = evaluator.evaluate_batched_with_hint(points, EvaluationHint::around(2));
    assert_scalars_close(&cold.unwrap(), &expected);
    let warm = evaluator.evaluate_batched_with_hint(points, EvaluationHint::around(2));
    assert_scalars_close(&warm.unwrap(), &expected);

    let stats = evaluator.stats_for_test();
    assert_eq!(stats.warm_edge_cut_assembly_visits, 7 * 2);
}

/// [AI Supplied] #718 Step 2 scaling gate for the #708 warm assembly law.
///
/// The claim under test is `points * chi_edge` **and nothing else**: the
/// assembly is a dot product over the cut bond, so doubling either factor
/// must double the work count and changing a descendant bond dimension
/// must not change it at all. A single fixed size cannot separate those
/// three statements, so each factor is swept independently at 1x/2x/4x
/// while the other two are held fixed. The exponent claim is carried by
/// the deterministic counter, never by wall clock.
///
/// Chain `0-1-2-3-4` centred on `2`: sorted neighbours of `2` are
/// `[1, 3]`, so the deterministic cut is the `(1, 2)` bond, which is
/// `bond_dims[1]` in [`typed_tree_from_edges`] edge order.
#[test]
fn warm_edge_cut_assembly_work_is_points_times_edge_bond_only() {
    const CUT_BONDS: [usize; 3] = [2, 4, 8];
    const POINT_COUNTS: [usize; 3] = [2, 4, 8];
    const DESCENDANT_BONDS: [usize; 3] = [2, 4, 8];

    fn measure_warm_assembly_visits(
        cut_bond: usize,
        descendant_bond: usize,
        n_points: usize,
    ) -> usize {
        let (tree, indices) = typed_tree_from_edges::<f64>(
            &[(0, 1), (1, 2), (2, 3), (3, 4)],
            &[descendant_bond, cut_bond, descendant_bond, descendant_bond],
        );
        let values = (0..n_points)
            .flat_map(|point| (0..indices.len()).map(move |site| (site + point) % 2))
            .collect::<Vec<_>>();
        let shape = [indices.len(), n_points];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(2),
                ..Default::default()
            },
        )
        .unwrap();

        let cold = evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(2))
            .unwrap();
        assert_scalars_close(&cold, &expected);
        let warm = evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(2))
            .unwrap();
        assert_scalars_close(&warm, &expected);

        let stats = evaluator.stats_for_test();
        assert_eq!(
            stats.message_cache_misses, 0,
            "the second identical call must be fully warm: {stats:?}"
        );
        assert!(
            stats.warm_edge_cut_assembly_visits > 0,
            "fixture did not take the warm edge-cut route: {stats:?}"
        );
        stats.warm_edge_cut_assembly_visits
    }

    // Factor 1: the requested point count, at fixed cut bond and fixed
    // descendant bonds.
    for &n_points in &POINT_COUNTS {
        assert_eq!(
            measure_warm_assembly_visits(4, 4, n_points),
            n_points * 4,
            "warm assembly must be linear in the point count"
        );
    }

    // Factor 2: the cut bond dimension, at a fixed point count.
    for &cut_bond in &CUT_BONDS {
        assert_eq!(
            measure_warm_assembly_visits(cut_bond, 4, 4),
            4 * cut_bond,
            "warm assembly must be linear in the cut bond dimension"
        );
    }

    // Factor 3: the descendant bonds, which the law says do not appear.
    // This is the property the #708 redesign actually bought: the old
    // vertex centre paid `d * product(incident bonds)` here.
    let descendant_visits =
        DESCENDANT_BONDS.map(|descendant_bond| measure_warm_assembly_visits(4, descendant_bond, 4));
    assert_eq!(
        descendant_visits,
        [4 * 4, 4 * 4, 4 * 4],
        "warm assembly must not depend on any descendant bond dimension"
    );
}

fn typed_tree_from_edges<T: CachedEvaluatorTestScalar>(
    edges: &[(usize, usize)],
    bond_dims: &[usize],
) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    assert_eq!(edges.len(), bond_dims.len());
    let n_nodes = edges
        .iter()
        .flat_map(|&(left, right)| [left, right])
        .max()
        .map_or(1, |node| node + 1);
    let physical = (0..n_nodes)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let bonds = bond_dims
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(n_nodes);
    for (node, physical_index) in physical.iter().enumerate() {
        let mut tensor_indices = vec![physical_index.clone()];
        for (edge, &(left, right)) in edges.iter().enumerate() {
            if node == left || node == right {
                tensor_indices.push(bonds[edge].clone());
            }
        }
        let size = tensor_indices.iter().map(IndexLike::dim).product();
        tensors.push(
            IdxTensor::from_dense(
                tensor_indices,
                (0..size)
                    .map(|value| {
                        T::from_parts(
                            (value + node + 1) as f64 * 0.125,
                            (value + 2 * node + 1) as f64 * 0.0625,
                        )
                    })
                    .collect(),
            )
            .unwrap(),
        );
    }
    (
        TreeTN::from_tensors(tensors, (0..n_nodes).collect()).unwrap(),
        physical,
    )
}

fn flatten_points(points: &[[usize; 6]], n_indices: usize) -> Vec<usize> {
    points
        .iter()
        .flat_map(|point| point[..n_indices].iter().copied())
        .collect()
}

fn assert_edge_cut_batch_variants<T: CachedEvaluatorTestScalar>(
    tree: &TreeTN<IdxTensor, usize>,
    indices: &[DynIndex],
    center: usize,
    batches: &[(Vec<usize>, usize)],
) {
    let mut evaluator = TreeTNCachedEvaluator::new(
        tree,
        indices,
        CachedEvaluatorOptions {
            center: Some(center),
            ..Default::default()
        },
    )
    .unwrap();
    for (values, n_points) in batches {
        let shape = [indices.len(), *n_points];
        let points = ColMajorArrayRef::new(values, &shape).unwrap();
        let expected = tree.evaluate(indices, points).unwrap();
        let actual = evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(center))
            .unwrap();
        assert_typed_results::<T>(&actual, &expected);
    }
}

/// [AI Supplied] Differential matrix for the complete edge-cut route.
/// Each fixture is evaluated cold, as a full hit, with reordered and
/// duplicate columns, with a partial hit/miss batch, and once again to
/// exercise persistent reuse. The ordinary TreeTN evaluator is the sole
/// numerical oracle.
fn edge_cut_cold_partial_reordered_and_repeated_batches_match_all_fixtures<
    T: CachedEvaluatorTestScalar,
>() {
    fn batches_for_path() -> Vec<(Vec<usize>, usize)> {
        vec![
            (
                flatten_points(
                    &[
                        [0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [2, 1, 2, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0],
                    ],
                    3,
                ),
                4,
            ),
            (
                flatten_points(
                    &[
                        [0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [2, 1, 2, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0],
                    ],
                    3,
                ),
                4,
            ),
            (
                flatten_points(
                    &[
                        [2, 1, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [2, 1, 2, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                    ],
                    3,
                ),
                5,
            ),
            (
                flatten_points(
                    &[
                        [0, 1, 1, 0, 0, 0],
                        [2, 0, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [1, 1, 2, 0, 0, 0],
                    ],
                    3,
                ),
                4,
            ),
            (
                flatten_points(
                    &[
                        [0, 1, 1, 0, 0, 0],
                        [2, 0, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [1, 1, 2, 0, 0, 0],
                    ],
                    3,
                ),
                4,
            ),
        ]
    }

    let path_batches = batches_for_path();
    let (path, path_indices) = typed_three_node_chain::<T>();
    assert_edge_cut_batch_variants::<T>(&path, &path_indices, 1, &path_batches);

    let y_batches = vec![
        (
            flatten_points(
                &[
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                ],
                4,
            ),
            4,
        ),
        (
            flatten_points(
                &[
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                ],
                4,
            ),
            4,
        ),
        (
            flatten_points(
                &[
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                ],
                4,
            ),
            4,
        ),
        (
            flatten_points(
                &[[1, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]],
                4,
            ),
            3,
        ),
        (
            flatten_points(
                &[[1, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]],
                4,
            ),
            3,
        ),
    ];
    let (y, y_indices) = typed_unequal_y_tree::<T>();
    assert_edge_cut_batch_variants::<T>(&y, &y_indices, 0, &y_batches);

    let comb_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)];
    let (comb, comb_indices) = typed_tree_from_edges::<T>(&comb_edges, &[2, 2, 2, 2, 2]);
    let comb_batches = vec![
        (
            flatten_points(
                &[
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                ],
                6,
            ),
            4,
        ),
        (
            flatten_points(
                &[
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                ],
                6,
            ),
            4,
        ),
        (
            flatten_points(
                &[
                    [0, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                ],
                6,
            ),
            4,
        ),
        (
            flatten_points(
                &[[1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0]],
                6,
            ),
            3,
        ),
        (
            flatten_points(
                &[[1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0]],
                6,
            ),
            3,
        ),
    ];
    assert_edge_cut_batch_variants::<T>(&comb, &comb_indices, 0, &comb_batches);
}

#[test]
fn edge_cut_differential_matrix_real_and_complex() {
    edge_cut_cold_partial_reordered_and_repeated_batches_match_all_fixtures::<f32>();
    edge_cut_cold_partial_reordered_and_repeated_batches_match_all_fixtures::<f64>();
    edge_cut_cold_partial_reordered_and_repeated_batches_match_all_fixtures::<Complex32>();
    edge_cut_cold_partial_reordered_and_repeated_batches_match_all_fixtures::<Complex64>();
}
