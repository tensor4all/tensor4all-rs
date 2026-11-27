//! `evaluate_candidates_typed` (#1481): the candidate matrix of one edge,
//! checked against assembling its points and calling
//! `evaluate_batched_typed`.

use super::edge_cut::typed_tree_from_edges;
use super::*;

/// The comb 0 - 1 - 2 with the branch 1 - 3 - 4, every site of dimension 2,
/// and an index list that is not in node order: position `p` holds the site
/// of node `[4, 0, 3, 1, 2][p]`.
fn permuted_comb<T: CachedEvaluatorTestScalar>() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let (tree, physical) =
        typed_tree_from_edges::<T>(&[(0, 1), (1, 2), (1, 3), (3, 4)], &[2, 3, 2, 3]);
    let indices = [4usize, 0, 3, 1, 2]
        .iter()
        .map(|&node| physical[node].clone())
        .collect();
    (tree, indices)
}

/// Sites of nodes {3, 4} (the subtree behind the edge 1 - 3), deliberately
/// not sorted, and their candidates; the third repeats the first.
const LEFT_SITES: [usize; 2] = [2, 0];
fn left_candidates() -> Vec<Vec<usize>> {
    vec![vec![0, 1], vec![1, 1], vec![0, 1], vec![1, 0]]
}

/// Sites of nodes {0, 1, 2}, not sorted, and their candidates.
const RIGHT_SITES: [usize; 3] = [4, 1, 3];
fn right_candidates() -> Vec<Vec<usize>> {
    vec![vec![0, 0, 0], vec![1, 0, 1], vec![1, 1, 0]]
}

fn assemble(
    n_indices: usize,
    left: CandidateSide<'_>,
    right: CandidateSide<'_>,
    pairs: &[(usize, usize)],
) -> Vec<usize> {
    let mut data = Vec::with_capacity(n_indices * pairs.len());
    for &(i, j) in pairs {
        let mut point = vec![usize::MAX; n_indices];
        for (side, candidate) in [(left, i), (right, j)] {
            for (&site, &value) in side.sites.iter().zip(&side.candidates[candidate]) {
                point[site] = value;
            }
        }
        assert!(point.iter().all(|&value| value != usize::MAX));
        data.extend(point);
    }
    data
}

/// The flat-point oracle: the same entries assembled into global points and
/// evaluated by a separate evaluator through `evaluate_batched_typed`.
fn reference<T: CachedEvaluatorTestScalar>(
    tree: &TreeTN<IdxTensor, usize>,
    indices: &[DynIndex],
    left: CandidateSide<'_>,
    right: CandidateSide<'_>,
    pairs: &[(usize, usize)],
) -> Vec<T> {
    let data = assemble(indices.len(), left, right, pairs);
    let shape = [indices.len(), pairs.len()];
    let points = ColMajorArrayRef::new(&data, &shape).unwrap();
    TreeTNCachedEvaluator::new(tree, indices, CachedEvaluatorOptions::default())
        .unwrap()
        .evaluate_batched_typed::<T>(points, EvaluationHint::default())
        .unwrap()
}

fn all_pairs(n_left: usize, n_right: usize) -> Vec<(usize, usize)> {
    (0..n_right)
        .flat_map(|j| (0..n_left).map(move |i| (i, j)))
        .collect()
}

fn assert_close<T: CachedEvaluatorTestScalar>(actual: &[T], expected: &[T]) {
    let actual = actual
        .iter()
        .cloned()
        .map(T::into_any_scalar)
        .collect::<Vec<_>>();
    let expected = expected
        .iter()
        .cloned()
        .map(T::into_any_scalar)
        .collect::<Vec<_>>();
    assert_typed_results::<T>(&actual, &expected);
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn all_is_column_major_with_left_fastest<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = permuted_comb::<T>();
    let (lc, rc) = (left_candidates(), right_candidates());
    let left = CandidateSide {
        sites: &LEFT_SITES,
        candidates: &lc,
    };
    let right = CandidateSide {
        sites: &RIGHT_SITES,
        candidates: &rc,
    };
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

    let values = evaluator
        .evaluate_candidates_typed::<T>(left, right, CandidateEntries::All)
        .unwrap();
    assert_eq!(values.len(), lc.len() * rc.len());
    // Entry (i, j) sits at i + n_left * j: each one against its own point.
    for j in 0..rc.len() {
        for i in 0..lc.len() {
            let expected = reference::<T>(&tree, &indices, left, right, &[(i, j)]);
            assert_close(&values[i + lc.len() * j..=i + lc.len() * j], &expected);
        }
    }
    // Swapping the sides transposes the matrix.
    let swapped = evaluator
        .evaluate_candidates_typed::<T>(right, left, CandidateEntries::All)
        .unwrap();
    let transposed = (0..lc.len())
        .flat_map(|i| (0..rc.len()).map(move |j| (i, j)))
        .map(|(i, j)| values[i + lc.len() * j])
        .collect::<Vec<_>>();
    assert_close(&swapped, &transposed);
}

#[test]
fn all_is_column_major_with_left_fastest_for_every_scalar_kind() {
    all_is_column_major_with_left_fastest::<f32>();
    all_is_column_major_with_left_fastest::<f64>();
    all_is_column_major_with_left_fastest::<Complex32>();
    all_is_column_major_with_left_fastest::<Complex64>();
}

fn pairs_follow_the_given_order<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = permuted_comb::<T>();
    let (lc, rc) = (left_candidates(), right_candidates());
    let left = CandidateSide {
        sites: &LEFT_SITES,
        candidates: &lc,
    };
    let right = CandidateSide {
        sites: &RIGHT_SITES,
        candidates: &rc,
    };
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    // Not in matrix order, with a repeated pair; candidate 2 is never named.
    let pairs = [(3, 2), (0, 0), (1, 2), (3, 2), (0, 1)];

    let values = evaluator
        .evaluate_candidates_typed::<T>(left, right, CandidateEntries::Pairs(&pairs))
        .unwrap();
    assert_close(
        &values,
        &reference::<T>(&tree, &indices, left, right, &pairs),
    );
    // Each entry equals its own single-point evaluation, so the order is the
    // order given, not a sorted or deduplicated one.
    for (k, pair) in pairs.iter().enumerate() {
        assert_close(
            &values[k..=k],
            &reference::<T>(&tree, &indices, left, right, std::slice::from_ref(pair)),
        );
    }
    // The pairs agree with the matching entries of the full matrix.
    let all = evaluator
        .evaluate_candidates_typed::<T>(left, right, CandidateEntries::All)
        .unwrap();
    let gathered = pairs
        .iter()
        .map(|&(i, j)| all[i + lc.len() * j])
        .collect::<Vec<_>>();
    assert_close(&values, &gathered);
}

#[test]
fn pairs_follow_the_given_order_for_every_scalar_kind() {
    pairs_follow_the_given_order::<f32>();
    pairs_follow_the_given_order::<f64>();
    pairs_follow_the_given_order::<Complex32>();
    pairs_follow_the_given_order::<Complex64>();
}

fn equals_flat_evaluation_serial_and_on_lanes<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = permuted_comb::<T>();
    let (lc, rc) = (left_candidates(), right_candidates());
    let left = CandidateSide {
        sites: &LEFT_SITES,
        candidates: &lc,
    };
    let right = CandidateSide {
        sites: &RIGHT_SITES,
        candidates: &rc,
    };
    let full = all_pairs(lc.len(), rc.len());
    let expected_all = reference::<T>(&tree, &indices, left, right, &full);
    let pairs = [(1, 0), (2, 2), (0, 1), (3, 0), (1, 1), (0, 2), (2, 0)];
    let expected_pairs = reference::<T>(&tree, &indices, left, right, &pairs);
    let pool = pool(3);

    let mut serial_pairs = None;
    for parallelism in [
        Parallelism::Serial,
        Parallelism::Pool {
            pool: &pool,
            budget: 2,
        },
        Parallelism::Pool {
            pool: &pool,
            budget: 3,
        },
    ] {
        // A fresh evaluator each time, so every run computes its messages.
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
        let all = evaluator
            .evaluate_candidates_typed_with_parallelism::<T>(
                left,
                right,
                CandidateEntries::All,
                parallelism,
            )
            .unwrap();
        assert_close(&all, &expected_all);
        let values = evaluator
            .evaluate_candidates_typed_with_parallelism::<T>(
                left,
                right,
                CandidateEntries::Pairs(&pairs),
                parallelism,
            )
            .unwrap();
        assert_close(&values, &expected_pairs);
        // Lanes split the pairs on whole entries: bit-identical to serial.
        match &serial_pairs {
            None => serial_pairs = Some(values),
            Some(serial) => assert_eq!(&values, serial),
        }
    }
}

#[test]
fn equals_flat_evaluation_serial_and_with_more_than_one_lane() {
    equals_flat_evaluation_serial_and_on_lanes::<f32>();
    equals_flat_evaluation_serial_and_on_lanes::<f64>();
    equals_flat_evaluation_serial_and_on_lanes::<Complex32>();
    equals_flat_evaluation_serial_and_on_lanes::<Complex64>();
}

#[test]
fn every_edge_of_the_tree_matches_flat_evaluation() {
    // Both orientations of every edge, on a tree with a degree-3 node, with
    // one point per internal chunk so the message path chunks too.
    let edges = [(0, 1), (1, 2), (1, 3), (3, 4)];
    let (tree, physical) = typed_tree_from_edges::<Complex64>(&edges, &[2, 3, 2, 3]);
    let components: [&[usize]; 4] = [&[0], &[0, 1, 3, 4], &[0, 1, 2], &[0, 1, 2, 3]];
    for max_batch_points in [None, Some(1)] {
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &physical,
            CachedEvaluatorOptions {
                max_batch_points,
                ..Default::default()
            },
        )
        .unwrap();
        for component in components {
            let complement = (0..5)
                .filter(|s| !component.contains(s))
                .collect::<Vec<_>>();
            let odometer = |n_sites: usize| {
                (0..1usize << n_sites)
                    .map(|code| (0..n_sites).map(|k| (code >> k) & 1).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            };
            let (a, b) = (odometer(component.len()), odometer(complement.len()));
            for (left, right) in [
                (
                    CandidateSide {
                        sites: component,
                        candidates: &a,
                    },
                    CandidateSide {
                        sites: &complement,
                        candidates: &b,
                    },
                ),
                (
                    CandidateSide {
                        sites: &complement,
                        candidates: &b,
                    },
                    CandidateSide {
                        sites: component,
                        candidates: &a,
                    },
                ),
            ] {
                let values = evaluator
                    .evaluate_candidates_typed::<Complex64>(left, right, CandidateEntries::All)
                    .unwrap();
                let full = all_pairs(left.candidates.len(), right.candidates.len());
                assert_close(
                    &values,
                    &reference::<Complex64>(&tree, &physical, left, right, &full),
                );
            }
        }
    }
}

#[test]
fn all_is_assembled_in_bounded_blocks_with_the_same_values() {
    let (tree, indices) = permuted_comb::<f64>();
    let (lc, rc) = (left_candidates(), right_candidates());
    let left = CandidateSide {
        sites: &LEFT_SITES,
        candidates: &lc,
    };
    let right = CandidateSide {
        sites: &RIGHT_SITES,
        candidates: &rc,
    };
    let expected = reference::<f64>(&tree, &indices, left, right, &all_pairs(lc.len(), rc.len()));
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    // One right column per block (and a block smaller than one column).
    for block_entries in [1, 4, 5, 8, usize::MAX] {
        let values = evaluator
            .evaluate_candidates_cached(
                left,
                right,
                CandidateEntries::All,
                Parallelism::Serial,
                block_entries,
            )
            .and_then(|values| cached_values_into_typed::<f64>(values).map_err(Into::into))
            .unwrap();
        assert_close(&values, &expected);
    }
}

#[test]
fn candidate_messages_go_through_the_shared_message_caches() {
    let (tree, indices) = permuted_comb::<f64>();
    let (lc, rc) = (left_candidates(), right_candidates());
    let left = CandidateSide {
        sites: &LEFT_SITES,
        candidates: &lc,
    };
    let right = CandidateSide {
        sites: &RIGHT_SITES,
        candidates: &rc,
    };
    let expected = reference::<f64>(&tree, &indices, left, right, &all_pairs(lc.len(), rc.len()));

    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let cold = evaluator
        .evaluate_candidates_typed::<f64>(left, right, CandidateEntries::All)
        .unwrap();
    assert!(evaluator.stats_for_test().message_cache_misses > 0);
    // The two directed messages of the edge 1 - 3 are now in the ordinary
    // per-edge caches, and a repeated request is answered from them.
    assert!(evaluator.message_caches[&(3, 1)].key_count() > 0);
    assert!(evaluator.message_caches[&(1, 3)].key_count() > 0);
    let warm = evaluator
        .evaluate_candidates_typed::<f64>(left, right, CandidateEntries::All)
        .unwrap();
    let stats = evaluator.stats_for_test();
    assert_eq!(stats.message_cache_misses, 0);
    assert!(stats.message_cache_hits > 0);
    assert_eq!(cold, warm);
    assert_close(&warm, &expected);
    // The same messages serve a hinted flat batch across the same edge.
    let data = assemble(indices.len(), left, right, &[(1, 2)]);
    let shape = [indices.len(), 1];
    let points = ColMajorArrayRef::new(&data, &shape).unwrap();
    evaluator
        .evaluate_batched_typed::<f64>(points, EvaluationHint::around(3))
        .unwrap();
    assert!(evaluator.message_caches[&(1, 3)].hits() > 0);

    // Without retention the values are unchanged.
    let mut uncached = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            message_cache_max_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    for _ in 0..2 {
        let values = uncached
            .evaluate_candidates_typed::<f64>(left, right, CandidateEntries::All)
            .unwrap();
        assert_close(&values, &expected);
    }
}

#[test]
fn mixed_real_and_complex_nodes_are_promoted() {
    let (tree, physical) = typed_tree_from_edges::<f64>(&[(0, 1), (1, 2)], &[2, 3]);
    let complex = IdxTensor::from_dense(
        tree.tensor(tree.node_index(&2).unwrap())
            .unwrap()
            .indices()
            .to_vec(),
        (0..6)
            .map(|k| Complex64::new(k as f64 * 0.5 - 1.0, 0.25 * k as f64))
            .collect(),
    )
    .unwrap();
    let mut tensors = (0..2)
        .map(|node| {
            tree.tensor(tree.node_index(&node).unwrap())
                .unwrap()
                .clone()
        })
        .collect::<Vec<_>>();
    tensors.push(complex);
    let mixed = TreeTN::<_, usize>::from_tensors(tensors, vec![0, 1, 2]).unwrap();
    let a = vec![vec![0], vec![1]];
    let b = vec![vec![0, 0], vec![1, 0], vec![1, 1]];
    // The complex node is the left leaf, so the right message (1 -> 2, fed by
    // 0 -> 1) is real and the product promotes a real and a complex column.
    // Directions where a real node receives a complex child message fail in
    // the existing message path for flat batches as well, so they are not
    // exercised here.
    let left = CandidateSide {
        sites: &[2],
        candidates: &a,
    };
    let right = CandidateSide {
        sites: &[0, 1],
        candidates: &b,
    };
    let mut evaluator =
        TreeTNCachedEvaluator::new(&mixed, &physical, CachedEvaluatorOptions::default()).unwrap();
    let values = evaluator
        .evaluate_candidates_typed::<Complex64>(left, right, CandidateEntries::All)
        .unwrap();
    let expected = reference::<Complex64>(&mixed, &physical, left, right, &all_pairs(2, 3));
    assert_close(&values, &expected);
    assert!(values.iter().any(|value| value.im.abs() > 1e-6));
}

/// Nodes `0 - 1 - 2` where only the end nodes carry a site, so both edges
/// split the sites the same way.
fn tree_with_siteless_middle(names: [usize; 3]) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let a = DynIndex::new_dyn(2);
    let c = DynIndex::new_dyn(3);
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(3);
    let value = |k: usize| ((k * 7 % 11) as f64 - 5.0) / 4.0;
    let t0 =
        IdxTensor::from_dense(vec![a.clone(), b01.clone()], (0..4).map(value).collect()).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01, b12.clone()], (4..10).map(value).collect()).unwrap();
    let t2 = IdxTensor::from_dense(vec![b12, c.clone()], (10..19).map(value).collect()).unwrap();
    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], names.to_vec()).unwrap();
    (tree, vec![a, c])
}

#[test]
fn an_ambiguous_split_uses_the_edge_nearest_the_root() {
    let a = vec![vec![0], vec![1]];
    let c = vec![vec![0], vec![2], vec![1]];
    // `names[k]` is the name of the k-th node of the chain; the site-less one
    // is in the middle. The plan's first node (the smallest name) is the root.
    // Middle named 1: the root 0 is an end, so the edge {0, 1} is nearest.
    // Middle named 0: the root is the middle, both edges touch it, and the
    // tie goes to the edge whose other end comes first, {0, 1}.
    for (names, chosen, other) in [([0, 1, 2], (0, 1), (1, 2)), ([1, 0, 2], (0, 1), (0, 2))] {
        let (tree, indices) = tree_with_siteless_middle(names);
        for (left, right) in [
            (
                CandidateSide {
                    sites: &[0],
                    candidates: &a,
                },
                CandidateSide {
                    sites: &[1],
                    candidates: &c,
                },
            ),
            (
                CandidateSide {
                    sites: &[1],
                    candidates: &c,
                },
                CandidateSide {
                    sites: &[0],
                    candidates: &a,
                },
            ),
        ] {
            let mut evaluator =
                TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default())
                    .unwrap();
            let values = evaluator
                .evaluate_candidates_typed::<f64>(left, right, CandidateEntries::All)
                .unwrap();
            let full = all_pairs(left.candidates.len(), right.candidates.len());
            assert_close(
                &values,
                &reference::<f64>(&tree, &indices, left, right, &full),
            );
            // Both directions of the chosen edge were contracted; the other
            // edge carries only the message that feeds one of them.
            let used = |edge: (usize, usize)| {
                evaluator
                    .message_caches
                    .get(&edge)
                    .is_some_and(|cache| cache.key_count() > 0)
            };
            assert!(used(chosen) && used((chosen.1, chosen.0)), "{names:?}");
            assert!(!(used(other) && used((other.1, other.0))), "{names:?}");
        }
    }
}

#[test]
fn empty_sides_and_empty_pairs_give_empty_results() {
    let (tree, indices) = permuted_comb::<f64>();
    let (lc, rc) = (left_candidates(), right_candidates());
    let none: Vec<Vec<usize>> = Vec::new();
    let left = CandidateSide {
        sites: &LEFT_SITES,
        candidates: &lc,
    };
    let right = CandidateSide {
        sites: &RIGHT_SITES,
        candidates: &rc,
    };
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let empty_left = CandidateSide {
        sites: &LEFT_SITES,
        candidates: &none,
    };
    let empty_right = CandidateSide {
        sites: &RIGHT_SITES,
        candidates: &none,
    };
    for (left, right, entries) in [
        (empty_left, right, CandidateEntries::All),
        (left, empty_right, CandidateEntries::All),
        (left, right, CandidateEntries::Pairs(&[])),
    ] {
        assert!(evaluator
            .evaluate_candidates_typed::<f64>(left, right, entries)
            .unwrap()
            .is_empty());
    }
}

fn candidate_error(error: TreeTNOperationError) -> CandidateInputError {
    match error {
        TreeTNOperationError::InvalidCandidates { payload, .. } => payload,
        other => panic!("expected InvalidCandidates, got {other:?}"),
    }
}

#[test]
fn each_invalid_request_has_its_own_error() {
    let (tree, indices) = permuted_comb::<f64>();
    let (lc, rc) = (left_candidates(), right_candidates());
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let mut run = |left_sites: &[usize],
                   left_candidates: &[Vec<usize>],
                   right_sites: &[usize],
                   right_candidates: &[Vec<usize>],
                   entries: CandidateEntries<'_>| {
        candidate_error(
            evaluator
                .evaluate_candidates_typed::<f64>(
                    CandidateSide {
                        sites: left_sites,
                        candidates: left_candidates,
                    },
                    CandidateSide {
                        sites: right_sites,
                        candidates: right_candidates,
                    },
                    entries,
                )
                .unwrap_err(),
        )
    };
    let one = |n: usize| vec![vec![0; n]];

    assert_eq!(
        run(&[2, 5], &one(2), &RIGHT_SITES, &rc, CandidateEntries::All),
        CandidateInputError::SiteOutOfRange {
            side: "left",
            site: 5,
            n_indices: 5
        }
    );
    assert_eq!(
        run(&LEFT_SITES, &lc, &[4, 1, 7], &one(3), CandidateEntries::All),
        CandidateInputError::SiteOutOfRange {
            side: "right",
            site: 7,
            n_indices: 5
        }
    );
    assert_eq!(
        run(
            &[2, 2],
            &one(2),
            &[4, 1, 3, 0],
            &one(4),
            CandidateEntries::All
        ),
        CandidateInputError::DuplicateSite { site: 2 }
    );
    assert_eq!(
        run(
            &LEFT_SITES,
            &lc,
            &[4, 1, 3, 0],
            &one(4),
            CandidateEntries::All
        ),
        CandidateInputError::DuplicateSite { site: 0 }
    );
    assert_eq!(
        run(&LEFT_SITES, &lc, &[4, 3], &one(2), CandidateEntries::All),
        CandidateInputError::MissingSite { site: 1 }
    );
    // Nodes {0, 4} (positions 1 and 0) are not the sites behind any edge.
    assert_eq!(
        run(&[1, 0], &one(2), &[2, 3, 4], &one(3), CandidateEntries::All),
        CandidateInputError::NotAnEdgeSplit
    );
    assert_eq!(
        run(
            &LEFT_SITES,
            &[vec![0, 1], vec![1]],
            &RIGHT_SITES,
            &rc,
            CandidateEntries::All
        ),
        CandidateInputError::CandidateLength {
            side: "left",
            candidate: 1,
            len: 1,
            expected: 2
        }
    );
    assert_eq!(
        run(
            &LEFT_SITES,
            &lc,
            &RIGHT_SITES,
            &[vec![0, 0, 0, 0]],
            CandidateEntries::All
        ),
        CandidateInputError::CandidateLength {
            side: "right",
            candidate: 0,
            len: 4,
            expected: 3
        }
    );
    assert_eq!(
        run(
            &LEFT_SITES,
            &[vec![0, 1], vec![1, 2]],
            &RIGHT_SITES,
            &rc,
            CandidateEntries::All
        ),
        CandidateInputError::CandidateValueOutOfRange {
            side: "left",
            candidate: 1,
            site: 0,
            value: 2,
            dim: 2,
        }
    );
    assert_eq!(
        run(
            &LEFT_SITES,
            &lc,
            &RIGHT_SITES,
            &[vec![0, 0, 0], vec![0, 3, 0]],
            CandidateEntries::All
        ),
        CandidateInputError::CandidateValueOutOfRange {
            side: "right",
            candidate: 1,
            site: 1,
            value: 3,
            dim: 2,
        }
    );
    assert_eq!(
        run(
            &LEFT_SITES,
            &lc,
            &RIGHT_SITES,
            &rc,
            CandidateEntries::Pairs(&[(0, 0), (4, 1)])
        ),
        CandidateInputError::PairOutOfRange {
            pair: 1,
            left: 4,
            right: 1,
            n_left: 4,
            n_right: 3
        }
    );
    assert_eq!(
        run(
            &LEFT_SITES,
            &lc,
            &RIGHT_SITES,
            &rc,
            CandidateEntries::Pairs(&[(0, 3)])
        ),
        CandidateInputError::PairOutOfRange {
            pair: 0,
            left: 0,
            right: 3,
            n_left: 4,
            n_right: 3
        }
    );
    // A matrix whose size overflows cannot be allocated as candidate lists,
    // so the check is exercised on the sizes directly.
    assert_eq!(
        super::super::candidates::checked_matrix_len(usize::MAX / 2, 3),
        Err(CandidateInputError::MatrixSizeOverflow {
            n_left: usize::MAX / 2,
            n_right: 3
        })
    );
    assert_eq!(super::super::candidates::checked_matrix_len(4, 3), Ok(12));

    // The error renders its remedy and is the variant's source.
    let error = evaluator
        .evaluate_candidates_typed::<f64>(
            CandidateSide {
                sites: &LEFT_SITES,
                candidates: &lc,
            },
            CandidateSide {
                sites: &[4, 3],
                candidates: &one(2),
            },
            CandidateEntries::All,
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("must list every index"),
        "{error}"
    );
    let source = std::error::Error::source(&error).unwrap();
    assert_eq!(
        source.downcast_ref::<CandidateInputError>(),
        Some(&CandidateInputError::MissingSite { site: 1 })
    );
}

#[test]
fn a_single_node_tree_has_no_edge_to_split() {
    let s = DynIndex::new_dyn(2);
    let tree = TreeTN::<_, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &[s], CachedEvaluatorOptions::default()).unwrap();
    let none: Vec<Vec<usize>> = vec![vec![]];
    let one = vec![vec![1]];
    let error = evaluator
        .evaluate_candidates_typed::<f64>(
            CandidateSide {
                sites: &[],
                candidates: &none,
            },
            CandidateSide {
                sites: &[0],
                candidates: &one,
            },
            CandidateEntries::All,
        )
        .unwrap_err();
    assert_eq!(candidate_error(error), CandidateInputError::NotAnEdgeSplit);
}

#[test]
fn a_complex_tree_rejects_a_real_request() {
    let (tree, indices) = permuted_comb::<Complex64>();
    let (lc, rc) = (left_candidates(), right_candidates());
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let error = evaluator
        .evaluate_candidates_typed::<f64>(
            CandidateSide {
                sites: &LEFT_SITES,
                candidates: &lc,
            },
            CandidateSide {
                sites: &RIGHT_SITES,
                candidates: &rc,
            },
            CandidateEntries::Pairs(&[(0, 0)]),
        )
        .unwrap_err();
    let TreeTNOperationError::ScalarKindMismatch { payload, .. } = error else {
        panic!("expected ScalarKindMismatch, got {error:?}");
    };
    assert_eq!((payload.stored, payload.requested), ("c64", "f64"));
}
