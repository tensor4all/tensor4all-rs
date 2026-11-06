//! The persistent packed message cache: hits, misses, byte accounting, budgets,
//! and reuse across calls.

use super::*;

fn assert_cache_capacity_and_clear_reuse<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_three_node_chain::<T>();
    let values = [0usize, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0];
    let points = ColMajorArrayRef::new(&values, &[3usize, 4usize]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();

    let mut zero_budget = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            message_cache_max_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let zero_cold = zero_budget.evaluate_batched(points).unwrap();
    let zero_warm = zero_budget.evaluate_batched(points).unwrap();
    assert_typed_results::<T>(&zero_cold, &expected);
    assert_typed_results::<T>(&zero_warm, &expected);
    assert!(zero_budget
        .message_caches
        .values()
        .all(|cache| cache.retained_bytes() == 0));
    let zero_stats = zero_budget.stats_for_test();
    assert_eq!(zero_stats.message_cache_hits, 0);
    assert_eq!(zero_stats.message_cache_key_count, 0);
    assert_eq!(zero_stats.message_cache_logical_bytes, 0);
    assert!(zero_stats.message_cache_owned_bytes_estimate > 0);

    let one_column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let mut bounded = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            message_cache_max_bytes: one_column_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    let bounded_result = bounded.evaluate_batched(points).unwrap();
    assert_typed_results::<T>(&bounded_result, &expected);
    assert!(bounded
        .message_caches
        .values()
        .all(|cache| cache.retained_bytes() <= one_column_bytes));
    let bounded_stats = bounded.stats_for_test();
    assert!(bounded_stats.message_cache_key_count > 0);
    assert!(bounded_stats.message_cache_logical_bytes <= one_column_bytes * 2);
    assert!(
        bounded_stats.message_cache_owned_bytes_estimate
            >= bounded_stats.message_cache_logical_bytes
    );
    bounded.message_caches.clear();
    let after_clear = bounded.evaluate_batched(points).unwrap();
    assert_typed_results::<T>(&after_clear, &expected);
    assert!(
        bounded.stats_for_test().message_cache_misses > 0,
        "clearing message caches must force a fresh miss"
    );
}

/// [AI Supplied] Retention and invalidation coverage for the four scalar
/// kinds. Zero-budget and partially retaining caches must still return the
/// ordinary result, and clearing/reusing a cache must not expose stale data.
#[test]
fn cached_evaluator_capacity_zero_over_budget_and_clear_reuse_match() {
    assert_cache_capacity_and_clear_reuse::<f32>();
    assert_cache_capacity_and_clear_reuse::<f64>();
    assert_cache_capacity_and_clear_reuse::<Complex32>();
    assert_cache_capacity_and_clear_reuse::<Complex64>();
}

/// [AI Supplied] Logical payload and owned-storage accounting must be
/// distinguishable: spare vector capacity and the key map are not part of
/// the logical byte budget but are still evaluator-owned storage.
#[test]
fn packed_message_cache_reports_logical_and_owned_bytes() {
    let indexer = FlatIndexer::try_new(&[2]).unwrap();
    let key = indexer.encode(&[1]).unwrap();
    let mut cache = PackedMessageCache::<IndexKey, CachedScalar>::new(2, usize::MAX);
    cache
        .get_or_compute_batch(&[key], |_| {
            Ok::<_, anyhow::Error>(vec![vec![CachedScalar::F64(1.0), CachedScalar::F64(2.0)]])
        })
        .unwrap();

    let logical = 2 * std::mem::size_of::<CachedScalar>();
    assert_eq!(cache.logical_payload_bytes(), logical);
    assert_eq!(cache.retained_bytes(), logical);
    assert!(cache.owned_retained_bytes_estimate() >= logical);
    assert!(cache.owned_retained_bytes_estimate() > logical);
}

fn assert_wide_key_owned_bytes<T: CachedEvaluatorTestScalar>() {
    let values = vec![T::from_parts(1.25, -0.5), T::from_parts(-2.0, 0.125)];
    let column_bytes = values.len() * std::mem::size_of::<T>();
    let narrow_key = FlatIndexer::try_new(&[2]).unwrap().encode(&[1]).unwrap();
    let mut narrow = PackedMessageCache::<IndexKey, T>::new(2, column_bytes);
    narrow
        .get_or_compute_batch(&[narrow_key], |_| {
            Ok::<_, anyhow::Error>(vec![values.clone()])
        })
        .unwrap();
    assert_eq!(narrow.key_heap_bytes, 0);

    for width in [130usize, 300, 700] {
        let indexer = FlatIndexer::try_new(&vec![2; width]).unwrap();
        let first = indexer.encode(&vec![1; width]).unwrap();
        let mut other_values = vec![1; width];
        other_values[0] = 0;
        let refused = indexer.encode(&other_values).unwrap();
        let mut cache = PackedMessageCache::<IndexKey, T>::new(2, column_bytes);
        let slots = cache
            .get_or_compute_batch(
                &[first.clone(), first.clone(), refused.clone()],
                |missing| {
                    assert_eq!(missing, &[first.clone(), refused.clone()]);
                    Ok::<_, anyhow::Error>(vec![values.clone(), values.clone()])
                },
            )
            .unwrap();
        let CacheSlot::Cached(position) = slots[0] else {
            panic!("the first column must be retained");
        };
        assert_eq!(slots[1], slots[0], "duplicate keys must share one column");
        assert_eq!(cache.column(position), values);
        assert_eq!(slots[2], CacheSlot::Uncached(values.clone()));
        assert_eq!(cache.key_count(), 1);
        let admitted_heap_bytes = cache.positions.keys().next().unwrap().owned_heap_bytes();
        assert!(admitted_heap_bytes >= width.div_ceil(64) * 8);
        assert_eq!(cache.key_heap_bytes, admitted_heap_bytes);
        // Both maps and packed buffers have identical capacities and
        // inline types. The wide key's heap is their sole size difference.
        assert_eq!(cache.positions.capacity(), narrow.positions.capacity());
        assert_eq!(cache.columns.capacity(), narrow.columns.capacity());
        assert_eq!(
            cache.owned_retained_bytes_estimate(),
            narrow.owned_retained_bytes_estimate() + admitted_heap_bytes
        );
        assert_eq!(cache.logical_payload_bytes(), column_bytes);

        let owned_before = cache.owned_retained_bytes_estimate();
        let hits = cache
            .get_or_compute_batch(&[first.clone(), first], |_| {
                panic!("cached keys must not invoke compute_missing")
            })
            .unwrap_or_else(|error: anyhow::Error| panic!("cache hit failed: {error}"));
        assert_eq!(hits, vec![slots[0].clone(), slots[0].clone()]);
        assert_eq!(cache.owned_retained_bytes_estimate(), owned_before);
        let miss = cache
            .get_or_compute_batch(std::slice::from_ref(&refused), |missing| {
                assert_eq!(missing, std::slice::from_ref(&refused));
                Ok::<_, anyhow::Error>(vec![values.clone()])
            })
            .unwrap();
        assert_eq!(miss, vec![CacheSlot::Uncached(values.clone())]);
        assert_eq!(cache.key_heap_bytes, admitted_heap_bytes);
        assert_eq!(cache.owned_retained_bytes_estimate(), owned_before);

        let failure = cache.get_or_compute_batch(std::slice::from_ref(&refused), |_| {
            Err::<Vec<Vec<T>>, _>(anyhow::anyhow!("forced contraction failure"))
        });
        assert!(failure.is_err());
        let invalid = cache.get_or_compute_batch(std::slice::from_ref(&refused), |_| {
            Ok::<_, anyhow::Error>(vec![Vec::new()])
        });
        assert!(invalid.is_err());
        assert_eq!(cache.key_heap_bytes, admitted_heap_bytes);
        assert_eq!(cache.owned_retained_bytes_estimate(), owned_before);

        let mut disabled = PackedMessageCache::<IndexKey, T>::new(2, 0);
        disabled
            .get_or_compute_batch(&[refused], |_| Ok::<_, anyhow::Error>(vec![values.clone()]))
            .unwrap();
        assert_eq!(disabled.key_count(), 0);
        assert_eq!(disabled.key_heap_bytes, 0);
    }
}

/// Wide retained keys own boxed/limb storage in addition to their inline
/// map buckets. Hits, refused misses, and errors must not charge it again.
#[test]
fn packed_message_cache_accounts_for_admitted_wide_key_heap_storage() {
    assert_wide_key_owned_bytes::<f32>();
    assert_wide_key_owned_bytes::<f64>();
    assert_wide_key_owned_bytes::<Complex32>();
    assert_wide_key_owned_bytes::<Complex64>();
}

#[test]
fn packed_message_cache_accumulates_multiple_wide_keys_once() {
    let indexer = FlatIndexer::try_new(&[2; 300]).unwrap();
    let first = indexer.encode(&[1; 300]).unwrap();
    let second = indexer.encode(&[0; 300]).unwrap();
    let mut cache = PackedMessageCache::<IndexKey, f64>::new(1, usize::MAX);
    cache
        .get_or_compute_batch(&[first.clone(), second.clone(), first.clone()], |missing| {
            assert_eq!(missing, &[first.clone(), second.clone()]);
            Ok::<_, anyhow::Error>(vec![vec![1.25], vec![-2.0]])
        })
        .unwrap();
    assert_eq!(cache.key_count(), 2);
    assert_eq!(cache.key_heap_bytes, 128);
    cache
        .get_or_compute_batch(&[second, first], |_| {
            Err::<Vec<Vec<f64>>, _>(anyhow::anyhow!("cache hits must not compute"))
        })
        .unwrap();
    assert_eq!(cache.key_heap_bytes, 128);
}

/// [AI Supplied] Packed message lookup must preserve the full bit width of
/// a component key; wide keys must not be truncated to a machine word.
#[test]
fn packed_message_cache_accepts_wide_index_keys() {
    let indexer = FlatIndexer::try_new(&[2; 130]).unwrap();
    let key = indexer.encode(&vec![1usize; 130]).unwrap();
    let mut cache = PackedMessageCache::<IndexKey, CachedScalar>::new(1, usize::MAX);
    let slots = cache
        .get_or_compute_batch(std::slice::from_ref(&key), |_| {
            Ok::<_, anyhow::Error>(vec![vec![CachedScalar::F64(3.0)]])
        })
        .unwrap();
    assert!(matches!(slots.as_slice(), [CacheSlot::Cached(0)]));
    assert!(cache.contains(&key));
    assert_eq!(cache.key_count(), 1);
}

#[test]
fn packed_message_cache_computes_and_stores_new_keys() {
    let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
    let mut compute_calls = 0usize;

    let slots = cache
        .get_or_compute_batch(&[1u32, 2u32], |missing: &[u32]| {
            compute_calls += 1;
            Ok::<_, anyhow::Error>(
                missing
                    .iter()
                    .map(|k| vec![*k as f64, (*k as f64) * 10.0])
                    .collect(),
            )
        })
        .unwrap();

    assert_eq!(compute_calls, 1);
    let CacheSlot::Cached(p0) = slots[0] else {
        panic!("expected a cached slot: {:?}", slots[0])
    };
    let CacheSlot::Cached(p1) = slots[1] else {
        panic!("expected a cached slot: {:?}", slots[1])
    };
    assert_eq!(cache.column(p0), &[1.0, 10.0]);
    assert_eq!(cache.column(p1), &[2.0, 20.0]);
}

#[test]
fn packed_message_cache_reuses_columns_across_calls() {
    let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
    let mut compute_calls = 0usize;
    let compute = |missing: &[u32]| -> std::result::Result<Vec<Vec<f64>>, anyhow::Error> {
        Ok(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
    };

    let first = cache
        .get_or_compute_batch(&[1u32, 2u32], |missing| {
            compute_calls += 1;
            compute(missing)
        })
        .unwrap();

    // A later call across a batch that repeats key 1 and adds new key 3
    // must recompute only the miss (3), not the already-cached hit (1).
    let second = cache
        .get_or_compute_batch(&[1u32, 3u32], |missing| {
            compute_calls += 1;
            assert_eq!(missing, &[3u32], "must not recompute an already-cached key");
            compute(missing)
        })
        .unwrap();

    assert_eq!(compute_calls, 2);
    assert_eq!(second[0], first[0], "key 1 must resolve to the same column");
    let CacheSlot::Cached(p3) = second[1] else {
        panic!("expected a cached slot: {:?}", second[1])
    };
    assert_eq!(cache.column(p3), &[3.0, 0.0]);
}

#[test]
fn packed_message_cache_reports_cumulative_hits_and_misses() {
    let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
    let compute = |missing: &[u32]| -> std::result::Result<Vec<Vec<f64>>, anyhow::Error> {
        Ok(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
    };

    cache.get_or_compute_batch(&[1u32, 2u32], compute).unwrap();
    assert_eq!((cache.hits(), cache.misses()), (0, 2));

    // key 1 repeats (a hit), key 3 is new (a miss); duplicate key 1 within
    // the same batch counts as two hits, not one.
    cache
        .get_or_compute_batch(&[1u32, 1u32, 3u32], compute)
        .unwrap();
    assert_eq!((cache.hits(), cache.misses()), (2, 3));
}

#[test]
fn packed_message_cache_get_all_cached_returns_none_on_any_miss_without_counting() {
    let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
    cache
        .get_or_compute_batch(&[1u32], |missing| {
            Ok::<_, anyhow::Error>(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
        })
        .unwrap();

    assert!(cache.get_all_cached(&[1u32, 2u32]).is_none());
    assert_eq!(
        (cache.hits(), cache.misses()),
        (0, 1),
        "a partial-miss lookup must not touch the counters"
    );

    let positions = cache.get_all_cached(&[1u32]).unwrap();
    assert_eq!(cache.column(positions[0]), &[1.0, 0.0]);
    assert_eq!((cache.hits(), cache.misses()), (1, 1));
}

#[test]
fn packed_message_cache_records_confirmed_partial_hits() {
    let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
    cache
        .get_or_compute_batch(&[1u32, 2u32], |missing| {
            Ok::<_, anyhow::Error>(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
        })
        .unwrap();

    let hit_keys = [1u32, 1u32, 2u32];
    assert!(hit_keys.iter().all(|key| cache.contains(key)));
    cache.record_hits(hit_keys.len());

    assert_eq!((cache.hits(), cache.misses()), (3, 2));
}

#[test]
fn packed_message_cache_stops_retaining_once_budget_is_exhausted() {
    // bond_dim=2, f64 -> 16 bytes per column. Budget fits exactly one.
    let mut cache = PackedMessageCache::<u32, f64>::new(2, 16);

    let slots = cache
        .get_or_compute_batch(&[1u32, 2u32], |missing| {
            Ok::<_, anyhow::Error>(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
        })
        .unwrap();

    let CacheSlot::Cached(position) = slots[0] else {
        panic!(
            "first key should fit the budget and be cached: {:?}",
            slots[0]
        );
    };
    assert_eq!(cache.column(position), &[1.0, 0.0]);

    let CacheSlot::Uncached(ref values) = slots[1] else {
        panic!(
            "second key exceeds the budget and must not be retained: {:?}",
            slots[1]
        );
    };
    assert_eq!(values, &[2.0, 0.0]);
    assert!(
        !cache.contains(&2u32),
        "an over-budget key must not be recorded as cached"
    );

    // Recomputed on a later call, since it was never retained.
    let mut recompute_calls = 0usize;
    let slots_again = cache
        .get_or_compute_batch(&[2u32], |missing| {
            recompute_calls += 1;
            Ok::<_, anyhow::Error>(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
        })
        .unwrap();
    assert_eq!(recompute_calls, 1);
    assert!(matches!(slots_again[0], CacheSlot::Uncached(_)));
}

/// `get_or_compute_batch` trusts `compute_missing` to return exactly one
/// column per requested missing key. If a caller's closure violates that
/// contract (returns fewer), the call must report a descriptive error
/// rather than panicking.
#[test]
fn packed_message_cache_reports_an_error_when_compute_missing_returns_too_few_columns() {
    let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);

    let result = cache.get_or_compute_batch(&[1u32, 2u32], |missing| {
        // Only ever returns a column for the first requested key,
        // regardless of how many were actually missing.
        Ok::<_, anyhow::Error>(vec![vec![missing[0] as f64, 0.0]])
    });

    let error = result.expect_err("a short compute_missing result must not panic");
    assert!(
        error.to_string().contains("fewer columns"),
        "unexpected error: {error}"
    );
}

/// A persistent per-edge message cache must not change results and must
/// actually get used across separate `evaluate_batched` calls on the same
/// evaluator -- the whole point of #626/#646's review-requested cache.
#[test]
fn evaluate_batched_reuses_persistent_message_cache_across_calls() {
    let (tree, indices) = varied_three_node_chain();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [3usize, 2usize];

    let values1 = [0usize, 0, 0, 0, 0, 1];
    let points1 = ColMajorArrayRef::new(&values1, &shape).unwrap();
    let actual1 = evaluator.evaluate_batched(points1).unwrap();
    let expected1 = tree.evaluate(&indices, points1).unwrap();
    assert_scalars_close(&actual1, &expected1);

    // Repeats point0=(0,0,0) from the first call; point1=(1,0,0) is new.
    let values2 = [0usize, 0, 0, 1, 0, 0];
    let points2 = ColMajorArrayRef::new(&values2, &shape).unwrap();
    let actual2 = evaluator.evaluate_batched(points2).unwrap();
    let expected2 = tree.evaluate(&indices, points2).unwrap();
    assert_scalars_close(&actual2, &expected2);

    let stats = evaluator.stats_for_test();
    assert!(
        stats.message_cache_hits > 0,
        "expected at least one message cache hit on the second call: {stats:?}"
    );
}

#[test]
fn complex_cached_evaluator_preserves_values_across_cache_reuse() {
    let (tree, indices) = complex_three_node_chain();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [3usize, 2usize];
    let values1 = [0usize, 0, 0, 0, 0, 1];
    let points1 = ColMajorArrayRef::new(&values1, &shape).unwrap();
    let actual1 = evaluator.evaluate_batched(points1).unwrap();
    let expected1 = tree.evaluate(&indices, points1).unwrap();
    for (actual, expected) in actual1.iter().zip(expected1.iter()) {
        assert!((actual.real() - expected.real()).abs() < 1.0e-12);
        assert!((actual.imag() - expected.imag()).abs() < 1.0e-12);
    }

    let values2 = [0usize, 0, 0, 1, 0, 0];
    let points2 = ColMajorArrayRef::new(&values2, &shape).unwrap();
    let actual2 = evaluator.evaluate_batched(points2).unwrap();
    let expected2 = tree.evaluate(&indices, points2).unwrap();
    for (actual, expected) in actual2.iter().zip(expected2.iter()) {
        assert!((actual.real() - expected.real()).abs() < 1.0e-12);
        assert!((actual.imag() - expected.imag()).abs() < 1.0e-12);
    }
    assert!(evaluator.stats_for_test().message_cache_hits > 0);
}

#[test]
fn message_cache_reuses_a_subtree_when_another_site_changes() {
    let (tree, indices) = varied_three_node_chain();
    let shape = [3usize, 1usize];
    let first_values = [0usize, 0, 0];
    let second_values = [0usize, 0, 1];
    let first = ColMajorArrayRef::new(&first_values, &shape).unwrap();
    let second = ColMajorArrayRef::new(&second_values, &shape).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();

    let actual_first = evaluator.evaluate_batched(first).unwrap();
    let actual_second = evaluator.evaluate_batched(second).unwrap();
    let expected_first = tree.evaluate(&indices, first).unwrap();
    let expected_second = tree.evaluate(&indices, second).unwrap();
    assert_scalars_close(&actual_first, &expected_first);
    assert_scalars_close(&actual_second, &expected_second);
    assert!(
        evaluator.stats_for_test().message_cache_hits > 0,
        "changing a site outside node 0's subtree should reuse its cached message: {:?}",
        evaluator.stats_for_test()
    );
}

#[test]
fn zero_message_cache_budget_preserves_results_without_retaining_payload() {
    let (tree, indices) = varied_three_node_chain();
    let values = [0usize, 0, 0, 1, 0, 0];
    let shape = [3usize, 2usize];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            message_cache_max_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();

    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
    assert!(!evaluator.message_caches.is_empty());
    assert!(evaluator
        .message_caches
        .values()
        .all(|cache| cache.retained_bytes() == 0));
    assert_eq!(evaluator.stats_for_test().message_cache_hits, 0);
}
