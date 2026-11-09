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
            message_cache_max_bytes: 2 * one_column_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    let bounded_result = bounded.evaluate_batched(points).unwrap();
    assert_typed_results::<T>(&bounded_result, &expected);
    let budget = bounded.options.message_cache_max_bytes;
    assert_eq!(budget, 2 * one_column_bytes);
    let bounded_stats = bounded.stats_for_test();
    assert!(bounded_stats.message_cache_key_count > 0);
    assert!(bounded_stats.message_cache_logical_bytes <= budget);
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

/// [AI Supplied] #1183: the shared budget reclaims whole columns from one
/// edge's cache oldest first, and every surviving position still points at
/// its own column in the packed buffer.
#[test]
fn packed_message_cache_evicts_oldest_columns_first() {
    let mut cache = PackedMessageCache::<u32, u32>::new(2, usize::MAX);
    let keys = [0_u32, 1, 2, 3];
    cache
        .get_or_compute_batch(&keys, |missing| {
            Ok::<_, anyhow::Error>(missing.iter().map(|key| vec![*key, key + 10]).collect())
        })
        .unwrap();
    assert_eq!(cache.key_count(), 4);

    // Nine bytes of an eight-byte column rounds up to two whole columns.
    assert_eq!(cache.evict_oldest_bytes(9), 16);
    assert_eq!(cache.key_count(), 2);
    assert_eq!(cache.logical_payload_bytes(), 16);
    assert!(!cache.contains(&0) && !cache.contains(&1));
    assert_eq!(cache.position(&2), Some(0));
    assert_eq!(cache.column(0), &[2, 12]);
    assert_eq!(cache.position(&3), Some(1));
    assert_eq!(cache.column(1), &[3, 13]);

    // Evicting past the end empties the buffer and frees nothing more. An
    // emptied cache also gives back its packed buffer: the budget bounds
    // logical payload, so a released edge must not keep the allocation of the
    // columns it no longer has.
    assert_eq!(cache.evict_oldest_bytes(1024), 16);
    assert_eq!(cache.key_count(), 0);
    assert_eq!(cache.logical_payload_bytes(), 0);
    assert_eq!(cache.columns.capacity(), 0);
    assert_eq!(cache.evict_oldest_bytes(1024), 0);

    // A wide key takes its key-owned heap payload with it.
    let indexer = FlatIndexer::try_new(&vec![2; 130]).unwrap();
    let kept = indexer.encode(&vec![1; 130]).unwrap();
    let mut dropped_values = vec![1; 130];
    dropped_values[0] = 0;
    let dropped = indexer.encode(&dropped_values).unwrap();
    let mut wide = PackedMessageCache::<IndexKey, u32>::new(2, usize::MAX);
    wide.get_or_compute_batch(&[dropped, kept.clone()], |missing| {
        Ok::<_, anyhow::Error>(missing.iter().map(|_| vec![1, 2]).collect())
    })
    .unwrap();
    let kept_heap_bytes = kept.owned_heap_bytes();
    assert_eq!(wide.key_heap_bytes, kept_heap_bytes + kept_heap_bytes);
    wide.evict_oldest_bytes(1);
    assert_eq!(wide.key_count(), 1);
    assert_eq!(wide.key_heap_bytes, kept_heap_bytes);
}

/// [AI Supplied] #1183: releasing a few columns from a large cache is not a
/// copy of the cache. Every release used to shift the surviving columns down
/// and renumber their positions, so 4 096 one-column releases of a
/// 4 096-column cache moved 201 277 440 bytes; releasing one column now only
/// advances `head`.
#[test]
fn packed_message_cache_releases_a_prefix_without_moving_the_survivors() {
    const BOND_DIM: usize = 3;
    let total = 4096usize;
    let column_bytes = BOND_DIM * std::mem::size_of::<u64>();
    let mut cache = PackedMessageCache::<u32, u64>::new(BOND_DIM, usize::MAX);
    let keys = (0..total as u32).collect::<Vec<_>>();
    cache
        .get_or_compute_batch(&keys, |missing| {
            Ok::<_, anyhow::Error>(
                missing
                    .iter()
                    .map(|key| vec![*key as u64; BOND_DIM])
                    .collect(),
            )
        })
        .unwrap();
    let newest = keys[total - 1];
    let address = cache.column(cache.position(&newest).unwrap()).as_ptr();

    // Half the buffer may still be dead prefix before it is reclaimed, so
    // these releases leave both the survivors and their addresses alone.
    for _ in 0..total / 2 - 1 {
        assert_eq!(cache.evict_oldest_bytes(column_bytes), column_bytes);
    }
    assert_eq!(cache.key_count(), total - (total / 2 - 1));
    assert_eq!(cache.head, total / 2 - 1);
    // The dead prefix is not retained payload and a released key is not served.
    assert_eq!(
        cache.logical_payload_bytes(),
        cache.key_count() * column_bytes
    );
    assert!(cache.position(&keys[0]).is_none());
    let position = cache.position(&newest).expect("the newest column survives");
    assert_eq!(cache.column(position), &[newest as u64; BOND_DIM]);
    assert_eq!(
        cache.column(position).as_ptr(),
        address,
        "a release must not move the surviving columns"
    );

    // Crossing that point reclaims the dead prefix once, and the survivors
    // still read back.
    assert_eq!(cache.evict_oldest_bytes(column_bytes), column_bytes);
    assert_eq!(
        cache.head, 0,
        "the released prefix is reclaimed once it is as large as the survivors"
    );
    assert_eq!(
        cache.column(cache.position(&newest).unwrap()),
        &[newest as u64; BOND_DIM]
    );
}

fn retained_message_cache_bytes(evaluator: &TreeTNCachedEvaluator<'_, usize>) -> usize {
    evaluator
        .message_caches
        .values()
        .map(PackedMessageCache::logical_payload_bytes)
        .sum()
}

fn alternating_points(indices: &[DynIndex], point_count: usize) -> Vec<usize> {
    let mut values = Vec::with_capacity(indices.len() * point_count);
    for point in 0..point_count {
        for site in 0..indices.len() {
            values.push((point >> site) & 1);
        }
    }
    values
}

/// [AI Supplied] #1183: the message-cache budget is one total for the
/// evaluator, not an allowance per directed edge: adding edges does not grow it.
#[test]
fn message_cache_budget_does_not_grow_with_edge_count() {
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let option = 4 * column_bytes;
    for ((tree, indices), center) in [
        (varied_three_node_chain(), 1usize),
        (five_node_chain(), 2usize),
        (four_arm_star_tree(), 0usize),
    ] {
        let values = alternating_points(&indices, 8);
        let shape = [indices.len(), 8];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(center),
                message_cache_max_bytes: option,
                ..Default::default()
            },
        )
        .unwrap();
        // Two, four and four undirected edges respectively: the total stays put
        // instead of scaling with them.
        assert_eq!(
            evaluator.options.message_cache_max_bytes, option,
            "the budget must not depend on the edge count"
        );

        let actual = evaluator.evaluate_batched(points).unwrap();
        assert_scalars_close(&actual, &tree.evaluate(&indices, points).unwrap());
        assert!(retained_message_cache_bytes(&evaluator) <= option);
    }

    // A ten-arm star: ten undirected (twenty directed) edges. Under the old
    // per-edge default this evaluator's ceiling was 8 MiB times twenty.
    let spine_site = DynIndex::new_dyn(2);
    let arms = 10usize;
    let bonds: Vec<DynIndex> = (0..arms).map(|_| DynIndex::new_dyn(2)).collect();
    let leaf_sites: Vec<DynIndex> = (0..arms).map(|_| DynIndex::new_dyn(2)).collect();
    let mut spine_indices = vec![spine_site.clone()];
    spine_indices.extend(bonds.iter().cloned());
    let spine_elements = spine_indices
        .iter()
        .map(|index| index.dim())
        .product::<usize>();
    let spine = IdxTensor::from_dense(spine_indices, vec![1.0_f64; spine_elements]).unwrap();
    let mut tensors = vec![spine];
    for (bond, site) in bonds.into_iter().zip(leaf_sites.iter().cloned()) {
        tensors.push(IdxTensor::from_dense(vec![bond, site], vec![1.0_f64; 4]).unwrap());
    }
    let wide_tree = TreeTN::<_, usize>::from_tensors(tensors, (0..=arms).collect()).unwrap();
    let wide_indices = std::iter::once(spine_site)
        .chain(leaf_sites.iter().cloned())
        .collect::<Vec<_>>();
    let mut wide_evaluator = TreeTNCachedEvaluator::new(
        &wide_tree,
        &wide_indices,
        CachedEvaluatorOptions {
            center: Some(0),
            message_cache_max_bytes: option,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        wide_evaluator.options.message_cache_max_bytes, option,
        "ten edges must not multiply the budget"
    );
    let wide_values = alternating_points(&wide_indices, 8);
    let wide_shape = [wide_indices.len(), 8];
    let wide_points = ColMajorArrayRef::new(&wide_values, &wide_shape).unwrap();
    let wide_actual = wide_evaluator.evaluate_batched(wide_points).unwrap();
    assert_scalars_close(
        &wide_actual,
        &wide_tree.evaluate(&wide_indices, wide_points).unwrap(),
    );
    assert!(retained_message_cache_bytes(&wide_evaluator) <= option);
}

/// [AI Supplied] #1183: the total is shared, so one reused edge may keep the
/// whole budget rather than an equal split of it.
#[test]
fn message_cache_lets_one_edge_use_the_whole_total() {
    // One undirected edge, so two directed ones: an equal split would give this
    // edge one of the budget's two columns.
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();
    let t1 = IdxTensor::from_dense(vec![bond, s1.clone()], vec![0.5_f64, 1.5, 2.5, 3.5]).unwrap();
    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
    let indices = vec![s0, s1];
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let values = alternating_points(&indices, 4);
    let shape = [indices.len(), 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            message_cache_max_bytes: 2 * column_bytes,
            ..Default::default()
        },
    )
    .unwrap();

    let actual = evaluator.evaluate_batched(points).unwrap();
    assert_scalars_close(&actual, &tree.evaluate(&indices, points).unwrap());
    assert_eq!(evaluator.options.message_cache_max_bytes, 2 * column_bytes);
    assert_eq!(
        retained_message_cache_bytes(&evaluator),
        2 * column_bytes,
        "one edge must be able to hold the whole shared total"
    );
    assert!(
        evaluator
            .message_caches
            .values()
            .any(|cache| cache.key_count() >= 2),
        "an equal split would have given this edge a single column"
    );
}

/// [AI Supplied] #1183: a budget with room for every column keeps the previous
/// behaviour -- the second call is all hits and nothing is recomputed.
#[test]
fn message_cache_ample_budget_keeps_every_column_warm() {
    let (tree, indices) = five_node_chain();
    let values = alternating_points(&indices, 8);
    let shape = [indices.len(), 8];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();

    let cold = evaluator.evaluate_batched(points).unwrap();
    assert_scalars_close(&cold, &expected);
    assert!(evaluator.stats_for_test().message_cache_misses > 0);

    let warm = evaluator.evaluate_batched(points).unwrap();
    assert_scalars_close(&warm, &expected);
    let stats = evaluator.stats_for_test();
    assert_eq!(stats.message_cache_misses, 0, "{stats:?}");
    assert_eq!(
        stats.message_cache_hits, stats.directed_message_count,
        "{stats:?}"
    );
}

/// [AI Supplied] #1183: a request that could not be retained even with the
/// whole budget must not evict a neighbour's columns.
#[test]
fn message_cache_does_not_evict_for_a_column_that_cannot_fit() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            message_cache_max_bytes: 2 * column_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    insert_packed_columns(&mut evaluator, (1, 2), 2, 2, 1);

    // One column is five times the whole total, so it can never be retained.
    let cap = evaluator.reserve_message_cache_bytes(&(2, 3), 5 * column_bytes, 5 * column_bytes);
    assert_eq!(
        cap, 0,
        "no room can exist for a column larger than the total"
    );
    assert_eq!(
        evaluator.message_caches[&(1, 2)].key_count(),
        2,
        "a request that cannot be retained must not evict anything"
    );
}

/// [AI Supplied] #1183: a parent whose columns were already hits must survive
/// its children's admissions: computing a child can now evict columns of the
/// message being assembled, so the merge may not look up an evicted key.
#[test]
fn message_cache_partially_cached_parent_survives_child_admission() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let narrow_values = alternating_points(&indices, 2);
    let narrow_shape = [indices.len(), 2];
    let narrow = ColMajorArrayRef::new(&narrow_values, &narrow_shape).unwrap();
    let wide_values = alternating_points(&indices, 8);
    let wide_shape = [indices.len(), 8];
    let wide = ColMajorArrayRef::new(&wide_values, &wide_shape).unwrap();

    // Unbounded retention gives the real columns of node 1's message to node 0
    // for the narrow batch. Its keys are a subset of the wide batch's keys.
    let mut generous = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            message_cache_max_bytes: usize::MAX,
            ..Default::default()
        },
    )
    .unwrap();
    generous.evaluate_batched(narrow).unwrap();
    let seeded = generous
        .message_caches
        .remove(&(1, 0))
        .expect("the narrow batch must cache node 1's message to node 0");
    assert!(seeded.key_count() > 0);

    // The same total, all of it already taken by that parent message.
    let mut tight = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            message_cache_max_bytes: 2 * column_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    tight.message_caches.insert((1, 0), seeded);
    tight.resync_message_cache_retained_bytes();

    let wide_actual = tight.evaluate_batched(wide).unwrap();
    assert_scalars_close(&wide_actual, &tree.evaluate(&indices, wide).unwrap());
    assert!(retained_message_cache_bytes(&tight) <= 2 * column_bytes);
}

/// [AI Supplied] #1183: with the budget exhausted, a re-rooting call still
/// returns exact values, keeps the retained payload within the budget, and
/// releases the previous rooting's columns instead of refusing everything.
#[test]
fn message_cache_stays_correct_while_the_shared_budget_is_exhausted() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let values = alternating_points(&indices, 8);
    let shape = [indices.len(), 8];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            message_cache_max_bytes: column_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();

    let first = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(0))
        .unwrap();
    assert_scalars_close(&first, &expected);
    let populated = evaluator
        .message_caches
        .iter()
        .filter(|(_, cache)| cache.key_count() > 0)
        .map(|(edge, _)| *edge)
        .collect::<Vec<_>>();
    assert!(!populated.is_empty(), "the cold call must retain columns");

    let second = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(4))
        .unwrap();
    assert_scalars_close(&second, &expected);
    assert!(
        populated.iter().any(|edge| evaluator
            .message_caches
            .get(edge)
            .is_none_or(|cache| cache.key_count() == 0)),
        "the exhausted budget must release a previous rooting's columns"
    );
    assert!(retained_message_cache_bytes(&evaluator) <= evaluator.options.message_cache_max_bytes);
    // The accounting must be exercised after real admissions: the other test in
    // this file that compares the running total with a fresh sum only reserves
    // against empty caches, so it compares 0 with 0.
    let summed = retained_message_cache_bytes(&evaluator);
    assert!(
        evaluator.message_cache_retained_bytes > 0,
        "admissions that retained a column must be accounted in the running total"
    );
    assert_eq!(
        evaluator.message_cache_retained_bytes, summed,
        "the running total must match the caches it mirrors after real admissions"
    );
}

/// [AI Supplied] #1183: a warm hit is a use of the edge that served it.
/// Without the stamp the steady branch of a plane would look stale to the shared
/// budget's least-recently-used eviction and lose its columns to a churning
/// neighbour.
#[test]
fn message_cache_warm_hits_refresh_the_use_stamp() {
    let (tree, indices) = five_node_chain();
    let values = alternating_points(&indices, 8);
    let shape = [indices.len(), 8];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    evaluator.evaluate_batched(points).unwrap();
    let stamps_before = message_cache_stamps(&evaluator);
    evaluator.reset_message_cache_use_stamps();
    let stamps_reset = message_cache_stamps(&evaluator);
    // The reset itself must leave a consistent state: one index entry per cache
    // that holds payload, keyed by that cache's new stamp.
    assert_message_cache_index_is_exact(&evaluator);
    assert_ne!(
        stamps_reset, stamps_before,
        "the reset must change the stamps"
    );

    let warm = evaluator.evaluate_batched(points).unwrap();
    assert_scalars_close(&warm, &tree.evaluate(&indices, points).unwrap());
    assert_eq!(evaluator.stats_for_test().message_cache_misses, 0);
    assert!(
        evaluator
            .message_caches
            .values()
            .any(|cache| cache.last_used() > 0),
        "the edges that served the all-hit call must be re-stamped as used"
    );
    // Every cache that holds payload is indexed exactly once, under its own
    // current stamp: a re-stamp that forgot the index would leave the cache
    // missing from it, and would leave a stale entry keyed by the old epoch.
    assert_message_cache_index_is_exact(&evaluator);
    // The all-hit call re-stamps the edges it serves and never moves a stamp
    // backwards. Not every resident is served by every call, so this asserts the
    // direction of the change, not that every edge was touched.
    let restamped = stamps_reset
        .iter()
        .filter(|(edge, reset)| evaluator.message_caches[edge].last_used() > *reset)
        .count();
    assert!(restamped > 0, "the all-hit call re-stamped no cache at all");
    for (edge, reset) in &stamps_reset {
        assert!(
            evaluator.message_caches[edge].last_used() >= *reset,
            "{edge:?} moved backwards: {reset} -> {}",
            evaluator.message_caches[edge].last_used()
        );
    }
}

/// [AI Supplied] #1183: the all-hit cut path re-stamps both directions it
/// serves. It is the plane's hot path -- the same x values, the hint at the top
/// of the x branch -- so without the stamp the branch that is reused on every
/// line looks stale next to a churning neighbour and loses its columns.
#[test]
fn message_cache_warm_cut_refreshes_both_directions() {
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();
    let t1 = IdxTensor::from_dense(vec![bond, s1.clone()], vec![0.5_f64, 1.5, 2.5, 3.5]).unwrap();
    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
    let indices = vec![s0, s1];
    let values = alternating_points(&indices, 4);
    let shape = [indices.len(), 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(1))
        .unwrap();
    let stamps_before = [(0, 1), (1, 0)].map(|edge| {
        (
            edge,
            evaluator
                .message_caches
                .get(&edge)
                .map(|cache| cache.last_used()),
        )
    });
    evaluator.reset_message_cache_use_stamps();
    let stamps_reset = message_cache_stamps(&evaluator);
    assert_message_cache_index_is_exact(&evaluator);
    // The reset itself hands every cache its own stamp; a shared one would put
    // two caches under one index entry before the warm call even runs.
    let mut reset_stamps = stamps_reset
        .iter()
        .map(|(_, stamp)| *stamp)
        .collect::<Vec<_>>();
    reset_stamps.sort_unstable();
    reset_stamps.dedup();
    assert_eq!(reset_stamps.len(), stamps_reset.len(), "{stamps_reset:?}");

    let warm = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(1))
        .unwrap();
    assert_scalars_close(&warm, &tree.evaluate(&indices, points).unwrap());
    let stats = evaluator.stats_for_test();
    assert_eq!(stats.message_cache_misses, 0, "{stats:?}");
    assert!(stats.warm_edge_cut_assembly_visits > 0, "{stats:?}");
    assert!(
        [(0, 1), (1, 0)].iter().all(|edge| evaluator
            .message_caches
            .get(edge)
            .is_none_or(|cache| cache.last_used() > 0)),
        "both directions of the warm cut must be re-stamped as used"
    );

    // Both directions are indexed, each under its own current stamp. Sharing one
    // epoch between them is what an earlier version did: the index then keeps a
    // single entry for two caches, and every consumer silently sees one edge.
    assert_message_cache_index_is_exact(&evaluator);
    let left = evaluator.message_caches[&(0, 1)].last_used();
    let right = evaluator.message_caches[&(1, 0)].last_used();
    assert_ne!(left, right, "the two directions must not share a use stamp");
    assert_eq!(
        evaluator.message_cache_order.len(),
        2,
        "{:?}",
        evaluator.message_cache_order
    );
    assert_eq!(evaluator.message_cache_order.get(&left), Some(&(0, 1)));
    assert_eq!(evaluator.message_cache_order.get(&right), Some(&(1, 0)));
    for (edge, before) in stamps_before {
        let after = evaluator.message_caches[&edge].last_used();
        assert_ne!(
            Some(after),
            before,
            "{edge:?} kept its pre-reset stamp, so the reset proved nothing"
        );
    }
}

/// The use stamp of every cache that holds payload, keyed by edge.
fn message_cache_stamps(
    evaluator: &TreeTNCachedEvaluator<'_, usize>,
) -> Vec<((usize, usize), u64)> {
    evaluator
        .message_caches
        .iter()
        .filter(|(_, cache)| cache.key_count() > 0)
        .map(|(edge, cache)| (*edge, cache.last_used()))
        .collect()
}

/// One index entry per cache that holds payload, keyed by that cache's stamp.
fn assert_message_cache_index_is_exact(evaluator: &TreeTNCachedEvaluator<'_, usize>) {
    let residents = message_cache_stamps(evaluator);
    assert_eq!(
        evaluator.message_cache_order.len(),
        residents.len(),
        "the index must hold one entry per cache that retains payload"
    );
    for (edge, stamp) in &residents {
        assert_eq!(
            evaluator.message_cache_order.get(stamp),
            Some(edge),
            "{edge:?} is not indexed under its current stamp {stamp}"
        );
    }
    for (stamp, edge) in &evaluator.message_cache_order {
        let cache = evaluator
            .message_caches
            .get(edge)
            .expect("every indexed edge is a cache of this evaluator");
        assert!(
            cache.key_count() > 0,
            "{edge:?} is indexed but holds no payload"
        );
        assert_eq!(
            cache.last_used(),
            *stamp,
            "{edge:?} is indexed under a stale stamp"
        );
    }
}

/// Inserts a packed cache holding `keys` columns of `bond_dim` elements on
/// `edge`, stamped `last_used` so the cross-edge LRU order is deterministic.
fn insert_packed_columns(
    evaluator: &mut TreeTNCachedEvaluator<'_, usize>,
    edge: (usize, usize),
    keys: usize,
    bond_dim: usize,
    last_used: u64,
) {
    let indexer = FlatIndexer::try_new(&[8]).unwrap();
    let key_list = (0..keys)
        .map(|key| indexer.encode(&[key]).unwrap())
        .collect::<Vec<_>>();
    let mut cache = PackedMessageCache::<IndexKey, CachedScalar>::new(bond_dim, usize::MAX);
    cache
        .get_or_compute_batch(&key_list, |missing| {
            Ok::<_, anyhow::Error>(
                missing
                    .iter()
                    .map(|_| {
                        (0..bond_dim)
                            .map(|element| CachedScalar::F64(1.0 + element as f64))
                            .collect()
                    })
                    .collect(),
            )
        })
        .unwrap();
    cache.touch(last_used);
    evaluator.message_caches.insert(edge, cache);
    evaluator.resync_message_cache_retained_bytes();
}

/// [AI Supplied] #1183: the shared budget reclaims columns from the least
/// recently used edge first, never from the edge that is being used.
#[test]
fn message_cache_reclaims_the_least_recently_used_edge_first() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            message_cache_max_bytes: 4 * column_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    // The shared total is four columns, filled by these two-column edges.
    insert_packed_columns(&mut evaluator, (0, 1), 2, 2, 1);
    insert_packed_columns(&mut evaluator, (1, 2), 2, 2, 2);
    assert_eq!(retained_message_cache_bytes(&evaluator), 4 * column_bytes);

    // A new edge's two columns have to come out of the older edge only.
    let cap = evaluator.reserve_message_cache_bytes(&(2, 3), 2 * column_bytes, column_bytes);
    assert!(cap >= 2 * column_bytes, "{cap}");
    assert_eq!(evaluator.message_caches[&(0, 1)].key_count(), 0);
    assert_eq!(evaluator.message_caches[&(1, 2)].key_count(), 2);
}

/// [AI Supplied] #1183: a reservation is whole incoming columns, so a neighbour
/// is never evicted for part of a column that admission cannot retain anyway.
#[test]
fn message_cache_does_not_evict_for_a_partial_incoming_column() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    // One two-element neighbour column plus one four-element incoming column is
    // the whole total, so exactly one of the two requested columns fits beside
    // the neighbour.
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            message_cache_max_bytes: 3 * column_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    insert_packed_columns(&mut evaluator, (0, 1), 1, 2, 1);

    let cap = evaluator.reserve_message_cache_bytes(&(1, 2), 4 * column_bytes, 2 * column_bytes);
    assert_eq!(
        cap,
        2 * column_bytes,
        "only the one column that fits may be reserved"
    );
    assert_eq!(
        evaluator.message_caches[&(0, 1)].key_count(),
        1,
        "a partial incoming column must not evict the neighbour's column"
    );
    assert_eq!(retained_message_cache_bytes(&evaluator), column_bytes);
}

/// [AI Supplied] #1183: the total is a byte budget, so its boundary is exact. A
/// column that lands on the total is retained and nothing is evicted; the same
/// column one byte past the total releases a whole neighbour column. This is the
/// off-by-one the column-granular eviction has to get right.
#[test]
fn message_cache_evicts_only_once_the_total_is_exceeded_by_one_byte() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let evaluator_with = |total: usize| {
        TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(2),
                message_cache_max_bytes: total,
                ..Default::default()
            },
        )
        .unwrap()
    };

    // Exactly the total: two neighbour columns plus the one being admitted are
    // the whole budget, so the neighbour keeps both of its columns.
    let mut exact = evaluator_with(3 * column_bytes);
    insert_packed_columns(&mut exact, (0, 1), 2, 2, 1);
    let cap = exact.reserve_message_cache_bytes(&(1, 2), column_bytes, column_bytes);
    assert_eq!(cap, column_bytes, "the admitted column must fit exactly");
    assert_eq!(
        exact.message_caches[&(0, 1)].key_count(),
        2,
        "landing exactly on the total must not evict"
    );

    // One byte short of the total: the same column overshoots by exactly one
    // byte, which releases a whole neighbour column rather than a byte of it.
    let mut one_byte_over = evaluator_with(3 * column_bytes - 1);
    insert_packed_columns(&mut one_byte_over, (0, 1), 2, 2, 1);
    let cap = one_byte_over.reserve_message_cache_bytes(&(1, 2), column_bytes, column_bytes);
    assert_eq!(
        one_byte_over.message_caches[&(0, 1)].key_count(),
        1,
        "one byte over the total must release a whole neighbour column"
    );
    assert_eq!(
        retained_message_cache_bytes(&one_byte_over),
        column_bytes,
        "the surviving total stays at or below the budget"
    );
    assert_eq!(cap, 2 * column_bytes - 1);
}

/// [AI Supplied] #1183: one admission reads one edge's payload. Summing every
/// directed cache instead made a cold sweep over `edges` caches read
/// `edges * (edges - 1)` payloads, and did so even when retention was disabled
/// (`0`) or unbounded (`usize::MAX`), where nothing can be reclaimed at all.
#[test]
fn message_cache_accounting_does_not_scan_every_edge() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let edges = 1024usize;
    for total in [0usize, usize::MAX, 64 << 20] {
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(2),
                message_cache_max_bytes: total,
                ..Default::default()
            },
        )
        .unwrap();
        for edge in 0..edges {
            evaluator.message_caches.insert(
                (edge, edges + edge),
                PackedMessageCache::<IndexKey, CachedScalar>::new(2, 0),
            );
        }
        evaluator.resync_message_cache_retained_bytes();
        reset_message_cache_payload_reads();
        for edge in 0..edges {
            evaluator.reserve_message_cache_bytes(
                &(edge, edges + edge),
                column_bytes,
                column_bytes,
            );
        }

        // Exactly one payload read per reservation under a finite budget -- the
        // edge being admitted -- and none at all when retention is disabled or
        // unbounded, where nothing can be reclaimed. A per-other-edge scan puts
        // `edges * (edges - 1)` more reads here, and dropping the one own read
        // would stop the running total from tracking admissions.
        let reads = message_cache_payload_reads();
        let expected = if total == 0 || total == usize::MAX {
            0
        } else {
            edges
        };
        assert_eq!(
            reads, expected,
            "budget {total}: {reads} payload reads for {edges} edges"
        );
        let summed = retained_message_cache_bytes(&evaluator);
        assert_eq!(
            evaluator.message_cache_retained_bytes, summed,
            "the running total must match the caches it mirrors"
        );
    }
}

fn reset_message_cache_payload_reads() {
    MESSAGE_CACHE_PAYLOAD_READS.with(|reads| reads.set(0));
}

fn message_cache_payload_reads() -> usize {
    MESSAGE_CACHE_PAYLOAD_READS.with(std::cell::Cell::get)
}

/// [AI Supplied] #1183 review: a victim search skips the caches that hold no
/// payload instead of collecting and sorting them.
#[test]
fn message_cache_victim_search_skips_caches_that_hold_nothing() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let narrow_values = alternating_points(&indices, 2);
    let narrow_shape = [indices.len(), 2];
    let narrow = ColMajorArrayRef::new(&narrow_values, &narrow_shape).unwrap();

    // One live cache, then 1 024 historical empty ones on the same evaluator.
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            message_cache_max_bytes: usize::MAX,
            ..Default::default()
        },
    )
    .unwrap();
    evaluator.evaluate_batched(narrow).unwrap();
    let live = evaluator
        .message_caches
        .remove(&(1, 0))
        .expect("the narrow batch must cache node 1's message to node 0");
    assert!(live.key_count() > 0);
    let edges = 1024usize;
    let mut shared = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            message_cache_max_bytes: live.logical_payload_bytes(),
            ..Default::default()
        },
    )
    .unwrap();
    shared.message_caches.insert((1, 0), live);
    for edge in 0..edges {
        shared.message_caches.insert(
            (edge, edges + edge),
            PackedMessageCache::<IndexKey, CachedScalar>::new(2, 0),
        );
    }
    shared.resync_message_cache_retained_bytes();

    reset_message_cache_victim_candidates();
    // One column for another edge, against a total the live cache already fills.
    shared.reserve_message_cache_bytes(&(edges, 2 * edges), column_bytes, column_bytes);
    assert_eq!(
        message_cache_victim_candidates(),
        1,
        "a victim search must consider only the one cache that holds payload"
    );
}

fn reset_message_cache_victim_candidates() {
    MESSAGE_CACHE_VICTIM_CANDIDATES.with(|candidates| candidates.set(0));
}

fn message_cache_victim_candidates() -> usize {
    MESSAGE_CACHE_VICTIM_CANDIDATES.with(std::cell::Cell::get)
}

fn reset_message_cache_victim_visits() {
    MESSAGE_CACHE_VICTIM_VISITS.with(|visits| visits.set(0));
}

fn message_cache_victim_visits() -> usize {
    MESSAGE_CACHE_VICTIM_VISITS.with(std::cell::Cell::get)
}

fn reset_message_cache_last_used_reads() {
    MESSAGE_CACHE_LAST_USED_READS.with(|reads| reads.set(0));
}

fn message_cache_last_used_reads() -> usize {
    MESSAGE_CACHE_LAST_USED_READS.with(std::cell::Cell::get)
}

/// #1209: the LRU index holds exactly the caches that retain payload, each
/// under its own use stamp, after a real run through the admission path.
///
/// The accounting fixture above inserts caches and re-syncs the index through a
/// test-only rebuild, so it cannot see whether *admission* maintains the index.
/// This one only calls `evaluate_batched`, with a one-column total so every
/// admission has to evict: a cache created by this call is stamped while it is
/// still empty and only starts retaining at the admission that follows.
#[test]
fn message_cache_index_holds_exactly_the_caches_that_retain_payload() {
    let (tree, indices) = five_node_chain();
    let column_bytes = std::mem::size_of::<CachedScalar>() * 2;
    let values = alternating_points(&indices, 8);
    let shape = [indices.len(), 8];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            message_cache_max_bytes: column_bytes,
            ..Default::default()
        },
    )
    .unwrap();

    // Two rootings, so caches are created, evicted and reused under pressure.
    for center in [0, 4, 2] {
        let actual = evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(center))
            .unwrap();
        assert_scalars_close(&actual, &expected);
    }

    let residents = evaluator
        .message_caches
        .iter()
        .filter(|(_, cache)| cache.key_count() > 0)
        .collect::<Vec<_>>();
    assert!(
        !residents.is_empty(),
        "a finite budget must retain at least one column to make this test meaningful"
    );
    assert_eq!(
        evaluator.message_cache_order.len(),
        residents.len(),
        "the index must hold one entry per cache that retains payload"
    );
    for (epoch, edge) in &evaluator.message_cache_order {
        let cache = evaluator
            .message_caches
            .get(edge)
            .expect("every indexed edge is a cache of this evaluator");
        assert!(
            cache.key_count() > 0,
            "{edge:?} is indexed but holds no payload"
        );
        assert_eq!(
            cache.last_used(),
            *epoch,
            "{edge:?} is indexed under a stale use stamp"
        );
    }
    for (edge, cache) in residents {
        assert_eq!(
            evaluator.message_cache_order.get(&cache.last_used()),
            Some(edge),
            "{edge:?} retains payload but is not in the index"
        );
    }
    assert!(retained_message_cache_bytes(&evaluator) <= column_bytes);
}

/// #1209: admitting one column per edge under a one-column total must cost the
/// caches that hold payload, not the edges the evaluator has seen.
///
/// The fixture is the one the review used: the 1 024-cache accounting set, a
/// one-column total, one column retained elsewhere, then one column reserved and
/// admitted per edge while the running total is updated. Every admission makes
/// the previous resident the only victim, so an eviction that walked the cache
/// map would visit each of the 1 024 historical caches once per admission.
#[test]
fn message_cache_pressured_admission_visits_only_the_caches_that_hold_payload() {
    let (tree, indices) = five_node_chain();
    let edges = 1024usize;
    let bond_dim = 2usize;
    let column_bytes = std::mem::size_of::<CachedScalar>() * bond_dim;
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            // Exactly one column: every admission has to evict the resident.
            message_cache_max_bytes: column_bytes,
            ..Default::default()
        },
    )
    .unwrap();

    // The accounting fixture's 1 024 historical edge caches. They hold no
    // payload, so an eviction has nothing to reclaim from them, but they are
    // the edges the evaluator has seen and they stay in the cache map.
    for edge_index in 0..edges {
        evaluator.message_caches.insert(
            (edge_index, edges + edge_index),
            PackedMessageCache::<IndexKey, CachedScalar>::new(bond_dim, 0),
        );
    }
    // The one resident, on an edge that is not itself an admission target, so
    // every admission below has a victim that is not the edge being used.
    let resident = (2 * edges, 2 * edges + 1);
    insert_packed_columns(&mut evaluator, resident, 1, bond_dim, 1);
    assert_eq!(retained_message_cache_bytes(&evaluator), column_bytes);

    reset_message_cache_last_used_reads();
    reset_message_cache_victim_visits();
    reset_message_cache_victim_candidates();
    let mut admitted = 0usize;
    let (mut reads, mut visits) = (0usize, 0usize);
    for edge_index in 0..edges {
        // The same edges the fixture populated, so the cache map keeps its size
        // and the walk cost is what the issue reports, not growth of the map.
        let edge = (edge_index, edges + edge_index);
        // Count what choosing victims costs, not what the test's own insertion
        // helper does afterwards.
        reset_message_cache_last_used_reads();
        reset_message_cache_victim_visits();
        let cap = evaluator.reserve_message_cache_bytes(&edge, column_bytes, column_bytes);
        reads += message_cache_last_used_reads();
        visits += message_cache_victim_visits();
        if cap < column_bytes {
            continue;
        }
        // Admit the column this reservation made room for. The insertion also
        // refreshes the running total from the map, so nothing here has to
        // mirror the admission path's arithmetic.
        insert_packed_columns(&mut evaluator, edge, 1, bond_dim, edge_index as u64 + 2);
        admitted += 1;
        assert!(
            evaluator.message_cache_retained_bytes <= evaluator.options.message_cache_max_bytes,
            "admission {edge_index} retained {} bytes over a {column_bytes}-byte total",
            evaluator.message_cache_retained_bytes
        );
    }
    assert_eq!(admitted, edges, "every edge must admit its one column");
    assert_eq!(
        retained_message_cache_bytes(&evaluator),
        column_bytes,
        "a one-column total leaves exactly one resident at the end"
    );

    // One resident at a time, so a victim search proportional to the caches
    // that hold payload traverses about one cache per admission. Measured with
    // the pre-#1209 walk restored (the same counters, same counter position):
    // 1 049 600 traversals and 1 024 `last_used()` reads for this fixture,
    // against 1 024 and 0 here. The bound is stated generously -- two per
    // admission -- so one extra resident is allowed, while any walk over the
    // historical caches per admission is far outside it. The traversal counter
    // runs before every filter, so a walk over `message_caches` counts the empty
    // caches too; counting inside `map()` did not, which is how an earlier
    // version of this test could not see the regression.
    assert!(
        visits <= 2 * edges,
        "{visits} cache visits for {edges} pressured admissions; the search must follow the caches that hold payload"
    );
    assert!(
        reads <= 2 * edges,
        "{reads} last_used() reads for {edges} pressured admissions; the search must follow the caches that hold payload"
    );
}

/// [AI Supplied] #1183 review: re-admitting a key released with the dead prefix
/// replaces its stale entry, so its heap bytes are charged once, not twice.
#[test]
fn packed_message_cache_charges_a_readmitted_key_once() {
    // 129-bit keys live on the heap (`IndexKey::owned_heap_bytes`), 64-bit ones
    // do not, so this uses wide keys to observe the counter.
    let indexer = FlatIndexer::try_new(&vec![2usize; 129]).expect("a 129-bit index space");
    let key_at = |mut value: usize| {
        let mut bits = vec![1usize; 129];
        for bit in bits.iter_mut().skip(64) {
            *bit = value & 1;
            value >>= 1;
        }
        indexer.encode(&bits).expect("one key")
    };
    let (first, second, third) = (key_at(0), key_at(1), key_at(2));
    let per_column = tree_bytes(2);
    let mut cache = PackedMessageCache::<IndexKey, CachedScalar>::new(2, 3 * per_column);
    cache.max_bytes = 3 * per_column;
    let compute = |keys: &[IndexKey]| -> Result<Vec<Vec<CachedScalar>>> {
        Ok(keys
            .iter()
            .map(|_| vec![CachedScalar::F64(1.0); 2])
            .collect())
    };
    cache
        .get_or_compute_batch(&[first.clone(), second.clone(), third.clone()], compute)
        .expect("three columns cache");
    let charged = cache.key_heap_bytes;
    assert_eq!(charged, 3 * first.owned_heap_bytes());

    // Release exactly one column: the dead prefix keeps the released key's map
    // entry until a reclaim, so re-admitting it replaces a stale entry.
    cache.evict_oldest_bytes(per_column);
    assert_eq!(cache.key_count(), 2);
    cache
        .get_or_compute_batch(std::slice::from_ref(&first), compute)
        .expect("the released key is admitted again");
    assert_eq!(
        cache.key_heap_bytes, charged,
        "a re-admitted key must not be charged its heap bytes twice"
    );
}

fn tree_bytes(bond_dim: usize) -> usize {
    bond_dim * std::mem::size_of::<CachedScalar>()
}
