//! Tests for the caller-driven [`MultiIndexCache`].
//!
//! The cache is exercised the way its callers use it: look a point up,
//! evaluate the misses, insert the successful results.

use crate::cached_function::error::CacheKeyError;
use crate::cached_function::multi_index_cache::MultiIndexCache;
use crate::ColMajorArrayRef;
use std::cell::RefCell;
use std::rc::Rc;

/// A driver that owns a non-`Send` recorder, proving the cache imposes no
/// `Send + Sync + 'static` requirement on the evaluation callback.
fn driven_eval(
    cache: &mut MultiIndexCache<f64>,
    recorder: &Rc<RefCell<Vec<Vec<usize>>>>,
    idx: &[usize],
) -> f64 {
    if let Some(value) = cache.get(idx).unwrap() {
        return value;
    }
    recorder.borrow_mut().push(idx.to_vec());
    let value = (idx[0] * 4 + idx[1]) as f64;
    cache.insert(idx, value).unwrap();
    value
}

#[test]
fn a_borrowed_recorder_can_drive_the_cache() {
    let recorder = Rc::new(RefCell::new(Vec::new()));
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[3, 4]).unwrap();

    assert_eq!(driven_eval(&mut cache, &recorder, &[1, 2]), 6.0);
    assert_eq!(driven_eval(&mut cache, &recorder, &[1, 2]), 6.0);
    assert_eq!(driven_eval(&mut cache, &recorder, &[0, 0]), 0.0);

    assert_eq!(recorder.borrow().len(), 2, "only the misses are evaluated");
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 2);
    assert_eq!(cache.len(), 2);
}

#[test]
fn batch_lookup_separates_hits_from_misses() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    cache.insert(&[0, 0], 1.0).unwrap();

    // Point order is preserved; a mixed batch yields hit, miss, hit.
    let values = vec![cache.get(&[0, 0]).unwrap(), cache.get(&[1, 1]).unwrap()];
    assert_eq!(values, vec![Some(1.0), None]);
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 1);

    // An all-hit batch does not evaluate anything.
    assert_eq!(cache.get(&[0, 0]).unwrap(), Some(1.0));
    assert_eq!(cache.hits(), 2);
    assert_eq!(cache.misses(), 1);
}

#[test]
fn repeated_points_in_one_batch_are_deduplicated_by_is_cached() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();

    // A driver that collects unique misses from a batch containing duplicates.
    let batch = [[0usize, 1], [0, 1], [1, 0]];
    let mut misses: Vec<Vec<usize>> = Vec::new();
    for idx in batch.iter() {
        if !cache.is_cached(idx).unwrap() && !misses.iter().any(|m| m == idx) {
            misses.push(idx.to_vec());
        }
    }
    assert_eq!(misses.len(), 2, "the duplicate point is requested once");

    for idx in misses.iter() {
        cache.insert(idx, (idx[0] * 2 + idx[1]) as f64).unwrap();
    }
    for idx in batch.iter() {
        assert!(cache.is_cached(idx).unwrap());
    }
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.hits(), 0, "deduplication does not count as lookups");
}

#[test]
fn an_empty_request_changes_nothing() {
    let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.hits(), 0);
    assert_eq!(cache.misses(), 0);
    assert_eq!(cache.hit_ratio(), 0.0);
    assert_eq!(cache.retained_bytes(), 0);
}

#[test]
fn invalid_indices_are_rejected_before_encoding() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 3]).unwrap();

    // Wrong rank.
    assert!(matches!(
        cache.get(&[0]).unwrap_err(),
        CacheKeyError::InvalidIndexLength {
            expected: 2,
            got: 1
        }
    ));
    assert!(matches!(
        cache.insert(&[0, 0, 0], 1.0).unwrap_err(),
        CacheKeyError::InvalidIndexLength {
            expected: 2,
            got: 3
        }
    ));

    // Out of range on the second axis: `[1, 3]` would alias `[2, 0]` if the
    // coordinate were truncated instead of rejected.
    assert!(matches!(
        cache.is_cached(&[1, 3]).unwrap_err(),
        CacheKeyError::IndexOutOfBounds {
            axis: 1,
            index: 3,
            dim: 3
        }
    ));
    cache.insert(&[1, 1], 7.0).unwrap();
    assert!(cache.get(&[1, 3]).is_err());
    assert_eq!(cache.get(&[1, 1]).unwrap(), Some(7.0));

    // Rejected lookups are not counted and change nothing.
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 0);
    assert_eq!(cache.len(), 1);
}

#[test]
fn insertion_overwrites_and_clear_keeps_counters() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2]).unwrap();
    cache.insert(&[0], 1.0).unwrap();
    cache.insert(&[0], 2.0).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&[0]).unwrap(), Some(2.0));

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.get(&[0]).unwrap(), None);
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hit_ratio(), 0.5);
}

#[test]
fn values_that_are_not_send_are_rejected_at_compile_time_but_plain_values_work() {
    // `V` must be `Clone + Send + Sync + 'static` (the key backend stores it),
    // while the callback stays free: this test documents the value bound by
    // using a value type with heap payload and confirming its accounting (the
    // table holds the `Vec` headers; their heap payload is not counted).
    let mut cache: MultiIndexCache<Vec<f64>> = MultiIndexCache::new(&[2, 2]).unwrap();
    cache.insert(&[0, 1], vec![1.0, 2.0]).unwrap();
    assert_eq!(
        cache.retained_bytes(),
        crate::cached_function::memo_table::table_bytes::<u64, Vec<f64>>(4)
    );
    assert_eq!(cache.get(&[0, 1]).unwrap(), Some(vec![1.0, 2.0]));
}

#[test]
fn automatic_key_width_covers_every_transition_including_1024_bits() {
    let narrow: MultiIndexCache<f64> = MultiIndexCache::new(&[2; 64]).unwrap();
    assert_eq!(narrow.key_type(), "u64");
    let wide: MultiIndexCache<f64> = MultiIndexCache::new(&[2; 65]).unwrap();
    assert_eq!(wide.key_type(), "u128");
    let wider: MultiIndexCache<f64> = MultiIndexCache::new(&[2; 129]).unwrap();
    assert_eq!(wider.key_type(), "U256");
    let widest: MultiIndexCache<f64> = MultiIndexCache::new(&[2; 512]).unwrap();
    assert_eq!(widest.key_type(), "U512");
    let full: MultiIndexCache<f64> = MultiIndexCache::new(&[2; 1024]).unwrap();
    assert_eq!(full.key_type(), "U1024");

    // Exactly 1024 bits is representable; one more site is not.
    let mut full = full;
    let idx = vec![1usize; 1024];
    full.insert(&idx, 3.0).unwrap();
    assert_eq!(full.get(&idx).unwrap(), Some(3.0));
    assert_eq!(full.hits(), 1);

    assert!(matches!(
        MultiIndexCache::<f64>::new(&vec![2; 1025]).unwrap_err(),
        CacheKeyError::Overflow {
            total_bits: 1025,
            max_bits: 1024,
            ..
        }
    ));
}

#[test]
fn mixed_radix_keys_do_not_collide_between_index_shapes() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[3, 4]).unwrap();
    // [1, 0] and [0, 4] are distinct points; [0, 4] is invalid for dim 4.
    cache.insert(&[1, 0], 10.0).unwrap();
    cache.insert(&[0, 1], 1.0).unwrap();
    cache.insert(&[1, 1], 11.0).unwrap();
    assert_eq!(cache.get(&[1, 0]).unwrap(), Some(10.0));
    assert_eq!(cache.get(&[0, 1]).unwrap(), Some(1.0));
    assert_eq!(cache.get(&[1, 1]).unwrap(), Some(11.0));
    assert_eq!(cache.len(), 3);
}

#[test]
fn the_limit_evicts_the_oldest_entries_without_corrupting() {
    // Room for one table of 4 buckets: 3 entries of (u64 key, f64 value).
    let limit = crate::cached_function::memo_table::table_bytes::<u64, f64>(4);
    let mut cache: MultiIndexCache<f64> =
        MultiIndexCache::with_retained_byte_limit(&[8], limit).unwrap();
    for i in 0..3 {
        cache.insert(&[i], i as f64).unwrap();
    }
    assert_eq!((cache.len(), cache.retained_bytes()), (3, limit));
    cache.insert(&[3], 3.0).unwrap();
    assert_eq!((cache.evicted_entries(), cache.dropped_inserts()), (3, 0));
    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.get(&[3]).unwrap(),
        Some(3.0),
        "the new entry is cached"
    );
    assert_eq!(
        cache.get(&[0]).unwrap(),
        None,
        "an evicted entry is not served"
    );

    // Overwriting an existing entry never evicts.
    cache.insert(&[3], 30.0).unwrap();
    assert_eq!(cache.get(&[3]).unwrap(), Some(30.0));
    assert_eq!(cache.evicted_entries(), 3);

    // Raising the limit lets the table grow instead of evicting.
    cache.set_retained_byte_limit(1024);
    for i in 4..8 {
        cache.insert(&[i], i as f64).unwrap();
    }
    assert_eq!((cache.len(), cache.evicted_entries()), (5, 3));

    // `clear` frees the tables without resetting the counters.
    cache.clear();
    assert_eq!(cache.retained_bytes(), 0);
    assert_eq!(cache.evicted_entries(), 3);
    assert_eq!(cache.hits(), 2);
}

#[test]
fn rejected_insertions_leave_existing_entries_unchanged() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[3]).unwrap();
    cache.insert(&[1], 7.0).unwrap();

    assert!(cache.insert(&[9], 1.0).is_err());
    assert!(cache.insert(&[0, 0], 1.0).is_err());

    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&[1]).unwrap(), Some(7.0));
    assert_eq!(cache.misses(), 0, "the rejected insertions are not lookups");
}

/// FNV-1a over the bit patterns of `values`, the fingerprint the #1026
/// reproducer pins.
fn fingerprint(values: &[f64]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |acc, value| {
        (acc ^ value.to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Deterministic pseudo-random pool entry, so the fixture needs no RNG.
fn pool_entry(seed: usize, pool: usize) -> usize {
    let mut state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    state ^= state >> 33;
    (state >> 7) % pool
}

/// The fragments of `pool` binary multi-indices with 15 sites each.
fn binary_pool(pool: usize) -> Vec<Vec<usize>> {
    (0..pool)
        .map(|entry| (0..15).map(|site| (entry >> site) & 1).collect())
        .collect()
}

/// Run `batches` of `(left, right)` fragment requests through both entry points
/// and compare the values and the accounting after every batch.
///
/// The point route assembles the point of each entry and calls
/// `evaluate_batched`; the keyed route builds one key per side and calls
/// `evaluate_keyed`. Both start empty with the same retained-byte `limit` and
/// must stay indistinguishable.
fn assert_routes_agree(
    local_dims: &[usize],
    left_sites: &[usize],
    left: &[Vec<usize>],
    right_sites: &[usize],
    right: &[Vec<usize>],
    batches: &[Vec<(usize, usize)>],
    limit: usize,
) -> usize {
    let assemble = |(left_index, right_index): (usize, usize)| {
        let mut point = vec![0usize; local_dims.len()];
        for (&site, &value) in left_sites.iter().zip(&left[left_index]) {
            point[site] = value;
        }
        for (&site, &value) in right_sites.iter().zip(&right[right_index]) {
            point[site] = value;
        }
        point
    };

    let mut by_points: MultiIndexCache<f64> =
        MultiIndexCache::with_retained_byte_limit(local_dims, limit).unwrap();
    let mut by_keys: MultiIndexCache<f64> =
        MultiIndexCache::with_retained_byte_limit(local_dims, limit).unwrap();
    let left_side = by_keys.left_keys(left_sites, left).unwrap();
    let right_side = by_keys.right_keys(right_sites, right).unwrap();

    for (step, batch) in batches.iter().enumerate() {
        let points: Vec<Vec<usize>> = batch.iter().map(|&entry| assemble(entry)).collect();
        let flat: Vec<usize> = points.iter().flatten().copied().collect();
        let shape = [local_dims.len(), points.len()];
        let point_values = by_points
            .evaluate_batched(ColMajorArrayRef::new(&flat, &shape).unwrap(), |misses| {
                let n_sites = misses.shape()[0];
                Ok((0..misses.shape()[1])
                    .map(|column| target(&misses.data()[column * n_sites..(column + 1) * n_sites]))
                    .collect())
            })
            .unwrap();
        let key_values = by_keys
            .evaluate_keyed(&left_side, &right_side, batch, |misses| {
                Ok(misses
                    .iter()
                    .map(|&entry| target(&assemble(entry)))
                    .collect())
            })
            .unwrap();

        assert_eq!(
            fingerprint(&key_values),
            fingerprint(&point_values),
            "step {step}: values"
        );
        assert_eq!(by_keys.hits(), by_points.hits(), "step {step}: hits");
        assert_eq!(by_keys.misses(), by_points.misses(), "step {step}: misses");
        assert_eq!(by_keys.len(), by_points.len(), "step {step}: entries");
        assert_eq!(
            by_keys.dropped_inserts(),
            by_points.dropped_inserts(),
            "step {step}: dropped inserts"
        );
        assert_eq!(
            by_keys.retained_bytes(),
            by_points.retained_bytes(),
            "step {step}: retained bytes"
        );
        assert_eq!(
            by_keys.evicted_entries(),
            by_points.evicted_entries(),
            "step {step}: evicted"
        );
    }
    by_keys.evicted_entries()
}

#[test]
fn keyed_lookup_is_bit_for_bit_the_point_lookup() {
    // The shape of the #1026 reproducer, small enough for a unit test: 30 binary
    // sites split 15 + 15, pools of 300 fragments per side, matrices of 40 x 40
    // entries drawn from those pools in a deterministic sequence.
    let sites: Vec<usize> = (0..30).collect();
    let pool = 300;
    let left = binary_pool(pool);
    let right = binary_pool(pool);
    let entries_per_matrix = 40 * 40;
    let matrices: Vec<Vec<(usize, usize)>> = (0..20)
        .map(|matrix| {
            let base = 2 * matrix * entries_per_matrix;
            (0..entries_per_matrix)
                .map(|entry| {
                    (
                        pool_entry(base + entry, pool),
                        pool_entry(base + entry + 1, pool),
                    )
                })
                .collect()
        })
        .collect();

    // Cold matrices, the same matrices again (all hits), and a batch with
    // explicit repeats: the counters of both routes must agree at every step.
    let mut batches = matrices.clone();
    batches.extend(matrices.iter().rev().cloned());
    batches.push(vec![(0, 0), (0, 0), (7, 3), (0, 0), (7, 3)]);

    for limit in [crate::DEFAULT_RETAINED_BYTE_LIMIT, 4096] {
        // A 4096-byte budget cannot retain 40 x 40 entries, so both routes must
        // evict and re-evaluate on the warm pass identically.
        let evicted = assert_routes_agree(
            &vec![2; 30],
            &sites[..15],
            &left,
            &sites[15..],
            &right,
            &batches,
            limit,
        );
        assert_eq!(
            evicted > 0,
            limit == 4096,
            "limit {limit}: {evicted} evicted entries"
        );
    }
}

#[test]
fn keyed_lookup_is_bit_for_bit_the_point_lookup_on_u128_keys() {
    // 128 binary sites need 128 bits, so the cache selects a u128 key and the
    // composed key spans the whole width, top bit included: a key truncated to
    // 64 bits would make these points collide and return another entry's value.
    let local_dims = vec![2; 128];
    let sites: Vec<usize> = (0..128).collect();
    let cache: MultiIndexCache<f64> = MultiIndexCache::new(&local_dims).unwrap();
    assert_eq!(cache.key_type(), "u128");

    let ones = vec![1usize; 64];
    let zeros = vec![0usize; 64];
    let left = vec![ones.clone(), zeros.clone()];
    // Site 64 has coefficient 2^64 and site 127 has coefficient 2^127.
    let mut top = vec![0usize; 64];
    top[63] = 1;
    let mut low = vec![0usize; 64];
    low[0] = 1;
    let right = vec![zeros, low, top, vec![1usize; 64]];
    let batches = vec![
        vec![(0, 0), (0, 3), (1, 0), (1, 2), (0, 2)],
        vec![(0, 3), (1, 2), (0, 0), (1, 3)],
    ];
    assert_routes_agree(
        &local_dims,
        &sites[..64],
        &left,
        &sites[64..],
        &right,
        &batches,
        crate::DEFAULT_RETAINED_BYTE_LIMIT,
    );
}

fn target(point: &[usize]) -> f64 {
    point
        .iter()
        .enumerate()
        .map(|(site, &value)| (value + 1) as f64 * (site as f64 + 1.0))
        .sum()
}

#[test]
fn a_repeated_keyed_request_is_one_evaluation_and_one_miss_per_request() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    let left = cache.left_keys(&[0], &[vec![0usize], vec![1]]).unwrap();
    let right = cache.right_keys(&[1], &[vec![1usize]]).unwrap();
    let mut evaluated = 0;
    cache
        .evaluate_keyed(&left, &right, &[(0, 0), (0, 0), (0, 0)], |misses| {
            evaluated += misses.len();
            assert_eq!(misses, [(0, 0)]);
            Ok(vec![5.0])
        })
        .unwrap();
    assert_eq!(evaluated, 1);
    assert_eq!(cache.misses(), 3, "every request is a lookup");
    assert_eq!(cache.len(), 1);

    let values = cache
        .evaluate_keyed(&left, &right, &[(0, 0)], |_| Ok(Vec::new()))
        .unwrap();
    assert_eq!(values, vec![5.0]);
    assert_eq!(cache.hits(), 1);
}

#[test]
fn a_failed_batch_matches_the_point_route_and_inserts_nothing() {
    let batch = [(0usize, 0usize), (0, 1)];
    let mut by_keys: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    let left = by_keys.left_keys(&[0], &[vec![0usize]]).unwrap();
    let right = by_keys.right_keys(&[1], &[vec![0usize], vec![1]]).unwrap();
    let mut by_points: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    let points = [[0usize, 0], [0, 1]];
    let flat: Vec<usize> = points.iter().flatten().copied().collect();
    let shape = [2, points.len()];

    // A failing callback: neither route retains a value, and both keep the
    // lookups they counted before the failure.
    let error = by_keys
        .evaluate_keyed(&left, &right, &batch, |_| anyhow::bail!("unavailable"))
        .unwrap_err();
    assert_eq!(error.to_string(), "unavailable");
    let error = by_points
        .evaluate_batched(ColMajorArrayRef::new(&flat, &shape).unwrap(), |_| {
            anyhow::bail!("unavailable")
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "unavailable");
    for cache in [&by_keys, &by_points] {
        assert_eq!(cache.len(), 0, "a failed batch retains no entry");
        assert_eq!(cache.misses(), 2, "the lookups before the failure count");
        assert_eq!(cache.dropped_inserts(), 0);
    }

    // A callback that returns the wrong count is rejected the same way.
    let error = by_keys
        .evaluate_keyed(&left, &right, &batch, |_| Ok(vec![1.0]))
        .unwrap_err();
    assert!(matches!(
        error,
        crate::cached_function::multi_index_cache::CachedBatchError::Index(
            CacheKeyError::BatchResultLength {
                expected: 2,
                got: 1
            }
        )
    ));
    let error = by_points
        .evaluate_batched(ColMajorArrayRef::new(&flat, &shape).unwrap(), |_| {
            Ok(vec![1.0])
        })
        .unwrap_err();
    assert!(matches!(
        error,
        crate::cached_function::multi_index_cache::CachedBatchError::Index(
            CacheKeyError::BatchResultLength {
                expected: 2,
                got: 1
            }
        )
    ));
    assert_eq!(by_keys.len(), 0);
    assert_eq!(by_points.len(), 0);
    assert_eq!(by_keys.misses(), by_points.misses());
    assert_eq!(by_keys.hits(), by_points.hits());
}

#[test]
fn side_keys_of_another_cache_are_rejected() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    let other: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    let left = other.left_keys(&[0], &[vec![1usize]]).unwrap();
    let right = cache.right_keys(&[1], &[vec![1usize]]).unwrap();

    let error = cache
        .evaluate_keyed(&left, &right, &[(0, 0)], |_| Ok(vec![1.0]))
        .unwrap_err();
    assert!(matches!(
        error,
        crate::cached_function::multi_index_cache::CachedBatchError::Index(
            CacheKeyError::SideKeyOwner
        )
    ));
    assert_eq!(cache.misses(), 0, "a rejected request is not a lookup");
    assert_eq!(cache.len(), 0);
}

#[test]
fn side_keys_are_stale_after_clear() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    let left = cache.left_keys(&[0], &[vec![1usize]]).unwrap();
    let right = cache.right_keys(&[1], &[vec![1usize]]).unwrap();
    cache
        .evaluate_keyed(&left, &right, &[(0, 0)], |_| Ok(vec![1.0]))
        .unwrap();
    assert_eq!(cache.len(), 1);

    cache.clear();
    let error = cache
        .evaluate_keyed(&left, &right, &[(0, 0)], |_| Ok(vec![1.0]))
        .unwrap_err();
    assert!(matches!(
        error,
        crate::cached_function::multi_index_cache::CachedBatchError::Index(
            CacheKeyError::SideKeyStale
        )
    ));
    assert_eq!(cache.len(), 0);

    // Keys built after the clear work again.
    let left = cache.left_keys(&[0], &[vec![1usize]]).unwrap();
    let right = cache.right_keys(&[1], &[vec![1usize]]).unwrap();
    let values = cache
        .evaluate_keyed(&left, &right, &[(0, 0)], |_| Ok(vec![2.0]))
        .unwrap();
    assert_eq!(values, vec![2.0]);
}

#[test]
fn overlapping_sides_are_rejected() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    let left = cache.left_keys(&[0, 1], &[vec![1usize, 0]]).unwrap();
    let right = cache.right_keys(&[1], &[vec![1usize]]).unwrap();
    let error = cache
        .evaluate_keyed(&left, &right, &[(0, 0)], |_| Ok(vec![1.0]))
        .unwrap_err();
    assert!(matches!(
        error,
        crate::cached_function::multi_index_cache::CachedBatchError::Index(
            CacheKeyError::OverlappingSideSites { site: 1 }
        )
    ));
}

#[test]
fn requests_outside_their_side_are_rejected() {
    let mut cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    let left = cache.left_keys(&[0], &[vec![0usize]]).unwrap();
    let right = cache.right_keys(&[1], &[vec![0usize]]).unwrap();

    for (requests, side) in [
        (&[(1usize, 0usize)][..], "left"),
        (&[(0usize, 1usize)][..], "right"),
    ] {
        let error = cache
            .evaluate_keyed(&left, &right, requests, |_| Ok(vec![1.0]))
            .unwrap_err();
        match error {
            crate::cached_function::multi_index_cache::CachedBatchError::Index(
                CacheKeyError::SideKeyPosition {
                    side: named,
                    index,
                    len,
                },
            ) => {
                assert_eq!(named, side);
                assert_eq!((index, len), (1, 1));
            }
            other => panic!("unexpected error for {side}: {other}"),
        }
    }
    assert_eq!(cache.len(), 0);
}

#[test]
fn side_key_fragments_are_validated() {
    let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 3]).unwrap();

    assert!(matches!(
        cache.left_keys(&[2], &[vec![0usize]]).unwrap_err(),
        CacheKeyError::SideSiteOutOfBounds {
            site: 2,
            n_sites: 2
        }
    ));
    assert!(matches!(
        cache.left_keys(&[0, 0], &[vec![0usize, 0]]).unwrap_err(),
        CacheKeyError::RepeatedSideSite { site: 0 }
    ));
    assert!(matches!(
        cache.left_keys(&[0, 1], &[vec![0usize]]).unwrap_err(),
        CacheKeyError::InvalidIndexLength {
            expected: 2,
            got: 1
        }
    ));
    // Coordinate 3 is out of range for site 1, whose local dimension is 3.
    assert!(matches!(
        cache.right_keys(&[1], &[vec![3usize]]).unwrap_err(),
        CacheKeyError::IndexOutOfBounds {
            axis: 1,
            index: 3,
            dim: 3
        }
    ));
}

#[test]
fn a_zero_dimension_index_space_has_no_side_keys() {
    // A local dimension of 0 zeroes every later mixed-radix coefficient, so
    // distinct fragments would share one key; the space also holds no valid
    // point, so the cache refuses to key it at all.
    let cache: MultiIndexCache<f64> = MultiIndexCache::new(&[0, 2]).unwrap();
    assert!(matches!(
        cache.left_keys(&[1], &[vec![0usize]]).unwrap_err(),
        CacheKeyError::SideKeyDegenerate { site: 0 }
    ));
}

#[test]
fn wide_caches_have_no_side_keys() {
    // 129 binary sites select a 256-bit key, which has no side-key support.
    let wide: MultiIndexCache<f64> = MultiIndexCache::new(&vec![2; 129]).unwrap();
    assert!(!wide.supports_side_keys());
    assert!(matches!(
        wide.left_keys(&[0], &[vec![0usize]]).unwrap_err(),
        CacheKeyError::SideKeyUnsupported { key_type: "U256" }
    ));

    let narrow: MultiIndexCache<f64> = MultiIndexCache::new(&[2, 2]).unwrap();
    assert!(narrow.supports_side_keys());
}

#[test]
fn keyed_inserts_respect_the_payload_limit() {
    // Below the smallest table: nothing can be retained.
    let limit = crate::cached_function::memo_table::table_bytes::<u64, f64>(4) - 1;
    let mut cache: MultiIndexCache<f64> =
        MultiIndexCache::with_retained_byte_limit(&[2, 2], limit).unwrap();
    let left = cache.left_keys(&[0], &[vec![0usize], vec![1]]).unwrap();
    let right = cache.right_keys(&[1], &[vec![0usize]]).unwrap();
    cache
        .evaluate_keyed(&left, &right, &[(0, 0), (1, 0)], |misses| {
            Ok(misses
                .iter()
                .map(|&(left_index, _)| left_index as f64)
                .collect())
        })
        .unwrap();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.dropped_inserts(), 2, "both inserts are skipped");
    assert_eq!(cache.retained_bytes(), 0);

    // The skipped entry is evaluated again rather than served wrongly.
    let mut evaluated = 0;
    let values = cache
        .evaluate_keyed(&left, &right, &[(1, 0)], |misses| {
            evaluated += misses.len();
            Ok(vec![1.0])
        })
        .unwrap();
    assert_eq!(values, vec![1.0]);
    assert_eq!(evaluated, 1);
    assert_eq!(cache.hits(), 0);
    assert_eq!(cache.misses(), 3);
}
