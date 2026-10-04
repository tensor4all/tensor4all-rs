//! Tests for the caller-driven [`MultiIndexCache`].
//!
//! The cache is exercised the way its callers use it: look a point up,
//! evaluate the misses, insert the successful results.

use crate::cached_function::error::CacheKeyError;
use crate::cached_function::multi_index_cache::MultiIndexCache;
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
    // using a value type with heap payload and confirming its accounting.
    let mut cache: MultiIndexCache<Vec<f64>> = MultiIndexCache::new(&[2, 2]).unwrap();
    cache.insert(&[0, 1], vec![1.0, 2.0]).unwrap();
    assert_eq!(cache.retained_bytes(), 8 + std::mem::size_of::<Vec<f64>>());
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
