use super::*;

#[test]
fn full_cache_batch_stays_within_its_table_limit() {
    // One table of 4 buckets of (u64, f64) entries: 3 entries.
    let limit = super::super::memo_table::table_bytes::<u64, f64>(4);
    let mut cache = MultiIndexCache::<f64>::with_retained_byte_limit(&[128], limit).unwrap();
    cache.insert(&[0], 7.0).unwrap();
    let points: Vec<usize> = (1..128).collect();
    let out = cache
        .evaluate_batched(
            ColMajorArrayRef::new(&points, &[1, 127]).unwrap(),
            |misses| Ok(misses.data().iter().map(|&i| i as f64).collect()),
        )
        .unwrap();
    assert_eq!(out, points.iter().map(|&i| i as f64).collect::<Vec<_>>());
    assert!(cache.retained_bytes() <= limit);
    assert_eq!(cache.dropped_inserts(), 0);
    assert_eq!(cache.evicted_entries() + cache.len(), 128);
    // The latest point of the batch is still answered.
    assert_eq!(cache.get(&[127]).unwrap(), Some(127.0));
}
