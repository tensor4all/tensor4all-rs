use super::*;

#[test]
fn full_cache_batch_does_not_grow_persistent_capacity() {
    let mut cache = MultiIndexCache::<f64>::with_retained_byte_limit(&[128], 16).unwrap();
    cache.insert(&[0], 7.0).unwrap();
    let capacity = match &cache.backend {
        InnerCache::U64 { cache, .. } => super::super::read_lock(cache).capacity(),
        _ => unreachable!(),
    };
    let points: Vec<usize> = (1..128).collect();
    let out = cache
        .evaluate_batched(
            ColMajorArrayRef::new(&points, &[1, 127]).unwrap(),
            |misses| Ok(misses.data().iter().map(|&i| i as f64).collect()),
        )
        .unwrap();
    assert_eq!(out, points.iter().map(|&i| i as f64).collect::<Vec<_>>());
    assert_eq!((cache.len(), cache.dropped_inserts()), (1, 127));
    match &cache.backend {
        InnerCache::U64 { cache, .. } => {
            assert_eq!(super::super::read_lock(cache).capacity(), capacity);
        }
        _ => unreachable!(),
    }
}
