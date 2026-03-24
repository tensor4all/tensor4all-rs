use super::*;

#[test]
fn test_cached_function_basic() {
    let local_dims = vec![2, 3, 4];
    let mut cf = CachedFunction::new(|idx: &[usize]| idx.iter().sum::<usize>(), &local_dims);

    assert_eq!(cf.eval(&[0, 1, 2]), 3);
    assert_eq!(cf.num_evals(), 1);
    assert_eq!(cf.num_cache_hits(), 0);

    // Second call should use cache
    assert_eq!(cf.eval(&[0, 1, 2]), 3);
    assert_eq!(cf.num_evals(), 1);
    assert_eq!(cf.num_cache_hits(), 1);

    // Different index
    assert_eq!(cf.eval(&[1, 2, 3]), 6);
    assert_eq!(cf.num_evals(), 2);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_auto_key_selection_small() {
    // Small space: should use u64
    let local_dims = vec![2; 30]; // 2^30 < 2^64
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims);
    assert_eq!(cf.key_type(), "u64");
}

#[test]
fn test_auto_key_selection_large() {
    // Large space: should use u128
    let local_dims = vec![2; 100]; // 2^100 > 2^64
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims);
    assert_eq!(cf.key_type(), "u128");
}

#[test]
fn test_cached_function_clear() {
    let local_dims = vec![10, 10];
    let mut cf = CachedFunction::new(|idx: &[usize]| idx[0] + idx[1], &local_dims);
    cf.eval(&[1, 2]);
    cf.eval(&[3, 4]);
    assert_eq!(cf.cache_size(), 2);

    cf.clear_cache();
    assert_eq!(cf.cache_size(), 0);
}

#[test]
fn test_local_dims() {
    let local_dims = vec![2, 3, 4];
    let cf = CachedFunction::new(|_: &[usize]| 0, &local_dims);
    assert_eq!(cf.local_dims(), &[2, 3, 4]);
    assert_eq!(cf.num_sites(), 3);
}

#[test]
fn test_u128_cache_operations() {
    // Use enough dimensions to force u128 key type
    let local_dims = vec![2; 100];
    let mut cf = CachedFunction::new(|idx: &[usize]| idx.iter().sum::<usize>(), &local_dims);
    assert_eq!(cf.key_type(), "u128");

    let idx = vec![0; 100];
    assert_eq!(cf.eval(&idx), 0);
    assert_eq!(cf.num_evals(), 1);
    assert!(!cf.is_cached(&vec![1; 100]));
    assert!(cf.is_cached(&idx));

    // Cache hit
    assert_eq!(cf.eval(&idx), 0);
    assert_eq!(cf.num_cache_hits(), 1);
    assert_eq!(cf.cache_size(), 1);

    // Clear
    cf.clear_cache();
    assert_eq!(cf.cache_size(), 0);
}

#[test]
fn test_eval_no_cache_and_stats() {
    let local_dims = vec![2, 3];
    let mut cf = CachedFunction::new(|idx: &[usize]| idx[0] * 10 + idx[1], &local_dims);

    // eval_no_cache does not populate cache or affect stats
    assert_eq!(cf.eval_no_cache(&[1, 2]), 12);
    assert_eq!(cf.num_evals(), 0);
    assert_eq!(cf.total_calls(), 0);
    assert_eq!(cf.cache_hit_ratio(), 0.0);

    // Now eval to populate cache
    cf.eval(&[1, 2]);
    cf.eval(&[1, 2]); // cache hit
    assert_eq!(cf.total_calls(), 2);
    assert_eq!(cf.cache_hit_ratio(), 0.5);
    assert!(cf.is_cached(&[1, 2]));
}
