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
