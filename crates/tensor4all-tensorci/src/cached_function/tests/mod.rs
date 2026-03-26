use super::error::CacheKeyError;
use super::*;
use bnum::types::U256;

#[test]
fn test_cache_key_u64_basics() {
    use super::cache_key::CacheKey;

    assert_eq!(u64::BITS_COUNT, 64);
    assert_eq!(u64::ZERO, 0u64);
    assert_eq!(u64::ONE, 1u64);
    assert_eq!(u64::from_usize(42), 42u64);
    assert_eq!(u64::ONE.checked_mul(u64::from_usize(10)), Some(10u64));
    assert_eq!(u64::MAX.checked_mul(u64::from_usize(2)), None);
}

#[test]
fn test_cache_key_u128_basics() {
    use super::cache_key::CacheKey;

    assert_eq!(u128::BITS_COUNT, 128);
    assert_eq!(
        u128::from_usize(100).checked_mul(u128::from_usize(3)),
        Some(300u128)
    );
}

#[test]
fn test_cache_key_u256_basics() {
    use super::cache_key::CacheKey;

    assert_eq!(U256::BITS_COUNT, 256);
    let big = U256::from_usize(usize::MAX);
    assert!(big.checked_mul(U256::from_usize(2)).is_some());
}

#[test]
fn test_index_int_u8() {
    use super::index_int::IndexInt;

    let val: u8 = 42;
    let as_usize = val.to_usize();
    assert_eq!(as_usize, 42);
}

#[test]
fn test_compute_coeffs_u64() {
    let coeffs = super::compute_coeffs::<u64>(&[2, 3, 4]).unwrap();
    assert_eq!(coeffs, vec![1u64, 2, 6]);
}

#[test]
fn test_compute_coeffs_overflow() {
    let result = super::compute_coeffs::<u64>(&[2; 100]);
    assert!(result.is_err());
    match result.unwrap_err() {
        CacheKeyError::Overflow { .. } => {}
        other => panic!("expected overflow error, got {other:?}"),
    }
}

#[test]
fn test_total_bits_calculation() {
    assert_eq!(super::total_bits(&[2, 2, 2]), 3);
    assert_eq!(super::total_bits(&[4, 4]), 4);
    assert_eq!(super::total_bits(&[1, 2, 1]), 1);
    assert_eq!(super::total_bits(&[256]), 8);
}

#[test]
fn test_cached_function_basic() {
    let local_dims = vec![2, 3, 4];
    let cf = CachedFunction::new(|idx: &[usize]| idx.iter().sum::<usize>(), &local_dims).unwrap();

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
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "u64");
}

#[test]
fn test_auto_key_selection_large() {
    // Large space: should use u128
    let local_dims = vec![2; 100]; // 2^100 > 2^64
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "u128");
}

#[test]
fn test_auto_key_selection_u256() {
    let local_dims = vec![2; 200];
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "U256");
}

#[test]
fn test_auto_key_selection_u512() {
    let local_dims = vec![2; 300];
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "U512");
}

#[test]
fn test_auto_key_selection_u1024() {
    let local_dims = vec![2; 600];
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "U1024");
}

#[test]
fn test_overflow_error() {
    let local_dims = vec![2; 1025];
    let result = CachedFunction::new(|_: &[usize]| 0.0, &local_dims);
    assert!(result.is_err());
}

#[test]
fn test_cached_function_clear() {
    let local_dims = vec![10, 10];
    let cf = CachedFunction::new(|idx: &[usize]| idx[0] + idx[1], &local_dims).unwrap();
    cf.eval(&[1, 2]);
    cf.eval(&[3, 4]);
    assert_eq!(cf.cache_size(), 2);

    cf.clear_cache();
    assert_eq!(cf.cache_size(), 0);
}

#[test]
fn test_local_dims() {
    let local_dims = vec![2, 3, 4];
    let cf = CachedFunction::new(|_: &[usize]| 0, &local_dims).unwrap();
    assert_eq!(cf.local_dims(), &[2, 3, 4]);
    assert_eq!(cf.num_sites(), 3);
}

#[test]
fn test_u128_cache_operations() {
    // Use enough dimensions to force u128 key type
    let local_dims = vec![2; 100];
    let cf = CachedFunction::new(|idx: &[usize]| idx.iter().sum::<usize>(), &local_dims).unwrap();
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
    let cf = CachedFunction::new(|idx: &[usize]| idx[0] * 10 + idx[1], &local_dims).unwrap();

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
