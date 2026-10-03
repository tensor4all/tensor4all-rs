use super::error::CacheKeyError;
use super::*;
use bnum::types::{U1024, U2048, U256};
use std::sync::Arc;
use std::thread;

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
fn test_index_int_all_types() {
    use super::index_int::IndexInt;

    assert_eq!(42u8.to_usize(), 42);
    assert_eq!(1000u16.to_usize(), 1000);
    assert_eq!(100_000u32.to_usize(), 100_000);
    assert_eq!(999usize.to_usize(), 999);
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

    assert_eq!(cf.eval(&[0, 1, 2]).unwrap(), 3);
    assert_eq!(cf.num_evals(), 1);
    assert_eq!(cf.num_cache_hits(), 0);

    // Second call should use cache
    assert_eq!(cf.eval(&[0, 1, 2]).unwrap(), 3);
    assert_eq!(cf.num_evals(), 1);
    assert_eq!(cf.num_cache_hits(), 1);

    // Different index
    assert_eq!(cf.eval(&[1, 2, 3]).unwrap(), 6);
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
fn test_auto_key_selection_at_exact_key_width_boundaries() {
    // The largest key of an index space is `product(local_dims) - 1`, so a
    // cardinality of exactly `2^K` still fits K-bit keys (#781).
    for (sites, expected) in [
        (vec![2usize; 64], "u64"),
        (vec![2usize; 128], "u128"),
        (vec![2usize; 256], "U256"),
        (vec![2usize; 512], "U512"),
        (vec![2usize; 1024], "U1024"),
    ] {
        let n = sites.len();
        let cf = CachedFunction::new(|_: &[usize]| 0.0, &sites)
            .unwrap_or_else(|e| panic!("{n} binary sites must be accepted: {e}"));
        assert_eq!(cf.key_type(), expected, "{n} binary sites");
    }

    // Power-of-two dimensions whose exponents still sum to 64 bits.
    for sites in [vec![4usize; 32], vec![65536usize; 4]] {
        let cf = CachedFunction::new(|_: &[usize]| 0.0, &sites).unwrap();
        assert_eq!(cf.key_type(), "u64");
    }
}

#[test]
fn test_exact_boundary_largest_key_round_trips() {
    let local_dims = vec![2usize; 64];
    // `f` counts the set coordinates, so the three probes below have pairwise
    // distinct keys; a key collision would show up as an unexpected cache hit.
    let cf = CachedFunction::new(
        |idx: &[usize]| idx.iter().sum::<usize>() as f64,
        &local_dims,
    )
    .unwrap();

    assert_eq!(cf.eval(&vec![0usize; 64]).unwrap(), 0.0);
    assert_eq!(cf.eval(&vec![1usize; 64]).unwrap(), 64.0);
    let mut single = vec![0usize; 64];
    single[63] = 1;
    assert_eq!(cf.eval(&single).unwrap(), 1.0);
    assert_eq!(cf.num_evals(), 3);
    assert_eq!(cf.num_cache_hits(), 0);

    // Re-evaluating the largest-key index must hit its own entry.
    assert_eq!(cf.eval(&vec![1usize; 64]).unwrap(), 64.0);
    assert_eq!(cf.num_evals(), 3);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_dimension_one_sites_do_not_shift_keys() {
    // A trailing or interior dimension-one site must not shift the
    // coefficients of the remaining sites (#781).
    let cf = CachedFunction::new(
        |idx: &[usize]| (idx[0] + 10 * idx[2]) as f64,
        &[2usize, 1, 3],
    )
    .unwrap();
    assert_eq!(cf.eval(&[1, 0, 2]).unwrap(), 21.0);
    assert_eq!(cf.eval(&[0, 0, 1]).unwrap(), 10.0);
    // These two collide if the dimension-one site shifts the following stride.
    assert_eq!(cf.eval(&[1, 0, 0]).unwrap(), 1.0);
    assert_eq!(cf.eval(&[0, 0, 2]).unwrap(), 20.0);
    assert_eq!(cf.num_evals(), 4);
    assert_eq!(cf.num_cache_hits(), 0);
    // Same index again must be a cache hit, so the two keys differ.
    assert_eq!(cf.eval(&[1, 0, 2]).unwrap(), 21.0);
    assert_eq!(cf.num_cache_hits(), 1);

    // A trailing dimension-one site leaves the cardinality at exactly 2^64.
    let mut sites = vec![2usize; 64];
    sites.push(1);
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &sites).unwrap();
    assert_eq!(cf.key_type(), "u64");
}

#[test]
fn test_degenerate_index_spaces_are_accepted() {
    // Empty and all-singleton spaces hold exactly one key.
    for local_dims in [vec![], vec![1usize], vec![1, 1], vec![2usize]] {
        let cf = CachedFunction::new(
            |idx: &[usize]| idx.iter().sum::<usize>() as f64,
            &local_dims,
        )
        .unwrap();
        let index = vec![0usize; local_dims.len()];
        assert_eq!(cf.eval(&index).unwrap(), 0.0, "{local_dims:?}");
        assert_eq!(cf.eval(&index).unwrap(), 0.0, "{local_dims:?}");
        assert_eq!(cf.num_evals(), 1, "{local_dims:?}");
        assert_eq!(cf.num_cache_hits(), 1, "{local_dims:?}");
    }

    // The exact 64-bit boundary also works through the forced key type.
    let local_dims = vec![2usize; 64];
    let mut index = vec![1usize; 64];
    let cf = CachedFunction::with_key_type::<u64>(
        |idx: &[usize]| idx.iter().sum::<usize>() as f64,
        &local_dims,
    )
    .unwrap();
    assert_eq!(cf.eval(&index).unwrap(), 64.0);
    index[0] = 0;
    assert_eq!(cf.eval(&index).unwrap(), 63.0);
    assert_eq!(cf.num_evals(), 2);
    assert_eq!(cf.num_cache_hits(), 0);
}

#[test]
fn test_empty_index_space_rejects_evaluation_without_calling_the_oracle() {
    for local_dims in [vec![0usize, 2], vec![2, 0, 1]] {
        let cf = CachedFunction::new(
            |_: &[usize]| -> f64 { panic!("an empty index space has no valid index") },
            &local_dims,
        )
        .unwrap();
        let index = vec![0usize; local_dims.len()];
        assert!(
            matches!(cf.eval(&index), Err(CacheKeyError::IndexOutOfBounds { .. })),
            "{local_dims:?}"
        );
    }
}

#[test]
fn test_forced_key_type_overflow_still_rejected() {
    // 4^33 needs 66 bits, which does not fit the forced u64 key type.
    let result = CachedFunction::with_key_type::<u64>(|_: &[usize]| 0.0, &[4usize; 33]);
    assert!(matches!(result, Err(CacheKeyError::Overflow { .. })));
}

#[test]
fn test_u1024_boundary_keys_are_exact() {
    // The widest supported boundary: 1024 binary sites, whose largest key is
    // `U1024::MAX`. Each probe isolates one coordinate so a wrong stride or a
    // colliding key shows up as a cache hit or a wrong value.
    let local_dims = vec![2usize; 1024];
    let probe = |coordinate: usize| {
        let mut index = vec![0usize; 1024];
        index[coordinate] = 1;
        index
    };
    let cf = CachedFunction::new(
        |idx: &[usize]| {
            idx.iter()
                .enumerate()
                .map(|(site, &value)| (site + 1) * value)
                .sum::<usize>() as f64
        },
        &local_dims,
    )
    .unwrap();
    assert_eq!(cf.key_type(), "U1024");

    assert_eq!(cf.eval(&vec![0usize; 1024]).unwrap(), 0.0);
    assert_eq!(cf.eval(&probe(0)).unwrap(), 1.0);
    assert_eq!(cf.eval(&probe(511)).unwrap(), 512.0);
    assert_eq!(cf.eval(&probe(512)).unwrap(), 513.0);
    assert_eq!(cf.eval(&probe(1023)).unwrap(), 1024.0);
    assert_eq!(cf.num_evals(), 5);
    assert_eq!(cf.num_cache_hits(), 0);

    // The all-ones index is the largest key and must equal `U1024::MAX`.
    assert_eq!(cf.eval(&vec![1usize; 1024]).unwrap(), 524_800.0);
    assert_eq!(cf.num_evals(), 6);
}

#[test]
fn test_u1024_all_ones_flat_key_is_max() {
    let local_dims = vec![2usize; 1024];
    let coeffs = compute_coeffs::<U1024>(&local_dims).unwrap();
    assert_eq!(coeffs.len(), 1024);
    assert_eq!(coeffs[1023], U1024::ONE << 1023);
    assert_eq!(
        flat_index::<U1024, usize>(&vec![1usize; 1024], &coeffs),
        U1024::MAX
    );
    assert_eq!(
        flat_index::<U1024, usize>(&vec![0usize; 1024], &coeffs),
        U1024::ZERO
    );
}

impl cache_key::CacheKey for U2048 {
    const BITS_COUNT: u32 = 2048;
    const ZERO: Self = U2048::ZERO;
    const ONE: Self = U2048::ONE;

    fn from_usize(v: usize) -> Self {
        U2048::from(v as u64)
    }

    fn checked_mul(self, rhs: Self) -> Option<Self> {
        self.checked_mul(rhs)
    }

    fn wrapping_add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

#[test]
fn test_custom_key_type_u2048() {
    let local_dims = vec![2; 1025];
    assert!(CachedFunction::new(|_: &[usize]| 0.0, &local_dims).is_err());

    let cf = CachedFunction::with_key_type::<U2048>(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "custom");

    let idx_zeros = vec![0usize; 1025];
    let idx_ones = vec![1usize; 1025];
    assert_eq!(cf.eval(&idx_zeros).unwrap(), 0.0);
    assert_eq!(cf.eval(&idx_ones).unwrap(), 0.0);
    assert_eq!(cf.num_evals(), 2);

    assert_eq!(cf.eval(&idx_zeros).unwrap(), 0.0);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_eval_batch_no_batch_func() {
    let local_dims = vec![2, 3];
    let cf = CachedFunction::new(|idx: &[usize]| idx[0] * 10 + idx[1], &local_dims).unwrap();

    cf.eval(&[0, 1]).unwrap();
    assert_eq!(cf.num_evals(), 1);

    let indices = vec![vec![0, 1], vec![1, 2], vec![0, 0]];
    let results = cf.eval_batch(&indices).unwrap();
    assert_eq!(results, vec![1, 12, 0]);
    assert_eq!(cf.num_evals(), 3);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_eval_batch_with_batch_func() {
    let local_dims = vec![2, 3];
    let single_f = |idx: &[usize]| idx[0] * 10 + idx[1];
    let batch_f = |indices: &[Vec<usize>]| -> Vec<usize> {
        indices.iter().map(|idx| idx[0] * 10 + idx[1]).collect()
    };
    let cf = CachedFunction::with_batch(single_f, batch_f, &local_dims).unwrap();

    cf.eval(&[1, 0]).unwrap();

    let indices = vec![vec![1, 0], vec![0, 2], vec![1, 1]];
    let results = cf.eval_batch(&indices).unwrap();
    assert_eq!(results, vec![10, 2, 11]);
    assert_eq!(cf.num_cache_hits(), 1);
    assert_eq!(cf.num_evals(), 3);
}

#[test]
fn test_eval_batch_empty() {
    let local_dims = vec![2, 3];
    let cf = CachedFunction::new(|idx: &[usize]| idx[0], &local_dims).unwrap();
    let results = cf.eval_batch(&[]).unwrap();
    assert!(results.is_empty());
}

#[test]
fn eval_rejects_wrong_rank_and_out_of_range_indices() {
    let cf = CachedFunction::new(|idx: &[usize]| idx.iter().sum::<usize>(), &[2, 3]).unwrap();

    assert!(matches!(
        cf.eval(&[1]),
        Err(error::CacheKeyError::InvalidIndexLength { .. })
    ));
    assert!(matches!(
        cf.eval(&[1, 3]),
        Err(error::CacheKeyError::IndexOutOfBounds { axis: 1, .. })
    ));
}

#[test]
fn eval_batch_rejects_short_callback_without_cache_mutation() {
    let cf = CachedFunction::with_batch(
        |idx: &[usize]| idx[0],
        |_indices: &[Vec<usize>]| vec![7],
        &[2],
    )
    .unwrap();

    let error = cf.eval_batch(&[vec![0], vec![1]]).unwrap_err();
    assert!(matches!(
        error,
        error::CacheKeyError::BatchResultLength {
            expected: 2,
            got: 1
        }
    ));
    assert_eq!(cf.cache_size(), 0);
}

#[test]
fn test_thread_safety() {
    let local_dims = vec![10, 10];
    let cf =
        Arc::new(CachedFunction::new(|idx: &[usize]| idx[0] * 100 + idx[1], &local_dims).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let cf = Arc::clone(&cf);
            thread::spawn(move || {
                for i in 0..10 {
                    let idx = vec![(t * 2 + i) % 10, (t + i) % 10];
                    let val = cf.eval(&idx).unwrap();
                    assert_eq!(val, idx[0] * 100 + idx[1]);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    assert!(cf.cache_size() > 0);
    assert_eq!(cf.num_evals() + cf.num_cache_hits(), 40);
}

#[test]
fn test_thread_safety_batch() {
    let local_dims = vec![5, 5];
    let cf = Arc::new(CachedFunction::new(|idx: &[usize]| idx[0] + idx[1], &local_dims).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let cf = Arc::clone(&cf);
            thread::spawn(move || {
                let indices: Vec<Vec<usize>> = (0..5).map(|i| vec![(t + i) % 5, i % 5]).collect();
                let results = cf.eval_batch(&indices).unwrap();
                for (i, idx) in indices.iter().enumerate() {
                    assert_eq!(results[i], idx[0] + idx[1]);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_index_int_u8_cached_function() {
    let local_dims = vec![2; 8];
    let cf = CachedFunction::new(
        |idx: &[u8]| idx.iter().map(|&i| i as usize).sum::<usize>(),
        &local_dims,
    )
    .unwrap();
    assert_eq!(cf.key_type(), "u64");

    assert_eq!(cf.eval(&[0u8, 1, 0, 1, 0, 1, 0, 1]).unwrap(), 4);
    assert_eq!(cf.eval(&[0u8, 1, 0, 1, 0, 1, 0, 1]).unwrap(), 4);
    assert_eq!(cf.num_evals(), 1);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_cached_function_clear() {
    let local_dims = vec![10, 10];
    let cf = CachedFunction::new(|idx: &[usize]| idx[0] + idx[1], &local_dims).unwrap();
    cf.eval(&[1, 2]).unwrap();
    cf.eval(&[3, 4]).unwrap();
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
    assert_eq!(cf.eval(&idx).unwrap(), 0);
    assert_eq!(cf.num_evals(), 1);
    assert!(!cf.is_cached(&vec![1; 100]));
    assert!(cf.is_cached(&idx));

    // Cache hit
    assert_eq!(cf.eval(&idx).unwrap(), 0);
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
    assert_eq!(cf.eval_no_cache(&[1, 2]).unwrap(), 12);
    assert_eq!(cf.num_evals(), 0);
    assert_eq!(cf.total_calls(), 0);
    assert_eq!(cf.cache_hit_ratio(), 0.0);

    // Now eval to populate cache
    cf.eval(&[1, 2]).unwrap();
    cf.eval(&[1, 2]).unwrap(); // cache hit
    assert_eq!(cf.total_calls(), 2);
    assert_eq!(cf.cache_hit_ratio(), 0.5);
    assert!(cf.is_cached(&[1, 2]));
}
