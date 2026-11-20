use super::*;
use crate::tensortrain::SimpleTensorTrain;
use crate::types::tensor3_zeros;

#[test]
fn test_ttcache_evaluate() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let mut cache = TTCache::new(&tt);

    // Evaluate at various indices
    let val = cache.evaluate(&[0, 0, 0]).unwrap();
    assert!((val - 2.0).abs() < 1e-10);

    let val = cache.evaluate(&[1, 2, 1]).unwrap();
    assert!((val - 2.0).abs() < 1e-10);
}

#[test]
fn test_ttcache_caching() {
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 2).unwrap();
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 0, 1, 0.5);
    t0.set3(0, 1, 0, 2.0);
    t0.set3(0, 1, 1, 1.0);

    let mut t1: Tensor3<f64> = tensor3_zeros(2, 3, 1).unwrap();
    for l in 0..2 {
        for s in 0..3 {
            t1.set3(l, s, 0, (l + s + 1) as f64);
        }
    }

    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    let mut cache = TTCache::new(&tt);

    // First evaluation
    let val1 = cache.evaluate(&[0, 1]).unwrap();

    // Should be cached now
    assert!(!cache.cache_left[0].is_empty());

    // Second evaluation should use cache
    let val2 = cache.evaluate(&[0, 1]).unwrap();
    assert!((val1 - val2).abs() < 1e-10);
}

#[test]
fn test_ttcache_clear() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);

    // Populate cache
    let _ = cache.evaluate(&[0, 0]);
    assert!(!cache.cache_left[0].is_empty());

    // Clear cache
    cache.clear_cache();
    assert!(cache.cache_left[0].is_empty());
    assert!(cache.cache_right[0].is_empty());
}

#[test]
fn test_ttcache_evaluate_many() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let mut cache = TTCache::new(&tt);

    let indices = vec![vec![0, 0, 0], vec![0, 1, 0], vec![1, 2, 1], vec![0, 0, 1]];

    let results = cache.evaluate_many(&indices, None).unwrap();

    // All values should be 2.0 for a constant TT
    assert_eq!(results.len(), 4);
    for val in &results {
        assert!((val - 2.0).abs() < 1e-10);
    }
}

#[test]
fn test_ttcache_evaluate_many_matches_single() {
    // Create a non-trivial TT
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 2).unwrap();
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 0, 1, 0.5);
    t0.set3(0, 1, 0, 2.0);
    t0.set3(0, 1, 1, 1.0);

    let mut t1: Tensor3<f64> = tensor3_zeros(2, 3, 2).unwrap();
    for l in 0..2 {
        for s in 0..3 {
            for r in 0..2 {
                t1.set3(l, s, r, ((l + s + r) as f64) * 0.5 + 0.1);
            }
        }
    }

    let mut t2: Tensor3<f64> = tensor3_zeros(2, 2, 1).unwrap();
    for l in 0..2 {
        for s in 0..2 {
            t2.set3(l, s, 0, (l + s + 1) as f64);
        }
    }

    let tt = SimpleTensorTrain::new(vec![t0, t1, t2]).unwrap();
    let mut cache = TTCache::new(&tt);

    // Generate all indices
    let mut indices = Vec::new();
    for i0 in 0..2 {
        for i1 in 0..3 {
            for i2 in 0..2 {
                indices.push(vec![i0, i1, i2]);
            }
        }
    }

    // Evaluate using evaluate_many
    let batch_results = cache.evaluate_many(&indices, None).unwrap();

    // Compare with single evaluations
    for (idx, batch_val) in indices.iter().zip(batch_results.iter()) {
        let single_val = cache.evaluate(idx).unwrap();
        assert!(
            (batch_val - single_val).abs() < 1e-10,
            "Mismatch at {:?}: batch={}, single={}",
            idx,
            batch_val,
            single_val
        );
    }
}

#[test]
fn test_ttcache_evaluate_many_cache_efficiency() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2, 2, 2], 1.0);
    let mut cache = TTCache::new(&tt);

    // Indices with shared prefixes/suffixes
    let indices = vec![
        vec![0, 0, 0, 0],
        vec![0, 0, 0, 1],
        vec![0, 0, 1, 0],
        vec![0, 0, 1, 1],
        vec![1, 1, 0, 0],
        vec![1, 1, 0, 1],
    ];

    let results = cache.evaluate_many(&indices, None).unwrap();
    assert_eq!(results.len(), 6);

    // Check that caches are populated
    // The optimal split should create fewer unique entries than indices.len()
    let total_left_cached: usize = cache.cache_left.iter().map(|c| c.len()).sum();
    let total_right_cached: usize = cache.cache_right.iter().map(|c| c.len()).sum();

    // With shared prefixes/suffixes, we should have fewer cached entries
    // than if we computed each index independently
    assert!(total_left_cached + total_right_cached > 0);
}

#[test]
fn test_ttcache_evaluate_many_empty() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);

    let results = cache.evaluate_many(&[], None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_ttcache_evaluate_wrong_length() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);
    // Too few indices
    assert!(cache.evaluate(&[0]).is_err());
    // Too many indices
    assert!(cache.evaluate(&[0, 0, 0]).is_err());
}

#[test]
fn test_ttcache_partial_environment_wrong_length_errors() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);

    assert!(cache.evaluate_left(&[0, 0, 0]).is_err());
    assert!(cache.evaluate_right(&[0, 0, 0]).is_err());
}

#[test]
fn test_ttcache_evaluate_many_invalid_split() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);
    let indices = vec![vec![0, 0]];
    // split=0 is invalid
    assert!(cache.evaluate_many(&indices, Some(0)).is_err());
    // split > n is invalid
    assert!(cache.evaluate_many(&indices, Some(10)).is_err());
}

#[test]
fn test_ttcache_evaluate_many_wrong_index_length_errors() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);

    assert!(cache.evaluate_many(&[vec![0]], Some(1)).is_err());
    assert!(cache.evaluate_many(&[vec![0, 0, 0]], Some(1)).is_err());
}

#[test]
fn test_ttcache_evaluate_many_with_explicit_split() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let mut cache = TTCache::new(&tt);
    let indices = vec![vec![0, 0, 0], vec![1, 2, 1]];
    let results = cache.evaluate_many(&indices, Some(1)).unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!((*r - 2.0).abs() < 1e-10);
    }
}

#[test]
fn test_with_site_dims_mismatch_length() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    // Wrong number of site_dims entries
    assert!(TTCache::with_site_dims(&tt, vec![vec![2]]).is_err());
}

#[test]
fn test_with_site_dims_rejects_empty_site_dimensions() {
    let tt = SimpleTensorTrain::<f64>::constant(&[1], 1.0);
    let error = TTCache::with_site_dims(&tt, vec![Vec::new()]).unwrap_err();
    assert!(error.to_string().contains("at least one dimension"));
}

#[test]
fn test_with_site_dims_mismatch_product() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    // Product doesn't match site dim
    assert!(TTCache::with_site_dims(&tt, vec![vec![3], vec![3]]).is_err());
}

#[test]
fn test_with_site_dims_rejects_product_overflow() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2], 1.0);
    let error = TTCache::with_site_dims(&tt, vec![vec![usize::MAX, 2]]).unwrap_err();
    assert!(error.to_string().contains("overflows usize"));
}

#[test]
fn find_split_heuristic() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2, 2, 2], 1.0);
    let cache = TTCache::new(&tt);

    // Indices where split=1 is optimal:
    // Heuristic checks 1/4=1, 1/2=2, 3/4=3
    // split=1: unique_left=2, unique_right=2, total=4
    // split=2: unique_left=4, unique_right=1, total=5
    // split=3: unique_left=4, unique_right=1, total=5
    let indices = vec![
        vec![0, 0, 0, 0],
        vec![0, 1, 0, 0],
        vec![1, 0, 0, 0],
        vec![1, 1, 0, 0],
    ];

    let split = cache.find_split_heuristic(&indices);
    assert_eq!(split, 1);

    // Indices where split=3 is optimal:
    // split=1: unique_left=1, unique_right=4, total=5
    // split=2: unique_left=1, unique_right=4, total=5
    // split=3: unique_left=2, unique_right=2, total=4
    let indices2 = vec![
        vec![0, 0, 0, 0],
        vec![0, 0, 0, 1],
        vec![0, 0, 1, 0],
        vec![0, 0, 1, 1],
    ];

    let split2 = cache.find_split_heuristic(&indices2);
    assert_eq!(split2, 3);
}

/// The key width must count `ceil(log2(dim))` bits per site, so a binary site
/// costs one bit. 700 and 1024 binary sites were rejected before that fix
/// (700 sites need 1400 bits with the old over-estimate, and the first split
/// candidate of a 700-site space left a 525-site half at 1050 bits).
#[test]
fn test_key_width_boundary_for_binary_sites() {
    for sites in [700usize, 1024] {
        let tt = SimpleTensorTrain::<f64>::constant(&vec![2usize; sites], 3.0);
        let mut cache = TTCache::new(&tt);
        assert_eq!(
            cache.evaluate_many(&[vec![0usize; sites]], None).unwrap(),
            vec![3.0],
            "{sites} binary sites"
        );
        let mut edges = vec![0usize; sites];
        edges[0] = 1;
        edges[sites - 1] = 1;
        assert_eq!(
            cache.evaluate_many(&[edges], None).unwrap(),
            vec![3.0],
            "{sites} binary sites"
        );
    }
}

#[test]
fn batch_cache_supports_keys_wider_than_1024_bits() {
    let n = 1370;
    let mut tensors = Vec::with_capacity(n);
    for site in 0..n {
        let mut tensor = tensor3_zeros(1, 2, 1).unwrap();
        tensor.set3(0, 0, 0, 1.0);
        tensor.set3(
            0,
            1,
            0,
            if site == 0 {
                2.0
            } else if site == 1024 {
                5.0
            } else if site == n - 1 {
                3.0
            } else {
                1.0
            },
        );
        tensors.push(tensor);
    }
    let tt = SimpleTensorTrain::new(tensors).unwrap();
    let mut indices = vec![vec![0; n]; 5];
    indices[1][0] = 1;
    indices[2][1024] = 1;
    indices[3][n - 1] = 1;
    indices[4][0] = 1;
    indices[4][1024] = 1;
    indices[4][n - 1] = 1;
    let mut cache = TTCache::new(&tt);
    for split in [None, Some(1050), Some(n - 1050), Some(n)] {
        assert_eq!(
            cache.evaluate_many(&indices, split).unwrap(),
            vec![1.0, 2.0, 5.0, 3.0, 30.0]
        );
    }
}

#[test]
fn integer_cache_keys_preserve_mixed_dimensions_and_partial_lengths() {
    let tensors = [3, 1, 5, 2]
        .into_iter()
        .map(|dim| {
            let mut tensor = tensor3_zeros(1, dim, 1).unwrap();
            for index in 0..dim {
                tensor.set3(0, index, 0, (index + 1) as f64);
            }
            tensor
        })
        .collect();
    let tt = SimpleTensorTrain::new(tensors).unwrap();
    let mut cache =
        TTCache::with_site_dims(&tt, vec![vec![3], vec![1], vec![5], vec![1, 2]]).unwrap();
    for _ in 0..2 {
        assert_eq!(cache.evaluate_left(&[]).unwrap(), vec![1.0]);
        assert_eq!(cache.evaluate_left(&[2]).unwrap(), vec![3.0]);
        assert_eq!(cache.evaluate_left(&[2, 0]).unwrap(), vec![3.0]);
        assert_eq!(cache.evaluate_left(&[2, 0, 4]).unwrap(), vec![15.0]);
        assert_eq!(cache.evaluate_left(&[2, 0, 4, 1]).unwrap(), vec![30.0]);
        assert_eq!(cache.evaluate_right(&[]).unwrap(), vec![1.0]);
        assert_eq!(cache.evaluate_right(&[1]).unwrap(), vec![2.0]);
        assert_eq!(cache.evaluate_right(&[4, 1]).unwrap(), vec![10.0]);
        assert_eq!(cache.evaluate_right(&[0, 4, 1]).unwrap(), vec![10.0]);
        assert_eq!(cache.evaluate_right(&[2, 0, 4, 1]).unwrap(), vec![30.0]);
        for split in [None, Some(1), Some(2), Some(3), Some(4)] {
            assert_eq!(
                cache
                    .evaluate_many(
                        &[vec![0, 0, 0, 0], vec![2, 0, 4, 1], vec![1, 0, 2, 0]],
                        split
                    )
                    .unwrap(),
                vec![1.0, 30.0, 6.0]
            );
        }
        for (left, index, site, max) in [(true, vec![0, 1], 1, 1), (false, vec![5, 0], 2, 5)] {
            let error = if left {
                cache.evaluate_left(&index)
            } else {
                cache.evaluate_right(&index)
            }
            .unwrap_err();
            assert!(
                matches!(error, SimpleTensorTrainError::IndexOutOfBounds { site: s, index: i, max: m } if s == site && i == index[if left { 1 } else { 0 }] && m == max)
            );
        }
        cache.clear_cache();
        assert!(cache.cache_left.iter().all(|c| c.is_empty()));
        assert!(cache.cache_right.iter().all(|c| c.is_empty()));
    }
}

#[test]
fn cache_allows_partial_evaluation_around_zero_dimension_site() {
    let tt = SimpleTensorTrain::<f64>::zeros(&[2, 0, 3]);
    let mut cache = TTCache::new(&tt);
    assert_eq!(cache.evaluate_left(&[1]).unwrap(), vec![0.0]);
    assert_eq!(cache.evaluate_right(&[2]).unwrap(), vec![0.0]);
    assert!(matches!(
        cache.evaluate_left(&[1, 0]),
        Err(SimpleTensorTrainError::IndexOutOfBounds {
            site: 1,
            max: 0,
            ..
        })
    ));
    assert!(matches!(
        cache.evaluate_right(&[0, 2]),
        Err(SimpleTensorTrainError::IndexOutOfBounds {
            site: 1,
            max: 0,
            ..
        })
    ));
}

#[test]
fn full_left_batch_preserves_order_duplicates_and_arithmetic() {
    fn check<T: TTScalar<Real = f64> + PartialEq + std::fmt::Debug>(value: impl Fn(f64) -> T) {
        let tensors = (0..5)
            .map(|site| {
                let l = if site == 0 { 1 } else { 3 };
                let r = if site == 4 { 1 } else { 3 };
                Tensor3::from_fn([l, 2, r], |[l, s, r]| {
                    value(((1 + site + 3 * l + 5 * s + 7 * r) as f64).sin())
                })
                .unwrap()
            })
            .collect();
        let tt = SimpleTensorTrain::new(tensors).unwrap();
        let indices = vec![
            vec![1, 1, 0, 1, 0],
            vec![0, 1, 0, 0, 0],
            vec![1, 1, 0, 1, 0],
            vec![0, 0, 0, 0, 0],
            vec![1, 1, 0, 1, 1],
        ];
        let expected: Vec<_> = indices
            .iter()
            .map(|i| T::zero() + tt.evaluate(i).unwrap() * T::one())
            .collect();
        let mut cache = TTCache::new(&tt);
        for _ in 0..2 {
            assert_eq!(
                cache.evaluate_many(&indices, Some(tt.len())).unwrap(),
                expected
            );
            assert!(cache.cache_left.iter().all(|c| c.is_empty()));
            assert!(cache.cache_right.iter().all(|c| c.is_empty()));
            cache.evaluate_left(&indices[0][..2]).unwrap();
            assert_eq!(
                cache.evaluate_many(&indices, Some(tt.len())).unwrap(),
                expected
            );
            cache.clear_cache();
        }
    }
    check(|x| x);
    check(|x| num_complex::Complex64::new(x, x * 0.3));
}

#[test]
fn full_left_batch_checks_single_site_and_malformed_bonds() {
    let tt = SimpleTensorTrain::constant(&[3], -2.0_f64);
    assert_eq!(
        TTCache::new(&tt)
            .evaluate_many(&[vec![2], vec![0], vec![2]], Some(1))
            .unwrap(),
        vec![-2.0; 3]
    );
    assert!(TTCache::new(&tt)
        .evaluate_many(&[vec![3]], Some(1))
        .is_err());
    for tensors in [
        vec![
            tensor3_zeros(1, 2, 2).unwrap(),
            tensor3_zeros(3, 2, 1).unwrap(),
        ],
        vec![tensor3_zeros(1, 2, 2).unwrap()],
        vec![tensor3_zeros(2, 2, 1).unwrap()],
    ] {
        let mut tt = SimpleTensorTrain::<f64>::constant(&vec![2; tensors.len()], 1.0);
        tt.site_tensors_mut().clone_from_slice(&tensors);
        let error = TTCache::new(&tt)
            .evaluate_many(&[vec![0; tensors.len()]], Some(tensors.len()))
            .unwrap_err();
        assert!(matches!(
            error,
            SimpleTensorTrainError::InvalidOperation { .. }
        ));
    }
}
