use super::CacheKeyError;
use crate::{CachedBatchError, ColMajorArrayRef, MultiIndexCache};

fn bitwise_batch<V: Clone + Send + Sync + 'static>(values: Vec<V>, bits: impl Fn(&V) -> Vec<u64>) {
    let mut cache = MultiIndexCache::<V>::new(&[2, 2]).unwrap();
    cache.insert(&[0, 0], values[0].clone()).unwrap();
    let shape = [2, 4];
    let points = [0, 0, 1, 0, 0, 1, 1, 0];
    let mut calls = 0;
    let out = cache
        .evaluate_batched(ColMajorArrayRef::new(&points, &shape).unwrap(), |misses| {
            calls += 1;
            assert_eq!(misses.shape(), &[2, 2]);
            assert_eq!(misses.data(), &[1, 0, 0, 1]);
            Ok(vec![values[1].clone(), values[2].clone()])
        })
        .unwrap();
    let expected = [0, 1, 2, 1].map(|i| bits(&values[i]));
    assert_eq!(out.iter().map(&bits).collect::<Vec<_>>(), expected);
    let warm = cache
        .evaluate_batched(ColMajorArrayRef::new(&points, &shape).unwrap(), |_| {
            calls += 1;
            anyhow::bail!("all-hit batches must not call the target")
        })
        .unwrap();
    assert_eq!(warm.iter().map(bits).collect::<Vec<_>>(), expected);
    assert_eq!(calls, 1);
    assert_eq!((cache.hits(), cache.misses(), cache.len()), (5, 3, 3));
}

#[test]
fn cached_batch_preserves_bits_for_all_scalar_kinds() {
    use num_complex::{Complex32, Complex64};
    bitwise_batch(
        vec![
            -0.0f64,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::INFINITY,
        ],
        |v| vec![v.to_bits()],
    );
    bitwise_batch(
        vec![-0.0f32, f32::from_bits(0x7fc0_0001), f32::INFINITY],
        |v| vec![v.to_bits() as u64],
    );
    bitwise_batch(
        vec![
            Complex64::new(-0.0, 3.0),
            Complex64::new(f64::NAN, -0.0),
            Complex64::new(2.0, f64::INFINITY),
        ],
        |v| vec![v.re.to_bits(), v.im.to_bits()],
    );
    bitwise_batch(
        vec![
            Complex32::new(-0.0, 3.0),
            Complex32::new(f32::NAN, -0.0),
            Complex32::new(2.0, f32::INFINITY),
        ],
        |v| vec![v.re.to_bits() as u64, v.im.to_bits() as u64],
    );
}

#[test]
fn cached_batch_failure_and_wrong_lengths_do_not_insert_partial_results() {
    for wrong_length in [false, true] {
        let mut cache = MultiIndexCache::<f64>::new(&[3]).unwrap();
        cache.insert(&[0], 10.0).unwrap();
        let points = ColMajorArrayRef::new(&[0, 1, 2, 1], &[1, 4]).unwrap();
        let error = cache
            .evaluate_batched(points, |_| {
                if wrong_length {
                    Ok(vec![20.0])
                } else {
                    anyhow::bail!("retryable")
                }
            })
            .unwrap_err();
        if wrong_length {
            assert!(matches!(
                error,
                CachedBatchError::Index(CacheKeyError::BatchResultLength {
                    expected: 2,
                    got: 1
                })
            ));
        } else {
            assert_eq!(error.to_string(), "retryable");
            assert!(std::error::Error::source(&error).is_some());
        }
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_cached(&[1]).unwrap());
        assert!(!cache.is_cached(&[2]).unwrap());
        let retry = cache
            .evaluate_batched(points, |_| Ok(vec![20.0, 30.0]))
            .unwrap();
        assert_eq!(retry, vec![10.0, 20.0, 30.0, 20.0]);
    }
}

#[test]
fn cached_batch_rejects_shape_and_late_invalid_points_before_callback() {
    let mut cache = MultiIndexCache::<f64>::new(&[2]).unwrap();
    for (data, shape) in [
        (vec![0], vec![1]),
        (vec![0, 0], vec![2, 1]),
        (vec![0, 2], vec![1, 2]),
    ] {
        let error = cache
            .evaluate_batched(ColMajorArrayRef::new(&data, &shape).unwrap(), |_| {
                panic!("invalid batch reached the callback")
            })
            .unwrap_err();
        assert!(matches!(error, CachedBatchError::Index(_)));
        assert!(cache.is_empty());
    }
}

#[test]
fn cached_batch_limits_skip_only_distinct_inserts_and_preserve_values() {
    for (limit, retained, skipped) in [(0, 0, 2), (16, 1, 1), (32, 2, 0)] {
        let mut cache = MultiIndexCache::<f64>::with_retained_byte_limit(&[2], limit).unwrap();
        let values = cache
            .evaluate_batched(
                ColMajorArrayRef::new(&[1, 0, 1], &[1, 3]).unwrap(),
                |misses| {
                    assert_eq!(misses.data(), &[1, 0]);
                    Ok(vec![20.0, 10.0])
                },
            )
            .unwrap();
        assert_eq!(values, vec![20.0, 10.0, 20.0]);
        assert_eq!((cache.len(), cache.dropped_inserts()), (retained, skipped));
        assert!(cache.retained_bytes() <= limit);
    }
}

#[test]
fn cached_batch_empty_scalar_and_extended_integer_keys() {
    let mut cache = MultiIndexCache::<f64, u8>::new(&[2]).unwrap();
    assert_eq!(
        cache
            .evaluate_batched(ColMajorArrayRef::new(&[], &[1, 0]).unwrap(), |_| panic!(
                "empty callback"
            ))
            .unwrap(),
        Vec::<f64>::new()
    );
    let mut scalar = MultiIndexCache::<f64>::new(&[]).unwrap();
    assert_eq!(
        scalar
            .evaluate_batched(ColMajorArrayRef::new(&[], &[0, 3]).unwrap(), |misses| {
                assert_eq!(misses.shape(), &[0, 1]);
                Ok(vec![7.0])
            })
            .unwrap(),
        vec![7.0; 3]
    );
    assert_eq!(scalar.len(), 1);
    for n_sites in [65, 129, 257, 513] {
        let mut wide = MultiIndexCache::<f64>::new(&vec![2; n_sites]).unwrap();
        let mut data = vec![0; n_sites * 3];
        data[2 * n_sites - 1] = 1;
        let out = wide
            .evaluate_batched(
                ColMajorArrayRef::new(&data, &[n_sites, 3]).unwrap(),
                |misses| {
                    assert_eq!(misses.shape(), &[n_sites, 2]);
                    Ok(vec![2.0, 3.0])
                },
            )
            .unwrap();
        assert_eq!(out, vec![2.0, 3.0, 2.0]);
        assert_eq!(wide.len(), 2);
        assert_eq!(wide.get(&data[n_sites..2 * n_sites]).unwrap(), Some(3.0));
        assert_eq!(wide.get(&data[..n_sites]).unwrap(), Some(2.0));
    }
}

#[test]
fn cached_batch_encodes_each_request_only_once() {
    use crate::IndexInt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CONVERSIONS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Clone, Copy)]
    struct Counted(usize);
    impl IndexInt for Counted {
        fn to_usize(self) -> usize {
            CONVERSIONS.fetch_add(1, Ordering::Relaxed);
            self.0
        }
    }
    let mut cache = MultiIndexCache::<f64, Counted>::new(&[2, 2]).unwrap();
    let data = [0, 1, 1, 0, 0, 1].map(Counted);
    let values = cache
        .evaluate_batched(ColMajorArrayRef::new(&data, &[2, 3]).unwrap(), |_| {
            Ok(vec![10.0, 20.0])
        })
        .unwrap();
    assert_eq!(values, vec![10.0, 20.0, 10.0]);
    // Each coordinate is converted once for validation and once for encoding,
    // not again for pending lookup or retained insertion.
    assert_eq!(CONVERSIONS.load(Ordering::Relaxed), 2 * data.len());
}

#[test]
fn cached_batch_wide_keys_preserve_budget_order_and_failure_accounting() {
    for (n_sites, key_bytes) in [(1, 8), (65, 16), (129, 32), (257, 64), (513, 128)] {
        let dims = vec![2; n_sites];
        let mut cache =
            MultiIndexCache::<f64>::with_retained_byte_limit(&dims, key_bytes + 8).unwrap();
        let mut data = vec![0; 3 * n_sites];
        data[n_sites - 1] = 1;
        data[3 * n_sites - 1] = 1;
        let shape = [n_sites, 3];
        let batch = ColMajorArrayRef::new(&data, &shape).unwrap();
        let out = cache
            .evaluate_batched(batch, |misses| {
                assert_eq!(misses.shape(), &[n_sites, 2]);
                assert_eq!(misses.data(), &data[..2 * n_sites]);
                Ok(vec![30.0, 20.0])
            })
            .unwrap();
        assert_eq!(out, vec![30.0, 20.0, 30.0]);
        assert_eq!(
            (cache.hits(), cache.misses(), cache.dropped_inserts()),
            (0, 3, 1)
        );
        assert_eq!(cache.retained_bytes(), key_bytes + 8);
        assert_eq!(cache.get(&data[..n_sites]).unwrap(), Some(30.0));
        assert_eq!(cache.get(&data[n_sites..2 * n_sites]).unwrap(), None);
        // A hit and a miss precede an invalid coordinate. Neither the missing
        // value nor the invalid point may be retained or sent to the callback.
        data[3 * n_sites - 1] = 2;
        assert!(matches!(
            cache.evaluate_batched(
                ColMajorArrayRef::new(&data, &[n_sites, 3]).unwrap(),
                |_| panic!("invalid input reached callback")
            ),
            Err(CachedBatchError::Index(
                CacheKeyError::IndexOutOfBounds { .. }
            ))
        ));
        assert_eq!(
            (cache.hits(), cache.misses(), cache.dropped_inserts()),
            (2, 5, 1)
        );
        assert_eq!(cache.len(), 1);
    }
}
