use super::*;

#[test]
fn test_floating_zone_finds_error() {
    use tensor4all_simplett::Tensor3Ops;

    // Build a rank-1 TT approximation of constant 1.0
    let mut t0 = tensor4all_simplett::tensor3_zeros(1, 4, 1).unwrap();
    let mut t1 = tensor4all_simplett::tensor3_zeros(1, 4, 1).unwrap();
    for s in 0..4 {
        t0.set3(0, s, 0, 1.0);
        t1.set3(0, s, 0, 1.0);
    }
    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();

    // Verify TT evaluates to 1.0 everywhere
    assert!((tt.evaluate(&[0, 0]).unwrap() - 1.0).abs() < 1e-14);
    assert!((tt.evaluate(&[3, 3]).unwrap() - 1.0).abs() < 1e-14);

    // Exact function has non-constant behavior
    let f = |idx: &MultiIndex| (idx[0] * idx[1]) as f64;
    let local_dims = vec![4, 4];

    // Start from (1, 1) so initial error is not zero: |1*1 - 1| = 0
    // Actually start from (2, 2) so initial error is |4 - 1| = 3
    let (pivot, error) = floating_zone(&tt, &f, &local_dims, &vec![2, 2], f64::MAX).unwrap();

    // Error should be > 0 since tt=1 but f(i,j)=i*j varies
    // The maximum error should be at (3,3): |9-1|=8
    assert!(error > 0.0, "Error should be positive, got {}", error);
    assert_eq!(pivot, vec![3, 3], "Should find max error at (3,3)");
    assert!(
        (error - 8.0).abs() < 1e-10,
        "Error should be 8.0, got {}",
        error
    );
}

#[test]
fn test_estimate_true_error_sorted() {
    // Build a constant TT (all 0)
    let t0 = tensor4all_simplett::tensor3_zeros(1, 4, 1).unwrap();
    let t1 = tensor4all_simplett::tensor3_zeros(1, 4, 1).unwrap();
    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();

    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(7);

    let errors = estimate_true_error(&tt, &f, 10, None, &mut rng).unwrap();

    // Verify sorted in descending order
    for i in 0..errors.len().saturating_sub(1) {
        assert!(
            errors[i].1 >= errors[i + 1].1,
            "Errors should be sorted descending: {} < {}",
            errors[i].1,
            errors[i + 1].1
        );
    }
}

#[test]
fn test_dedup_pivots_ignores_interleaved_equal_errors() {
    // A stable sort by descending error leaves equal-error entries in input
    // order, so two copies of `p` can be separated by a distinct pivot `q`.
    // `Vec::dedup_by` would keep both copies; the set-based dedup must not.
    let p = vec![1usize, 2];
    let q = vec![3usize, 4];
    let mut entries = vec![(p.clone(), 1.0), (q.clone(), 1.0), (p.clone(), 1.0)];

    dedup_pivots(&mut entries);

    assert_eq!(entries.len(), 2, "duplicate pivot survived: {entries:?}");
    assert_eq!(entries[0].0, p, "the first (highest-error) copy must win");
    assert_eq!(entries[1].0, q);
}
