use super::*;

#[test]
fn test_default_global_pivot_finder() {
    // Simple function: f(i, j) = i * j
    let f = |idx: &MultiIndex| (idx[0] * idx[1]) as f64;
    let local_dims = vec![4, 4];

    // Build a deliberately bad TT (constant 0) to ensure large errors
    let tensors = vec![
        tensor4all_simplett::tensor3_zeros(1, 4, 1).unwrap(),
        tensor4all_simplett::tensor3_zeros(1, 4, 1).unwrap(),
    ];
    let tt = SimpleTensorTrain::new(tensors).unwrap();

    let input = GlobalPivotSearchInput {
        local_dims: local_dims.clone(),
        current_tt: tt,
        max_sample_value: 9.0,
        i_set: vec![vec![vec![]], vec![vec![0]]],
        j_set: vec![vec![vec![0]], vec![vec![]]],
    };

    let finder = DefaultGlobalPivotFinder::new(10, 3, 1.0);
    let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(7);

    let search = finder
        .find_global_pivots(&input, &f, 0.1, &mut rng)
        .unwrap();
    let pivots = &search.pivots;
    // The reported maximum covers every walked residual, accepted or not.
    for pivot in pivots {
        assert!(f(pivot) <= search.max_sampled_error, "{pivot:?}");
    }

    // Should find some pivots since the TT is zero but f is not
    // (except at i=0 or j=0)
    assert!(
        pivots.len() <= 3,
        "Should limit to max_nglobal_pivot=3, got {}",
        pivots.len()
    );
}

#[test]
fn test_custom_global_pivot_finder() {
    struct FixedPivotFinder;

    impl GlobalPivotFinder for FixedPivotFinder {
        fn find_global_pivots<T, F>(
            &self,
            _input: &GlobalPivotSearchInput<T>,
            _f: &F,
            _abs_tol: f64,
            _rng: &mut dyn RngCore,
        ) -> Result<GlobalPivotSearchResult>
        where
            T: Scalar + TTScalar,
            F: Fn(&MultiIndex) -> T,
        {
            // Always return a fixed pivot; this finder measures nothing.
            Ok(GlobalPivotSearchResult {
                pivots: vec![vec![1, 2]],
                max_sampled_error: 0.0,
            })
        }
    }

    let finder = FixedPivotFinder;
    let f = |_: &MultiIndex| 1.0f64;
    let tensors = vec![
        tensor4all_simplett::tensor3_zeros(1, 3, 1).unwrap(),
        tensor4all_simplett::tensor3_zeros(1, 3, 1).unwrap(),
    ];
    let tt = SimpleTensorTrain::new(tensors).unwrap();

    let input = GlobalPivotSearchInput {
        local_dims: vec![3, 3],
        current_tt: tt,
        max_sample_value: 1.0,
        i_set: vec![],
        j_set: vec![],
    };

    let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(7);
    let search = finder
        .find_global_pivots(&input, &f, 0.0, &mut rng)
        .unwrap();
    assert_eq!(search.pivots, vec![vec![1, 2]]);
    assert_eq!(search.max_sampled_error, 0.0);
}
