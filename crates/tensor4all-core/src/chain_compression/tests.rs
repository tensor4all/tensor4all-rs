use super::*;
use tensor4all_tensorbackend::CpuBackend;

fn diagonal_chain(small: f64) -> [ColMajorArray<f64>; 2] {
    let identity = ColMajorArray::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2]).unwrap();
    let diagonal = ColMajorArray::new(vec![3.0, 0.0, 0.0, small], vec![2, 2, 1]).unwrap();
    [identity, diagonal]
}

#[test]
fn lu_compression_preserves_thresholds_and_rank_cap() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    for (small, tolerance, max_bond_dim, normalize_error, rank) in [
        (1.0, 0.5, None, true, 1),
        (1.0, 0.5, None, false, 2),
        (1.0, 0.5, Some(1), false, 1),
        (1e-15, 0.0, None, true, 1),
        (1e-15, 0.0, None, false, 1),
    ] {
        for factorization in [ChainLuFactorization::RrLu, ChainLuFactorization::Luci] {
            let options = ChainLuOptions {
                tolerance,
                max_bond_dim,
                normalize_error,
                factorization,
            };
            let mut global = diagonal_chain(small);
            compress_chain_lu_mut(&mut global, &options).unwrap();
            let mut on_session = diagonal_chain(small);
            compress_chain_lu_on_session(&mut session, &mut on_session, &options).unwrap();
            assert_eq!(global, on_session, "{options:?}");

            let cores = global;
            assert_eq!(cores[0].shape(), &[1, 2, rank], "{options:?}");
            assert_eq!(cores[1].shape(), &[rank, 2, 1], "{options:?}");
            let left = Matrix::from_col_major_vec(2, rank, cores[0].data().to_vec());
            let right = Matrix::from_col_major_vec(rank, 2, cores[1].data().to_vec());
            let product = mat_mul(&left, &right).unwrap();
            let retained = if rank == 2 { small } else { 0.0 };
            assert_eq!(
                product.as_col_major_slice(),
                &[3.0, 0.0, 0.0, retained],
                "{options:?}"
            );
        }
    }
}

#[test]
fn lu_compression_rejects_invalid_options_before_touching_cores() {
    for (tolerance, max_bond_dim, message) in [
        (-1e-8, None, "finite and nonnegative, got -0.00000001"),
        (f64::NAN, None, "finite and nonnegative, got NaN"),
        (f64::INFINITY, None, "finite and nonnegative, got inf"),
        (0.0, Some(0), "max_bond_dim must be positive"),
    ] {
        let options = ChainLuOptions {
            tolerance,
            max_bond_dim,
            ..ChainLuOptions::default()
        };
        let mut cores = diagonal_chain(1.0);
        let error = compress_chain_lu_mut(&mut cores, &options).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(cores, diagonal_chain(1.0));
    }
}

#[test]
fn chain_validation_checks_shape_before_sweeping() {
    let invalid = ColMajorArray::new(vec![1.0_f64], vec![1, 1]).unwrap();
    assert!(matches!(
        compress_chain_lu_mut(&mut [invalid], &ChainLuOptions::default()),
        Err(ChainCompressionError::CoreRank { site: 0, rank: 2 })
    ));

    let left = ColMajorArray::new(vec![1.0_f64; 2], vec![1, 1, 2]).unwrap();
    let right = ColMajorArray::new(vec![1.0_f64], vec![1, 1, 1]).unwrap();
    let mut cores = [left, right];
    let before = cores.clone();
    assert!(matches!(
        compress_chain_lu_mut(&mut cores, &ChainLuOptions::default()),
        Err(ChainCompressionError::BondMismatch { site: 0, .. })
    ));
    assert_eq!(cores, before);
}

#[test]
fn unfolds_and_repacks_in_row_major_fused_order() {
    let data: Vec<_> = (0..12).collect();
    let left = swap_blocks(&data, 1, 2, 3).unwrap();
    assert_eq!(left, [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11]);
    assert_eq!(swap_blocks(&left, 1, 3, 2).unwrap(), data);
    let right = swap_blocks(&data, 2, 3, 2).unwrap();
    assert_eq!(right, [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]);
    assert_eq!(swap_blocks(&right, 2, 2, 3).unwrap(), data);
}

#[test]
fn unfolding_rejects_a_partial_block() {
    let data: Vec<_> = (0..13).collect();
    let error = swap_blocks(&data, 1, 2, 3).unwrap_err();
    assert!(matches!(
        error,
        ChainCompressionError::BufferLength { len: 13, block: 6 }
    ));
    assert!(error
        .to_string()
        .contains("length 13 is not a whole number of 6-element blocks"));
    // A size-one axis leaves the order unchanged but still checks the length.
    assert!(swap_blocks(&data, 2, 1, 3).is_err());
    assert!(swap_blocks(&data, 1, 0, 3).is_err());
}

#[test]
fn rejects_inconsistent_factors_before_multiplication() {
    let core = ColMajorArray::new(vec![1.0_f64], vec![1, 1, 1]).unwrap();
    let mut cores = [core.clone(), core];
    let result = compress_chain_cores_mut(
        &mut cores,
        &mut (),
        |(), _, _| Ok((Matrix::zeros(1, 1), Matrix::zeros(2, 1))),
        |(), _, _| panic!("inconsistent factors must be rejected before matrix multiplication"),
    );
    assert!(matches!(
        result,
        Err(ChainCompressionError::Factorization(
            MatrixCIError::DimensionMismatch {
                expected_rows: 1,
                actual_rows: 2,
                ..
            }
        ))
    ));
    assert!(cores.iter().all(|core| core.data() == [1.0]));
}
