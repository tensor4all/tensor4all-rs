use crate::matrixluci::block_rook::LazyBlockRookKernel;
use crate::matrixluci::dense::DenseLuKernel;
use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::source::{DenseMatrixSource, LazyMatrixSource};
use crate::matrixluci::types::PivotKernelOptions;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type LazyDenseSource = LazyMatrixSource<f64, Box<dyn Fn(&[usize], &[usize], &mut [f64])>>;

fn unique_pivot_test_matrix() -> Vec<f64> {
    vec![
        9.0, 0.2, 0.3, 0.4, //
        0.1, 8.0, 0.2, 0.3, //
        0.2, 0.1, 7.0, 0.2, //
        0.3, 0.2, 0.1, 6.0, //
    ]
}

fn dense_to_lazy(data: Vec<f64>, nrows: usize, ncols: usize) -> LazyDenseSource {
    LazyMatrixSource::new(
        nrows,
        ncols,
        Box::new(move |rows, cols, out: &mut [f64]| {
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = data[row + nrows * col];
                }
            }
        }),
    )
}

#[test]
fn lazy_block_rook_kernel_matches_dense_kernel_on_unique_pivot_matrix() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);

    let dense_out = DenseLuKernel
        .factorize(&dense, &PivotKernelOptions::no_truncation())
        .unwrap();
    let lazy_out = LazyBlockRookKernel
        .factorize(&lazy, &PivotKernelOptions::no_truncation())
        .unwrap();

    assert_eq!(lazy_out.row_indices, dense_out.row_indices);
    assert_eq!(lazy_out.col_indices, dense_out.col_indices);
    assert_eq!(lazy_out.rank, dense_out.rank);
    assert_eq!(lazy_out.pivot_errors, dense_out.pivot_errors);
}

#[test]
fn lazy_block_rook_kernel_matches_dense_kernel_for_abs_tol_stop() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);
    let options = PivotKernelOptions {
        abs_tol: 6.5,
        ..PivotKernelOptions::default()
    };

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    assert_eq!(lazy_out.row_indices, dense_out.row_indices);
    assert_eq!(lazy_out.col_indices, dense_out.col_indices);
    assert_eq!(lazy_out.rank, dense_out.rank);
    assert_eq!(lazy_out.pivot_errors, dense_out.pivot_errors);
}

#[test]
fn lazy_block_rook_kernel_avoids_full_matrix_request() {
    let data = unique_pivot_test_matrix();
    let max_requested = Arc::new(AtomicUsize::new(0));
    let lazy = LazyMatrixSource::new(4, 4, {
        let max_requested = max_requested.clone();
        move |rows, cols, out: &mut [f64]| {
            max_requested.fetch_max(rows.len() * cols.len(), Ordering::SeqCst);
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = data[row + 4 * col];
                }
            }
        }
    });

    let out = LazyBlockRookKernel
        .factorize(&lazy, &PivotKernelOptions::no_truncation())
        .unwrap();

    assert_eq!(out.rank, 4);
    assert!(max_requested.load(Ordering::SeqCst) < 16);
}

#[test]
fn lazy_block_rook_reconstructs_across_zero_fibers() {
    fn check<T: crate::MatrixLuciScalar>(value: T) {
        for (nrows, ncols) in [(2, 2), (3, 5), (5, 3), (16, 16)] {
            for diagonal_gap in [false, true] {
                for left_orthogonal in [false, true] {
                    let mut data = vec![T::zero(); nrows * ncols];
                    data[nrows * ncols - 1] = value;
                    if diagonal_gap {
                        data[0] = value;
                    }
                    let factors = crate::matrix_luci_factors_from_blocks(
                        nrows,
                        ncols,
                        |rows, cols, out| {
                            for (j, &col) in cols.iter().enumerate() {
                                for (i, &row) in rows.iter().enumerate() {
                                    out[i + rows.len() * j] = data[row + nrows * col];
                                }
                            }
                        },
                        crate::RrLUOptions {
                            left_orthogonal,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    assert_eq!(factors.rank, if diagonal_gap { 2 } else { 1 });
                    let reconstructed =
                        tensor4all_tensorbackend::mat_mul(&factors.left, &factors.right).unwrap();
                    let error = reconstructed
                        .as_col_major_slice()
                        .iter()
                        .zip(&data)
                        .map(|(&actual, &expected)| (actual - expected).abs_val())
                        .fold(0.0_f64, f64::max);
                    assert!(error < 1e-6, "reconstruction error: {error}");
                }
            }
        }
    }
    check(2.0_f32);
    check(2.0_f64);
    check(num_complex::Complex32::new(2.0, 1.0));
    check(num_complex::Complex64::new(2.0, 1.0));
}

#[test]
fn lazy_block_rook_certifies_zero_without_materializing_matrix() {
    for (nrows, ncols) in [(1, 1), (1, 8), (8, 1), (8, 16), (32, 64)] {
        let evaluated = AtomicUsize::new(0);
        let largest_block = AtomicUsize::new(0);
        let source = LazyMatrixSource::new(nrows, ncols, |rows, cols, out: &mut [f64]| {
            evaluated.fetch_add(out.len(), Ordering::Relaxed);
            largest_block.fetch_max(rows.len() * cols.len(), Ordering::Relaxed);
            out.fill(0.0);
        });
        let selection = LazyBlockRookKernel
            .factorize(&source, &PivotKernelOptions::no_truncation())
            .unwrap();
        assert_eq!(selection.rank, 0);
        assert_eq!(selection.pivot_errors, vec![0.0]);
        // Certifying zero requires inspecting all entries, but only a column
        // is materialized at a time, including rectangular and singleton cases.
        assert_eq!(evaluated.load(Ordering::Relaxed), nrows * ncols);
        assert_eq!(largest_block.load(Ordering::Relaxed), nrows);
    }
}

fn rank_cap_options(max_bond_dim: usize) -> PivotKernelOptions {
    PivotKernelOptions {
        max_bond_dim,
        rel_tol: 0.0,
        abs_tol: 0.0,
        left_orthogonal: true,
    }
}

/// Structured matrix whose diagonal magnitudes decrease in column order, so
/// complete pivoting (dense) and rook search (lazy) select the same pivots.
fn decreasing_pivot_matrix(nrows: usize, ncols: usize, seed: u64) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let full_rank = nrows.min(ncols);
    let mut data: Vec<f64> = (0..nrows * ncols)
        .map(|_| 0.01 * (rng.random::<f64>() - 0.5))
        .collect();
    for i in 0..full_rank {
        data[i + nrows * i] = (full_rank - i) as f64;
    }
    data
}

#[test]
fn lazy_rank_cap_error_matches_dense_kernel_across_shapes_and_caps() {
    for (nrows, ncols) in [(4usize, 4usize), (6, 6), (8, 5), (5, 8), (12, 12), (20, 10)] {
        let full_rank = nrows.min(ncols);
        for cap in 1..full_rank {
            for seed in 1..=5u64 {
                let data = decreasing_pivot_matrix(nrows, ncols, seed);
                let dense_source = DenseMatrixSource::from_column_major(&data, nrows, ncols);
                let lazy_source = dense_to_lazy(data.clone(), nrows, ncols);
                let options = rank_cap_options(cap);

                let dense_out = DenseLuKernel.factorize(&dense_source, &options).unwrap();
                let lazy_out = LazyBlockRookKernel
                    .factorize(&lazy_source, &options)
                    .unwrap();

                // Comparing errors is only meaningful on the same pivot set.
                assert_eq!(lazy_out.rank, cap, "{nrows}x{ncols} cap {cap}");
                assert_eq!(dense_out.row_indices, lazy_out.row_indices);
                assert_eq!(dense_out.col_indices, lazy_out.col_indices);
                assert_eq!(lazy_out.pivot_errors.len(), cap + 1);

                let dense_error = *dense_out.pivot_errors.last().unwrap();
                let lazy_error = *lazy_out.pivot_errors.last().unwrap();
                // Both kernels report the maximum magnitude of the remaining
                // residual, but the dense kernel reads it off its LU diagonal
                // and the lazy kernel subtracts A[:, J] A[I, J]^{-1} A[I, :],
                // so agreement holds to rounding noise only (measured worst
                // case: 1.3 ULP).
                let relative = (dense_error - lazy_error).abs() / dense_error.abs();
                assert!(
                    relative <= 4.0 * f64::EPSILON,
                    "{nrows}x{ncols} cap {cap} seed {seed}: dense {dense_error:e} lazy {lazy_error:e}"
                );
            }
        }
    }
}

#[test]
fn lazy_rank_cap_error_is_the_remaining_residual_maximum() {
    // A[0, 0] = 4 is the largest entry, so both kernels pivot there first and
    // the remaining 2x2 Schur complement is exactly [[3, 0], [0, 1]].
    fn check<T>(four: f64, two: f64, three: f64, one: f64)
    where
        T: crate::MatrixLuciScalar,
        DenseLuKernel: PivotKernel<T>,
        LazyBlockRookKernel: PivotKernel<T>,
    {
        let convert = |value: f64| T::from_f64(value);
        let data = vec![
            convert(four),
            convert(two),
            convert(0.0),
            convert(two),
            convert(four),
            convert(0.0),
            convert(0.0),
            convert(0.0),
            convert(one),
        ];
        let dense = DenseMatrixSource::from_column_major(&data, 3, 3);
        let lazy = LazyMatrixSource::new(3, 3, {
            let data = data.clone();
            move |rows, cols, out: &mut [T]| {
                for (j, &col) in cols.iter().enumerate() {
                    for (i, &row) in rows.iter().enumerate() {
                        out[i + rows.len() * j] = data[row + 3 * col];
                    }
                }
            }
        });
        let options = rank_cap_options(1);
        let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
        let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

        assert_eq!(dense_out.pivot_errors, vec![four, three]);
        assert_eq!(lazy_out.pivot_errors, vec![four, three]);
        // The buggy value was the last accepted pivot (4), not the residual (3).
        assert_ne!(lazy_out.pivot_errors[0], lazy_out.pivot_errors[1]);
    }
    check::<f64>(4.0, 2.0, 3.0, 1.0);
    check::<f32>(4.0, 2.0, 3.0, 1.0);
    check::<num_complex::Complex64>(4.0, 2.0, 3.0, 1.0);
    check::<num_complex::Complex32>(4.0, 2.0, 3.0, 1.0);
}

#[test]
fn lazy_rank_cap_error_ignores_the_last_accepted_pivot() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);
    let options = rank_cap_options(2);

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    assert_eq!(lazy_out.pivot_errors, dense_out.pivot_errors);
    let accepted = lazy_out.pivot_errors[lazy_out.rank - 1];
    assert_eq!(
        lazy_out.pivot_errors.last().copied(),
        Some(6.990983606557377)
    );
    assert_eq!(accepted, 7.997777777777777);
}

#[test]
fn lazy_zero_rank_cap_reports_the_whole_matrix_maximum() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);
    let options = rank_cap_options(0);

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    // Nothing was accepted, so the whole matrix is residual and its maximum is
    // the trailing entry. The previous lazy value here was NaN.
    assert_eq!(lazy_out.rank, 0);
    assert_eq!(dense_out.pivot_errors, vec![9.0]);
    assert_eq!(lazy_out.pivot_errors, vec![9.0]);
}

#[test]
fn full_rank_reports_zero_last_error_in_both_kernels() {
    for (nrows, ncols) in [(4usize, 4usize), (8, 5), (5, 8)] {
        let data = decreasing_pivot_matrix(nrows, ncols, 7);
        let dense = DenseMatrixSource::from_column_major(&data, nrows, ncols);
        let lazy = dense_to_lazy(data.clone(), nrows, ncols);
        let full_rank = nrows.min(ncols);

        for options in [
            PivotKernelOptions::no_truncation(),
            rank_cap_options(full_rank + 3),
        ] {
            let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
            let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();
            assert_eq!(dense_out.rank, full_rank);
            assert_eq!(lazy_out.rank, full_rank);
            assert_eq!(dense_out.pivot_errors.last().copied(), Some(0.0));
            assert_eq!(lazy_out.pivot_errors.last().copied(), Some(0.0));
        }
    }
}

#[test]
fn lazy_bond_error_reports_the_remaining_residual_at_a_rank_cap() {
    let data = unique_pivot_test_matrix();
    let factors = crate::matrix_luci_factors_from_blocks(
        4,
        4,
        |rows, cols, out| {
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = data[row + 4 * col];
                }
            }
        },
        crate::RrLUOptions {
            max_bond_dim: 2,
            rel_tol: 0.0,
            abs_tol: 0.0,
            left_orthogonal: true,
        },
    )
    .unwrap();

    assert_eq!(factors.rank, 2);
    // Downstream bond error: the largest discarded residual. Before the fix
    // this reported the last accepted pivot, 7.997777777777777.
    assert_eq!(
        factors.pivot_errors,
        vec![9.0, 7.997777777777777, 6.990983606557377]
    );
}

#[test]
fn lazy_zero_rank_cap_error_ignores_the_pivot_rule_difference() {
    // The rook search converges on the local maximum 3 at (1, 0) while complete
    // pivoting reaches the global maximum 10 at (2, 2). The trailing entry is
    // the maximum of the residual block, so both kernels still agree.
    let data = vec![
        1.0_f64, 3.0, 0.0, // column 0
        2.0, 0.5, 0.0, // column 1
        0.0, 0.0, 10.0, // column 2
    ];
    let dense = DenseMatrixSource::from_column_major(&data, 3, 3);
    let lazy = dense_to_lazy(data.clone(), 3, 3);
    let options = rank_cap_options(0);

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    assert_eq!(dense_out.pivot_errors, vec![10.0]);
    assert_eq!(lazy_out.pivot_errors, vec![10.0]);
}

#[test]
fn lazy_residual_scan_rejects_non_finite_residuals() {
    // A zero cap never enters the search loop, so the trailing measurement has
    // to report the non-finite residual itself.
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let lazy = dense_to_lazy(vec![invalid], 1, 1);
        let error = LazyBlockRookKernel
            .factorize(&lazy, &rank_cap_options(0))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::matrixluci::MatrixLuciError::NaNEncountered { .. }
        ));

        // The pivot search only evaluates its own pivot row and column, so the
        // trailing scan is what first sees this entry.
        let data = vec![4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, invalid];
        let lazy = dense_to_lazy(data, 3, 3);
        let error = LazyBlockRookKernel
            .factorize(&lazy, &rank_cap_options(1))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::matrixluci::MatrixLuciError::NaNEncountered { .. }
        ));
    }
}
