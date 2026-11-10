use crate::matrixluci::dense::DenseLuKernel;
use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::source::{DenseMatrixSource, LazyMatrixSource};
use crate::matrixluci::types::{PivotKernelOptions, PivotSelectionCore};
use crate::{rrlu, RrLUOptions};
use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use tenferro_tensor::TensorScalar;
use tensor4all_tensorbackend::{from_vec2d, full_piv_lu_matrix_owned, Matrix};

#[test]
fn dense_kernel_keeps_small_nonzero_pivots() {
    for scale in [1.0, 1e-20] {
        let data = [2.0 * scale, 0.0, 0.0, scale];
        let source = DenseMatrixSource::from_column_major(&data, 2, 2);
        for options in [
            PivotKernelOptions::default(),
            PivotKernelOptions::no_truncation(),
        ] {
            let selection = DenseLuKernel.factorize(&source, &options).unwrap();
            assert_eq!(selection.row_indices, vec![0, 1]);
            assert_eq!(selection.col_indices, vec![0, 1]);
            assert_eq!(selection.pivot_errors, vec![2.0 * scale, scale, 0.0]);
            assert_eq!(selection.rank, 2);
        }
    }
}

#[test]
fn dense_kernel_recovers_identity_pivots() {
    let src =
        DenseMatrixSource::from_column_major(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let kernel = DenseLuKernel;
    let out = kernel
        .factorize(&src, &PivotKernelOptions::no_truncation())
        .unwrap();
    assert_eq!(out.rank, 3);
    assert_eq!(out.row_indices.len(), 3);
    assert_eq!(out.col_indices.len(), 3);
}

#[test]
fn dense_kernel_reports_zero_rank_for_zero_matrix() {
    let src = DenseMatrixSource::from_column_major(&[0.0; 9], 3, 3);
    let kernel = DenseLuKernel;
    let out = kernel
        .factorize(&src, &PivotKernelOptions::no_truncation())
        .unwrap();
    assert_eq!(out.rank, 0);
}

#[test]
fn pivot_errors_cover_empty_and_zero_pivot_stops() {
    assert_eq!(
        DenseLuKernel::compute_pivot_errors::<f64>(&[], 0, 3, &PivotKernelOptions::default()),
        vec![0.0]
    );
    assert_eq!(
        DenseLuKernel::compute_pivot_errors::<f64>(
            &[2.0, 0.0],
            2,
            2,
            &PivotKernelOptions::no_truncation()
        ),
        vec![2.0, 0.0]
    );
    assert_eq!(
        DenseLuKernel::compute_pivot_errors::<f64>(&[0.0], 1, 1, &PivotKernelOptions::default()),
        vec![0.0]
    );
}

#[test]
fn rank_ceiling_errors_include_the_next_residual() {
    for (max_bond_dim, expected) in [(0, vec![4.0]), (1, vec![4.0, 2.0])] {
        assert_eq!(
            DenseLuKernel::compute_pivot_errors::<f64>(
                &[4.0, 2.0],
                2,
                2,
                &PivotKernelOptions {
                    max_bond_dim,
                    ..Default::default()
                }
            ),
            expected
        );
    }
    let rows = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let options = PivotKernelOptions {
        max_bond_dim: 1,
        ..Default::default()
    };
    let selection = dense_factorize(&rows, options.clone());
    assert_eq!(selection.rank, 1);
    assert_eq!(selection.pivot_errors, vec![1.0, 0.0]);
    assert_pivot_parity(
        rows,
        options,
        RrLUOptions {
            max_bond_dim: 1,
            ..Default::default()
        },
    );
}

#[test]
fn dense_kernel_uses_rectangular_rrlu_fallback() {
    let out = dense_factorize(
        &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 7.0]],
        PivotKernelOptions::no_truncation(),
    );

    assert_eq!(out.rank, 2);
    assert_eq!(out.row_indices.len(), 2);
    assert_eq!(out.col_indices.len(), 2);
    assert_eq!(out.pivot_errors.last().copied(), Some(0.0));
}

#[test]
fn dense_kernel_materializes_lazy_source() {
    let data = col_major_from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    let source = LazyMatrixSource::new(2, 2, |rows: &[usize], cols: &[usize], out: &mut [f64]| {
        for (j, &col) in cols.iter().enumerate() {
            for (i, &row) in rows.iter().enumerate() {
                out[i + rows.len() * j] = data[row + 2 * col];
            }
        }
    });

    let out = DenseLuKernel
        .factorize(&source, &PivotKernelOptions::no_truncation())
        .unwrap();

    assert_eq!(out.rank, 2);
    assert_eq!(out.row_indices, vec![0, 1]);
    assert_eq!(out.col_indices, vec![0, 1]);
}

fn col_major_from_rows(rows: &[Vec<f64>]) -> Vec<f64> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let mut out = Vec::with_capacity(nrows * ncols);
    for col in 0..ncols {
        for row in rows {
            out.push(row[col]);
        }
    }
    out
}

fn dense_factorize(
    rows: &[Vec<f64>],
    options: PivotKernelOptions,
) -> crate::matrixluci::PivotSelectionCore {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let data = col_major_from_rows(rows);
    let src = DenseMatrixSource::from_column_major(&data, nrows, ncols);
    DenseLuKernel.factorize(&src, &options).unwrap()
}

fn assert_pivot_parity(
    rows: Vec<Vec<f64>>,
    luci_options: PivotKernelOptions,
    legacy_options: RrLUOptions,
) {
    let new = dense_factorize(&rows, luci_options);
    let legacy = rrlu(&from_vec2d(rows), Some(legacy_options)).unwrap();

    assert_eq!(new.rank, legacy.npivots());
    assert_eq!(new.row_indices, legacy.row_indices());
    assert_eq!(new.col_indices, legacy.col_indices());
    assert_eq!(new.pivot_errors.len(), legacy.pivot_errors().len());
    for (lhs, rhs) in new.pivot_errors.iter().zip(legacy.pivot_errors()) {
        assert_abs_diff_eq!(*lhs, rhs, epsilon = 1e-12);
    }
}

#[test]
fn dense_kernel_matches_legacy_pivot_errors_for_full_rank() {
    let rows = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert_pivot_parity(
        rows,
        PivotKernelOptions::no_truncation(),
        RrLUOptions::default(),
    );
}

#[test]
fn dense_kernel_matches_legacy_pivot_errors_for_max_bond_dim_stop() {
    let rows = vec![
        vec![0.433088, 0.956638, 0.0907974, 0.0447859, 0.0196053],
        vec![0.855517, 0.782503, 0.291197, 0.540828, 0.358579],
        vec![0.37455, 0.536457, 0.205479, 0.75896, 0.701206],
        vec![0.47272, 0.0172539, 0.518177, 0.242864, 0.461635],
        vec![0.0676373, 0.450878, 0.672335, 0.77726, 0.540691],
    ];
    let luci_options = PivotKernelOptions {
        max_bond_dim: 2,
        ..PivotKernelOptions::default()
    };
    let legacy_options = RrLUOptions {
        max_bond_dim: 2,
        ..RrLUOptions::default()
    };
    assert_pivot_parity(rows, luci_options, legacy_options);
}

#[test]
fn dense_kernel_matches_legacy_pivot_errors_for_abs_tol_stop() {
    let rows = vec![
        vec![0.433088, 0.956638, 0.0907974, 0.0447859, 0.0196053],
        vec![0.855517, 0.782503, 0.291197, 0.540828, 0.358579],
        vec![0.37455, 0.536457, 0.205479, 0.75896, 0.701206],
        vec![0.47272, 0.0172539, 0.518177, 0.242864, 0.461635],
        vec![0.0676373, 0.450878, 0.672335, 0.77726, 0.540691],
    ];
    let luci_options = PivotKernelOptions {
        abs_tol: 0.5,
        ..PivotKernelOptions::default()
    };
    let legacy_options = RrLUOptions {
        abs_tol: 0.5,
        ..RrLUOptions::default()
    };
    assert_pivot_parity(rows, luci_options, legacy_options);
}

#[test]
fn dense_kernel_matches_legacy_pivot_errors_for_zero_matrix() {
    let rows = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    assert_pivot_parity(
        rows,
        PivotKernelOptions::no_truncation(),
        RrLUOptions::default(),
    );
}

#[test]
fn dense_kernel_matches_legacy_pivot_errors_for_right_orthogonal_mode() {
    let rows = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ];
    let luci_options = PivotKernelOptions {
        left_orthogonal: false,
        ..PivotKernelOptions::no_truncation()
    };
    let legacy_options = RrLUOptions {
        left_orthogonal: false,
        ..RrLUOptions::default()
    };
    assert_pivot_parity(rows, luci_options, legacy_options);
}

#[test]
fn dense_selection_extreme_scales() {
    use num_complex::Complex64;
    for phase in [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.5)] {
        let base: Vec<_> = [4.0, 1.0, 2.0, 3.0].iter().map(|&v| phase * v).collect();
        for options in [
            PivotKernelOptions::default(),
            PivotKernelOptions::no_truncation(),
        ] {
            let reference = DenseLuKernel
                .factorize(&DenseMatrixSource::from_column_major(&base, 2, 2), &options)
                .unwrap();
            for scale in [2f64.powi(-600), 2f64.powi(600)] {
                let data: Vec<_> = base.iter().map(|&v| v * scale).collect();
                let out = DenseLuKernel
                    .factorize(&DenseMatrixSource::from_column_major(&data, 2, 2), &options)
                    .unwrap();
                assert_eq!(out.rank, reference.rank);
                assert_eq!(out.row_indices, reference.row_indices);
                assert_eq!(out.col_indices, reference.col_indices);
                for (a, b) in out.pivot_errors.iter().zip(&reference.pivot_errors) {
                    assert!((a / scale - b).abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn dense_errors_stop_at_undividable_pivots_in_the_component_type() {
    let tiny = f32::from_bits(1) as f64;
    for options in [
        PivotKernelOptions::default(),
        PivotKernelOptions::no_truncation(),
    ] {
        assert_eq!(
            DenseLuKernel::compute_pivot_errors::<f32>(&[tiny], 1, 1, &options),
            vec![tiny]
        );
        assert_eq!(
            DenseLuKernel::compute_pivot_errors::<f32>(&[1.0, tiny], 2, 2, &options).len(),
            2
        );
    }
}

#[test]
fn dense_selection_reports_non_finite_input_and_diagonals() {
    use crate::matrixluci::MatrixLuciError;
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let data = [invalid];
        let source = DenseMatrixSource::from_column_major(&data, 1, 1);
        assert!(matches!(
            DenseLuKernel.factorize(&source, &PivotKernelOptions::default()),
            Err(MatrixLuciError::NaNEncountered { .. })
        ));
    }
    let values = [f64::MAX, f64::MAX, -f64::MAX, f64::MAX];
    let source = DenseMatrixSource::from_column_major(&values, 2, 2);
    assert!(matches!(
        DenseLuKernel.factorize(&source, &PivotKernelOptions::default()),
        Err(MatrixLuciError::NaNEncountered { .. })
    ));
}

/// Deterministic entries in `[-1, 1)` from an integer generator: no RNG dependency, no platform
/// drift, and the same matrix in every run.
fn lcg(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// The square path as it was before the factorization stopped at the rank cap: factor to full rank
/// through the backend, then truncate by scanning the diagonal.
fn previous_square_selection<T>(
    data: &[T],
    n: usize,
    options: &PivotKernelOptions,
) -> PivotSelectionCore
where
    T: crate::MatrixLuciScalar + TensorScalar,
{
    let matrix = Matrix::from_col_major_vec(n, n, data.to_vec());
    let decomp = full_piv_lu_matrix_owned(matrix).unwrap();
    let u_values = decomp.u.as_col_major_slice();
    let diag_abs = (0..n)
        .map(|i| u_values[i + n * i].abs_val())
        .collect::<Vec<_>>();
    let pivot_errors = DenseLuKernel::compute_pivot_errors::<T>(&diag_abs, n, n, options);
    let rank = pivot_errors.len().saturating_sub(1);
    let permutation = |matrix: &Matrix<T>| {
        let data = matrix.as_col_major_slice();
        (0..n)
            .map(|row| {
                (0..n)
                    .find(|&col| data[row + n * col].abs_val() > 0.5)
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let (rows, cols) = (permutation(&decomp.p), permutation(&decomp.q));
    PivotSelectionCore {
        row_indices: rows[..rank].to_vec(),
        col_indices: cols[..rank].to_vec(),
        pivot_errors,
        rank,
    }
}

fn square_selection<
    T: crate::MatrixLuciScalar + tensor4all_tensorbackend::CappedFullPivLuScalar,
>(
    data: &[T],
    n: usize,
    options: &PivotKernelOptions,
) -> (PivotSelectionCore, usize) {
    DenseLuKernel::factorize_square_with_backend(data, n, options).unwrap()
}

fn assert_selection_bits_equal(
    actual: &PivotSelectionCore,
    expected: &PivotSelectionCore,
    what: &str,
) {
    assert_eq!(actual.rank, expected.rank, "{what}: rank");
    assert_eq!(
        actual.row_indices, expected.row_indices,
        "{what}: row pivots"
    );
    assert_eq!(
        actual.col_indices, expected.col_indices,
        "{what}: column pivots"
    );
    assert_eq!(
        actual.pivot_errors.len(),
        expected.pivot_errors.len(),
        "{what}: pivot error count"
    );
    for (actual, expected) in actual.pivot_errors.iter().zip(&expected.pivot_errors) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{what}: pivot errors must agree bit for bit"
        );
    }
}

/// Within one lock, the backend's capped LU reproduces the `full_piv_lu_matrix_owned` + diagonal-scan
/// path it replaced, bit for bit: the pivots, the rank and the error history.
///
/// What this cannot prove is the cross-version claim. Both sides of the comparison are built against
/// the faer in this lock, so a change between 0.28.2 and 0.33.0 would move them together; that claim
/// rests on the measurement taken while the lock still held 0.28.2 and is not re-derivable here.
#[test]
fn square_kernel_keeps_the_previous_selection_bit_for_bit() {
    for n in [1usize, 2, 8, 37, 64] {
        let data = lcg(n * n, 11 + n as u64);
        let phased = data
            .iter()
            .map(|&value| Complex64::new(value, 0.5 * value + 0.25))
            .collect::<Vec<_>>();
        for options in [
            PivotKernelOptions::no_truncation(),
            PivotKernelOptions::default(),
        ] {
            let what = format!("f64 n={n} max_bond_dim={}", options.max_bond_dim);
            let (actual, pivots) = square_selection(&data, n, &options);
            assert_eq!(pivots, n, "{what}: an uncapped run takes every pivot");
            assert_selection_bits_equal(
                &actual,
                &previous_square_selection(&data, n, &options),
                &what,
            );

            let what = format!("c64 n={n} max_bond_dim={}", options.max_bond_dim);
            let (actual, _) = square_selection(&phased, n, &options);
            assert_selection_bits_equal(
                &actual,
                &previous_square_selection(&phased, n, &options),
                &what,
            );
        }
    }
}

/// The score the backend's complete-pivoting search maximises over a complex block. It depends on
/// the SIMD path faer dispatches to at run time and both are the contract (see
/// `tensor4all_tensorbackend::capped_full_piv_lu`; its tests pin the path against CPU detection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComplexPivotScore {
    /// Vector dispatch: `max(|re|, |im|)`.
    LargerPart,
    /// Scalar dispatch: `|re| + |im|`.
    PartSum,
}

impl ComplexPivotScore {
    const BOTH: [Self; 2] = [Self::LargerPart, Self::PartSum];

    fn of(self, value: Complex64) -> f64 {
        match self {
            Self::LargerPart => value.re.abs().max(value.im.abs()),
            Self::PartSum => value.re.abs() + value.im.abs(),
        }
    }

    /// The score this host's backend runs, probed on `diag(3, 2.5 + 2.5i)`: the larger part takes
    /// `3` (3 > 2.5) first, the part sum `2.5 + 2.5i` (5 > 3).
    fn of_this_host() -> Self {
        let probe = tensor4all_tensorbackend::capped_full_piv_lu(
            Matrix::from_col_major_vec(
                2,
                2,
                vec![
                    Complex64::new(3.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(2.5, 2.5),
                ],
            ),
            Some(1),
        )
        .unwrap();
        if probe.diagonal()[0] == Complex64::new(3.0, 0.0) {
            Self::LargerPart
        } else {
            assert_eq!(probe.diagonal()[0], Complex64::new(2.5, 2.5));
            Self::PartSum
        }
    }
}

/// Reference model of complete pivoting on a diagonal matrix under `score`: the pivots are the
/// diagonal in descending score, so the kernel's `pivot_errors` are their moduli in that order,
/// then `0.0` for the exhausted block (uncapped) or the next pivot's modulus as the residual
/// (capped at `cap`).
fn diagonal_pivot_errors(diag: &[Complex64], score: ComplexPivotScore, cap: usize) -> Vec<f64> {
    let mut order = diag.to_vec();
    order.sort_by(|a, b| score.of(*b).total_cmp(&score.of(*a)));
    let mut errors = order.iter().map(|value| value.norm()).collect::<Vec<_>>();
    errors.push(0.0);
    errors.truncate(cap + 1);
    errors
}

/// Both arms of the reference model, stated as literals on `diag(10, 3, 2.5 + 2.5i)`, so neither
/// score is the reference the other is checked against. This runs on every host; only the arm the
/// host dispatches to is also compared with the kernel below.
#[test]
fn the_diagonal_reference_model_orders_pivots_by_each_score() {
    let diag = [
        Complex64::new(10.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.5, 2.5),
    ];
    let modulus = Complex64::new(2.5, 2.5).norm();
    assert_eq!(modulus, 3.5355339059327378);
    assert_eq!(
        diagonal_pivot_errors(&diag, ComplexPivotScore::LargerPart, 3),
        vec![10.0, 3.0, modulus, 0.0]
    );
    assert_eq!(
        diagonal_pivot_errors(&diag, ComplexPivotScore::PartSum, 3),
        vec![10.0, modulus, 3.0, 0.0]
    );
    assert_eq!(
        diagonal_pivot_errors(&diag, ComplexPivotScore::LargerPart, 1),
        vec![10.0, 3.0]
    );
    assert_eq!(
        diagonal_pivot_errors(&diag, ComplexPivotScore::PartSum, 1),
        vec![10.0, modulus]
    );
    for score in ComplexPivotScore::BOTH {
        assert_eq!(
            diagonal_pivot_errors(&diag, score, 2).len(),
            3,
            "{score:?}: two pivots and a residual"
        );
    }
}

/// The residual a capped run reports is the pivot the untruncated factorization would take next,
/// for a complex matrix whose second pivot depends on the dispatched score: of `3` and
/// `2.5 + 2.5i` the larger part names `3` and the part sum (like the modulus, `3.54`) names the
/// complex entry. The expectation is the reference model under the score this host runs.
#[test]
fn a_capped_complex_run_reports_the_next_pivot_faer_would_take() {
    let score = ComplexPivotScore::of_this_host();
    let diag = [
        Complex64::new(10.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.5, 2.5),
    ];
    let zero = Complex64::new(0.0, 0.0);
    let data = [
        diag[0], zero, zero, //
        zero, diag[1], zero, //
        zero, zero, diag[2],
    ];
    let uncapped = PivotKernelOptions::no_truncation();
    let (full, _) = square_selection(&data, 3, &uncapped);
    assert_eq!(
        full.pivot_errors,
        diagonal_pivot_errors(&diag, score, 3),
        "{score:?}"
    );

    for cap in [1usize, 2] {
        let options = PivotKernelOptions {
            max_bond_dim: cap,
            ..uncapped
        };
        let (selection, pivots) = square_selection(&data, 3, &options);
        assert_eq!(
            pivots,
            cap + 1,
            "cap={cap}: the cap stops the elimination one pivot past itself"
        );
        assert_selection_bits_equal(
            &selection,
            &previous_square_selection(&data, 3, &options),
            &format!("c64 cap={cap}"),
        );
        // The residual is the next diagonal of the untruncated run under the same score, not the
        // trailing block's largest modulus.
        assert_eq!(
            selection.pivot_errors,
            diagonal_pivot_errors(&diag, score, cap),
            "{score:?} cap={cap}"
        );
        assert_eq!(selection.pivot_errors[cap], full.pivot_errors[cap]);
    }
}

/// A subnormal trailing block stops faer before the rank cap does, so the capped run must report
/// what the untruncated run reports - the trailing diagonal - instead of consuming that stop's
/// residual as one more pivot. `diag(1e-308, 9e-309, 8e-309)` capped at 2 keeps two pivots; the
/// residual-as-pivot reading returned rank 1 with a `0.0` error.
#[test]
fn a_subnormal_stop_before_the_cap_keeps_the_previous_selection() {
    let data = [
        1e-308f64, 0.0, 0.0, // column 0
        0.0, 9e-309, 0.0, // column 1
        0.0, 0.0, 8e-309, // column 2
    ];
    let uncapped = PivotKernelOptions::no_truncation();
    for cap in [1usize, 2, 3] {
        let options = PivotKernelOptions {
            max_bond_dim: cap,
            ..uncapped
        };
        let (selection, pivots) = square_selection(&data, 3, &options);
        assert_eq!(
            pivots,
            if cap < 3 { 0 } else { 3 },
            "cap={cap}: faer stopped before the cap"
        );
        assert_selection_bits_equal(
            &selection,
            &previous_square_selection(&data, 3, &options),
            &format!("subnormal cap={cap}"),
        );
    }

    // Stated directly, so a regression cannot hide behind both sides moving together: the caps below
    // the matrix's rank keep every subnormal pivot the untruncated run kept.
    let cap2 = PivotKernelOptions {
        max_bond_dim: 2,
        ..uncapped
    };
    let (selection, _) = square_selection(&data, 3, &cap2);
    assert_eq!(selection.rank, 2);
    assert_eq!(selection.pivot_errors, vec![1e-308, 9e-309, 8e-309]);
}

/// A rank cap stops the factorization at the cap: the pivots faer actually takes are the cap plus
/// the one extra pivot the residual needs, not `n`, and the selection is what the uncapped
/// factorization returns after truncation.
///
/// The expected count is `cap + 1` on purpose. A count clamped to the cap (`pivots.min(cap)`) reads
/// the same for a capped run and for a full factorization whose result was truncated afterwards, so
/// it cannot fail when the cap stops being passed to faer; with the raw `info.rank` it does.
#[test]
fn rank_cap_stops_the_square_factorization_at_the_cap() {
    let (n, data) = (64usize, lcg(64 * 64, 29));
    let uncapped = PivotKernelOptions::no_truncation();
    let (_, pivots) = square_selection(&data, n, &uncapped);
    assert_eq!(pivots, n, "the uncapped run takes every pivot");

    for cap in [1usize, 7, 16, 63] {
        let options = PivotKernelOptions {
            max_bond_dim: cap,
            ..PivotKernelOptions::no_truncation()
        };
        let (selection, pivots) = square_selection(&data, n, &options);
        assert_eq!(
            pivots,
            cap + 1,
            "cap={cap}: the factorization stops at the cap, plus the extra pivot asked for"
        );
        assert_eq!(
            selection.rank, cap,
            "cap={cap}: the selection is the capped rank"
        );
        assert_selection_bits_equal(
            &selection,
            &previous_square_selection(&data, n, &options),
            &format!("cap={cap}"),
        );
    }
}

/// The absolute tolerance keeps its meaning across the switch: a pivot equal to the tolerance is
/// accepted, one just below it stops the factorization.
#[test]
fn absolute_tolerance_boundary_is_the_previous_one() {
    // diag(4, 2) in column-major order; complete pivoting takes 4 then 2.
    let data = [4.0f64, 0.0, 0.0, 2.0];
    let with_abs_tol = |abs_tol: f64| PivotKernelOptions {
        abs_tol,
        ..PivotKernelOptions::no_truncation()
    };

    let at = square_selection(&data, 2, &with_abs_tol(2.0)).0;
    assert_eq!(at.rank, 2, "a pivot exactly at abs_tol is accepted");
    assert_eq!(at.pivot_errors, vec![4.0, 2.0, 0.0]);

    let below = square_selection(&data, 2, &with_abs_tol(2.0f64.next_down())).0;
    assert_eq!(below.rank, 2, "abs_tol just under the pivot keeps it");
    assert_eq!(below.pivot_errors, vec![4.0, 2.0, 0.0]);

    let above = square_selection(&data, 2, &with_abs_tol(2.0f64.next_up())).0;
    assert_eq!(above.rank, 1, "abs_tol just over the pivot stops there");
    assert_eq!(above.pivot_errors, vec![4.0, 2.0]);

    for abs_tol in [2.0, 2.0f64.next_down(), 2.0f64.next_up()] {
        assert_selection_bits_equal(
            &square_selection(&data, 2, &with_abs_tol(abs_tol)).0,
            &previous_square_selection(&data, 2, &with_abs_tol(abs_tol)),
            &format!("abs_tol={abs_tol}"),
        );
    }
}

/// The relative tolerance is measured against the largest accepted pivot - a running maximum, not
/// the first pivot - and a pivot exactly at `rel_tol * max_accepted` is accepted.
#[test]
fn relative_tolerance_boundary_is_the_previous_one() {
    // diag(4, 2): the second pivot is exactly `0.5 * 4`.
    let data = [4.0f64, 0.0, 0.0, 2.0];
    let with_rel_tol = |rel_tol: f64| PivotKernelOptions {
        rel_tol,
        ..PivotKernelOptions::no_truncation()
    };

    let at = square_selection(&data, 2, &with_rel_tol(0.5)).0;
    assert_eq!(at.rank, 2, "a pivot exactly at rel_tol is accepted");
    assert_eq!(at.pivot_errors, vec![4.0, 2.0, 0.0]);

    let below = square_selection(&data, 2, &with_rel_tol(0.5f64.next_down())).0;
    assert_eq!(below.rank, 2);

    let above = square_selection(&data, 2, &with_rel_tol(0.5f64.next_up())).0;
    assert_eq!(above.rank, 1, "rel_tol just over the ratio stops there");
    assert_eq!(above.pivot_errors, vec![4.0, 2.0]);

    // The threshold is `rel_tol * max_accepted`, and complete pivoting can grow after its first
    // pivot, so the two rules do not agree: on this matrix the pivots are 1, then 2 (the growth of
    // the `[[1, 1], [1, -1]]` block), then 0.75. At `rel_tol = 0.5` the first-pivot threshold is
    // 0.5, which would accept 0.75; the running maximum is 2, whose threshold is 1.0, so the kernel
    // stops at two pivots and reports 0.75 as the residual.
    let growth = [
        1.0f64, 1.0, 0.0, // column 0
        1.0, -1.0, 0.0, // column 1
        0.0, 0.0, 0.75, // column 2
    ];
    let grown = square_selection(&growth, 3, &with_rel_tol(0.5)).0;
    assert_eq!(
        grown.rank, 2,
        "the third pivot is below the running maximum"
    );
    assert_eq!(grown.pivot_errors, vec![1.0, 2.0, 0.75]);
    assert_selection_bits_equal(
        &grown,
        &previous_square_selection(&growth, 3, &with_rel_tol(0.5)),
        "growth",
    );

    for rel_tol in [0.5, 0.5f64.next_down(), 0.5f64.next_up()] {
        assert_selection_bits_equal(
            &square_selection(&data, 2, &with_rel_tol(rel_tol)).0,
            &previous_square_selection(&data, 2, &with_rel_tol(rel_tol)),
            &format!("rel_tol={rel_tol}"),
        );
    }
}
