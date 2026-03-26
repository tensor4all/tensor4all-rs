use crate::{DenseFaerLuKernel, DenseMatrixSource, PivotKernel, PivotKernelOptions};
use approx::assert_abs_diff_eq;
use tensor4all_tcicore::{from_vec2d, rrlu, RrLUOptions};

#[test]
fn dense_kernel_recovers_identity_pivots() {
    let src =
        DenseMatrixSource::from_column_major(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let kernel = DenseFaerLuKernel;
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
    let kernel = DenseFaerLuKernel;
    let out = kernel
        .factorize(&src, &PivotKernelOptions::no_truncation())
        .unwrap();
    assert_eq!(out.rank, 0);
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

fn dense_factorize(rows: &[Vec<f64>], options: PivotKernelOptions) -> crate::PivotSelectionCore {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let data = col_major_from_rows(rows);
    let src = DenseMatrixSource::from_column_major(&data, nrows, ncols);
    DenseFaerLuKernel.factorize(&src, &options).unwrap()
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
fn dense_kernel_matches_legacy_pivot_errors_for_max_rank_stop() {
    let rows = vec![
        vec![0.433088, 0.956638, 0.0907974, 0.0447859, 0.0196053],
        vec![0.855517, 0.782503, 0.291197, 0.540828, 0.358579],
        vec![0.37455, 0.536457, 0.205479, 0.75896, 0.701206],
        vec![0.47272, 0.0172539, 0.518177, 0.242864, 0.461635],
        vec![0.0676373, 0.450878, 0.672335, 0.77726, 0.540691],
    ];
    let luci_options = PivotKernelOptions {
        max_rank: 2,
        ..PivotKernelOptions::default()
    };
    let legacy_options = RrLUOptions {
        max_rank: 2,
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
