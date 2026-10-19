use super::*;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tenferro::TensorScalar;

#[test]
fn src_error_estimate_matches_real_upper_triangular_oracle() {
    let r = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 0.0, 1.0, 3.0]);

    let estimate = src_error_estimate(&r).expect("well-conditioned R should be accepted");
    let column0_norm_sq: f64 = 0.25 + 1.0 / 36.0;
    let column1_norm_sq: f64 = 1.0 / 9.0;
    let expected_error = (0.5 * (1.0 / column0_norm_sq + 1.0 / column1_norm_sq)).sqrt();
    let expected_norm = (14.0_f64 / 2.0).sqrt();

    assert!((estimate.error - expected_error).abs() < 1.0e-12);
    assert!((estimate.norm - expected_norm).abs() < 1.0e-12);
}

#[test]
fn src_error_estimate_uses_conjugate_adjoint_for_complex_r() {
    let r01 = Complex64::new(1.0, 2.0);
    let r11 = Complex64::new(3.0, -1.0);
    let r = Matrix::from_col_major_vec(
        2,
        2,
        vec![Complex64::new(2.0, 0.0), Complex64::zero(), r01, r11],
    );

    let estimate = src_error_estimate(&r).expect("well-conditioned R should be accepted");
    let g00 = Complex64::new(0.5, 0.0);
    let g11 = Complex64::new(1.0, 0.0) / r11.conj();
    let g10 = -(r01.conj() * g00 * g11);
    let column0_norm_sq = g00.norm_sqr() + g10.norm_sqr();
    let column1_norm_sq = g11.norm_sqr();
    let expected_error = (0.5 * (1.0 / column0_norm_sq + 1.0 / column1_norm_sq)).sqrt();
    let expected_norm = (r01.norm_sqr() + 4.0 + r11.norm_sqr()).sqrt() / 2.0_f64.sqrt();

    assert!((estimate.error - expected_error).abs() < 1.0e-12);
    assert!((estimate.norm - expected_norm).abs() < 1.0e-12);
}

#[test]
fn src_error_estimate_rejects_singular_and_non_square_r() {
    let singular = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 0.0, 1.0, 0.0]);
    let singular_error = src_error_estimate(&singular).unwrap_err();
    assert!(singular_error.to_string().contains("SRC"));

    let non_square = Matrix::from_col_major_vec(2, 3, vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let shape_error = src_error_estimate(&non_square).unwrap_err();
    assert!(shape_error.to_string().contains("square"));

    let non_triangular = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 1.0, 0.0, 1.0]);
    let triangular_error = src_error_estimate(&non_triangular).unwrap_err();
    assert!(triangular_error.to_string().contains("upper-triangular"));
}

#[test]
fn src_error_estimate_general_accepts_column_restored_factors() {
    let triangular = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 0.0, 1.0, 3.0]);
    let restored = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 3.0, 2.0, 0.0]);
    let expected = src_error_estimate(&triangular).unwrap();
    let actual = src_error_estimate_general(&restored).unwrap();
    assert!((actual.error - expected.error).abs() < 1.0e-12);
    assert!((actual.norm - expected.norm).abs() < 1.0e-12);

    let triangular_complex = Matrix::from_col_major_vec(
        2,
        2,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::zero(),
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
        ],
    );
    let restored_complex = Matrix::from_col_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(2.0, 0.0),
            Complex64::zero(),
        ],
    );
    let expected = src_error_estimate(&triangular_complex).unwrap();
    let actual = src_error_estimate_general(&restored_complex).unwrap();
    assert!((actual.error - expected.error).abs() < 1.0e-12);
    assert!((actual.norm - expected.norm).abs() < 1.0e-12);
}

#[test]
fn src_error_estimate_general_rejects_invalid_factors() {
    let empty = Matrix::<f64>::zeros(0, 0);
    assert!(src_error_estimate_general(&empty)
        .unwrap_err()
        .to_string()
        .contains("non-empty"));
    let non_square = Matrix::from_col_major_vec(1, 2, vec![1.0_f64, 2.0]);
    assert!(src_error_estimate_general(&non_square)
        .unwrap_err()
        .to_string()
        .contains("square"));
    let singular = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 2.0, 2.0, 4.0]);
    assert!(src_error_estimate_general(&singular).is_err());
    let non_finite = Matrix::from_col_major_vec(1, 1, vec![f64::NAN]);
    assert!(src_error_estimate_general(&non_finite)
        .unwrap_err()
        .to_string()
        .contains("finite"));
}

#[test]
fn src_error_estimate_supports_single_precision_scalars() {
    let r32 = Matrix::from_col_major_vec(2, 2, vec![2.0_f32, 0.0, 1.0, 3.0]);
    let estimate32 = src_error_estimate(&r32).expect("f32 R should be accepted");
    assert!((estimate32.error - 6.3_f64.sqrt()).abs() < 1.0e-5);

    let rc32 = Matrix::from_col_major_vec(
        2,
        2,
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::zero(),
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, -1.0),
        ],
    );
    let estimatec32 = src_error_estimate(&rc32).expect("Complex32 R should be accepted");
    assert!(estimatec32.error.is_finite());
    assert!(estimatec32.norm.is_finite());
}

fn row_major_values<T>(tensor: &TypedTensor<T>) -> Vec<T>
where
    T: TensorScalar + Copy,
{
    assert_eq!(tensor.shape().len(), 2, "test helper expects a matrix");
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let values = tensor
        .as_slice()
        .expect("typed test tensor should expose host values");
    let mut out = Vec::with_capacity(values.len());
    for row in 0..rows {
        for col in 0..cols {
            out.push(values[row + col * rows]);
        }
    }
    out
}

fn matmul_row_major<T>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T>
where
    T: Copy + Zero + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let mut out = vec![T::zero(); m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    out
}

fn scale_columns_complex(
    data: &[Complex64],
    rows: usize,
    cols: usize,
    scales: &[f64],
) -> Vec<Complex64> {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(scales.len(), cols);
    let mut out = vec![Complex64::zero(); data.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = data[i * cols + j] * Complex64::new(scales[j], 0.0);
        }
    }
    out
}

#[test]
fn qr_backend_reconstructs_real_matrix() {
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])
        .expect("valid QR test input");

    let input_values = row_major_values(&input);
    let (q, r) = qr_backend(input).unwrap();
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    let q_values = row_major_values(&q);
    let r_values = row_major_values(&r);
    let reconstructed = matmul_row_major(&q_values, 2, 2, &r_values, 2);

    for (actual, expected) in reconstructed.iter().zip(input_values.iter()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "QR reconstruction mismatch: {actual} vs {expected}"
        );
    }
}

#[test]
fn svd_backend_reconstructs_complex_matrix() {
    let input = TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(-3.0, 0.25),
            Complex64::new(2.0, 1.5),
            Complex64::new(4.0, -2.0),
        ],
    )
    .expect("valid SVD test input");

    let decomp = svd_backend(&input).unwrap();
    assert_eq!(decomp.u().shape(), &[2, 2]);
    assert_eq!(decomp.s().shape(), &[2]);
    assert_eq!(decomp.vt().shape(), &[2, 2]);
    let cloned = decomp.clone();
    assert_eq!(cloned.s().shape(), &[2]);

    let u = row_major_values(decomp.u());
    let s = decomp
        .s()
        .as_slice()
        .expect("SVD singular values should expose host values")
        .to_vec();
    let vt = row_major_values(decomp.vt());
    let us = scale_columns_complex(&u, 2, 2, &s);
    let reconstructed = matmul_row_major(&us, 2, 2, &vt, 2);
    let input_values = row_major_values(&input);

    for (actual, expected) in reconstructed.iter().zip(input_values.iter()) {
        assert!(
            (*actual - *expected).norm() < 1.0e-10,
            "SVD reconstruction mismatch: {actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn solve_backend_solves_real_system() {
    let a = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 1.0, 2.0])
        .expect("valid solve lhs test input");
    let b = TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 0.0])
        .expect("valid solve rhs test input");

    let x = solve_backend(&a, &b).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    let x_values = x
        .as_slice()
        .expect("solve result should expose host values");
    assert!((x_values[0] - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((x_values[1] + 1.0 / 3.0).abs() < 1.0e-12);
}

#[test]
fn solve_matrix_solves_real_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 2.0]]);
    let b = crate::from_vec2d(vec![vec![1.0_f64], vec![0.0]]);

    let x = solve_matrix(&a, &b).unwrap();

    assert_eq!(x.nrows(), 2);
    assert_eq!(x.ncols(), 1);
    assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-12);
}

#[test]
fn solve_matrix_owned_solves_real_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 2.0]]);
    let b = crate::from_vec2d(vec![vec![1.0_f64], vec![0.0]]);

    let x = solve_matrix_owned(a, b).unwrap();

    assert_eq!(x.nrows(), 2);
    assert_eq!(x.ncols(), 1);
    assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-12);
}

#[test]
fn solve_matrix_owned_promotes_f32_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f32, 1.0], vec![1.0, 2.0]]);
    let b = crate::from_vec2d(vec![vec![1.0_f32], vec![0.0]]);

    let x = solve_matrix_owned(a, b).unwrap();

    assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-6);
    assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-6);
}

#[test]
fn solve_matrix_owned_solves_complex64_system() {
    let a = crate::from_vec2d(vec![
        vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    ]);
    let b = crate::from_vec2d(vec![
        vec![Complex64::new(1.0, 0.0)],
        vec![Complex64::new(0.0, 0.0)],
    ]);

    let x = solve_matrix_owned(a, b).unwrap();

    assert!((x[[0, 0]].re - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((x[[1, 0]].re + 1.0 / 3.0).abs() < 1.0e-12);
    assert!(x[[0, 0]].im.abs() < 1.0e-12);
    assert!(x[[1, 0]].im.abs() < 1.0e-12);
}

#[test]
fn solve_matrix_promotes_f32_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f32, 1.0], vec![1.0, 2.0]]);
    let b = crate::from_vec2d(vec![vec![1.0_f32], vec![0.0]]);

    let x = solve_matrix(&a, &b).unwrap();

    assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-6);
    assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-6);
}

#[test]
fn solve_matrix_promotes_complex32_system() {
    let a = crate::from_vec2d(vec![
        vec![Complex32::new(2.0, 0.0), Complex32::new(1.0, 0.0)],
        vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
    ]);
    let b = crate::from_vec2d(vec![
        vec![Complex32::new(1.0, 0.0)],
        vec![Complex32::new(0.0, 0.0)],
    ]);

    let x = solve_matrix(&a, &b).unwrap();

    assert!((x[[0, 0]].re - 2.0 / 3.0).abs() < 1.0e-6);
    assert!((x[[1, 0]].re + 1.0 / 3.0).abs() < 1.0e-6);
    assert!(x[[0, 0]].im.abs() < 1.0e-6);
    assert!(x[[1, 0]].im.abs() < 1.0e-6);
}

#[test]
fn triangular_solve_matrix_solves_left_lower_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 0.0], vec![1.0, 3.0]]);
    let b = crate::from_vec2d(vec![vec![2.0_f64], vec![7.0]]);

    let x = triangular_solve_matrix(&a, &b, true, true, false, false).unwrap();

    assert_eq!(x.nrows(), 2);
    assert_eq!(x.ncols(), 1);
    assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
    assert!((x[[1, 0]] - 2.0).abs() < 1.0e-12);
}

#[test]
fn triangular_solve_matrix_owned_solves_left_lower_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 0.0], vec![1.0, 3.0]]);
    let b = crate::from_vec2d(vec![vec![2.0_f64], vec![7.0]]);

    let x = triangular_solve_matrix_owned(a, b, true, true, false, false).unwrap();

    assert_eq!(x.nrows(), 2);
    assert_eq!(x.ncols(), 1);
    assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
    assert!((x[[1, 0]] - 2.0).abs() < 1.0e-12);
}

#[test]
fn triangular_solve_matrix_solves_right_upper_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 1.0], vec![0.0, 3.0]]);
    let b = crate::from_vec2d(vec![vec![2.0_f64, 7.0]]);

    let x = triangular_solve_matrix(&a, &b, false, false, false, false).unwrap();

    assert_eq!(x.nrows(), 1);
    assert_eq!(x.ncols(), 2);
    assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
    assert!((x[[0, 1]] - 2.0).abs() < 1.0e-12);
}

#[test]
fn triangular_solve_matrix_owned_promotes_f32_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f32, 0.0], vec![1.0, 3.0]]);
    let b = crate::from_vec2d(vec![vec![2.0_f32], vec![7.0]]);

    let x = triangular_solve_matrix_owned(a, b, true, true, false, false).unwrap();

    assert!((x[[0, 0]] - 1.0).abs() < 1.0e-6);
    assert!((x[[1, 0]] - 2.0).abs() < 1.0e-6);
}

#[test]
fn triangular_solve_matrix_promotes_complex32_system() {
    let a = crate::from_vec2d(vec![
        vec![Complex32::new(2.0, 0.0), Complex32::new(0.0, 0.0)],
        vec![Complex32::new(1.0, 0.0), Complex32::new(3.0, 0.0)],
    ]);
    let b = crate::from_vec2d(vec![
        vec![Complex32::new(2.0, 0.0)],
        vec![Complex32::new(7.0, 0.0)],
    ]);

    let x = triangular_solve_matrix(&a, &b, true, true, false, false).unwrap();

    assert!((x[[0, 0]].re - 1.0).abs() < 1.0e-6);
    assert!((x[[1, 0]].re - 2.0).abs() < 1.0e-6);
    assert!(x[[0, 0]].im.abs() < 1.0e-6);
    assert!(x[[1, 0]].im.abs() < 1.0e-6);
}

#[test]
fn triangular_solve_matrix_owned_solves_complex64_system() {
    let a = crate::from_vec2d(vec![
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
    ]);
    let b = crate::from_vec2d(vec![
        vec![Complex64::new(2.0, 0.0)],
        vec![Complex64::new(7.0, 0.0)],
    ]);

    let x = triangular_solve_matrix_owned(a, b, true, true, false, false).unwrap();

    assert!((x[[0, 0]].re - 1.0).abs() < 1.0e-12);
    assert!((x[[1, 0]].re - 2.0).abs() < 1.0e-12);
    assert!(x[[0, 0]].im.abs() < 1.0e-12);
    assert!(x[[1, 0]].im.abs() < 1.0e-12);
}

#[test]
fn triangular_solve_backend_solves_typed_tensor_system() {
    let a = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 3.0])
        .expect("valid triangular solve lhs test input");
    let b = TypedTensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 7.0])
        .expect("valid triangular solve rhs test input");

    let x = triangular_solve_backend(&a, &b, true, true, false, false).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    let x_values = x
        .as_slice()
        .expect("triangular solve result should expose host values");
    assert!((x_values[0] - 1.0).abs() < 1.0e-12);
    assert!((x_values[1] - 2.0).abs() < 1.0e-12);
}

#[test]
fn try_into_typed_result_reports_dtype_mismatch() {
    let tensor = Complex64::into_tensor(vec![1], vec![Complex64::new(1.0, 0.0)])
        .expect("valid dtype-mismatch test tensor");

    let err = try_into_typed_result::<f64>("test_op", tensor).unwrap_err();

    assert!(err.to_string().contains("test_op: dtype mismatch"));
}

#[test]
fn full_piv_lu_backend_returns_square_factors() {
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0])
        .expect("valid LU test input");

    let decomp = full_piv_lu_backend(&input).unwrap();

    assert_eq!(decomp.p().shape(), &[2, 2]);
    assert_eq!(decomp.l().shape(), &[2, 2]);
    assert_eq!(decomp.u().shape(), &[2, 2]);
    assert_eq!(decomp.q().shape(), &[2, 2]);
    let cloned = decomp.clone();
    assert_eq!(cloned.u().shape(), &[2, 2]);
}

#[test]
fn full_piv_lu_matrix_returns_square_factors() {
    let input = crate::from_vec2d(vec![vec![0.0_f64, 1.0], vec![2.0, 3.0]]);

    let decomp = full_piv_lu_matrix(&input).unwrap();

    assert_eq!(decomp.p.nrows(), 2);
    assert_eq!(decomp.p.ncols(), 2);
    assert_eq!(decomp.l.nrows(), 2);
    assert_eq!(decomp.l.ncols(), 2);
    assert_eq!(decomp.u.nrows(), 2);
    assert_eq!(decomp.u.ncols(), 2);
    assert_eq!(decomp.q.nrows(), 2);
    assert_eq!(decomp.q.ncols(), 2);
}

#[test]
fn full_piv_lu_matrix_owned_matches_borrowed_factors() {
    let input = crate::from_vec2d(vec![vec![0.0_f64, 1.0], vec![2.0, 3.0]]);
    let borrowed = full_piv_lu_matrix(&input).unwrap();
    let owned = full_piv_lu_matrix_owned(input).unwrap();

    assert_eq!(
        owned.p.as_col_major_slice(),
        borrowed.p.as_col_major_slice()
    );
    assert_eq!(
        owned.l.as_col_major_slice(),
        borrowed.l.as_col_major_slice()
    );
    assert_eq!(
        owned.u.as_col_major_slice(),
        borrowed.u.as_col_major_slice()
    );
    assert_eq!(
        owned.q.as_col_major_slice(),
        borrowed.q.as_col_major_slice()
    );
}

#[test]
fn full_piv_lu_matrix_owned_preserves_shape_error() {
    let input = crate::Matrix::from_col_major_vec(2, 3, vec![1.0_f64; 6]);
    let borrowed_error = full_piv_lu_matrix(&input).unwrap_err().to_string();
    let owned_error = full_piv_lu_matrix_owned(input).unwrap_err().to_string();

    assert_eq!(owned_error, borrowed_error);
    assert!(owned_error.contains("incompatible shapes"), "{owned_error}");
}

// --- A caller-owned execution context for the full-pivot LU solve ---

fn one_worker_context() -> crate::CpuExecutionContext {
    let backend = tenferro_cpu::CpuBackend::with_threads(1).expect("1T CPU backend");
    crate::CpuExecutionContext::from_backend(backend)
}

/// `T * P = Pi1` with a nontrivial 2x2 pivot and a *rectangular* left-hand
/// side, so the column-major transposition in the solve is exercised rather
/// than an identity or square fixture. `convert` maps the real fixture values
/// into the scalar type under test.
fn rectangular_operands<T>(mut convert: impl FnMut(f64) -> T) -> (Vec<T>, Vec<T>) {
    let pivot = [2.0, -1.0, 4.0, 3.0]
        .into_iter()
        .map(&mut convert)
        .collect();
    let pi1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .map(&mut convert)
        .collect();
    (pi1, pivot)
}

#[test]
fn the_context_solve_agrees_with_the_legacy_solve_for_f64_and_complex64() {
    let context = one_worker_context();

    let (pi1, pivot) = rectangular_operands::<f64>(|value| value);
    let legacy = f64::solve_right_full_piv_lu(&pi1, 3, 2, &pivot, 2, 2).expect("legacy solve");
    let scoped =
        f64::solve_right_full_piv_lu_in(&context, &pi1, 3, 2, &pivot, 2, 2).expect("context solve");
    assert_eq!(legacy, scoped);

    let pivot = vec![
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.0, 3.0),
        Complex64::new(4.0, -2.0),
        Complex64::new(3.0, 5.0),
    ];
    let pi1 = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(3.0, 0.5),
        Complex64::new(4.0, -0.5),
        Complex64::new(5.0, 1.5),
        Complex64::new(6.0, -2.5),
    ];
    let legacy =
        Complex64::solve_right_full_piv_lu(&pi1, 3, 2, &pivot, 2, 2).expect("legacy solve");
    let scoped = Complex64::solve_right_full_piv_lu_in(&context, &pi1, 3, 2, &pivot, 2, 2)
        .expect("context solve");
    assert_eq!(legacy, scoped);
}

#[test]
fn the_narrow_scalars_also_reach_a_nonzero_context_solve() {
    let context = one_worker_context();

    let (pi1, pivot) = rectangular_operands::<f32>(|value| value as f32);
    let legacy = f32::solve_right_full_piv_lu(&pi1, 3, 2, &pivot, 2, 2).expect("legacy solve");
    let scoped =
        f32::solve_right_full_piv_lu_in(&context, &pi1, 3, 2, &pivot, 2, 2).expect("context solve");
    assert_eq!(legacy, scoped);

    let pivot = vec![
        Complex32::new(2.0, 1.0),
        Complex32::new(-1.0, 3.0),
        Complex32::new(4.0, -2.0),
        Complex32::new(3.0, 5.0),
    ];
    let pi1 = vec![
        Complex32::new(1.0, 2.0),
        Complex32::new(2.0, -1.0),
        Complex32::new(3.0, 0.5),
        Complex32::new(4.0, -0.5),
        Complex32::new(5.0, 1.5),
        Complex32::new(6.0, -2.5),
    ];
    let legacy =
        Complex32::solve_right_full_piv_lu(&pi1, 3, 2, &pivot, 2, 2).expect("legacy solve");
    let scoped = Complex32::solve_right_full_piv_lu_in(&context, &pi1, 3, 2, &pivot, 2, 2)
        .expect("context solve");
    assert_eq!(legacy, scoped);
}

#[test]
fn the_context_solve_keeps_the_shape_validation_and_its_precedence() {
    let context = one_worker_context();
    let pivot = vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
    let pi1 = vec![1.0_f64, 2.0, 3.0, 4.0];

    let legacy = f64::solve_right_full_piv_lu(&pi1, 2, 2, &pivot, 2, 3)
        .expect_err("a non-square pivot must be refused");
    let scoped = f64::solve_right_full_piv_lu_in(&context, &pi1, 2, 2, &pivot, 2, 3)
        .expect_err("a non-square pivot must be refused");
    assert!(scoped.source.to_string().contains("square pivot matrix"));
    assert_eq!(legacy.source.to_string(), scoped.source.to_string());

    // An incompatible left-hand side is refused too.
    let pivot = vec![1.0_f64, 0.0, 0.0, 1.0];
    let scoped = f64::solve_right_full_piv_lu_in(&context, &pi1, 2, 3, &pivot, 2, 2)
        .expect_err("an incompatible left-hand side must be refused");
    assert!(scoped
        .source
        .to_string()
        .contains("cannot solve T * P = Pi1"));

    // With both shapes wrong, the square-pivot check keeps precedence.
    let scoped = f64::solve_right_full_piv_lu_in(&context, &pi1, 2, 3, &pivot, 2, 3)
        .expect_err("both shape errors must be refused");
    assert!(scoped.source.to_string().contains("square pivot matrix"));
}

#[test]
fn the_session_solve_agrees_with_the_legacy_solve_on_one_entered_session() {
    let context = one_worker_context();
    let (pi1, pivot) = rectangular_operands::<f64>(|value| value);
    let legacy = f64::solve_right_full_piv_lu(&pi1, 3, 2, &pivot, 2, 2).expect("legacy solve");

    // One session the caller entered serves every scalar kind; the method itself
    // never enters another one.
    let scoped = context
        .with_concrete_session(|session| {
            let f64_values = f64::solve_right_full_piv_lu_on_session(
                session.backend(),
                &pi1,
                3,
                2,
                &pivot,
                2,
                2,
            )?;
            let f32_inputs = rectangular_operands::<f32>(|value| value as f32);
            let f32_values = f32::solve_right_full_piv_lu_on_session(
                session.backend(),
                &f32_inputs.0,
                3,
                2,
                &f32_inputs.1,
                2,
                2,
            )?;
            Ok::<_, crate::BackendLinalgError>((f64_values, f32_values))
        })
        .expect("session entry")
        .expect("session solve");

    assert_eq!(legacy, scoped.0);
    let (f32_pi1, f32_pivot) = rectangular_operands::<f32>(|value| value as f32);
    let f32_legacy =
        f32::solve_right_full_piv_lu(&f32_pi1, 3, 2, &f32_pivot, 2, 2).expect("legacy solve");
    assert_eq!(f32_legacy, scoped.1);
}

/// The session-taking solve is the entry a scheduler-owned lane uses: it runs on the
/// lane's own session view and opens no other session.
#[test]
fn the_session_solve_runs_on_a_lane_session() {
    use std::sync::Mutex;

    // One worker keeps the fixture independent of the suite's parallel test threads,
    // which observe the process CPU mask that a wider session narrows.
    let backend = tenferro_cpu::CpuBackend::with_threads(1).expect("1T CPU backend");
    let (pi1, pivot) = rectangular_operands::<f64>(|value| value);
    let legacy = f64::solve_right_full_piv_lu(&pi1, 3, 2, &pivot, 2, 2).expect("legacy solve");
    let lane_solutions = Mutex::new(Vec::new());

    let mut held = crate::explicit::HeldSession::open(&backend).expect("held session");
    held.phase(|_index, lane| {
        let mut view = lane.session();
        let solution =
            f64::solve_right_full_piv_lu_on_session(view.backend(), &pi1, 3, 2, &pivot, 2, 2)
                .map_err(|error| {
                    tenferro_tensor::Error::runtime_state_source("lane solve", error)
                })?;
        lane_solutions
            .lock()
            .expect("lane solutions")
            .push(solution);
        Ok::<(), tenferro_tensor::Error>(())
    })
    .expect("phase");
    held.close().expect("affinity restores");

    let solutions = lane_solutions.into_inner().expect("lane solutions");
    assert!(!solutions.is_empty(), "every lane solved on its own view");
    for solution in solutions {
        assert_eq!(solution, legacy);
    }
}

#[test]
fn the_session_solve_keeps_the_shape_validation_and_its_precedence() {
    let context = one_worker_context();
    let pivot = vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0];
    let pi1 = vec![1.0_f64, 2.0, 3.0, 4.0];

    let scoped = context
        .with_concrete_session(|session| {
            f64::solve_right_full_piv_lu_on_session(session.backend(), &pi1, 2, 2, &pivot, 2, 3)
        })
        .expect("session entry")
        .expect_err("a non-square pivot must be refused");
    assert!(scoped.source.to_string().contains("square pivot matrix"));

    let scoped = context
        .with_concrete_session(|session| {
            f64::solve_right_full_piv_lu_on_session(
                session.backend(),
                &pi1,
                2,
                3,
                &[1.0, 0.0, 0.0, 1.0],
                2,
                3,
            )
        })
        .expect("session entry")
        .expect_err("both shape errors must be refused");
    assert!(scoped.source.to_string().contains("square pivot matrix"));
}

#[test]
fn a_context_solve_from_a_rayon_worker_is_refused_typed() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let (pi1, pivot) = rectangular_operands::<f64>(|value| value);
    let context = one_worker_context();

    let error = pool
        .install(|| f64::solve_right_full_piv_lu_in(&context, &pi1, 3, 2, &pivot, 2, 2))
        .expect_err("a worker entry must be refused");
    let rendered = format!("{error}");
    assert!(
        rendered.contains("explicit concrete session"),
        "the refusal must stay visible as an entry refusal: {rendered}"
    );
    // The refusal keeps its own type as the source instead of being flattened
    // into the message.
    assert!(
        error
            .source
            .downcast_ref::<crate::CpuExecutionContextError>()
            .is_some(),
        "the entry refusal must be the error's source: {rendered}"
    );
}
