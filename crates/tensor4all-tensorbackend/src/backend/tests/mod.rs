use super::*;
use num_complex::Complex64;
use num_traits::Zero;

fn row_major_values<T>(tensor: &TypedTensor<T>) -> Vec<T>
where
    T: Copy,
{
    assert_eq!(tensor.shape.len(), 2, "test helper expects a matrix");
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    let values = tensor.as_slice();
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
    let input = TypedTensor::from_vec(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);

    let (q, r) = qr_backend(&input).unwrap();
    assert_eq!(q.shape, vec![2, 2]);
    assert_eq!(r.shape, vec![2, 2]);

    let q_values = row_major_values(&q);
    let r_values = row_major_values(&r);
    let reconstructed = matmul_row_major(&q_values, 2, 2, &r_values, 2);
    let input_values = row_major_values(&input);

    for (actual, expected) in reconstructed.iter().zip(input_values.iter()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "QR reconstruction mismatch: {actual} vs {expected}"
        );
    }
}

#[test]
fn svd_backend_reconstructs_complex_matrix() {
    let input = TypedTensor::from_vec(
        vec![2, 2],
        vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(-3.0, 0.25),
            Complex64::new(2.0, 1.5),
            Complex64::new(4.0, -2.0),
        ],
    );

    let decomp = svd_backend(&input).unwrap();
    assert_eq!(decomp.u.shape, vec![2, 2]);
    assert_eq!(decomp.s.shape, vec![2]);
    assert_eq!(decomp.vt.shape, vec![2, 2]);

    let u = row_major_values(&decomp.u);
    let s = decomp.s.as_slice().to_vec();
    let vt = row_major_values(&decomp.vt);
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
