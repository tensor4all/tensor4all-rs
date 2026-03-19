
use super::*;
use num_complex::Complex64;
use num_traits::Zero;
use tenferro::LogicalMemorySpace;
use tenferro_algebra::Scalar as TfScalar;
use tenferro_tensor::MemoryOrder;

fn row_major_values<T>(tensor: &TypedTensor<T>) -> Vec<T>
where
    T: TfScalar + Copy,
{
    let row_major = tensor.contiguous(MemoryOrder::RowMajor);
    let row_major = if row_major.logical_memory_space() == LogicalMemorySpace::MainMemory {
        row_major
    } else {
        row_major
            .to_memory_space_async(LogicalMemorySpace::MainMemory)
            .expect("tensor should be movable to host memory")
    };
    let offset = usize::try_from(row_major.offset()).expect("offset should be non-negative");
    let len = row_major.len();
    let values: &[T] = row_major
        .buffer()
        .as_slice()
        .expect("tensor should expose contiguous host buffer");
    values[offset..offset + len].to_vec()
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
    let input =
        TypedTensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::RowMajor).unwrap();

    let (q, r) = qr_backend(&input).unwrap();
    assert_eq!(q.dims(), &[2, 2]);
    assert_eq!(r.dims(), &[2, 2]);

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
    let input = TypedTensor::from_slice(
        &[
            Complex64::new(1.0, -0.5),
            Complex64::new(2.0, 1.5),
            Complex64::new(-3.0, 0.25),
            Complex64::new(4.0, -2.0),
        ],
        &[2, 2],
        MemoryOrder::RowMajor,
    )
    .unwrap();

    let decomp = svd_backend(&input).unwrap();
    assert_eq!(decomp.u.dims(), &[2, 2]);
    assert_eq!(decomp.s.dims(), &[2]);
    assert_eq!(decomp.vt.dims(), &[2, 2]);

    let u = row_major_values(&decomp.u);
    let s = row_major_values(&decomp.s);
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
