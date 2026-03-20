
use super::*;

#[test]
fn test_factorize_svd() {
    let mut matrix: Matrix2<f64> = matrix2_zeros(4, 3);
    for i in 0..4 {
        for j in 0..3 {
            matrix[[i, j]] = (i * 3 + j + 1) as f64;
        }
    }

    let options = FactorizeOptions {
        method: FactorizeMethod::SVD,
        tolerance: 1e-12,
        max_rank: 10,
        left_orthogonal: true,
        ..Default::default()
    };

    let result = factorize(&matrix, &options).unwrap();
    assert!(result.rank >= 1);
    assert!(result.rank <= 3); // Max rank is min(4, 3) = 3

    // Verify reconstruction: L @ R ≈ original
    let m = 4;
    let n = 3;
    for i in 0..m {
        for j in 0..n {
            let mut reconstructed = 0.0;
            for k in 0..result.rank {
                reconstructed += result.left[[i, k]] * result.right[[k, j]];
            }
            let original = matrix[[i, j]];
            assert!(
                (reconstructed - original).abs() < 1e-10,
                "Reconstruction failed at [{}, {}]: {} vs {}",
                i,
                j,
                reconstructed,
                original
            );
        }
    }
}

#[test]
fn test_factorize_lu() {
    let mut matrix: Matrix2<f64> = matrix2_zeros(4, 3);
    for i in 0..4 {
        for j in 0..3 {
            matrix[[i, j]] = (i * 3 + j) as f64;
        }
    }

    let options = FactorizeOptions {
        method: FactorizeMethod::LU,
        tolerance: 1e-12,
        max_rank: 10,
        left_orthogonal: true,
        ..Default::default()
    };

    let result = factorize_lu(&matrix, &options).unwrap();
    assert!(result.rank >= 1);
    assert!(result.rank <= 3); // Max rank is min(4, 3) = 3
}

#[test]
fn test_factorize_with_truncation() {
    // Create a rank-2 matrix
    let mut matrix: Matrix2<f64> = matrix2_zeros(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            // Rank-2: outer product of [1,2,3,4] and [1,1,1,1] + [1,1,1,1] and [1,2,3,4]
            matrix[[i, j]] = (i + 1) as f64 + (j + 1) as f64;
        }
    }

    let options = FactorizeOptions {
        method: FactorizeMethod::SVD,
        tolerance: 1e-10,
        max_rank: 2,
        left_orthogonal: true,
        ..Default::default()
    };

    let result = factorize(&matrix, &options).unwrap();
    assert!(result.rank <= 2);
}

#[test]
fn test_factorize_svd_complex64() {
    use num_complex::Complex64;

    let mut matrix: Matrix2<Complex64> = matrix2_zeros(4, 3);
    for i in 0..4 {
        for j in 0..3 {
            // Create complex values with both real and imaginary parts
            let re = (i * 3 + j + 1) as f64;
            let im = ((i + j) % 3) as f64 * 0.5;
            matrix[[i, j]] = Complex64::new(re, im);
        }
    }

    let options = FactorizeOptions {
        method: FactorizeMethod::SVD,
        tolerance: 1e-12,
        max_rank: 10,
        left_orthogonal: true,
        ..Default::default()
    };

    let result = factorize(&matrix, &options).unwrap();
    assert!(result.rank >= 1);
    assert!(result.rank <= 3); // Max rank is min(4, 3) = 3

    // Verify reconstruction: L @ R ≈ original
    let m = 4;
    let n = 3;
    let mut max_error: f64 = 0.0;
    for i in 0..m {
        for j in 0..n {
            let mut reconstructed = Complex64::new(0.0, 0.0);
            for k in 0..result.rank {
                reconstructed += result.left[[i, k]] * result.right[[k, j]];
            }
            let original = matrix[[i, j]];
            let error = (reconstructed - original).norm();
            max_error = max_error.max(error);
        }
    }
    assert!(
        max_error < 1e-10,
        "Reconstruction error too large: {}",
        max_error
    );
}
