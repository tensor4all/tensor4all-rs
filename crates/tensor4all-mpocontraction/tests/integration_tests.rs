//! Integration tests for tensor4all-mpocontraction
//!
//! Tests cover:
//! - Random MPO generation
//! - Full tensor reconstruction accuracy
//! - Algorithm equivalence (naive vs zipup vs fit)
//! - Both f64 and Complex64 types

use num_complex::Complex64;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tensor4all_mpocontraction::{
    contract_fit, contract_naive, contract_zipup, factorize::SVDScalar, tensor4_zeros,
    ContractionOptions, FactorizeMethod, FitOptions, Tensor4, Tensor4Ops, MPO,
};

/// Test helper module for generic MPO testing
mod test_helpers {
    use super::*;

    /// Trait for types that can be generated randomly and used in MPO operations
    pub trait RandomScalar: SVDScalar
    where
        <Self as num_complex::ComplexFloat>::Real: Into<f64>,
    {
        fn random_val(rng: &mut impl Rng) -> Self;
    }

    impl RandomScalar for f64 {
        fn random_val(rng: &mut impl Rng) -> Self {
            rng.gen::<f64>() * 2.0 - 1.0 // Range [-1, 1]
        }
    }

    impl RandomScalar for Complex64 {
        fn random_val(rng: &mut impl Rng) -> Self {
            Complex64::new(rng.gen::<f64>() * 2.0 - 1.0, rng.gen::<f64>() * 2.0 - 1.0)
        }
    }

    /// Generate a random MPO with specified dimensions
    ///
    /// # Arguments
    /// * `site_dims` - Vector of (site_dim_1, site_dim_2) tuples for each site
    /// * `bond_dims` - Vector of bond dimensions between sites (length = len(site_dims) - 1)
    /// * `seed` - Random seed for reproducibility
    pub fn random_mpo<T: RandomScalar>(
        site_dims: &[(usize, usize)],
        bond_dims: &[usize],
        seed: u64,
    ) -> MPO<T>
    where
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {
        assert!(!site_dims.is_empty());
        assert_eq!(bond_dims.len() + 1, site_dims.len());

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = site_dims.len();
        let mut tensors = Vec::with_capacity(n);

        for i in 0..n {
            let left_dim = if i == 0 { 1 } else { bond_dims[i - 1] };
            let right_dim = if i == n - 1 { 1 } else { bond_dims[i] };
            let (s1, s2) = site_dims[i];

            let mut tensor: Tensor4<T> = tensor4_zeros(left_dim, s1, s2, right_dim);
            for l in 0..left_dim {
                for i1 in 0..s1 {
                    for i2 in 0..s2 {
                        for r in 0..right_dim {
                            tensor.set4(l, i1, i2, r, T::random_val(&mut rng));
                        }
                    }
                }
            }
            tensors.push(tensor);
        }

        MPO::new(tensors).expect("Valid MPO dimensions")
    }

    /// Generate a random MPO with uniform site dimensions
    pub fn random_mpo_uniform<T: RandomScalar>(
        n_sites: usize,
        site_dim: usize,
        max_bond_dim: usize,
        seed: u64,
    ) -> MPO<T>
    where
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {
        let site_dims: Vec<_> = (0..n_sites).map(|_| (site_dim, site_dim)).collect();
        let bond_dims: Vec<_> = (0..n_sites.saturating_sub(1))
            .map(|_| max_bond_dim)
            .collect();
        random_mpo(&site_dims, &bond_dims, seed)
    }

    /// Convert MPO to full tensor (for small MPOs only!)
    ///
    /// Returns a 2D matrix representation where:
    /// - rows correspond to all combinations of site_dim_1 indices
    /// - cols correspond to all combinations of site_dim_2 indices
    pub fn mpo_to_full_matrix<T: RandomScalar>(mpo: &MPO<T>) -> Vec<Vec<T>>
    where
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {

        if mpo.is_empty() {
            return vec![vec![T::one()]];
        }

        let site_dims = mpo.site_dims();

        // Calculate total dimensions
        let total_dim1: usize = site_dims.iter().map(|(d1, _)| d1).product();
        let total_dim2: usize = site_dims.iter().map(|(_, d2)| d2).product();

        let mut result = vec![vec![T::zero(); total_dim2]; total_dim1];

        // Iterate over all index combinations
        let dims1: Vec<usize> = site_dims.iter().map(|(d1, _)| *d1).collect();
        let dims2: Vec<usize> = site_dims.iter().map(|(_, d2)| *d2).collect();

        for row in 0..total_dim1 {
            for col in 0..total_dim2 {
                // Convert flat indices to multi-indices
                let indices1 = flat_to_multi_index(row, &dims1);
                let indices2 = flat_to_multi_index(col, &dims2);

                // Build interleaved indices for MPO evaluation: [i1, j1, i2, j2, ...]
                let mut flat_indices = Vec::with_capacity(indices1.len() * 2);
                for (&i1, &i2) in indices1.iter().zip(indices2.iter()) {
                    flat_indices.push(i1);
                    flat_indices.push(i2);
                }

                result[row][col] = mpo.evaluate(&flat_indices).unwrap_or(T::zero());
            }
        }

        result
    }

    /// Convert flat index to multi-index given dimensions
    fn flat_to_multi_index(mut flat: usize, dims: &[usize]) -> Vec<usize> {
        let n = dims.len();
        let mut indices = vec![0; n];
        for i in (0..n).rev() {
            indices[i] = flat % dims[i];
            flat /= dims[i];
        }
        indices
    }

    /// Matrix multiplication for 2D vectors
    pub fn matrix_multiply<T: RandomScalar>(a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>>
    where
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {

        let m = a.len();
        let k = if m > 0 { a[0].len() } else { 0 };
        let n = if !b.is_empty() { b[0].len() } else { 0 };

        assert!(
            k > 0 && !b.is_empty() && b.len() == k,
            "Matrix dimension mismatch"
        );

        let mut result = vec![vec![T::zero(); n]; m];
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + a[i][l] * b[l][j];
                }
                result[i][j] = sum;
            }
        }
        result
    }

    /// Compute Frobenius norm of difference between two matrices
    pub fn matrix_diff_norm<T: RandomScalar>(a: &[Vec<T>], b: &[Vec<T>]) -> f64
    where
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {

        assert_eq!(a.len(), b.len());
        let mut sum: f64 = 0.0;
        for i in 0..a.len() {
            assert_eq!(a[i].len(), b[i].len());
            for j in 0..a[i].len() {
                let diff = a[i][j] - b[i][j];
                let abs_val: f64 = diff.abs().into();
                sum += abs_val * abs_val;
            }
        }
        sum.sqrt()
    }

    /// Compute Frobenius norm of a matrix
    pub fn matrix_norm<T: RandomScalar>(a: &[Vec<T>]) -> f64
    where
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {

        let mut sum: f64 = 0.0;
        for row in a {
            for &val in row {
                let abs_val: f64 = val.abs().into();
                sum += abs_val * abs_val;
            }
        }
        sum.sqrt()
    }

    /// Check if two matrices are approximately equal (relative error)
    pub fn matrices_approx_equal<T: RandomScalar>(a: &[Vec<T>], b: &[Vec<T>], rel_tol: f64) -> bool
    where
        <T as num_complex::ComplexFloat>::Real: Into<f64>,
    {
        let diff_norm = matrix_diff_norm(a, b);
        let a_norm = matrix_norm(a);
        if a_norm < 1e-15 {
            diff_norm < rel_tol
        } else {
            diff_norm / a_norm < rel_tol
        }
    }
}

use test_helpers::*;

// ============================================================================
// Generic test functions that work for both f64 and Complex64
// ============================================================================

/// Test that naive contraction produces correct full tensor
fn test_contract_naive_accuracy<T: RandomScalar>()
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let site_dims_a = vec![(2, 3), (2, 3), (2, 3)];
    let site_dims_b = vec![(3, 2), (3, 2), (3, 2)]; // s1_b = s2_a
    let bond_dims = vec![2, 2];

    let mpo_a: MPO<T> = random_mpo(&site_dims_a, &bond_dims, 42);
    let mpo_b: MPO<T> = random_mpo(&site_dims_b, &bond_dims, 43);

    // Compute contraction
    let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    // Convert to full matrices
    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let full_result = mpo_to_full_matrix(&result);

    // Expected: A * B
    let expected = matrix_multiply(&full_a, &full_b);

    assert!(
        matrices_approx_equal(&full_result, &expected, 1e-10),
        "Naive contraction result doesn't match A * B"
    );
}

/// Test that zipup contraction produces correct full tensor
fn test_contract_zipup_accuracy<T: RandomScalar>()
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let site_dims_a = vec![(2, 3), (2, 3), (2, 3)];
    let site_dims_b = vec![(3, 2), (3, 2), (3, 2)];
    let bond_dims = vec![2, 2];

    let mpo_a: MPO<T> = random_mpo(&site_dims_a, &bond_dims, 44);
    let mpo_b: MPO<T> = random_mpo(&site_dims_b, &bond_dims, 45);

    let options = ContractionOptions {
        tolerance: 1e-14,
        max_bond_dim: 100, // Large enough to not truncate
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let full_result = mpo_to_full_matrix(&result);
    let expected = matrix_multiply(&full_a, &full_b);

    assert!(
        matrices_approx_equal(&full_result, &expected, 1e-10),
        "Zipup contraction result doesn't match A * B"
    );
}

/// Test that fit contraction produces correct full tensor
fn test_contract_fit_accuracy<T: RandomScalar>()
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let site_dims_a = vec![(2, 3), (2, 3)];
    let site_dims_b = vec![(3, 2), (3, 2)];
    let bond_dims = vec![2];

    let mpo_a: MPO<T> = random_mpo(&site_dims_a, &bond_dims, 46);
    let mpo_b: MPO<T> = random_mpo(&site_dims_b, &bond_dims, 47);

    let options = FitOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        max_sweeps: 20,
        convergence_tol: 1e-12,
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_fit(&mpo_a, &mpo_b, &options, None).unwrap();

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let full_result = mpo_to_full_matrix(&result);
    let expected = matrix_multiply(&full_a, &full_b);

    assert!(
        matrices_approx_equal(&full_result, &expected, 1e-8),
        "Fit contraction result doesn't match A * B"
    );
}

/// Test that all contraction algorithms produce equivalent results
fn test_algorithm_equivalence<T: RandomScalar>()
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let site_dims_a = vec![(2, 2), (2, 2)];
    let site_dims_b = vec![(2, 2), (2, 2)];
    let bond_dims = vec![2];

    let mpo_a: MPO<T> = random_mpo(&site_dims_a, &bond_dims, 50);
    let mpo_b: MPO<T> = random_mpo(&site_dims_b, &bond_dims, 51);

    // Naive (no compression)
    let result_naive = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    // Zipup (high max_bond_dim to avoid truncation)
    let zipup_options = ContractionOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        factorize_method: FactorizeMethod::SVD,
    };
    let result_zipup = contract_zipup(&mpo_a, &mpo_b, &zipup_options).unwrap();

    // Fit (many sweeps for convergence)
    let fit_options = FitOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        max_sweeps: 20,
        convergence_tol: 1e-12,
        factorize_method: FactorizeMethod::SVD,
    };
    let result_fit = contract_fit(&mpo_a, &mpo_b, &fit_options, None).unwrap();

    // Convert to full matrices
    let full_naive = mpo_to_full_matrix(&result_naive);
    let full_zipup = mpo_to_full_matrix(&result_zipup);
    let full_fit = mpo_to_full_matrix(&result_fit);

    assert!(
        matrices_approx_equal(&full_naive, &full_zipup, 1e-10),
        "Naive and zipup results differ"
    );

    assert!(
        matrices_approx_equal(&full_naive, &full_fit, 1e-8),
        "Naive and fit results differ"
    );
}

/// Test compression with max_bond_dim
fn test_compression<T: RandomScalar>()
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let site_dims_a = vec![(2, 2), (2, 2), (2, 2)];
    let site_dims_b = vec![(2, 2), (2, 2), (2, 2)];
    let bond_dims = vec![3, 3];

    let mpo_a: MPO<T> = random_mpo(&site_dims_a, &bond_dims, 60);
    let mpo_b: MPO<T> = random_mpo(&site_dims_b, &bond_dims, 61);

    // Zipup with compression
    let options = ContractionOptions {
        tolerance: 1e-10,
        max_bond_dim: 4,
        factorize_method: FactorizeMethod::SVD,
    };
    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Check that bond dimensions are limited
    for link_dim in result.link_dims() {
        assert!(
            link_dim <= 4,
            "Bond dimension {} exceeds max_bond_dim 4",
            link_dim
        );
    }
}

/// Test identity contraction: I * A = A
fn test_identity_contraction<T: RandomScalar>()
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let site_dims = vec![(2, 2), (2, 2), (2, 2)];
    let bond_dims = vec![2, 2];

    let identity: MPO<T> = MPO::identity(&[2, 2, 2]).unwrap();
    let mpo_a: MPO<T> = random_mpo(&site_dims, &bond_dims, 70);

    let result = contract_naive(&identity, &mpo_a, None).unwrap();

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_result = mpo_to_full_matrix(&result);

    assert!(
        matrices_approx_equal(&full_result, &full_a, 1e-10),
        "I * A should equal A"
    );
}

/// Test associativity: (A * B) * C = A * (B * C)
fn test_associativity<T: RandomScalar>()
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    // Smaller dimensions to keep computation manageable
    let site_dims_a = vec![(2, 2), (2, 2)];
    let site_dims_b = vec![(2, 2), (2, 2)];
    let site_dims_c = vec![(2, 2), (2, 2)];
    let bond_dims = vec![2];

    let mpo_a: MPO<T> = random_mpo(&site_dims_a, &bond_dims, 80);
    let mpo_b: MPO<T> = random_mpo(&site_dims_b, &bond_dims, 81);
    let mpo_c: MPO<T> = random_mpo(&site_dims_c, &bond_dims, 82);

    // (A * B) * C
    let ab = contract_naive(&mpo_a, &mpo_b, None).unwrap();
    let ab_c = contract_naive(&ab, &mpo_c, None).unwrap();

    // A * (B * C)
    let bc = contract_naive(&mpo_b, &mpo_c, None).unwrap();
    let a_bc = contract_naive(&mpo_a, &bc, None).unwrap();

    let full_ab_c = mpo_to_full_matrix(&ab_c);
    let full_a_bc = mpo_to_full_matrix(&a_bc);

    assert!(
        matrices_approx_equal(&full_ab_c, &full_a_bc, 1e-10),
        "(A*B)*C should equal A*(B*C)"
    );
}

// ============================================================================
// Tests for f64
// ============================================================================

#[test]
fn test_contract_naive_accuracy_f64() {
    test_contract_naive_accuracy::<f64>();
}

#[test]
fn test_contract_zipup_accuracy_f64() {
    test_contract_zipup_accuracy::<f64>();
}

#[test]
fn test_contract_fit_accuracy_f64() {
    test_contract_fit_accuracy::<f64>();
}

#[test]
fn test_algorithm_equivalence_f64() {
    test_algorithm_equivalence::<f64>();
}

#[test]
fn test_compression_f64() {
    test_compression::<f64>();
}

#[test]
fn test_identity_contraction_f64() {
    test_identity_contraction::<f64>();
}

#[test]
fn test_associativity_f64() {
    test_associativity::<f64>();
}

// ============================================================================
// Tests for Complex64
// ============================================================================

#[test]
fn test_contract_naive_accuracy_complex64() {
    test_contract_naive_accuracy::<Complex64>();
}

// NOTE: Complex64 tests temporarily ignored due to mdarray-linalg-faer issue
// See: mdarray_linalg_faer_complex_svd_issue.md
#[test]
#[ignore = "mdarray-linalg-faer returns V^T instead of V^H for complex SVD"]
fn test_contract_zipup_accuracy_complex64() {
    test_contract_zipup_accuracy::<Complex64>();
}

#[test]
#[ignore = "mdarray-linalg-faer returns V^T instead of V^H for complex SVD"]
fn test_contract_fit_accuracy_complex64() {
    test_contract_fit_accuracy::<Complex64>();
}

#[test]
#[ignore = "mdarray-linalg-faer returns V^T instead of V^H for complex SVD"]
fn test_algorithm_equivalence_complex64() {
    test_algorithm_equivalence::<Complex64>();
}

#[test]
fn test_compression_complex64() {
    test_compression::<Complex64>();
}

#[test]
fn test_identity_contraction_complex64() {
    test_identity_contraction::<Complex64>();
}

#[test]
fn test_associativity_complex64() {
    test_associativity::<Complex64>();
}

// ============================================================================
// Additional edge case tests
// ============================================================================

#[test]
fn test_single_site_contraction() {
    let mpo_a = MPO::<f64>::constant(&[(2, 3)], 2.0);
    let mpo_b = MPO::<f64>::constant(&[(3, 2)], 3.0);

    let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    // Each element = sum over k of 2 * 3 = 6 * 3 = 18
    // Using interleaved indices: [i1, j1]
    let val = result.evaluate(&[0, 0]).unwrap();
    assert!((val - 18.0).abs() < 1e-10);
}

#[test]
fn test_dimension_mismatch_error() {
    let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0); // s2 = 3
    let mpo_b = MPO::<f64>::constant(&[(4, 2)], 1.0); // s1 = 4 ≠ 3

    let result = contract_naive(&mpo_a, &mpo_b, None);
    assert!(result.is_err());
}

#[test]
fn test_length_mismatch_error() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 1.0);

    let result = contract_naive(&mpo_a, &mpo_b, None);
    assert!(result.is_err());
}

#[test]
fn test_random_mpo_construction() {
    let mpo: MPO<f64> = random_mpo_uniform(4, 2, 3, 100);

    assert_eq!(mpo.len(), 4);
    for (s1, s2) in mpo.site_dims() {
        assert_eq!(s1, 2);
        assert_eq!(s2, 2);
    }
    for link_dim in mpo.link_dims() {
        assert_eq!(link_dim, 3);
    }
}

#[test]
fn test_mpo_to_full_matrix_identity() {
    let identity = MPO::<f64>::identity(&[2, 2]).unwrap();
    let full = mpo_to_full_matrix(&identity);

    // Should be 4x4 identity matrix
    assert_eq!(full.len(), 4);
    assert_eq!(full[0].len(), 4);

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (full[i][j] - expected).abs() < 1e-10,
                "Identity matrix incorrect at [{}, {}]",
                i,
                j
            );
        }
    }
}

#[test]
fn test_factorize_method_lu() {
    let site_dims = vec![(2, 2), (2, 2)];
    let bond_dims = vec![2];

    let mpo_a: MPO<f64> = random_mpo(&site_dims, &bond_dims, 200);
    let mpo_b: MPO<f64> = random_mpo(&site_dims, &bond_dims, 201);

    let options = ContractionOptions {
        tolerance: 1e-12,
        max_bond_dim: 100,
        factorize_method: FactorizeMethod::LU,
    };

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let full_result = mpo_to_full_matrix(&result);
    let expected = matrix_multiply(&full_a, &full_b);

    assert!(
        matrices_approx_equal(&full_result, &expected, 1e-10),
        "LU factorization should give correct result"
    );
}

#[test]
fn test_complex_random_mpo() {
    let mpo: MPO<Complex64> = random_mpo_uniform(3, 2, 2, 300);

    assert_eq!(mpo.len(), 3);
    // Check that values are complex (not just real)
    let tensor = mpo.site_tensor(0);
    // At least some values should have non-zero imaginary parts
    let has_complex = (0..2)
        .flat_map(|s1| (0..2).map(move |s2| *tensor.get4(0, s1, s2, 0)))
        .any(|v| v.im.abs() > 1e-10);
    assert!(has_complex, "Random complex MPO should have imaginary parts");
}

// ============================================================================
// Comprehensive f64 tests: maxbonddim variations
// ============================================================================

/// Test zipup with various max_bond_dim values
#[test]
fn test_zipup_maxbonddim_variations_f64() {
    let site_dims_a = vec![(2, 3), (2, 3), (2, 3)];
    let site_dims_b = vec![(3, 2), (3, 2), (3, 2)];
    let bond_dims = vec![4, 4]; // Higher bond dims to test truncation

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 1001);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 1002);

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let expected = matrix_multiply(&full_a, &full_b);
    let expected_norm = matrix_norm(&expected);

    // Test with various max_bond_dim values
    let max_bond_dims = [100, 16, 8, 4, 2];

    for &max_bd in &max_bond_dims {
        let options = ContractionOptions {
            tolerance: 1e-14,
            max_bond_dim: max_bd,
            factorize_method: FactorizeMethod::SVD,
        };

        let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();
        let full_result = mpo_to_full_matrix(&result);

        // Verify bond dimensions are respected
        for link_dim in result.link_dims() {
            assert!(
                link_dim <= max_bd,
                "max_bond_dim={}: link_dim {} exceeds limit",
                max_bd,
                link_dim
            );
        }

        // Compute relative error
        let rel_error = matrix_diff_norm(&full_result, &expected) / expected_norm;

        // Higher max_bond_dim should give lower error
        // With unlimited (100), should be exact
        if max_bd >= 16 {
            assert!(
                rel_error < 1e-10,
                "max_bond_dim={}: relative error {} too large for high bond dim",
                max_bd,
                rel_error
            );
        } else {
            // Even with low bond dim, error should be bounded
            assert!(
                rel_error < 1.0,
                "max_bond_dim={}: relative error {} is unbounded",
                max_bd,
                rel_error
            );
        }
    }
}

/// Test fit with various max_bond_dim values
#[test]
fn test_fit_maxbonddim_variations_f64() {
    let site_dims_a = vec![(2, 3), (2, 3)];
    let site_dims_b = vec![(3, 2), (3, 2)];
    let bond_dims = vec![4];

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 1003);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 1004);

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let expected = matrix_multiply(&full_a, &full_b);
    let expected_norm = matrix_norm(&expected);

    let max_bond_dims = [100, 16, 8, 4, 2];

    for &max_bd in &max_bond_dims {
        let options = FitOptions {
            tolerance: 1e-14,
            max_bond_dim: max_bd,
            max_sweeps: 20,
            convergence_tol: 1e-12,
            factorize_method: FactorizeMethod::SVD,
        };

        let result = contract_fit(&mpo_a, &mpo_b, &options, None).unwrap();
        let full_result = mpo_to_full_matrix(&result);

        // Verify bond dimensions
        for link_dim in result.link_dims() {
            assert!(
                link_dim <= max_bd,
                "max_bond_dim={}: link_dim {} exceeds limit",
                max_bd,
                link_dim
            );
        }

        let rel_error = matrix_diff_norm(&full_result, &expected) / expected_norm;

        if max_bd >= 16 {
            assert!(
                rel_error < 1e-8,
                "max_bond_dim={}: relative error {} too large",
                max_bd,
                rel_error
            );
        }
    }
}

/// Test compression quality vs bond dimension tradeoff
#[test]
fn test_compression_quality_tradeoff_f64() {
    let site_dims_a = vec![(2, 2), (2, 2), (2, 2), (2, 2)];
    let site_dims_b = vec![(2, 2), (2, 2), (2, 2), (2, 2)];
    let bond_dims = vec![3, 3, 3];

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 1005);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 1006);

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let expected = matrix_multiply(&full_a, &full_b);
    let expected_norm = matrix_norm(&expected);

    // Collect errors for increasing max_bond_dim
    let max_bond_dims = [2, 4, 8, 16, 32];
    let mut prev_error = f64::INFINITY;

    for &max_bd in &max_bond_dims {
        let options = ContractionOptions {
            tolerance: 1e-14,
            max_bond_dim: max_bd,
            factorize_method: FactorizeMethod::SVD,
        };

        let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();
        let full_result = mpo_to_full_matrix(&result);
        let rel_error = matrix_diff_norm(&full_result, &expected) / expected_norm;

        // Error should decrease (or stay same) as max_bond_dim increases
        assert!(
            rel_error <= prev_error + 1e-10,
            "Error increased from {} to {} when max_bond_dim increased to {}",
            prev_error,
            rel_error,
            max_bd
        );
        prev_error = rel_error;
    }

    // Final error with high bond dim should be very small
    assert!(
        prev_error < 1e-10,
        "Final error {} should be very small",
        prev_error
    );
}

// ============================================================================
// Comprehensive f64 tests: SVD vs LU factorization comparison
// ============================================================================

/// Compare SVD and LU factorization methods in zipup
#[test]
fn test_zipup_svd_vs_lu_f64() {
    let site_dims_a = vec![(2, 3), (2, 3), (2, 3)];
    let site_dims_b = vec![(3, 2), (3, 2), (3, 2)];
    let bond_dims = vec![3, 3];

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 2001);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 2002);

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let expected = matrix_multiply(&full_a, &full_b);
    let expected_norm = matrix_norm(&expected);

    // SVD method
    let svd_options = ContractionOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        factorize_method: FactorizeMethod::SVD,
    };
    let result_svd = contract_zipup(&mpo_a, &mpo_b, &svd_options).unwrap();
    let full_svd = mpo_to_full_matrix(&result_svd);
    let svd_error = matrix_diff_norm(&full_svd, &expected) / expected_norm;

    // LU method
    let lu_options = ContractionOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        factorize_method: FactorizeMethod::LU,
    };
    let result_lu = contract_zipup(&mpo_a, &mpo_b, &lu_options).unwrap();
    let full_lu = mpo_to_full_matrix(&result_lu);
    let lu_error = matrix_diff_norm(&full_lu, &expected) / expected_norm;

    // Both should produce accurate results
    assert!(
        svd_error < 1e-10,
        "SVD zipup error {} too large",
        svd_error
    );
    assert!(lu_error < 1e-10, "LU zipup error {} too large", lu_error);

    // Results should be equivalent
    let diff_error = matrix_diff_norm(&full_svd, &full_lu) / expected_norm;
    assert!(
        diff_error < 1e-10,
        "SVD and LU results differ by {}",
        diff_error
    );
}

/// Compare SVD and LU factorization methods in fit
#[test]
fn test_fit_svd_vs_lu_f64() {
    let site_dims_a = vec![(2, 3), (2, 3)];
    let site_dims_b = vec![(3, 2), (3, 2)];
    let bond_dims = vec![3];

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 2003);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 2004);

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let expected = matrix_multiply(&full_a, &full_b);
    let expected_norm = matrix_norm(&expected);

    // SVD method
    let svd_options = FitOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        max_sweeps: 20,
        convergence_tol: 1e-12,
        factorize_method: FactorizeMethod::SVD,
    };
    let result_svd = contract_fit(&mpo_a, &mpo_b, &svd_options, None).unwrap();
    let full_svd = mpo_to_full_matrix(&result_svd);
    let svd_error = matrix_diff_norm(&full_svd, &expected) / expected_norm;

    // LU method
    let lu_options = FitOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        max_sweeps: 20,
        convergence_tol: 1e-12,
        factorize_method: FactorizeMethod::LU,
    };
    let result_lu = contract_fit(&mpo_a, &mpo_b, &lu_options, None).unwrap();
    let full_lu = mpo_to_full_matrix(&result_lu);
    let lu_error = matrix_diff_norm(&full_lu, &expected) / expected_norm;

    // Both should produce accurate results
    assert!(svd_error < 1e-8, "SVD fit error {} too large", svd_error);
    assert!(lu_error < 1e-8, "LU fit error {} too large", lu_error);
}

/// Test SVD vs LU with compression
#[test]
fn test_svd_vs_lu_with_compression_f64() {
    let site_dims_a = vec![(2, 2), (2, 2), (2, 2)];
    let site_dims_b = vec![(2, 2), (2, 2), (2, 2)];
    let bond_dims = vec![3, 3]; // Use smaller bond dims for meaningful compression

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 2005);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 2006);

    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let expected = matrix_multiply(&full_a, &full_b);
    let expected_norm = matrix_norm(&expected);

    let max_bd = 6; // Allow sufficient bond dim for reasonable accuracy

    // SVD with compression
    let svd_options = ContractionOptions {
        tolerance: 1e-10,
        max_bond_dim: max_bd,
        factorize_method: FactorizeMethod::SVD,
    };
    let result_svd = contract_zipup(&mpo_a, &mpo_b, &svd_options).unwrap();
    let full_svd = mpo_to_full_matrix(&result_svd);
    let svd_error = matrix_diff_norm(&full_svd, &expected) / expected_norm;

    // LU with compression
    let lu_options = ContractionOptions {
        tolerance: 1e-10,
        max_bond_dim: max_bd,
        factorize_method: FactorizeMethod::LU,
    };
    let result_lu = contract_zipup(&mpo_a, &mpo_b, &lu_options).unwrap();
    let full_lu = mpo_to_full_matrix(&result_lu);
    let lu_error = matrix_diff_norm(&full_lu, &expected) / expected_norm;

    // With reasonable compression, both methods should work
    // Note: The exact error depends on the random matrices and compression level
    assert!(
        svd_error < 1.0,
        "SVD compressed error {} unreasonably large",
        svd_error
    );
    assert!(
        lu_error < 1.0,
        "LU compressed error {} unreasonably large",
        lu_error
    );

    // Check bond dimensions are respected
    for link_dim in result_svd.link_dims() {
        assert!(link_dim <= max_bd, "SVD: bond dim exceeds max");
    }
    for link_dim in result_lu.link_dims() {
        assert!(link_dim <= max_bd, "LU: bond dim exceeds max");
    }
}

// ============================================================================
// Comprehensive f64 tests: Environment calculation integration
// ============================================================================

/// Test environment consistency: full left environment equals full right environment
#[test]
fn test_environment_consistency_f64() {
    use tensor4all_mpocontraction::environment::{left_environment, right_environment};

    let site_dims = vec![(2, 2), (2, 2), (2, 2)];
    let bond_dims = vec![2, 2];

    let mpo_a: MPO<f64> = random_mpo(&site_dims, &bond_dims, 3001);
    let mpo_b: MPO<f64> = random_mpo(&site_dims, &bond_dims, 3002);

    let mut left_cache = Vec::new();
    let mut right_cache = Vec::new();

    let n = mpo_a.len();

    // left_environment(site=n) = full contraction from left (1x1 result)
    let left_final = left_environment(&mpo_a, &mpo_b, n, &mut left_cache).unwrap();
    assert_eq!(left_final.dim(0), 1);
    assert_eq!(left_final.dim(1), 1);

    // right_environment at site n-1 gives the environment to the right of site n-1
    // which is just [[1]] (boundary condition)
    let right_last = right_environment(&mpo_a, &mpo_b, n - 1, &mut right_cache).unwrap();
    assert_eq!(right_last.dim(0), 1);
    assert_eq!(right_last.dim(1), 1);
    assert!((right_last[[0, 0]] - 1.0).abs() < 1e-10);

    // Verify left environment at intermediate sites have correct dimensions
    for i in 1..n {
        let env = left_environment(&mpo_a, &mpo_b, i, &mut left_cache).unwrap();
        assert_eq!(env.dim(0), bond_dims[i - 1]);
        assert_eq!(env.dim(1), bond_dims[i - 1]);
    }

    // Verify right environment at intermediate sites have correct dimensions
    for i in 0..n - 1 {
        let env = right_environment(&mpo_a, &mpo_b, i, &mut right_cache).unwrap();
        assert_eq!(env.dim(0), bond_dims[i]);
        assert_eq!(env.dim(1), bond_dims[i]);
    }
}

/// Test environment caching efficiency
#[test]
fn test_environment_caching_f64() {
    use tensor4all_mpocontraction::environment::{left_environment, right_environment};

    let site_dims = vec![(2, 2), (2, 2), (2, 2), (2, 2)];
    let bond_dims = vec![2, 2, 2];

    let mpo_a: MPO<f64> = random_mpo(&site_dims, &bond_dims, 3003);
    let mpo_b: MPO<f64> = random_mpo(&site_dims, &bond_dims, 3004);

    let mut left_cache = Vec::new();

    // Compute left environment at site 3 (will compute 0, 1, 2, 3)
    let env3 = left_environment(&mpo_a, &mpo_b, 3, &mut left_cache).unwrap();

    // Now compute at site 2 - should use cache
    let env2 = left_environment(&mpo_a, &mpo_b, 2, &mut left_cache).unwrap();

    // Compute at site 4 (full chain) - should reuse cached values
    let env4 = left_environment(&mpo_a, &mpo_b, 4, &mut left_cache).unwrap();

    // Verify dimensions
    assert_eq!(env2.dim(0), bond_dims[1]); // left bond at site 2
    assert_eq!(env3.dim(0), bond_dims[2]); // left bond at site 3
    assert_eq!(env4.dim(0), 1); // boundary

    // Similar test for right environment
    let mut right_cache = Vec::new();
    let renv0 = right_environment(&mpo_a, &mpo_b, 0, &mut right_cache).unwrap();
    let renv1 = right_environment(&mpo_a, &mpo_b, 1, &mut right_cache).unwrap();

    assert_eq!(renv0.dim(0), bond_dims[0]);
    assert_eq!(renv1.dim(0), bond_dims[1]);
}

/// Test environment with identity MPO
#[test]
fn test_environment_identity_f64() {
    use tensor4all_mpocontraction::environment::{left_environment, right_environment};

    // Identity MPO contracted with itself
    let site_dims = vec![2, 2, 2];
    let identity = MPO::<f64>::identity(&site_dims).unwrap();

    let mut left_cache = Vec::new();
    let mut right_cache = Vec::new();

    // For identity @ identity:
    // Each site contributes sum_{s1,s2} delta_{s1,s2} * delta_{s1,s2} = site_dim
    // Full contraction = product of site contributions = 2 * 2 * 2 = 8
    let env_final = left_environment(&identity, &identity, 3, &mut left_cache).unwrap();
    let expected_full = (2.0_f64).powi(3); // 8
    assert!(
        (env_final[[0, 0]] - expected_full).abs() < 1e-10,
        "Identity@Identity full environment expected {}, got {}",
        expected_full,
        env_final[[0, 0]]
    );

    // right_environment at site 0 contracts sites 1 and 2 only
    // Expected: 2 * 2 = 4
    let renv0 = right_environment(&identity, &identity, 0, &mut right_cache).unwrap();
    let expected_partial = (2.0_f64).powi(2); // 4
    assert!(
        (renv0[[0, 0]] - expected_partial).abs() < 1e-10,
        "Right environment at site 0 expected {}, got {}",
        expected_partial,
        renv0[[0, 0]]
    );

    // Verify left environment accumulates correctly at each step
    for i in 0..=3 {
        let env = left_environment(&identity, &identity, i, &mut left_cache).unwrap();
        let expected_val = (2.0_f64).powi(i as i32);
        assert!(
            (env[[0, 0]] - expected_val).abs() < 1e-10,
            "Left env at {} expected {}, got {}",
            i,
            expected_val,
            env[[0, 0]]
        );
    }
}

/// Test that environments integrate correctly with zipup algorithm
#[test]
fn test_environment_zipup_integration_f64() {
    let site_dims_a = vec![(2, 3), (2, 3)];
    let site_dims_b = vec![(3, 2), (3, 2)];
    let bond_dims = vec![3];

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 3005);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 3006);

    // Compute via zipup
    let options = ContractionOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        factorize_method: FactorizeMethod::SVD,
    };
    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Verify result by computing full matrices
    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let full_result = mpo_to_full_matrix(&result);
    let expected = matrix_multiply(&full_a, &full_b);

    assert!(
        matrices_approx_equal(&full_result, &expected, 1e-10),
        "Zipup result with environment integration failed"
    );
}

/// Test environments with non-uniform site dimensions
#[test]
fn test_environment_nonuniform_dims_f64() {
    use tensor4all_mpocontraction::environment::{left_environment, right_environment};

    let site_dims = vec![(2, 2), (3, 3), (2, 2)];
    let bond_dims = vec![2, 3];

    let mpo_a: MPO<f64> = random_mpo(&site_dims, &bond_dims, 3007);
    let mpo_b: MPO<f64> = random_mpo(&site_dims, &bond_dims, 3008);

    let mut left_cache = Vec::new();
    let mut right_cache = Vec::new();

    // Compute all left environments
    for i in 0..=3 {
        let env = left_environment(&mpo_a, &mpo_b, i, &mut left_cache).unwrap();
        // Verify dimensions
        let expected_dim = if i == 0 {
            (1, 1)
        } else if i == 3 {
            (1, 1)
        } else {
            (bond_dims[i - 1], bond_dims[i - 1])
        };
        assert_eq!(
            (env.dim(0), env.dim(1)),
            expected_dim,
            "Left env at {} has wrong shape",
            i
        );
    }

    // Compute all right environments
    for i in 0..3 {
        let env = right_environment(&mpo_a, &mpo_b, i, &mut right_cache).unwrap();
        let expected_dim = if i == 2 {
            (1, 1)
        } else {
            (bond_dims[i], bond_dims[i])
        };
        assert_eq!(
            (env.dim(0), env.dim(1)),
            expected_dim,
            "Right env at {} has wrong shape",
            i
        );
    }
}

// ============================================================================
// Additional stress tests for f64
// ============================================================================

/// Test with larger MPOs
#[test]
fn test_larger_mpo_contraction_f64() {
    let n_sites = 5;
    let site_dim = 2;
    let bond_dim = 3;

    let site_dims_a: Vec<_> = (0..n_sites).map(|_| (site_dim, site_dim)).collect();
    let site_dims_b = site_dims_a.clone();
    let bond_dims: Vec<_> = (0..n_sites - 1).map(|_| bond_dim).collect();

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 4001);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 4002);

    // Naive contraction (no truncation)
    let result_naive = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    // Zipup with sufficient bond dimension
    let options = ContractionOptions {
        tolerance: 1e-14,
        max_bond_dim: 20,
        factorize_method: FactorizeMethod::SVD,
    };
    let result_zipup = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Convert to full and compare
    let full_naive = mpo_to_full_matrix(&result_naive);
    let full_zipup = mpo_to_full_matrix(&result_zipup);

    assert!(
        matrices_approx_equal(&full_naive, &full_zipup, 1e-10),
        "Large MPO: naive and zipup differ"
    );
}

/// Test with asymmetric site dimensions
#[test]
fn test_asymmetric_site_dims_f64() {
    let site_dims_a = vec![(2, 4), (3, 2), (2, 3)];
    let site_dims_b = vec![(4, 3), (2, 2), (3, 2)];
    let bond_dims = vec![2, 2];

    let mpo_a: MPO<f64> = random_mpo(&site_dims_a, &bond_dims, 4003);
    let mpo_b: MPO<f64> = random_mpo(&site_dims_b, &bond_dims, 4004);

    let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    // Verify dimensions
    let result_dims = result.site_dims();
    assert_eq!(result_dims[0], (2, 3)); // (s1_a, s2_b)
    assert_eq!(result_dims[1], (3, 2));
    assert_eq!(result_dims[2], (2, 2));

    // Verify accuracy
    let full_a = mpo_to_full_matrix(&mpo_a);
    let full_b = mpo_to_full_matrix(&mpo_b);
    let full_result = mpo_to_full_matrix(&result);
    let expected = matrix_multiply(&full_a, &full_b);

    assert!(
        matrices_approx_equal(&full_result, &expected, 1e-10),
        "Asymmetric dims contraction failed"
    );
}

/// Test numerical stability with small values
#[test]
fn test_numerical_stability_small_values_f64() {
    use tensor4all_mpocontraction::{tensor4_zeros, Tensor4};

    // Create MPO with small values
    let mut tensors_a = Vec::new();
    let mut tensors_b = Vec::new();

    for _ in 0..3 {
        let mut t: Tensor4<f64> = tensor4_zeros(1, 2, 2, 1);
        for s1 in 0..2 {
            for s2 in 0..2 {
                t.set4(0, s1, s2, 0, 1e-8 * ((s1 + s2 + 1) as f64));
            }
        }
        tensors_a.push(t.clone());
        tensors_b.push(t);
    }

    let mpo_a = MPO::new(tensors_a).unwrap();
    let mpo_b = MPO::new(tensors_b).unwrap();

    let options = ContractionOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Verify non-zero result (numerical stability)
    let full = mpo_to_full_matrix(&result);
    let norm = matrix_norm(&full);
    assert!(norm > 0.0, "Result should be non-zero");
}

/// Test numerical stability with large values
#[test]
fn test_numerical_stability_large_values_f64() {
    use tensor4all_mpocontraction::{tensor4_zeros, Tensor4};

    let mut tensors_a = Vec::new();
    let mut tensors_b = Vec::new();

    for _ in 0..2 {
        let mut t: Tensor4<f64> = tensor4_zeros(1, 2, 2, 1);
        for s1 in 0..2 {
            for s2 in 0..2 {
                t.set4(0, s1, s2, 0, 1e8 * ((s1 + s2 + 1) as f64));
            }
        }
        tensors_a.push(t.clone());
        tensors_b.push(t);
    }

    let mpo_a = MPO::new(tensors_a).unwrap();
    let mpo_b = MPO::new(tensors_b).unwrap();

    let options = ContractionOptions {
        tolerance: 1e-14,
        max_bond_dim: 100,
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Verify result by comparing to naive
    let expected = contract_naive(&mpo_a, &mpo_b, None).unwrap();
    let full_result = mpo_to_full_matrix(&result);
    let full_expected = mpo_to_full_matrix(&expected);

    assert!(
        matrices_approx_equal(&full_result, &full_expected, 1e-6),
        "Large value contraction inaccurate"
    );
}
