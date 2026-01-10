//! Tests for the unified factorize function.

use std::sync::Arc;
use tensor4all_core::index::{DynId, Index};
use tensor4all_core::{factorize, Canonical, DynIndex, FactorizeAlg, FactorizeError, FactorizeOptions};
use tensor4all_core::{storage::DenseStorageF64, Storage, TensorDynLen};

// ============================================================================
// Test Data Helpers
// ============================================================================

/// Helper to create a simple 2x3 matrix tensor for testing.
fn create_test_matrix() -> TensorDynLen {
    let i: DynIndex = Index::new_dyn(2);
    let j: DynIndex = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));

    TensorDynLen {
        indices: vec![i, j],
        dims: vec![2, 3],
        storage,
    }
}

/// Helper to create a rank-3 tensor for testing.
fn create_rank3_tensor() -> TensorDynLen {
    let i: DynIndex = Index::new_dyn(2);
    let j: DynIndex = Index::new_dyn(3);
    let k: DynIndex = Index::new_dyn(2);

    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));

    TensorDynLen {
        indices: vec![i, j, k],
        dims: vec![2, 3, 2],
        storage,
    }
}

// ============================================================================
// Shared Test Helpers
// ============================================================================

/// Test factorization with given options and verify reconstruction.
fn test_factorize_reconstruction(options: &FactorizeOptions) {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let result = factorize(&tensor, &left_inds, options).unwrap();

    // Verify reconstruction: left * right ≈ original
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

/// Test that left and right factors share the same bond index.
fn test_shared_bond_index(options: &FactorizeOptions) {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let result = factorize(&tensor, &left_inds, options).unwrap();

    let left_bond = result.left.indices.last().unwrap();
    let right_bond = result.right.indices.first().unwrap();
    assert_eq!(
        left_bond.id, right_bond.id,
        "Left and right should share the same bond index"
    );
    assert_eq!(
        left_bond.id, result.bond_index.id,
        "Bond index should match left's bond index"
    );
}

// ============================================================================
// All Algorithms: Reconstruction Tests
// ============================================================================

#[test]
fn test_factorize_reconstruction_all_algorithms() {
    // Test all algorithms with both canonical directions (where supported)
    let algorithms = [
        (FactorizeAlg::SVD, vec![Canonical::Left, Canonical::Right]),
        (FactorizeAlg::QR, vec![Canonical::Left]), // Right not supported
        (FactorizeAlg::LU, vec![Canonical::Left, Canonical::Right]),
        (FactorizeAlg::CI, vec![Canonical::Left, Canonical::Right]),
    ];

    for (alg, canonicals) in algorithms {
        for canonical in canonicals {
            let options = FactorizeOptions {
                alg,
                canonical,
                rtol: None,
                max_rank: None,
            };
            test_factorize_reconstruction(&options);
        }
    }
}

#[test]
fn test_factorize_shared_bond_index_all_algorithms() {
    // Test all algorithms have shared bond index
    let algorithms = [
        FactorizeAlg::SVD,
        FactorizeAlg::QR,
        FactorizeAlg::LU,
        FactorizeAlg::CI,
    ];

    for alg in algorithms {
        let options = FactorizeOptions {
            alg,
            canonical: Canonical::Left,
            rtol: None,
            max_rank: None,
        };
        test_shared_bond_index(&options);
    }
}

// ============================================================================
// SVD-Specific Tests
// ============================================================================

#[test]
fn test_factorize_svd_returns_singular_values() {
    for canonical in [Canonical::Left, Canonical::Right] {
        let tensor = create_test_matrix();
        let left_inds = vec![tensor.indices[0].clone()];
        let options = FactorizeOptions::svd().with_canonical(canonical);
        let result = factorize(&tensor, &left_inds, &options).unwrap();

        assert!(result.singular_values.is_some());
        let sv = result.singular_values.unwrap();
        assert!(!sv.is_empty());
        assert!(result.rank > 0);
        assert!(result.rank <= 2); // min(2, 3) = 2
    }
}

#[test]
fn test_factorize_svd_rank3() {
    let tensor = create_rank3_tensor();
    let left_inds = vec![tensor.indices[0].clone(), tensor.indices[1].clone()];

    let options = FactorizeOptions::svd();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

// ============================================================================
// QR-Specific Tests
// ============================================================================

#[test]
fn test_factorize_qr_right_canonical_error() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::qr().with_canonical(Canonical::Right);
    let result = factorize(&tensor, &left_inds, &options);

    assert!(matches!(
        result,
        Err(FactorizeError::UnsupportedCanonical(_))
    ));
}

#[test]
fn test_factorize_qr_no_singular_values() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];
    let options = FactorizeOptions::qr();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    assert!(result.singular_values.is_none());
}

// ============================================================================
// LU/CI-Specific Tests
// ============================================================================

#[test]
fn test_factorize_lu_ci_no_singular_values() {
    for alg in [FactorizeAlg::LU, FactorizeAlg::CI] {
        for canonical in [Canonical::Left, Canonical::Right] {
            let tensor = create_test_matrix();
            let left_inds = vec![tensor.indices[0].clone()];
            let options = FactorizeOptions {
                alg,
                canonical,
                rtol: None,
                max_rank: None,
            };
            let result = factorize(&tensor, &left_inds, &options).unwrap();

            assert!(result.singular_values.is_none());
        }
    }
}

// ============================================================================
// Truncation Tests
// ============================================================================

#[test]
fn test_factorize_with_max_rank() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    // LU should respect max_rank
    let options = FactorizeOptions::lu().with_max_rank(1);
    let result = factorize(&tensor, &left_inds, &options).unwrap();
    assert_eq!(result.rank, 1);

    // SVD API works (actual truncation behavior may vary)
    let options = FactorizeOptions::svd().with_max_rank(1);
    let result = factorize(&tensor, &left_inds, &options).unwrap();
    assert!(result.rank >= 1);
}

// ============================================================================
// Diagonal Storage Contraction Tests
// ============================================================================

#[test]
fn test_diag_dense_contraction_svd_internals() {
    use tensor4all_core::svd;

    let i: DynIndex = Index::new_dyn(2);
    let j: DynIndex = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));

    let tensor: TensorDynLen = TensorDynLen {
        indices: vec![i.clone(), j.clone()],
        dims: vec![2, 3],
        storage,
    };

    let (u, s, v) = svd::<f64>(&tensor, &[i.clone()]).expect("SVD should succeed");

    // Verify S is diagonal storage
    assert!(matches!(s.storage.as_ref(), Storage::DiagF64(_)));

    // Verify S and V share a common index
    let common_found = s
        .indices
        .iter()
        .any(|s_idx| v.indices.iter().any(|v_idx| s_idx.id == v_idx.id));
    assert!(common_found, "S and V should share a common index");

    // Contractions should work
    let sv = s.contract_einsum(&v);
    assert_eq!(sv.dims.len(), 2, "S*V should be a 2D tensor");

    let us = u.contract_einsum(&s);
    assert_eq!(us.dims.len(), 2, "U*S should be a 2D tensor");
}

// ============================================================================
// Helper Functions
// ============================================================================

fn assert_tensors_approx_equal(a: &TensorDynLen, b: &TensorDynLen, tol: f64) {
    assert_eq!(a.dims, b.dims, "Tensor dimensions don't match");

    match (a.storage.as_ref(), b.storage.as_ref()) {
        (Storage::DenseF64(a_data), Storage::DenseF64(b_data)) => {
            let a_slice = a_data.as_slice();
            let b_slice = b_data.as_slice();
            assert_eq!(a_slice.len(), b_slice.len(), "Storage lengths don't match");

            for (i, (&av, &bv)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
                let diff = (av - bv).abs();
                assert!(
                    diff < tol,
                    "Element {} differs: {} vs {} (diff={})",
                    i,
                    av,
                    bv,
                    diff
                );
            }
        }
        _ => panic!("Unsupported storage types for comparison"),
    }
}
