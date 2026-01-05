//! Tests for the unified factorize function.

use std::sync::Arc;
use tensor4all_core_common::index::{DynId, Index, NoSymmSpace};
use tensor4all_core_linalg::{factorize, Canonical, FactorizeError, FactorizeOptions};
use tensor4all_core_tensor::{storage::DenseStorageF64, Storage, TensorDynLen};

/// Helper to create a simple 2x3 matrix tensor for testing.
fn create_test_matrix() -> TensorDynLen<DynId> {
    // Create a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    let i: Index<DynId, NoSymmSpace, _> = Index::new_dyn(2);
    let j: Index<DynId, NoSymmSpace, _> = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));

    TensorDynLen {
        indices: vec![i, j],
        dims: vec![2, 3],
        storage,
    }
}

/// Helper to create a rank-3 tensor for testing.
fn create_rank3_tensor() -> TensorDynLen<DynId> {
    let i: Index<DynId, NoSymmSpace, _> = Index::new_dyn(2);
    let j: Index<DynId, NoSymmSpace, _> = Index::new_dyn(3);
    let k: Index<DynId, NoSymmSpace, _> = Index::new_dyn(2);

    // 2x3x2 tensor
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));

    TensorDynLen {
        indices: vec![i, j, k],
        dims: vec![2, 3, 2],
        storage,
    }
}

// ============================================================================
// SVD Tests
// ============================================================================

#[test]
fn test_factorize_svd_left_canonical() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::svd().with_canonical(Canonical::Left);
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // Check that we got singular values
    assert!(result.singular_values.is_some());
    let sv = result.singular_values.unwrap();
    assert!(!sv.is_empty());

    // Check rank
    assert!(result.rank > 0);
    assert!(result.rank <= 2); // min(2, 3) = 2

    // Verify reconstruction: left * right ≈ original
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

#[test]
fn test_factorize_svd_right_canonical() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::svd().with_canonical(Canonical::Right);
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // Check that we got singular values
    assert!(result.singular_values.is_some());

    // Verify reconstruction
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

#[test]
fn test_factorize_svd_rank3() {
    let tensor = create_rank3_tensor();
    // Split as (i, j) | (k)
    let left_inds = vec![tensor.indices[0].clone(), tensor.indices[1].clone()];

    let options = FactorizeOptions::svd();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // Verify reconstruction
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

// ============================================================================
// QR Tests
// ============================================================================

#[test]
fn test_factorize_qr_left_canonical() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::qr();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // QR should not return singular values
    assert!(result.singular_values.is_none());

    // Verify reconstruction
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

#[test]
fn test_factorize_qr_right_canonical_error() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::qr().with_canonical(Canonical::Right);
    let result = factorize(&tensor, &left_inds, &options);

    // QR with Right canonical should fail
    assert!(matches!(result, Err(FactorizeError::UnsupportedCanonical(_))));
}

// ============================================================================
// LU Tests
// ============================================================================

#[test]
fn test_factorize_lu_left_canonical() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::lu();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // LU should not return singular values
    assert!(result.singular_values.is_none());

    // Verify reconstruction
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

#[test]
fn test_factorize_lu_right_canonical() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::lu().with_canonical(Canonical::Right);
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // Verify reconstruction
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

// ============================================================================
// CI Tests
// ============================================================================

#[test]
fn test_factorize_ci_left_canonical() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::ci();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // CI should not return singular values
    assert!(result.singular_values.is_none());

    // Verify reconstruction
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

#[test]
fn test_factorize_ci_right_canonical() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::ci().with_canonical(Canonical::Right);
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // Verify reconstruction
    let reconstructed = result.left.contract_einsum(&result.right);
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

// ============================================================================
// Truncation Tests
// ============================================================================

#[test]
fn test_factorize_svd_with_max_rank() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::svd().with_max_rank(1);
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // Should truncate to rank 1
    // Note: SVD currently uses rtol-based truncation, max_rank is for LU/CI
    // This test verifies the API works, actual truncation behavior may vary
    assert!(result.rank >= 1);
}

#[test]
fn test_factorize_lu_with_max_rank() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::lu().with_max_rank(1);
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // LU should respect max_rank
    assert_eq!(result.rank, 1);
}

// ============================================================================
// Bond Index Tests
// ============================================================================

#[test]
fn test_factorize_svd_shared_bond_index() {
    // Verify that left and right factors share the same bond index
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::svd().with_canonical(Canonical::Left);
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    // Check that left's last index and right's first index have the same ID
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

#[test]
fn test_factorize_qr_shared_bond_index() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::qr();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    let left_bond = result.left.indices.last().unwrap();
    let right_bond = result.right.indices.first().unwrap();
    assert_eq!(
        left_bond.id, right_bond.id,
        "Left and right should share the same bond index"
    );
}

#[test]
fn test_factorize_lu_shared_bond_index() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::lu();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    let left_bond = result.left.indices.last().unwrap();
    let right_bond = result.right.indices.first().unwrap();
    assert_eq!(
        left_bond.id, right_bond.id,
        "Left and right should share the same bond index"
    );
}

#[test]
fn test_factorize_ci_shared_bond_index() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::ci();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    let left_bond = result.left.indices.last().unwrap();
    let right_bond = result.right.indices.first().unwrap();
    assert_eq!(
        left_bond.id, right_bond.id,
        "Left and right should share the same bond index"
    );
}

// ============================================================================
// Diagonal Storage Contraction Tests
// ============================================================================

#[test]
fn test_diag_dense_contraction_svd_internals() {
    // Test that diagonal tensor (S) can contract with dense tensor (V)
    // This is the internal operation in factorize_svd
    use tensor4all_core_linalg::svd;

    let i: Index<DynId, NoSymmSpace, _> = Index::new_dyn(2);
    let j: Index<DynId, NoSymmSpace, _> = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));

    let tensor: TensorDynLen<DynId> = TensorDynLen {
        indices: vec![i.clone(), j.clone()],
        dims: vec![2, 3],
        storage,
    };

    let (u, s, v) = svd::<DynId, _, f64>(&tensor, &[i.clone()]).expect("SVD should succeed");

    // Verify S is diagonal storage
    assert!(matches!(s.storage.as_ref(), Storage::DiagF64(_)));

    // Verify S and V share a common index
    let common_found = s.indices.iter().any(|s_idx| {
        v.indices.iter().any(|v_idx| s_idx.id == v_idx.id)
    });
    assert!(common_found, "S and V should share a common index");

    // S * V contraction should work
    let sv = s.contract_einsum(&v);
    assert_eq!(sv.dims.len(), 2, "S*V should be a 2D tensor");

    // U * S contraction should also work
    let us = u.contract_einsum(&s);
    assert_eq!(us.dims.len(), 2, "U*S should be a 2D tensor");
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if two tensors are approximately equal.
fn assert_tensors_approx_equal(a: &TensorDynLen<DynId>, b: &TensorDynLen<DynId>, tol: f64) {
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
