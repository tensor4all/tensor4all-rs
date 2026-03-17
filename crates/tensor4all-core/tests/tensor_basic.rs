use num_complex::{Complex32, Complex64};
use std::sync::Arc;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::storage::{DenseStorageC64, DenseStorageF64, DiagStorageC64};
use tensor4all_core::{AnyScalar, DenseStorageFactory, Storage, TensorDynLen, TensorElement};

/// Helper to create DenseF64 storage with shape information
fn make_dense_f64(data: Vec<f64>, dims: &[usize]) -> Storage {
    Storage::DenseF64(DenseStorageF64::from_vec_with_shape(data, dims))
}

/// Helper to create DenseC64 storage with shape information
fn make_dense_c64(data: Vec<Complex64>, dims: &[usize]) -> Storage {
    Storage::DenseC64(DenseStorageC64::from_vec_with_shape(data, dims))
}

#[test]
fn test_storage_dense_f64() {
    // Create a zero-initialized tensor with 10 elements
    let storage = Storage::new_dense_f64(10);
    assert_eq!(storage.len(), 10);
    assert!(storage.is_dense());
    assert!(storage.is_f64());
    assert_eq!(storage.sum_f64(), 0.0);
    assert_eq!(
        storage.to_dense_f64_col_major_vec(&[10]).unwrap(),
        vec![0.0; 10]
    );
}

#[test]
fn test_storage_dense_c64() {
    // Create a zero-initialized tensor with 10 elements
    let storage = Storage::new_dense_c64(10);
    assert_eq!(storage.len(), 10);
    assert!(storage.is_dense());
    assert!(storage.is_c64());
    assert_eq!(storage.sum_c64(), Complex64::new(0.0, 0.0));
    assert_eq!(
        storage.to_dense_c64_col_major_vec(&[10]).unwrap(),
        vec![Complex64::new(0.0, 0.0); 10]
    );
}

#[test]
fn test_storage_factory_f64() {
    let storage = <f64 as DenseStorageFactory>::new_dense(7);
    assert!(storage.is_dense());
    assert!(storage.is_f64());
    assert_eq!(storage.len(), 7);
    assert_eq!(
        storage.to_dense_f64_col_major_vec(&[7]).unwrap(),
        vec![0.0; 7]
    );
}

#[test]
fn test_storage_factory_c64() {
    let storage = <Complex64 as DenseStorageFactory>::new_dense(9);
    assert!(storage.is_dense());
    assert!(storage.is_c64());
    assert_eq!(storage.len(), 9);
    assert_eq!(
        storage.to_dense_c64_col_major_vec(&[9]).unwrap(),
        vec![Complex64::new(0.0, 0.0); 9]
    );
}

#[test]
fn test_cow_storage() {
    // Test that Arc-based storage allows COW semantics
    let storage1 = Arc::new(make_dense_f64(vec![1.0, 2.0], &[2]));
    let storage2 = Arc::clone(&storage1);

    // Initially, both point to the same storage
    assert!(Arc::ptr_eq(&storage1, &storage2));

    // Check values are correct
    match storage1.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v.get(0), 1.0);
            assert_eq!(v.get(1), 2.0);
        }
        other => panic!("expected DenseF64, got {other:?}"),
    }
}

#[test]
fn test_tensor_dyn_len_creation() {
    let indices = vec![Index::new_dyn(2), Index::new_dyn(3)];
    let storage = Arc::new(<f64 as DenseStorageFactory>::new_dense_with_shape(&[2, 3]));

    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);
    assert_eq!(tensor.indices.len(), 2);
    let dims = tensor.dims();
    assert_eq!(dims.len(), 2);
    assert_eq!(dims[0], 2);
    assert_eq!(dims[1], 3);
}

#[test]
fn test_tensor_shared_storage() {
    let indices = vec![Index::new_dyn(2)];
    let storage = Arc::new(make_dense_f64(vec![1.0, 2.0], &[2]));

    let tensor1 = TensorDynLen::new(indices.clone(), Arc::clone(&storage));
    let tensor2 = TensorDynLen::new(indices, storage);

    // Canonical payload is native; storage snapshots need not share allocation.
    assert_eq!(tensor1.to_vec_f64().unwrap(), tensor2.to_vec_f64().unwrap());

    let cloned = tensor1.clone();
    assert_eq!(tensor1.to_vec_f64().unwrap(), cloned.to_vec_f64().unwrap());

    // Check that both tensors have the expected storage
    match tensor1.storage().as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v.dims(), vec![2]);
        }
        other => panic!("expected DenseF64, got {other:?}"),
    }
}

#[test]
fn test_tensor_sum_f64_no_match() {
    let indices = vec![Index::new_dyn(3)];
    let storage = Arc::new(make_dense_f64(vec![1.0, 2.0, 3.0], &[3]));

    let t: TensorDynLen = TensorDynLen::new(indices, storage);
    let sum_f64 = t.sum_f64();
    assert_eq!(sum_f64, 6.0);

    let sum_any: AnyScalar = t.sum();
    assert!(!sum_any.is_complex());
    assert!((sum_any.real() - 6.0).abs() < 1e-10);
}

#[test]
fn test_tensor_sum_c64() {
    let indices = vec![Index::new_dyn(2)];
    let storage = Arc::new(make_dense_c64(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        &[2],
    ));

    // Now always returns AnyScalar
    let t: TensorDynLen = TensorDynLen::new(indices, storage);
    let sum_any: AnyScalar = t.sum();
    assert!(sum_any.is_complex());
    let z: Complex64 = sum_any.into();
    assert!((z.re - 4.0).abs() < 1e-10);
    assert!((z.im - 1.0).abs() < 1e-10);
}

#[test]
#[should_panic(expected = "Tensor indices must all be unique")]
fn test_tensor_duplicate_indices_new() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone(), i.clone()]; // duplicate i
    let storage = Arc::new(<f64 as DenseStorageFactory>::new_dense_with_shape(&[
        2, 3, 2,
    ]));

    let _tensor: TensorDynLen = TensorDynLen::new(indices, storage);
}

#[test]
#[should_panic(expected = "Tensor indices must all be unique")]
fn test_tensor_duplicate_indices_from_indices() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone(), i.clone()]; // duplicate i
    let storage = Arc::new(<f64 as DenseStorageFactory>::new_dense_with_shape(&[
        2, 3, 2,
    ]));

    let _tensor: TensorDynLen = TensorDynLen::from_indices(indices, storage);
}

// ============================================================================
// Index Replacement Tests
// ============================================================================

#[test]
fn test_replaceind_basic() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let new_i = Index::new_dyn(2); // Same dimension, different ID

    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data, &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Replace index i with new_i
    let replaced = tensor.replaceind(&i, &new_i);

    // Check that the first index was replaced
    assert_eq!(replaced.indices[0].id, new_i.id);
    // Check that the second index was not affected
    assert_eq!(replaced.indices[1].id, j.id);
    // Check that dimensions are unchanged
    assert_eq!(replaced.dims(), vec![2, 3]);
    assert_eq!(tensor.to_vec_f64().unwrap(), replaced.to_vec_f64().unwrap());
}

#[test]
fn test_replaceind_no_match() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4); // Not in tensor
    let new_k = Index::new_dyn(4);

    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data, &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Replace index k (not in tensor) - should return unchanged tensor
    let replaced = tensor.replaceind(&k, &new_k);

    // Check that indices are unchanged
    assert_eq!(replaced.indices[0].id, i.id);
    assert_eq!(replaced.indices[1].id, j.id);
}

#[test]
fn test_replaceinds_basic() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let new_i = Index::new_dyn(2);
    let new_j = Index::new_dyn(3);
    let new_k = Index::new_dyn(4);

    let indices = vec![i.clone(), j.clone(), k.clone()];
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let storage = Arc::new(make_dense_f64(data, &[2, 3, 4]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Replace all indices
    let replaced = tensor.replaceinds(
        &[i.clone(), j.clone(), k.clone()],
        &[new_i.clone(), new_j.clone(), new_k.clone()],
    );

    // Check that all indices were replaced
    assert_eq!(replaced.indices[0].id, new_i.id);
    assert_eq!(replaced.indices[1].id, new_j.id);
    assert_eq!(replaced.indices[2].id, new_k.id);
    // Check that dimensions are unchanged
    assert_eq!(replaced.dims(), vec![2, 3, 4]);
}

#[test]
fn test_replaceinds_partial() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let new_i = Index::new_dyn(2);
    // Only replace i, not j or k

    let indices = vec![i.clone(), j.clone(), k.clone()];
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let storage = Arc::new(make_dense_f64(data, &[2, 3, 4]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Replace only i
    let replaced = tensor.replaceinds(std::slice::from_ref(&i), std::slice::from_ref(&new_i));

    // Check that i was replaced
    assert_eq!(replaced.indices[0].id, new_i.id);
    // Check that j and k are unchanged
    assert_eq!(replaced.indices[1].id, j.id);
    assert_eq!(replaced.indices[2].id, k.id);
}

#[test]
#[should_panic(expected = "old_indices and new_indices must have the same length")]
fn test_replaceinds_length_mismatch() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let new_i = Index::new_dyn(2);
    let new_j = Index::new_dyn(3);

    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data, &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Should panic - length mismatch
    let _replaced = tensor.replaceinds(std::slice::from_ref(&i), &[new_i, new_j]);
}

#[test]
#[should_panic(expected = "Index space mismatch")]
fn test_replaceind_dimension_mismatch() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let wrong_size = Index::new_dyn(5); // Different dimension

    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data, &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Should panic - dimension mismatch
    let _replaced = tensor.replaceind(&i, &wrong_size);
}

#[test]
#[should_panic(expected = "Index space mismatch")]
fn test_replaceinds_dimension_mismatch() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let new_i = Index::new_dyn(2);
    let wrong_size = Index::new_dyn(5); // Different dimension

    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data, &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Should panic - dimension mismatch
    let _replaced = tensor.replaceinds(&[i.clone(), j.clone()], &[new_i, wrong_size]);
}

#[test]
fn test_replaceinds_does_not_reorder_data() {
    // Test that replaceinds changes index IDs but does NOT reorder storage data.
    // This is the key difference from permuteinds.
    //
    // Create a 2×3 tensor with raw low-level buffer [1, 2, 3, 4, 5, 6].
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data.clone(), &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Create new indices with same dimensions but different IDs
    let new_i = Index::new_dyn(2);
    let new_j = Index::new_dyn(3);

    // Use replaceinds to change index IDs (but keep same order)
    let replaced = tensor.replaceinds(&[i.clone(), j.clone()], &[new_i.clone(), new_j.clone()]);

    // Check indices were replaced
    assert_eq!(replaced.indices[0].id, new_i.id);
    assert_eq!(replaced.indices[1].id, new_j.id);
    assert_eq!(replaced.dims(), vec![2, 3]);

    // CRITICAL: replaceinds does NOT reorder data
    // The storage data should remain unchanged
    match replaced.storage().as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.as_slice(), &data, "replaceinds should not reorder data");
        }
        _ => panic!("expected DenseF64"),
    }
}

#[test]
fn test_replaceinds_with_different_order_does_not_reorder_data() {
    // Test that replaceinds with indices in different order still does NOT reorder data.
    // This demonstrates the bug: if you need to reorder indices, use permuteinds instead.
    //
    // Create a 2×3 tensor backed by the raw low-level buffer [1, 2, 3, 4, 5, 6].
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data.clone(), &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Create new indices with same dimensions
    let new_i = Index::new_dyn(2);
    let new_j = Index::new_dyn(3);

    // Use replaceinds with indices in different order
    // This changes the index order in the result, but does NOT reorder the data
    // NOTE: replaceinds requires matching dimensions, so we can't directly swap i and j
    // This test demonstrates that replaceinds doesn't reorder data even when used correctly
    // For actual index reordering, use permuteinds instead
    let replaced = tensor.replaceinds(&[i.clone(), j.clone()], &[new_i.clone(), new_j.clone()]);

    // Check indices were replaced (order unchanged, just IDs changed)
    assert_eq!(replaced.indices[0].id, new_i.id);
    assert_eq!(replaced.indices[1].id, new_j.id);
    // Dimensions remain the same: [2, 3]
    assert_eq!(replaced.dims(), vec![2, 3]);

    // CRITICAL: replaceinds does NOT reorder data
    // The storage data remains unchanged
    match replaced.storage().as_ref() {
        Storage::DenseF64(v) => {
            // Data is unchanged
            assert_eq!(v.as_slice(), &data, "replaceinds should not reorder data");
        }
        _ => panic!("expected DenseF64"),
    }
}

#[test]
fn test_permuteinds_reorders_data() {
    // Test that permuteinds changes index order AND reorders storage data.
    // This is the correct way to reorder indices.
    //
    // Create a 2×3 tensor backed by the raw low-level buffer [1, 2, 3, 4, 5, 6].
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data, &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Use permuteinds to swap indices: [j, i] instead of [i, j]
    // This should reorder both indices AND data
    let permuted = tensor.permute_indices(&[j.clone(), i.clone()]);

    // Check indices were permuted
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    assert_eq!(permuted.dims(), vec![3, 2]);

    // CRITICAL: permuteinds DOES reorder data
    // The data should be reordered to match the new index order
    // Original logical values: [[1, 2, 3], [4, 5, 6]] with shape [2, 3]
    // Permuted logical values: [[1, 4], [2, 5], [3, 6]] with shape [3, 2]
    // Raw low-level buffer: [1, 4, 2, 5, 3, 6]
    match permuted.storage().as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(
                v.as_slice(),
                &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
                "permuteinds should reorder data to match new index order"
            );
        }
        _ => panic!("expected DenseF64"),
    }
}

#[test]
fn test_replaceinds_vs_permuteinds_comparison() {
    // Direct comparison: replaceinds vs permuteinds when index order changes.
    // This test demonstrates why permuteinds should be used when reordering indices.
    //
    // Create a 2×3 tensor: [[1, 2, 3], [4, 5, 6]]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(make_dense_f64(data, &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(indices, storage);

    // Create new indices with same dimensions
    let new_i = Index::new_dyn(2);
    let new_j = Index::new_dyn(3);

    // Method 1: replaceinds (only changes IDs, not order)
    // This changes index IDs but NOT data order
    // Note: replaceinds requires matching dimensions, so we can't swap i and j directly
    // This test demonstrates that replaceinds doesn't reorder data
    // In practice, you should use permuteinds when you need to change index order
    let replaced = tensor.replaceinds(&[i.clone(), j.clone()], &[new_i.clone(), new_j.clone()]);
    assert_eq!(replaced.dims(), vec![2, 3]);
    let replaced_data = match replaced.storage().as_ref() {
        Storage::DenseF64(v) => v.as_slice().to_vec(),
        _ => panic!("expected DenseF64"),
    };

    // Method 2: permuteinds (CORRECT when order changes)
    // This changes both index order AND data order
    let permuted = tensor.permute_indices(&[j.clone(), i.clone()]);
    assert_eq!(permuted.dims(), vec![3, 2]);
    let permuted_data = match permuted.storage().as_ref() {
        Storage::DenseF64(v) => v.as_slice().to_vec(),
        _ => panic!("expected DenseF64"),
    };

    // The data should be DIFFERENT because permuteinds reorders data
    // replaced: [1, 2, 3, 4, 5, 6] (unchanged, same shape [2, 3])
    // permuted: [1, 4, 2, 5, 3, 6] (reordered, shape [3, 2])
    assert_ne!(
        replaced_data, permuted_data,
        "replaceinds and permuteinds should produce different data: replaceinds doesn't reorder, permuteinds does"
    );
    assert_eq!(
        permuted_data,
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        "permuteinds should correctly reorder data"
    );
}

// ============================================================================
// Complex Conjugation Tests
// ============================================================================

#[test]
fn test_storage_conj_f64() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let storage = make_dense_f64(data.clone(), &[4]);
    let conj_storage = storage.conj();

    // For real numbers, conj is identity
    match conj_storage {
        Storage::DenseF64(v) => {
            assert_eq!(v.as_slice(), &data);
        }
        _ => panic!("Expected DenseF64"),
    }
}

#[test]
fn test_storage_conj_c64() {
    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -4.0),
        Complex64::new(0.0, 5.0),
    ];
    let storage = make_dense_c64(data, &[3]);
    let conj_storage = storage.conj();

    match conj_storage {
        Storage::DenseC64(v) => {
            let expected = vec![
                Complex64::new(1.0, -2.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(0.0, -5.0),
            ];
            assert_eq!(v.as_slice(), &expected);
        }
        _ => panic!("Expected DenseC64"),
    }
}

#[test]
fn test_storage_conj_diag_c64() {
    let data = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -2.0)];
    let storage = Storage::DiagC64(DiagStorageC64::from_vec(data));
    let conj_storage = storage.conj();

    match conj_storage {
        Storage::DiagC64(v) => {
            let expected = vec![Complex64::new(1.0, -1.0), Complex64::new(2.0, 2.0)];
            assert_eq!(v.as_slice(), &expected);
        }
        _ => panic!("Expected DiagC64"),
    }
}

#[test]
fn test_tensor_conj_f64() {
    let i = Index::new_dyn(2);
    let data = vec![1.0, 2.0];
    let storage = Arc::new(make_dense_f64(data.clone(), &[2]));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i.clone()], storage);

    let conj_tensor = tensor.conj();

    // Indices should be the same
    assert_eq!(conj_tensor.indices[0].id, i.id);
    // Dims should be the same
    assert_eq!(conj_tensor.dims(), vec![2]);
    // Data should be the same (real conj is identity)
    match conj_tensor.storage().as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.as_slice(), &data);
        }
        _ => panic!("Expected DenseF64"),
    }
}

#[test]
fn test_tensor_conj_c64() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -2.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(0.0, 4.0),
        Complex64::new(-1.0, 1.0),
        Complex64::new(5.0, 5.0),
    ];
    let storage = Arc::new(make_dense_c64(data, &[2, 3]));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i.clone(), j.clone()], storage);

    let conj_tensor = tensor.conj();

    // Indices should be preserved
    assert_eq!(conj_tensor.indices[0].id, i.id);
    assert_eq!(conj_tensor.indices[1].id, j.id);
    // Dims should be preserved
    assert_eq!(conj_tensor.dims(), vec![2, 3]);
    // Data should be conjugated
    match conj_tensor.storage().as_ref() {
        Storage::DenseC64(v) => {
            let expected = vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, 2.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, -4.0),
                Complex64::new(-1.0, -1.0),
                Complex64::new(5.0, -5.0),
            ];
            assert_eq!(v.as_slice(), &expected);
        }
        _ => panic!("Expected DenseC64"),
    }
}

// ============================================================================
// High-level API tests for TensorDynLen
// ============================================================================

#[test]
fn test_from_dense_f64() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = TensorDynLen::from_dense(vec![i, j], data.clone()).unwrap();

    assert_eq!(tensor.dims(), vec![2, 3]);
    assert!(tensor.is_f64());
    assert!(!tensor.is_complex());
    assert_eq!(tensor.as_slice_f64().unwrap(), &data[..]);
    assert_eq!(tensor.to_vec_f64().unwrap(), data);
}

#[test]
fn test_from_dense_c64() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let data: Vec<Complex64> = (1..=6)
        .map(|x| Complex64::new(x as f64, x as f64))
        .collect();
    let tensor = TensorDynLen::from_dense(vec![i, j], data.clone()).unwrap();

    assert_eq!(tensor.dims(), vec![2, 3]);
    assert!(!tensor.is_f64());
    assert!(tensor.is_complex());
    assert_eq!(tensor.as_slice_c64().unwrap(), &data[..]);
    assert_eq!(tensor.to_vec_c64().unwrap(), data);
}

fn assert_dense_constructor_dims<T>(data: Vec<T>)
where
    T: TensorElement,
    TensorDynLen: std::fmt::Debug,
{
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![i, j], data).unwrap();
    assert_eq!(tensor.dims(), vec![2, 2]);
}

#[test]
fn test_from_dense_generic_supports_all_supported_element_types() {
    assert_dense_constructor_dims::<f32>(vec![1.0, 2.0, 3.0, 4.0]);
    assert_dense_constructor_dims::<f64>(vec![1.0, 2.0, 3.0, 4.0]);
    assert_dense_constructor_dims::<Complex32>(vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, -0.5),
        Complex32::new(3.0, 1.5),
        Complex32::new(4.0, -1.5),
    ]);
    assert_dense_constructor_dims::<Complex64>(vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, -0.5),
        Complex64::new(3.0, 1.5),
        Complex64::new(4.0, -1.5),
    ]);
}

#[test]
fn test_from_dense_generic_preserves_column_major_input_order() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let permuted = tensor.permute_indices(&[j, i]);
    assert_eq!(
        permuted.to_vec_f64().unwrap(),
        vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn test_from_dense_generic_rejects_length_mismatch() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let err = TensorDynLen::from_dense(vec![i, j], vec![1.0_f64, 2.0, 3.0]).unwrap_err();
    assert!(
        err.to_string().contains("length"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_from_diag_generic_supports_all_supported_element_types() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);

    assert!(
        TensorDynLen::from_diag(vec![i.clone(), j.clone()], vec![1.0_f32, 2.0, 3.0])
            .unwrap()
            .is_diag()
    );
    assert!(
        TensorDynLen::from_diag(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0])
            .unwrap()
            .is_diag()
    );
    assert!(TensorDynLen::from_diag(
        vec![i.clone(), j.clone()],
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.5),
            Complex32::new(3.0, -0.5),
        ],
    )
    .unwrap()
    .is_diag());
    assert!(TensorDynLen::from_diag(
        vec![i, j],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(3.0, -0.5),
        ],
    )
    .unwrap()
    .is_diag());
}

#[test]
fn test_from_diag_generic_rejects_mismatched_index_dims() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let err = TensorDynLen::from_diag(vec![i, j], vec![1.0_f64, 2.0]).unwrap_err();
    assert!(
        err.to_string().contains("same dimension"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_from_diag_generic_rejects_payload_length_mismatch() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let err = TensorDynLen::from_diag(vec![i, j], vec![1.0_f64, 2.0]).unwrap_err();
    assert!(
        err.to_string().contains("length"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_scalar_f64() {
    let scalar = TensorDynLen::scalar(42.0).unwrap();
    assert_eq!(scalar.dims(), Vec::<usize>::new());
    assert!(scalar.is_f64());
    assert_eq!(scalar.as_slice_f64().unwrap(), &[42.0]);
}

#[test]
fn test_scalar_c64() {
    let z = Complex64::new(1.0, 2.0);
    let scalar = TensorDynLen::scalar(z).unwrap();
    assert_eq!(scalar.dims(), Vec::<usize>::new());
    assert!(scalar.is_complex());
    assert_eq!(scalar.as_slice_c64().unwrap(), &[z]);
}

#[test]
fn test_zeros_f64() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::zeros::<f64>(vec![i, j]).unwrap();

    assert_eq!(tensor.dims(), vec![2, 3]);
    assert!(tensor.is_f64());
    let data = tensor.as_slice_f64().unwrap();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_zeros_c64() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::zeros::<num_complex::Complex64>(vec![i, j]).unwrap();

    assert_eq!(tensor.dims(), vec![2, 3]);
    assert!(tensor.is_complex());
    let data = tensor.as_slice_c64().unwrap();
    assert!(data.iter().all(|&x| x == Complex64::new(0.0, 0.0)));
}

#[test]
fn test_as_slice_error_on_wrong_type() {
    let i = Index::new_dyn(2);
    // Create f64 tensor but try to get c64 slice
    let tensor_f64 = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    assert!(tensor_f64.as_slice_c64().is_err());
    assert!(tensor_f64.to_vec_c64().is_err());

    // Create c64 tensor but try to get f64 slice
    let tensor_c64 = TensorDynLen::from_dense(
        vec![i],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();
    assert!(tensor_c64.as_slice_f64().is_err());
    assert!(tensor_c64.to_vec_f64().is_err());
}
