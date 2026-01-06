use tensor4all_core::index::{DefaultIndex as Index, DynId};
use tensor4all_core::{AnyScalar, DenseStorageFactory, Storage, TensorDynLen, make_mut_storage};
use std::sync::Arc;
use num_complex::Complex64;

#[test]
fn test_storage_dense_f64() {
    let storage = Storage::new_dense_f64(10);
    assert_eq!(storage.len(), 0);
    assert_eq!(storage.sum_f64(), 0.0);
    
    match storage {
        Storage::DenseF64(v) => {
            assert_eq!(v.capacity(), 10);
        }
        Storage::DenseC64(_) => panic!("expected DenseF64"),
        Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
}

#[test]
fn test_storage_dense_c64() {
    let storage = Storage::new_dense_c64(10);
    assert_eq!(storage.len(), 0);
    assert_eq!(storage.sum_c64(), Complex64::new(0.0, 0.0));

    match storage {
        Storage::DenseC64(v) => {
            assert_eq!(v.capacity(), 10);
        }
        Storage::DenseF64(_) => panic!("expected DenseC64"),
        Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseC64"),
    }
}

#[test]
fn test_storage_factory_f64() {
    let storage = <f64 as DenseStorageFactory>::new_dense(7);
    match storage {
            Storage::DenseF64(v) => assert_eq!(v.capacity(), 7),
            Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
}

#[test]
fn test_storage_factory_c64() {
    let storage = <Complex64 as DenseStorageFactory>::new_dense(9);
    match storage {
            Storage::DenseC64(v) => assert_eq!(v.capacity(), 9),
            Storage::DenseF64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseC64"),
    }
}

#[test]
fn test_cow_storage() {
    let mut storage1 = Arc::new(Storage::new_dense_f64(0));
    let storage2 = Arc::clone(&storage1);
    
    // Initially, both point to the same storage
    assert!(Arc::ptr_eq(&storage1, &storage2));
    
    // Mutate storage1 (should clone due to COW)
    {
        let mut_storage = make_mut_storage(&mut storage1);
        match mut_storage {
            Storage::DenseF64(v) => {
                v.push(1.0);
                v.push(2.0);
            }
            Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
        }
    }
    
    // After mutation, they should be different
    assert!(!Arc::ptr_eq(&storage1, &storage2));
    
    // storage2 should be unchanged
    match storage2.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 0);
        }
        Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
    
    // storage1 should have the new data
    match storage1.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v.get(0), 1.0);
            assert_eq!(v.get(1), 2.0);
        }
        Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
}

#[test]
fn test_tensor_dyn_len_creation() {
    let indices = vec![
        Index::new_dyn(2),
        Index::new_dyn(3),
    ];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    assert_eq!(tensor.indices.len(), 2);
    assert_eq!(tensor.dims.len(), 2);
    assert_eq!(tensor.dims[0], 2);
    assert_eq!(tensor.dims[1], 3);
}

#[test]
#[should_panic(expected = "indices and dims must have the same length")]
fn test_tensor_dyn_len_mismatch() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2, 3]; // mismatch
    let storage = Arc::new(Storage::new_dense_f64(6));
    
    let _tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
}


#[test]
fn test_tensor_cow() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2];
    let storage = Arc::new(Storage::new_dense_f64(2));
    
    let mut tensor1 = TensorDynLen::<DynId>::new(indices.clone(), dims.clone(), Arc::clone(&storage));
    let tensor2 = TensorDynLen::<DynId>::new(indices, dims, storage);
    
    // Initially, both tensors share the same storage
    assert!(Arc::ptr_eq(&tensor1.storage, &tensor2.storage));
    
    // Mutate tensor1's storage (should clone due to COW)
    {
        let mut_storage = tensor1.storage_mut();
        match mut_storage {
            Storage::DenseF64(v) => {
                v.push(42.0);
            }
            Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
        }
    }
    
    // After mutation, they should have different storage
    assert!(!Arc::ptr_eq(&tensor1.storage, &tensor2.storage));
    
    // tensor2's storage should be unchanged
    match tensor2.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 0);
        }
        Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
    
    // tensor1's storage should have the new data
    match tensor1.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 1);
            assert_eq!(v.get(0), 42.0);
        }
        Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
}

#[test]
fn test_tensor_sum_f64_no_match() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2];
    let mut storage = Arc::new(Storage::new_dense_f64(0));
    {
        let s = make_mut_storage(&mut storage);
        match s {
            Storage::DenseF64(v) => v.extend([1.0, 2.0, 3.0].iter().copied()),
            Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
        }
    }

    let t: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    let sum_f64 = t.sum_f64();
    assert_eq!(sum_f64, 6.0);

    let sum_any: AnyScalar = t.sum();
    assert_eq!(sum_any, AnyScalar::F64(6.0));
}

#[test]
fn test_tensor_sum_c64() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2];
    let mut storage = Arc::new(Storage::new_dense_c64(0));
    {
        let s = make_mut_storage(&mut storage);
        match s {
            Storage::DenseC64(v) => v.extend([Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)]),
            Storage::DenseF64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseC64"),
        }
    }

    // Now always returns AnyScalar
    let t: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    let sum_any: AnyScalar = t.sum();
    assert_eq!(sum_any, AnyScalar::C64(Complex64::new(4.0, 1.0)));
}

#[test]
#[should_panic(expected = "Tensor indices must all be unique")]
fn test_tensor_duplicate_indices_new() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone(), i.clone()]; // duplicate i
    let dims = vec![2, 3, 2];
    let storage = Arc::new(Storage::new_dense_f64(12));
    
    let _tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
}

#[test]
#[should_panic(expected = "Tensor indices must all be unique")]
fn test_tensor_duplicate_indices_from_indices() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone(), i.clone()]; // duplicate i
    let storage = Arc::new(Storage::new_dense_f64(12));

    let _tensor: TensorDynLen<DynId> = TensorDynLen::from_indices(indices, storage);
}

// ============================================================================
// Index Replacement Tests
// ============================================================================

#[test]
fn test_replaceind_basic() {
    use tensor4all_core::storage::DenseStorageF64;

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let new_i = Index::new_dyn(2);  // Same dimension, different ID

    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, vec![2, 3], storage);

    // Replace index i with new_i
    let replaced = tensor.replaceind(&i, &new_i);

    // Check that the first index was replaced
    assert_eq!(replaced.indices[0].id, new_i.id);
    // Check that the second index was not affected
    assert_eq!(replaced.indices[1].id, j.id);
    // Check that dimensions are unchanged
    assert_eq!(replaced.dims, vec![2, 3]);
    // Check that storage is shared (no data copy)
    assert!(Arc::ptr_eq(&tensor.storage, &replaced.storage));
}

#[test]
fn test_replaceind_no_match() {
    use tensor4all_core::storage::DenseStorageF64;

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);  // Not in tensor
    let new_k = Index::new_dyn(4);

    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, vec![2, 3], storage);

    // Replace index k (not in tensor) - should return unchanged tensor
    let replaced = tensor.replaceind(&k, &new_k);

    // Check that indices are unchanged
    assert_eq!(replaced.indices[0].id, i.id);
    assert_eq!(replaced.indices[1].id, j.id);
}

#[test]
fn test_replaceinds_basic() {
    use tensor4all_core::storage::DenseStorageF64;

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let new_i = Index::new_dyn(2);
    let new_j = Index::new_dyn(3);
    let new_k = Index::new_dyn(4);

    let indices = vec![i.clone(), j.clone(), k.clone()];
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, vec![2, 3, 4], storage);

    // Replace all indices
    let replaced = tensor.replaceinds(
        &[i.clone(), j.clone(), k.clone()],
        &[new_i.clone(), new_j.clone(), new_k.clone()]
    );

    // Check that all indices were replaced
    assert_eq!(replaced.indices[0].id, new_i.id);
    assert_eq!(replaced.indices[1].id, new_j.id);
    assert_eq!(replaced.indices[2].id, new_k.id);
    // Check that dimensions are unchanged
    assert_eq!(replaced.dims, vec![2, 3, 4]);
}

#[test]
fn test_replaceinds_partial() {
    use tensor4all_core::storage::DenseStorageF64;

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let new_i = Index::new_dyn(2);
    // Only replace i, not j or k

    let indices = vec![i.clone(), j.clone(), k.clone()];
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, vec![2, 3, 4], storage);

    // Replace only i
    let replaced = tensor.replaceinds(&[i.clone()], &[new_i.clone()]);

    // Check that i was replaced
    assert_eq!(replaced.indices[0].id, new_i.id);
    // Check that j and k are unchanged
    assert_eq!(replaced.indices[1].id, j.id);
    assert_eq!(replaced.indices[2].id, k.id);
}

#[test]
#[should_panic(expected = "old_indices and new_indices must have the same length")]
fn test_replaceinds_length_mismatch() {
    use tensor4all_core::storage::DenseStorageF64;

    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let new_i = Index::new_dyn(2);
    let new_j = Index::new_dyn(3);

    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, vec![2, 3], storage);

    // Should panic - length mismatch
    let _replaced = tensor.replaceinds(&[i.clone()], &[new_i.clone(), new_j.clone()]);
}

// ============================================================================
// Complex Conjugation Tests
// ============================================================================

#[test]
fn test_storage_conj_f64() {
    use tensor4all_core::storage::DenseStorageF64;

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let storage = Storage::DenseF64(DenseStorageF64::from_vec(data.clone()));
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
    use tensor4all_core::storage::DenseStorageC64;

    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -4.0),
        Complex64::new(0.0, 5.0),
    ];
    let storage = Storage::DenseC64(DenseStorageC64::from_vec(data));
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
    use tensor4all_core::storage::DiagStorageC64;

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
    use tensor4all_core::storage::DenseStorageF64;

    let i = Index::new_dyn(2);
    let data = vec![1.0, 2.0];
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data.clone())));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(vec![i.clone()], vec![2], storage);

    let conj_tensor = tensor.conj();

    // Indices should be the same
    assert_eq!(conj_tensor.indices[0].id, i.id);
    // Dims should be the same
    assert_eq!(conj_tensor.dims, vec![2]);
    // Data should be the same (real conj is identity)
    match conj_tensor.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.as_slice(), &data);
        }
        _ => panic!("Expected DenseF64"),
    }
}

#[test]
fn test_tensor_conj_c64() {
    use tensor4all_core::storage::DenseStorageC64;

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
    let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(
        vec![i.clone(), j.clone()],
        vec![2, 3],
        storage,
    );

    let conj_tensor = tensor.conj();

    // Indices should be preserved
    assert_eq!(conj_tensor.indices[0].id, i.id);
    assert_eq!(conj_tensor.indices[1].id, j.id);
    // Dims should be preserved
    assert_eq!(conj_tensor.dims, vec![2, 3]);
    // Data should be conjugated
    match conj_tensor.storage.as_ref() {
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

