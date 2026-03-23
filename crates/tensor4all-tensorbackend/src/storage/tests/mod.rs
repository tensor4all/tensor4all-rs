use super::*;

/// Helper to extract f64 data from storage
fn extract_f64(storage: &Storage) -> Vec<f64> {
    match storage.repr() {
        StorageRepr::DenseF64(ds) => ds.as_slice().to_vec(),
        StorageRepr::StructuredF64(ds) => ds.data().to_vec(),
        _ => panic!("Expected f64 dense-compatible storage"),
    }
}

/// Helper to extract Complex64 data from storage
fn extract_c64(storage: &Storage) -> Vec<Complex64> {
    match storage.repr() {
        StorageRepr::DenseC64(ds) => ds.as_slice().to_vec(),
        StorageRepr::StructuredC64(ds) => ds.data().to_vec(),
        _ => panic!("Expected c64 dense-compatible storage"),
    }
}

// ===== Legacy diagonal kernel generic tests =====

#[test]
fn test_diag_storage_generic_f64() {
    let diag: DiagStorage<f64> = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(diag.len(), 3);
    assert_eq!(diag.get(0), 1.0);
    assert_eq!(diag.get(1), 2.0);
    assert_eq!(diag.get(2), 3.0);
}

#[test]
fn test_diag_storage_generic_c64() {
    let diag: DiagStorage<Complex64> =
        DiagStorage::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
    assert_eq!(diag.len(), 2);
    assert_eq!(diag.get(0), Complex64::new(1.0, 2.0));
    assert_eq!(diag.get(1), Complex64::new(3.0, 4.0));
}

#[test]
fn test_diag_to_dense_vec_2d() {
    // 2D diagonal tensor [3, 3] with diag = [1, 2, 3]
    let diag: DiagStorage<f64> = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
    let dense = diag.to_dense_vec(&[3, 3]);
    // Expected: [[1,0,0], [0,2,0], [0,0,3]] in row-major
    assert_eq!(dense, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
}

#[test]
fn test_diag_to_dense_vec_3d() {
    // 3D diagonal tensor [2, 2, 2] with diag = [1, 2]
    let diag: DiagStorage<f64> = DiagStorage::from_vec(vec![1.0, 2.0]);
    let dense = diag.to_dense_vec(&[2, 2, 2]);
    // Position (0,0,0) = 1.0, position (1,1,1) = 2.0, others = 0
    // Row-major: [[[1,0],[0,0]], [[0,0],[0,2]]]
    // Linear: 1,0,0,0,0,0,0,2
    assert_eq!(dense, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
}

// ===== Diag × Dense contraction tests =====

#[test]
fn test_contract_diag_dense_2d_all_contracted() {
    // Diag tensor [3, 3] with diag = [1, 2, 3]
    // Dense tensor [3, 3] with all 1s
    // Contract all axes: result = sum_t diag[t] * dense[t, t] = 1*1 + 2*1 + 3*1 = 6
    let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
    let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0; 9], &[3, 3]));

    let result = contract_storage(&diag, &[3, 3], &[0, 1], &dense, &[3, 3], &[0, 1], &[]);

    let data = extract_f64(&result);
    assert_eq!(data.len(), 1);
    assert!((data[0] - 6.0).abs() < 1e-10);
}

#[test]
fn test_contract_diag_dense_2d_one_axis() {
    // Diag tensor [3, 3] with diag = [1, 2, 3]
    // Dense tensor [3, 2]
    // Contract axis 1 of diag with axis 0 of dense
    // Result[i, j] = diag[i, i] * dense[i, j] (since diag is only non-zero when i=k)
    //              = diag[i] * dense[i, j]
    let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
    // Dense = [[1,2], [3,4], [5,6]] in row-major
    let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
    ));

    let result = contract_storage(&diag, &[3, 3], &[1], &dense, &[3, 2], &[0], &[3, 2]);

    let data = extract_f64(&result);
    // Result should be:
    // [diag[0]*dense[0,:], diag[1]*dense[1,:], diag[2]*dense[2,:]]
    // = [1*[1,2], 2*[3,4], 3*[5,6]]
    // = [[1,2], [6,8], [15,18]]
    // Row-major: [1, 2, 6, 8, 15, 18]
    assert_eq!(data.len(), 6);
    assert!((data[0] - 1.0).abs() < 1e-10);
    assert!((data[1] - 2.0).abs() < 1e-10);
    assert!((data[2] - 6.0).abs() < 1e-10);
    assert!((data[3] - 8.0).abs() < 1e-10);
    assert!((data[4] - 15.0).abs() < 1e-10);
    assert!((data[5] - 18.0).abs() < 1e-10);
}

#[test]
fn test_contract_dense_diag_2d_one_axis() {
    // Dense × Diag (reversed order)
    // Dense tensor [2, 3]
    // Diag tensor [3, 3] with diag = [1, 2, 3]
    // Contract axis 1 of dense with axis 0 of diag
    let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    ));
    let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));

    let result = contract_storage(&dense, &[2, 3], &[1], &diag, &[3, 3], &[0], &[2, 3]);

    let data = extract_f64(&result);
    // Result[i, j] = dense[i, k] * diag[k, j] summed over k
    // But diag is only non-zero when k=j, so:
    // Result[i, j] = dense[i, j] * diag[j]
    // = [[1*1, 2*2, 3*3], [4*1, 5*2, 6*3]]
    // = [[1, 4, 9], [4, 10, 18]]
    // Row-major: [1, 4, 9, 4, 10, 18]
    assert_eq!(data.len(), 6);
    assert!((data[0] - 1.0).abs() < 1e-10);
    assert!((data[1] - 4.0).abs() < 1e-10);
    assert!((data[2] - 9.0).abs() < 1e-10);
    assert!((data[3] - 4.0).abs() < 1e-10);
    assert!((data[4] - 10.0).abs() < 1e-10);
    assert!((data[5] - 18.0).abs() < 1e-10);
}

#[test]
fn test_contract_diag_dense_3d() {
    // Diag tensor [2, 2, 2] with diag = [1, 2]
    // Dense tensor [2, 3]
    // Contract axis 2 of diag with axis 0 of dense
    // Result has shape [2, 2, 3] but only diagonal in first two indices is non-zero
    let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0]));
    let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    ));

    let result = contract_storage(&diag, &[2, 2, 2], &[2], &dense, &[2, 3], &[0], &[2, 2, 3]);

    let data = extract_f64(&result);
    assert_eq!(data.len(), 12);
    // Result shape [2, 2, 3]
    // diag only non-zero at (t, t, t), so result[i, j, k] = diag[i] * dense[i, k] if i==j, else 0
    // Result[0, 0, :] = diag[0] * dense[0, :] = 1 * [1, 2, 3] = [1, 2, 3]
    // Result[0, 1, :] = 0 (diag is zero when i != j)
    // Result[1, 0, :] = 0
    // Result[1, 1, :] = diag[1] * dense[1, :] = 2 * [4, 5, 6] = [8, 10, 12]
    // Row-major: [1,2,3, 0,0,0, 0,0,0, 8,10,12]
    assert!((data[0] - 1.0).abs() < 1e-10);
    assert!((data[1] - 2.0).abs() < 1e-10);
    assert!((data[2] - 3.0).abs() < 1e-10);
    assert!((data[3] - 0.0).abs() < 1e-10);
    assert!((data[4] - 0.0).abs() < 1e-10);
    assert!((data[5] - 0.0).abs() < 1e-10);
    assert!((data[6] - 0.0).abs() < 1e-10);
    assert!((data[7] - 0.0).abs() < 1e-10);
    assert!((data[8] - 0.0).abs() < 1e-10);
    assert!((data[9] - 8.0).abs() < 1e-10);
    assert!((data[10] - 10.0).abs() < 1e-10);
    assert!((data[11] - 12.0).abs() < 1e-10);
}

// ===== Type promotion tests =====

#[test]
fn test_contract_diag_f64_dense_c64() {
    // Diag<f64> × Dense<Complex64> should produce Dense<Complex64>
    let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0]));
    let dense = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
        ],
        &[2, 2],
    ));

    let result = contract_storage(&diag, &[2, 2], &[1], &dense, &[2, 2], &[0], &[2, 2]);

    let data = extract_c64(&result);
    assert_eq!(data.len(), 4);
    // Result[i, j] = diag[i] * dense[i, j]
    // Result[0, 0] = 1 * (1+1i) = 1+1i
    // Result[0, 1] = 1 * (2+2i) = 2+2i
    // Result[1, 0] = 2 * (3+3i) = 6+6i
    // Result[1, 1] = 2 * (4+4i) = 8+8i
    assert!((data[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
    assert!((data[1] - Complex64::new(2.0, 2.0)).norm() < 1e-10);
    assert!((data[2] - Complex64::new(6.0, 6.0)).norm() < 1e-10);
    assert!((data[3] - Complex64::new(8.0, 8.0)).norm() < 1e-10);
}

#[test]
fn test_contract_diag_c64_dense_f64() {
    // Diag<Complex64> × Dense<f64> should produce Dense<Complex64>
    let diag = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 2.0),
    ]));
    let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
        vec![1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    ));

    let result = contract_storage(&diag, &[2, 2], &[1], &dense, &[2, 2], &[0], &[2, 2]);

    let data = extract_c64(&result);
    assert_eq!(data.len(), 4);
    // Result[i, j] = diag[i] * dense[i, j]
    // Result[0, 0] = (1+1i) * 1 = 1+1i
    // Result[0, 1] = (1+1i) * 2 = 2+2i
    // Result[1, 0] = (2+2i) * 3 = 6+6i
    // Result[1, 1] = (2+2i) * 4 = 8+8i
    assert!((data[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
    assert!((data[1] - Complex64::new(2.0, 2.0)).norm() < 1e-10);
    assert!((data[2] - Complex64::new(6.0, 6.0)).norm() < 1e-10);
    assert!((data[3] - Complex64::new(8.0, 8.0)).norm() < 1e-10);
}

#[test]
fn test_contract_dense_f64_diag_c64() {
    // Dense<f64> × Diag<Complex64> should produce Dense<Complex64>
    let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(
        vec![1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    ));
    let diag = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 2.0),
    ]));

    let result = contract_storage(&dense, &[2, 2], &[1], &diag, &[2, 2], &[0], &[2, 2]);

    let data = extract_c64(&result);
    assert_eq!(data.len(), 4);
    // Result[i, j] = dense[i, j] * diag[j]
    // Result[0, 0] = 1 * (1+1i) = 1+1i
    // Result[0, 1] = 2 * (2+2i) = 4+4i
    // Result[1, 0] = 3 * (1+1i) = 3+3i
    // Result[1, 1] = 4 * (2+2i) = 8+8i
    assert!((data[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
    assert!((data[1] - Complex64::new(4.0, 4.0)).norm() < 1e-10);
    assert!((data[2] - Complex64::new(3.0, 3.0)).norm() < 1e-10);
    assert!((data[3] - Complex64::new(8.0, 8.0)).norm() < 1e-10);
}

#[test]
fn test_contract_dense_c64_diag_f64() {
    // Dense<Complex64> × Diag<f64> should produce Dense<Complex64>
    let dense = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
        ],
        &[2, 2],
    ));
    let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0]));

    let result = contract_storage(&dense, &[2, 2], &[1], &diag, &[2, 2], &[0], &[2, 2]);

    let data = extract_c64(&result);
    assert_eq!(data.len(), 4);
    // Result[i, j] = dense[i, j] * diag[j]
    // Result[0, 0] = (1+1i) * 1 = 1+1i
    // Result[0, 1] = (2+2i) * 2 = 4+4i
    // Result[1, 0] = (3+3i) * 1 = 3+3i
    // Result[1, 1] = (4+4i) * 2 = 8+8i
    assert!((data[0] - Complex64::new(1.0, 1.0)).norm() < 1e-10);
    assert!((data[1] - Complex64::new(4.0, 4.0)).norm() < 1e-10);
    assert!((data[2] - Complex64::new(3.0, 3.0)).norm() < 1e-10);
    assert!((data[3] - Complex64::new(8.0, 8.0)).norm() < 1e-10);
}

// ===== Diag × Diag contraction tests =====

#[test]
fn test_contract_diag_diag_all_contracted() {
    // Diag [3, 3] × Diag [3, 3] with all indices contracted
    // Result = sum_t diag1[t] * diag2[t] (inner product)
    let diag1 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
    let diag2 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![4.0, 5.0, 6.0]));

    let result = contract_storage(&diag1, &[3, 3], &[0, 1], &diag2, &[3, 3], &[0, 1], &[]);

    let data = extract_f64(&result);
    assert_eq!(data.len(), 1);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((data[0] - 32.0).abs() < 1e-10);
}

#[test]
fn test_contract_diag_diag_partial() {
    // Diag [3, 3] × Diag [3, 3] with one axis contracted
    // Result is a diagonal tensor
    let diag1 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0, 3.0]));
    let diag2 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![4.0, 5.0, 6.0]));

    let result = contract_storage(&diag1, &[3, 3], &[1], &diag2, &[3, 3], &[0], &[3, 3]);

    // Result is element-wise product: [1*4, 2*5, 3*6] = [4, 10, 18]
    match result.repr() {
        StorageRepr::DiagF64(d) => {
            assert_eq!(d.as_slice(), &[4.0, 10.0, 18.0]);
        }
        _ => panic!("Expected DiagF64"),
    }
}

// ===== Type inspection tests =====

#[test]
fn test_is_f64() {
    let dense_f64 = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
    let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(1.0, 0.0)],
        &[1],
    ));
    let diag_f64 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0]));
    let diag_c64 = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

    assert!(dense_f64.is_f64());
    assert!(!dense_c64.is_f64());
    assert!(diag_f64.is_f64());
    assert!(!diag_c64.is_f64());
}

#[test]
fn test_is_c64() {
    let dense_f64 = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
    let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(1.0, 0.0)],
        &[1],
    ));
    let diag_f64 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0]));
    let diag_c64 = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

    assert!(!dense_f64.is_c64());
    assert!(dense_c64.is_c64());
    assert!(!diag_f64.is_c64());
    assert!(diag_c64.is_c64());
}

#[test]
fn test_is_complex() {
    let dense_f64 = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
    let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(1.0, 0.0)],
        &[1],
    ));
    let diag_f64 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0]));
    let diag_c64 = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![Complex64::new(1.0, 0.0)]));

    // is_complex is an alias for is_c64
    assert!(!dense_f64.is_complex());
    assert!(dense_c64.is_complex());
    assert!(!diag_f64.is_complex());
    assert!(diag_c64.is_complex());
}

// ===== Legacy dense kernel tests =====

#[test]
fn test_dense_from_scalar() {
    let ds = DenseStorage::from_scalar(42.0_f64);
    assert_eq!(ds.rank(), 0);
    assert_eq!(ds.len(), 1);
    assert!(!ds.is_empty());
    assert_eq!(ds.dims(), Vec::<usize>::new());
    assert_eq!(ds.as_slice(), &[42.0]);
}

#[test]
fn test_dense_from_scalar_c64() {
    let val = Complex64::new(1.0, 2.0);
    let ds = DenseStorage::from_scalar(val);
    assert_eq!(ds.rank(), 0);
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.as_slice(), &[val]);
}

#[test]
fn test_dense_from_tensor_into_tensor_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ds = DenseStorage::from_vec_with_shape(data.clone(), &[2, 3]);
    let tensor = ds.into_tensor();
    assert_eq!(tensor.len(), 6);
    let ds2 = DenseStorage::from_tensor(tensor);
    assert_eq!(ds2.dims(), vec![2, 3]);
    assert_eq!(ds2.as_slice(), &data[..]);
}

#[test]
fn test_dense_get_set() {
    let mut ds = DenseStorage::from_vec_with_shape(vec![10.0, 20.0, 30.0], &[3]);
    assert_eq!(ds.get(0), 10.0);
    assert_eq!(ds.get(1), 20.0);
    assert_eq!(ds.get(2), 30.0);

    ds.set(1, 99.0);
    assert_eq!(ds.get(1), 99.0);
}

#[test]
fn test_dense_len_is_empty_dims_rank() {
    let ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(ds.len(), 4);
    assert!(!ds.is_empty());
    assert_eq!(ds.dims(), vec![2, 2]);
    assert_eq!(ds.rank(), 2);
}

#[test]
fn test_dense_iter() {
    let ds = DenseStorage::from_vec_with_shape(vec![5.0, 10.0, 15.0], &[3]);
    let collected: Vec<f64> = ds.iter().copied().collect();
    assert_eq!(collected, vec![5.0, 10.0, 15.0]);
}

#[test]
fn test_dense_as_slice_as_mut_slice() {
    let mut ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]);
    assert_eq!(ds.as_slice(), &[1.0, 2.0, 3.0]);

    let slice = ds.as_mut_slice();
    slice[0] = 100.0;
    assert_eq!(ds.as_slice(), &[100.0, 2.0, 3.0]);
}

#[test]
fn test_dense_into_vec() {
    let ds = DenseStorage::from_vec_with_shape(vec![7.0, 8.0, 9.0], &[3]);
    let v = ds.into_vec();
    assert_eq!(v, vec![7.0, 8.0, 9.0]);
}

#[test]
fn test_dense_permute() {
    // 2x3 tensor, permute to 3x2
    let ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let permuted = ds.permute(&[1, 0]);
    assert_eq!(permuted.dims(), vec![3, 2]);
    // Original row-major [2,3]: [[1,2,3],[4,5,6]]
    // Transposed row-major [3,2]: [[1,4],[2,5],[3,6]]
    assert_eq!(permuted.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_dense_contract_matrix_multiply() {
    // Matrix multiply: [2,3] x [3,2] -> [2,2]
    let a = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = DenseStorage::from_vec_with_shape(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let result = a.contract(&[1], &b, &[0]);
    assert_eq!(result.dims(), vec![2, 2]);
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
    let data = result.as_slice();
    assert!((data[0] - 58.0).abs() < 1e-10);
    assert!((data[1] - 64.0).abs() < 1e-10);
    assert!((data[2] - 139.0).abs() < 1e-10);
    assert!((data[3] - 154.0).abs() < 1e-10);
}

#[test]
fn test_dense_contract_inner_product() {
    // Inner product: [3] x [3] -> scalar
    let a = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]);
    let b = DenseStorage::from_vec_with_shape(vec![4.0, 5.0, 6.0], &[3]);
    let result = a.contract(&[0], &b, &[0]);
    // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
    assert_eq!(result.len(), 1);
    assert!((result.as_slice()[0] - 32.0).abs() < 1e-10);
}

#[test]
fn test_dense_random_f64() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);
    let ds = DenseStorage::<f64>::random(&mut rng, &[3, 4]);
    assert_eq!(ds.dims(), vec![3, 4]);
    assert_eq!(ds.len(), 12);
    // Values should not all be zero (with overwhelming probability)
    let nonzero = ds.as_slice().iter().any(|&x| x.abs() > 1e-10);
    assert!(nonzero);
}

#[test]
fn test_dense_random_1d_f64() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(123);
    let ds = DenseStorage::<f64>::random_1d(&mut rng, 5);
    assert_eq!(ds.dims(), vec![5]);
    assert_eq!(ds.len(), 5);
}

#[test]
fn test_dense_random_c64() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);
    let ds = DenseStorage::<Complex64>::random(&mut rng, &[2, 3]);
    assert_eq!(ds.dims(), vec![2, 3]);
    assert_eq!(ds.len(), 6);
    // At least one element should have nonzero imaginary part
    let has_imag = ds.as_slice().iter().any(|z| z.im.abs() > 1e-10);
    assert!(has_imag);
}

#[test]
fn test_dense_random_1d_c64() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(99);
    let ds = DenseStorage::<Complex64>::random_1d(&mut rng, 4);
    assert_eq!(ds.dims(), vec![4]);
    assert_eq!(ds.len(), 4);
}

// ===== Legacy diagonal kernel tests =====

#[test]
fn test_diag_as_slice_as_mut_slice() {
    let mut diag = DiagStorage::from_vec(vec![10.0, 20.0, 30.0]);
    assert_eq!(diag.as_slice(), &[10.0, 20.0, 30.0]);

    let slice = diag.as_mut_slice();
    slice[1] = 99.0;
    assert_eq!(diag.as_slice(), &[10.0, 99.0, 30.0]);
}

#[test]
fn test_diag_into_vec() {
    let diag = DiagStorage::from_vec(vec![5.0, 6.0, 7.0]);
    let v = diag.into_vec();
    assert_eq!(v, vec![5.0, 6.0, 7.0]);
}

#[test]
fn test_diag_is_empty() {
    let empty: DiagStorage<f64> = DiagStorage::from_vec(vec![]);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);

    let nonempty = DiagStorage::from_vec(vec![1.0]);
    assert!(!nonempty.is_empty());
}

#[test]
fn test_diag_set() {
    let mut diag = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
    diag.set(0, 100.0);
    diag.set(2, 300.0);
    assert_eq!(diag.get(0), 100.0);
    assert_eq!(diag.get(1), 2.0);
    assert_eq!(diag.get(2), 300.0);
}

#[test]
fn test_diag_to_dense_vec_1d() {
    // 1D diagonal tensor [3] with diag = [1, 2, 3]
    // This is just the vector itself
    let diag = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
    let dense = diag.to_dense_vec(&[3]);
    assert_eq!(dense, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_diag_to_dense_vec_c64() {
    let diag = DiagStorage::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
    let dense = diag.to_dense_vec(&[2, 2]);
    // [[1+2i, 0], [0, 3+4i]] in row-major
    assert_eq!(dense[0], Complex64::new(1.0, 2.0));
    assert_eq!(dense[1], Complex64::zero());
    assert_eq!(dense[2], Complex64::zero());
    assert_eq!(dense[3], Complex64::new(3.0, 4.0));
}

#[test]
fn test_diag_contract_diag_diag_scalar_result() {
    // All indices contracted: inner product
    let d1 = DiagStorage::from_vec(vec![1.0, 2.0, 3.0]);
    let d2 = DiagStorage::from_vec(vec![4.0, 5.0, 6.0]);
    let result = d1.contract_diag_diag(
        &[3, 3],
        &d2,
        &[3, 3],
        &[],
        |v| Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, &[])),
        |v| Storage::diag_f64_legacy(DiagStorage::from_vec(v)),
    );
    let data = extract_f64(&result);
    assert_eq!(data.len(), 1);
    // 1*4 + 2*5 + 3*6 = 32
    assert!((data[0] - 32.0).abs() < 1e-10);
}

#[test]
fn test_diag_contract_diag_diag_diag_result() {
    // Partial contraction: element-wise product
    let d1 = DiagStorage::from_vec(vec![2.0, 3.0]);
    let d2 = DiagStorage::from_vec(vec![5.0, 7.0]);
    let result = d1.contract_diag_diag(
        &[2, 2],
        &d2,
        &[2, 2],
        &[2, 2],
        |v| Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, &[2, 2])),
        |v| Storage::diag_f64_legacy(DiagStorage::from_vec(v)),
    );
    match result.repr() {
        StorageRepr::DiagF64(d) => {
            assert_eq!(d.as_slice(), &[10.0, 21.0]);
        }
        _ => panic!("Expected DiagF64"),
    }
}

#[test]
fn test_diag_contract_diag_dense_basic() {
    // Diag [2,2] diag=[1,2], Dense [2,3] = [[1,2,3],[4,5,6]]
    // Contract axis 1 of diag with axis 0 of dense
    // Result[i,j] = diag[i] * dense[i,j]
    let diag = DiagStorage::from_vec(vec![1.0, 2.0]);
    let dense = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = diag.contract_diag_dense(&[2, 2], &[1], &dense, &[2, 3], &[0], &[2, 3], |v| {
        Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(v, &[2, 3]))
    });
    let data = extract_f64(&result);
    assert_eq!(data.len(), 6);
    // Result = [[1*1, 1*2, 1*3], [2*4, 2*5, 2*6]] = [[1,2,3],[8,10,12]]
    assert!((data[0] - 1.0).abs() < 1e-10);
    assert!((data[1] - 2.0).abs() < 1e-10);
    assert!((data[2] - 3.0).abs() < 1e-10);
    assert!((data[3] - 8.0).abs() < 1e-10);
    assert!((data[4] - 10.0).abs() < 1e-10);
    assert!((data[5] - 12.0).abs() < 1e-10);
}

#[test]
fn test_dense_tensor_ref_and_mut() {
    let mut ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]);
    // Test tensor() for read access
    assert_eq!(ds.tensor().len(), 3);
    // Test tensor_mut() for write access
    ds.tensor_mut()[0] = 99.0;
    assert_eq!(ds.get(0), 99.0);
}

#[test]
fn test_dense_deref() {
    let ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]);
    // Deref gives access to tensor methods
    let _len = ds.len();
    assert_eq!(_len, 2);
}

#[test]
fn test_dense_contract_c64() {
    // Verify contraction works for Complex64 too
    let a = DenseStorage::from_vec_with_shape(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
        ],
        &[2, 2],
    );
    let b = DenseStorage::from_vec_with_shape(
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
        &[2],
    );
    let result = a.contract(&[1], &b, &[0]);
    assert_eq!(result.dims(), vec![2]);
    // C[0] = (1+0i)*(1+0i) + (0+1i)*(0+1i) = 1 + (-1) = 0
    // C[1] = (1+1i)*(1+0i) + (2+0i)*(0+1i) = 1+1i + 0+2i = 1+3i
    assert!((result.as_slice()[0] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((result.as_slice()[1] - Complex64::new(1.0, 3.0)).norm() < 1e-10);
}

#[test]
fn test_dense_permute_3d() {
    // 3D tensor [2, 3, 1], permute to [3, 1, 2]
    let ds = DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let permuted = ds.permute(&[1, 2, 0]);
    assert_eq!(permuted.dims(), vec![3, 1, 2]);
    assert_eq!(permuted.len(), 6);
}

// ===== Storage-level tests for is_diag =====

#[test]
fn test_storage_is_diag() {
    let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
    let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0]));
    assert!(!dense.is_diag());
    assert!(diag.is_diag());
}

#[test]
fn structured_storage_rejects_noncanonical_axis_classes() {
    let err = StructuredStorage::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        vec![1, 2],
        vec![1, 0, 0],
    )
    .unwrap_err();

    assert!(err.to_string().contains("canonical"));
}

#[test]
fn structured_storage_column_major_helpers_cover_contiguous_padded_and_empty_payloads() {
    let dense =
        StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(dense.logical_dims(), vec![2, 3]);
    assert!(dense.is_dense());
    assert!(!dense.is_diag());
    assert_eq!(
        dense.dense_col_major_view_if_contiguous().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        dense.payload_col_major_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let padded = StructuredStorage::new(
        vec![10.0, 20.0, -1.0, 30.0, 40.0, -1.0, 50.0, 60.0],
        vec![2, 3],
        vec![1, 3],
        vec![0, 1],
    )
    .unwrap();
    assert!(padded.dense_col_major_view_if_contiguous().is_none());
    assert_eq!(
        padded.payload_col_major_vec(),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );

    let empty = StructuredStorage::from_dense_col_major(Vec::<f64>::new(), &[0, 3]);
    assert!(empty.is_empty());
    assert_eq!(empty.payload_col_major_vec(), Vec::<f64>::new());
}

#[test]
fn structured_storage_permute_and_map_copy_preserve_metadata() {
    let storage = StructuredStorage::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        vec![0, 1, 0],
    )
    .unwrap();
    assert_eq!(storage.logical_dims(), vec![2, 3, 2]);

    let permuted = storage.permute_logical_axes(&[0, 2, 1]);
    assert_eq!(permuted.axis_classes(), &[0, 0, 1]);
    assert_eq!(permuted.logical_dims(), vec![2, 2, 3]);

    let mapped = permuted.map_copy(|x| x * 10.0);
    assert_eq!(mapped.data(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    assert_eq!(mapped.payload_dims(), &[2, 3]);
    assert_eq!(mapped.strides(), &[1, 2]);
    assert_eq!(mapped.axis_classes(), &[0, 0, 1]);
}

#[test]
fn structured_storage_validates_payload_rank_and_required_len() {
    let rank_err =
        StructuredStorage::<f64>::new(vec![1.0, 2.0], vec![2], vec![1], vec![0, 1]).unwrap_err();
    assert!(rank_err.to_string().contains("payload rank"));

    let len_err = StructuredStorage::<f64>::new(vec![1.0, 2.0], vec![2, 2], vec![1, 3], vec![0, 1])
        .unwrap_err();
    assert!(len_err.to_string().contains("required len"));

    let scalar_diag = StructuredStorage::from_diag_col_major(vec![42.0], 0);
    assert_eq!(scalar_diag.payload_dims(), &[] as &[usize]);
    assert_eq!(scalar_diag.logical_rank(), 0);
    assert!(scalar_diag.is_dense());
    assert!(!scalar_diag.is_diag());
    assert_eq!(scalar_diag.payload_col_major_vec(), vec![42.0]);
}

// ===== Storage len / is_empty =====

#[test]
fn test_storage_len_is_empty() {
    let dense = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
    assert_eq!(dense.len(), 2);
    assert!(!dense.is_empty());

    let diag = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![]));
    assert_eq!(diag.len(), 0);
    assert!(diag.is_empty());
}

// ===== Storage zero-constructor tests =====

#[test]
fn test_storage_new_dense_f64() {
    let s = Storage::new_dense_f64(3);
    assert_eq!(s.len(), 3);
    assert!(s.is_f64());
    let data = extract_f64(&s);
    assert_eq!(data, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_storage_new_dense_c64() {
    let s = Storage::new_dense_c64(2);
    assert_eq!(s.len(), 2);
    assert!(s.is_c64());
}

#[test]
fn test_storage_from_dense_f64_col_major_zeros_with_shape() {
    let s = Storage::from_dense_f64_col_major(vec![0.0; 6], &[2, 3]).unwrap();
    assert_eq!(s.len(), 6);
}

#[test]
fn test_storage_from_dense_c64_col_major_zeros_with_shape() {
    let s = Storage::from_dense_c64_col_major(vec![Complex64::new(0.0, 0.0); 6], &[3, 2]).unwrap();
    assert_eq!(s.len(), 6);
}

// ===== SumFromStorage tests =====

#[test]
fn test_sum_from_storage_f64() {
    let s = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]));
    let sum: f64 = f64::sum_from_storage(&s);
    assert!((sum - 6.0).abs() < 1e-10);
}

#[test]
fn test_sum_from_storage_diag_f64() {
    let s = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![10.0, 20.0]));
    let sum: f64 = f64::sum_from_storage(&s);
    assert!((sum - 30.0).abs() < 1e-10);
}

#[test]
fn test_storage_sum_f64_method() {
    let s = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0, 3.0], &[3]));
    assert!((s.sum_f64() - 6.0).abs() < 1e-10);
}

#[test]
fn test_storage_sum_c64_method() {
    let s = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
    ));
    let sum = s.sum_c64();
    assert!((sum - Complex64::new(4.0, 6.0)).norm() < 1e-10);
}

#[test]
fn test_storage_max_abs_and_to_dense_storage_cover_complex_and_diag() {
    let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)],
        &[2],
    ));
    assert!((dense_c64.max_abs() - 5.0).abs() < 1e-10);
    match dense_c64.to_dense_storage(&[2]).repr() {
        StorageRepr::StructuredC64(ds) => assert_eq!(
            ds.payload_col_major_vec().as_slice(),
            &[Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)]
        ),
        StorageRepr::DenseC64(ds) => assert_eq!(
            ds.as_slice(),
            &[Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)]
        ),
        other => panic!("expected DenseC64, got {other:?}"),
    }

    let diag_c64 = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
        Complex64::new(0.0, 2.0),
        Complex64::new(3.0, 4.0),
    ]));
    assert!((diag_c64.max_abs() - 5.0).abs() < 1e-10);
    match diag_c64.to_dense_storage(&[2, 2]).repr() {
        StorageRepr::StructuredC64(ds) => {
            assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[
                    Complex64::new(0.0, 2.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(3.0, 4.0),
                ]
            );
        }
        StorageRepr::DenseC64(ds) => {
            assert_eq!(
                ds.as_slice(),
                &[
                    Complex64::new(0.0, 2.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(3.0, 4.0),
                ]
            );
        }
        other => panic!("expected DenseC64, got {other:?}"),
    }
}

#[test]
fn test_storage_projection_promotion_and_conjugation_helpers() {
    let dense_c64 = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)],
        &[2],
    ));
    match dense_c64.extract_real_part().repr() {
        StorageRepr::DenseF64(ds) => assert_eq!(ds.as_slice(), &[1.0, 3.0]),
        StorageRepr::StructuredF64(ds) => {
            assert_eq!(ds.payload_col_major_vec().as_slice(), &[1.0, 3.0])
        }
        other => panic!("expected DenseF64, got {other:?}"),
    }
    match dense_c64.extract_imag_part(&[2]).repr() {
        StorageRepr::DenseF64(ds) => assert_eq!(ds.as_slice(), &[-2.0, 4.0]),
        StorageRepr::StructuredF64(ds) => {
            assert_eq!(ds.payload_col_major_vec().as_slice(), &[-2.0, 4.0])
        }
        other => panic!("expected DenseF64, got {other:?}"),
    }
    match dense_c64.conj().repr() {
        StorageRepr::DenseC64(ds) => {
            assert_eq!(
                ds.as_slice(),
                &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]
            )
        }
        StorageRepr::StructuredC64(ds) => {
            assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]
            )
        }
        other => panic!("expected DenseC64, got {other:?}"),
    }

    let diag_f64 = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![2.0, -1.0]));
    match diag_f64.extract_imag_part(&[2, 2]).repr() {
        StorageRepr::DiagF64(ds) => assert_eq!(ds.as_slice(), &[0.0, 0.0]),
        StorageRepr::StructuredF64(ds) => {
            assert_eq!(ds.payload_col_major_vec().as_slice(), &[0.0, 0.0])
        }
        other => panic!("expected DiagF64, got {other:?}"),
    }
    match diag_f64.to_complex_storage().repr() {
        StorageRepr::DiagC64(ds) => {
            assert_eq!(
                ds.as_slice(),
                &[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)]
            )
        }
        StorageRepr::StructuredC64(ds) => {
            assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)]
            )
        }
        other => panic!("expected DiagC64, got {other:?}"),
    }
    let real = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
    let imag = Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![0.5, -1.5], &[2]));
    match Storage::combine_to_complex(&real, &imag).repr() {
        StorageRepr::DenseC64(ds) => {
            assert_eq!(
                ds.as_slice(),
                &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -1.5)]
            )
        }
        StorageRepr::StructuredC64(ds) => {
            assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -1.5)]
            )
        }
        other => panic!("expected DenseC64, got {other:?}"),
    }
}

#[test]
fn test_storage_try_add_and_try_sub_cover_all_variants_and_errors() {
    let dense_f64_a =
        Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0, 2.0], &[2]));
    let dense_f64_b =
        Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![3.0, -1.0], &[2]));
    match dense_f64_a.try_add(&dense_f64_b).unwrap().repr() {
        StorageRepr::DenseF64(ds) => assert_eq!(ds.as_slice(), &[4.0, 1.0]),
        other => panic!("expected DenseF64, got {other:?}"),
    }
    match dense_f64_a.try_sub(&dense_f64_b).unwrap().repr() {
        StorageRepr::DenseF64(ds) => assert_eq!(ds.as_slice(), &[-2.0, 3.0]),
        other => panic!("expected DenseF64, got {other:?}"),
    }

    let dense_c64_a = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -2.0)],
        &[2],
    ));
    let dense_c64_b = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(-1.0, 0.5), Complex64::new(3.0, 1.0)],
        &[2],
    ));
    assert!(matches!(
        dense_c64_a.try_add(&dense_c64_b).unwrap().repr(),
        StorageRepr::DenseC64(_)
    ));
    assert!(matches!(
        dense_c64_a.try_sub(&dense_c64_b).unwrap().repr(),
        StorageRepr::DenseC64(_)
    ));

    let diag_f64_a = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![1.0, 2.0]));
    let diag_f64_b = Storage::diag_f64_legacy(DiagStorage::from_vec(vec![0.5, -3.0]));
    assert!(matches!(
        diag_f64_a.try_add(&diag_f64_b).unwrap().repr(),
        StorageRepr::DiagF64(_)
    ));
    assert!(matches!(
        diag_f64_a.try_sub(&diag_f64_b).unwrap().repr(),
        StorageRepr::DiagF64(_)
    ));

    let diag_c64_a = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 2.0),
    ]));
    let diag_c64_b = Storage::diag_c64_legacy(DiagStorage::from_vec(vec![
        Complex64::new(0.5, 0.5),
        Complex64::new(-3.0, 1.0),
    ]));
    assert!(matches!(
        diag_c64_a.try_add(&diag_c64_b).unwrap().repr(),
        StorageRepr::DiagC64(_)
    ));
    assert!(matches!(
        diag_c64_a.try_sub(&diag_c64_b).unwrap().repr(),
        StorageRepr::DiagC64(_)
    ));

    let mismatched_len =
        Storage::dense_f64_legacy(DenseStorage::from_vec_with_shape(vec![1.0], &[1]));
    let err = dense_f64_a.try_add(&mismatched_len).unwrap_err();
    assert!(err.contains("Storage lengths must match for addition"));
    let err = dense_f64_a.try_sub(&mismatched_len).unwrap_err();
    assert!(err.contains("Storage lengths must match for subtraction"));

    let mismatched_type = Storage::dense_c64_legacy(DenseStorage::from_vec_with_shape(
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        &[2],
    ));
    let err = dense_f64_a.try_add(&mismatched_type).unwrap_err();
    assert!(err.contains("Storage types must match for addition"));
    let err = dense_f64_a.try_sub(&mismatched_type).unwrap_err();
    assert!(err.contains("Storage types must match for subtraction"));
}
