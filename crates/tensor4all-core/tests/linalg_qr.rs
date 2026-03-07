use num_complex::Complex64;
use std::sync::Arc;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{qr, qr_c64};
use tensor4all_core::{Storage, TensorDynLen, TensorLike};

#[test]
fn test_qr_identity() {
    // Test QR of a 2×2 identity matrix
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create identity matrix: [[1, 0], [0, 1]]
    let mut data = vec![0.0; 4];
    data[0] = 1.0; // [0, 0]
    data[3] = 1.0; // [1, 1]

    let storage = Arc::new(Storage::DenseF64(
        tensor4all_core::storage::DenseStorageF64::from_vec_with_shape(data, &[2, 2]),
    ));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i.clone(), j.clone()], storage);

    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).expect("QR should succeed");

    // Check dimensions
    assert_eq!(q.dims(), vec![2, 2]);
    assert_eq!(r.dims(), vec![2, 2]);

    // Check indices
    assert_eq!(q.indices.len(), 2);
    assert_eq!(r.indices.len(), 2);

    // Check that Q and R share the bond index
    assert_eq!(q.indices[1].id, r.indices[0].id);

    // Check that bond index has "Link" tag
    assert!(q.indices[1].tags().has_tag("Link"));
    assert!(r.indices[0].tags().has_tag("Link"));
}

#[test]
fn test_qr_simple_matrix() {
    // Test QR of a simple 2×3 matrix: [[1, 2, 3], [4, 5, 6]]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_core::storage::DenseStorageF64::from_vec_with_shape(data, &[2, 3]),
    ));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i.clone(), j.clone()], storage);

    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).expect("QR should succeed");

    // Check dimensions: m=2, n=3, k=min(2,3)=2
    assert_eq!(q.dims(), vec![2, 2]);
    assert_eq!(r.dims(), vec![2, 3]);

    // Check indices
    assert_eq!(q.indices.len(), 2);
    assert_eq!(r.indices.len(), 2);

    // Check that Q and R share the bond index
    assert_eq!(q.indices[1].id, r.indices[0].id);

    // Check that bond index has "Link" tag
    assert!(q.indices[1].tags().has_tag("Link"));
    assert!(r.indices[0].tags().has_tag("Link"));
}

#[test]
fn test_qr_reconstruction() {
    // Test that Q * R reconstructs the original matrix
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);

    // Create a random-ish matrix
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_core::storage::DenseStorageF64::from_vec_with_shape(data.clone(), &[3, 4]),
    ));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i.clone(), j.clone()], storage);

    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).expect("QR should succeed");

    // Reconstruct: A = Q * R
    let reconstructed = q.contract(&r);

    // Check reconstruction accuracy
    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0),
        "QR reconstruction failed: maxabs diff = {}",
        (&tensor - &reconstructed).maxabs()
    );
}

#[test]
fn test_qr_invalid_rank() {
    // Test that QR fails for rank-1 tensors
    let i = Index::new_dyn(2);

    let storage = Arc::new(Storage::new_dense_f64(2));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i.clone()], storage);

    let result = qr::<f64>(&tensor, std::slice::from_ref(&i));
    assert!(result.is_err());
    // Expected: unfold_split returns an error for rank < 2
    if result.is_ok() {
        panic!("Expected error but got Ok");
    }
}

#[test]
fn test_qr_invalid_split() {
    // Test that QR fails when left_inds is empty or contains all indices
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i.clone(), j.clone()], storage);

    // Empty left_inds should fail
    let result = qr::<f64>(&tensor, &[]);
    assert!(result.is_err(), "Expected error for empty left_inds");

    // All indices in left_inds should fail
    let result = qr::<f64>(&tensor, &[i, j]);
    assert!(
        result.is_err(),
        "Expected error for all indices in left_inds"
    );
}

#[test]
fn test_qr_rank3() {
    // Test QR of a rank-3 tensor: split first index vs remaining two
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Create a 2×3×4 tensor with some data
    let data = (0..24).map(|x| x as f64).collect::<Vec<_>>();
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_core::storage::DenseStorageF64::from_vec_with_shape(data, &[2, 3, 4]),
    ));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i.clone(), j.clone(), k.clone()], storage);

    // Split: left = [i], right = [j, k]
    // This unfolds to a 2×12 matrix
    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).expect("QR should succeed");

    // Check dimensions:
    // Q should be [i, bond] = [2, min(2, 12)] = [2, 2]
    // R should be [bond, j, k] = [2, 3, 4]
    assert_eq!(q.dims(), vec![2, 2]);
    assert_eq!(r.dims(), vec![2, 3, 4]);

    // Check indices
    assert_eq!(q.indices.len(), 2);
    assert_eq!(r.indices.len(), 3);

    // Check that Q has left index first, then bond
    assert_eq!(q.indices[0].id, i.id);

    // Check that R has bond first, then right indices
    assert_eq!(r.indices[0].id, q.indices[1].id); // bond index
    assert_eq!(r.indices[1].id, j.id);
    assert_eq!(r.indices[2].id, k.id);

    // Check that Q and R share the bond index
    assert_eq!(q.indices[1].id, r.indices[0].id);

    // Check that bond index has "Link" tag
    assert!(q.indices[1].tags().has_tag("Link"));
    assert!(r.indices[0].tags().has_tag("Link"));
}

#[test]
fn test_qr_nontrivial_split_reconstruction() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let l = Index::new_dyn(2);

    let data = (1..=24).map(|x| x as f64).collect::<Vec<_>>();
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_core::storage::DenseStorageF64::from_vec_with_shape(data, &[2, 3, 2, 2]),
    ));
    let tensor = TensorDynLen::new(vec![i.clone(), j.clone(), k.clone(), l.clone()], storage);

    let (q, r) = qr::<f64>(&tensor, &[i.clone(), k.clone()]).expect("QR should succeed");
    let reconstructed = q.contract(&r);

    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0),
        "QR nontrivial split reconstruction failed: maxabs diff = {}",
        (&tensor - &reconstructed).maxabs()
    );
}

#[test]
fn test_qr_complex_reconstruction() {
    // Complex matrix: [[i, 0], [0, 2]]
    let i_idx = Index::new_dyn(2);
    let j_idx = Index::new_dyn(2);

    let data = vec![
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let storage = Arc::new(Storage::DenseC64(
        tensor4all_core::storage::DenseStorageC64::from_vec_with_shape(data.clone(), &[2, 2]),
    ));
    let tensor: TensorDynLen = TensorDynLen::new(vec![i_idx.clone(), j_idx.clone()], storage);

    let (q, r) = qr_c64(&tensor, std::slice::from_ref(&i_idx)).expect("Complex QR should succeed");

    let reconstructed = q.contract(&r);
    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0),
        "Complex QR reconstruction failed: maxabs diff = {}",
        (&tensor - &reconstructed).maxabs()
    );
}
