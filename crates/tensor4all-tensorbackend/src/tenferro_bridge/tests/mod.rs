use super::*;
use crate::storage::Storage;

fn assert_storage_eq(lhs: &Storage, rhs: &Storage) {
    match (lhs.repr(), rhs.repr()) {
        (StorageRepr::DenseF64(a), StorageRepr::DenseF64(b)) => {
            assert_eq!(a.dims(), b.dims());
            assert_eq!(a.as_slice(), b.as_slice());
        }
        (StorageRepr::DenseC64(a), StorageRepr::DenseC64(b)) => {
            assert_eq!(a.dims(), b.dims());
            assert_eq!(a.as_slice(), b.as_slice());
        }
        (StorageRepr::DiagF64(a), StorageRepr::DiagF64(b)) => {
            assert_eq!(a.as_slice(), b.as_slice());
        }
        (StorageRepr::DiagC64(a), StorageRepr::DiagC64(b)) => {
            assert_eq!(a.as_slice(), b.as_slice());
        }
        (StorageRepr::StructuredF64(a), StorageRepr::StructuredF64(b)) => {
            assert_eq!(a.payload_dims(), b.payload_dims());
            assert_eq!(a.strides(), b.strides());
            assert_eq!(a.axis_classes(), b.axis_classes());
            assert_eq!(a.data(), b.data());
        }
        (StorageRepr::StructuredC64(a), StorageRepr::StructuredC64(b)) => {
            assert_eq!(a.payload_dims(), b.payload_dims());
            assert_eq!(a.strides(), b.strides());
            assert_eq!(a.axis_classes(), b.axis_classes());
            assert_eq!(a.data(), b.data());
        }
        _ => panic!(
            "storage mismatch: lhs variant {:?}, rhs variant {:?}",
            std::mem::discriminant(lhs.repr()),
            std::mem::discriminant(rhs.repr())
        ),
    }
}

#[test]
fn storage_native_roundtrip_dense_f64() {
    let storage = Storage::from_dense_f64_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();

    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();

    let expected = Storage::from_dense_f64_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn storage_native_roundtrip_diag_preserves_diag_layout() {
    let storage = Storage::from_diag_f64_col_major(vec![2.0, -1.0, 4.0], 2).unwrap();

    let native = storage_to_native_tensor(&storage, &[3, 3]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();

    assert!(native.is_diag());
    let expected = Storage::from_diag_f64_col_major(vec![2.0, -1.0, 4.0], 2).unwrap();
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn native_dense_materialization_sets_default_runtime_if_needed() {
    let native = storage_to_native_tensor(
        &Storage::from_diag_f64_col_major(vec![2.0, -1.0, 4.0], 2).unwrap(),
        &[3, 3],
    )
    .unwrap();

    let dense = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();

    assert_eq!(
        dense,
        vec![
            2.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, //
            0.0, 0.0, 4.0,
        ]
    );
}

#[test]
fn storage_native_roundtrip_structured_preserves_axis_classes() {
    let payload = NativeTensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let native = NativeTensor::with_axis_classes(payload, &[0, 1, 1]).unwrap();

    let storage = native_tensor_primal_to_storage(&native).unwrap();
    let roundtrip = storage_to_native_tensor(&storage, &[2, 2, 2]).unwrap();

    match storage.repr() {
        StorageRepr::StructuredF64(value) => {
            assert_eq!(value.axis_classes(), &[0, 1, 1]);
            assert_eq!(value.payload_dims(), &[2, 2]);
        }
        other => panic!("expected StructuredF64 storage, got {other:?}"),
    }
    assert_eq!(roundtrip.dims(), &[2, 2, 2]);
    assert_eq!(roundtrip.axis_classes(), &[0, 1, 1]);
    assert!(!roundtrip.is_dense());
    assert!(!roundtrip.is_diag());
}

#[test]
fn sum_native_tensor_returns_rank0_scalar() {
    let storage = Storage::from_dense_c64_col_major(
        vec![Complex64::new(1.0, -1.0), Complex64::new(-0.5, 2.0)],
        &[2],
    )
    .unwrap();
    let native = storage_to_native_tensor(&storage, &[2]).unwrap();

    let sum = sum_native_tensor(&native).unwrap();

    assert!(sum.is_complex());
    assert_eq!(sum.as_c64(), Some(Complex64::new(0.5, 1.0)));
}

#[test]
fn native_snapshot_materializes_lazy_conjugation() {
    let storage = Storage::from_dense_c64_col_major(
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(-3.0, 4.5),
            Complex64::new(2.5, 0.25),
        ],
        &[2, 2],
    )
    .unwrap();
    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();

    let conjugated = conj_native_tensor(&native).unwrap();
    let snapshot = native_tensor_primal_to_storage(&conjugated).unwrap();

    let expected = Storage::from_dense_c64_col_major(
        vec![
            Complex64::new(1.0, -2.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-3.0, -4.5),
            Complex64::new(2.5, -0.25),
        ],
        &[2, 2],
    )
    .unwrap();
    assert_storage_eq(&snapshot, &expected);
}

#[test]
fn native_einsum_accepts_unsorted_nonfirst_operand_labels() {
    let lhs = storage_to_native_tensor(
        &Storage::from_dense_f64_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();
    let rhs = storage_to_native_tensor(
        &Storage::from_dense_f64_col_major(vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0], &[3, 2])
            .unwrap(),
        &[3, 2],
    )
    .unwrap();

    let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[2, 1])], &[0, 2]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();

    let expected =
        Storage::from_dense_f64_col_major(vec![50.0, 110.0, 110.0, 250.0, 170.0, 390.0], &[2, 3])
            .unwrap();
    assert_storage_eq(&snapshot, &expected);
}

#[test]
fn contract_native_tensor_restores_rhs_free_axis_order() {
    let lhs = storage_to_native_tensor(
        &Storage::from_dense_f64_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();
    let rhs = storage_to_native_tensor(
        &Storage::from_dense_f64_col_major(vec![10.0, 30.0, 20.0, 40.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();

    let out = contract_native_tensor(&lhs, &[1], &rhs, &[1]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();

    let expected =
        Storage::from_dense_f64_col_major(vec![50.0, 110.0, 110.0, 250.0], &[2, 2]).unwrap();
    assert_storage_eq(&snapshot, &expected);
}

#[test]
fn dense_native_tensor_column_major_roundtrip_preserves_linearization() {
    let native =
        dense_native_tensor_from_col_major(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let values = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();

    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn permute_storage_native_dense_matches_expected_data() {
    let storage =
        Storage::from_dense_f64_col_major(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]).unwrap();

    let native = permute_storage_native(&storage, &[2, 3], &[1, 0]).unwrap();

    let expected =
        Storage::from_dense_f64_col_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    assert_storage_eq(&native, &expected);
}

#[test]
fn reshape_col_major_native_tensor_handles_noncontiguous_permuted_input() {
    let native = dense_native_tensor_from_col_major(
        &(1..=24).map(|x| x as f64).collect::<Vec<_>>(),
        &[2, 3, 2, 2],
    )
    .unwrap();
    let permuted = permute_native_tensor(&native, &[0, 2, 1, 3]).unwrap();
    let permuted_values = native_tensor_primal_to_dense_f64_col_major(&permuted).unwrap();

    let reshaped = reshape_col_major_native_tensor(&permuted, &[4, 6]).unwrap();
    let reshaped_values = native_tensor_primal_to_dense_f64_col_major(&reshaped).unwrap();

    assert_eq!(reshaped_values, permuted_values);
}
