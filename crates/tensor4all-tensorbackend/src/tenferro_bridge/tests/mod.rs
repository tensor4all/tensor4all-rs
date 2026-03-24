use super::*;
use crate::storage::Storage;

fn assert_storage_eq(lhs: &Storage, rhs: &Storage) {
    match (lhs.repr(), rhs.repr()) {
        (StorageRepr::F64(a), StorageRepr::F64(b)) => {
            assert_eq!(a.payload_dims(), b.payload_dims());
            assert_eq!(a.strides(), b.strides());
            assert_eq!(a.axis_classes(), b.axis_classes());
            assert_eq!(a.data(), b.data());
        }
        (StorageRepr::C64(a), StorageRepr::C64(b)) => {
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
        StorageRepr::F64(value) => {
            assert_eq!(value.axis_classes(), &[0, 1, 1]);
            assert_eq!(value.payload_dims(), &[2, 2]);
        }
        other => panic!("expected F64 storage, got {other:?}"),
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

// ===== contract_storage_native =====

#[test]
fn contract_storage_native_dense_f64() {
    let a = Storage::from_dense_f64_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let b = Storage::from_dense_f64_col_major(vec![5.0, 7.0, 6.0, 8.0], &[2, 2]).unwrap();

    let result = contract_storage_native(&a, &[2, 2], &[1], &b, &[2, 2], &[0], &[2, 2]).unwrap();

    // A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = A@B = [[19,22],[43,50]]
    // col-major: [19, 43, 22, 50]
    let expected =
        Storage::from_dense_f64_col_major(vec![19.0, 43.0, 22.0, 50.0], &[2, 2]).unwrap();
    assert_storage_eq(&result, &expected);
}

// ===== outer_product_storage_native =====

#[test]
fn outer_product_storage_native_produces_correct_shape() {
    let a = Storage::from_dense_f64_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_f64_col_major(vec![3.0, 4.0, 5.0], &[3]).unwrap();

    let result = outer_product_storage_native(&a, &[2], &b, &[3], &[2, 3]).unwrap();

    // a[i] * b[j]: col-major [2,3]
    // [1*3, 2*3, 1*4, 2*4, 1*5, 2*5] = [3, 6, 4, 8, 5, 10]
    let expected =
        Storage::from_dense_f64_col_major(vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0], &[2, 3]).unwrap();
    assert_storage_eq(&result, &expected);
}

// ===== scale_storage_native =====

#[test]
fn scale_storage_native_scales_elements() {
    let s = Storage::from_dense_f64_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let scalar = crate::AnyScalar::new_real(2.0);

    let result = scale_storage_native(&s, &[3], &scalar).unwrap();

    let expected = Storage::from_dense_f64_col_major(vec![2.0, 4.0, 6.0], &[3]).unwrap();
    assert_storage_eq(&result, &expected);
}

// ===== axpby_storage_native =====

#[test]
fn axpby_storage_native_linear_combination() {
    let lhs = Storage::from_dense_f64_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let rhs = Storage::from_dense_f64_col_major(vec![3.0, 4.0], &[2]).unwrap();
    let a = crate::AnyScalar::new_real(2.0);
    let b = crate::AnyScalar::new_real(3.0);

    let result = axpby_storage_native(&lhs, &[2], &a, &rhs, &[2], &b).unwrap();

    // 2*1 + 3*3 = 11, 2*2 + 3*4 = 16
    let expected = Storage::from_dense_f64_col_major(vec![11.0, 16.0], &[2]).unwrap();
    assert_storage_eq(&result, &expected);
}

// ===== native_tensor_primal_to_diag =====

#[test]
fn native_tensor_primal_to_diag_f64_extracts_diagonal() {
    let storage = Storage::from_diag_f64_col_major(vec![2.0, -1.0, 4.0], 2).unwrap();
    let native = storage_to_native_tensor(&storage, &[3, 3]).unwrap();

    let diag_values = native_tensor_primal_to_diag_f64(&native).unwrap();

    assert_eq!(diag_values, vec![2.0, -1.0, 4.0]);
}

#[test]
fn native_tensor_primal_to_diag_c64_extracts_diagonal() {
    let storage = Storage::from_diag_c64_col_major(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        2,
    )
    .unwrap();
    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();

    let diag_values = native_tensor_primal_to_diag_c64(&native).unwrap();

    assert_eq!(
        diag_values,
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]
    );
}

// ===== dense_native_tensor_from_col_major for c64 =====

#[test]
fn dense_native_tensor_from_col_major_c64_roundtrip() {
    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let native = dense_native_tensor_from_col_major(&data, &[2]).unwrap();

    let values = native_tensor_primal_to_dense_c64_col_major(&native).unwrap();

    assert_eq!(values, data);
}

// ===== diag_native_tensor_from_col_major =====

#[test]
fn diag_native_tensor_from_col_major_f64_roundtrip() {
    let data = vec![1.0_f64, 2.0, 3.0];
    let native = diag_native_tensor_from_col_major(&data, 2).unwrap();

    assert!(native.is_diag());
    let diag_values = native_tensor_primal_to_diag_f64(&native).unwrap();
    assert_eq!(diag_values, data);
}

// ===== structured storage roundtrip for c64 =====

#[test]
fn storage_native_roundtrip_structured_c64_preserves_axis_classes() {
    let payload =
        NativeTensor::from_slice(&[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)], &[2])
            .unwrap();
    let native = NativeTensor::with_axis_classes(payload, &[0, 0]).unwrap();

    let storage = native_tensor_primal_to_storage(&native).unwrap();
    let roundtrip = storage_to_native_tensor(&storage, &[2, 2]).unwrap();

    match storage.repr() {
        StorageRepr::C64(value) => {
            assert_eq!(value.axis_classes(), &[0, 0]);
        }
        other => panic!("expected C64 storage, got {other:?}"),
    }
    assert_eq!(roundtrip.dims(), &[2, 2]);
    assert_eq!(roundtrip.axis_classes(), &[0, 0]);
}

// ===== sum_native_tensor for f64 =====

#[test]
fn sum_native_tensor_returns_f64_scalar() {
    let storage = Storage::from_dense_f64_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let native = storage_to_native_tensor(&storage, &[3]).unwrap();

    let sum = sum_native_tensor(&native).unwrap();

    assert!(sum.is_real());
    assert_eq!(sum.real(), 6.0);
}

// ===== tangent_native_tensor =====

#[test]
fn tangent_native_tensor_returns_none_for_primal() {
    let native = dense_native_tensor_from_col_major(&[1.0_f64, 2.0], &[2]).unwrap();

    let tangent = tangent_native_tensor(&native);

    assert!(tangent.is_none());
}

// ===== einsum with empty operands =====

#[test]
fn einsum_native_tensors_rejects_empty_operands() {
    let result = einsum_native_tensors(&[], &[]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("at least one operand"));
}

// ===== build_binary_einsum_ids error paths =====

#[test]
fn build_binary_einsum_ids_rejects_mismatched_axes_len() {
    let result = build_binary_einsum_ids(2, &[0], 2, &[0, 1]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("length mismatch"));
}

#[test]
fn build_binary_einsum_ids_rejects_out_of_range_axis() {
    let result = build_binary_einsum_ids(2, &[5], 2, &[0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of range"));
}

#[test]
fn build_binary_einsum_ids_rejects_duplicate_axis() {
    let result = build_binary_einsum_ids(3, &[0, 0], 3, &[0, 1]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("duplicate"));
}
