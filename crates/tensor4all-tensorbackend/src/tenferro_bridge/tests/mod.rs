use super::*;
use crate::storage::Storage;
use crate::tensor_element::TensorElement;
use num_complex::{Complex32, Complex64};

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

fn recorded_native_einsum_call_count(path: NativeEinsumPath) -> usize {
    NATIVE_EINSUM_PROFILE_STATE.with(|state| {
        state
            .borrow()
            .iter()
            .filter(|(signature, _)| signature.path == path)
            .map(|(_, entry)| entry.calls)
            .sum()
    })
}

#[test]
fn storage_native_roundtrip_dense_f64() {
    let storage = Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    assert_storage_eq(&roundtrip, &storage);
}

#[test]
fn storage_native_roundtrip_diag_densifies_at_public_bridge() {
    let storage = Storage::from_diag_col_major(vec![2.0, -1.0, 4.0], 2).unwrap();
    let native = storage_to_native_tensor(&storage, &[3, 3]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    let expected = storage.to_dense_storage(&[3, 3]);

    assert_eq!(native.shape(), &[3, 3]);
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn storage_native_roundtrip_structured_c64_densifies_at_public_bridge() {
    let storage = Storage::new_structured(
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        vec![2],
        vec![1],
        vec![0, 0],
    )
    .unwrap();
    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    let expected = storage.to_dense_storage(&[2, 2]);

    assert_eq!(native.shape(), &[2, 2]);
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn dense_native_tensor_column_major_roundtrip_preserves_linearization() {
    let native =
        dense_native_tensor_from_col_major(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let values = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn dense_native_tensor_from_col_major_c64_roundtrip() {
    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let native = dense_native_tensor_from_col_major(&data, &[2]).unwrap();
    let values = native_tensor_primal_to_dense_c64_col_major(&native).unwrap();
    assert_eq!(values, data);
}

#[test]
fn diag_native_tensor_from_col_major_f64_roundtrip() {
    let data = vec![1.0_f64, 2.0, 3.0];
    let native = diag_native_tensor_from_col_major(&data, 2).unwrap();
    let diag_values = native_tensor_primal_to_diag_f64(&native).unwrap();
    let dense = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();

    assert_eq!(diag_values, data);
    assert_eq!(
        dense,
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ]
    );
}

#[test]
fn dense_native_tensor_from_col_major_f32_roundtrip() {
    let native = dense_native_tensor_from_col_major(&[1.25_f32, -2.5_f32], &[2]).unwrap();
    let values = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();
    let snapshot = native_tensor_primal_to_storage(&native).unwrap();

    assert_eq!(values, vec![1.25, -2.5]);
    assert_storage_eq(
        &snapshot,
        &Storage::from_dense_col_major(vec![1.25, -2.5], &[2]).unwrap(),
    );
}

#[test]
fn diag_native_tensor_from_col_major_c32_promotes_to_c64_values() {
    let data = vec![Complex32::new(1.0, -0.5), Complex32::new(-2.0, 0.25)];
    let native = diag_native_tensor_from_col_major(&data, 2).unwrap();
    let diag_values = native_tensor_primal_to_diag_c64(&native).unwrap();
    let dense_values = native_tensor_primal_to_dense_c64_col_major(&native).unwrap();
    let snapshot = native_tensor_primal_to_storage(&native).unwrap();

    assert_eq!(
        diag_values,
        vec![Complex64::new(1.0, -0.5), Complex64::new(-2.0, 0.25)]
    );
    assert_eq!(
        dense_values,
        vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-2.0, 0.25),
        ]
    );
    assert_storage_eq(
        &snapshot,
        &Storage::from_dense_col_major(
            vec![
                Complex64::new(1.0, -0.5),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-2.0, 0.25),
            ],
            &[2, 2],
        )
        .unwrap(),
    );
}

#[test]
fn sum_native_tensor_returns_rank0_scalar() {
    let storage = Storage::from_dense_col_major(
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
fn sum_native_tensor_preserves_rank0_scalar() {
    let native = Complex64::scalar_native_tensor(Complex64::new(1.5, -0.25)).unwrap();
    let sum = sum_native_tensor(&native).unwrap();

    assert_eq!(sum.as_c64(), Some(Complex64::new(1.5, -0.25)));
}

#[test]
fn native_snapshot_materializes_lazy_conjugation() {
    let storage = Storage::from_dense_col_major(
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

    let expected = Storage::from_dense_col_major(
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
        &Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();
    let rhs = storage_to_native_tensor(
        &Storage::from_dense_col_major(vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0], &[3, 2]).unwrap(),
        &[3, 2],
    )
    .unwrap();

    let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[2, 1])], &[0, 2]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();

    let expected =
        Storage::from_dense_col_major(vec![50.0, 110.0, 110.0, 250.0, 170.0, 390.0], &[2, 3])
            .unwrap();
    assert_storage_eq(&snapshot, &expected);
}

#[test]
fn einsum_native_tensors_dense_binary_records_frontend_fallback_profile() {
    struct ProfileGuard;

    impl Drop for ProfileGuard {
        fn drop(&mut self) {
            set_native_einsum_profile_enabled_for_tests(false);
            reset_native_einsum_profile();
        }
    }

    reset_native_einsum_profile();
    set_native_einsum_profile_enabled_for_tests(true);
    let _guard = ProfileGuard;

    let lhs = dense_native_tensor_from_col_major(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let rhs =
        dense_native_tensor_from_col_major(&[10.0_f64, 30.0, 50.0, 20.0, 40.0, 60.0], &[3, 2])
            .unwrap();
    let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[2, 1])], &[0, 2]).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::FrontendFallback),
        1
    );
}

#[test]
fn contract_native_tensor_restores_rhs_free_axis_order() {
    let lhs = storage_to_native_tensor(
        &Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();
    let rhs = storage_to_native_tensor(
        &Storage::from_dense_col_major(vec![10.0, 30.0, 20.0, 40.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();

    let out = contract_native_tensor(&lhs, &[1], &rhs, &[1]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();
    let expected = Storage::from_dense_col_major(vec![50.0, 110.0, 110.0, 250.0], &[2, 2]).unwrap();

    assert_storage_eq(&snapshot, &expected);
}

#[test]
fn permute_storage_native_dense_matches_expected_data() {
    let storage =
        Storage::from_dense_col_major(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]).unwrap();
    let permuted = permute_storage_native(&storage, &[2, 3], &[1, 0]).unwrap();
    let expected =
        Storage::from_dense_col_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    assert_storage_eq(&permuted, &expected);
}

#[test]
fn reshape_col_major_native_tensor_handles_permuted_input() {
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

#[test]
fn scale_native_tensor_promotes_real_scalar_for_complex_tensor() {
    let native = dense_native_tensor_from_col_major(
        &[Complex64::new(1.0, 2.0), Complex64::new(-0.5, 1.0)],
        &[2],
    )
    .unwrap();
    let scalar = crate::AnyScalar::new_real(2.0);
    let scaled = scale_native_tensor(&native, &scalar).unwrap();
    let values = native_tensor_primal_to_dense_c64_col_major(&scaled).unwrap();

    assert_eq!(
        values,
        vec![Complex64::new(2.0, 4.0), Complex64::new(-1.0, 2.0)]
    );
}

#[test]
fn scale_and_axpby_native_tensor_cover_f32_paths() {
    let lhs = dense_native_tensor_from_col_major(&[1.0_f32, -2.0_f32], &[2]).unwrap();
    let rhs = dense_native_tensor_from_col_major(&[0.5_f32, 4.0_f32], &[2]).unwrap();

    let scaled = scale_native_tensor(&lhs, &crate::AnyScalar::from_value(0.5_f32)).unwrap();
    let scaled_values = native_tensor_primal_to_dense_f64_col_major(&scaled).unwrap();
    assert_eq!(scaled_values, vec![0.5, -1.0]);

    let combined = axpby_native_tensor(
        &lhs,
        &crate::AnyScalar::from_value(2.0_f32),
        &rhs,
        &crate::AnyScalar::from_value(-1.0_f32),
    )
    .unwrap();
    let combined_values = native_tensor_primal_to_dense_f64_col_major(&combined).unwrap();
    assert_eq!(combined_values, vec![1.5, -8.0]);
}

#[test]
fn axpby_native_tensor_promotes_real_scalars_for_complex_tensors() {
    let lhs = dense_native_tensor_from_col_major(
        &[Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.5)],
        &[2],
    )
    .unwrap();
    let rhs = dense_native_tensor_from_col_major(
        &[Complex64::new(0.5, -1.0), Complex64::new(2.0, 1.0)],
        &[2],
    )
    .unwrap();
    let a = crate::AnyScalar::new_real(2.0);
    let b = crate::AnyScalar::new_real(-1.0);
    let combined = axpby_native_tensor(&lhs, &a, &rhs, &b).unwrap();
    let values = native_tensor_primal_to_dense_c64_col_major(&combined).unwrap();

    assert_eq!(
        values,
        vec![Complex64::new(1.5, 5.0), Complex64::new(-4.0, 0.0)]
    );
}

#[test]
fn scale_and_axpby_storage_native_roundtrip() {
    let storage = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let scalar = crate::AnyScalar::new_real(2.0);
    let scaled = scale_storage_native(&storage, &[3], &scalar).unwrap();
    let expected_scaled = Storage::from_dense_col_major(vec![2.0, 4.0, 6.0], &[3]).unwrap();
    assert_storage_eq(&scaled, &expected_scaled);

    let rhs = Storage::from_dense_col_major(vec![3.0, 4.0, 5.0], &[3]).unwrap();
    let combined = axpby_storage_native(
        &storage,
        &[3],
        &crate::AnyScalar::new_real(2.0),
        &rhs,
        &[3],
        &crate::AnyScalar::new_real(3.0),
    )
    .unwrap();
    let expected_combined = Storage::from_dense_col_major(vec![11.0, 16.0, 21.0], &[3]).unwrap();
    assert_storage_eq(&combined, &expected_combined);
}

#[test]
fn outer_product_storage_native_produces_correct_shape() {
    let a = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_col_major(vec![3.0, 4.0, 5.0], &[3]).unwrap();
    let result = outer_product_storage_native(&a, &[2], &b, &[3], &[2, 3]).unwrap();
    let expected =
        Storage::from_dense_col_major(vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0], &[2, 3]).unwrap();
    assert_storage_eq(&result, &expected);
}

#[test]
fn qr_and_svd_native_tensor_return_expected_shapes() {
    let native = dense_native_tensor_from_col_major(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();

    let (q, r) = qr_native_tensor(&native).unwrap();
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    let (u, s, vt) = svd_native_tensor(&native).unwrap();
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);
}

#[test]
fn tangent_native_tensor_returns_none_for_primal() {
    let native = dense_native_tensor_from_col_major(&[1.0_f64, 2.0], &[2]).unwrap();
    assert!(tangent_native_tensor(&native).is_none());
}

#[test]
fn dense_and_diag_extractors_reject_wrong_dtypes() {
    let real = dense_native_tensor_from_col_major(&[1.0_f64, 2.0], &[2]).unwrap();
    let complex = dense_native_tensor_from_col_major(&[Complex64::new(1.0, -1.0)], &[1]).unwrap();

    assert!(native_tensor_primal_to_dense_f64_col_major(&complex)
        .unwrap_err()
        .to_string()
        .contains("expected real native tensor"));
    assert!(native_tensor_primal_to_dense_c64_col_major(&real)
        .unwrap_err()
        .to_string()
        .contains("expected complex native tensor"));
    assert!(native_tensor_primal_to_diag_f64(&complex)
        .unwrap_err()
        .to_string()
        .contains("expected real native tensor"));
    assert!(native_tensor_primal_to_diag_c64(&real)
        .unwrap_err()
        .to_string()
        .contains("expected complex native tensor"));
}

#[test]
fn einsum_native_tensors_rejects_empty_operands() {
    let result = einsum_native_tensors(&[], &[]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("at least one operand"));
}

#[test]
fn ids_to_subscript_and_build_einsum_subscripts_cover_helper_paths() {
    assert_eq!(ids_to_subscript(&[0, 25, 51]).unwrap(), "azZ");
    assert_eq!(
        build_einsum_subscripts(&[&[0, 1], &[2, 1]], &[0, 2]).unwrap(),
        "ab,cb->ac"
    );
    assert!(ids_to_subscript(&[52])
        .unwrap_err()
        .to_string()
        .contains("exceeds supported label range"));
}

#[test]
fn build_binary_einsum_ids_rejects_mismatched_axes_len() {
    let result = build_binary_einsum_ids(2, &[0], 2, &[0, 1]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("length mismatch"));
}

#[test]
fn build_binary_einsum_ids_success_assigns_free_axes_after_contract_axes() {
    let (lhs_ids, rhs_ids, output_ids) = build_binary_einsum_ids(3, &[1], 2, &[0]).unwrap();

    assert_eq!(lhs_ids, vec![1, 0, 2]);
    assert_eq!(rhs_ids, vec![0, 3]);
    assert_eq!(output_ids, vec![1, 2, 3]);
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
