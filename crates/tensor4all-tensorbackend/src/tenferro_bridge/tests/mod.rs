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

    let expected = Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn storage_native_roundtrip_diag_densifies_at_public_bridge() {
    let storage = Storage::from_diag_col_major(vec![2.0, -1.0, 4.0], 2).unwrap();

    let native = storage_to_native_tensor(&storage, &[3, 3]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();

    assert!(!native.is_diag());
    let expected = Storage::from_dense_col_major(
        vec![
            2.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, //
            0.0, 0.0, 4.0,
        ],
        &[3, 3],
    )
    .unwrap();
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn native_dense_materialization_sets_default_runtime_if_needed() {
    reset_runtime_caches_for_tests();
    let native = storage_to_native_tensor(
        &Storage::from_diag_col_major(vec![2.0, -1.0, 4.0], 2).unwrap(),
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
    assert_eq!(default_runtime_install_count_for_tests(), 1);
}

#[test]
fn native_dense_materialization_reinstalls_scoped_default_runtime_per_call() {
    reset_runtime_caches_for_tests();
    let native = storage_to_native_tensor(
        &Storage::from_diag_col_major(vec![2.0, -1.0, 4.0], 2).unwrap(),
        &[3, 3],
    )
    .unwrap();

    let first = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();
    let installs_after_first = default_runtime_install_count_for_tests();
    let second = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();

    assert_eq!(first, second);
    assert_eq!(installs_after_first, 1);
    assert_eq!(default_runtime_install_count_for_tests(), 2);
}

#[test]
fn native_dense_materialization_survives_external_runtime_scope_changes() {
    reset_runtime_caches_for_tests();
    let native = storage_to_native_tensor(
        &Storage::from_diag_col_major(vec![2.0, -1.0, 4.0], 2).unwrap(),
        &[3, 3],
    )
    .unwrap();

    let outer_guard = tenferro::set_default_runtime(tenferro::RuntimeContext::Cpu(
        tenferro_prims::CpuContext::new(2),
    ));
    let first = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();
    drop(outer_guard);

    let second = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();

    assert_eq!(first, second);
}

#[test]
fn storage_native_roundtrip_structured_densifies_at_public_bridge() {
    let storage = Storage::new_structured(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        vec![2, 2],
        vec![1, 2],
        vec![0, 1, 1],
    )
    .unwrap();

    let native = storage_to_native_tensor(&storage, &[2, 2, 2]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    let expected = storage.to_dense_storage(&[2, 2, 2]);

    assert_eq!(native.dims(), &[2, 2, 2]);
    assert!(native.is_dense());
    assert_storage_eq(&roundtrip, &expected);
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
fn native_einsum_promotes_detached_real_operands_for_complex_result() {
    let lhs = dense_native_tensor_from_col_major(
        &[
            Complex64::new(1.0, 2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, -1.0),
        ],
        &[2, 2],
    )
    .unwrap();
    let rhs = dense_native_tensor_from_col_major(&[2.0_f64, -1.0], &[2]).unwrap();

    let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[1])], &[0]).unwrap();
    let values = native_tensor_primal_to_dense_c64_col_major(&out).unwrap();

    assert_eq!(
        values,
        vec![Complex64::new(2.0, 4.0), Complex64::new(-3.0, 1.0)]
    );
}

#[test]
fn einsum_native_tensors_dense_primal_binary_routes_through_frontend_fallback() {
    struct ProfileGuard;

    impl Drop for ProfileGuard {
        fn drop(&mut self) {
            set_native_einsum_profile_enabled_for_tests(false);
            reset_native_einsum_profile();
        }
    }

    reset_runtime_caches_for_tests();
    reset_native_einsum_profile();
    set_native_einsum_profile_enabled_for_tests(true);
    let _guard = ProfileGuard;
    let lhs = dense_native_tensor_from_col_major(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let rhs =
        dense_native_tensor_from_col_major(&[10.0_f64, 30.0, 50.0, 20.0, 40.0, 60.0], &[3, 2])
            .unwrap();

    let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[2, 1])], &[0, 2]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();
    let expected =
        Storage::from_dense_col_major(vec![50.0, 110.0, 110.0, 250.0, 170.0, 390.0], &[2, 3])
            .unwrap();

    assert_eq!(out.dims(), &[2, 3]);
    assert_storage_eq(&snapshot, &expected);
    assert_eq!(cpu_context_install_count_for_tests(), 0);
    assert_eq!(default_runtime_install_count_for_tests(), 1);
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::FrontendFallback),
        1
    );
}

#[test]
fn contract_native_tensor_restores_rhs_free_axis_order() {
    reset_runtime_caches_for_tests();
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
fn with_tenferro_ctx_reuses_cached_cpu_context() {
    reset_runtime_caches_for_tests();

    let a = tenferro_tensor::Tensor::<f64>::from_slice(
        &[1.0, 3.0, 2.0, 4.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let first = with_tenferro_ctx("test_qr", |ctx| {
        tenferro_linalg::qr(ctx, &a).map_err(|e| anyhow::anyhow!("qr failed: {e}"))
    })
    .unwrap();
    let installs_after_first = cpu_context_install_count_for_tests();
    let second = with_tenferro_ctx("test_qr", |ctx| {
        tenferro_linalg::qr(ctx, &a).map_err(|e| anyhow::anyhow!("qr failed: {e}"))
    })
    .unwrap();

    assert_eq!(first.q.dims(), second.q.dims());
    assert_eq!(first.r.dims(), second.r.dims());
    assert_eq!(cpu_context_install_count_for_tests(), installs_after_first);
    assert_eq!(installs_after_first, 1);
}

#[test]
fn with_tenferro_ctx_allows_same_thread_nested_calls() {
    reset_runtime_caches_for_tests();

    let nested = std::panic::catch_unwind(|| {
        with_tenferro_ctx("outer", |_outer| {
            with_tenferro_ctx("inner", |_inner| Ok::<_, anyhow::Error>(()))
        })
    });

    assert!(nested.is_ok(), "nested with_tenferro_ctx panicked");
    assert!(nested.unwrap().is_ok());
}

#[test]
fn cpu_threads_uses_tenferro_default_when_env_is_unset() {
    assert_eq!(
        cpu_threads_from_env_value(None).unwrap(),
        CpuContext::default_num_threads()
    );
}

#[test]
fn cpu_threads_rejects_zero_override() {
    let err = cpu_threads_from_env_value(Some("0")).expect_err("zero override must be rejected");
    assert!(err.to_string().contains("T4A_TENFERRO_CPU_THREADS"));
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
        Storage::from_dense_col_major(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]).unwrap();

    let native = permute_storage_native(&storage, &[2, 3], &[1, 0]).unwrap();

    let expected =
        Storage::from_dense_col_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
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
    let a = Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let b = Storage::from_dense_col_major(vec![5.0, 7.0, 6.0, 8.0], &[2, 2]).unwrap();

    let result = contract_storage_native(&a, &[2, 2], &[1], &b, &[2, 2], &[0], &[2, 2]).unwrap();

    // A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = A@B = [[19,22],[43,50]]
    // col-major: [19, 43, 22, 50]
    let expected = Storage::from_dense_col_major(vec![19.0, 43.0, 22.0, 50.0], &[2, 2]).unwrap();
    assert_storage_eq(&result, &expected);
}

// ===== outer_product_storage_native =====

#[test]
fn outer_product_storage_native_produces_correct_shape() {
    let a = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_col_major(vec![3.0, 4.0, 5.0], &[3]).unwrap();

    let result = outer_product_storage_native(&a, &[2], &b, &[3], &[2, 3]).unwrap();

    // a[i] * b[j]: col-major [2,3]
    // [1*3, 2*3, 1*4, 2*4, 1*5, 2*5] = [3, 6, 4, 8, 5, 10]
    let expected =
        Storage::from_dense_col_major(vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0], &[2, 3]).unwrap();
    assert_storage_eq(&result, &expected);
}

// ===== scale_storage_native =====

#[test]
fn scale_storage_native_scales_elements() {
    let s = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let scalar = crate::AnyScalar::new_real(2.0);

    let result = scale_storage_native(&s, &[3], &scalar).unwrap();

    let expected = Storage::from_dense_col_major(vec![2.0, 4.0, 6.0], &[3]).unwrap();
    assert_storage_eq(&result, &expected);
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

// ===== axpby_storage_native =====

#[test]
fn axpby_storage_native_linear_combination() {
    let lhs = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let rhs = Storage::from_dense_col_major(vec![3.0, 4.0], &[2]).unwrap();
    let a = crate::AnyScalar::new_real(2.0);
    let b = crate::AnyScalar::new_real(3.0);

    let result = axpby_storage_native(&lhs, &[2], &a, &rhs, &[2], &b).unwrap();

    // 2*1 + 3*3 = 11, 2*2 + 3*4 = 16
    let expected = Storage::from_dense_col_major(vec![11.0, 16.0], &[2]).unwrap();
    assert_storage_eq(&result, &expected);
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

// ===== native_tensor_primal_to_diag =====

#[test]
fn native_tensor_primal_to_diag_f64_extracts_diagonal() {
    let storage = Storage::from_diag_col_major(vec![2.0, -1.0, 4.0], 2).unwrap();
    let native = storage_to_native_tensor(&storage, &[3, 3]).unwrap();

    let diag_values = native_tensor_primal_to_diag_f64(&native).unwrap();

    assert_eq!(diag_values, vec![2.0, -1.0, 4.0]);
}

#[test]
fn native_tensor_primal_to_diag_c64_extracts_diagonal() {
    let storage =
        Storage::from_diag_col_major(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)], 2)
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

    assert!(!native.is_diag());
    let diag_values = native_tensor_primal_to_diag_f64(&native).unwrap();
    assert_eq!(diag_values, data);
    let dense = native_tensor_primal_to_dense_f64_col_major(&native).unwrap();
    assert_eq!(
        dense,
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ]
    );
}

// ===== structured storage roundtrip for c64 =====

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

    assert_eq!(native.dims(), &[2, 2]);
    assert!(native.is_dense());
    assert_storage_eq(&roundtrip, &expected);
}

// ===== sum_native_tensor for f64 =====

#[test]
fn sum_native_tensor_returns_f64_scalar() {
    let storage = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
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

#[test]
fn qr_native_tensor_dense_primal_uses_default_runtime_path() {
    reset_runtime_caches_for_tests();
    let native = dense_native_tensor_from_col_major(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();

    let (q, r) = qr_native_tensor(&native).unwrap();

    assert_eq!(q.dims(), &[2, 2]);
    assert_eq!(r.dims(), &[2, 2]);
    assert_eq!(cpu_context_install_count_for_tests(), 0);
    assert_eq!(default_runtime_install_count_for_tests(), 1);
}

#[test]
fn svd_native_tensor_dense_primal_uses_default_runtime_path() {
    reset_runtime_caches_for_tests();
    let native = dense_native_tensor_from_col_major(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();

    let (u, s, vt) = svd_native_tensor(&native).unwrap();

    assert_eq!(u.dims(), &[2, 2]);
    assert_eq!(s.dims(), &[2]);
    assert_eq!(vt.dims(), &[2, 2]);
    assert_eq!(cpu_context_install_count_for_tests(), 0);
    assert_eq!(default_runtime_install_count_for_tests(), 1);
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
