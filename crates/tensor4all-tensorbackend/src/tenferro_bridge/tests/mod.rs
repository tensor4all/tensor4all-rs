use super::*;
use crate::storage::{Storage, StorageRepr};
use crate::tensor_element::TensorElement;
use num_complex::{Complex32, Complex64};
use tenferro::TensorView;

fn assert_storage_eq(lhs: &Storage, rhs: &Storage) {
    match (lhs.repr(), rhs.repr()) {
        (StorageRepr::F32(a), StorageRepr::F32(b)) => {
            assert_eq!(a.payload_dims(), b.payload_dims());
            assert_eq!(a.strides(), b.strides());
            assert_eq!(a.axis_classes(), b.axis_classes());
            assert_eq!(a.data(), b.data());
        }
        (StorageRepr::C32(a), StorageRepr::C32(b)) => {
            assert_eq!(a.payload_dims(), b.payload_dims());
            assert_eq!(a.strides(), b.strides());
            assert_eq!(a.axis_classes(), b.axis_classes());
            assert_eq!(a.data(), b.data());
        }
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

/// Whether the process-global graph runtime currently reports prepared plans.
///
/// The context builds its graph runtime lazily, so this also pays the first
/// initialization. The aggregate cache statistics can be unavailable while
/// another test owns a cache owner, which is the documented "management does not
/// wait on a run" behaviour rather than a failure here, so a busy owner is
/// retried briefly.
fn default_graph_runtime_has_prepared_plan_cache_entries() -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match crate::context::default_context().graph_cache_stats() {
            Ok(stats) => return stats.prepared_plans.entries > 0,
            Err(error) if std::time::Instant::now() < deadline => {
                let _ = error;
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(error) => panic!("default runtime never provided cache stats: {error}"),
        }
    }
}

struct ProfileGuard;

impl ProfileGuard {
    fn enable() -> Self {
        reset_native_einsum_profile();
        set_native_einsum_profile_enabled_for_tests(true);
        Self
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        set_native_einsum_profile_enabled_for_tests(false);
        reset_native_einsum_profile();
    }
}

#[test]
fn storage_native_roundtrip_dense_f64() {
    let storage = Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    assert_storage_eq(&roundtrip, &storage);
}

#[test]
fn storage_native_roundtrip_keeps_the_narrow_dtypes() {
    // The session route consumes compact storage through these bridges, so a narrow payload must reach
    // the session as its own dtype instead of a widened one.
    let f32_storage =
        Storage::from_dense_col_major(vec![1.5_f32, -2.5, 3.25, 4.0], &[2, 2]).unwrap();
    let native = storage_to_native_tensor(&f32_storage, &[2, 2]).unwrap();
    assert_eq!(native.dtype(), tenferro::DType::F32);
    assert_eq!(
        native.as_slice::<f32>().unwrap(),
        &[1.5_f32, -2.5, 3.25, 4.0]
    );
    assert_storage_eq(
        &native_tensor_primal_to_storage(&native).unwrap(),
        &f32_storage,
    );

    let c32_storage = Storage::from_dense_col_major(
        vec![Complex32::new(1.0, -1.0), Complex32::new(0.5, 2.0)],
        &[2],
    )
    .unwrap();
    let native = storage_to_native_tensor(&c32_storage, &[2]).unwrap();
    assert_eq!(native.dtype(), tenferro::DType::C32);
    assert_storage_eq(
        &native_tensor_primal_to_storage(&native).unwrap(),
        &c32_storage,
    );

    // The payload bridge preserves the dtype and, for a contiguous payload, borrows it.
    let payload = storage_payload_native(&f32_storage).unwrap();
    assert_eq!(payload.dtype(), tenferro::DType::F32);
    assert_eq!(payload.shape(), &[2, 2]);
    match storage_payload_native_read_input(&f32_storage).unwrap() {
        NativeTensorReadInput::Borrowed(read) => {
            assert_eq!(read.dtype(), tenferro::DType::F32);
        }
        NativeTensorReadInput::Owned(_) => panic!("a contiguous f32 payload is borrowed"),
    }

    // Rebuilding storage from a narrow native payload keeps the dtype.
    let rebuilt = storage_from_payload_native(payload, &[2, 2], vec![0, 1]).unwrap();
    assert!(rebuilt.is_f32());
    assert_storage_eq(&rebuilt, &f32_storage);
}

#[test]
fn storage_native_roundtrip_diag_densifies_at_public_bridge() {
    let storage = Storage::from_diag_col_major(vec![2.0, -1.0, 4.0], 2).unwrap();
    let native = storage_to_native_tensor(&storage, &[3, 3]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    let expected = storage.to_dense_storage(&[3, 3]).unwrap();

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
    let expected = storage.to_dense_storage(&[2, 2]).unwrap();

    assert_eq!(native.shape(), &[2, 2]);
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn dense_native_tensor_column_major_roundtrip_preserves_linearization() {
    let native =
        dense_native_tensor_from_col_major(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let values = native_tensor_primal_to_dense_col_major::<f64>(&native).unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn dense_native_tensor_from_col_major_c64_roundtrip() {
    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let native = dense_native_tensor_from_col_major(&data, &[2]).unwrap();
    let values = native_tensor_primal_to_dense_col_major::<Complex64>(&native).unwrap();
    assert_eq!(values, data);
}

#[test]
fn owned_dense_native_tensor_preserves_values_for_all_scalar_kinds() {
    fn assert_owned<T>(data: Vec<T>)
    where
        T: TensorElement + PartialEq + std::fmt::Debug,
    {
        let input_ptr = data.as_ptr();
        let native = dense_native_tensor_from_col_major_owned(data, &[2, 2]).unwrap();
        assert_eq!(native.shape(), &[2, 2]);
        assert_eq!(native.as_slice::<T>().unwrap().as_ptr(), input_ptr);
        assert_eq!(
            native_tensor_primal_to_dense_col_major::<T>(&native)
                .unwrap()
                .len(),
            4
        );
    }

    assert_owned(vec![1.0_f32, 2.0, 3.0, 4.0]);
    assert_owned(vec![1.0_f64, 2.0, 3.0, 4.0]);
    assert_owned(vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, 1.5),
        Complex32::new(3.0, 2.5),
        Complex32::new(4.0, 3.5),
    ]);
    assert_owned(vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, 1.5),
        Complex64::new(3.0, 2.5),
        Complex64::new(4.0, 3.5),
    ]);
}

#[test]
fn owned_dense_native_tensor_rejects_shape_mismatch_and_overflow() {
    let err = dense_native_tensor_from_col_major_owned(vec![1.0_f64; 3], &[2, 2]).unwrap_err();
    assert!(err.to_string().contains("does not match"), "{err}");

    let err =
        dense_native_tensor_from_col_major_owned(vec![1.0_f64; 4], &[65536, 65536, 65536, 65536])
            .unwrap_err();
    assert!(err.to_string().contains("overflow"), "{err}");
}

#[test]
#[ignore = "paired release efficiency measurement; correctness runs in the full suite"]
fn owned_dense_native_tensor_paired_release_measurement() {
    use std::hint::black_box;
    use std::time::Instant;

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        values[values.len() / 2]
    }

    let dims = [128, 128];
    let template = (0..(dims[0] * dims[1]))
        .map(|index| index as f64 * 0.001 - 1.0)
        .collect::<Vec<_>>();
    let samples = 9;
    let repetitions = 32;
    let mut borrowed_ms = Vec::with_capacity(samples);
    let mut owned_ms = Vec::with_capacity(samples);
    let mut checksum = 0.0;

    for _sample in 0..samples {
        let borrowed_started = Instant::now();
        for _ in 0..repetitions {
            let data = black_box(template.clone());
            let tensor = dense_native_tensor_from_col_major(&data, &dims).unwrap();
            checksum += black_box(tensor.as_slice::<f64>().unwrap()[0]);
        }
        borrowed_ms.push(borrowed_started.elapsed().as_secs_f64() * 1_000.0);

        let owned_started = Instant::now();
        for _ in 0..repetitions {
            let data = black_box(template.clone());
            let tensor = dense_native_tensor_from_col_major_owned(data, &dims).unwrap();
            checksum += black_box(tensor.as_slice::<f64>().unwrap()[0]);
        }
        owned_ms.push(owned_started.elapsed().as_secs_f64() * 1_000.0);
    }

    let borrowed_median = median(&mut borrowed_ms);
    let owned_median = median(&mut owned_ms);
    println!(
        "#716 paired release dense constructor: dims={dims:?}, repetitions={repetitions}, samples={samples}, borrowed_median_ms={borrowed_median:.3}, owned_median_ms={owned_median:.3}, reduction_pct={:.1}, borrowed_all_ms={borrowed_ms:?}, owned_all_ms={owned_ms:?}, checksum={checksum}",
        (1.0 - owned_median / borrowed_median) * 100.0,
    );
}

#[test]
fn diag_native_tensor_from_col_major_f64_roundtrip() {
    let data = vec![1.0_f64, 2.0, 3.0];
    let native = diag_native_tensor_from_col_major(&data, 2).unwrap();
    let diag_values = native_tensor_primal_to_diag::<f64>(&native).unwrap();
    let dense = native_tensor_primal_to_dense_col_major::<f64>(&native).unwrap();

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
    // Strict materialization preserves the scalar kind: an f32 native tensor
    // must not be readable as f64.
    assert!(native_tensor_primal_to_dense_col_major::<f64>(&native).is_err());
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f32>(&native).unwrap(),
        vec![1.25_f32, -2.5_f32]
    );
    let snapshot = native_tensor_primal_to_storage(&native).unwrap();
    // Snapshot materialization is dtype preserving: the compact payload stays
    // f32 instead of widening to f64.
    assert!(snapshot.is_f32());
    assert_storage_eq(
        &snapshot,
        &Storage::from_dense_col_major(vec![1.25_f32, -2.5], &[2]).unwrap(),
    );
}

#[test]
fn diag_native_tensor_from_col_major_overflow_returns_error() {
    // diag_len^logical_rank == 2^16^4 == 2^64 wraps to 0 in unchecked
    // arithmetic; the checked product must surface a typed error instead of
    // an empty allocation followed by an OOB panic.
    let data = vec![1.0_f64; 65536];
    let err = diag_native_tensor_from_col_major(&data, 4).unwrap_err();
    assert!(err.to_string().contains("overflow"), "{err}");
}

#[test]
fn dense_native_tensor_from_col_major_overflow_returns_error() {
    // Same wrap for the dense constructor's shape product feeding the length
    // check: dims product overflows before any allocation.
    let data = vec![1.0_f64; 4];
    let err = dense_native_tensor_from_col_major(&data, &[65536, 65536, 65536, 65536]).unwrap_err();
    assert!(err.to_string().contains("overflow"), "{err}");
}

#[test]
fn diag_native_tensor_from_col_major_zero_size_is_ok() {
    // Zero diag length is a valid edge case (empty diagonal tensor).
    let data: Vec<f64> = vec![];
    let native = diag_native_tensor_from_col_major(&data, 2).unwrap();
    assert_eq!(native.shape(), &[0, 0]);
}

#[test]
fn diag_native_tensor_from_col_major_c32_promotes_to_c64_values() {
    let data = vec![Complex32::new(1.0, -0.5), Complex32::new(-2.0, 0.25)];
    let native = diag_native_tensor_from_col_major(&data, 2).unwrap();
    let diag_values = native_tensor_primal_to_diag::<Complex64>(&native).unwrap();
    // Strict dense materialization rejects the c32 source; only the
    // promoting diag extractor converts c32 -> c64.
    assert!(native_tensor_primal_to_dense_col_major::<Complex64>(&native).is_err());
    let snapshot = native_tensor_primal_to_storage(&native).unwrap();

    assert_eq!(
        diag_values,
        vec![Complex64::new(1.0, -0.5), Complex64::new(-2.0, 0.25)]
    );

    // Snapshot materialization is dtype preserving: the compact payload stays
    // Complex32 instead of widening to Complex64.
    assert!(snapshot.is_c32());
    assert_storage_eq(
        &snapshot,
        &Storage::from_dense_col_major(
            vec![
                Complex32::new(1.0, -0.5),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(-2.0, 0.25),
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
fn einsum_native_tensors_supports_retained_shared_nary_label() {
    let a = NativeTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 11.0, 13.0])
        .expect("valid n-ary einsum test tensor a");
    let b = NativeTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("valid n-ary einsum test tensor b");
    let c = NativeTensor::from_vec_col_major(vec![2, 2], vec![11.0_f64, 13.0, 17.0, 19.0])
        .expect("valid n-ary einsum test tensor c");

    let out = einsum_native_tensors(
        &[(&a, &[0, 1]), (&b, &[0, 2]), (&c, &[0, 3])],
        &[0, 1, 2, 3],
    )
    .unwrap();
    let values = native_tensor_primal_to_dense_col_major::<f64>(&out).unwrap();

    let mut expected = vec![0.0; 24];
    for b_idx in 0..2 {
        for i_idx in 0..2 {
            for j_idx in 0..3 {
                for k_idx in 0..2 {
                    let a_offset = b_idx + 2 * i_idx;
                    let b_offset = b_idx + 2 * j_idx;
                    let c_offset = b_idx + 2 * k_idx;
                    let out_offset = b_idx + 2 * (i_idx + 2 * (j_idx + 3 * k_idx));
                    expected[out_offset] = a.as_slice::<f64>().expect("tensor a host values")
                        [a_offset]
                        * b.as_slice::<f64>().expect("tensor b host values")[b_offset]
                        * c.as_slice::<f64>().expect("tensor c host values")[c_offset];
                }
            }
        }
    }

    assert_eq!(out.shape(), &[2, 2, 3, 2]);
    assert_eq!(values, expected);
    assert_eq!(values[0], 55.0);

    let read_a = NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&a));
    let read_b = NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&b));
    let read_c = NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&c));
    let session_out = einsum_native_tensor_reads(
        &[(&read_a, &[0, 1]), (&read_b, &[0, 2]), (&read_c, &[0, 3])],
        &[0, 1, 2, 3],
    )
    .expect("session-native retained-label einsum should succeed");
    assert_eq!(session_out.shape(), out.shape());
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&session_out).unwrap(),
        values
    );
}

#[test]
fn einsum_native_subscripts_matches_known_matmul_trace_and_reduction() {
    // Column-major A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]].
    let lhs = NativeTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])
        .expect("valid lhs");
    let rhs = NativeTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0])
        .expect("valid rhs");

    let product = einsum_native_subscripts("ij,jk->ik", &[&lhs, &rhs])
        .expect("string-subscript matmul should succeed");
    assert_eq!(product.shape(), &[2, 2]);
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&product).unwrap(),
        vec![19.0, 43.0, 22.0, 50.0]
    );

    let trace =
        einsum_native_subscripts("ii->", &[&lhs]).expect("string-subscript trace should succeed");
    assert_eq!(trace.shape(), &[] as &[usize]);
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&trace).unwrap(),
        vec![5.0]
    );

    let column_sums = einsum_native_subscripts("ij->j", &[&lhs])
        .expect("string-subscript reduction should succeed");
    assert_eq!(column_sums.shape(), &[2]);
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&column_sums).unwrap(),
        vec![4.0, 6.0]
    );
}

#[test]
fn einsum_native_subscripts_matches_known_complex_matmul_and_trace() {
    // Column-major A = [[1+2i, 0], [0, 1]], B = [[1, 1], [1, 1]].
    let lhs = NativeTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .expect("valid complex lhs");
    let rhs = NativeTensor::from_vec_col_major(vec![2, 2], vec![Complex64::new(1.0, 0.0); 4])
        .expect("valid complex rhs");

    let product = einsum_native_subscripts("ij,jk->ik", &[&lhs, &rhs])
        .expect("complex string-subscript matmul should succeed");
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<Complex64>(&product).unwrap(),
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(1.0, 0.0),
        ]
    );

    let trace = einsum_native_subscripts("ii->", &[&lhs])
        .expect("complex string-subscript trace should succeed");
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<Complex64>(&trace).unwrap(),
        vec![Complex64::new(2.0, 2.0)]
    );
}

#[test]
fn einsum_native_subscripts_rejects_operand_count_mismatch_and_parse_errors() {
    let lhs = NativeTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])
        .expect("valid lhs");

    let too_few = einsum_native_subscripts("ij,jk->ik", &[&lhs]).unwrap_err();
    let message = too_few.to_string();
    assert!(message.contains("2 input term(s)"), "unexpected: {message}");
    assert!(message.contains("1 operand(s)"), "unexpected: {message}");

    let too_many = einsum_native_subscripts("ij->ij", &[&lhs, &lhs]).unwrap_err();
    let message = too_many.to_string();
    assert!(message.contains("1 input term(s)"), "unexpected: {message}");
    assert!(message.contains("2 operand(s)"), "unexpected: {message}");

    let parse = einsum_native_subscripts("ij,jk", &[&lhs, &lhs]).unwrap_err();
    assert!(
        parse.to_string().contains("invalid einsum subscripts"),
        "unexpected: {parse}"
    );
}

#[test]
fn einsum_native_tensor_reads_accepts_non_contiguous_borrowed_view() {
    reset_concrete_einsum_plan_cache();
    let backing = [1.0_f64, 2.0, 3.0, 4.0];
    let lhs_view = TensorView::F64(
        tenferro::TypedTensorView::from_slice(vec![2, 2], vec![2, 1], 0, &backing)
            .expect("valid non-contiguous lhs view"),
    );
    assert!(!lhs_view
        .is_col_major_contiguous()
        .expect("view layout should be valid"));
    let lhs = NativeTensorReadInput::Borrowed(TensorRead::from_view(lhs_view));
    let rhs_tensor = NativeTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])
        .expect("valid identity rhs");
    let rhs = NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&rhs_tensor));

    let contiguous_lhs_tensor =
        NativeTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])
            .expect("valid contiguous lhs");
    let contiguous_lhs =
        NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&contiguous_lhs_tensor));
    let output =
        einsum_native_tensor_reads(&[(&contiguous_lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2])
            .expect("borrowed contiguous einsum should succeed");
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&output).unwrap(),
        vec![1.0, 3.0, 2.0, 4.0]
    );
    assert_eq!(concrete_einsum_plan_cache_len(), 1);

    let repeated = einsum_native_tensor_reads(&[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2])
        .expect("repeated borrowed einsum should reuse its session plan");
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&repeated).unwrap(),
        vec![1.0, 3.0, 2.0, 4.0]
    );
    assert_eq!(concrete_einsum_plan_cache_len(), 1);
}

#[test]
fn einsum_native_tensor_reads_promotes_non_contiguous_borrowed_view() {
    reset_concrete_einsum_plan_cache();
    let backing = [1.0_f64, 2.0, 3.0, 4.0];
    let lhs_view = TensorView::F64(
        tenferro::TypedTensorView::from_slice(vec![2, 2], vec![2, 1], 0, &backing)
            .expect("valid non-contiguous lhs view"),
    );
    let lhs = NativeTensorReadInput::Borrowed(TensorRead::from_view(lhs_view));
    let rhs_tensor = NativeTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .expect("valid complex identity rhs");
    let rhs = NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&rhs_tensor));

    let output = einsum_native_tensor_reads(&[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2])
        .expect("mixed borrowed non-contiguous einsum should succeed");

    assert_eq!(
        native_tensor_primal_to_dense_col_major::<Complex64>(&output).unwrap(),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]
    );
    assert_eq!(concrete_einsum_plan_cache_len(), 0);
}

#[test]
fn same_integer_dtype_retains_graph_promotion() {
    reset_concrete_einsum_plan_cache();
    let lhs_tensor =
        NativeTensor::from_vec_col_major(vec![1, 2], vec![2_i64, 3]).expect("valid integer lhs");
    let rhs_tensor =
        NativeTensor::from_vec_col_major(vec![2, 1], vec![5_i64, 7]).expect("valid integer rhs");
    let lhs = NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&lhs_tensor));
    let rhs = NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&rhs_tensor));

    let output = einsum_native_tensor_reads(&[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2])
        .expect("integer graph fallback should promote and contract");

    assert_eq!(output.dtype(), DType::F64);
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&output).unwrap(),
        vec![31.0]
    );
    assert_eq!(concrete_einsum_plan_cache_len(), 0);
}

#[test]
fn concrete_einsum_plan_cache_clears_at_its_bound() {
    reset_concrete_einsum_plan_cache();
    let tensor = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .expect("valid bounded-cache tensor");
    let input = NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&tensor));

    for label in 0..=CONCRETE_EINSUM_PLAN_CACHE_CAPACITY {
        let output = einsum_native_tensor_reads(&[(&input, &[label])], &[label])
            .expect("single-input session plan should execute");
        assert_eq!(
            native_tensor_primal_to_dense_col_major::<f64>(&output).unwrap(),
            vec![1.0, 2.0]
        );
    }

    assert_eq!(concrete_einsum_plan_cache_len(), 1);
}

#[test]
fn einsum_native_tensors_populates_process_global_prepared_plan_cache() {
    let a = NativeTensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24])
        .expect("valid cache einsum test tensor a");
    let b = NativeTensor::from_vec_col_major(vec![4, 5], vec![2.0_f64; 20])
        .expect("valid cache einsum test tensor b");
    let c = NativeTensor::from_vec_col_major(vec![3, 2], vec![3.0_f64; 6])
        .expect("valid cache einsum test tensor c");

    let out =
        einsum_native_tensors(&[(&a, &[0, 1, 2]), (&b, &[2, 3]), (&c, &[1, 0])], &[3]).unwrap();

    assert_eq!(out.shape(), &[5]);
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&out).unwrap(),
        vec![144.0; 5]
    );
    assert!(default_graph_runtime_has_prepared_plan_cache_entries());
}

#[test]
fn einsum_native_tensors_mixed_dtype_records_borrowed_conversion_profile() {
    let _guard = ProfileGuard::enable();
    let lhs = NativeTensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0])
        .expect("valid mixed-dtype einsum lhs");
    let rhs = NativeTensor::from_vec_col_major(vec![2, 3], vec![5.0_f64, 6.0, 7.0, 8.0, 9.0, 10.0])
        .expect("valid mixed-dtype einsum rhs");

    let borrowed = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2]).unwrap();
    let owned =
        einsum_native_tensors_owned(vec![(lhs, vec![0, 1]), (rhs, vec![1, 2])], &[0, 2]).unwrap();

    assert_eq!(owned.shape(), &[2, 3]);
    assert_eq!(owned.dtype(), DType::F64);
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&owned).unwrap(),
        native_tensor_primal_to_dense_col_major::<f64>(&borrowed).unwrap()
    );
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::BorrowedWithConversions),
        1
    );
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::Borrowed),
        0
    );
}

#[test]
fn einsum_native_tensors_dense_binary_records_borrowed_profile() {
    let _guard = ProfileGuard::enable();

    let lhs = dense_native_tensor_from_col_major(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let rhs =
        dense_native_tensor_from_col_major(&[10.0_f64, 30.0, 50.0, 20.0, 40.0, 60.0], &[3, 2])
            .unwrap();
    let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[2, 1])], &[0, 2]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();
    let expected =
        Storage::from_dense_col_major(vec![50.0, 110.0, 110.0, 250.0, 170.0, 390.0], &[2, 3])
            .unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_storage_eq(&snapshot, &expected);
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::Borrowed),
        1
    );
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::BorrowedWithConversions),
        0
    );
}

#[test]
fn native_read_input_owned_and_plan_helpers_cover_debug_paths() {
    let tensor = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .expect("valid native read-input test tensor");
    let tensor_bytes = native_tensor_bytes(&tensor);
    let input = NativeTensorReadInput::Owned(tensor);
    assert_eq!(input.dtype(), DType::F64);
    assert_eq!(input.shape(), &[2]);
    assert_eq!(input.as_read().shape(), &[2]);

    assert_eq!(native_einsum_path_trace_min_bytes(), 0);
    assert_eq!(native_einsum_path_trace_max_signatures(), 64);
    assert_eq!(native_einsum_pool_trace_min_output_bytes(), 0);
    assert_eq!(native_einsum_pool_trace_min_retained_bytes(), 0);

    assert_eq!(dtype_size_bytes(DType::F32), 4);
    assert_eq!(dtype_size_bytes(DType::F64), 8);
    assert_eq!(dtype_size_bytes(DType::C32), 8);
    assert_eq!(dtype_size_bytes(DType::C64), 16);
    assert_eq!(dtype_size_bytes(DType::I64), 8);
    assert_eq!(tensor_bytes, 16);
    assert_eq!(format_label('x' as u32), "x");
    assert_eq!(format_label(0x110000), "1114112");
    assert_eq!(format_labels(&[]), "scalar");
    assert_eq!(format_labels(&['a' as u32, 'b' as u32]), "ab");

    let subscripts = Subscripts {
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
    };
    let dims = label_dims(&subscripts, &[vec![2, 3], vec![3, 4]]).unwrap();
    assert_eq!(labels_size(&[0, 2], &dims), 8);
    assert_eq!(union_labels(&[0, 1], &[1, 2]), vec![0, 1, 2]);

    let bad_len = label_dims(&subscripts, &[vec![2], vec![3, 4]]).unwrap_err();
    assert!(bad_len.to_string().contains("do not match shape"));
    let bad_dim = label_dims(&subscripts, &[vec![2, 3], vec![4, 4]]).unwrap_err();
    assert!(bad_dim.to_string().contains("inconsistent dimension"));

    let signature = NativeEinsumSignature {
        path: NativeEinsumPath::Borrowed,
        operands: vec![
            NativeOperandSignature {
                shape: vec![2, 3],
                ids: vec![0, 1],
                dtype: DType::F64,
            },
            NativeOperandSignature {
                shape: vec![3, 4],
                ids: vec![1, 2],
                dtype: DType::F64,
            },
        ],
        output_ids: vec![0, 2],
    };
    let time_report = native_einsum_time_optimized_plan_report(&signature).unwrap();
    let balanced_report = native_einsum_balanced_plan_report(&signature).unwrap();
    assert!(time_report.peak_intermediate_bytes >= 8);
    assert!(!time_report.lines.is_empty());
    assert!(!balanced_report.lines.is_empty());
}

#[test]
fn native_einsum_profile_print_and_c32_arithmetic_paths() {
    let _guard = ProfileGuard::enable();

    let lhs = NativeTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 2.0), Complex32::new(-3.0, 0.5)],
    )
    .expect("valid complex32 scale lhs");
    let rhs = NativeTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(0.5, -1.0), Complex32::new(4.0, 2.0)],
    )
    .expect("valid complex32 scale rhs");

    let scaled = scale_native_tensor(
        &lhs,
        &crate::BackendScalar::from_value(Complex32::new(2.0, -1.0)),
    )
    .unwrap();
    assert_eq!(scaled.dtype(), DType::C32);
    assert_eq!(
        scaled
            .as_slice::<Complex32>()
            .expect("scaled tensor host values"),
        &[Complex32::new(4.0, 3.0), Complex32::new(-5.5, 4.0)]
    );

    let combined = axpby_native_tensor(
        &lhs,
        &crate::BackendScalar::from_value(Complex32::new(1.0, 0.0)),
        &rhs,
        &crate::BackendScalar::from_value(Complex32::new(0.0, 1.0)),
    )
    .unwrap();
    assert_eq!(combined.dtype(), DType::C32);
    assert_eq!(
        combined
            .as_slice::<Complex32>()
            .expect("combined tensor host values"),
        &[Complex32::new(2.0, 2.5), Complex32::new(-5.0, 4.5)]
    );

    let contraction = einsum_native_tensors(&[(&lhs, &[0]), (&rhs, &[0])], &[]).unwrap();
    assert_eq!(contraction.shape(), &[] as &[usize]);
    print_and_reset_native_einsum_profile();
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::Borrowed),
        0
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
    let permuted_values = native_tensor_primal_to_dense_col_major::<f64>(&permuted).unwrap();

    let reshaped = reshape_col_major_native_tensor(&permuted, &[4, 6]).unwrap();
    let reshaped_values = native_tensor_primal_to_dense_col_major::<f64>(&reshaped).unwrap();

    assert_eq!(reshaped_values, permuted_values);
}

#[test]
fn scale_native_tensor_promotes_real_scalar_for_complex_tensor() {
    let native = dense_native_tensor_from_col_major(
        &[Complex64::new(1.0, 2.0), Complex64::new(-0.5, 1.0)],
        &[2],
    )
    .unwrap();
    let scalar = crate::BackendScalar::new_real(2.0);
    let scaled = scale_native_tensor(&native, &scalar).unwrap();
    let values = native_tensor_primal_to_dense_col_major::<Complex64>(&scaled).unwrap();

    assert_eq!(
        values,
        vec![Complex64::new(2.0, 4.0), Complex64::new(-1.0, 2.0)]
    );
}

#[test]
fn scale_and_axpby_native_tensor_cover_f32_paths() {
    let lhs = dense_native_tensor_from_col_major(&[1.0_f32, -2.0_f32], &[2]).unwrap();
    let rhs = dense_native_tensor_from_col_major(&[0.5_f32, 4.0_f32], &[2]).unwrap();

    let scaled = scale_native_tensor(&lhs, &crate::BackendScalar::from_value(0.5_f32)).unwrap();
    let scaled_values = native_tensor_primal_to_dense_col_major::<f32>(&scaled).unwrap();
    assert_eq!(scaled_values, vec![0.5_f32, -1.0_f32]);

    let combined = axpby_native_tensor(
        &lhs,
        &crate::BackendScalar::from_value(2.0_f32),
        &rhs,
        &crate::BackendScalar::from_value(-1.0_f32),
    )
    .unwrap();
    let combined_values = native_tensor_primal_to_dense_col_major::<f32>(&combined).unwrap();
    assert_eq!(combined_values, vec![1.5_f32, -8.0_f32]);
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
    let a = crate::BackendScalar::new_real(2.0);
    let b = crate::BackendScalar::new_real(-1.0);
    let combined = axpby_native_tensor(&lhs, &a, &rhs, &b).unwrap();
    let values = native_tensor_primal_to_dense_col_major::<Complex64>(&combined).unwrap();

    assert_eq!(
        values,
        vec![Complex64::new(1.5, 5.0), Complex64::new(-4.0, 0.0)]
    );
}

#[test]
fn scale_and_axpby_storage_native_roundtrip() {
    let storage = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let scalar = crate::BackendScalar::new_real(2.0);
    let scaled = scale_storage_native(&storage, &[3], &scalar).unwrap();
    let expected_scaled = Storage::from_dense_col_major(vec![2.0, 4.0, 6.0], &[3]).unwrap();
    assert_storage_eq(&scaled, &expected_scaled);

    let rhs = Storage::from_dense_col_major(vec![3.0, 4.0, 5.0], &[3]).unwrap();
    let combined = axpby_storage_native(
        &storage,
        &[3],
        &crate::BackendScalar::new_real(2.0),
        &rhs,
        &[3],
        &crate::BackendScalar::new_real(3.0),
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

    assert!(native_tensor_primal_to_dense_col_major::<f64>(&complex)
        .unwrap_err()
        .to_string()
        .contains("expected real native tensor"));
    assert!(native_tensor_primal_to_dense_col_major::<Complex64>(&real)
        .unwrap_err()
        .to_string()
        .contains("expected complex native tensor"));
    assert!(native_tensor_primal_to_diag::<f64>(&complex)
        .unwrap_err()
        .to_string()
        .contains("expected real native tensor"));
    assert!(native_tensor_primal_to_diag::<Complex64>(&real)
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
fn native_einsum_rejects_oversized_labels_on_all_input_paths() {
    let oversized = u32::MAX as usize + 1;
    let owned_tensor = NativeTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let owned = einsum_native_tensors_owned(vec![(owned_tensor, vec![oversized])], &[0]);
    assert!(owned.unwrap_err().to_string().contains("u32 range"));

    let borrowed_tensor = NativeTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let borrowed = einsum_native_tensors(&[(&borrowed_tensor, &[oversized])], &[0]);
    assert!(borrowed.unwrap_err().to_string().contains("u32 range"));

    let read_tensor = NativeTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let read = NativeTensorReadInput::Owned(read_tensor);
    let read_result = einsum_native_tensor_reads(&[(&read, &[oversized])], &[0]);
    assert!(read_result.unwrap_err().to_string().contains("u32 range"));
}

#[test]
fn native_einsum_rejects_oversized_output_labels() {
    let oversized = u32::MAX as usize + 1;
    let tensor = NativeTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();

    let result = einsum_native_tensors(&[(&tensor, &[0])], &[oversized]);
    assert!(result.unwrap_err().to_string().contains("u32 range"));
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

#[test]
fn concrete_einsum_plan_cache_keeps_issue_891_working_set_resident() {
    reset_concrete_einsum_plan_cache();
    // 400 distinct A[i,j] B[j,k] C[k,l] complex shapes: the churn cycle from
    // issue #891, which the old 256-entry clear-on-full bound re-prepared on
    // every call. Two passes must keep all 400 plans resident.
    const SIGNATURES: usize = 400;
    let tensors = (0..SIGNATURES)
        .map(|k| {
            let shape = [
                2 + k % 7,
                2 + (k / 7) % 7,
                2 + (k / 49) % 7,
                2 + (k / 343) % 7,
            ];
            [
                complex_test_tensor(&[shape[0], shape[1]], 0),
                complex_test_tensor(&[shape[1], shape[2]], 1),
                complex_test_tensor(&[shape[2], shape[3]], 2),
            ]
        })
        .collect::<Vec<_>>();
    for _ in 0..2 {
        for t in &tensors {
            let reads = [
                NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&t[0])),
                NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&t[1])),
                NativeTensorReadInput::Borrowed(TensorRead::from_tensor(&t[2])),
            ];
            einsum_native_tensor_reads(
                &[
                    (&reads[0], &[0, 1]),
                    (&reads[1], &[1, 2]),
                    (&reads[2], &[2, 3]),
                ],
                &[0, 3],
            )
            .expect("working-set contraction should succeed");
        }
    }
    assert_eq!(concrete_einsum_plan_cache_len(), SIGNATURES);
}

fn complex_test_tensor(shape: &[usize], seed: usize) -> NativeTensor {
    let len: usize = shape.iter().product();
    let values = (0..len)
        .map(|index| Complex64::new(((index + seed) % 13) as f64 * 0.1, 0.0))
        .collect::<Vec<_>>();
    NativeTensor::from_vec_col_major(shape.to_vec(), values).expect("valid complex test tensor")
}
