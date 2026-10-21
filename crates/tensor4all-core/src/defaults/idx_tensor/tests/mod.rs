use super::*;
use num_complex::{Complex32, Complex64};
use std::cell::Cell;
use tensor4all_tensorbackend::StorageError;

/// `norm_squared` of `x`, and the eager-payload reads it took.
fn norm_squared_and_reads(x: &IdxTensor) -> (f64, usize) {
    EAGER_PAYLOAD_READS.with(|reads| reads.set(0));
    let value = x.norm_squared().unwrap();
    (value, EAGER_PAYLOAD_READS.with(Cell::get))
}

#[test]
fn norm_squared_reads_a_dense_eager_payload_once() {
    let data = (0..12)
        .map(|i| Complex64::new(i as f64, 1.0))
        .collect::<Vec<_>>();
    let x = IdxTensor::from_dense(vec![DynIndex::new_dyn(3), DynIndex::new_dyn(4)], data).unwrap();
    let expected = (0..12).map(|i| (i * i) as f64 + 1.0).sum::<f64>();
    let (value, reads) = norm_squared_and_reads(&x);
    assert!((value - expected).abs() <= 1e-12 * expected);
    // Once per call, not once per element.
    assert_eq!(reads, 1);
}

#[test]
fn norm_squared_reads_a_diagonal_payload_once() {
    let x = IdxTensor::from_diag(
        vec![DynIndex::new_dyn(3), DynIndex::new_dyn(3)],
        vec![1.0_f64, 2.0, 3.0],
    )
    .unwrap();
    let (value, reads) = norm_squared_and_reads(&x);
    assert!((value - 14.0).abs() <= 1e-12);
    assert_eq!(reads, 1);
}

#[cfg(not(feature = "tenferro-cuda"))]
#[test]
fn unsupported_context_error_is_classifiable_and_actionable() {
    let error = unsupported_execution_context_error();
    match &error {
        IdxTensorError::UnsupportedExecutionContext { required_feature } => {
            assert_eq!(*required_feature, "tensor4all-core/tenferro-cuda");
        }
        other => panic!("unexpected error: {other}"),
    }
    let message = error.to_string();
    assert!(
        message.contains("tensor4all-core/tenferro-cuda"),
        "diagnostic must name the required feature: {message}"
    );
}

#[cfg(not(feature = "tenferro-cuda"))]
#[test]
fn unsupported_context_error_survives_factorize_wrapping() {
    let wrapped = unsupported_factorize_context_error();
    let FactorizeError::ComputationError(inner) = &wrapped else {
        panic!("unexpected error: {wrapped}");
    };
    assert!(inner.downcast_ref::<IdxTensorError>().is_some());
    assert!(
        std::error::Error::source(&wrapped)
            .and_then(|source| source.downcast_ref::<IdxTensorError>())
            .is_some(),
        "the typed error must stay reachable through the source chain"
    );
}

#[cfg(not(feature = "tenferro-cuda"))]
#[test]
fn supported_cpu_context_check_does_not_initialize_runtime() {
    let cpu = Arc::new(tensor4all_tensorbackend::CpuExecutionContext::from_backend(
        tensor4all_tensorbackend::CpuBackend::new(),
    ));
    let context = ExecutionContext::Cpu(Arc::clone(&cpu));

    assert!(format!("{cpu:?}").contains("eager_initialized: false"));
    ensure_supported_execution_context(&context).unwrap();
    assert!(
        format!("{cpu:?}").contains("eager_initialized: false"),
        "the support check must not initialise the eager runtime"
    );
}

#[test]
fn structured_contraction_does_not_install_logical_dense_cache() {
    let n = 8;
    let left = DynIndex::new_dyn(n);
    let site = DynIndex::new_dyn(3);
    let right = DynIndex::new_dyn(n);
    let far = DynIndex::new_dyn(n);
    let end = DynIndex::new_dyn(n);
    let a = IdxTensor::from_copy_selector(left, site.clone(), right.clone(), 1, 1.0_f64).unwrap();
    let b = IdxTensor::from_copy_selector(right, site.clone(), far.clone(), 1, 2.0_f64).unwrap();
    let c = IdxTensor::from_copy_selector(far, site.clone(), end, 1, 3.0_f64).unwrap();
    let result = crate::defaults::contract::contract_with_options(
        &[&a, &b, &c],
        crate::defaults::contract::ContractionOptions::new()
            .with_retain_indices(std::slice::from_ref(&site)),
    )
    .unwrap();

    assert_eq!(result.storage_kind(), StorageKind::Structured);
    assert!(result.eager_cache.get().is_none());
    assert_eq!(result.storage().unwrap().payload_len(), n * 3);
}

#[test]
fn binary_contraction_axis_classes_preserve_uncontracted_order() {
    let axis_classes =
        IdxTensor::binary_contraction_axis_classes(&[0, 1, 0], &[1], &[0, 1], &[0]).unwrap();

    assert_eq!(axis_classes, vec![0, 0, 1]);
}

#[test]
fn structured_metrics_use_authoritative_compact_payload_for_all_dtypes() {
    fn check(tensor: IdxTensor, expected_sum: f64, expected_norm_squared: f64) {
        assert!(matches!(tensor.storage, IdxTensorStorage::Compact(_)));
        assert!(tensor.eager_cache.get().is_none());
        assert!((tensor.sum().unwrap().real() - expected_sum).abs() < 1.0e-6);
        assert!((tensor.norm_squared().unwrap() - expected_norm_squared).abs() < 1.0e-6);
        assert!((tensor.maxabs().unwrap() - 2.0).abs() < 1.0e-6);
        assert!(tensor.isapprox(&tensor, 0.0, 0.0).unwrap());
        assert!(tensor.eager_cache.get().is_none());
    }

    let indices = || vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    check(
        IdxTensor::from_diag(indices(), vec![1.0_f32, 2.0]).unwrap(),
        3.0,
        5.0,
    );
    check(
        IdxTensor::from_diag(indices(), vec![1.0_f64, 2.0]).unwrap(),
        3.0,
        5.0,
    );
    check(
        IdxTensor::from_diag(
            indices(),
            vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
        )
        .unwrap(),
        3.0,
        5.0,
    );
    check(
        IdxTensor::from_diag(
            indices(),
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        )
        .unwrap(),
        3.0,
        5.0,
    );
}

#[test]
fn payload_storage_error_retains_typed_source() {
    let storage = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    let error = storage.scalar_at(&[2]).unwrap_err();
    assert!(matches!(error, StorageError::InvalidStructuredStorage(_)));
}

#[test]
fn materialization_error_retains_backend_source() {
    let native = NativeTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let inner = EagerTensor::from_tensor_in(native, default_eager_ctx().unwrap()).unwrap();
    let storage = IdxTensorStorage::Eager {
        inner: Arc::new(inner),
        axis_classes: vec![0, 0],
    };

    let error = storage.materialize(2).unwrap_err();
    assert!(matches!(error, TensorStorageError::Materialization { .. }));
    assert!(std::error::Error::source(&error).is_some());
}

fn conjugate_with_injected_failure(
    tensor: &IdxTensor,
    target: *const EagerTensor,
    message: &'static str,
) -> IdxTensor {
    let calls = Cell::new(0usize);
    let conjugated = tensor.conj_with(&|inner| {
        calls.set(calls.get() + 1);
        if std::ptr::eq(inner, target) {
            Err(Arc::new(std::io::Error::other(message)) as _)
        } else {
            conjugate_eager(inner)
        }
    });
    assert!(calls.get() > 0, "injected closure was not reached");
    conjugated
}

fn assert_unwrapped_conjugation_error(
    tensor: IdxTensor,
    target: *const EagerTensor,
    message: &'static str,
) -> IdxTensor {
    let conjugated = conjugate_with_injected_failure(&tensor, target, message);
    let error = conjugated.to_storage().unwrap_err();
    assert!(matches!(error, TensorStorageError::Conjugation { .. }));
    let source = std::error::Error::source(&error).unwrap();
    assert_eq!(source.to_string(), message);
    assert!(
        source.source().is_none(),
        "source was wrapped more than once"
    );
    conjugated
}

#[test]
fn authoritative_storage_conjugation_failure_is_deferred_without_detaching() {
    let i = DynIndex::new_dyn(2);
    let native = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let inner = EagerTensor::requires_grad_in(native, default_eager_ctx().unwrap()).unwrap();
    let tensor = IdxTensor::from_inner(vec![i], inner).unwrap();
    let source = match &tensor.storage {
        IdxTensorStorage::Eager { inner, .. } => Arc::clone(inner),
        IdxTensorStorage::Compact(payload) => Arc::clone(&payload.payload),
        IdxTensorStorage::Materialized(_) | IdxTensorStorage::Deferred { .. } => {
            panic!("tracked eager source expected")
        }
    };

    let conjugated = conjugate_with_injected_failure(
        &tensor,
        Arc::as_ptr(&source),
        "forced authoritative eager conjugation failure",
    );
    assert!(conjugated.tracks_grad());
    assert!(conjugated.is_f64());
    assert!(!conjugated.is_complex());
    assert!(!conjugated.is_diag());
    assert_eq!(conjugated.dims(), vec![2]);

    let error = conjugated.to_storage().unwrap_err();
    let source = std::error::Error::source(&error).unwrap();
    assert_eq!(
        source.to_string(),
        "forced authoritative eager conjugation failure"
    );
    assert!(conjugated.detach().is_err());
}

#[test]
fn structured_payload_conjugation_failure_retains_graph_and_blocks_detached_primal() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_diag(
        vec![i, j],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    )
    .unwrap()
    .enable_grad()
    .unwrap();

    let target = Arc::as_ptr(
        &tensor
            .storage
            .compact_payload()
            .expect("tracked compact payload")
            .payload,
    );
    let conjugated = assert_unwrapped_conjugation_error(
        tensor,
        target,
        "forced structured AD conjugation failure",
    );
    assert!(conjugated.tracks_grad());
    assert!(conjugated.detach().is_err());
    assert!(conjugated.clone().enable_grad().is_err());
    assert!(conjugated.sum().is_err());
    assert!(conjugated.grad().is_err());
    assert!(conjugated.clear_grad().is_err());
    assert!(conjugated.maxabs().is_err());
    assert!(conjugated.norm_squared().is_err());

    let twice_conjugated = conjugated.conj();
    let error = twice_conjugated.to_storage().unwrap_err();
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        "forced structured AD conjugation failure"
    );
}

#[test]
fn eager_cache_conjugation_failure_is_deferred_with_original_diagnostic() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_diag(vec![i, j], vec![1.0_f64, 2.0]).unwrap();
    tensor.as_inner().unwrap();
    let target = Arc::as_ptr(tensor.eager_cache.get().unwrap());

    let conjugated = assert_unwrapped_conjugation_error(
        tensor,
        target,
        "forced eager cache conjugation failure",
    );
    assert!(!conjugated.tracks_grad());
    assert!(conjugated.detach().is_err());
}

#[test]
fn session_identity_alignment_shares_storage_and_cache_for_every_representation() {
    use tensor4all_tensorbackend::CpuExecutionContext;

    let backend = tensor4all_tensorbackend::CpuBackend::with_threads(1).expect("1T CPU backend");
    let context = CpuExecutionContext::from_backend(backend);
    let i = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);

    let dense =
        IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let diagonal = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();
    // A compact payload materializes its logical-dense cache, so sharing a populated cache is
    // observable here rather than being an empty-Arc comparison.
    diagonal.as_inner().unwrap();
    assert!(
        diagonal.eager_cache.get().is_some(),
        "the compact payload populated its eager cache"
    );
    let tracked = dense.clone().enable_grad().expect("tracked operand");
    let deferred = {
        let tensor = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0]).unwrap();
        tensor.as_inner().unwrap();
        let target = Arc::as_ptr(tensor.eager_cache.get().unwrap());
        conjugate_with_injected_failure(&tensor, target, "forced eager cache conjugation failure")
    };

    // The identity order: the same index list, which is the branch this test is about.
    let order = vec![i.clone(), k.clone()];
    for (label, tensor) in [
        ("dense", dense),
        ("diagonal", diagonal),
        ("tracked", tracked),
        ("deferred", deferred),
    ] {
        let before_kind = tensor.storage_kind();
        let before_classes = tensor.storage.axis_classes().to_vec();
        let before_cache = Arc::as_ptr(&tensor.eager_cache);
        let identity = context
            .with_concrete_session(|session| {
                crate::defaults::contract_session::permute_indices_on_session(
                    session, &tensor, &order,
                )
            })
            .expect("session entry")
            .expect("identity alignment is metadata only");

        assert_eq!(identity.indices(), order.as_slice(), "{label}");
        assert_eq!(identity.storage_kind(), before_kind, "{label}");
        assert_eq!(
            identity.storage.axis_classes(),
            before_classes.as_slice(),
            "{label}"
        );
        assert_eq!(
            Arc::as_ptr(&identity.eager_cache),
            before_cache,
            "{label}: the eager cache is shared, not rebuilt"
        );
        assert_eq!(identity.tracks_grad(), tensor.tracks_grad(), "{label}");
        match (&tensor.storage, &identity.storage) {
            (
                IdxTensorStorage::Deferred { error: before, .. },
                IdxTensorStorage::Deferred { error: after, .. },
            ) => assert_eq!(
                before.to_string(),
                after.to_string(),
                "{label}: the retained diagnostic is the same one"
            ),
            _ => assert_eq!(
                identity.to_vec::<f64>().unwrap(),
                tensor.to_vec::<f64>().unwrap(),
                "{label}"
            ),
        }
    }
}

#[test]
fn session_operand_reports_a_deferred_error_before_any_representation_refusal() {
    // The deferred representation is reached through the existing injected-failure fixture: a dense
    // tracked value whose conjugation failed retains its graph and its original diagnostic. The route
    // must return that storage error rather than its generic tracked-value refusal.
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_diag(vec![i, j], vec![1.0_f64, 2.0]).unwrap();
    tensor.as_inner().unwrap();
    let target = Arc::as_ptr(tensor.eager_cache.get().unwrap());
    let deferred =
        conjugate_with_injected_failure(&tensor, target, "forced eager cache conjugation failure");

    let error = match deferred.session_dense_operand("contract_pair_on_session") {
        Ok(_) => panic!("a deferred operand must be refused"),
        Err(error) => error,
    };
    match error {
        IdxTensorError::Storage { source } => assert!(
            matches!(source, TensorStorageError::Conjugation { .. }),
            "the deferred diagnostic must survive: {source:?}"
        ),
        other => panic!("expected the deferred storage error, got {other:?}"),
    }
}

#[test]
fn encode_col_major_linear_rejects_offset_overflow() {
    let error =
        encode_col_major_linear(&[usize::MAX - 1, usize::MAX - 1], &[usize::MAX, usize::MAX])
            .unwrap_err();
    assert!(error.to_string().contains("linear offset overflow"));
}

#[test]
fn factorize_probe_batch_incremental_matches_the_column_based_path() {
    let row = DynIndex::new_dyn(6);
    let batch = DynIndex::new_dyn(4);
    let data: Vec<f64> = (0..24).map(|i| i as f64 * 0.37 - 1.5).collect();
    let batch_tensor =
        IdxTensor::from_dense(vec![row.clone(), batch.clone()], data.clone()).unwrap();

    let columns: Vec<IdxTensor> = (0..4)
        .map(|position| {
            batch_tensor
                .select_indices(std::slice::from_ref(&batch), &[position])
                .unwrap()
        })
        .collect();
    let column_refs: Vec<&IdxTensor> = columns.iter().collect();

    let from_batch = IdxTensor::factorize_probe_batch_incremental(
        None,
        &batch_tensor,
        &batch,
        std::slice::from_ref(&row),
    )
    .unwrap();
    let from_columns =
        IdxTensor::factorize_probe_columns_incremental(None, &column_refs, &column_refs, &[row])
            .unwrap();

    assert_eq!(from_batch.rank, from_columns.rank);
    let batch_data = from_batch.left.to_vec::<f64>().unwrap();
    let columns_data = from_columns.left.to_vec::<f64>().unwrap();
    assert_eq!(batch_data.len(), columns_data.len());
    let max_diff = batch_data
        .iter()
        .zip(columns_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-12,
        "batch-native and column-based factorizations disagree: max diff = {}",
        max_diff
    );
}

/// #988: promoting mixed-dtype operands must cast through the owner runtime
/// (not a context-free native tensor) and must keep the autodiff graph, so a
/// gradient reaches the real leaf through the promotion.
#[test]
fn promote_eager_operands_casts_in_the_owner_runtime_and_keeps_grad() {
    let runtime = default_eager_ctx().unwrap();
    let real_leaf = EagerTensor::requires_grad_in(
        NativeTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let complex_leaf = EagerTensor::requires_grad_in(
        NativeTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        )
        .unwrap(),
        runtime.clone(),
    )
    .unwrap();

    let (real_cast, complex_cast) =
        IdxTensor::promote_eager_operands(&real_leaf, &complex_leaf).unwrap();

    assert_eq!(real_cast.dtype(), tenferro::DType::C64);
    assert_eq!(complex_cast.dtype(), tenferro::DType::C64);
    assert_eq!(real_cast.ctx_id(), runtime.id());
    assert!(real_cast.tracks_grad());
    assert!(complex_cast.tracks_grad());

    // A reduction of the promoted value must back-propagate through the cast to
    // the original real leaf.
    let promoted = IdxTensor::from_inner(vec![DynIndex::new_dyn(2)], real_cast).unwrap();
    promoted.sum().unwrap().backward().unwrap();
    assert!(real_leaf.grad().unwrap().is_some());
}

// ============================================================================
// #988: numerical gradients through the public mixed-dtype eager operations
// ============================================================================

use crate::index_like::IndexLike;

/// Fresh, caller-scoped CPU eager runtime for the mixed-dtype gradient checks.
fn scoped_eager_runtime() -> Arc<EagerRuntime> {
    tensor4all_tensorbackend::CpuExecutionContext::from_backend(
        tensor4all_tensorbackend::CpuBackend::with_threads(1).expect("1T CPU backend"),
    )
    .eager_runtime()
    .expect("scoped eager runtime")
}

fn dims_of(indices: &[DynIndex]) -> Vec<usize> {
    indices.iter().map(|index| index.dim()).collect()
}

fn tracked_real_leaf(runtime: &Arc<EagerRuntime>, indices: &[DynIndex], data: &[f64]) -> IdxTensor {
    let inner = EagerTensor::requires_grad_in(
        NativeTensor::from_vec_col_major(dims_of(indices), data.to_vec()).unwrap(),
        Arc::clone(runtime),
    )
    .unwrap();
    IdxTensor::from_inner(indices.to_vec(), inner).unwrap()
}

fn tracked_complex_leaf(
    runtime: &Arc<EagerRuntime>,
    indices: &[DynIndex],
    data: &[Complex64],
) -> IdxTensor {
    let inner = EagerTensor::requires_grad_in(
        NativeTensor::from_vec_col_major(dims_of(indices), data.to_vec()).unwrap(),
        Arc::clone(runtime),
    )
    .unwrap();
    IdxTensor::from_inner(indices.to_vec(), inner).unwrap()
}

fn untracked_real_leaf(
    runtime: &Arc<EagerRuntime>,
    indices: &[DynIndex],
    data: &[f64],
) -> IdxTensor {
    let inner = EagerTensor::from_tensor_in(
        NativeTensor::from_vec_col_major(dims_of(indices), data.to_vec()).unwrap(),
        Arc::clone(runtime),
    )
    .unwrap();
    IdxTensor::from_inner(indices.to_vec(), inner).unwrap()
}

fn untracked_complex_leaf(
    runtime: &Arc<EagerRuntime>,
    indices: &[DynIndex],
    data: &[Complex64],
) -> IdxTensor {
    let inner = EagerTensor::from_tensor_in(
        NativeTensor::from_vec_col_major(dims_of(indices), data.to_vec()).unwrap(),
        Arc::clone(runtime),
    )
    .unwrap();
    IdxTensor::from_inner(indices.to_vec(), inner).unwrap()
}

/// Deterministic complex weight used to reduce an operation result to the real
/// scalar `L = Re(sum_k W_k result_k)`.
fn gradient_weight(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|k| Complex64::new(k as f64 - 2.5, 0.5 * k as f64 - 1.0))
        .collect()
}

fn weighted_real_loss(result: &IdxTensor, weight: &[Complex64]) -> f64 {
    result
        .to_vec::<Complex64>()
        .unwrap()
        .iter()
        .zip(weight)
        .map(|(value, weight)| (weight * value).re)
        .sum::<f64>()
}

/// Check reverse-mode gradients through one public mixed-dtype operation
/// against central finite differences of the real loss
/// `L = Re(sum_k W_k result_k)`.
///
/// `real_first` selects the operand order. The tracked leaves live in a fresh
/// scoped CPU runtime, and the result must stay in that runtime.
fn check_mixed_dtype_gradient(
    label: &str,
    real_indices: &[DynIndex],
    real_data: &[f64],
    complex_indices: &[DynIndex],
    complex_data: &[Complex64],
    real_first: bool,
    apply: &dyn Fn(&IdxTensor, &IdxTensor) -> IdxTensor,
) {
    let runtime = scoped_eager_runtime();
    let real = tracked_real_leaf(&runtime, real_indices, real_data);
    let complex = tracked_complex_leaf(&runtime, complex_indices, complex_data);

    let result = if real_first {
        apply(&real, &complex)
    } else {
        apply(&complex, &real)
    };
    assert_eq!(
        result.try_materialized_inner().unwrap().runtime().id(),
        runtime.id(),
        "{label}: the mixed-dtype result left the scoped runtime"
    );

    let weight_data = gradient_weight(result.to_vec::<Complex64>().unwrap().len());
    let weight = IdxTensor::from_inner(
        result.indices().to_vec(),
        EagerTensor::from_tensor_in(
            NativeTensor::from_vec_col_major(dims_of(result.indices()), weight_data.clone())
                .unwrap(),
            Arc::clone(&runtime),
        )
        .unwrap(),
    )
    .unwrap();

    crate::contract_pair(&result, &weight)
        .unwrap()
        .sum()
        .unwrap()
        .real_part()
        .backward()
        .unwrap();

    let grad_real = real.grad().unwrap().unwrap().to_vec::<f64>().unwrap();
    let grad_complex = complex
        .grad()
        .unwrap()
        .unwrap()
        .to_vec::<Complex64>()
        .unwrap();

    let forward = |real_data: &[f64], complex_data: &[Complex64]| -> f64 {
        let real = untracked_real_leaf(&runtime, real_indices, real_data);
        let complex = untracked_complex_leaf(&runtime, complex_indices, complex_data);
        let result = if real_first {
            apply(&real, &complex)
        } else {
            apply(&complex, &real)
        };
        weighted_real_loss(&result, &weight_data)
    };

    let step = 1.0e-6;
    let tol = 1.0e-6;
    for index in 0..real_data.len() {
        let mut plus = real_data.to_vec();
        let mut minus = real_data.to_vec();
        plus[index] += step;
        minus[index] -= step;
        let expected =
            (forward(&plus, complex_data) - forward(&minus, complex_data)) / (2.0 * step);
        assert!(
            (grad_real[index] - expected).abs() <= tol,
            "{label}: d/d(real[{index}]) = {}, finite difference {expected}",
            grad_real[index]
        );
    }

    for index in 0..complex_data.len() {
        let mut real_plus = complex_data.to_vec();
        let mut real_minus = complex_data.to_vec();
        real_plus[index].re += step;
        real_minus[index].re -= step;
        let expected_real =
            (forward(real_data, &real_plus) - forward(real_data, &real_minus)) / (2.0 * step);

        let mut imag_plus = complex_data.to_vec();
        let mut imag_minus = complex_data.to_vec();
        imag_plus[index].im += step;
        imag_minus[index].im -= step;
        let expected_imag =
            (forward(real_data, &imag_plus) - forward(real_data, &imag_minus)) / (2.0 * step);

        let actual = grad_complex[index];
        assert!(
            (actual.re - expected_real).abs() <= tol && (actual.im - expected_imag).abs() <= tol,
            "{label}: d/d(complex[{index}]) = {actual}, finite difference \
             {expected_real} + i {expected_imag}"
        );
    }
}

/// Run the gradient check for `f64 x c64` and `c64 x f64`.
fn check_both_operand_orders(
    label: &str,
    real_indices: &[DynIndex],
    real_data: &[f64],
    complex_indices: &[DynIndex],
    complex_data: &[Complex64],
    apply: &dyn Fn(&IdxTensor, &IdxTensor) -> IdxTensor,
) {
    check_mixed_dtype_gradient(
        &format!("{label}, f64 x c64"),
        real_indices,
        real_data,
        complex_indices,
        complex_data,
        true,
        apply,
    );
    check_mixed_dtype_gradient(
        &format!("{label}, c64 x f64"),
        real_indices,
        real_data,
        complex_indices,
        complex_data,
        false,
        apply,
    );
}

#[test]
fn mixed_dtype_pairwise_contraction_gradients_match_finite_differences() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let real_data = vec![0.7_f64, -1.3, 0.9, 1.1, -0.4, 2.0];
    let complex_data = vec![
        Complex64::new(0.5, 1.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(1.0, -0.75),
    ];
    check_both_operand_orders(
        "pairwise contraction",
        &[i, j.clone()],
        &real_data,
        std::slice::from_ref(&j),
        &complex_data,
        &|lhs, rhs| crate::contract_pair(lhs, rhs).unwrap(),
    );
}

#[test]
fn mixed_dtype_tensordot_gradients_match_finite_differences() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let real_data = vec![0.7_f64, -1.3, 0.9, 1.1, -0.4, 2.0];
    let complex_data = vec![
        Complex64::new(0.5, 1.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(1.0, -0.75),
    ];
    let pairs = [(j.clone(), j.clone())];
    check_both_operand_orders(
        "tensordot",
        &[i, j.clone()],
        &real_data,
        std::slice::from_ref(&j),
        &complex_data,
        &|lhs, rhs| crate::tensordot(lhs, rhs, &pairs).unwrap(),
    );
}

#[test]
fn mixed_dtype_outer_product_gradients_match_finite_differences() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let real_data = vec![0.7_f64, -1.3];
    let complex_data = vec![
        Complex64::new(0.5, 1.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(1.0, -0.75),
    ];
    check_both_operand_orders(
        "outer product",
        std::slice::from_ref(&i),
        &real_data,
        std::slice::from_ref(&j),
        &complex_data,
        &|lhs, rhs| crate::outer_product(lhs, rhs).unwrap(),
    );
}
