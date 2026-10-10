//! Mixed-feature regression for #869.
//!
//! This crate builds `tensor4all-core` without its `tenferro-cuda` feature while
//! `tensor4all-tensorbackend` has it, i.e. the configuration in which
//! `ExecutionContext::Cuda` exists but core has no CUDA support compiled in.
//! core must build, keep working for CPU contexts, and reject a CUDA context
//! with [`IdxTensorError::UnsupportedExecutionContext`] before any runtime
//! initialisation, transfer, upload, or output mutation.
//!
//! Run from this directory: `cargo test --locked --test rejection -- --test-threads=1`
//! (see README.md; requires a machine with a CUDA toolkit and a visible device).

use std::sync::Arc;

use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tensor4all_core::{
    factorize_in, Canonical, DynIndex, ExecutionContext, FactorizeAlg, FactorizeOptions, IdxTensor,
    IdxTensorError, TensorContractionLike, TensorFactorizationLike,
};
use tensor4all_tensorbackend::{CpuExecutionContext, CudaExecutionContext, Storage};

fn cuda_context() -> (Arc<CudaExecutionContext>, ExecutionContext) {
    let cuda = Arc::new(CudaExecutionContext::new().expect("a CUDA device and toolkit"));
    (Arc::clone(&cuda), ExecutionContext::Cuda(cuda))
}

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )))
}

fn assert_unsupported(error: &IdxTensorError) {
    match error {
        IdxTensorError::UnsupportedExecutionContext { required_feature } => {
            assert_eq!(*required_feature, "tensor4all-core/tenferro-cuda");
        }
        other => panic!("expected an unsupported-context rejection, got: {other}"),
    }
    assert!(
        error.to_string().contains("tensor4all-core/tenferro-cuda"),
        "diagnostic must name the required feature: {error}"
    );
}

/// Every wrapped failure must keep the typed cause reachable, not stringified.
fn assert_source_chain_has_unsupported(error: &(dyn std::error::Error + 'static)) {
    let mut cause = error.source();
    while let Some(current) = cause {
        if let Some(typed) = current.downcast_ref::<IdxTensorError>() {
            assert_unsupported(typed);
            return;
        }
        cause = current.source();
    }
    panic!("source chain does not carry the typed unsupported-context error: {error}");
}

/// Eager-storage fixture: `from_dense` wraps the payload in the process-global
/// eager runtime, so `storage.eager()` is `Some`.
fn eager_tensor() -> IdxTensor {
    IdxTensor::from_dense(
        vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
        vec![3.0_f64, 0.0, 0.0, 4.0],
    )
    .unwrap()
}

/// Materialized-storage fixture: `from_storage` keeps the payload compact, so
/// `storage.eager()` is `None` and `validate_context` takes its early branch.
fn materialized_tensor() -> IdxTensor {
    let storage = Storage::from_dense_col_major(vec![3.0_f64, 0.0, 0.0, 4.0], &[2, 2]).unwrap();
    IdxTensor::from_storage(
        vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
        Arc::new(storage),
    )
    .unwrap()
}

/// The rejection must never initialise the CUDA eager runtime.
fn assert_cuda_runtime_untouched(cuda: &CudaExecutionContext) {
    assert!(
        format!("{cuda:?}").contains("eager_initialized: false"),
        "a rejection must not initialise the CUDA eager runtime"
    );
}

#[test]
fn validation_rejects_a_cuda_context_for_materialized_and_eager_storage() {
    let (cuda, context) = cuda_context();

    assert_unsupported(
        &materialized_tensor()
            .validate_context(&context)
            .unwrap_err(),
    );
    assert_cuda_runtime_untouched(&cuda);
    assert_unsupported(&eager_tensor().validate_context(&context).unwrap_err());
    assert_cuda_runtime_untouched(&cuda);
}

#[test]
fn readback_scaling_and_norm_reject_a_cuda_context_without_mutation() {
    let (cuda, context) = cuda_context();
    let tensor = materialized_tensor();

    assert_unsupported(&tensor.read_decision_data(&context).unwrap_err());
    assert_cuda_runtime_untouched(&cuda);
    for factor in [0.0_f64, 1.0, 2.0] {
        assert_unsupported(&tensor.scale_in(factor, &context).unwrap_err());
    }
    assert_cuda_runtime_untouched(&cuda);
    assert_unsupported(&tensor.norm_in(&context).unwrap_err());
    assert_cuda_runtime_untouched(&cuda);

    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![3.0, 0.0, 0.0, 4.0],
        "a rejected call must not mutate the operand"
    );
}

#[test]
fn construction_entry_points_reject_a_cuda_context() {
    let (cuda, context) = cuda_context();
    let index = DynIndex::new_dyn(2);

    assert_unsupported(
        &IdxTensor::from_dense_in(&context, vec![index.clone()], vec![1.0_f64, 2.0]).unwrap_err(),
    );
    assert_unsupported(
        &IdxTensor::from_dense_in(
            &context,
            vec![index.clone()],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        )
        .unwrap_err(),
    );
    assert_unsupported(&IdxTensor::ones_in(&context, &[index]).unwrap_err());
    // Degenerate and empty shapes must not silently complete. An empty index
    // list reaches the context check; a zero-sized index is already rejected by
    // the dimension validation that runs first.
    assert_unsupported(&IdxTensor::ones_in(&context, &[]).unwrap_err());
    assert!(
        IdxTensor::ones_in(&context, &[DynIndex::new_dyn(0)]).is_err(),
        "a zero-sized request must not succeed"
    );
    assert_cuda_runtime_untouched(&cuda);
}

#[test]
fn factorization_entry_points_reject_a_cuda_context() {
    let (cuda, context) = cuda_context();
    let tensor = materialized_tensor();
    let left = tensor.indices()[0].clone();

    let error = factorize_in(
        &tensor,
        std::slice::from_ref(&left),
        &FactorizeOptions::qr(),
        &context,
    )
    .expect_err("factorize_in must reject a CUDA context");
    assert_source_chain_has_unsupported(&error);

    let error = tensor
        .factorize_full_rank_in(
            std::slice::from_ref(&left),
            FactorizeAlg::QR,
            Canonical::Left,
            &context,
        )
        .expect_err("factorize_full_rank_in must reject a CUDA context");
    assert_source_chain_has_unsupported(&error);

    let error = TensorFactorizationLike::src_error_estimate_in(&tensor, &context)
        .expect_err("src_error_estimate_in must reject a CUDA context");
    assert_source_chain_has_unsupported(&error);

    let error = IdxTensor::factorize_probe_batch_incremental_in(
        None,
        &tensor,
        &left,
        std::slice::from_ref(&left),
        &context,
    )
    .expect_err("factorize_probe_batch_incremental_in must reject a CUDA context");
    assert_source_chain_has_unsupported(&error);
    assert_cuda_runtime_untouched(&cuda);
}

#[test]
fn qr_and_svd_reject_a_cuda_context() {
    use tensor4all_core::qr::{qr_with_in, QrOptions};
    use tensor4all_core::svd::{svd_with_in, SvdOptions};

    let (cuda, context) = cuda_context();
    let tensor = materialized_tensor();
    let left = tensor.indices()[0].clone();

    let error = qr_with_in::<f64>(
        &tensor,
        std::slice::from_ref(&left),
        &QrOptions::new(),
        &context,
    )
    .expect_err("qr_with_in must reject a CUDA context");
    assert_source_chain_has_unsupported(&error);

    let error = svd_with_in::<f64>(&tensor, &[left], &SvdOptions::new(), &context)
        .expect_err("svd_with_in must reject a CUDA context");
    assert_source_chain_has_unsupported(&error);
    assert_cuda_runtime_untouched(&cuda);
}

#[test]
fn rejection_does_not_initialize_the_cuda_eager_runtime() {
    let (cuda, context) = cuda_context();
    let tensor = materialized_tensor();

    assert_unsupported(&tensor.validate_context(&context).unwrap_err());
    assert_cuda_runtime_untouched(&cuda);
}

#[test]
fn the_same_build_keeps_cpu_contexts_working() {
    let context = cpu_context();
    let index = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense_in(&context, vec![index], vec![3.0_f64, 4.0]).unwrap();

    tensor.validate_context(&context).unwrap();
    assert!((tensor.norm_in(&context).unwrap() - 5.0).abs() < 1e-12);
    assert_eq!(
        tensor
            .scale_in(2.0, &context)
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![6.0, 8.0]
    );
    assert_eq!(tensor.read_decision_data(&context).unwrap(), vec![3.0, 4.0]);
    assert_eq!(
        IdxTensor::ones_in(&context, &[])
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![1.0]
    );

    let matrix = IdxTensor::from_dense_in(
        &context,
        vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
        vec![3.0_f64, 0.0, 0.0, 4.0],
    )
    .unwrap();
    let result = factorize_in(
        &matrix,
        std::slice::from_ref(&matrix.indices()[0]),
        &FactorizeOptions::qr(),
        &context,
    )
    .unwrap();
    let recovered = result.left.contract_pair(&result.right).unwrap();
    assert!(recovered.isapprox(&matrix, 1e-12, 1e-12).unwrap());
}
