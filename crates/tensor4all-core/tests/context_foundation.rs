use std::sync::Arc;

use num_complex::Complex64;

use tenferro_cpu::CpuBackend;
use tensor4all_core::{
    contract_pair_with_operand_options, outer_product, tensordot, DynIndex, ExecutionContext,
    IdxTensor, PairwiseContractionOptions, TensorContractionLike,
};
use tensor4all_tensorbackend::CpuExecutionContext;

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )))
}

#[test]
fn context_scoped_construction_uses_the_supplied_cpu_runtime() {
    let context = cpu_context();
    let index = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense_in(&context, vec![index], vec![1.0_f64, 2.0]).unwrap();

    tensor.validate_context(&context).unwrap();
    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);

    let ones = IdxTensor::ones_in(&context, tensor.indices()).unwrap();
    assert_eq!(ones.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
}

#[test]
fn context_validation_rejects_a_different_cpu_runtime() {
    let first = cpu_context();
    let second = cpu_context();
    let tensor =
        IdxTensor::from_dense_in(&first, vec![DynIndex::new_dyn(1)], vec![3.0_f64]).unwrap();

    assert!(tensor.validate_context(&second).is_err());
}

#[test]
fn decision_readback_returns_values_in_the_supplied_cpu_runtime() {
    let context = cpu_context();
    let tensor = IdxTensor::from_dense_in(
        &context,
        vec![DynIndex::new_dyn(3)],
        vec![0.5_f64, 2.0, 1.0],
    )
    .unwrap();

    assert_eq!(
        tensor.read_decision_data(&context).unwrap(),
        vec![0.5, 2.0, 1.0]
    );

    let other = cpu_context();
    assert!(tensor.read_decision_data(&other).is_err());
}

#[cfg(feature = "tenferro-cuda")]
#[test]
#[ignore]
fn context_scoped_cuda_construction_stays_in_the_selected_runtime() {
    let cuda = tensor4all_tensorbackend::CudaExecutionContext::new().unwrap();
    let context = ExecutionContext::Cuda(Arc::new(cuda));
    let tensor =
        IdxTensor::from_dense_in(&context, vec![DynIndex::new_dyn(2)], vec![1.0_f64, 2.0]).unwrap();

    tensor.validate_context(&context).unwrap();
}

#[cfg(feature = "tenferro-cuda")]
#[test]
#[ignore]
fn decision_readback_downloads_through_the_owning_cuda_runtime() {
    let cuda = tensor4all_tensorbackend::CudaExecutionContext::new().unwrap();
    let context = ExecutionContext::Cuda(Arc::new(cuda));
    let tensor =
        IdxTensor::from_dense_in(&context, vec![DynIndex::new_dyn(2)], vec![1.0_f64, 2.0]).unwrap();

    assert_eq!(tensor.read_decision_data(&context).unwrap(), vec![1.0, 2.0]);

    let foreign = ExecutionContext::Cuda(Arc::new(
        tensor4all_tensorbackend::CudaExecutionContext::new().unwrap(),
    ));
    assert!(tensor.read_decision_data(&foreign).is_err());
}

#[test]
fn context_scoped_cpu_scaling_and_norm_preserve_values() {
    let context = cpu_context();
    let tensor =
        IdxTensor::from_dense_in(&context, vec![DynIndex::new_dyn(2)], vec![3.0_f64, 4.0]).unwrap();

    assert!((tensor.norm_in(&context).unwrap() - 5.0).abs() < 1e-12);
    for factor in [0.0_f64, 1.0, 2.0] {
        let scaled = tensor.scale_in(factor, &context).unwrap();
        for (actual, expected) in scaled
            .to_vec::<f64>()
            .unwrap()
            .iter()
            .zip([3.0 * factor, 4.0 * factor])
        {
            assert!((actual - expected).abs() < 1e-12, "factor {factor}");
        }
        assert!((scaled.norm_in(&context).unwrap() - 5.0 * factor).abs() < 1e-12);
    }
    // Scaling must not mutate the operand held in the context.
    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![3.0, 4.0]);

    let ones = IdxTensor::ones_in(&context, &[]).unwrap();
    assert_eq!(ones.to_vec::<f64>().unwrap(), vec![1.0]);
}

#[test]
fn context_scoped_complex_cpu_operations_preserve_values() {
    let context = cpu_context();
    let tensor = IdxTensor::from_dense_in(
        &context,
        vec![DynIndex::new_dyn(2)],
        vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
    )
    .unwrap();

    assert!((tensor.norm_in(&context).unwrap() - 5.0).abs() < 1e-12);
    assert_eq!(tensor.read_decision_data(&context).unwrap(), vec![3.0, 0.0]);

    let scaled = tensor.scale_in(2.0, &context).unwrap();
    assert_eq!(
        scaled.to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(6.0, 8.0), Complex64::new(0.0, 0.0)]
    );
}

/// A 2x3 real tensor and a length-3 complex vector in the same context, used by
/// the mixed-dtype #988 regressions. The pair contracts over the shared `j`
/// index and yields `[11 - 2i, 14 - 2i]` in both operand orders.
fn mixed_dtype_fixture(context: &ExecutionContext) -> (IdxTensor, IdxTensor, DynIndex, DynIndex) {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let real = IdxTensor::from_dense_in(
        context,
        vec![i.clone(), j.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let complex = IdxTensor::from_dense_in(
        context,
        vec![j.clone()],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();
    (real, complex, i, j)
}

const MIXED_PAIRWISE_EXPECTED: [Complex64; 2] =
    [Complex64::new(11.0, -2.0), Complex64::new(14.0, -2.0)];

#[test]
fn mixed_dtype_pairwise_contraction_stays_in_caller_context() {
    let context = cpu_context();
    let (real, complex, _i, _j) = mixed_dtype_fixture(&context);

    // Real on the left, complex on the right.
    let real_first = real.contract_pair(&complex).unwrap();
    real_first.validate_context(&context).unwrap();
    assert_eq!(
        real_first.to_vec::<Complex64>().unwrap(),
        MIXED_PAIRWISE_EXPECTED
    );

    // Complex on the left, real on the right.
    let complex_first = complex.contract_pair(&real).unwrap();
    complex_first.validate_context(&context).unwrap();
    assert_eq!(
        complex_first.to_vec::<Complex64>().unwrap(),
        MIXED_PAIRWISE_EXPECTED
    );
}

#[test]
fn mixed_dtype_tensordot_stays_in_caller_context() {
    let context = cpu_context();
    let (real, complex, _i, j) = mixed_dtype_fixture(&context);

    for (lhs, rhs) in [(&real, &complex), (&complex, &real)] {
        let result = tensordot(lhs, rhs, &[(j.clone(), j.clone())]).unwrap();
        result.validate_context(&context).unwrap();
        assert_eq!(
            result.to_vec::<Complex64>().unwrap(),
            MIXED_PAIRWISE_EXPECTED
        );
    }
}

#[test]
fn mixed_dtype_outer_product_stays_in_caller_context() {
    let context = cpu_context();
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let real = IdxTensor::from_dense_in(&context, vec![i], vec![1.0_f64, 2.0]).unwrap();
    let complex = IdxTensor::from_dense_in(
        &context,
        vec![j],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();

    let real_first = outer_product(&real, &complex).unwrap();
    real_first.validate_context(&context).unwrap();
    assert_eq!(
        real_first.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]
    );

    let complex_first = outer_product(&complex, &real).unwrap();
    complex_first.validate_context(&context).unwrap();
    assert_eq!(
        complex_first.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(4.0, 0.0),
        ]
    );
}

/// The SRC batch broadcast shape from #988: a complex `local` tensordot the
/// process-real `ones_in` batch tensor. Before the fix the f64 x c64 branch
/// rebuilt both operands as context-free native tensors and the result left
/// the caller's context.
#[test]
fn mixed_dtype_src_style_ones_broadcast_stays_in_caller_context() {
    let context = cpu_context();
    let site = DynIndex::new_dyn(2);
    let batch = DynIndex::new_dyn(3);
    let local = IdxTensor::from_dense_in(
        &context,
        vec![site.clone()],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    )
    .unwrap();
    let ones = IdxTensor::ones_in(&context, std::slice::from_ref(&batch)).unwrap();

    let result = local.outer_product(&ones).unwrap();
    result.validate_context(&context).unwrap();
    assert_eq!(result.indices(), &[site, batch]);
    assert_eq!(
        result.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
        ]
    );
}

#[test]
fn mixed_dtype_pairwise_conjugation_stays_in_caller_context() {
    let context = cpu_context();
    let (real, complex, _i, _j) = mixed_dtype_fixture(&context);

    let flagged = contract_pair_with_operand_options(
        &complex,
        &real,
        PairwiseContractionOptions::new().with_lhs_conj(true),
    )
    .unwrap();
    flagged.validate_context(&context).unwrap();

    let reference = complex.conj().contract_pair(&real).unwrap();
    reference.validate_context(&context).unwrap();
    assert_eq!(
        flagged.to_vec::<Complex64>().unwrap(),
        reference.to_vec::<Complex64>().unwrap()
    );
}
