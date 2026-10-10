use std::sync::Arc;

use num_complex::Complex64;

use tenferro_cpu::CpuBackend;
use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor};
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
