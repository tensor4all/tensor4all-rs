//! Scaling and scalar extraction retain the caller-owned AD graph.

use super::*;
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};

fn context() -> Arc<CpuExecutionContext> {
    Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    ))
}

#[test]
fn dense_publication_reuses_ownership_and_validates_shape() {
    let owner = context();
    let execution = ExecutionContext::Cpu(Arc::clone(&owner));
    let index = DynIndex::new_dyn(2);
    let plain = IdxTensor::from_dense_compact(vec![index.clone()], vec![1.0_f64, 2.0]).unwrap();
    let owned =
        IdxTensor::from_dense_in(&execution, vec![index.clone()], vec![1.0_f64, 2.0]).unwrap();
    for template in [&plain, &owned] {
        let result =
            IdxTensor::from_dense_like(template, vec![index.clone()], vec![3.0_f64, 4.0]).unwrap();
        assert_eq!(
            result.eager_runtime().is_some(),
            template.eager_runtime().is_some()
        );
        if template.eager_runtime().is_some() {
            result.validate_context(&execution).unwrap();
        }
        assert_eq!(result.dense_host_values::<f64>().unwrap(), vec![3.0, 4.0]);
        assert!(IdxTensor::from_dense_like(template, vec![index.clone()], vec![1.0_f64]).is_err());
        assert!(IdxTensor::from_dense_like(
            template,
            vec![index.clone(), index.clone()],
            vec![1.0_f64; 4]
        )
        .is_err());
    }
}

fn tracked(context: &CpuExecutionContext, indices: Vec<DynIndex>, data: Vec<f64>) -> IdxTensor {
    let shape = indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
    let runtime = context.eager_runtime().unwrap();
    let native = NativeTensor::from_vec_col_major(shape, data).unwrap();
    IdxTensor::from_inner(indices, runtime.variable_from(native).unwrap()).unwrap()
}

#[test]
fn untracked_scalar_rehoming_preserves_tensor_gradients() {
    let context = context();
    let tensor = tracked(&context, vec![DynIndex::new_dyn(2)], vec![3.0, 4.0]);
    let scaled = tensor.scale(AnyScalar::new_real(2.0)).unwrap();
    scaled
        .validate_context(&ExecutionContext::Cpu(Arc::clone(&context)))
        .unwrap();
    scaled.sum().unwrap().backward().unwrap();
    assert_eq!(
        tensor.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0, 2.0]
    );
}

#[test]
fn same_runtime_tracked_coefficients_keep_both_gradient_paths() {
    let context = context();
    let tensor = tracked(&context, vec![DynIndex::new_dyn(2)], vec![3.0, 4.0]);
    let coefficient = tracked(&context, vec![], vec![2.0]);
    let scaled = tensor.scale(coefficient.to_scalar().unwrap()).unwrap();
    scaled
        .validate_context(&ExecutionContext::Cpu(Arc::clone(&context)))
        .unwrap();
    scaled.sum().unwrap().backward().unwrap();
    assert_eq!(
        tensor.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0, 2.0]
    );
    assert_eq!(
        coefficient
            .grad()
            .unwrap()
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![7.0]
    );
}

#[test]
fn scalar_extraction_preserves_runtime_tracking_and_rejects_external_indices() {
    let context = context();
    let scalar = tracked(&context, vec![], vec![3.0]);
    let value = scalar.to_scalar().unwrap();
    assert_eq!(value.real(), 3.0);
    assert!(value.tracks_grad());
    value.backward().unwrap();
    assert_eq!(
        scalar.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0]
    );
    let vector = tracked(&context, vec![DynIndex::new_dyn(1)], vec![3.0]);
    assert!(matches!(
        vector.to_scalar(),
        Err(IdxTensorError::ShapeMismatch { .. })
    ));
}
