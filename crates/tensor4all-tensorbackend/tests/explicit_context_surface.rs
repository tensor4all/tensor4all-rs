//! The explicit-context surface runs numerical work on the caller's context.
//!
//! Compiled without `global-defaults` (`--no-default-features --features
//! explicit-context,tenferro-cpu-faer`) these calls cannot reach a
//! process-global context, because none exists in that build.

use tenferro::Tensor as NativeTensor;
use tensor4all_tensorbackend::{
    einsum_native_tensors_in, einsum_native_tensors_owned_in, hermitian_eigendecomposition_in,
    lowest_hermitian_eigenpair_in, CpuBackend, CpuExecutionContext, Matrix, NativeEinsumPlan,
};

fn context() -> CpuExecutionContext {
    CpuExecutionContext::from_backend(CpuBackend::with_threads(1).expect("one-thread backend"))
}

#[test]
fn einsum_runs_on_caller_context() {
    let ctx = context();
    let lhs = NativeTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = NativeTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();

    let borrowed =
        einsum_native_tensors_in(&ctx, &[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2]).unwrap();
    let owned =
        einsum_native_tensors_owned_in(&ctx, vec![(lhs, vec![0, 1]), (rhs, vec![1, 2])], &[0, 2])
            .unwrap();

    assert_eq!(borrowed.as_slice::<f64>().unwrap(), &[3.0; 4]);
    assert_eq!(owned.as_slice::<f64>().unwrap(), &[3.0; 4]);
}

#[test]
fn einsum_promotes_mixed_dtypes_on_caller_context() {
    let ctx = context();
    let lhs = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let rhs = NativeTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let dot = einsum_native_tensors_in(&ctx, &[(&lhs, &[0]), (&rhs, &[0])], &[]).unwrap();
    assert_eq!(dot.as_slice::<f64>().unwrap(), &[11.0]);
}

#[test]
fn einsum_plan_runs_on_caller_context() {
    use tenferro_tensor::TensorRead;
    let ctx = context();
    let plan = NativeEinsumPlan::new("i,i->").unwrap();
    let a = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = NativeTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let dot = plan
        .execute_reads_in(
            &ctx,
            &[TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)],
        )
        .unwrap();
    assert_eq!(dot.as_slice::<f64>().unwrap(), &[11.0]);
}

#[test]
fn hermitian_eigensolver_runs_on_caller_context() {
    let ctx = context();
    let matrix = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0]);

    let decomposition = hermitian_eigendecomposition_in(&ctx, &matrix, 1.0e-12).unwrap();
    assert!((decomposition.eigenvalues[0] - 1.0).abs() < 1.0e-12);
    assert!((decomposition.eigenvalues[1] - 3.0).abs() < 1.0e-12);

    let pair = lowest_hermitian_eigenpair_in(&ctx, &matrix, 1.0e-12).unwrap();
    assert!((pair.eigenvalue - 1.0).abs() < 1.0e-12);
}

#[test]
fn hermitian_eigensolver_rejects_invalid_tolerance_before_the_context() {
    let ctx = context();
    let matrix = Matrix::from_col_major_vec(1, 1, vec![1.0_f64]);
    assert!(hermitian_eigendecomposition_in(&ctx, &matrix, f64::NAN).is_err());
}
