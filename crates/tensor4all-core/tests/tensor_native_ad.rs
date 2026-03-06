use tensor4all_core::{Index, TensorDynLen};
use tensor4all_tensorbackend::tenferro_dyadtensor::{AdMode, AdTensor, DynAdTensor};
use tensor4all_tensorbackend::tenferro_tensor::{MemoryOrder, Tensor};

fn native_f64_tensor(primal: &[f64], tangent: &[f64], dims: &[usize]) -> DynAdTensor {
    let primal = Tensor::<f64>::from_slice(primal, dims, MemoryOrder::RowMajor)
        .expect("valid primal tensor");
    let tangent = Tensor::<f64>::from_slice(tangent, dims, MemoryOrder::RowMajor)
        .expect("valid tangent tensor");
    AdTensor::new_forward(primal, tangent).into()
}

#[test]
fn sum_preserves_forward_native_payload() {
    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::from_native(
        vec![i],
        native_f64_tensor(&[1.0, 2.0], &[0.25, -0.75], &[2]),
    )
    .unwrap();

    let sum = tensor.sum();

    assert_eq!(sum.mode(), AdMode::Forward);
    assert_eq!(sum.primal().as_f64(), Some(3.0));
    assert_eq!(sum.tangent().and_then(|x| x.as_f64()), Some(-0.5));
}

#[test]
fn only_preserves_forward_native_payload() {
    let tensor =
        TensorDynLen::from_native(vec![], native_f64_tensor(&[2.5], &[0.75], &[])).unwrap();

    let only = tensor.only();

    assert_eq!(only.mode(), AdMode::Forward);
    assert_eq!(only.primal().as_f64(), Some(2.5));
    assert_eq!(only.tangent().and_then(|x| x.as_f64()), Some(0.75));
}

#[test]
fn inner_product_preserves_forward_native_payload() {
    let i = Index::new_dyn(2);
    let lhs = TensorDynLen::from_native(
        vec![i.clone()],
        native_f64_tensor(&[1.0, 2.0], &[0.1, 0.2], &[2]),
    )
    .unwrap();
    let rhs =
        TensorDynLen::from_native(vec![i], native_f64_tensor(&[3.0, 4.0], &[1.0, -1.0], &[2]))
            .unwrap();

    let inner = lhs.inner_product(&rhs).unwrap();

    assert_eq!(inner.mode(), AdMode::Forward);
    assert_eq!(inner.primal().as_f64(), Some(11.0));
    let tangent = inner.tangent().and_then(|x| x.as_f64()).unwrap();
    assert!(
        (tangent - 0.1).abs() < 1e-12,
        "unexpected tangent: {tangent}"
    );
}
