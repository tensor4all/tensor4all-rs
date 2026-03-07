use tensor4all_core::{Index, TensorDynLen};
use tensor4all_itensorlike::{CanonicalForm, TensorTrain, TruncateOptions};
use tensor4all_tensorbackend::tenferro_dyadtensor::{AdMode, AdTensor, DynAdTensor};
use tensor4all_tensorbackend::tenferro_tensor::{MemoryOrder, Tensor};

fn native_f64_tensor(primal: &[f64], tangent: &[f64], dims: &[usize]) -> DynAdTensor {
    let primal = Tensor::<f64>::from_slice(primal, dims, MemoryOrder::RowMajor)
        .expect("valid primal tensor");
    let tangent = Tensor::<f64>::from_slice(tangent, dims, MemoryOrder::RowMajor)
        .expect("valid tangent tensor");
    AdTensor::new_forward(primal, tangent).unwrap().into()
}

fn make_two_site_tt() -> TensorTrain {
    let s0 = Index::new_dyn(2);
    let s1 = Index::new_dyn(2);
    let bond = Index::new_dyn(2);

    let t0 = TensorDynLen::from_native(
        vec![s0, bond.clone()],
        native_f64_tensor(&[1.0, 0.0, 0.0, 2.0], &[0.1, 0.0, 0.0, -0.2], &[2, 2]),
    )
    .unwrap();
    let t1 = TensorDynLen::from_native(
        vec![bond, s1],
        native_f64_tensor(&[3.0, 0.0, 0.0, 4.0], &[0.3, 0.0, 0.0, -0.4], &[2, 2]),
    )
    .unwrap();

    TensorTrain::new(vec![t0, t1]).unwrap()
}

#[test]
fn orthogonalize_preserves_forward_native_payload() {
    let mut tt = make_two_site_tt();

    tt.orthogonalize_with(1, CanonicalForm::Unitary).unwrap();

    for site in 0..tt.len() {
        let tensor = tt.tensor(site);
        let mode = tensor.as_native().map(DynAdTensor::mode);
        assert_eq!(mode, Some(AdMode::Forward), "site {site} lost native mode");
        assert!(
            tensor.sum().tangent().is_some(),
            "site {site} lost tangent information"
        );
    }
}

#[test]
fn truncate_preserves_forward_native_payload() {
    let mut tt = make_two_site_tt();

    tt.truncate(&TruncateOptions::svd().with_max_rank(1))
        .unwrap();

    assert_eq!(tt.tensor(0).dims()[1], 1);
    assert_eq!(tt.tensor(1).dims()[0], 1);
    for site in 0..tt.len() {
        let tensor = tt.tensor(site);
        let mode = tensor.as_native().map(DynAdTensor::mode);
        assert_eq!(mode, Some(AdMode::Forward), "site {site} lost native mode");
        assert!(
            tensor.sum().tangent().is_some(),
            "site {site} lost tangent information"
        );
    }
}
