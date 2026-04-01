use tensor4all_core::{contract_multi, AllowedPairs, Index, Storage, TensorDynLen};

fn with_runtime<R>(f: impl FnOnce() -> R) -> R {
    let _guard = tenferro::set_default_runtime(tenferro::RuntimeContext::Cpu(
        tenferro_prims::CpuContext::new(1),
    ));
    f()
}

#[test]
fn plain_dense_storage_auto_seeds_native_payload() {
    let i = Index::new_dyn(2);
    let tensor = TensorDynLen::from_storage(
        vec![i],
        Storage::from_dense_col_major(vec![1.0, 2.0], &[2])
            .map(std::sync::Arc::new)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
    assert!(!tensor.requires_grad());
}

#[test]
fn plain_diag_storage_auto_seeds_native_dense_payload() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::from_storage(
        vec![i, j],
        Storage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2)
            .map(std::sync::Arc::new)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ]
    );
    assert!(!tensor.is_diag());
}

#[test]
fn backward_ad_contraction_accumulates_gradient() {
    with_runtime(|| {
        let i = Index::new_dyn(3);

        let mut a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0]).unwrap();
        a.set_requires_grad(true).unwrap();

        let ones = TensorDynLen::from_dense(vec![i], vec![1.0, 1.0, 1.0]).unwrap();

        let result = contract_multi(&[&a, &ones], AllowedPairs::All).unwrap();
        assert!(
            result.indices().is_empty(),
            "contraction result should be rank-0"
        );

        result.backward(None).unwrap();

        let grad = a.grad().unwrap().expect("gradient missing after backward");
        let grad_vec = grad.to_vec::<f64>().unwrap();
        for (j, &g) in grad_vec.iter().enumerate() {
            assert!((g - 1.0).abs() < 1e-10, "grad[{j}] = {g}, expected 1.0");
        }
    });
}

#[test]
fn backward_ad_gradient_matches_finite_diff() {
    with_runtime(|| {
        let i = Index::new_dyn(2);
        let j = Index::new_dyn(2);
        let eps = 1e-6;

        let data = vec![1.0, 2.0, 3.0, 4.0];

        let mut a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data.clone()).unwrap();
        a.set_requires_grad(true).unwrap();

        let result = contract_multi(&[&a, &a], AllowedPairs::All).unwrap();
        result.backward(None).unwrap();

        let grad = a.grad().unwrap().expect("gradient missing");
        let grad_vec = grad.to_vec::<f64>().unwrap();

        for idx in 0..data.len() {
            let f_eval = |delta: f64| -> f64 {
                let mut d = data.clone();
                d[idx] += delta;
                let t = TensorDynLen::from_dense(vec![i.clone(), j.clone()], d).unwrap();
                contract_multi(&[&t, &t], AllowedPairs::All)
                    .unwrap()
                    .to_vec::<f64>()
                    .unwrap()[0]
            };
            let fd = (f_eval(eps) - f_eval(-eps)) / (2.0 * eps);
            let ad = grad_vec[idx];
            let err = (ad - fd).abs();
            assert!(
                err < 1e-4,
                "backward grad[{idx}] = {ad}, finite diff = {fd}, err = {err}"
            );
        }
    });
}
