//! Tests for reverse-mode automatic differentiation through TreeTN operations.

use tensor4all_core::{contract_multi, AllowedPairs, DynIndex, IndexLike, TensorDynLen};
use tensor4all_treetn::TreeTN;

fn make_three_site_mps_data() -> (Vec<Vec<DynIndex>>, Vec<Vec<f64>>) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(2);

    let index_sets = vec![
        vec![s0, bond01.clone()],
        vec![bond01, s1, bond12.clone()],
        vec![bond12, s2],
    ];

    let data = vec![
        vec![1.0, 0.5, 0.3, 2.0],
        vec![3.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 2.0],
        vec![1.5, 0.2, 0.4, 1.0],
    ];

    (index_sets, data)
}

fn with_runtime<R>(f: impl FnOnce() -> R) -> R {
    let _guard = tenferro::set_default_runtime(tenferro::RuntimeContext::Cpu(
        tenferro_prims::CpuContext::new(1),
    ));
    f()
}

#[test]
fn backward_ad_to_dense_propagates_gradients() {
    with_runtime(|| {
        let (index_sets, data) = make_three_site_mps_data();

        let tensors: Vec<TensorDynLen> = index_sets
            .iter()
            .zip(&data)
            .map(|(idx, d)| {
                let mut t = TensorDynLen::from_dense(idx.clone(), d.clone()).unwrap();
                t.set_requires_grad(true).unwrap();
                t
            })
            .collect();

        let ttn = TreeTN::from_tensors(tensors, vec![0, 1, 2]).unwrap();
        let dense = ttn.to_dense().unwrap();

        // Contract dense with ones to get scalar = sum(dense)
        let ones = TensorDynLen::from_dense(
            dense.indices().to_vec(),
            vec![1.0; dense.indices().iter().map(|i| i.dim()).product::<usize>()],
        )
        .unwrap();
        let scalar = contract_multi(&[&dense, &ones], AllowedPairs::All).unwrap();

        scalar.backward(None).unwrap();

        for (i, &ni) in ttn.node_indices().iter().enumerate() {
            let grad = ttn.tensor(ni).unwrap().grad().unwrap();
            assert!(grad.is_some(), "node {i} has no gradient after backward");
        }
    });
}

#[test]
fn backward_ad_gradient_matches_finite_diff() {
    with_runtime(|| {
        let (index_sets, data) = make_three_site_mps_data();
        let eps = 1e-6;

        let tensors: Vec<TensorDynLen> = index_sets
            .iter()
            .zip(&data)
            .map(|(idx, d)| {
                let mut t = TensorDynLen::from_dense(idx.clone(), d.clone()).unwrap();
                t.set_requires_grad(true).unwrap();
                t
            })
            .collect();

        let ttn = TreeTN::from_tensors(tensors, vec![0, 1, 2]).unwrap();
        let dense = ttn.to_dense().unwrap();

        let ones = TensorDynLen::from_dense(
            dense.indices().to_vec(),
            vec![1.0; dense.indices().iter().map(|i| i.dim()).product::<usize>()],
        )
        .unwrap();
        let scalar = contract_multi(&[&dense, &ones], AllowedPairs::All).unwrap();

        scalar.backward(None).unwrap();

        // Verify gradient of first node vs finite difference
        let node_0 = ttn.node_index(&0).unwrap();
        let grad_t0 = ttn
            .tensor(node_0)
            .unwrap()
            .grad()
            .unwrap()
            .expect("t0 gradient missing");
        let grad_t0_vec = grad_t0.to_vec::<f64>().unwrap();

        for elem_idx in 0..data[0].len() {
            let make_perturbed_sum = |delta: f64| -> f64 {
                let tensors: Vec<TensorDynLen> = index_sets
                    .iter()
                    .zip(&data)
                    .enumerate()
                    .map(|(i, (idx, d))| {
                        let mut d = d.clone();
                        if i == 0 {
                            d[elem_idx] += delta;
                        }
                        TensorDynLen::from_dense(idx.clone(), d).unwrap()
                    })
                    .collect();
                let ttn = TreeTN::from_tensors(tensors, vec![0, 1, 2]).unwrap();
                ttn.to_dense().unwrap().sum().real()
            };

            let fd = (make_perturbed_sum(eps) - make_perturbed_sum(-eps)) / (2.0 * eps);
            let ad = grad_t0_vec[elem_idx];
            let err = (ad - fd).abs();
            assert!(
                err < 1e-4,
                "backward grad[0][{elem_idx}] = {ad}, finite diff = {fd}, err = {err}"
            );
        }
    });
}
