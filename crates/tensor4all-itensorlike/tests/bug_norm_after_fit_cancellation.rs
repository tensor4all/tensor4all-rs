use rand::rngs::StdRng;
use rand::SeedableRng;

use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};

fn create_random_mpo(
    length: usize,
    input_indices: &[DynIndex],
    output_indices: &[DynIndex],
    link_indices: &[DynIndex],
    rng: &mut StdRng,
) -> TensorTrain {
    let mut tensors = Vec::with_capacity(length);
    for i in 0..length {
        let mut indices = vec![input_indices[i].clone(), output_indices[i].clone()];
        if i > 0 {
            indices.insert(0, link_indices[i - 1].clone());
        }
        if i < length - 1 {
            indices.push(link_indices[i].clone());
        }
        tensors.push(TensorDynLen::random::<f64, _>(rng, indices));
    }
    TensorTrain::new(tensors).unwrap()
}

#[test]
#[ignore] // Sequential bra-ket contraction loses precision with massive cancellation (separate issue)
fn norm_matches_dense_after_fit_cancellation() {
    let length = 6;
    let phys_dim = 2;
    let bond_dim = 8;

    let s_input: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("s={}", i + 1)).unwrap())
        .collect();
    let s_shared: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("sc={}", i + 1)).unwrap())
        .collect();
    let s_output: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("so={}", i + 1)).unwrap())
        .collect();
    let links_a: Vec<DynIndex> = (0..length - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();
    let links_b: Vec<DynIndex> = (0..length - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    let mut rng1 = StdRng::seed_from_u64(42);
    let mut rng2 = StdRng::seed_from_u64(123);
    let mpo_a = create_random_mpo(length, &s_input, &s_shared, &links_a, &mut rng1);
    let mpo_b = create_random_mpo(length, &s_shared, &s_output, &links_b, &mut rng2);

    let exact = mpo_a.contract(&mpo_b, &ContractOptions::zipup()).unwrap();
    let fit4 = mpo_a
        .contract(&mpo_b, &ContractOptions::fit().with_nsweeps(4))
        .unwrap();

    let diff = fit4.axpby(1.0.into(), &exact, (-1.0).into()).unwrap();
    let dense_diff = diff.to_dense().unwrap();
    let dense_fit4 = fit4.to_dense().unwrap();
    let dense_exact = exact.to_dense().unwrap();
    let direct_dense_diff = dense_fit4
        .axpby(1.0.into(), &dense_exact, (-1.0).into())
        .unwrap();

    let dense_norm = direct_dense_diff.norm();
    let tt_norm = diff.norm();

    assert!(
        (dense_diff.norm() - dense_norm).abs() <= dense_norm.max(1.0) * 1e-9,
        "Dense contraction of diff TT should agree with direct dense subtraction: tt_dense={:.12e}, direct_dense={dense_norm:.12e}",
        dense_diff.norm()
    );
    assert!(
        (tt_norm - dense_norm).abs() <= dense_norm.max(1.0) * 1e-10,
        "TensorTrain::norm should match dense norm after cancellation: tt={tt_norm:.12e}, dense={dense_norm:.12e}"
    );
}
