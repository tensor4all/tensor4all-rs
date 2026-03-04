use std::sync::Arc;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::storage::DenseStorageF64;
use tensor4all_core::{AllowedPairs, DynIndex, Storage, TensorDynLen, TensorLike};

fn make_tensor(indices: Vec<DynIndex>, data: Vec<f64>, dims: &[usize]) -> TensorDynLen {
    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
        data, dims,
    )));
    TensorDynLen::new(indices, storage)
}

#[test]
fn test_contract_multi_pair_matches_binary_contract() {
    let l01 = Index::new_dyn(3);
    let s1 = Index::new_dyn(2);
    let l12 = Index::new_dyn(3);
    let s2 = Index::new_dyn(2);

    // t1[l01, s1, l12]
    let t1 = make_tensor(
        vec![l01.clone(), s1.clone(), l12.clone()],
        (1..=18).map(|x| x as f64).collect(),
        &[3, 2, 3],
    );
    // t2[l12, s2]
    let t2 = make_tensor(
        vec![l12.clone(), s2.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[3, 2],
    );

    let binary = t1.contract(&t2);
    let multi =
        <TensorDynLen as TensorLike>::contract(&[&t1, &t2], AllowedPairs::All).expect("contract");

    assert!(
        multi.isapprox(&binary, 1e-12, 0.0),
        "multi-contract and binary contract differ: maxabs diff = {}",
        (&multi - &binary).maxabs()
    );
}

#[test]
fn test_contract_multi_three_matches_sequential_binary_contract() {
    let i = Index::new_dyn(2);
    let a = Index::new_dyn(3);
    let b = Index::new_dyn(2);
    let c = Index::new_dyn(3);
    let k = Index::new_dyn(2);

    let t0 = make_tensor(
        vec![i.clone(), a.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[2, 3],
    );
    let t1 = make_tensor(
        vec![a.clone(), b.clone(), c.clone()],
        (1..=18).map(|x| x as f64).collect(),
        &[3, 2, 3],
    );
    let t2 = make_tensor(
        vec![c.clone(), k.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[3, 2],
    );

    let sequential = t0.contract(&t1).contract(&t2);
    let multi = <TensorDynLen as TensorLike>::contract(&[&t0, &t1, &t2], AllowedPairs::All)
        .expect("contract");

    assert!(
        multi.isapprox(&sequential, 1e-12, 0.0),
        "3-tensor multi-contract and sequential contract differ: maxabs diff = {}",
        (&multi - &sequential).maxabs()
    );
}
