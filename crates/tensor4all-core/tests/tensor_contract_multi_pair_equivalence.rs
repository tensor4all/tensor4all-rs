use std::sync::Arc;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::storage::DenseStorageF64;
use tensor4all_core::{
    factorize, svd, AllowedPairs, Canonical, DynIndex, FactorizeOptions, Storage, TensorDynLen,
    TensorLike,
};
use tensor4all_tensorbackend::permute_storage_native;

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

#[test]
fn test_contract_multi_pair_matches_binary_contract_for_zero_masked_inputs() {
    let s0 = Index::new_dyn(2);
    let l01 = Index::new_dyn(3);
    let s1 = Index::new_dyn(2);

    let t0 = make_tensor(
        vec![s0, l01.clone()],
        vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        &[2, 3],
    );
    let t1 = make_tensor(vec![l01, s1], (1..=6).map(|x| x as f64).collect(), &[3, 2]);

    let binary = t0.contract(&t1);
    let multi =
        <TensorDynLen as TensorLike>::contract(&[&t0, &t1], AllowedPairs::All).expect("contract");

    assert!(
        multi.isapprox(&binary, 1e-12, 0.0),
        "zero-masked multi-contract and binary contract differ: maxabs diff = {}",
        (&multi - &binary).maxabs()
    );
}

#[test]
fn test_zipup_zero_masked_root_multi_matches_sequential_binary_contract() {
    let s0 = Index::new_dyn(2);
    let s1 = Index::new_dyn(2);
    let s2 = Index::new_dyn(2);
    let l01 = Index::new_dyn(3);
    let l12 = Index::new_dyn(3);

    let a0 = make_tensor(
        vec![s0.clone(), l01.clone()],
        vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        &[2, 3],
    );
    let a1 = make_tensor(
        vec![l01.clone(), s1.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[3, 2],
    );
    let b0 = make_tensor(
        vec![s1.clone(), l12.clone()],
        (1..=6).map(|x| x as f64).collect(),
        &[2, 3],
    );
    let b1 = make_tensor(
        vec![l12.clone(), s2.clone()],
        vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0],
        &[3, 2],
    );

    let leaf = <TensorDynLen as TensorLike>::contract(&[&a0, &b0], AllowedPairs::All)
        .expect("leaf contract");
    let permuted_leaf = leaf.permute_indices(&[s0.clone(), s1.clone(), l01.clone(), l12.clone()]);
    let expected_permuted_storage =
        permute_storage_native(leaf.storage().as_ref(), &leaf.dims(), &[0, 2, 1, 3])
            .expect("expected permute storage");
    let expected_permuted = TensorDynLen::new(
        vec![s0.clone(), s1.clone(), l01.clone(), l12.clone()],
        Arc::new(expected_permuted_storage),
    );
    assert!(
        permuted_leaf.isapprox(&expected_permuted, 1e-12, 0.0),
        "native permute for leaf does not match storage baseline: maxabs diff = {}",
        (&permuted_leaf - &expected_permuted).maxabs()
    );

    let (u, s, v) = svd::<f64>(&leaf, &[s0.clone(), s1.clone()]).expect("svd");
    let vh = v.conj().permute(&[2, 0, 1]);
    let svh = s.contract(&vh);
    let svh = svh.replaceind(
        &s.indices[1].clone(),
        &v.indices[v.indices.len() - 1].clone(),
    );
    let svd_reconstructed = u.contract(&svh);
    assert!(
        svd_reconstructed.isapprox(&leaf, 1e-10, 0.0),
        "svd leaf does not reconstruct: maxabs diff = {}",
        (&svd_reconstructed - &leaf).maxabs()
    );

    let factorized = factorize(
        &leaf,
        &[s0.clone(), s1.clone()],
        &FactorizeOptions::svd().with_canonical(Canonical::Left),
    )
    .expect("factorize");

    let reconstructed_leaf = factorized.left.contract(&factorized.right);
    assert!(
        reconstructed_leaf.isapprox(&leaf, 1e-10, 0.0),
        "factorized leaf does not reconstruct: maxabs diff = {}",
        (&reconstructed_leaf - &leaf).maxabs()
    );

    let sequential = factorized.right.contract(&a1).contract(&b1);
    let multi =
        <TensorDynLen as TensorLike>::contract(&[&factorized.right, &a1, &b1], AllowedPairs::All)
            .expect("root contract");

    assert!(
        multi.isapprox(&sequential, 1e-10, 0.0),
        "zipup root multi-contract and sequential binary contract differ: maxabs diff = {}",
        (&multi - &sequential).maxabs()
    );
}
