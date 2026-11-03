use std::sync::Arc;

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{
    tensor::RandomScalar, DynIndex, ExecutionContext, FactorizeAlg, IdxTensor, SvdTruncationPolicy,
};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
use tensor4all_treetn::{
    contraction::{contract, contract_with_rng, contract_with_rng_in, ContractionOptions},
    TreeTN,
};

fn fixture<T: RandomScalar>() -> (TreeTN<IdxTensor, usize>, TreeTN<IdxTensor, usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(1108);
    let shared: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let mut make = || {
        let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(3)).collect();
        let tensors = (0..4)
            .map(|site| {
                let mut indices = vec![shared[site].clone(), DynIndex::new_dyn(2)];
                if site > 0 {
                    indices.push(bonds[site - 1].clone());
                }
                if site < 3 {
                    indices.push(bonds[site].clone());
                }
                IdxTensor::random::<T, _>(&mut rng, indices).unwrap()
            })
            .collect();
        TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
    };
    (make(), make())
}

fn check<T: RandomScalar>() {
    let (a, b) = fixture::<T>();
    let exact = a.contract_naive(&b).unwrap();
    for options in [
        ContractionOptions::zipup(),
        ContractionOptions::fit().with_nfullsweeps(2),
    ] {
        let options = options
            .with_max_bond_dim(64)
            .with_factorize_alg(FactorizeAlg::RSVD);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let result = contract_with_rng(&a, &b, &0, options.clone(), &mut rng)
            .unwrap()
            .to_dense()
            .unwrap();
        let residual = result.sub(&exact).unwrap().norm().unwrap() / exact.norm().unwrap();
        assert!(
            residual < 1e-11,
            "full-rank RSVD contraction residual {residual}"
        );
        assert!(rng.get_word_pos() > 0);
        let seeded = contract(&a, &b, &0, options.clone())
            .unwrap()
            .to_dense()
            .unwrap();
        assert_eq!(result.sub(&seeded).unwrap().maxabs().unwrap(), 0.);
        let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
            CpuBackend::with_threads(1).unwrap(),
        )));
        let mut scoped_a = a.clone();
        let mut scoped_b = b.clone();
        scoped_a.adopt_into_in(&context).unwrap();
        scoped_b.adopt_into_in(&context).unwrap();
        let scoped = contract_with_rng_in(&scoped_a, &scoped_b, &0, options, &mut rng, &context)
            .unwrap()
            .to_dense()
            .unwrap();
        let scoped =
            IdxTensor::from_dense(scoped.indices().to_vec(), scoped.to_vec::<T>().unwrap())
                .unwrap();
        let residual = scoped.sub(&exact).unwrap().norm().unwrap() / exact.norm().unwrap();
        assert!(
            residual < 1e-11,
            "context RSVD contraction residual {residual}"
        );
    }
}

#[test]
fn fit_and_zipup_real() {
    check::<f64>();
}
#[test]
fn fit_and_zipup_complex() {
    check::<Complex64>();
}

#[test]
fn truncated_results_compare_with_deterministic_path() {
    let (a, b) = fixture::<Complex64>();
    let exact = a.contract_naive(&b).unwrap();
    for options in [
        ContractionOptions::zipup(),
        ContractionOptions::fit().with_nfullsweeps(2),
    ] {
        let options = options.with_max_bond_dim(2);
        let deterministic = contract(&a, &b, &0, options.clone())
            .unwrap()
            .to_dense()
            .unwrap();
        let random = contract(&a, &b, &0, options.with_factorize_alg(FactorizeAlg::RSVD))
            .unwrap()
            .to_dense()
            .unwrap();
        let optimal = deterministic.sub(&exact).unwrap().norm().unwrap();
        let approximate = random.sub(&exact).unwrap().norm().unwrap();
        assert!(
            approximate <= 2.0 * optimal,
            "RSVD error {approximate}, deterministic error {optimal}"
        );
    }
}

#[test]
fn invalid_policies_and_missing_caps_leave_rng_untouched() {
    let (a, b) = fixture::<f64>();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for base in [
        ContractionOptions::zipup(),
        ContractionOptions::fit().with_nfullsweeps(0),
    ] {
        let base = base.with_factorize_alg(FactorizeAlg::RSVD);
        for options in [
            base.clone(),
            base.clone().with_max_bond_dim(0),
            base.with_max_bond_dim(2)
                .with_svd_policy(SvdTruncationPolicy::new(1e-8)),
        ] {
            assert!(contract_with_rng(&a, &b, &0, options, &mut rng).is_err());
            assert_eq!(rng.get_word_pos(), 0);
        }
    }
}

#[test]
fn long_separable_product_stays_a_tensor_network() {
    let length = 40;
    let shared: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(2)).collect();
    let make = || {
        let bonds: Vec<_> = (1..length).map(|_| DynIndex::new_dyn(1)).collect();
        let tensors = (0..length)
            .map(|site| {
                let mut indices = vec![shared[site].clone(), DynIndex::new_dyn(2)];
                if site > 0 {
                    indices.push(bonds[site - 1].clone());
                }
                if site + 1 < length {
                    indices.push(bonds[site].clone());
                }
                IdxTensor::from_dense(indices, vec![1.; 4]).unwrap()
            })
            .collect();
        TreeTN::from_tensors(tensors, (0..length).collect()).unwrap()
    };
    let (a, b) = (make(), make());
    for options in [ContractionOptions::zipup(), ContractionOptions::fit()] {
        let options = options
            .with_max_bond_dim(1)
            .with_factorize_alg(FactorizeAlg::RSVD);
        let mut result = contract(&a, &b, &0, options).unwrap();
        assert_eq!(result.node_count(), length);
        assert!(result.link_dims().iter().all(|&dim| dim == 1));
        let expected_norm = 2.0_f64.powi(2 * length as i32);
        let residual = (result.norm().unwrap() / expected_norm - 1.).abs();
        assert!(
            residual < 1e-11,
            "separable product norm relative residual={residual}"
        );
    }
}

#[test]
fn direct_fit_and_sum_fit_use_rsvd() {
    use tensor4all_core::AnyScalar;
    use tensor4all_treetn::{contract_fit, fit_sum, FitOptions};
    let (a, b) = fixture::<Complex64>();
    let exact = a.contract_naive(&b).unwrap();
    let options = FitOptions::new(1)
        .with_factorize_alg(FactorizeAlg::RSVD)
        .with_max_bond_dim(64);
    let fitted = contract_fit(&a, &b, &0, options.clone()).unwrap();
    let dense = fitted.to_dense().unwrap();
    assert!(dense.sub(&exact).unwrap().norm().unwrap() / exact.norm().unwrap() < 1e-11);
    let summed = fit_sum(&[fitted.clone(), fitted.clone()], &fitted, &0, options)
        .unwrap()
        .to_dense()
        .unwrap();
    let expected = exact.scale(AnyScalar::new_real(2.)).unwrap();
    assert!(summed.sub(&expected).unwrap().norm().unwrap() / expected.norm().unwrap() < 1e-11);
}

#[test]
fn context_refuses_tracked_networks_before_rng_consumption() {
    let (mut a, mut b) = fixture::<f64>();
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )));
    a.adopt_into_in(&context).unwrap();
    b.adopt_into_in(&context).unwrap();
    let node = a.node_indices()[0];
    let tensor = a.tensor_mut(node).unwrap();
    *tensor = tensor.clone().enable_grad().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    for options in [ContractionOptions::zipup(), ContractionOptions::fit()] {
        let options = options
            .with_factorize_alg(FactorizeAlg::RSVD)
            .with_max_bond_dim(2);
        let error = contract_with_rng_in(&a, &b, &0, options, &mut rng, &context).unwrap_err();
        assert!(matches!(
            error
                .source
                .downcast_ref::<tensor4all_core::FactorizeError>(),
            Some(tensor4all_core::FactorizeError::UnsupportedStorage(_))
        ));
        assert_eq!(rng.get_word_pos(), 0);
    }
}
