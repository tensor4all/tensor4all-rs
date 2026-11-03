use std::sync::Arc;

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::tensor::RandomScalar;
use tensor4all_core::{
    factorize, factorize_with_rng, factorize_with_rng_in, factorize_with_rng_on_session,
    randomized_range, randomized_svd, Canonical, DynIndex, ExecutionContext, FactorizeError,
    FactorizeOptions, IdxTensor, IndexLike, RangeOptions, RsvdOptions, SvdTruncationPolicy,
    TensorContractionLike,
};
use tensor4all_tensorbackend::{explicit::Session, CpuBackend, CpuExecutionContext};

fn fixture<T: RandomScalar>() -> (IdxTensor, DynIndex) {
    let mut rng = ChaCha8Rng::seed_from_u64(1108);
    let i = DynIndex::new_dyn(7);
    let j = DynIndex::new_dyn(5);
    let r = DynIndex::new_dyn(2);
    let x = IdxTensor::random::<T, _>(&mut rng, vec![i.clone(), r.clone()]).unwrap();
    let y = IdxTensor::random::<T, _>(&mut rng, vec![r, j]).unwrap();
    (x.contract_pair(&y).unwrap(), i)
}

fn check_exact<T: RandomScalar>() {
    let (a, i) = fixture::<T>();
    for q in [0, 1, 2] {
        let mut rng = ChaCha8Rng::seed_from_u64(14);
        let options = RsvdOptions {
            max_bond_dim: 2,
            range: RangeOptions {
                oversampling: 1,
                power_iterations: q,
            },
        };
        let result = randomized_svd(&a, std::slice::from_ref(&i), &options, &mut rng).unwrap();
        let recovered = result
            .u
            .contract_pair(&result.s)
            .unwrap()
            .contract_pair(&result.vh)
            .unwrap();
        let residual = a.sub(&recovered).unwrap().norm().unwrap() / a.norm().unwrap();
        assert!(residual < 1e-12, "RSVD q={q} relative residual={residual}");
        assert_eq!(result.singular_values.len(), 2);
        assert!(
            result.timings.gemm + result.timings.qr + result.timings.small_svd
                <= result.timings.total
        );
        assert!(result.singular_values[0] >= result.singular_values[1]);
        for canonical in [Canonical::Left, Canonical::Right] {
            let mut options = FactorizeOptions::rsvd(2).with_canonical(canonical);
            options.randomized.power_iterations = q;
            let result =
                factorize_with_rng(&a, std::slice::from_ref(&i), &options, &mut rng).unwrap();
            let recovered = result.left.contract_pair(&result.right).unwrap();
            let residual = a.sub(&recovered).unwrap().norm().unwrap() / a.norm().unwrap();
            assert!(
                residual < 1e-12,
                "canonical={canonical:?} residual={residual}"
            );
        }
    }
}

#[test]
fn exact_low_rank_real() {
    check_exact::<f64>();
}
#[test]
fn exact_low_rank_complex() {
    check_exact::<Complex64>();
}

fn check_projector<T: RandomScalar>() {
    let (a, i) = fixture::<T>();
    let mut rng = ChaCha8Rng::seed_from_u64(14);
    let mut reference_rng = rng.clone();
    IdxTensor::random::<T, _>(
        &mut reference_rng,
        vec![a.indices()[1].clone(), DynIndex::new_dyn(2)],
    )
    .unwrap();
    let q = randomized_range(
        &a,
        std::slice::from_ref(&i),
        2,
        &RangeOptions {
            oversampling: 0,
            power_iterations: 1,
        },
        &mut rng,
    )
    .unwrap();
    assert_eq!(rng.get_word_pos(), reference_rng.get_word_pos());
    let bond = q.indices().last().unwrap();
    let new_bond = bond.sim();
    let gram = q
        .conj()
        .contract_pair(&q.replaceind(bond, &new_bond).unwrap())
        .unwrap();
    let expected =
        IdxTensor::from_dense(vec![bond.clone(), new_bond], vec![1., 0., 0., 1.]).unwrap();
    assert!(gram.sub(&expected).unwrap().maxabs().unwrap() < 1e-12);
    let projected = q
        .contract_pair(&q.conj().contract_pair(&a).unwrap())
        .unwrap();
    let residual = a.sub(&projected).unwrap().norm().unwrap() / a.norm().unwrap();
    assert!(residual < 1e-12, "projector relative residual={residual}");

    // Insert Q Q† on a shared input bond before contracting the two tensors.
    let output = DynIndex::new_dyn(3);
    let b = IdxTensor::random::<T, _>(&mut rng, vec![i, output]).unwrap();
    let exact = a.contract_pair(&b).unwrap();
    // A^T B = (A^T conj(Q)) (Q^T B), since Q Q† A = A.
    let compressed_left = a.contract_pair(&q.conj()).unwrap();
    let compressed_right = q.contract_pair(&b).unwrap();
    let projected = compressed_left.contract_pair(&compressed_right).unwrap();
    let residual = exact.sub(&projected).unwrap().norm().unwrap() / exact.norm().unwrap();
    assert!(
        residual < 1e-12,
        "projected two-site contraction residual={residual}"
    );
}

#[test]
fn bond_projector_real() {
    check_projector::<f64>();
}
#[test]
fn bond_projector_complex() {
    check_projector::<Complex64>();
}

#[test]
fn decaying_spectrum_matches_deterministic_error() {
    let i = DynIndex::new_dyn(24);
    let j = DynIndex::new_dyn(24);
    let mut data = vec![0.; 24 * 24];
    for p in 0..24 {
        data[p + 24 * p] = 0.6_f64.powi(p as i32);
    }
    let a = IdxTensor::from_dense(vec![i.clone(), j], data).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(13);
    let random = factorize_with_rng(
        &a,
        std::slice::from_ref(&i),
        &FactorizeOptions::rsvd(5),
        &mut rng,
    )
    .unwrap();
    let deterministic = factorize(&a, &[i], &FactorizeOptions::svd().with_max_bond_dim(5)).unwrap();
    let error = |r: tensor4all_core::FactorizeResult<IdxTensor>| {
        a.sub(&r.left.contract_pair(&r.right).unwrap())
            .unwrap()
            .norm()
            .unwrap()
    };
    let optimal = error(deterministic);
    let approximate = error(random);
    assert!(approximate >= optimal * (1. - 1e-12));
    assert!(
        approximate <= 1.02 * optimal,
        "RSVD error {approximate}, optimal {optimal}"
    );
}

#[test]
fn reproducible_stream_and_no_randomness_for_deterministic_calls() {
    let (a, i) = fixture::<Complex64>();
    let mut rng = ChaCha8Rng::seed_from_u64(14);
    let mut reference = rng.clone();
    for _ in 0..2 {
        let options = FactorizeOptions::rsvd(2);
        let x = factorize_with_rng(&a, std::slice::from_ref(&i), &options, &mut rng).unwrap();
        let y = factorize_with_rng(&a, std::slice::from_ref(&i), &options, &mut reference).unwrap();
        let x = x.left.contract_pair(&x.right).unwrap();
        let y = y.left.contract_pair(&y.right).unwrap();
        assert_eq!(x.sub(&y).unwrap().maxabs().unwrap(), 0.);
    }
    assert!(rng.get_word_pos() > 0);
    assert_eq!(rng.get_word_pos(), reference.get_word_pos());
    factorize_with_rng(&a, &[i], &FactorizeOptions::svd(), &mut rng).unwrap();
    assert_eq!(rng.get_word_pos(), reference.get_word_pos());
}

#[test]
fn invalid_inputs_leave_rng_untouched() {
    let (a, i) = fixture::<f64>();
    let mut rng = ChaCha8Rng::seed_from_u64(14);
    for left in [
        vec![],
        a.indices().to_vec(),
        vec![i.clone(), i.clone()],
        vec![i.prime()],
    ] {
        assert!(randomized_svd(&a, &left, &RsvdOptions::new(2), &mut rng).is_err());
    }
    for rank in [0, usize::MAX] {
        assert!(randomized_svd(
            &a,
            std::slice::from_ref(&i),
            &RsvdOptions::new(rank),
            &mut rng
        )
        .is_err());
    }
    for canonical in [Canonical::Left, Canonical::Right] {
        let options = FactorizeOptions::rsvd(2)
            .with_canonical(canonical)
            .with_svd_policy(SvdTruncationPolicy::new(1e-8));
        assert!(matches!(
            factorize_with_rng(&a, std::slice::from_ref(&i), &options, &mut rng),
            Err(FactorizeError::UnsupportedTruncationPolicy(_))
        ));
    }
    assert_eq!(rng.get_word_pos(), 0);
}

fn check_routes<T: RandomScalar>() {
    let (a, i) = fixture::<T>();
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )));
    let a =
        IdxTensor::from_dense_in(&context, a.indices().to_vec(), a.to_vec::<T>().unwrap()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(14);
    let options = FactorizeOptions::rsvd(2);
    let result =
        factorize_with_rng_in(&a, std::slice::from_ref(&i), &options, &mut rng, &context).unwrap();
    result.left.validate_context(&context).unwrap();
    result.right.validate_context(&context).unwrap();
    let reconstructed = result.left.contract_pair(&result.right).unwrap();
    let actual = IdxTensor::from_dense(
        reconstructed.indices().to_vec(),
        reconstructed.to_vec::<T>().unwrap(),
    )
    .unwrap();
    let expected = IdxTensor::from_dense(a.indices().to_vec(), a.to_vec::<T>().unwrap()).unwrap();
    assert!(expected.sub(&actual).unwrap().norm().unwrap() / expected.norm().unwrap() < 1e-12);
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let result = {
        let mut session = Session::new(backend.engine_mut());
        factorize_with_rng_on_session(&mut session, &a, &[i], &options, &mut rng).unwrap()
    };
    let reconstructed = result.left.contract_pair(&result.right).unwrap();
    let actual = IdxTensor::from_dense(
        reconstructed.indices().to_vec(),
        reconstructed.to_vec::<T>().unwrap(),
    )
    .unwrap();
    let expected = IdxTensor::from_dense(a.indices().to_vec(), a.to_vec::<T>().unwrap()).unwrap();
    assert!(expected.sub(&actual).unwrap().norm().unwrap() / expected.norm().unwrap() < 1e-12);
}

#[test]
fn context_and_session_real() {
    check_routes::<f64>();
}
#[test]
fn context_and_session_complex() {
    check_routes::<Complex64>();
}

#[test]
fn zero_matrix_keeps_requested_rank_and_oversized_rank_is_capped() {
    let i = DynIndex::new_dyn(3);
    let j = DynIndex::new_dyn(4);
    let a = IdxTensor::from_dense(vec![i.clone(), j], vec![0.; 12]).unwrap();
    for rank in [2, 9] {
        let result =
            factorize(&a, std::slice::from_ref(&i), &FactorizeOptions::rsvd(rank)).unwrap();
        assert_eq!(result.rank, rank.min(3));
        assert_eq!(
            result
                .left
                .contract_pair(&result.right)
                .unwrap()
                .maxabs()
                .unwrap(),
            0.
        );
    }
}

#[test]
fn partial_rsvd_gradient_matches_finite_differences() {
    let data = vec![2., 0.5, 0.2, 0.7, 3., 0.4];
    let options = {
        let mut options = FactorizeOptions::rsvd(1);
        options.randomized.oversampling = 0;
        options
    };
    let evaluate = |data: Vec<f64>, tracked: bool| {
        let i = DynIndex::new_dyn(3);
        let j = DynIndex::new_dyn(2);
        let a = IdxTensor::from_dense(vec![i.clone(), j], data).unwrap();
        let a = if tracked { a.enable_grad().unwrap() } else { a };
        let result = factorize(&a, &[i], &options).unwrap();
        let loss = result
            .left
            .contract_pair(&result.right)
            .unwrap()
            .sum()
            .unwrap();
        (a, loss)
    };
    let (a, loss) = evaluate(data.clone(), true);
    assert!(loss.tracks_grad());
    loss.backward().unwrap();
    let grad = a.grad().unwrap().unwrap().to_vec::<f64>().unwrap();
    for position in 0..data.len() {
        let mut plus = data.clone();
        let mut minus = data.clone();
        plus[position] += 1e-6;
        minus[position] -= 1e-6;
        let expected = (evaluate(plus, false).1.real() - evaluate(minus, false).1.real()) / 2e-6;
        assert!(
            (grad[position] - expected).abs() < 2e-5,
            "gradient[{position}]={}, finite difference={expected}",
            grad[position]
        );
    }
}

#[test]
fn explicit_session_rejects_tracked_inputs_without_consuming_randomness() {
    let (a, i) = fixture::<f64>();
    let a = a.enable_grad().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    let error =
        factorize_with_rng_on_session(&mut session, &a, &[i], &FactorizeOptions::rsvd(2), &mut rng)
            .unwrap_err();
    assert!(matches!(error, FactorizeError::UnsupportedStorage(_)));
    assert_eq!(rng.get_word_pos(), 0);
}

#[test]
fn foreign_context_is_rejected_before_randomness() {
    let (a, i) = fixture::<f64>();
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    assert!(
        factorize_with_rng_in(&a, &[i], &FactorizeOptions::rsvd(2), &mut rng, &context).is_err()
    );
    assert_eq!(rng.get_word_pos(), 0);
}

#[test]
fn rectangular_permuted_layout_and_same_id_prime_indices() {
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let i = DynIndex::new_dyn(3);
    let j = i.prime();
    let k = DynIndex::new_dyn(2);
    let a =
        IdxTensor::random::<Complex64, _>(&mut rng, vec![i.clone(), j.clone(), k.clone()]).unwrap();
    let a = a.permute_indices(&[k, j, i.clone()]).unwrap();
    for canonical in [Canonical::Left, Canonical::Right] {
        let options = FactorizeOptions::rsvd(3).with_canonical(canonical);
        let result = factorize_with_rng(&a, std::slice::from_ref(&i), &options, &mut rng).unwrap();
        let recovered = result.left.contract_pair(&result.right).unwrap();
        let residual = a.sub(&recovered).unwrap().norm().unwrap() / a.norm().unwrap();
        assert!(
            residual < 1e-12,
            "permuted same-ID split residual {residual}"
        );
    }
}

#[test]
fn every_threshold_policy_is_unsupported_with_a_remedy() {
    let (a, i) = fixture::<f64>();
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    for absolute in [false, true] {
        for squared in [false, true] {
            for tail_sum in [false, true] {
                let mut policy = SvdTruncationPolicy::new(1e-8);
                if absolute {
                    policy = policy.with_absolute();
                }
                if squared {
                    policy = policy.with_squared_values();
                }
                if tail_sum {
                    policy = policy.with_discarded_tail_sum();
                }
                let options = FactorizeOptions::rsvd(2).with_svd_policy(policy);
                let error = factorize_with_rng(&a, std::slice::from_ref(&i), &options, &mut rng)
                    .unwrap_err();
                assert!(
                    matches!(error, FactorizeError::UnsupportedTruncationPolicy(p) if p == policy)
                );
                assert!(error.to_string().contains("fixed max_bond_dim"));
                assert!(error.to_string().contains("deterministic SVD"));
            }
        }
    }
    assert_eq!(rng.get_word_pos(), 0);
}

#[test]
fn range_context_and_session_projectors() {
    use tensor4all_core::{randomized_range_in, randomized_range_on_session};
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )));
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let a = IdxTensor::from_dense_in(
        &context,
        vec![i.clone(), j],
        vec![
            Complex64::new(3., 2.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
        ],
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let options = RangeOptions {
        oversampling: 0,
        power_iterations: 1,
    };
    let q = randomized_range_in(
        &a,
        std::slice::from_ref(&i),
        1,
        &options,
        &mut rng,
        &context,
    )
    .unwrap();
    q.validate_context(&context).unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let qs = {
        let mut session = Session::new(backend.engine_mut());
        randomized_range_on_session(&mut session, &a, &[i], 1, &options, &mut rng).unwrap()
    };
    for q in [q, qs] {
        let values = q.to_vec::<Complex64>().unwrap();
        assert!((values[0].norm() - 1.).abs() < 1e-12 && values[1].norm() < 1e-12);
    }
}
