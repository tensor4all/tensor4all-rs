//! Full-rank CI factorization on an explicit session.
//!
//! `factorize_full_rank_on_session` serves `FactorizeAlg::CI` with the
//! session's pivoted LU, and the run enters no process-global runtime (it
//! completes while the default engine is borrowed). The factors equal those of
//! `factorize_full_rank` bit for bit for `f64`, `f32`, `Complex64` and
//! `Complex32`, on both canonical sides, and reconstruct the tensor to working
//! precision. Both routes' triangular solves promote the narrow scalar types to
//! 64 bits the same way (tensor4all-rs#1244). LU keeps its typed refusal.

use num_complex::{Complex32, Complex64};
use tensor4all_core::{
    factorize_full_rank, factorize_full_rank_on_session, Canonical, DynIndex, FactorizeAlg,
    FactorizeError, IdxTensor, TensorContractionLike, TensorElement,
};
use tensor4all_tensorbackend::{
    explicit::Session, with_default_backend, CpuBackend, StorageScalar,
};

fn values<T: TensorElement + StorageScalar>(tensor: &IdxTensor) -> Vec<T> {
    tensor.dense_host_values::<T>().expect("host values")
}

fn check<T>(scalar: fn(f64, f64) -> T)
where
    T: TensorElement + StorageScalar + PartialEq + std::fmt::Debug,
{
    // A rank-deficient 3 x 2 x 4 tensor split as (i, k) | (j): rank two.
    let (i, j, k) = (
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(4),
    );
    let data: Vec<T> = (0..24)
        .map(|n| {
            let (a, b) = ((n % 3) as f64, (n / 12) as f64);
            scalar(1.0 + a * (1.0 + b) + 0.25 * (n % 6) as f64, 0.5 * a - b)
        })
        .collect();
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();
    let left = [i.clone(), k.clone()];
    for canonical in [Canonical::Left, Canonical::Right] {
        let global = factorize_full_rank(&tensor, &left, FactorizeAlg::CI, canonical).unwrap();
        let mut backend = CpuBackend::with_threads(1).unwrap();
        let session = with_default_backend(|_engine| {
            let mut session = Session::new(backend.engine_mut());
            factorize_full_rank_on_session(
                &mut session,
                &tensor,
                &left,
                FactorizeAlg::CI,
                canonical,
            )
            .expect("the session CI factorization must not reach the borrowed default engine")
        });
        assert_eq!(session.rank, global.rank, "{canonical:?}");
        assert_eq!(session.left.dims(), global.left.dims());
        assert_eq!(session.right.dims(), global.right.dims());
        assert_eq!(
            values::<T>(&session.left),
            values::<T>(&global.left),
            "{canonical:?}"
        );
        assert_eq!(
            values::<T>(&session.right),
            values::<T>(&global.right),
            "{canonical:?}"
        );
        // Full rank: the factors reconstruct the tensor to working precision.
        let product = session.left.contract_pair(&session.right).unwrap();
        let residual = product.sub(&tensor).unwrap().maxabs().unwrap();
        let scale = tensor.maxabs().unwrap();
        let epsilon = if tensor.is_f32() || tensor.is_c32() {
            1e-5
        } else {
            1e-13
        };
        assert!(
            residual <= epsilon * scale,
            "{canonical:?}: residual {residual:e}"
        );
    }
}

#[test]
fn session_ci_factors_equal_the_compatibility_route_without_entering_it() {
    check::<f64>(|re, _| re);
    check::<f32>(|re, _| re as f32);
    check::<Complex64>(Complex64::new);
    check::<Complex32>(|re, im| Complex32::new(re as f32, im as f32));
}

#[test]
fn session_lu_is_still_refused() {
    let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let tensor = IdxTensor::from_dense(vec![i.clone(), j], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    let error = factorize_full_rank_on_session(
        &mut session,
        &tensor,
        &[i],
        FactorizeAlg::LU,
        Canonical::Left,
    )
    .expect_err("LU has no session route");
    assert!(
        matches!(error, FactorizeError::UnsupportedStorage(_)),
        "{error:?}"
    );
}
