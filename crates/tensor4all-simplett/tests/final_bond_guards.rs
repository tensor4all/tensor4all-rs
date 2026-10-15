//! The reductions must not read a single element out of a final environment that is not 1x1.
//!
//! A tensor train whose last core has a non-unit right bond is reachable through the public accessors
//! (`site_tensors_mut`, `set_site_tensor`, `SiteTensorTrain::to_tensor_train`), and `evaluate()` already refuses
//! it. Before this guard, `sum`, `norm2`, `norm` and `log_norm` instead returned the projection onto the `r == 0`
//! component: a plausible but wrong scalar, which is the worst failure mode for a numerics library. Issue #913.

use tensor4all_simplett::{
    tensor3_zeros, AbstractTensorTrain, SimpleTensorTrain, SimpleTensorTrainError, SiteTensorTrain,
    TTCache, Tensor3Ops,
};

/// A valid constant train, for the control assertions.
fn valid() -> SimpleTensorTrain<f64> {
    SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0)
}

/// The same train with its last core replaced through the public mutable accessor, so its final right bond is 2.
///
/// Every entry is set so that the wrong answer would be visible rather than accidentally equal: the `r == 0`
/// component holds 1, the `r == 1` component holds 2.
fn with_a_broken_final_bond() -> SimpleTensorTrain<f64> {
    let mut site = SiteTensorTrain::from_tensor_train(&valid(), 0).expect("site tensor train");
    let cores = site.site_tensors_mut();
    let last = cores.len() - 1;
    let mut core = tensor3_zeros::<f64>(1, 2, 2);
    for physical in 0..2 {
        core.set3(0, physical, 0, 1.0);
        core.set3(0, physical, 1, 2.0);
    }
    cores[last] = core;
    site.to_tensor_train()
}

#[test]
fn a_valid_train_still_reports_the_same_reductions() {
    let tt = valid();
    // The constant train over 2*3 = 6 entries of 1.0: sum 6, squared norm 6, norm sqrt(6), log norm half of ln 6.
    assert!((tt.sum().expect("sum") - 6.0).abs() < 1e-12);
    assert!((tt.norm2().expect("norm2") - 6.0).abs() < 1e-12);
    assert!((tt.norm().expect("norm") - 6.0_f64.sqrt()).abs() < 1e-12);
    assert!((tt.log_norm().expect("log_norm") - 0.5 * 6.0_f64.ln()).abs() < 1e-12);
    // The sibling that already guarded agrees with them.
    assert!((tt.evaluate(&[0, 0]).expect("evaluate") - 1.0).abs() < 1e-12);
}

#[test]
fn the_reductions_refuse_a_broken_final_bond_instead_of_projecting() {
    let tt = with_a_broken_final_bond();
    // The reference the guards must agree with.
    assert!(matches!(
        tt.evaluate(&[0, 0]),
        Err(SimpleTensorTrainError::InvalidOperation { .. })
    ));

    for (name, result) in [
        ("sum", tt.sum().map(|_| ())),
        ("norm2", tt.norm2().map(|_| ())),
        ("norm", tt.norm().map(|_| ())),
        ("log_norm", tt.log_norm().map(|_| ())),
    ] {
        match result {
            Err(SimpleTensorTrainError::InvalidOperation { message }) => {
                assert!(
                    message.contains("expected 1"),
                    "{name} should say what it expected: {message}"
                );
            }
            other => panic!("{name} must refuse a broken final bond, got {other:?}"),
        }
    }
}

#[test]
fn the_guarded_siblings_refuse_it_too() {
    let tt = with_a_broken_final_bond();
    let other = valid();

    // `inner_product` already returned a Result; it must now refuse the malformed environment as well.
    assert!(matches!(
        tt.inner_product(&other),
        Err(SimpleTensorTrainError::InvalidOperation { .. })
    ));

    // `partial_sum` over every site walks the same final environment.
    assert!(matches!(
        tt.partial_sum(&[0, 1]),
        Err(SimpleTensorTrainError::InvalidOperation { .. })
    ));
}

#[test]
fn the_batched_evaluation_refuses_it_instead_of_truncating() {
    let tt = with_a_broken_final_bond();
    let mut cache = TTCache::new(&tt);

    // The batch route pairs a left and a right environment per query. `zip` would truncate the pair to the
    // shorter side - the same projection the reductions must not return - so it has to refuse instead.
    let outcome = cache.evaluate_many(&[vec![0, 0]], Some(1));
    assert!(matches!(
        outcome,
        Err(SimpleTensorTrainError::InvalidOperation { .. })
    ));
}
