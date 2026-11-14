//! The reductions must not project away a non-unit leading bond of the first core.
//!
//! A tensor train whose first core has a leading bond above one is reachable through the public accessors
//! (`site_tensors_mut`, `set_site_tensor`, `SiteTensorTrain::to_tensor_train`). Every reduction seeds its
//! environment from the `l == 0` component of that core, so before this guard `evaluate`, `sum`, `norm2`, `norm`
//! and `log_norm` returned the projection onto that component: a plausible but wrong scalar. This is the
//! left-boundary sibling of #913.

use tensor4all_simplett::{
    tensor3_zeros, AbstractTensorTrain, SimpleTensorTrain, SimpleTensorTrainError, SiteTensorTrain,
    Tensor3Ops,
};

/// A valid constant train, for the control assertions.
fn valid() -> SimpleTensorTrain<f64> {
    SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0)
}

/// The same train with its first core replaced through the public mutable accessor, so its leading bond is 2.
///
/// Every entry is set so that the wrong answer would be visible rather than accidentally equal: the `l == 0`
/// component holds 1, the `l == 1` component holds 2.
fn with_a_broken_leading_bond() -> SimpleTensorTrain<f64> {
    let mut site = SiteTensorTrain::from_tensor_train(&valid(), 0).expect("site tensor train");
    let cores = site.site_tensors_mut();
    let mut core = tensor3_zeros::<f64>(2, 2, 1).unwrap();
    for physical in 0..2 {
        core.set3(0, physical, 0, 1.0);
        core.set3(1, physical, 0, 2.0);
    }
    cores[0] = core;
    site.to_tensor_train()
}

#[test]
fn a_valid_train_still_reports_the_same_reductions() {
    let tt = valid();
    assert!((tt.sum().expect("sum") - 6.0).abs() < 1e-12);
    assert!((tt.norm2().expect("norm2") - 6.0).abs() < 1e-12);
    assert!((tt.norm().expect("norm") - 6.0_f64.sqrt()).abs() < 1e-12);
    assert!((tt.log_norm().expect("log_norm") - 0.5 * 6.0_f64.ln()).abs() < 1e-12);
    assert!((tt.evaluate(&[0, 0]).expect("evaluate") - 1.0).abs() < 1e-12);
}

#[test]
fn the_reductions_refuse_a_broken_leading_bond_instead_of_projecting() {
    let tt = with_a_broken_leading_bond();

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
            other => panic!("{name} must refuse a broken leading bond, got {other:?}"),
        }
    }
}
