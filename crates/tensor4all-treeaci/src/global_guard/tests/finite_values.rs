//! Guard finite-value and coordinate-retention regressions for #854.

use super::super::{find_global_pivots, InputEvaluators};
use crate::{state::TreeAciState, TreeAciOptions, TreeElementwiseBatch};
use rand::SeedableRng;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

fn delta_tree() -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(1);
    TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![DynIndex::new_dyn(2), bond.clone()], vec![1.0_f64, 0.0])
                .unwrap(),
            IdxTensor::from_dense(vec![bond, DynIndex::new_dyn(2)], vec![1.0_f64, 0.0]).unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap()
}

fn nonfinite(value: f64) {
    let inputs = vec![delta_tree()];
    let options = TreeAciOptions {
        scale_tolerance: false,
        ..Default::default()
    };
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut operator = |_batch: TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output.fill(value);
        Ok(())
    };
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut rng,
        &mut operator,
        &[],
    );
    eprintln!("NONFINITE_GUARD value={value:?} report={report:?}");
    assert!(report.is_err(), "non-finite guard values must be rejected");
}

#[test]
fn nan_must_error() {
    nonfinite(f64::NAN);
}
#[test]
fn infinity_must_error() {
    nonfinite(f64::INFINITY);
}

#[test]
fn finite_control_preserves_pivot_coordinates() {
    let inputs = vec![delta_tree()];
    let options = TreeAciOptions {
        scale_tolerance: false,
        ..Default::default()
    };
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    // Install an exact zero approximation; the target remains one at (0,0).
    let mut operator = |batch: TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
        for (p, value) in out.iter_mut().enumerate() {
            *value = batch.get(0, p)?;
        }
        Ok(())
    };
    crate::schedule::run_directional_pass(
        &mut state,
        &options,
        crate::schedule::PassDirection::Forward,
        &mut |_batch: TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
            out.fill(0.0);
            Ok(())
        },
    )
    .unwrap();
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let report = find_global_pivots(
        &state,
        &mut evaluators,
        &options,
        &mut rng,
        &mut operator,
        &[],
    )
    .unwrap();
    assert_eq!(report.pivots, vec![vec![0, 0]]);
}

#[test]
fn nonfinite_walk_target_after_a_finite_start_is_rejected() {
    use rand::Rng;
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let inputs = vec![delta_tree()];
        let options = TreeAciOptions {
            nsearch_global_pivots: 1,
            scale_tolerance: false,
            ..Default::default()
        };
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        crate::schedule::run_directional_pass(
            &mut state,
            &options,
            crate::schedule::PassDirection::Forward,
            &mut |_batch: TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
                out.fill(0.0);
                Ok(())
            },
        )
        .unwrap();
        let mut peek = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let start: Vec<usize> = (0..2).map(|_| peek.random_range(0..2)).collect();
        let start_value = if start == [0, 0] { 1.0 } else { 0.0 };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
        let mut calls = 0;
        let mut operator = |batch: TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
            calls += 1;
            for (point, output) in out.iter_mut().enumerate() {
                *output = if batch.get(0, point)? == start_value {
                    1.0
                } else {
                    bad
                };
            }
            Ok(())
        };
        let result = find_global_pivots(
            &state,
            &mut evaluators,
            &options,
            &mut rng,
            &mut operator,
            &[],
        );
        assert!(
            calls > 2,
            "the finite start and its walk initialization must be evaluated"
        );
        assert!(matches!(
            result,
            Err(crate::TreeAciError::NonFiniteValue {
                context: "guard target"
            })
        ));
    }
}

#[test]
fn residual_rejects_nonfinite_approximation_and_finite_subtraction_overflow() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            super::super::guard_residual(1.0, bad),
            Err(crate::TreeAciError::NonFiniteValue {
                context: "guard residual"
            })
        ));
    }
    assert!(super::super::guard_residual(f64::MAX, -f64::MAX).is_err());
    assert_eq!(super::super::guard_residual(5.0, 3.0).unwrap(), 2.0);
}

#[test]
fn overflowing_guard_threshold_is_rejected() {
    let inputs = vec![delta_tree()];
    let options = TreeAciOptions {
        global_tolerance_margin: 1e200,
        ..Default::default()
    };
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut operator = |_batch: TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
        out.fill(1e200);
        Ok(())
    };
    assert!(matches!(
        find_global_pivots(
            &state,
            &mut evaluators,
            &options,
            &mut rng,
            &mut operator,
            &[]
        ),
        Err(crate::TreeAciError::NonFiniteValue {
            context: "guard threshold"
        })
    ));
}
