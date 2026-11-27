use super::LocalThreshold;
use crate::StopRule;

const TOL: f64 = 1e-6;

#[test]
fn errors_below_tolerance_keep_the_factor_exactly_one() {
    let mut threshold = LocalThreshold::new(15, TOL, StopRule::Sampled);
    assert_eq!(threshold.factor(), 1.0);
    assert_eq!(threshold.last_applied(), 1.0);
    for error in [0.0, 0.5 * TOL, TOL * (1.0 - f64::EPSILON)] {
        threshold.record(error, TOL);
        assert_eq!(threshold.factor().to_bits(), 1.0_f64.to_bits());
    }
    // `abs_tol * 1.0` is `abs_tol` bit for bit, so the local updates see the
    // untightened threshold.
    let abs_tol = TOL * 0.731;
    assert_eq!((abs_tol * threshold.factor()).to_bits(), abs_tol.to_bits());
    assert_eq!(threshold.last_applied(), 1.0);
    assert!(!threshold.floor_window_failed(&[0.0; 3], TOL, 3));
}

#[test]
fn failing_errors_halve_down_to_the_edge_count_floor() {
    let mut threshold = LocalThreshold::new(15, TOL, StopRule::Sampled);
    let mut applied = Vec::new();
    // `error == tolerance` and NaN fail the stop's `error < tolerance` too.
    for error in [2.0 * TOL, TOL, f64::NAN, 1.3 * TOL, 1.1 * TOL, 5.0 * TOL] {
        threshold.record(error, TOL);
        applied.push(threshold.last_applied());
    }
    assert_eq!(applied, vec![1.0, 0.5, 0.25, 0.125, 1.0 / 15.0, 1.0 / 15.0]);
    assert_eq!(threshold.factor(), 1.0 / 15.0);
}

#[test]
fn the_factor_never_loosens() {
    let mut threshold = LocalThreshold::new(15, TOL, StopRule::Sampled);
    threshold.record(3.0 * TOL, TOL);
    threshold.record(3.0 * TOL, TOL);
    assert_eq!(threshold.factor(), 0.25);
    for _ in 0..4 {
        threshold.record(0.1 * TOL, TOL);
        assert_eq!(threshold.factor(), 0.25);
    }
}

#[test]
fn the_floor_is_one_for_a_single_edge_or_none() {
    for n_edges in [0, 1] {
        let mut threshold = LocalThreshold::new(n_edges, TOL, StopRule::Sampled);
        threshold.record(3.0 * TOL, TOL);
        assert_eq!(threshold.factor(), 1.0, "{n_edges} edges");
    }
    // A non-power-of-two edge count clamps at exactly 1 / E.
    let mut threshold = LocalThreshold::new(3, TOL, StopRule::Sampled);
    threshold.record(3.0 * TOL, TOL);
    threshold.record(3.0 * TOL, TOL);
    assert_eq!(threshold.factor(), 1.0 / 3.0);
}

#[test]
fn a_single_edge_or_none_never_arms_the_floor() {
    // With E <= 1 the floor is 1.0, so the factor cannot descend. Three
    // failing sweeps "at the floor" would be three untightened sweeps, and
    // the run must end as without the adaptive threshold (`MaxIterations` or
    // `Converged`), however many iterations fail.
    for n_edges in [0, 1] {
        let mut threshold = LocalThreshold::new(n_edges, TOL, StopRule::Sampled);
        let mut errors = Vec::new();
        for n in 0..30 {
            // `error == tolerance` and NaN fail the stop's comparison too.
            let error = match n % 3 {
                0 => 2.0 * TOL,
                1 => TOL,
                _ => f64::NAN,
            };
            errors.push(error);
            threshold.record(error, TOL);
            assert_eq!(threshold.factor().to_bits(), 1.0_f64.to_bits());
            assert_eq!(threshold.last_applied().to_bits(), 1.0_f64.to_bits());
            for window in 1..=3 {
                assert!(
                    !threshold.floor_window_failed(&errors, TOL, window),
                    "{n_edges} edges, iteration {n}, window {window}"
                );
            }
        }
    }
}

#[test]
fn zero_tolerance_never_tightens_or_reaches_the_floor() {
    // E = 2, where a positive tolerance would arm the floor stop.
    let mut threshold = LocalThreshold::new(2, 0.0, StopRule::Sampled);
    let errors = [0.0, 1.0, 1.0, 1.0];
    for &error in &errors {
        threshold.record(error, 0.0);
    }
    assert_eq!(threshold.factor(), 1.0);
    assert!(!threshold.floor_window_failed(&errors, 0.0, 3));
}

#[test]
fn the_floor_window_needs_three_failing_sweeps_at_the_floor() {
    // E = 2: the floor 1/2 is reached after one halving.
    let mut threshold = LocalThreshold::new(2, TOL, StopRule::Sampled);
    let mut errors = Vec::new();
    let mut verdicts = Vec::new();
    for error in [
        4.0 * TOL,
        2.0 * TOL,
        2.0 * TOL,
        0.5 * TOL,
        2.0 * TOL,
        2.0 * TOL,
        2.0 * TOL,
    ] {
        errors.push(error);
        threshold.record(error, TOL);
        verdicts.push(threshold.floor_window_failed(&errors, TOL, 3));
    }
    // Iteration 0 swept at 1.0, so the first window at the floor is 1..=3,
    // which a below-tolerance error at iteration 3 interrupts; the next
    // complete failing window ends at iteration 6.
    assert_eq!(
        verdicts,
        vec![false, false, false, false, false, false, true]
    );
}

#[test]
fn the_floor_window_ignores_a_short_or_mismatched_history() {
    // E = 2: iteration 0 sweeps at 1.0, every later one at the floor 1/2.
    let mut threshold = LocalThreshold::new(2, TOL, StopRule::Sampled);
    for _ in 0..3 {
        threshold.record(2.0 * TOL, TOL);
    }
    // Iterations 1 and 2 are a complete failing window of two at the floor.
    assert!(threshold.floor_window_failed(&[2.0 * TOL; 3], TOL, 2));
    // A history shorter than what was recorded is not that window.
    assert!(!threshold.floor_window_failed(&[2.0 * TOL; 2], TOL, 2));
    // Too short for the window, an empty window, iteration 0 not at the floor.
    assert!(!threshold.floor_window_failed(&[2.0 * TOL; 2], TOL, 3));
    assert!(!threshold.floor_window_failed(&[2.0 * TOL; 3], TOL, 0));
    assert!(!threshold.floor_window_failed(&[2.0 * TOL; 3], TOL, 3));
    threshold.record(2.0 * TOL, TOL);
    assert!(threshold.floor_window_failed(&[2.0 * TOL; 4], TOL, 3));
}

#[test]
fn per_bond_never_tightens_or_reaches_the_floor() {
    // Under `PerBond` the recorded error is the bond error, which a bond cap
    // can keep above `tolerance`; the factor must stay exactly 1.0 there and
    // the floor stop must never fire (#1405). With E >= 2 the same errors
    // under `Sampled` halve the factor and complete a floor window (see the
    // tests above), so both the factor and the window check tell the gate
    // apart.
    for n_edges in [1, 2, 15] {
        let mut threshold = LocalThreshold::new(n_edges, TOL, StopRule::PerBond);
        let errors = [2.0 * TOL, f64::NAN, TOL, 5.0 * TOL, 3.0 * TOL];
        for (n, &error) in errors.iter().enumerate() {
            threshold.record(error, TOL);
            assert_eq!(threshold.factor().to_bits(), 1.0_f64.to_bits());
            assert_eq!(threshold.last_applied().to_bits(), 1.0_f64.to_bits());
            assert!(!threshold.floor_window_failed(&errors[..=n], TOL, 3));
        }
    }
}

#[test]
fn floor_stop_uses_the_configured_history_window() {
    let mut threshold = super::LocalThreshold::new(2, 1.0, crate::StopRule::Sampled);
    threshold.record(2.0, 1.0);
    threshold.record(2.0, 1.0);
    assert!(threshold.floor_window_failed(&[2.0, 2.0], 1.0, 1));
    assert!(!threshold.floor_window_failed(&[2.0, 2.0], 1.0, 2));
    threshold.record(2.0, 1.0);
    assert!(threshold.floor_window_failed(&[2.0, 2.0, 2.0], 1.0, 2));
    assert!(!threshold.floor_window_failed(&[2.0, 2.0, 2.0], 1.0, 4));
}
