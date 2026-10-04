//! `patched_interpolate` with the patch-size bounds of M5: the minimum patch
//! size (`min_patch_bits`) and the acceptance of capped patches
//! (`capped_patches`), with the `ToleranceNotMet` report.
//!
//! Unless stated otherwise a test runs on `chain("s", 4, 2)` with
//! `f(x) = 1 + x0 + 2 x1 + 4 x2 + 8 x3` (values 1 to 16, `sum f^2 = 1496`),
//! the given reference norm `sqrt(1496)`, `rtol = 1e-12`, a bond cap of 3,
//! no retries, and seed 0. Every patch has at most 16 points, so every patch
//! is measured exhaustively and the RMS values have closed forms.

mod adaptive_common;

use adaptive_common::*;
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    CappedPatches, GlobalL2Error, MeasurementMethod, NormReport, PatchRecord, PatchStatus,
    PatchedInterpolationError, PatchedInterpolationOptions, PatchedInterpolationReport,
    PatchedInterpolationResult, ToleranceNotMetBasis, VerificationOptions, GLOBAL_ROUNDING_MARGIN,
    MEASUREMENT_ROUNDING_FACTOR,
};
use tensor4all_partitionedtreetn::ErrorNorm;
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationTermination, TreeInterpolator,
};

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

/// `1 + sum_k 2^k x_k`: the values 1 to `2^n` on `n` binary sites.
fn binary_ramp(p: &[usize]) -> f64 {
    1.0 + p
        .iter()
        .enumerate()
        .map(|(k, &x)| (x << k) as f64)
        .sum::<f64>()
}

/// `2^(sum x)`: rank one across every bond.
fn rank_one(p: &[usize]) -> f64 {
    2.0_f64.powi(p.iter().sum::<usize>() as i32)
}

/// `2^(sum x) + 3^(sum x)`: rank two across every bond and in every child.
fn rank_two(p: &[usize]) -> f64 {
    let s = p.iter().sum::<usize>() as i32;
    2.0_f64.powi(s) + 3.0_f64.powi(s)
}

/// The shared options on a problem whose function has L2 norm `norm`.
fn base_options(norm: f64) -> PatchedInterpolationOptions {
    l2_given(3, norm, 1e-12).with_verification(VerificationOptions::new().with_retries(0))
}

fn chain4() -> Problem {
    chain("s", 4, 2)
}

const RAMP_NORM_SQ: f64 = 1496.0;

fn ramp_options() -> PatchedInterpolationOptions {
    base_options(RAMP_NORM_SQ.sqrt())
}

fn zero_steps() -> ScriptedEngine {
    ScriptedEngine::new(vec![Step::converged(Network::Constant(0.0))])
}

/// The RMS values of the records, in report order.
fn record_rms(report: &PatchedInterpolationReport) -> Vec<f64> {
    report
        .accepted
        .iter()
        .map(|record| record.acceptance.as_ref().unwrap().rms)
        .collect()
}

fn statuses(report: &PatchedInterpolationReport) -> Vec<PatchStatus> {
    report.accepted.iter().map(|record| record.status).collect()
}

/// Agreement to a few ulps of the larger magnitude.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 8.0 * f64::EPSILON * a.abs().max(b.abs())
}

/// The global error of an L2 run, which must be `ToleranceNotMet`:
/// `(measured_rms, unmet_fraction, basis)`.
fn unmet(report: &PatchedInterpolationReport) -> (f64, f64, ToleranceNotMetBasis) {
    match &report.norm.l2_error().unwrap().global {
        GlobalL2Error::ToleranceNotMet {
            measured_rms,
            unmet_fraction,
            basis,
            ..
        } => (*measured_rms, *unmet_fraction, basis.clone()),
        other => panic!("expected ToleranceNotMet, got {other:?}"),
    }
}

fn certified_fraction(report: &PatchedInterpolationReport) -> f64 {
    report.norm.l2_error().unwrap().certified_fraction
}

fn terminations(report: &PatchedInterpolationReport) -> Vec<InterpolationTermination> {
    report
        .accepted
        .iter()
        .map(|record| record.termination)
        .collect()
}

fn run_ok<E>(
    engine: &E,
    problem: &Problem,
    f: &(dyn Fn(&[usize]) -> f64 + Sync),
    options: &PatchedInterpolationOptions,
) -> PatchedInterpolationResult<Name>
where
    E: TreeInterpolator<f64> + Sync,
{
    run(engine, problem, f, &[], options).unwrap()
}

// ---------------------------------------------------------------------------
// The minimum patch size
// ---------------------------------------------------------------------------

#[test]
fn a_converged_failing_patch_is_retained_at_the_minimum() {
    let problem = chain4();
    let (reference, norm) = dense_reference(&problem, &binary_ramp);
    let options = ramp_options().with_min_patch_bits(3);
    let result = run_ok(&zero_steps(), &problem, &binary_ramp, &options);
    let report = &result.report;

    // The root (4 bits) splits at s0; its 3-bit children cannot split below
    // the minimum and are retained with their failed measurements: the odd
    // and the even values.
    assert_eq!(report.splits, 1);
    assert_eq!(
        statuses(report),
        [PatchStatus::ToleranceNotMet, PatchStatus::ToleranceNotMet]
    );
    let rms = record_rms(report);
    assert!(close(rms[0], 85.0_f64.sqrt()) && close(rms[1], 102.0_f64.sqrt()));
    assert_eq!(report.verification_failures, 3);
    assert_eq!(report.function_evaluations, 16);
    let (measured_rms, unmet_fraction, basis) = unmet(report);
    assert!(close(measured_rms, 93.5_f64.sqrt()));
    assert_eq!(unmet_fraction, 1.0);
    // The zero approximation gives no relative statement.
    assert!(matches!(
        basis,
        ToleranceNotMetBasis::ExactOrExhaustive {
            relative_error_bound: None,
            ..
        }
    ));
    assert_eq!(certified_fraction(report), 0.0);
    assert!(!report.tolerance_met());
    assert!(close(dense_l2_residual(&result, &reference), norm));

    // Without the minimum the same engine splits down to exact patches.
    let result = run_ok(&zero_steps(), &problem, &binary_ramp, &ramp_options());
    assert_eq!(result.report.splits, 7);
    assert!(result.report.tolerance_met());
    assert!(matches!(
        result.report.norm.l2_error().unwrap().global,
        GlobalL2Error::Certified { .. }
    ));
}

#[test]
fn retries_run_before_the_minimum_and_the_last_run_is_kept() {
    let problem = chain4();
    let options = ramp_options()
        .with_min_patch_bits(3)
        .with_verification(VerificationOptions::new().with_retries(1));
    // Calls: root run 0, root run 1, child 0 run 0, child 0 run 1, child 1
    // run 0, child 1 run 1. The constants 8 and 9 center the children.
    let engine = ScriptedEngine::new(
        [0.0, 0.0, 0.0, 8.0, 0.0, 9.0]
            .into_iter()
            .map(|value| Step::converged(Network::Constant(value)))
            .collect(),
    );
    let result = run_ok(&engine, &problem, &binary_ramp, &options);
    let report = &result.report;
    assert_eq!(engine.calls().len(), 6);
    assert_eq!(report.engine_retries, 3);
    assert!(report
        .accepted
        .iter()
        .all(|record| record.retries_used == 1));
    let rms = record_rms(report);
    assert!(close(rms[0], 21.0_f64.sqrt()) && close(rms[1], 21.0_f64.sqrt()));

    // ||f~||^2 = 8 * 64 + 8 * 81 over 16 points.
    let error = report.norm.l2_error().unwrap();
    let approximation_rms = error.approximation_rms.unwrap();
    assert!((approximation_rms - 72.5_f64.sqrt()).abs() <= 1e-12 * approximation_rms);
    let (measured_rms, _, basis) = unmet(report);
    let ToleranceNotMetBasis::ExactOrExhaustive {
        relative_error_bound: Some(bound),
        ..
    } = basis
    else {
        panic!("expected a relative bound, got {basis:?}");
    };
    // The M3 formula with the measured value in place of tau.
    let rounding = MEASUREMENT_ROUNDING_FACTOR * f64::EPSILON * 72.5_f64.sqrt();
    let upper = 21.0_f64.sqrt() * (1.0 + GLOBAL_ROUNDING_MARGIN) + rounding;
    let expected = upper / ((1.0 - GLOBAL_ROUNDING_MARGIN) * 72.5_f64.sqrt() - upper);
    assert!(close(measured_rms, 21.0_f64.sqrt()));
    assert!((bound - expected).abs() <= 1e-6 * expected);
}

#[test]
fn patches_within_and_outside_their_allowance_mix() {
    let problem = chain4();
    let options = ramp_options().with_min_patch_bits(3);
    let engine = ScriptedEngine::new(vec![
        Step::converged(Network::Constant(0.0)),
        Step::converged(Network::Exact),
        Step::converged(Network::Constant(0.0)),
    ]);
    let result = run_ok(&engine, &problem, &binary_ramp, &options);
    let report = &result.report;
    assert_eq!(
        statuses(report),
        [PatchStatus::WithinTolerance, PatchStatus::ToleranceNotMet]
    );
    let rms = record_rms(report);
    assert!(rms[0] <= tau(report));
    assert!(close(rms[1], 102.0_f64.sqrt()));
    assert_eq!(certified_fraction(report), 0.5);
    let (measured_rms, unmet_fraction, _) = unmet(report);
    assert_eq!(unmet_fraction, 0.5);
    assert!((measured_rms - 51.0_f64.sqrt()).abs() <= 1e-12 * measured_rms);
    // The global mean square is the volume-weighted sum of the records'.
    let combined: f64 = rms.iter().map(|rms| 0.5 * rms * rms).sum();
    assert!(close(measured_rms * measured_rms, combined));
}

#[test]
fn a_blocked_run_that_did_not_converge_is_measured_once() {
    // No maximum: the capped runs are not eligible. The root splits
    // unmeasured; only the two blocked children are measured.
    let problem = chain4();
    let options = ramp_options().with_min_patch_bits(3);
    let engine = ScriptedEngine::new(vec![Step::capped()]);
    let result = run_ok(&engine, &problem, &binary_ramp, &options);
    let report = &result.report;
    assert_eq!(report.verification_failures, 2);
    assert_eq!(report.function_evaluations, 16);
    assert_eq!(
        statuses(report),
        [PatchStatus::ToleranceNotMet, PatchStatus::ToleranceNotMet]
    );
    assert_eq!(
        terminations(report),
        [InterpolationTermination::BondCapReached; 2]
    );
    let rms = record_rms(report);
    assert!(close(rms[0], 85.0_f64.sqrt()) && close(rms[1], 102.0_f64.sqrt()));

    // An IterationLimit run that passes its measurement is within tolerance.
    let (_, norm) = dense_reference(&problem, &rank_one);
    let options = base_options(norm).with_min_patch_bits(3);
    let engine = DenseEngine::with_fault(Fault::IterationLimit);
    let result = run_ok(&engine, &problem, &rank_one, &options);
    assert_eq!(result.report.splits, 1);
    assert_eq!(statuses(&result.report), [PatchStatus::WithinTolerance; 2]);
    assert_eq!(
        terminations(&result.report),
        [InterpolationTermination::IterationLimit; 2]
    );
    assert!(result.report.tolerance_met());
}

// ---------------------------------------------------------------------------
// Capped patches
// ---------------------------------------------------------------------------

#[test]
fn a_capped_patch_within_the_bound_is_accepted_on_its_measurement() {
    // The exact network of rank_two has rank 2: at a cap of 2 the dense
    // engine reports BondCapReached.
    let problem = chain4();
    let (reference, norm) = dense_reference(&problem, &rank_two);
    let base = l2_given(2, norm, 1e-10);

    let options = base
        .clone()
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 4 });
    let result = run_ok(&DenseEngine::new(), &problem, &rank_two, &options);
    let report = &result.report;
    assert_eq!(report.splits, 0);
    let record = &report.accepted[0];
    assert_eq!(record.termination, InterpolationTermination::BondCapReached);
    assert_eq!(record.max_bond_dim, 2);
    assert_eq!(record.status, PatchStatus::WithinTolerance);
    assert!(matches!(
        report.norm.l2_error().unwrap().global,
        GlobalL2Error::Certified { .. }
    ));
    let delta = report.norm.delta().unwrap();
    assert!(dense_l2_residual(&result, &reference) <= delta);

    // A bound of 3 bits splits the 4-bit root once; its children are
    // accepted at the cap.
    let options = base
        .clone()
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 3 });
    let result = run_ok(&DenseEngine::new(), &problem, &rank_two, &options);
    assert_eq!(result.report.splits, 1);
    assert_eq!(result.report.verification_failures, 0);
    assert_eq!(
        terminations(&result.report),
        [InterpolationTermination::BondCapReached; 2]
    );
    assert!(dense_l2_residual(&result, &reference) <= delta);

    // Without a bound every capped patch splits, down to exact patches.
    let result = run_ok(&DenseEngine::new(), &problem, &rank_two, &base);
    assert_eq!(result.report.splits, 7);
    assert_eq!(result.report.accepted.len(), 8);
}

#[test]
fn a_converged_patch_is_never_split_for_its_size() {
    let problem = chain4();
    let (_, norm) = dense_reference(&problem, &rank_one);
    let options =
        l2_given(2, norm, 1e-10).with_capped_patches(CappedPatches::AcceptUpTo { bits: 0 });
    let result = run_ok(&DenseEngine::new(), &problem, &rank_one, &options);
    assert_eq!(result.report.splits, 0);
    assert_eq!(
        terminations(&result.report),
        [InterpolationTermination::Converged]
    );
}

#[test]
fn a_failed_capped_measurement_splits_without_a_retry() {
    let problem = chain4();
    let options = ramp_options()
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 4 })
        .with_verification(VerificationOptions::new().with_retries(1));
    let engine = ScriptedEngine::new(vec![Step::capped(), Step::converged(Network::Exact)]);
    let result = run_ok(&engine, &problem, &binary_ramp, &options);
    let report = &result.report;
    assert_eq!(
        (report.verification_failures, report.engine_retries),
        (1, 0)
    );
    assert_eq!(report.splits, 1);
    // The children start from the worst points of the root measurement:
    // 15 at (0, 1, 1, 1) and 16 at (1, 1, 1, 1), in child coordinates.
    let calls = engine.calls();
    assert_eq!(calls.len(), 3);
    for child in &calls[1..] {
        assert_eq!(child.initial_pivots[0], vec![1, 1, 1]);
    }
}

#[test]
fn a_failed_capped_measurement_at_an_exhausted_patch_order_is_a_verification_failure() {
    let problem = chain4();
    let options = ramp_options()
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 4 })
        .with_patch_order(vec![problem.site("s0", 0)]);
    let engine = ScriptedEngine::new(vec![Step::capped()]);
    // FIFO: the child s0 = 0 fails first; the child s0 = 1 is never reached.
    match run(&engine, &problem, &binary_ramp, &[], &options) {
        Err(PatchedInterpolationError::VerificationFailed {
            projector,
            measurement,
        }) => {
            assert_eq!(projector.get(&problem.site("s0", 0)), Some(0));
            assert!(close(measurement.rms, 85.0_f64.sqrt()));
        }
        other => panic!("expected VerificationFailed, got {other:?}"),
    }
    assert_eq!(engine.calls().len(), 2);
}

#[test]
fn a_failed_capped_run_that_is_blocked_is_not_measured_twice() {
    let problem = chain4();
    let options = ramp_options()
        .with_min_patch_bits(4)
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 4 });
    let engine = ScriptedEngine::new(vec![Step::capped()]);
    let result = run_ok(&engine, &problem, &binary_ramp, &options);
    let report = &result.report;
    assert_eq!(report.splits, 0);
    assert_eq!(report.verification_failures, 1);
    let record = &report.accepted[0];
    assert_eq!(record.status, PatchStatus::ToleranceNotMet);
    assert_eq!(record.termination, InterpolationTermination::BondCapReached);
    assert!(close(record_rms(report)[0], 93.5_f64.sqrt()));
}

#[test]
fn iteration_limit_runs_are_not_capped_eligible() {
    let problem = chain4();
    let (_, norm) = dense_reference(&problem, &rank_one);
    let options = base_options(norm).with_capped_patches(CappedPatches::AcceptUpTo { bits: 4 });
    let engine = DenseEngine::with_fault(Fault::IterationLimit);
    let result = run_ok(&engine, &problem, &rank_one, &options);
    assert_eq!(result.report.splits, 7);
    assert_eq!(result.report.accepted.len(), 8);
    assert_eq!(result.report.verification_failures, 0);
}

#[test]
fn a_network_above_the_cap_or_malformed_is_an_engine_error() {
    let problem = chain4();
    let rank_three = |p: &[usize]| {
        let s = p.iter().sum::<usize>() as i32;
        1.0 + 2.0_f64.powi(s) + 3.0_f64.powi(s)
    };
    let (_, norm) = dense_reference(&problem, &rank_three);
    let engine_message = |result| match expect_interpolation(result) {
        (_, InterpolationError::Engine { source }) => source.to_string(),
        (_, other) => panic!("expected an engine error, got {other:?}"),
    };

    // Rank 3 at the middle bond, above the cap of 2: capped-eligible, and
    // separately blocked without a maximum.
    for options in [
        l2_given(2, norm, 1e-10).with_capped_patches(CappedPatches::AcceptUpTo { bits: 4 }),
        l2_given(2, norm, 1e-10).with_min_patch_bits(4),
    ] {
        let message = engine_message(run(
            &DenseEngine::new(),
            &problem,
            &rank_three,
            &[],
            &options,
        ));
        assert!(message.contains("above the cap 2"), "{message}");
    }

    // An empty network and a network over other sites, both blocked.
    let (_, norm) = dense_reference(&problem, &rank_two);
    let options = l2_given(2, norm, 1e-10).with_min_patch_bits(4);
    let message = engine_message(run(
        &DenseEngine::with_fault(Fault::CapAfterPivots),
        &problem,
        &rank_two,
        &[],
        &options,
    ));
    assert!(message.contains("nodes"), "{message}");
    let message = engine_message(run(
        &DenseEngine::with_fault(Fault::WrongLayout),
        &problem,
        &rank_two,
        &[],
        &options,
    ));
    assert!(message.contains("expected the active sites"), "{message}");
}

// ---------------------------------------------------------------------------
// Measurement streams and SampledMax
// ---------------------------------------------------------------------------

/// `chain("s", 6, 2)` measured on 4 samples, never exhaustively.
fn sampled_six() -> (Problem, PatchedInterpolationOptions) {
    let problem = chain("s", 6, 2);
    let (_, norm) = dense_reference(&problem, &binary_ramp);
    let options = l2_given(3, norm, 1e-12).with_verification(
        VerificationOptions::new()
            .with_samples(4)
            .with_max_exhaustive_points(0)
            .with_retries(0),
    );
    (problem, options)
}

/// The RMS of `f` (the zero approximation's residual) at the points drawn
/// from `seed` in the active coordinates of the patch with `fixed` sites.
fn sampled_rms(fixed: &[Option<usize>], seed: u64) -> f64 {
    let n_active = fixed.iter().filter(|value| value.is_none()).count();
    let points = streams::draw(&vec![2; n_active], 4, seed);
    let mean_square = points
        .iter()
        .map(|local| {
            let mut local = local.iter().copied();
            let full: Vec<usize> = fixed
                .iter()
                .map(|value| value.unwrap_or_else(|| local.next().unwrap()))
                .collect();
            binary_ramp(&full).powi(2)
        })
        .sum::<f64>()
        / points.len() as f64;
    mean_square.sqrt()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

#[test]
fn a_blocked_rerun_is_measured_on_the_stream_of_its_attempt() {
    // The root (6 bits) is blocked by a minimum of 6. Run 0 converges and
    // fails; run 1 reaches the cap.
    let (problem, options) = sampled_six();
    let verification = options.verification.with_retries(1);
    let options = options
        .with_min_patch_bits(6)
        .with_verification(verification);
    let state = streams::path_state(0, &[]);
    let root = [None; 6];
    let expected_acceptance = sampled_rms(&root, streams::verify(state, 1));
    let expected_audit = sampled_rms(&root, streams::audit(state));
    // Eligible through the maximum, or blocked and unmeasured without it:
    // both measure run 1 on verification stream 1.
    for capped in [CappedPatches::AcceptUpTo { bits: 6 }, CappedPatches::Split] {
        let engine = ScriptedEngine::new(vec![
            Step::converged(Network::Constant(0.0)),
            Step::capped(),
        ]);
        let options = options.clone().with_capped_patches(capped);
        let result = run_ok(&engine, &problem, &binary_ramp, &options);
        let record = &result.report.accepted[0];
        assert_eq!(record.status, PatchStatus::ToleranceNotMet);
        assert_eq!(record.termination, InterpolationTermination::BondCapReached);
        assert_eq!(record.retries_used, 1);
        let acceptance = record.acceptance.as_ref().unwrap();
        assert_eq!(acceptance.method, MeasurementMethod::Sampled);
        assert!(near(acceptance.rms, expected_acceptance));
        assert!(near(record.audit.as_ref().unwrap().rms, expected_audit));
        assert_eq!(result.report.verification_failures, 2);
    }
}

#[test]
fn a_sampled_blocked_patch_is_audited() {
    let (problem, options) = sampled_six();
    let options = options.with_min_patch_bits(5);
    // The 32-point children are sampled on stream 0 and audited.
    let child = |value: usize| {
        let mut fixed = [None; 6];
        fixed[0] = Some(value);
        let state = streams::path_state(0, &[(0, value)]);
        (
            sampled_rms(&fixed, streams::verify(state, 0)),
            sampled_rms(&fixed, streams::audit(state)),
        )
    };
    let children = [child(0), child(1)];

    let result = run_ok(&zero_steps(), &problem, &binary_ramp, &options);
    let report = &result.report;
    assert_eq!(statuses(report), [PatchStatus::ToleranceNotMet; 2]);
    for (record, (acceptance, audit)) in report.accepted.iter().zip(children) {
        assert!(near(record.acceptance.as_ref().unwrap().rms, acceptance));
        assert!(near(record.audit.as_ref().unwrap().rms, audit));
    }
    let (measured_rms, unmet_fraction, basis) = unmet(report);
    assert_eq!(unmet_fraction, 1.0);
    assert!(matches!(basis, ToleranceNotMetBasis::Audited { .. }));
    let audited = (0.5 * children[0].1.powi(2) + 0.5 * children[1].1.powi(2)).sqrt();
    assert!(near(measured_rms, audited));
    assert_eq!(certified_fraction(report), 0.0);

    // Without audits the acceptance statistics are combined.
    let verification = options.verification.with_audit(false);
    let options = options.with_verification(verification);
    let result = run_ok(&zero_steps(), &problem, &binary_ramp, &options);
    let (measured_rms, _, basis) = unmet(&result.report);
    assert_eq!(basis, ToleranceNotMetBasis::AcceptanceOnly);
    let statistics = (0.5 * children[0].0.powi(2) + 0.5 * children[1].0.powi(2)).sqrt();
    assert!(near(measured_rms, statistics));
}

#[test]
fn sampled_max_judges_capped_and_blocked_runs_by_the_engine_estimate() {
    // Engine tolerance 0.1 * 16 = 1.6.
    let problem = chain4();
    let base = PatchedInterpolationOptions::new(3)
        .with_error_norm(ErrorNorm::sampled_max_with_reference(16.0))
        .with_tolerance(tol(0.1));
    let run_steps = |steps: Vec<Step>, options: &PatchedInterpolationOptions| {
        run_ok(&ScriptedEngine::new(steps), &problem, &binary_ramp, options)
    };
    let capped = base
        .clone()
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 4 });

    // A capped-eligible run at or below the tolerance is accepted.
    let result = run_steps(vec![Step::capped().with_error_estimate(1.6)], &capped);
    assert_eq!(result.report.splits, 0);
    let record = &result.report.accepted[0];
    assert_eq!(record.termination, InterpolationTermination::BondCapReached);
    assert_eq!(record.status, PatchStatus::WithinTolerance);
    assert!(record.acceptance.is_none());
    assert!(matches!(
        result.report.norm,
        NormReport::SampledMax { engine_tolerance, .. } if engine_tolerance == 1.6
    ));

    // Above it, the patch splits; the converged children are accepted.
    let steps = vec![
        Step::capped().with_error_estimate(2.0),
        Step::converged(Network::Constant(0.0)),
    ];
    let result = run_steps(steps.clone(), &capped);
    assert_eq!(result.report.splits, 1);
    assert!(result.report.tolerance_met());

    // Blocked: judged above the tolerance with a maximum, or unjudged and
    // estimated above it without one.
    for options in [capped.clone(), base.clone()] {
        let result = run_steps(steps.clone(), &options.with_min_patch_bits(4));
        assert_eq!(statuses(&result.report), [PatchStatus::ToleranceNotMet]);
        assert!(!result.report.tolerance_met());
    }

    // A blocked IterationLimit run at the tolerance is within it.
    let step = Step::capped()
        .with_termination(InterpolationTermination::IterationLimit)
        .with_error_estimate(1.0);
    let result = run_steps(vec![step], &base.clone().with_min_patch_bits(4));
    assert_eq!(statuses(&result.report), [PatchStatus::WithinTolerance]);
    assert_eq!(
        terminations(&result.report),
        [InterpolationTermination::IterationLimit]
    );
}

// ---------------------------------------------------------------------------
// Generalized bits
// ---------------------------------------------------------------------------

#[test]
fn a_fused_site_counts_as_one_bit() {
    // Three sites of dimension 4: the root has 3 bits and splits at q0 into
    // four 2-bit children, which a minimum of 2 blocks. Counting log2 of the
    // volume would give them 4 bits and let them split.
    let problem = chain("q", 3, 4);
    let f = |p: &[usize]| 1.0 + (p[0] + 4 * p[1] + 16 * p[2]) as f64;
    let (_, norm) = dense_reference(&problem, &f);
    let result = run_ok(
        &zero_steps(),
        &problem,
        &f,
        &base_options(norm).with_min_patch_bits(2),
    );
    let report = &result.report;
    assert_eq!(report.splits, 1);
    assert_eq!(statuses(report), [PatchStatus::ToleranceNotMet; 4]);
    for (rms, expected) in record_rms(report)
        .iter()
        .zip([1301.0, 1364.0, 1429.0, 1496.0])
    {
        assert!(close(rms * rms, expected), "{rms}");
    }
    let (measured_rms, _, _) = unmet(report);
    assert!(close(measured_rms * measured_rms, 1397.5));

    // A capped 2-bit root of two fused sites is within a 2-bit bound.
    let problem = chain("q", 2, 4);
    let g = |p: &[usize]| rank_two(&[p[0] + p[1]]);
    let (_, norm) = dense_reference(&problem, &g);
    let options =
        l2_given(2, norm, 1e-10).with_capped_patches(CappedPatches::AcceptUpTo { bits: 2 });
    let result = run_ok(&DenseEngine::new(), &problem, &g, &options);
    assert_eq!(result.report.splits, 0);
    assert_eq!(
        terminations(&result.report),
        [InterpolationTermination::BondCapReached]
    );

    // A site of dimension 3 is one bit as well; no layout is rejected.
    let problem = Problem::new(
        &[("q0", &[3]), ("q1", &[2]), ("q2", &[2])],
        &[("q0", "q1"), ("q1", "q2")],
    );
    let h = |p: &[usize]| 1.0 + (p[0] + 3 * p[1] + 6 * p[2]) as f64;
    let options = base_options(650.0_f64.sqrt())
        .with_min_patch_bits(2)
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 3 });
    let result = run_ok(&zero_steps(), &problem, &h, &options);
    let report = &result.report;
    assert_eq!(report.splits, 1);
    // rms^2 = a^2 + 9 a + 31.5 for a = q0 + 1.
    for (rms, expected) in record_rms(report).iter().zip([41.5, 53.5, 67.5]) {
        assert!(close(rms * rms, expected), "{rms}");
    }
    let (measured_rms, _, _) = unmet(report);
    assert!(close(measured_rms * measured_rms, 650.0 / 12.0));
}

// ---------------------------------------------------------------------------
// Options and interactions
// ---------------------------------------------------------------------------

#[test]
fn a_capped_bound_below_the_minimum_is_rejected_before_any_evaluation() {
    let problem = chain4();
    let invalid = ramp_options()
        .with_min_patch_bits(3)
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 2 });
    let message = expect_invalid(
        run_with(&zero_steps(), &problem, never, &[], &invalid),
        "capped_patches",
    );
    assert!(message.contains("min_patch_bits (3)"), "{message}");
    // Validated after max_patches and before the verification options.
    expect_invalid(
        run_with(
            &zero_steps(),
            &problem,
            never,
            &[],
            &invalid.clone().with_max_patches(0),
        ),
        "max_patches",
    );
    let samples = invalid
        .clone()
        .with_verification(VerificationOptions::new().with_samples(1));
    expect_invalid(
        run_with(&zero_steps(), &problem, never, &[], &samples),
        "capped_patches",
    );
    // Equal bounds are valid.
    let equal = ramp_options()
        .with_min_patch_bits(3)
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 3 });
    assert!(run(&zero_steps(), &problem, &binary_ramp, &[], &equal).is_ok());
}

#[test]
fn an_exhausted_patch_order_keeps_its_errors_under_the_minimum() {
    let problem = chain4();
    let s0 = problem.site("s0", 0);
    let s1 = problem.site("s1", 0);
    // A minimum of 1 never blocks; the children have no split site left.
    let options = ramp_options()
        .with_min_patch_bits(1)
        .with_patch_order(vec![s0.clone()]);
    match run(&zero_steps(), &problem, &binary_ramp, &[], &options) {
        Err(PatchedInterpolationError::VerificationFailed { projector, .. }) => {
            assert_eq!(projector.get(&s0), Some(0));
        }
        other => panic!("expected VerificationFailed, got {other:?}"),
    }
    // With a split site left the children are blocked and retained.
    let options = ramp_options()
        .with_min_patch_bits(3)
        .with_patch_order(vec![s0, s1]);
    let result = run_ok(&zero_steps(), &problem, &binary_ramp, &options);
    assert_eq!(statuses(&result.report), [PatchStatus::ToleranceNotMet; 2]);
}

#[test]
fn settings_without_effect_equal_the_defaults() {
    // Patches that reach the engine have at least two active sites, and the
    // split of a two-site patch gives exact patches.
    let problem = chain4();
    let (_, norm) = dense_reference(&problem, &rank_two);
    let dense = l2_given(2, norm, 1e-10);
    let scripted = ramp_options();
    let variants = |base: &PatchedInterpolationOptions| {
        vec![
            base.clone().with_min_patch_bits(0),
            base.clone().with_min_patch_bits(1),
            base.clone()
                .with_capped_patches(CappedPatches::AcceptUpTo { bits: 0 }),
            base.clone()
                .with_capped_patches(CappedPatches::AcceptUpTo { bits: 1 }),
        ]
    };
    let unset = run_ok(&DenseEngine::new(), &problem, &rank_two, &dense);
    for options in variants(&dense) {
        let result = run_ok(&DenseEngine::new(), &problem, &rank_two, &options);
        assert_same_run(&problem, &unset, &result);
    }
    let unset = run_ok(&zero_steps(), &problem, &binary_ramp, &scripted);
    for options in variants(&scripted) {
        let result = run_ok(&zero_steps(), &problem, &binary_ramp, &options);
        assert_same_run(&problem, &unset, &result);
    }
}

#[test]
fn zero_patches_are_unaffected_by_the_minimum() {
    let problem = chain4();
    let f = |p: &[usize]| (p[0] * (1 + p[1] + 2 * p[2] + 4 * p[3])) as f64;
    let (_, norm) = dense_reference(&problem, &f);
    let result = run_ok(
        &zero_steps(),
        &problem,
        &f,
        &base_options(norm).with_min_patch_bits(3),
    );
    let report = &result.report;
    let s0 = problem.site("s0", 0);
    // The half s0 = 0 vanishes: a zero patch with an exhaustive acceptance.
    assert_eq!(report.zero_patches.len(), 1);
    let zero = &report.zero_patches[0];
    assert_eq!(zero.projector.get(&s0), Some(0));
    assert_eq!(
        zero.acceptance.as_ref().unwrap().method,
        MeasurementMethod::Exhaustive
    );
    // The half s0 = 1 holds the values 1 to 8 and is retained.
    assert_eq!(statuses(report), [PatchStatus::ToleranceNotMet]);
    assert!(close(record_rms(report)[0].powi(2), 25.5));
    let (measured_rms, unmet_fraction, _) = unmet(report);
    assert!(close(measured_rms * measured_rms, 12.75));
    assert_eq!(unmet_fraction, 0.5);
    assert_eq!(certified_fraction(report), 0.5);
}

#[test]
fn both_bounds_act_on_a_branched_tree_with_treetci() {
    // The junction r of quantics_tree has degree three and carries no site;
    // 128 points, so every patch is measured exhaustively.
    let problem = quantics_tree();
    assert_eq!(max_degree(&problem), 3);
    let (reference, norm) = dense_reference(&problem, &tree_peak);
    let n_sites = problem.sites.len();
    let active = |record: &PatchRecord| n_sites - record.projector.len();
    let capped_accepted_above = |report: &PatchedInterpolationReport, cap: usize, min: usize| {
        report.accepted.iter().any(|record| {
            record.termination == InterpolationTermination::BondCapReached
                && record.status == PatchStatus::WithinTolerance
                && record.max_bond_dim <= cap
                && active(record) > min
        })
    };

    // Cap 2: a 5-bit capped patch is accepted through the bound of 5, and
    // 4-bit patches that miss their allowance are retained by the minimum.
    let options = l2_given(2, norm, 1e-3)
        .with_min_patch_bits(4)
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 5 });
    let result = run_ok(
        &TreeTciInterpolator::default(),
        &problem,
        &tree_peak,
        &options,
    );
    let report = &result.report;
    assert!(capped_accepted_above(report, 2, 4));
    assert!(report
        .accepted
        .iter()
        .any(|record| record.status == PatchStatus::ToleranceNotMet && active(record) <= 4));
    assert!(!report.tolerance_met());
    let (measured_rms, unmet_fraction, basis) = unmet(report);
    assert!(unmet_fraction > 0.0 && unmet_fraction < 1.0);
    assert!(matches!(
        basis,
        ToleranceNotMetBasis::ExactOrExhaustive { .. }
    ));
    let dense_rms = dense_l2_residual(&result, &reference) / 128.0_f64.sqrt();
    assert!((measured_rms - dense_rms).abs() <= 1e-12 * dense_rms);

    // Cap 3: capped patches up to 5 bits are accepted and every patch meets
    // its allowance; the dense error satisfies the M3 certificate.
    let options = l2_given(3, norm, 1e-3)
        .with_min_patch_bits(3)
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 5 });
    let result = run_ok(
        &TreeTciInterpolator::default(),
        &problem,
        &tree_peak,
        &options,
    );
    let report = &result.report;
    assert!(capped_accepted_above(report, 3, 3));
    assert!(report.tolerance_met());
    let GlobalL2Error::Certified {
        rounding_allowance_rms: Some(rounding),
        ..
    } = report.norm.l2_error().unwrap().global
    else {
        panic!("expected a certified error");
    };
    let delta = report.norm.delta().unwrap();
    let bound = delta * (1.0 + GLOBAL_ROUNDING_MARGIN) + 128.0_f64.sqrt() * rounding;
    assert!(dense_l2_residual(&result, &reference) <= bound);
}
