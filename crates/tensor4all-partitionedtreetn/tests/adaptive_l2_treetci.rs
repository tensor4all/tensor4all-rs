//! `patched_interpolate` under `ErrorNorm::L2` with the TreeTCI engine,
//! checked against dense references on small branched trees.

mod adaptive_common;

use std::sync::OnceLock;

use adaptive_common::*;
use tensor4all_core::IdxTensor;
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    GlobalL2Error, MeasurementMethod, PatchedInterpolationOptions, PatchedInterpolationResult,
    VerificationOptions, GLOBAL_ROUNDING_MARGIN, MEASUREMENT_ROUNDING_FACTOR,
};
use tensor4all_partitionedtreetn::{ErrorNorm, L2Reference};
use tensor4all_treetci::TreeTciInterpolator;

/// L2 options with the given reference norm.
fn l2_options(problem: &Problem, norm: f64) -> PatchedInterpolationOptions {
    PatchedInterpolationOptions::new(4)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
        .with_tolerance(tol(1e-6))
        .with_patch_order(extended_order(problem))
        .with_seed(5)
}

/// The certified run of test 1, shared with the rounding-model check.
struct CertifiedRun {
    reference: IdxTensor,
    norm: f64,
    result: PatchedInterpolationResult<Name>,
}

fn certified_run() -> &'static CertifiedRun {
    static RUN: OnceLock<CertifiedRun> = OnceLock::new();
    RUN.get_or_init(|| {
        let problem = extended_quantics_tree();
        let (reference, norm) = dense_reference(&problem, &extended_peak);
        // Every patch, the root included, has at most 1024 points and is
        // measured exhaustively.
        let options = l2_options(&problem, norm);
        assert_eq!(options.verification.max_exhaustive_points, 1024);
        let result = run(
            &TreeTciInterpolator::default(),
            &problem,
            &extended_peak,
            &[],
            &options,
        )
        .unwrap();
        CertifiedRun {
            reference,
            norm,
            result,
        }
    })
}

/// `MEASUREMENT_ROUNDING_FACTOR * eps * ||f~||`, the absolute rounding term
/// of one evaluation path.
fn rounding_term(result: &PatchedInterpolationResult<Name>) -> f64 {
    let error = result.report.norm.l2_error().unwrap();
    MEASUREMENT_ROUNDING_FACTOR * f64::EPSILON * error.approximation_norm().unwrap()
}

#[test]
fn the_extended_tree_is_branched_with_a_generic_path_node() {
    let problem = extended_quantics_tree();
    assert!(max_degree(&problem) >= 3);
    assert_eq!(problem.dims().iter().product::<usize>(), 768);
    assert!(problem.node_sites["r"].is_empty());
    assert_eq!(problem.node_sites["w"].len(), 2);
}

/// Test 1: a certified run against a dense reference.
#[test]
fn certified_l2_error_bounds_the_dense_error() {
    let CertifiedRun {
        reference,
        norm,
        result,
    } = certified_run();
    let report = &result.report;
    // Some patch is accepted from the engine with rank at least two, and is
    // measured exhaustively.
    assert!(report.accepted.iter().any(|record| {
        record.max_bond_dim >= 2
            && record.acceptance.as_ref().unwrap().method == MeasurementMethod::Exhaustive
    }));
    let error = report.norm.l2_error().unwrap();
    assert_eq!(error.certified_fraction, 1.0);
    let GlobalL2Error::Certified {
        rounding_limited,
        relative_error_bound,
        ..
    } = &error.global
    else {
        panic!("expected a certified error, got {:?}", error.global);
    };
    assert_eq!(*rounding_limited, Some(false));
    let bound = relative_error_bound.expect("a relative bound");

    let delta = report.norm.delta().unwrap();
    let diff = dense_l2_residual(result, reference);
    let rounding = rounding_term(result);
    assert!(
        diff <= delta * (1.0 + GLOBAL_ROUNDING_MARGIN) + 2.0 * rounding,
        "diff {diff:e}, delta {delta:e}, rounding {rounding:e}"
    );
    assert!(diff / norm <= bound + rounding / norm);
}

/// The calibration check of the rounding model (not a contract assertion):
/// the measured certified error agrees with the dense error up to the
/// rounding of both evaluation paths, with no `delta` slack. A failure means
/// `MEASUREMENT_ROUNDING_FACTOR` must be revisited, not that the
/// certificate is wrong.
#[test]
fn rounding_model_calibration() {
    let CertifiedRun {
        reference, result, ..
    } = certified_run();
    let error = result.report.norm.l2_error().unwrap();
    let measured = error.error_norm().unwrap();
    let diff = dense_l2_residual(result, reference);
    let rounding = rounding_term(result);
    assert!(
        (diff - measured).abs() <= GLOBAL_ROUNDING_MARGIN * diff + 2.0 * rounding,
        "dense {diff:e}, measured {measured:e}, rounding {rounding:e}"
    );
}

/// Fixed-seed regression constants of test 2 (not probabilistic claims):
/// the dense error is at most `SAMPLED_DELTA_FACTOR * delta`, and the audited
/// mean square lies within `SAMPLED_STANDARD_ERRORS` standard errors of the
/// true mean square.
const SAMPLED_DELTA_FACTOR: f64 = 1.0;
const SAMPLED_STANDARD_ERRORS: f64 = 4.0;

/// Test 2: the same problem measured on samples.
#[test]
fn sampled_l2_error_is_audited_or_acceptance_only() {
    let problem = extended_quantics_tree();
    let (reference, norm) = dense_reference(&problem, &extended_peak);
    let sampled = VerificationOptions::new()
        .with_samples(16)
        .with_max_exhaustive_points(0);
    let options = l2_options(&problem, norm).with_verification(sampled);
    let engine = TreeTciInterpolator::default();
    let result = run(&engine, &problem, &extended_peak, &[], &options).unwrap();
    let report = &result.report;
    assert!(report.accepted.iter().any(|record| {
        record.acceptance.as_ref().unwrap().method == MeasurementMethod::Sampled
            && record.audit.is_some()
    }));
    assert!(report.audit_evaluations > 0);
    assert!(report.audit_evaluations <= report.measurement_evaluations);

    let delta = report.norm.delta().unwrap();
    let diff = dense_l2_residual(&result, &reference);
    assert!(diff <= SAMPLED_DELTA_FACTOR * delta, "diff {diff:e}");
    let error = report.norm.l2_error().unwrap();
    let GlobalL2Error::Audited {
        rms_error_estimate,
        mean_square_rel_std_error,
        ..
    } = error.global
    else {
        panic!("expected an audited error, got {:?}", error.global);
    };
    if mean_square_rel_std_error > 0.0 {
        let estimate = rms_error_estimate * rms_error_estimate;
        let truth = diff * diff / error.domain_points;
        assert!(
            (estimate - truth).abs()
                <= SAMPLED_STANDARD_ERRORS * mean_square_rel_std_error * estimate,
            "estimate {estimate:e}, truth {truth:e}, rel se {mean_square_rel_std_error:e}"
        );
    }

    // Without the audit the same run is acceptance-only.
    let options = options.with_verification(sampled.with_audit(false));
    let result = run(&engine, &problem, &extended_peak, &[], &options).unwrap();
    let error = result.report.norm.l2_error().unwrap();
    assert!(matches!(error.global, GlobalL2Error::AcceptanceOnly { .. }));
    assert_eq!(result.report.audit_evaluations, 0);
    assert!(result
        .report
        .accepted
        .iter()
        .all(|record| record.audit.is_none()));
}

// ---------------------------------------------------------------------------
// Test 14: determinism on fresh threads
// ---------------------------------------------------------------------------

/// Run the same L2 problem in two fresh threads, each with its own problem
/// (fresh site IDs), and compare. Fingerprints are positional and ID-free.
fn assert_deterministic_across_threads(problem: fn() -> Problem, f: fn(&[usize]) -> f64) {
    let one_run = move || {
        std::thread::spawn(move || {
            let problem = problem();
            let (_, norm) = dense_reference(&problem, &f);
            let options = PatchedInterpolationOptions::new(3)
                .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
                .with_tolerance(tol(1e-6))
                .with_seed(11);
            let result = run(&TreeTciInterpolator::default(), &problem, &f, &[], &options).unwrap();
            run_digest(result, &problem)
        })
        .join()
        .unwrap()
    };
    assert_same_runs_across_problems(&one_run(), &one_run());
}

fn raw_kernel_peak(p: &[usize]) -> f64 {
    let (x, y, u) = (quantics(&p[0..2]), quantics(&p[2..4]), quantics(&p[5..7]));
    (1.0 + 0.5 * p[4] as f64) * gaussian(x, 0.3, 0.12) * gaussian(y, 0.6, 0.12) + 0.2 * (x * y + u)
}

#[test]
fn l2_runs_are_reproducible_on_fresh_threads_on_a_raw_kernel_tree() {
    assert!(max_degree(&raw_kernel_tree()) >= 3);
    assert_deterministic_across_threads(raw_kernel_tree, raw_kernel_peak);
}

#[test]
#[ignore = "open question 8 of docs/design/tree-patching-error-contract.md: the generic IdxTensor \
            path of TreeTNCachedEvaluator is not reproducible across threads (omeco breaks \
            contraction-path cost ties in HashMap order)"]
fn l2_runs_are_reproducible_on_fresh_threads_on_a_generic_path_tree() {
    assert_deterministic_across_threads(extended_quantics_tree, extended_peak);
}
