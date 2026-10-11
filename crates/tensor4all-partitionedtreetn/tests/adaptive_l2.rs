//! `patched_interpolate` under `ErrorNorm::L2` with the dense and scripted
//! test engines: the verification, retry, split, zero-patch, reference, and
//! error paths of the M3 error contract.

mod adaptive_common;

use std::collections::HashSet;

use adaptive_common::*;
use num_complex::{Complex32, Complex64};
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, CommonScalar, TensorElement};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate, GlobalL2Error, L2ReferenceSource, MeasurementMethod, NormReport,
    PatchedInterpolationError, PatchedInterpolationOptions, PatchedInterpolationReport,
    VerificationOptions, GLOBAL_ROUNDING_MARGIN, MEASUREMENT_ROUNDING_FACTOR,
};
use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference, Projector};
use tensor4all_treetn::interpolation::InterpolationError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Every acceptance measurement of the report, accepted and zero patches.
fn acceptances(report: &PatchedInterpolationReport) -> Vec<f64> {
    report
        .accepted
        .iter()
        .map(|record| record.acceptance.as_ref().unwrap().rms)
        .chain(
            report
                .zero_patches
                .iter()
                .map(|record| record.acceptance.as_ref().unwrap().rms),
        )
        .collect()
}

/// Exact-reference check of a certified run: `||f - f~|| <= delta (1 +
/// margin) + 2 R`, with `R` the rounding term of one evaluation path.
fn assert_certified_bound<T>(
    result: &tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationResult<Name>,
    problem: &Problem,
    f: &dyn Fn(&[usize]) -> T,
) where
    T: tensor4all_core::CommonScalar + tensor4all_core::TensorElement,
{
    let error = result.report.norm.l2_error().unwrap();
    assert!(
        matches!(error.global, GlobalL2Error::Certified { .. }),
        "{:?}",
        error.global
    );
    let (reference, _) = dense_reference(problem, f);
    let diff = dense_l2_residual(result, &reference);
    let rounding =
        MEASUREMENT_ROUNDING_FACTOR * f64::EPSILON * error.approximation_norm().unwrap_or(0.0);
    let delta = result.report.norm.delta().unwrap();
    assert!(
        diff <= delta * (1.0 + GLOBAL_ROUNDING_MARGIN) + 2.0 * rounding,
        "diff {diff:e}, delta {delta:e}"
    );
}

/// The full-domain point `point` restricted to the sites of a call.
fn local(problem: &Problem, call: &Call, point: &[usize]) -> Vec<usize> {
    call.site_order
        .iter()
        .map(|site| point[problem.position(site)])
        .collect()
}

fn no_pivots(problem: &Problem) -> ColMajorArray<usize> {
    ColMajorArray::new(vec![], vec![problem.sites.len(), 0]).unwrap()
}

// ---------------------------------------------------------------------------
// Budget arithmetic on a dense-engine run (test 11)
// ---------------------------------------------------------------------------

#[test]
fn dense_engine_runs_meet_the_l2_budget() {
    let problem = branched();
    assert_eq!(max_degree(&problem), 3);
    let j0 = problem.site("j", 0);
    let f = switch_on(problem.position(&j0));
    let (_, norm) = dense_reference(&problem, &f);
    let options = l2_given(2, norm, 1e-8).with_patch_order(vec![j0]);
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    let report = &result.report;
    assert_eq!(report.splits, 1);
    let tau = tau(report);
    assert!(acceptances(report).iter().all(|&rms| rms <= tau));

    // The global error combines the patches by volume.
    let error = report.norm.l2_error().unwrap();
    assert_eq!(error.domain_points, 96.0);
    assert_eq!(error.certified_fraction, 1.0);
    let combined: f64 = report
        .accepted
        .iter()
        .map(|record| {
            let m = record.acceptance.as_ref().unwrap();
            m.patch_points / error.domain_points * m.rms * m.rms
        })
        .sum::<f64>()
        .sqrt();
    let GlobalL2Error::Certified { rms_error, .. } = error.global else {
        panic!("expected a certified error");
    };
    assert!((rms_error - combined).abs() <= GLOBAL_ROUNDING_MARGIN * combined);
    assert!(rms_error <= tau * (1.0 + GLOBAL_ROUNDING_MARGIN));
    // The approximation norm matches the norm of the materialized partition,
    // computed densely and as a network norm. The direct sum has a site-free
    // leaf with a bond of dimension two, which `TreeTN::norm` handles since
    // #799.
    let approximation = error.approximation_norm().unwrap();
    let sum = result.partition.to_treetn().unwrap();
    let direct = sum.contract_to_tensor().unwrap().norm().unwrap();
    assert!(
        (approximation - direct).abs() <= 1e-12 * direct,
        "approximation {approximation:e}, direct {direct:e}"
    );
    let network = sum.clone().norm().unwrap();
    assert!(
        (network - direct).abs() <= 1e-12 * direct,
        "network norm {network:e}, direct {direct:e}"
    );
    assert_certified_bound(&result, &problem, &f);
}

// ---------------------------------------------------------------------------
// A missed feature (test 3)
// ---------------------------------------------------------------------------

/// A product function with a narrow spike at `SPIKE` (site order a0, b0, c0,
/// d0, d1, j0 of `branched`).
const SPIKE: [usize; 6] = [2, 1, 1, 1, 0, 1];

fn spiked(p: &[usize]) -> f64 {
    product(p, 0) + if p == SPIKE { 3.0 } else { 0.0 }
}

#[test]
fn a_missed_feature_is_caught_and_rerun_with_the_worst_point() {
    let problem = branched();
    let (_, norm) = dense_reference(&problem, &spiked);
    let options = l2_given(3, norm, 1e-10);
    assert_eq!(options.verification.retries, 1);
    let engine = SpikeBlindEngine::new(&problem.sites, &SPIKE);
    let result = run(&engine, &problem, &spiked, &[], &options).unwrap();
    let calls = engine.calls();
    // Run 0 misses the spike and converges with rank one; the exhaustive
    // verification rejects it, and run 1 receives the spike first.
    assert_eq!(calls.len(), 2);
    assert!(!calls[0].initial_pivots.contains(&SPIKE.to_vec()));
    let base = calls[0].initial_pivots.len();
    assert_eq!(calls[1].initial_pivots[..base], calls[0].initial_pivots[..]);
    assert_eq!(calls[1].initial_pivots[base], SPIKE.to_vec());
    assert_ne!(calls[0].seed, calls[1].seed);
    let report = &result.report;
    assert_eq!(report.splits, 0);
    assert_eq!(
        (report.verification_failures, report.engine_retries),
        (1, 1)
    );
    let record = &report.accepted[0];
    assert_eq!(record.retries_used, 1);
    assert_eq!(record.max_bond_dim, 2);
    assert_eq!(
        record.acceptance.as_ref().unwrap().method,
        MeasurementMethod::Exhaustive
    );
    assert_certified_bound(&result, &problem, &spiked);
}

#[test]
fn without_retries_a_missed_feature_splits_and_the_child_gets_the_worst_point() {
    let problem = branched();
    let j0 = problem.site("j", 0);
    let (_, norm) = dense_reference(&problem, &spiked);
    let options = l2_given(3, norm, 1e-10)
        .with_verification(VerificationOptions::new().with_retries(0))
        .with_patch_order(vec![j0.clone()]);
    let engine = SpikeBlindEngine::new(&problem.sites, &SPIKE);
    let result = run(&engine, &problem, &spiked, &[], &options).unwrap();
    let report = &result.report;
    assert_eq!(report.splits, 1);
    assert_eq!(
        (report.verification_failures, report.engine_retries),
        (1, 0)
    );
    // The child that contains the spike receives it as a candidate.
    let calls = engine.calls();
    let child = calls.iter().find(|call| {
        !call.site_order.contains(&j0) && {
            let spike = local(&problem, call, &SPIKE);
            call.initial_pivots.contains(&spike)
        }
    });
    assert!(child.is_some(), "no child received the spike");
    assert!(report.accepted.iter().all(|r| r.retries_used == 0));
    assert_certified_bound(&result, &problem, &spiked);
}

/// `f = 1` except at two points with residuals 5 and 3 against the
/// constant network 1.
fn two_spikes(p: &[usize]) -> f64 {
    match p {
        [1, 2] => 6.0,
        [3, 0] => 4.0,
        _ => 1.0,
    }
}

/// The added list of a rerun computed from the documented rule.
fn expected_added(
    base: &[Vec<usize>],
    worst: &[Vec<usize>],
    outcome: &[Vec<usize>],
    limit: usize,
) -> Vec<Vec<usize>> {
    let mut seen: HashSet<Vec<usize>> = base.iter().cloned().collect();
    worst
        .iter()
        .chain(outcome)
        .filter(|p| seen.insert((*p).clone()))
        .take(limit)
        .cloned()
        .collect()
}

#[test]
fn rerun_pivots_put_the_worst_points_before_the_outcome_pivots_without_accumulating() {
    let problem = single_node(&[4, 4]);
    let (_, norm) = dense_reference(&problem, &two_spikes);
    let first_outcome = vec![vec![0, 0], vec![2, 2], vec![3, 3], vec![0, 3], vec![1, 1]];
    let second_outcome = vec![vec![2, 1], vec![1, 3], vec![3, 2]];
    let constant = Step::converged(Network::Constant(1.0));
    let engine = ScriptedEngine::new(vec![
        constant.clone().with_pivots(first_outcome.clone()),
        constant.clone().with_pivots(second_outcome.clone()),
        constant,
        Step::converged(Network::Exact),
    ]);
    let cap = 5;
    let options = l2_given(cap, norm, 1e-6)
        .with_n_initial_pivots(2)
        .with_verification(VerificationOptions::new().with_retries(2))
        .with_patch_order(vec![problem.site("only", 0)]);
    let result = run(&engine, &problem, &two_spikes, &[vec![0, 0]], &options).unwrap();
    let calls = engine.calls();
    let base = calls[0].initial_pivots.clone();
    assert_eq!(base[0], vec![0, 0]);
    let worst = [vec![1, 2], vec![3, 0]];
    let added = |outcome: &[Vec<usize>]| {
        let mut initial = base.clone();
        initial.extend(expected_added(&base, &worst, outcome, cap - 1));
        initial
    };
    assert_eq!(calls[1].initial_pivots, added(&first_outcome));
    assert_eq!(calls[2].initial_pivots, added(&second_outcome));
    // At most max_bond_dim - 1 added points, and none of the first rerun's
    // outcome pivots carried into the second.
    assert_eq!(calls[1].initial_pivots.len() - base.len(), cap - 1);
    assert!(!calls[2].initial_pivots.contains(&vec![2, 2]));
    let report = &result.report;
    assert_eq!(
        (report.verification_failures, report.engine_retries),
        (3, 2)
    );
    assert_eq!(report.splits, 1);
}

// ---------------------------------------------------------------------------
// A sampled retry on a fresh stream (tests 4 and 15)
// ---------------------------------------------------------------------------

#[test]
fn a_sampled_retry_measures_on_a_fresh_stream_and_audit_points_never_become_pivots() {
    let problem = chain("s", 12, 2);
    let one = |_: &[usize]| 1.0;
    let engine = ScriptedEngine::new(vec![
        Step::converged(Network::Constant(0.0)),
        Step::converged(Network::Constant(1.0)),
    ]);
    let seed = 3;
    let options = l2_given(2, 64.0, 1e-6).with_seed(seed).with_verification(
        VerificationOptions::new()
            .with_samples(16)
            .with_max_exhaustive_points(0),
    );
    let recorder = Recorder::default();
    let evaluate = recording_evaluator(&one, problem.sites.len(), &recorder);
    let result = run_with(&engine, &problem, evaluate, &[], &options).unwrap();
    let record = &result.report.accepted[0];
    assert_eq!(record.retries_used, 1);
    assert_eq!(
        record.acceptance.as_ref().unwrap().method,
        MeasurementMethod::Sampled
    );
    assert!(record.audit.is_some());

    let dims = problem.dims();
    let state = streams::path_state(seed, &[]);
    let stream = |stream_seed| streams::draw(&dims, 16, stream_seed);
    let evaluated = recorder.seen.lock().unwrap().clone();
    let all_evaluated = |points: &[Vec<usize>]| points.iter().all(|p| evaluated.contains(p));
    let (first, second) = (
        stream(streams::verify(state, 0)),
        stream(streams::verify(state, 1)),
    );
    assert_ne!(first, second);
    assert!(all_evaluated(&first));
    assert!(all_evaluated(&second));
    // A stream the run did not use is not evaluated.
    assert!(!all_evaluated(&stream(streams::verify(state, 2))));
    let audit = stream(streams::audit(state));
    assert!(all_evaluated(&audit));
    for call in engine.calls() {
        assert!(audit.iter().all(|p| !call.initial_pivots.contains(p)));
    }
    // The rerun received the first worst point of the failed measurement.
    let calls = engine.calls();
    assert!(calls[1].initial_pivots.contains(&first[0]));
}

#[test]
fn measured_values_reach_the_children_without_reevaluation() {
    let problem = branched();
    let j0 = problem.site("j", 0);
    let f = switch_on(problem.position(&j0));
    let (_, norm) = dense_reference(&problem, &f);
    let engine = ScriptedEngine::new(vec![
        Step::converged(Network::Constant(0.0)),
        Step::converged(Network::Exact),
    ]);
    let options = l2_given(2, norm, 1e-8)
        .with_verification(VerificationOptions::new().with_retries(0))
        .with_patch_order(vec![j0]);
    let result = run(&engine, &problem, &f, &[], &options).unwrap();
    let report = &result.report;
    // The root's exhaustive measurement evaluated every point that its
    // candidates had not; the children evaluate nothing new.
    let candidates = engine.calls()[0].initial_pivots.len();
    assert_eq!(report.function_evaluations, 96);
    assert_eq!(report.measurement_evaluations, 96 - candidates);
    assert_eq!(report.accepted.len(), 2);
}

// ---------------------------------------------------------------------------
// Split-time candidates (test 5)
// ---------------------------------------------------------------------------

/// `f = 1` plus spikes of decreasing size at A, B, C, D, E (site order p0,
/// p1, p2 of `single_node(&[2, 4, 4])`).
const SPIKES: [([usize; 3], f64); 5] = [
    ([0, 0, 1], 9.0),
    ([1, 3, 3], 7.0),
    ([0, 2, 0], 5.0),
    ([1, 0, 2], 3.0),
    ([0, 3, 2], 2.0),
];

fn spikes(p: &[usize]) -> f64 {
    1.0 + SPIKES
        .iter()
        .find(|(point, _)| point == p)
        .map_or(0.0, |(_, size)| *size)
}

/// Run the split-time candidate scenario and return the initial pivots of
/// the two children, in child coordinates.
fn split_children(
    recycle: bool,
    n_initial_pivots: usize,
    rerun_caps: bool,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, PatchedInterpolationReport) {
    let problem = single_node(&[2, 4, 4]);
    let (_, norm) = dense_reference(&problem, &spikes);
    let root =
        Step::converged(Network::Constant(1.0)).with_pivots(vec![vec![0, 1, 2], vec![1, 1, 0]]);
    let mut steps = vec![root];
    let retries = if rerun_caps {
        steps.push(Step::capped().with_pivots(vec![vec![0, 3, 0], vec![1, 2, 1]]));
        1
    } else {
        0
    };
    steps.push(Step::converged(Network::Exact));
    let engine = ScriptedEngine::new(steps);
    let options = l2_given(4, norm, 1e-6)
        .with_recycle_pivots(recycle)
        .with_n_initial_pivots(n_initial_pivots)
        .with_verification(VerificationOptions::new().with_retries(retries))
        .with_patch_order(vec![problem.site("only", 0)]);
    let result = run(
        &engine,
        &problem,
        &spikes,
        &[vec![0, 1, 1], vec![1, 2, 2]],
        &options,
    )
    .unwrap();
    let calls = engine.calls();
    let children = &calls[calls.len() - 2..];
    assert!(children.iter().all(|call| call.site_order.len() == 2));
    (
        children[0].initial_pivots.clone(),
        children[1].initial_pivots.clone(),
        result.report,
    )
}

#[test]
fn split_children_start_from_user_recycled_and_worst_points_then_random_fill() {
    // Worst points of the root, truncated to max_bond_dim - 1 = 3 for the
    // whole patch: A (child 0), B (child 1), C (child 0).
    let (first, second, report) = split_children(true, 6, false);
    assert_eq!(
        (report.verification_failures, report.engine_retries),
        (1, 0)
    );
    // User pivot, recycled pivot, then the worst points in descending |r|.
    assert_eq!(first[..4], [vec![1, 1], vec![1, 2], vec![0, 1], vec![2, 0]]);
    assert_eq!(first.len(), 6);
    assert_eq!(second[..3], [vec![2, 2], vec![1, 0], vec![3, 3]]);
    assert_eq!(second.len(), 6);
    let distinct: HashSet<&Vec<usize>> = first.iter().collect();
    assert_eq!(distinct.len(), first.len());

    // Without recycling the recycled pivots are absent.
    let (first, second, _) = split_children(false, 6, false);
    assert_eq!(first[..3], [vec![1, 1], vec![0, 1], vec![2, 0]]);
    assert_eq!(second[..2], [vec![2, 2], vec![3, 3]]);
    assert_eq!(first.len(), 6);

    // Earlier sources reaching n_initial_pivots leave no random fill.
    let (first, second, _) = split_children(true, 2, false);
    assert_eq!(first, [vec![1, 1], vec![1, 2], vec![0, 1], vec![2, 0]]);
    assert_eq!(second, [vec![2, 2], vec![1, 0], vec![3, 3]]);
}

#[test]
fn a_rerun_that_does_not_converge_passes_the_preceding_worst_points() {
    let (first, second, report) = split_children(true, 2, true);
    assert_eq!(
        (report.verification_failures, report.engine_retries),
        (1, 1)
    );
    // Recycled pivots come from the capped rerun, worst points from the
    // failed measurement before it.
    assert_eq!(first, [vec![1, 1], vec![3, 0], vec![0, 1], vec![2, 0]]);
    assert_eq!(second, [vec![2, 2], vec![2, 1], vec![3, 3]]);
}

// ---------------------------------------------------------------------------
// Failure at the end of the order (test 6)
// ---------------------------------------------------------------------------

#[test]
fn failures_at_the_end_of_the_order_are_typed() {
    let problem = single_node(&[2, 4, 4]);
    let p0 = problem.site("only", 0);
    let two = |_: &[usize]| 2.0;
    let options = l2_given(2, 2.0 * 32f64.sqrt(), 1e-6)
        .with_verification(VerificationOptions::new().with_retries(0))
        .with_patch_order(vec![p0.clone()]);
    let wrong = || ScriptedEngine::new(vec![Step::converged(Network::Constant(1.0))]);
    match run(&wrong(), &problem, &two, &[], &options) {
        Err(PatchedInterpolationError::VerificationFailed {
            projector,
            measurement,
        }) => {
            assert_eq!(projector, Projector::from_pairs([(p0.clone(), 0)]).unwrap());
            assert_eq!(measurement.method, MeasurementMethod::Exhaustive);
            assert_eq!(measurement.rms, 1.0);
        }
        other => panic!("expected VerificationFailed, got {other:?}"),
    }
    // A patch that does not converge keeps the M2 error.
    let capped = ScriptedEngine::new(vec![Step::capped()]);
    match run(&capped, &problem, &two, &[], &options) {
        Err(PatchedInterpolationError::NoSplitIndexLeft { projector }) => {
            assert_eq!(projector, Projector::from_pairs([(p0, 0)]).unwrap());
        }
        other => panic!("expected NoSplitIndexLeft, got {other:?}"),
    }
    // The patch limit after a failed verification.
    match run(
        &wrong(),
        &problem,
        &two,
        &[],
        &options.clone().with_max_patches(1),
    ) {
        Err(PatchedInterpolationError::ResourceLimit { resource, limit }) => {
            assert_eq!((resource, limit), ("max_patches", 1));
        }
        other => panic!("expected ResourceLimit, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Tolerances below roundoff (test 7)
// ---------------------------------------------------------------------------

#[test]
fn tolerances_below_roundoff_follow_the_retry_and_split_path() {
    let problem = branched();
    let f = |p: &[usize]| product(p, 0);
    let (_, norm) = dense_reference(&problem, &f);
    // Far below eps times any value scale of f.
    for tolerance in [
        ErrorTolerance {
            rtol: 1e-30,
            atol: 0.0,
        },
        ErrorTolerance {
            rtol: 0.0,
            atol: 0.0,
        },
    ] {
        let options = PatchedInterpolationOptions::new(2)
            .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
            .with_tolerance(tolerance);
        let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
        let report = &result.report;
        assert!(report.verification_failures >= 1);
        assert!(report.splits >= 1);
        let tau = tau(report);
        assert!(acceptances(report).iter().all(|&rms| rms <= tau));
        assert!(report.accepted.iter().any(|record| {
            record.acceptance.as_ref().unwrap().method == MeasurementMethod::Exact
        }));
        let error = report.norm.l2_error().unwrap();
        assert!(matches!(error.global, GlobalL2Error::Certified { .. }));

        // Constrained runs report their normal errors.
        let limited = options.clone().with_max_patches(3);
        assert!(matches!(
            run(&DenseEngine::new(), &problem, &f, &[], &limited),
            Err(PatchedInterpolationError::ResourceLimit { .. })
        ));
        let partial = options.with_patch_order(vec![problem.site("a", 0)]);
        assert!(matches!(
            run(&DenseEngine::new(), &problem, &f, &[], &partial),
            Err(PatchedInterpolationError::VerificationFailed { .. })
        ));
    }
}

// ---------------------------------------------------------------------------
// Zero patches (test 8)
// ---------------------------------------------------------------------------

/// Seed whose single random root candidate of `branched` has b0 = 0.
const ZERO_CANDIDATE_SEED: u64 = 1;

#[test]
fn a_failed_zero_screen_hands_its_largest_points_to_the_engine() {
    let problem = branched();
    let pb = problem.position(&problem.site("b", 0));
    // Nonzero on half of the domain (b0 = 1).
    let f = move |p: &[usize]| if p[pb] == 1 { product(p, 0) } else { 0.0 };
    let (_, norm) = dense_reference(&problem, &f);
    let options = l2_given(2, norm, 1e-8)
        .with_n_initial_pivots(1)
        .with_seed(ZERO_CANDIDATE_SEED);
    let engine = DenseEngine::new();
    let result = run(&engine, &problem, &f, &[], &options).unwrap();
    let seen = engine.seen();
    let initial = &seen[0].initial_pivots;
    // The candidate is a zero; the screen adds max_bond_dim - 1 = 1 point,
    // the largest |f| of the exhaustive screen.
    assert_eq!(initial.len(), 2);
    assert_eq!(f(&initial[0]), 0.0);
    let largest = full_domain(&problem.dims())
        .iter()
        .map(|p| f(p))
        .fold(0.0, f64::max);
    assert_eq!(f(&initial[1]), largest);
    let report = &result.report;
    assert_eq!(report.zero_patches.len(), 0);
    assert_eq!(report.accepted.len(), 1);
    assert_eq!(report.accepted[0].retries_used, 0);
    assert_eq!(
        (report.verification_failures, report.engine_retries),
        (0, 0)
    );
    // The screen evaluated every point but the candidate; the verification
    // then needs no new value.
    assert_eq!(report.function_evaluations, 96);
    assert_eq!(report.measurement_evaluations, 95);
    assert_certified_bound(&result, &problem, &f);
}

#[test]
fn a_truly_zero_region_is_a_verified_zero_patch() {
    let problem = branched();
    let (j0, a0) = (problem.site("j", 0), problem.site("a", 0));
    let (pj, pa) = (problem.position(&j0), problem.position(&a0));
    let f = move |p: &[usize]| if p[pj] == 0 { 0.0 } else { product(p, p[pa]) };
    let (_, norm) = dense_reference(&problem, &f);
    let options = l2_given(2, norm, 1e-12).with_patch_order(vec![j0.clone(), a0]);
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    let zero = &result.report.zero_patches;
    assert_eq!(zero.len(), 1);
    assert_eq!(zero[0].projector, Projector::from_pairs([(j0, 0)]).unwrap());
    let measurement = zero[0].acceptance.as_ref().unwrap();
    assert_eq!(measurement.method, MeasurementMethod::Exhaustive);
    assert_eq!((measurement.points, measurement.rms), (48, 0.0));
    assert!(zero[0].audit.is_none());
    assert_certified_bound(&result, &problem, &f);
}

#[test]
fn a_run_whose_approximation_does_not_exceed_its_error_has_no_relative_statement() {
    let problem = branched();
    // Nonzero only at one point outside the single candidate; within atol.
    let f = |p: &[usize]| if p == SPIKE { 1e-3 } else { 0.0 };
    let options = PatchedInterpolationOptions::new(2)
        .with_tolerance(ErrorTolerance {
            rtol: 0.0,
            atol: 1.0,
        })
        .with_n_initial_pivots(1)
        .with_seed(ZERO_CANDIDATE_SEED);
    let engine = DenseEngine::new();
    let result = run(&engine, &problem, &f, &[], &options).unwrap();
    assert!(engine.seen().is_empty());
    assert!(result.partition.is_empty());
    let error = result.report.norm.l2_error().unwrap();
    assert_eq!(error.approximation_rms, Some(0.0));
    let GlobalL2Error::Certified {
        rms_error,
        relative_error_bound,
        ..
    } = error.global
    else {
        panic!("expected a certified error");
    };
    assert!(rms_error > 0.0);
    assert_eq!(relative_error_bound, None);
}

// ---------------------------------------------------------------------------
// The exhaustive threshold and exact small patches (tests 9 and 10)
// ---------------------------------------------------------------------------

#[test]
fn the_exhaustive_threshold_is_inclusive() {
    let problem = single_node(&[4, 4]);
    let f = |p: &[usize]| product(p, 0);
    let (_, norm) = dense_reference(&problem, &f);
    for (samples, max_exhaustive, method) in [
        (16, 0, MeasurementMethod::Exhaustive),
        (15, 0, MeasurementMethod::Sampled),
        (2, 16, MeasurementMethod::Exhaustive),
        (2, 15, MeasurementMethod::Sampled),
    ] {
        let options = l2_given(2, norm, 1e-8).with_verification(
            VerificationOptions::new()
                .with_samples(samples)
                .with_max_exhaustive_points(max_exhaustive),
        );
        let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
        let measurement = result.report.accepted[0].acceptance.clone().unwrap();
        assert_eq!(measurement.method, method, "{samples} {max_exhaustive}");
        let expected_points = if method == MeasurementMethod::Exhaustive {
            16
        } else {
            samples
        };
        assert_eq!(measurement.points, expected_points);
        assert_eq!(measurement.patch_points, 16.0);
    }
}

#[test]
fn exact_small_patches_carry_exact_measurements_and_add_no_evaluations() {
    // Rank two at the root; every child has one active site.
    let problem = Problem::new(&[("s", &[3]), ("t", &[8])], &[("s", "t")]);
    let f = |p: &[usize]| match p[0] {
        0 => 0.0,
        1 => 1.0 + p[1] as f64,
        _ => 2.0 - p[1] as f64,
    };
    let (_, norm) = dense_reference(&problem, &f);
    let result = run(
        &DenseEngine::new(),
        &problem,
        &f,
        &[],
        &l2_given(2, norm, 1e-12),
    )
    .unwrap();
    let report = &result.report;
    assert_eq!(report.splits, 1);
    assert_eq!(report.measurement_evaluations, 0);
    let measurements = report
        .accepted
        .iter()
        .map(|record| record.acceptance.clone().unwrap())
        .chain(
            report
                .zero_patches
                .iter()
                .map(|record| record.acceptance.clone().unwrap()),
        );
    for measurement in measurements {
        assert_eq!(measurement.method, MeasurementMethod::Exact);
        assert_eq!((measurement.points, measurement.rms), (8, 0.0));
        assert_eq!(measurement.max_residual, 0.0);
    }
    assert_eq!(report.zero_patches.len(), 1);
    let error = report.norm.l2_error().unwrap();
    assert!(matches!(
        error.global,
        GlobalL2Error::Certified { rms_error, .. } if rms_error == 0.0
    ));
}

// ---------------------------------------------------------------------------
// Overflow and rounding-limited reports (test 11)
// ---------------------------------------------------------------------------

#[test]
fn an_overflowing_approximation_norm_is_reported_as_none() {
    // ||f~|| = 2e155 exceeds sqrt(f64::MAX); the exact path builds the patch.
    let problem = single_node(&[4]);
    let f = |_: &[usize]| 1e155;
    let options = PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(2e155)))
        .with_tolerance(tol(1e-6));
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    let error = result.report.norm.l2_error().unwrap();
    assert_eq!(error.approximation_rms, None);
    assert_eq!(error.approximation_norm(), None);
    assert!(matches!(
        error.global,
        GlobalL2Error::Certified {
            rms_error,
            rounding_allowance_rms: None,
            rounding_limited: None,
            relative_error_bound: None,
            ..
        } if rms_error == 0.0
    ));
}

#[test]
fn a_tau_below_the_rounding_term_is_rounding_limited() {
    let problem = single_node(&[4]);
    let f = |_: &[usize]| 1.0;
    let options = PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(2.0)))
        .with_tolerance(tol(1e-20));
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    let error = result.report.norm.l2_error().unwrap();
    let GlobalL2Error::Certified {
        rounding_allowance_rms,
        rounding_limited,
        ..
    } = error.global
    else {
        panic!("expected a certified error");
    };
    assert!(rounding_allowance_rms.unwrap() > tau(&result.report));
    assert_eq!(rounding_limited, Some(true));
}

/// The rounding allowance scales with the machine epsilon of the evaluated
/// scalar type: `tau = 1e-6` resolves the `f64` allowance (about `1.4e-13`)
/// but not the `f32` one (about `7.7e-5`).
fn check_rounding_allowance_follows_the_scalar_type<T>(epsilon: f64, limited: bool)
where
    T: CommonScalar + TensorElement,
{
    let problem = single_node(&[4]);
    let f = |_: &[usize]| T::from_f64(1.0);
    let options = l2_given(2, 2.0, 1e-6);
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    let error = result.report.norm.l2_error().unwrap();
    let approximation_rms = error.approximation_rms.unwrap();
    let GlobalL2Error::Certified {
        rms_error,
        rounding_allowance_rms,
        rounding_limited,
        ..
    } = error.global
    else {
        panic!("expected a certified error");
    };
    assert_eq!(rms_error, 0.0);
    assert_eq!(
        rounding_allowance_rms,
        Some(MEASUREMENT_ROUNDING_FACTOR * epsilon * approximation_rms)
    );
    assert_eq!(rounding_limited, Some(limited));
}

#[test]
fn the_rounding_allowance_follows_the_scalar_type() {
    let f32_epsilon = f64::from(f32::EPSILON);
    check_rounding_allowance_follows_the_scalar_type::<f64>(f64::EPSILON, false);
    check_rounding_allowance_follows_the_scalar_type::<Complex64>(f64::EPSILON, false);
    check_rounding_allowance_follows_the_scalar_type::<f32>(f32_epsilon, true);
    check_rounding_allowance_follows_the_scalar_type::<Complex32>(f32_epsilon, true);
}

// ---------------------------------------------------------------------------
// References (test 12)
// ---------------------------------------------------------------------------

#[test]
fn every_reference_source_pins_tau() {
    let problem = branched();
    let f = |p: &[usize]| product(p, 0);
    let (_, norm) = dense_reference(&problem, &f);
    let root_points = 96f64.sqrt();

    let given = run(
        &DenseEngine::new(),
        &problem,
        &f,
        &[],
        &l2_given(2, norm, 1e-6),
    )
    .unwrap();
    let NormReport::L2 {
        reference_rms,
        source,
        tau,
        ..
    } = given.report.norm
    else {
        panic!("expected an L2 report");
    };
    assert_eq!(source, L2ReferenceSource::Given);
    assert_eq!(reference_rms, Some(norm / root_points));
    assert_eq!(tau, 1e-6 * (norm / root_points));
    assert!((given.report.norm.reference_norm().unwrap() - norm).abs() <= 1e-12 * norm);

    // Not needed when rtol = 0, whatever the reference option.
    for reference in [L2Reference::Required, L2Reference::MonteCarlo] {
        let options = PatchedInterpolationOptions::new(2)
            .with_error_norm(ErrorNorm::l2(reference))
            .with_tolerance(ErrorTolerance {
                rtol: 0.0,
                atol: 1e-3,
            });
        let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
        let NormReport::L2 {
            reference_rms,
            source,
            tau,
            ..
        } = result.report.norm
        else {
            panic!("expected an L2 report");
        };
        assert_eq!(source, L2ReferenceSource::NotNeeded);
        assert_eq!(reference_rms, None);
        assert_eq!(tau, 1e-3 / root_points);
        assert_eq!(result.report.norm.reference_norm(), None);
    }

    // The Monte Carlo estimate: the RMS of f over the reference stream.
    let options = PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::MonteCarlo))
        .with_tolerance(tol(1e-6))
        .with_seed(4);
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    let NormReport::L2 {
        reference_rms,
        source,
        ..
    } = result.report.norm
    else {
        panic!("expected an L2 report");
    };
    let L2ReferenceSource::MonteCarlo {
        samples,
        mean_square_rel_std_error,
        ..
    } = source
    else {
        panic!("expected a Monte Carlo reference, got {source:?}");
    };
    assert_eq!(samples, 64);
    assert!(mean_square_rel_std_error > 0.0);
    let points = streams::draw(
        &problem.dims(),
        64,
        streams::scale(streams::path_state(4, &[])),
    );
    let expected = (points.iter().map(|p| f(p) * f(p)).sum::<f64>() / 64.0).sqrt();
    let reference_rms = reference_rms.unwrap();
    assert!((reference_rms - expected).abs() <= 1e-14 * expected);
    assert!(result.report.measurement_evaluations >= 1);
}

#[test]
fn an_exact_root_gives_the_reference_and_an_all_zero_exact_root_is_empty() {
    let problem = single_node(&[4]);
    let f = |p: &[usize]| [3.0, 0.0, 4.0, 0.0][p[0]];
    let result = run(
        &DenseEngine::new(),
        &problem,
        &f,
        &[],
        &PatchedInterpolationOptions::new(2).with_tolerance(tol(1e-6)),
    )
    .unwrap();
    let NormReport::L2 {
        reference_rms,
        source,
        ..
    } = result.report.norm
    else {
        panic!("expected an L2 report");
    };
    assert_eq!(source, L2ReferenceSource::ExactRoot);
    assert_eq!(reference_rms, Some(2.5));

    let problem = single_node(&[3]);
    let zero = |_: &[usize]| 0.0;
    for options in [
        PatchedInterpolationOptions::new(2),
        l2_given(2, 1.0, 1e-6),
        PatchedInterpolationOptions::new(2).with_tolerance(tol(0.0)),
        PatchedInterpolationOptions::new(2).with_error_norm(ErrorNorm::l2(L2Reference::MonteCarlo)),
    ] {
        let result = run(&DenseEngine::new(), &problem, &zero, &[], &options).unwrap();
        assert!(result.partition.is_empty());
        assert_eq!(result.report.zero_patches.len(), 1);
        assert_eq!(
            result.report.zero_patches[0]
                .acceptance
                .as_ref()
                .unwrap()
                .method,
            MeasurementMethod::Exact
        );
    }
}

#[test]
fn a_required_reference_fails_before_any_evaluation() {
    let problem = branched();
    let message = expect_invalid(
        patched_interpolate(
            &DenseEngine::new(),
            problem.topology.clone(),
            problem.node_sites.clone(),
            no_pivots(&problem),
            never,
            &PatchedInterpolationOptions::new(2),
        ),
        "needs a reference norm",
    );
    assert!(message.contains("L2Reference::Given"));
    assert!(message.contains("L2Reference::MonteCarlo"));
}

#[test]
fn a_zero_monte_carlo_reference_needs_an_absolute_floor() {
    let problem = branched();
    let zero = |_: &[usize]| 0.0;
    let monte_carlo = PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::MonteCarlo))
        .with_tolerance(tol(1e-6));
    let message = expect_invalid(
        run(&DenseEngine::new(), &problem, &zero, &[], &monte_carlo),
        "Monte Carlo reference norm is zero",
    );
    assert!(message.contains("tolerance.atol"));
    // With an absolute floor the root is accepted as a zero patch.
    let floor = monte_carlo.with_tolerance(ErrorTolerance {
        rtol: 1e-6,
        atol: 1e-9,
    });
    let result = run(&DenseEngine::new(), &problem, &zero, &[], &floor).unwrap();
    assert!(result.partition.is_empty());
    assert_eq!(result.report.zero_patches.len(), 1);
}

// ---------------------------------------------------------------------------
// SampledMax (test 13)
// ---------------------------------------------------------------------------

#[test]
fn sampled_max_engine_tolerance_is_raised_by_atol() {
    let problem = branched();
    let f = |p: &[usize]| product(p, 0);
    for (atol, expected) in [(1e-3, 1e-3), (1e-12, 2e-8)] {
        let options = sampled_max(2)
            .with_error_norm(ErrorNorm::sampled_max_with_reference(2.0))
            .with_tolerance(ErrorTolerance { rtol: 1e-8, atol });
        let engine = DenseEngine::new();
        let result = run(&engine, &problem, &f, &[], &options).unwrap();
        assert_eq!(engine.seen()[0].tolerance, expected);
        assert!(matches!(
            result.report.norm,
            NormReport::SampledMax { engine_tolerance, .. } if engine_tolerance == expected
        ));
        assert_eq!(result.report.measurement_evaluations, 0);
        assert!(result.report.accepted[0].acceptance.is_none());
    }
}

#[test]
fn a_domain_too_large_for_f64_is_accepted_only_under_sampled_max() {
    // 2^1100 points overflow f64.
    let problem = chain("s", 1100, 2);
    let f = |p: &[usize]| {
        p.iter()
            .enumerate()
            .map(|(i, &x)| 1.0 + 1e-4 * ((i % 7) * x) as f64)
            .product::<f64>()
    };
    let pivots = [vec![0; 1100]];
    let options = sampled_max(2).with_error_norm(ErrorNorm::sampled_max_with_reference(2.0));
    let result = run(&FiberEngine, &problem, &f, &pivots, &options).unwrap();
    assert_eq!(result.report.accepted.len(), 1);

    let message = expect_invalid(
        patched_interpolate(
            &FiberEngine,
            problem.topology.clone(),
            problem.node_sites.clone(),
            problem.pivots(&pivots),
            never,
            &l2_given(2, 1.0, 1e-6),
        ),
        "more points than f64",
    );
    assert!(message.contains("ErrorNorm::sampled_max()"));
}

// ---------------------------------------------------------------------------
// Determinism on fresh threads (test 14)
// ---------------------------------------------------------------------------

fn dense_run_on_a_fresh_thread(
    problem: fn() -> Problem,
    f: fn(&[usize]) -> f64,
    verification: VerificationOptions,
) -> RunDigest {
    std::thread::spawn(move || {
        let problem = problem();
        let (_, norm) = dense_reference(&problem, &f);
        let options = l2_given(2, norm, 1e-10).with_verification(verification);
        let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
        run_digest(result, &problem)
    })
    .join()
    .unwrap()
}

fn raw_kernel_switch(p: &[usize]) -> f64 {
    product(p, p[4])
}

fn generic_path_switch(p: &[usize]) -> f64 {
    product(p, p[8])
}

#[test]
fn dense_engine_l2_runs_are_reproducible_on_fresh_threads_on_a_raw_kernel_tree() {
    // Exhaustive and sampled measurements.
    for verification in [
        VerificationOptions::new(),
        VerificationOptions::new()
            .with_samples(8)
            .with_max_exhaustive_points(0),
    ] {
        let first = dense_run_on_a_fresh_thread(raw_kernel_tree, raw_kernel_switch, verification);
        let second = dense_run_on_a_fresh_thread(raw_kernel_tree, raw_kernel_switch, verification);
        assert!(first.report.splits >= 1);
        assert_same_runs_across_problems(&first, &second);
    }
}

#[test]
#[ignore = "open question 8 of docs/design/tree-patching-error-contract.md: the generic IdxTensor \
            path of TreeTNCachedEvaluator is not guaranteed reproducible across threads (omeco \
            breaks contraction-path cost ties in HashMap order); these rank-one patches happen \
            to agree, the TreeTCI patches of adaptive_l2_treetci do not"]
fn dense_engine_l2_runs_are_reproducible_on_fresh_threads_on_a_generic_path_tree() {
    for verification in [
        VerificationOptions::new(),
        VerificationOptions::new()
            .with_samples(8)
            .with_max_exhaustive_points(0),
    ] {
        let first =
            dense_run_on_a_fresh_thread(extended_quantics_tree, generic_path_switch, verification);
        let second =
            dense_run_on_a_fresh_thread(extended_quantics_tree, generic_path_switch, verification);
        assert_same_runs_across_problems(&first, &second);
    }
}

// ---------------------------------------------------------------------------
// Errors (test 17)
// ---------------------------------------------------------------------------

#[test]
fn placeholder_norms_fail_before_every_other_check() {
    let problem = branched();
    for norm in [ErrorNorm::MaxAbs, ErrorNorm::WeightedL2] {
        // Invalid tolerance, cap, and pivots as well: the norm comes first.
        let options = PatchedInterpolationOptions::new(0)
            .with_error_norm(norm)
            .with_tolerance(tol(-1.0));
        let bad_pivots = ColMajorArray::new(vec![0], vec![1, 1]).unwrap();
        match patched_interpolate(
            &DenseEngine::new(),
            problem.topology.clone(),
            problem.node_sites.clone(),
            bad_pivots,
            never,
            &options,
        ) {
            Err(error @ PatchedInterpolationError::UnsupportedNorm { .. }) => {
                assert!(matches!(
                    error,
                    PatchedInterpolationError::UnsupportedNorm { norm: n } if n == norm
                ));
                let text = error.to_string();
                assert!(text.contains("ErrorNorm::L2 (the default) or ErrorNorm::SampledMax"));
            }
            other => panic!("expected UnsupportedNorm, got {other:?}"),
        }
    }
}

#[test]
fn new_invalid_inputs_are_rejected_before_any_evaluation() {
    let problem = single_node(&[3, 2]);
    let base = l2_given(2, 1.0, 1e-6);
    let cases = [
        (
            base.clone().with_tolerance(ErrorTolerance {
                rtol: 1e-6,
                atol: -1.0,
            }),
            "tolerance.atol",
        ),
        (
            base.clone().with_tolerance(ErrorTolerance {
                rtol: 1e-6,
                atol: f64::NAN,
            }),
            "tolerance.atol",
        ),
        (
            base.clone().with_tolerance(ErrorTolerance {
                rtol: 1e-6,
                atol: f64::INFINITY,
            }),
            "tolerance.atol",
        ),
        (base.clone().with_tolerance(tol(f64::NAN)), "tolerance.rtol"),
        (
            base.clone()
                .with_error_norm(ErrorNorm::l2(L2Reference::Given(0.0))),
            "L2Reference::Given",
        ),
        (
            base.clone()
                .with_error_norm(ErrorNorm::l2(L2Reference::Given(-1.0))),
            "L2Reference::Given",
        ),
        (
            base.clone()
                .with_error_norm(ErrorNorm::l2(L2Reference::Given(f64::NAN))),
            "L2Reference::Given",
        ),
        (
            base.clone()
                .with_error_norm(ErrorNorm::l2(L2Reference::Given(f64::INFINITY))),
            "L2Reference::Given",
        ),
        (
            base.clone()
                .with_verification(VerificationOptions::new().with_samples(1)),
            "verification.samples",
        ),
        (
            base.clone()
                .with_verification(VerificationOptions::new().with_samples(0)),
            "verification.samples",
        ),
        // Verification options are validated under every norm.
        (
            sampled_max(2).with_verification(VerificationOptions::new().with_samples(1)),
            "verification.samples",
        ),
    ];
    for (options, needle) in cases {
        expect_invalid(
            patched_interpolate(
                &DenseEngine::new(),
                problem.topology.clone(),
                problem.node_sites.clone(),
                no_pivots(&problem),
                never,
                &options,
            ),
            needle,
        );
    }
}

#[test]
fn a_non_finite_network_value_is_an_engine_error() {
    let problem = single_node(&[2, 2]);
    let f = |_: &[usize]| 1.0;
    let engine = ScriptedEngine::new(vec![Step::converged(Network::NonFinite)]);
    let (projector, source) =
        expect_interpolation(run(&engine, &problem, &f, &[], &l2_given(2, 2.0, 1e-6)));
    assert!(projector.is_empty());
    assert!(matches!(source, InterpolationError::Engine { .. }));
    assert!(source.to_string().contains("non-finite"), "{source}");
}

#[test]
fn verification_failures_display_their_remedy() {
    let error = PatchedInterpolationError::VerificationFailed {
        projector: Projector::new(),
        measurement: tensor4all_partitionedtreetn::adaptive_interpolation::L2Measurement::clone(
            &run(
                &DenseEngine::new(),
                &single_node(&[2]),
                &|_: &[usize]| 1.0,
                &[],
                &PatchedInterpolationOptions::new(2),
            )
            .unwrap()
            .report
            .accepted[0]
                .acceptance
                .clone()
                .unwrap(),
        ),
    };
    let text = error.to_string();
    assert!(text.contains("list more sites in patch_order"));
    assert!(text.contains("raise rtol or atol"));
    assert!(text.contains("check the reference norm"));
}

// ---------------------------------------------------------------------------
// Complex scalars (test 18)
// ---------------------------------------------------------------------------

#[test]
fn complex_residual_magnitudes_are_measured_by_abs_val() {
    let problem = branched();
    let j0 = problem.site("j", 0);
    let pj = problem.position(&j0);
    let phases = [Complex64::new(0.6, 0.8), Complex64::new(-0.8, 0.6)];
    let f = move |p: &[usize]| phases[p[pj]] * product(p, p[pj]);
    let (_, norm) = dense_reference(&problem, &f);
    let options = l2_given(2, norm, 1e-10).with_patch_order(vec![j0]);
    let result = run(&DenseEngine::new(), &problem, &f, &[], &options).unwrap();
    assert_eq!(result.report.accepted.len(), 2);
    for patch in result.partition.values() {
        let data = patch.data();
        for name in data.node_names() {
            assert!(data
                .tensor(data.node_index(&name).unwrap())
                .unwrap()
                .is_c64());
        }
    }
    assert_certified_bound(&result, &problem, &f);

    // A complex constant network that is off by i: residual magnitude 1.
    let problem = single_node(&[2, 2, 2]);
    let g = |_: &[usize]| Complex64::new(1.0, 1.0);
    let engine = ScriptedEngine::new(vec![Step::converged(Network::Constant(1.0))]);
    let options = PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(1.0)))
        .with_tolerance(tol(1e-6))
        .with_verification(VerificationOptions::new().with_retries(0))
        .with_patch_order(vec![problem.site("only", 0)]);
    let error = run(&engine, &problem, &g, &[], &options).unwrap_err();
    let PatchedInterpolationError::VerificationFailed { measurement, .. } = error else {
        panic!("expected VerificationFailed, got {error:?}");
    };
    assert_eq!(measurement.rms, 1.0);
    assert_eq!(measurement.max_residual, 1.0);
}

// ---------------------------------------------------------------------------
// Review fixes: overflowing plans, the rerun list, the log-norm guard, and
// the L2 error branches
// ---------------------------------------------------------------------------

/// A product over 65 binary sites (2^65 points, more than usize holds) and
/// its exact L2 norm.
fn wide_product() -> (Problem, impl Fn(&[usize]) -> f64 + Sync + Copy, f64) {
    let problem = chain("s", 65, 2);
    let factor = |i: usize, x: usize| 1.0 + 0.1 * ((i % 3) * x) as f64;
    let f = move |p: &[usize]| {
        p.iter()
            .enumerate()
            .map(|(i, &x)| factor(i, x))
            .product::<f64>()
    };
    let norm = (0..65)
        .map(|i| (factor(i, 0).powi(2) + factor(i, 1).powi(2)).sqrt())
        .product::<f64>();
    (problem, f, norm)
}

#[test]
fn unbounded_exhaustive_limits_never_measure_an_overflowing_patch_exhaustively() {
    let (problem, f, norm) = wide_product();
    let pivots = [vec![0; 65], vec![1; 65]];
    let options = l2_given(2, norm, 1e-6)
        .with_verification(VerificationOptions::new().with_max_exhaustive_points(usize::MAX));
    let result = run(&FiberEngine, &problem, &f, &pivots, &options).unwrap();
    let record = &result.report.accepted[0];
    let measurement = record.acceptance.as_ref().unwrap();
    assert_eq!(measurement.method, MeasurementMethod::Sampled);
    assert_eq!(measurement.points, 64);
    assert_eq!(measurement.patch_points, 2f64.powi(65));
    assert!(record.audit.is_some());
    assert!(matches!(
        result.report.norm.l2_error().unwrap().global,
        GlobalL2Error::Audited { .. }
    ));

    // A sample count whose point list overflows usize is rejected before any
    // evaluation.
    let options = l2_given(2, norm, 1e-6)
        .with_verification(VerificationOptions::new().with_samples(usize::MAX));
    expect_invalid(
        patched_interpolate(
            &FiberEngine,
            problem.topology.clone(),
            problem.node_sites.clone(),
            problem.pivots(&pivots),
            never,
            &options,
        ),
        "verification.samples",
    );
}

/// A point list that fits a `Vec` by length but cannot be reserved is
/// `InvalidInput` naming the verification options, reported during the
/// measurement, after the root's candidates were evaluated. The scenario
/// needs a 64-bit address space: on narrower targets the list length already
/// exceeds `Vec` capacity and is rejected before any evaluation.
#[test]
#[cfg(target_pointer_width = "64")]
fn an_unreservable_measurement_point_list_is_invalid_input_after_evaluations() {
    // 2^50 points of 50 binary sites: the exhaustive list holds 50 * 2^50
    // entries (about 4.5e17 bytes). That fits a Vec's byte limit, but no
    // 64-bit address space (at most 2^57 bytes) can provide it, so the
    // fallible reservation fails instead of aborting.
    let problem = chain("s", 50, 2);
    let options = l2_given(2, 1.0, 1e-6)
        .with_verification(VerificationOptions::new().with_max_exhaustive_points(1usize << 50));
    let calls = std::sync::atomic::AtomicUsize::new(0);
    // f = 0: every candidate sample is zero, so the zero screen measures the
    // whole root exhaustively.
    let zero = |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(vec![0.0; batch.shape()[1]])
    };
    let message = expect_invalid(
        patched_interpolate(
            &FiberEngine,
            problem.topology.clone(),
            problem.node_sites.clone(),
            no_pivots(&problem),
            zero,
            &options,
        ),
        "verification point list",
    );
    assert!(
        message.contains("verification.max_exhaustive_points"),
        "{message:?}"
    );
    assert!(message.contains("could not reserve exhaustive point list"));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

/// `f = 1` plus spikes A (+9), B (+7), C (+5) on `single_node(&[4, 4])`.
fn three_spikes(p: &[usize]) -> f64 {
    match p {
        [1, 2] => 10.0,
        [3, 0] => 8.0,
        [0, 3] => 6.0,
        _ => 1.0,
    }
}

#[test]
fn a_base_candidate_among_the_worst_points_does_not_shorten_the_added_list() {
    let problem = single_node(&[4, 4]);
    let (_, norm) = dense_reference(&problem, &three_spikes);
    let constant = Step::converged(Network::Constant(1.0));
    let engine = ScriptedEngine::new(vec![
        constant.clone().with_pivots(vec![vec![2, 2], vec![1, 1]]),
        constant,
        Step::converged(Network::Exact),
    ]);
    // A is a user pivot, so a base candidate; max_bond_dim - 1 = 2.
    let options = l2_given(3, norm, 1e-6)
        .with_n_initial_pivots(1)
        .with_patch_order(vec![problem.site("only", 0)]);
    run(&engine, &problem, &three_spikes, &[vec![1, 2]], &options).unwrap();
    let calls = engine.calls();
    let base = calls[0].initial_pivots.clone();
    assert_eq!(base, [vec![1, 2]]);
    // Step 9: drop the base candidates, then truncate: B and C, not B and an
    // outcome pivot.
    let mut expected = base.clone();
    expected.extend([vec![3, 0], vec![0, 3]]);
    assert_eq!(calls[1].initial_pivots, expected);
}

/// Returns, for the problem `a(2) - b(2) - e()` with `e` site-free, the exact
/// network of `f(x, y) = (1 + x)(3 + y)` whose bond `b - e` has dimension
/// two: `b(y, k) = (3 + y) / 2` and `e = [1, 1]`.
struct WideLeafEngine;

impl tensor4all_treetn::interpolation::TreeInterpolator<f64> for WideLeafEngine {
    fn interpolate<V, F>(
        &self,
        problem: &tensor4all_treetn::interpolation::InterpolationProblem<V>,
        _evaluate: F,
    ) -> Result<tensor4all_treetn::interpolation::InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>>,
    {
        use tensor4all_core::{DynIndex, IdxTensor};
        let sites: Vec<(V, Vec<DynIndex>)> = problem
            .node_sites()
            .iter()
            .map(|(n, s)| (n.clone(), s.clone()))
            .collect();
        let (ab, be) = (DynIndex::new_dyn(1), DynIndex::new_dyn(2));
        let a =
            IdxTensor::from_dense(vec![sites[0].1[0].clone(), ab.clone()], vec![1.0, 2.0]).unwrap();
        let b = IdxTensor::from_dense(
            vec![sites[1].1[0].clone(), ab, be.clone()],
            vec![1.5, 2.0, 1.5, 2.0],
        )
        .unwrap();
        let e = IdxTensor::from_dense(vec![be], vec![1.0, 1.0]).unwrap();
        let names = sites.into_iter().map(|(n, _)| n).collect();
        Ok(tensor4all_treetn::interpolation::InterpolationOutcome {
            network: tensor4all_treetn::TreeTN::from_tensors(vec![a, b, e], names).unwrap(),
            termination: tensor4all_treetn::interpolation::InterpolationTermination::Converged,
            error_estimate: 0.0,
            max_sample_magnitude: 8.0,
            pivots: None,
        })
    }
}

#[test]
fn a_site_free_leaf_with_a_wide_bond_reports_the_dense_approximation_norm() {
    let problem = Problem::new(
        &[("a", &[2]), ("b", &[2]), ("e", &[])],
        &[("a", "b"), ("b", "e")],
    );
    let f = |p: &[usize]| ((1 + p[0]) * (3 + p[1])) as f64;
    let (_, norm) = dense_reference(&problem, &f);
    // ||f|| = sqrt((1 + 4) (9 + 16)).
    assert!((norm - 125.0f64.sqrt()).abs() < 1e-12);
    let result = run(
        &WideLeafEngine,
        &problem,
        &f,
        &[],
        &l2_given(3, norm, 1e-10),
    )
    .unwrap();
    let error = result.report.norm.l2_error().unwrap();
    // The patch is exact, so its TreeTN::log_norm must reproduce the dense
    // norm although the site-free leaf `e` has a bond of dimension two.
    let approximation_norm = error.approximation_norm().unwrap();
    assert!(
        (approximation_norm - norm).abs() < 1e-12 * norm,
        "approximation norm {approximation_norm}, dense {norm}"
    );
    assert!(matches!(
        error.global,
        GlobalL2Error::Certified {
            rms_error,
            rounding_allowance_rms: Some(_),
            rounding_limited: Some(false),
            relative_error_bound: Some(bound),
            ..
        } if rms_error == 0.0 && bound < 1e-12
    ));
}

/// Regression for the `tensor4all-treetn` site-free-leaf norm defect fixed
/// by #799: `TreeTN::log_norm` overestimated the norm when a site-free leaf
/// that is not the smallest node name had a bond of dimension two or more.
#[test]
fn treetn_log_norm_of_a_site_free_leaf_with_a_wide_bond() {
    use tensor4all_core::{DynIndex, IdxTensor};
    use tensor4all_treetn::TreeTN;
    // f(x, y) = a(x) sum_k b(y, k) e(k) = [1, 0]_x [3, 4]_y, norm 5.
    let (sa, sb) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let (ab, be) = (DynIndex::new_dyn(1), DynIndex::new_dyn(2));
    let a = IdxTensor::from_dense(vec![sa, ab.clone()], vec![1.0, 0.0]).unwrap();
    let b = IdxTensor::from_dense(vec![sb, ab, be.clone()], vec![3.0, 0.0, 0.0, 4.0]).unwrap();
    let e = IdxTensor::from_dense(vec![be], vec![1.0, 1.0]).unwrap();
    let mut network = TreeTN::<IdxTensor, String>::from_tensors(
        vec![a, b, e],
        vec!["a".to_string(), "b".to_string(), "e".to_string()],
    )
    .unwrap();
    let dense = network.contract_to_tensor().unwrap().norm().unwrap();
    assert!((dense - 5.0).abs() < 1e-12);
    let log_norm = network.log_norm().unwrap();
    assert!(
        (log_norm.exp() - 5.0).abs() < 1e-12,
        "log_norm.exp() = {}",
        log_norm.exp()
    );
}

/// An evaluator of `f` that fails on its call number `fail_on` and counts
/// its calls.
fn failing_on(
    f: fn(&[usize]) -> f64,
    n_sites: usize,
    fail_on: usize,
    calls: &std::sync::atomic::AtomicUsize,
) -> impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>> + Send + Sync + '_ {
    move |batch| {
        let call = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        anyhow::ensure!(call != fail_on, "user failure on call {call}");
        Ok(batch.data().chunks(n_sites).map(f).collect())
    }
}

/// Run a scenario without failure to count its evaluator calls, check that
/// the last call is the measurement under test, then fail exactly that call.
fn assert_last_call_failure_is_reported<E>(
    engine: impl Fn() -> E,
    problem: &Problem,
    f: fn(&[usize]) -> f64,
    options: &PatchedInterpolationOptions,
    expected_calls: usize,
) where
    E: tensor4all_treetn::interpolation::TreeInterpolator<f64> + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    run_with(
        &engine(),
        problem,
        failing_on(f, problem.sites.len(), usize::MAX, &calls),
        &[],
        options,
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
    let calls = AtomicUsize::new(0);
    let (projector, source) = expect_interpolation(run_with(
        &engine(),
        problem,
        failing_on(f, problem.sites.len(), expected_calls - 1, &calls),
        &[],
        options,
    ));
    assert!(projector.is_empty());
    match source {
        InterpolationError::Evaluator { source } => {
            assert!(format!("{source:#}").contains("user failure"), "{source:#}")
        }
        other => panic!("expected Evaluator, got {other:?}"),
    }
}

#[test]
fn evaluator_failures_in_l2_measurements_are_evaluator_errors() {
    let problem = branched();
    // Monte Carlo reference: the first call of the run.
    let monte_carlo = PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::MonteCarlo))
        .with_tolerance(tol(1e-6));
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let (_, source) = expect_interpolation(run_with(
        &DenseEngine::new(),
        &problem,
        failing_on(|p| product(p, 0), problem.sites.len(), 0, &calls),
        &[],
        &monte_carlo,
    ));
    assert!(matches!(source, InterpolationError::Evaluator { .. }));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Zero screen: candidates, then the exhaustive screen of the zero root.
    assert_last_call_failure_is_reported(
        DenseEngine::new,
        &problem,
        |_| 0.0,
        &l2_given(2, 1.0, 1e-6),
        2,
    );
    // Verification: candidates (re-read from the cache by the engine), then
    // the exhaustive measurement.
    assert_last_call_failure_is_reported(
        || ScriptedEngine::new(vec![Step::converged(Network::Constant(1.0))]),
        &problem,
        |_| 1.0,
        &l2_given(2, 96f64.sqrt(), 1e-6),
        2,
    );
    // Audit: candidates, the sampled verification, then the audit.
    let wide = chain("s", 12, 2);
    let sampled = l2_given(2, 64.0, 1e-6).with_verification(
        VerificationOptions::new()
            .with_samples(16)
            .with_max_exhaustive_points(0),
    );
    assert_last_call_failure_is_reported(
        || ScriptedEngine::new(vec![Step::converged(Network::Constant(1.0))]),
        &wide,
        |_| 1.0,
        &sampled,
        3,
    );
}

#[test]
fn malformed_outcome_pivots_fail_a_rerun_without_recycling() {
    let problem = single_node(&[4, 4]);
    let two = |_: &[usize]| 2.0;
    let engine = ScriptedEngine::new(vec![
        Step::converged(Network::Constant(1.0)).with_pivots(vec![vec![9, 9]])
    ]);
    let options = l2_given(2, 8.0, 1e-6);
    assert!(!options.recycle_pivots);
    assert!(options.verification.retries >= 1);
    let (projector, source) = expect_interpolation(run(&engine, &problem, &two, &[], &options));
    assert!(projector.is_empty());
    assert!(matches!(source, InterpolationError::Engine { .. }));
    assert!(source.to_string().contains("coordinate 9"), "{source}");
}

#[test]
fn a_capped_rerun_after_a_failed_verification_keeps_no_split_index_left() {
    let problem = single_node(&[2, 4, 4]);
    let p0 = problem.site("only", 0);
    let two = |_: &[usize]| 2.0;
    // Root: fails, rerun capped, split. Child 0: fails, rerun capped, and no
    // split site is left.
    let wrong = Step::converged(Network::Constant(1.0));
    let engine = ScriptedEngine::new(vec![wrong.clone(), Step::capped(), wrong, Step::capped()]);
    let options = l2_given(2, 2.0 * 32f64.sqrt(), 1e-6).with_patch_order(vec![p0.clone()]);
    match run(&engine, &problem, &two, &[], &options) {
        Err(PatchedInterpolationError::NoSplitIndexLeft { projector }) => {
            assert_eq!(projector, Projector::from_pairs([(p0, 0)]).unwrap());
        }
        other => panic!("expected NoSplitIndexLeft, got {other:?}"),
    }
    let calls = engine.calls();
    assert_eq!(calls.len(), 4);
}
