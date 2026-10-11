//! Diagnostic measurements behind the "Evidence for the open questions" of
//! `docs/design/tree-patching-error-contract.md`. They are not contract
//! tests: each prints what it measures. Run one with
//!
//! ```text
//! cargo test --release -p tensor4all-partitionedtreetn --test adaptive_l2_diagnostics \
//!     -- --ignored --nocapture --test-threads 1 <name>
//! ```
//!
//! The numbers quoted in the record are evaluation and patch counts, which
//! do not depend on timing or thread settings.

mod adaptive_common;

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Mutex;

use adaptive_common::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    GlobalL2Error, MeasurementMethod, NormReport, PatchedInterpolationOptions,
    PatchedInterpolationReport, PatchedInterpolationResult, VerificationOptions,
};
use tensor4all_partitionedtreetn::{ErrorNorm, L2Reference};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationOutcome, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};

/// Bits per variable of the diagnostic tree.
const BITS: usize = 6;

/// Two quantics variables of `BITS` bits on the branches of a site-free
/// junction `r`, plus a binary flag `z`: degree three at `r`, 2^13 points.
/// Site order x0..x5, y0..y5, z.
fn quantics_2d() -> Problem {
    let mut nodes: Vec<(String, Vec<usize>)> = vec![("r".into(), vec![]), ("z".into(), vec![2])];
    let mut edges: Vec<(String, String)> = vec![("r".into(), "z".into())];
    for v in ["x", "y"] {
        for k in 0..BITS {
            nodes.push((format!("{v}{k}"), vec![2]));
            let parent = if k == 0 {
                "r".to_string()
            } else {
                format!("{v}{}", k - 1)
            };
            edges.push((parent, format!("{v}{k}")));
        }
    }
    let nodes: Vec<(&str, &[usize])> = nodes
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();
    let edges: Vec<(&str, &str)> = edges
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    Problem::new(&nodes, &edges)
}

/// Most significant bits first, interleaved, then the flag.
fn msb_order(problem: &Problem) -> Vec<DynIndex> {
    let mut order = Vec::new();
    for k in 0..BITS {
        order.push(problem.site(&format!("x{k}"), 0));
        order.push(problem.site(&format!("y{k}"), 0));
    }
    order.push(problem.site("z", 0));
    order
}

fn xyz(p: &[usize]) -> (f64, f64, f64) {
    (
        quantics(&p[0..BITS]),
        quantics(&p[BITS..2 * BITS]),
        p[2 * BITS] as f64,
    )
}

/// A smooth function, `rms / max = 0.66`.
fn smooth(p: &[usize]) -> f64 {
    let (x, y, z) = xyz(p);
    (1.0 + 0.5 * z) * (1.0 / (1.0 + 2.0 * (x + y).powi(2)) + (3.0 * x * y).sin())
}

/// A localized cusp, `rms / max = 0.040`.
fn localized(p: &[usize]) -> f64 {
    let (x, y, z) = xyz(p);
    let r = ((x - 0.3).powi(2) + (y - 0.6).powi(2)).sqrt();
    (1.0 + 0.5 * z) * (-r / 0.03).exp()
}

type TestFunction = fn(&[usize]) -> f64;

const FUNCTIONS: [(&str, TestFunction); 2] = [("smooth", smooth), ("localized", localized)];

fn options(
    cap: usize,
    rtol: f64,
    norm: ErrorNorm,
    problem: &Problem,
) -> PatchedInterpolationOptions {
    PatchedInterpolationOptions::new(cap)
        .with_error_norm(norm)
        .with_tolerance(tol(rtol))
        .with_patch_order(msb_order(problem))
        .with_seed(1)
}

fn label(report: &PatchedInterpolationReport) -> &'static str {
    match report.norm.l2_error().map(|e| &e.global) {
        Some(GlobalL2Error::Certified { .. }) => "certified",
        Some(GlobalL2Error::Audited { .. }) => "audited",
        Some(GlobalL2Error::AcceptanceOnly { .. }) => "acceptance-only",
        _ => "no L2 report",
    }
}

fn summary(
    name: &str,
    result: &PatchedInterpolationResult<Name>,
    reference: &IdxTensor,
    norm: f64,
) {
    let r = &result.report;
    let diff = dense_l2_residual(result, reference);
    eprintln!(
        "{name}: patches {} zero {} splits {} evals {} measurement {} audit {} failures {} \
         retries {} achieved E/||f|| {:.2e} ({})",
        r.accepted.len(),
        r.zero_patches.len(),
        r.splits,
        r.function_evaluations,
        r.measurement_evaluations,
        r.audit_evaluations,
        r.verification_failures,
        r.engine_retries,
        diff / norm,
        label(r)
    );
}

/// `tau / (rtol rms_P(f))` per accepted patch: the effective local relative
/// tolerance of the volume-proportional budget, in units of `rtol`.
fn local_tolerances(
    result: &PatchedInterpolationResult<Name>,
    problem: &Problem,
    f: fn(&[usize]) -> f64,
    rtol: f64,
) -> Vec<f64> {
    let NormReport::L2 { tau, .. } = result.report.norm else {
        return Vec::new();
    };
    let points = full_domain(&problem.dims());
    let mut ratios: Vec<f64> = result
        .report
        .accepted
        .iter()
        .map(|record| {
            let inside: Vec<f64> = points
                .iter()
                .filter(|p| {
                    record
                        .projector
                        .iter()
                        .all(|(s, &v)| p[problem.position(s)] == v)
                })
                .map(|p| f(p))
                .collect();
            let rms = (inside.iter().map(|v| v * v).sum::<f64>() / inside.len() as f64).sqrt();
            tau / (rtol * rms)
        })
        .collect();
    ratios.sort_by(f64::total_cmp);
    ratios
}

/// OQ2: the effective local relative tolerance of the volume-proportional
/// budget, and (as a separate diagnostic of the change of norm, at
/// unmatched accuracy) the same rtol under SampledMax.
#[test]
#[ignore = "diagnostic for open question 2; run in release"]
fn oq2_volume_budget_local_tolerances() {
    let problem = quantics_2d();
    for (name, f) in FUNCTIONS {
        let (reference, norm) = dense_reference(&problem, &f);
        let max_abs = max_abs(&problem, &f);
        for rtol in [1e-4, 1e-6] {
            let l2 = options(4, rtol, ErrorNorm::l2(L2Reference::Given(norm)), &problem);
            let result = run(&TreeTciInterpolator::default(), &problem, &f, &[], &l2).unwrap();
            summary(&format!("{name} L2 rtol {rtol}"), &result, &reference, norm);
            let local = local_tolerances(&result, &problem, f, rtol);
            eprintln!(
                "   tau / (rtol rms_P(f)): min {:.2e} median {:.2e} max {:.2e}; below 1: {} of {}",
                local[0],
                local[local.len() / 2],
                local[local.len() - 1],
                local.iter().filter(|&&t| t < 1.0).count(),
                local.len()
            );
            let max = options(
                4,
                rtol,
                ErrorNorm::sampled_max_with_reference(max_abs),
                &problem,
            );
            let result = run(&TreeTciInterpolator::default(), &problem, &f, &[], &max).unwrap();
            summary(
                &format!("{name} SampledMax rtol {rtol}"),
                &result,
                &reference,
                norm,
            );
        }
    }
}

/// OQ3: the spread of the Monte Carlo reference estimate (the driver's
/// estimator, `verification.samples = 64` uniform points) over 10000 draws.
#[test]
#[ignore = "diagnostic for open question 3"]
fn oq3_monte_carlo_reference_spread() {
    let problem = quantics_2d();
    let points = full_domain(&problem.dims());
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    for (name, f) in FUNCTIONS {
        let values: Vec<f64> = points.iter().map(|p| f(p)).collect();
        let rms = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt();
        let mut ratios: Vec<f64> = (0..10_000)
            .map(|_| {
                let mean = (0..64)
                    .map(|_| values[rng.random_range(0..values.len())].powi(2))
                    .sum::<f64>()
                    / 64.0;
                mean.sqrt() / rms
            })
            .collect();
        ratios.sort_by(f64::total_cmp);
        let q = |p: f64| ratios[((ratios.len() - 1) as f64 * p) as usize];
        let share = |pred: &dyn Fn(f64) -> bool| {
            ratios.iter().filter(|&&r| pred(r)).count() as f64 / ratios.len() as f64
        };
        eprintln!(
            "{name}: S_MC / S quantiles 1% {:.2} 10% {:.2} 50% {:.2} 90% {:.2} 99% {:.2}; \
             P(> 1.5) {:.3}, P(< 0.5) {:.3}",
            q(0.01),
            q(0.1),
            q(0.5),
            q(0.9),
            q(0.99),
            share(&|r| r > 1.5),
            share(&|r| r < 0.5)
        );
    }
}

/// Wraps TreeTCI and measures every `BondCapReached` outcome exhaustively on
/// its active domain against the engine tolerance (`tau` under L2).
#[derive(Default)]
struct CapProbe {
    inner: TreeTciInterpolator,
    capped: Mutex<Vec<f64>>,
}

impl TreeInterpolator<f64> for CapProbe {
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>>,
    {
        let outcome = self.inner.interpolate(problem, &evaluate)?;
        if outcome.termination == InterpolationTermination::BondCapReached {
            let sites = problem.site_order().to_vec();
            let dims: Vec<usize> = sites.iter().map(IndexLike::dim).collect();
            let points = full_domain(&dims);
            let flat = points.concat();
            let shape = [dims.len(), points.len()];
            let values = evaluate(ColMajorArrayRef::new(&flat, &shape).unwrap()).unwrap();
            let reference = IdxTensor::from_dense(sites, values).unwrap();
            let dense = outcome.network.contract_to_tensor().unwrap();
            let diff = dense.sub(&reference).unwrap().norm().unwrap();
            let rms = diff / (points.len() as f64).sqrt();
            self.capped
                .lock()
                .unwrap()
                .push(rms / problem.absolute_tolerance());
        }
        Ok(outcome)
    }
}

/// OQ4: how many capped outcomes would have fit their allowance.
#[test]
#[ignore = "diagnostic for open question 4; run in release"]
fn oq4_capped_outcomes_that_fit() {
    let problem = quantics_2d();
    for (name, f) in FUNCTIONS {
        let (_, norm) = dense_reference(&problem, &f);
        let l2 = options(4, 1e-4, ErrorNorm::l2(L2Reference::Given(norm)), &problem);
        let probe = CapProbe::default();
        let result = run(&probe, &problem, &f, &[], &l2).unwrap();
        let ratios = probe.capped.lock().unwrap().clone();
        eprintln!(
            "{name}: capped outcomes {}, of which fit their allowance {}; splits {}",
            ratios.len(),
            ratios.iter().filter(|&&r| r <= 1.0).count(),
            result.report.splits
        );
    }
}

/// OQ5: the evaluations an audit adds.
#[test]
#[ignore = "diagnostic for open question 5; run in release"]
fn oq5_audit_overhead() {
    let problem = quantics_2d();
    for (name, f) in FUNCTIONS {
        let (reference, norm) = dense_reference(&problem, &f);
        for audit in [true, false] {
            let l2 = options(6, 1e-4, ErrorNorm::l2(L2Reference::Given(norm)), &problem)
                .with_verification(VerificationOptions::new().with_audit(audit));
            let result = run(&TreeTciInterpolator::default(), &problem, &f, &[], &l2).unwrap();
            summary(&format!("{name} audit {audit}"), &result, &reference, norm);
            let sampled = result
                .report
                .accepted
                .iter()
                .filter(|r| r.acceptance.as_ref().unwrap().method == MeasurementMethod::Sampled)
                .count();
            eprintln!(
                "   sampled acceptances {sampled}; global {:?}",
                result.report.norm.l2_error().unwrap().global
            );
        }
    }
}

/// OQ8: two L2 runs with TreeTCI on a generic-path tree in fresh threads.
#[test]
#[ignore = "diagnostic for open question 8; run in release"]
fn oq8_generic_path_runs_across_threads() {
    let one = || {
        std::thread::spawn(|| {
            let problem = extended_quantics_tree();
            let (_, norm) = dense_reference(&problem, &extended_peak);
            let options = PatchedInterpolationOptions::new(3)
                .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
                .with_tolerance(tol(1e-6))
                .with_seed(11);
            let result = run(
                &TreeTciInterpolator::default(),
                &problem,
                &extended_peak,
                &[],
                &options,
            )
            .unwrap();
            (fingerprint(&result, &problem), result.report)
        })
        .join()
        .unwrap()
    };
    let (first_fingerprint, first) = one();
    let (second_fingerprint, second) = one();
    eprintln!(
        "patches {} vs {}, splits {} vs {}, same node data bits {}",
        first.accepted.len(),
        second.accepted.len(),
        first.splits,
        second.splits,
        first_fingerprint == second_fingerprint
    );
    let rms = |r: &PatchedInterpolationReport| -> Vec<f64> {
        r.accepted
            .iter()
            .map(|x| x.acceptance.as_ref().unwrap().rms)
            .collect()
    };
    let (a, b) = (rms(&first), rms(&second));
    let differing = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    eprintln!(
        "acceptance rms differs in {differing} of {} patches",
        a.len()
    );
}
