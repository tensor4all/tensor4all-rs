//! Frozen outputs of the M2 `patched_interpolate` driver (golden test).
//!
//! The scenarios below mirror the M2 test scenarios. Their outputs were
//! recorded on the unmodified M2 driver and committed in
//! `tests/golden/adaptive_m2.json`; since M3 they run under
//! `ErrorNorm::SampledMax`, which must reproduce them. Two classes:
//!
//! - **discrete outputs**, compared exactly: projector keys and their
//!   canonical order, zero projectors, split count, function evaluations and
//!   cache hits, terminations, per-patch bond dimensions, the ID-free leg
//!   layout of every stored node (sites by position, bonds by neighbor and
//!   dimension), and, for TreeTCI, the termination of every engine call;
//! - **floating outputs** (raw column-major node data in that leg order and
//!   the floating report values), compared within [`M2_GOLDEN_RTOL`] relative
//!   to each tensor's largest magnitude (each value's own magnitude for
//!   report values). Each stored node may be negated, because a factorization
//!   may move signs between nodes and their bond legs; the product of those
//!   per-node signs is compared per accepted patch, so a gauge that preserves
//!   the represented function passes and one that flips its sign does not.
//!
//! The discrete outputs rest on floating-point decisions, so the scenarios
//! keep their decisions away from the thresholds. The dense-engine scenarios
//! use dyadic functions: their values are exactly representable and their
//! ranks are exact. Every TreeTCI scenario runs through [`RecordingTreeTci`],
//! which records the tolerance, termination, and error estimate of every
//! engine call; a scenario is admitted only if every ratio of error estimate
//! to tolerance lies outside
//! `[1 / GOLDEN_DECISION_SEPARATION, GOLDEN_DECISION_SEPARATION]`. TreeTCI's
//! per-sweep history, rank truncations, and LU pivot choices cannot be
//! screened this way: a TreeTCI scenario whose discrete output differs on
//! another platform is evidence for open question 9 of
//! `docs/design/tree-patching-error-contract.md` and must not be silently
//! re-recorded.
//!
//! Re-recording is an explicit, reviewed step: run the ignored test
//! `record_m2_golden` with `T4A_RECORD_M2_GOLDEN=1`.

use crate::adaptive_common;

use std::sync::Mutex;

use adaptive_common::*;
use num_complex::Complex64;
use serde_json::{json, Value};
use tensor4all_core::{ColMajorArrayRef, CommonScalar, IndexLike, TensorElement};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    PatchedInterpolationOptions, PatchedInterpolationResult,
};
use tensor4all_partitionedtreetn::ErrorNorm;
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{
    EvalContext, InterpolationError, InterpolationOutcome, InterpolationProblem,
    InterpolationTermination, TreeInterpolator,
};

/// Committed golden outputs.
const GOLDEN_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/adaptive_m2.json");

/// Environment variable that allows `record_m2_golden` to overwrite
/// [`GOLDEN_PATH`].
const RECORD_ENV: &str = "T4A_RECORD_M2_GOLDEN";

/// Relative tolerance of the floating golden outputs: per tensor, relative to
/// its largest magnitude; per report value, relative to its magnitude.
const M2_GOLDEN_RTOL: f64 = 1e-10;

/// Minimum separation of every TreeTCI error estimate from its tolerance, as
/// a ratio in either direction.
const GOLDEN_DECISION_SEPARATION: f64 = 10.0;

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// A product of per-site factors with dyadic coefficients: every value is
/// exactly representable, so the ranks the dense engine sees are exact.
fn dyadic_product(point: &[usize], variant: usize) -> f64 {
    point
        .iter()
        .enumerate()
        .map(|(i, &x)| 1.0 + ((variant + 1) * (i + 2) * x) as f64 / 8.0 + (x * x) as f64 / 16.0)
        .product()
}

/// `f = g_{x_s}(x)` with dyadic product functions.
fn dyadic_switch_on(position: usize) -> impl Fn(&[usize]) -> f64 + Sync {
    move |point| dyadic_product(point, point[position])
}

type RealFn = Box<dyn Fn(&[usize]) -> f64 + Sync>;
type ComplexFn = Box<dyn Fn(&[usize]) -> Complex64 + Sync>;

/// The function of a scenario.
enum Function {
    Real(RealFn),
    Complex(ComplexFn),
}

/// The engine of a scenario.
#[derive(Clone, Copy)]
enum Engine {
    Dense(Fault),
    TreeTci,
}

/// One golden scenario.
struct Scenario {
    name: &'static str,
    problem: Problem,
    function: Function,
    pivots: Vec<Vec<usize>>,
    options: PatchedInterpolationOptions,
    engine: Engine,
}

/// M2 options (`ErrorNorm::SampledMax`) with a given max-norm reference
/// and relative tolerance.
fn options_with(cap: usize, rtol: f64, scale: f64) -> PatchedInterpolationOptions {
    sampled_max(cap)
        .with_tolerance(tol(rtol))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(scale))
}

fn dense_scenarios() -> Vec<Scenario> {
    let mut scenarios = Vec::new();

    let problem = branched();
    let j0 = problem.site("j", 0);
    scenarios.push(Scenario {
        name: "dense_split_at_junction",
        function: Function::Real(Box::new(dyadic_switch_on(problem.position(&j0)))),
        options: options_with(2, 1e-12, 10.0).with_patch_order(vec![j0]),
        problem,
        pivots: vec![],
        engine: Engine::Dense(Fault::None),
    });

    let problem = branched();
    let d1 = problem.site("d", 1);
    scenarios.push(Scenario {
        name: "dense_split_at_multi_site_node",
        function: Function::Real(Box::new(dyadic_switch_on(problem.position(&d1)))),
        options: options_with(2, 1e-12, 10.0).with_patch_order(vec![d1]),
        problem,
        pivots: vec![],
        engine: Engine::Dense(Fault::None),
    });

    let problem = branched();
    let (d0, d1) = (problem.site("d", 0), problem.site("d", 1));
    let (p0, p1) = (problem.position(&d0), problem.position(&d1));
    scenarios.push(Scenario {
        name: "dense_every_site_of_a_node_fixed",
        function: Function::Real(Box::new(move |p: &[usize]| {
            dyadic_product(p, 2 * p[p0] + p[p1])
        })),
        options: options_with(2, 1e-12, 10.0).with_patch_order(vec![d0, d1]),
        problem,
        pivots: vec![],
        engine: Engine::Dense(Fault::None),
    });

    scenarios.push(Scenario {
        name: "dense_canonical_order_chain",
        problem: chain3(),
        function: Function::Real(Box::new(|p: &[usize]| {
            if p[0] == 1 {
                dyadic_product(p, 0)
            } else {
                dyadic_product(p, p[1])
            }
        })),
        options: options_with(2, 1e-12, 10.0),
        pivots: vec![],
        engine: Engine::Dense(Fault::None),
    });

    let problem = branched();
    let (j0, a0) = (problem.site("j", 0), problem.site("a", 0));
    let (pj, pa) = (problem.position(&j0), problem.position(&a0));
    scenarios.push(Scenario {
        name: "dense_vanishing_region",
        function: Function::Real(Box::new(move |p: &[usize]| {
            if p[pj] == 0 {
                0.0
            } else {
                dyadic_product(p, p[pa])
            }
        })),
        options: options_with(2, 1e-12, 10.0).with_patch_order(vec![j0, a0]),
        problem,
        pivots: vec![],
        engine: Engine::Dense(Fault::None),
    });

    // The scale is pinned from the single root candidate f(1, 0) = 1.
    scenarios.push(Scenario {
        name: "dense_exact_one_site_patches",
        problem: Problem::new(&[("s", &[3]), ("t", &[8])], &[("s", "t")]),
        function: Function::Real(Box::new(|p: &[usize]| match p[0] {
            0 => 0.0,
            1 => 1.0 + p[1] as f64,
            _ if p[1] == 5 => -3.0,
            _ => 0.0,
        })),
        options: sampled_max(2).with_n_initial_pivots(1),
        pivots: vec![vec![1, 0]],
        engine: Engine::Dense(Fault::None),
    });

    scenarios.push(Scenario {
        name: "dense_recycled_pivots",
        problem: chain3(),
        function: Function::Real(Box::new(|p: &[usize]| dyadic_product(p, 2 * p[0] + p[1]))),
        options: options_with(2, 1e-12, 10.0)
            .with_n_initial_pivots(1)
            .with_seed(3)
            .with_recycle_pivots(true),
        pivots: vec![],
        engine: Engine::Dense(Fault::None),
    });

    let problem = branched();
    let (j0, a0) = (problem.site("j", 0), problem.site("a", 0));
    let (pj, pa) = (problem.position(&j0), problem.position(&a0));
    scenarios.push(Scenario {
        name: "dense_seeded_branched",
        function: Function::Real(Box::new(move |p: &[usize]| {
            dyadic_product(p, p[pj] + 2 * p[pa])
        })),
        options: options_with(2, 1e-12, 20.0)
            .with_seed(99)
            .with_patch_order(vec![j0, a0]),
        problem,
        pivots: vec![],
        engine: Engine::Dense(Fault::None),
    });

    let problem = branched();
    let j0 = problem.site("j", 0);
    let pj = problem.position(&j0);
    let phases = [Complex64::new(0.25, 0.5), Complex64::new(0.375, -0.125)];
    scenarios.push(Scenario {
        name: "dense_complex",
        function: Function::Complex(Box::new(move |p: &[usize]| {
            phases[p[pj]] * dyadic_product(p, p[pj])
        })),
        options: options_with(2, 1e-12, 10.0).with_patch_order(vec![j0]),
        problem,
        pivots: vec![],
        engine: Engine::Dense(Fault::None),
    });

    scenarios.push(Scenario {
        name: "dense_iteration_limit",
        problem: chain3(),
        function: Function::Real(Box::new(|p: &[usize]| dyadic_product(p, 0))),
        options: sampled_max(4).with_error_norm(ErrorNorm::sampled_max_with_reference(2.0)),
        pivots: vec![],
        engine: Engine::Dense(Fault::IterationLimit),
    });

    scenarios
}

fn treetci_scenarios() -> Vec<Scenario> {
    let mut scenarios = Vec::new();

    // The M2 chain scenario (two Gaussian peaks on a seven-bit quantics
    // chain) is not admitted: at every tried rtol from 1e-6 to 1e-12 some
    // TreeTCI call has an error-to-tolerance ratio between 0.1 and 10 (0.22
    // and 0.64 at the M2 rtol of 1e-8). See "Implementation decisions" in
    // docs/design/tree-patching-error-contract.md.
    for (name, recycle) in [
        ("treetci_quantics_tree", false),
        ("treetci_quantics_tree_recycled", true),
    ] {
        let problem = quantics_tree();
        let order = ["x0", "y0", "x1", "y1", "x2", "y2", "z"]
            .iter()
            .map(|node| problem.site(node, 0))
            .collect();
        scenarios.push(Scenario {
            name,
            options: options_with(4, 1e-8, max_abs(&problem, &tree_peak))
                .with_patch_order(order)
                .with_recycle_pivots(recycle)
                .with_seed(5),
            function: Function::Real(Box::new(tree_peak)),
            problem,
            pivots: vec![vec![1, 0, 0, 1, 0, 0, 1]],
            engine: Engine::TreeTci,
        });
    }
    scenarios
}

// ---------------------------------------------------------------------------
// The recording TreeTCI engine
// ---------------------------------------------------------------------------

/// One engine call: the absolute tolerance, the verdict, and the estimate.
#[derive(Clone, Debug)]
struct EngineCall {
    tolerance: f64,
    termination: InterpolationTermination,
    error_estimate: f64,
}

/// Delegates every call to [`TreeTciInterpolator`] and records the absolute
/// tolerance, the termination, and the error estimate of each call.
#[derive(Default)]
struct RecordingTreeTci {
    inner: TreeTciInterpolator,
    calls: Mutex<Vec<EngineCall>>,
}

impl TreeInterpolator<f64> for RecordingTreeTci {
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        ctx: &mut EvalContext<'_>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<f64>>,
    {
        let outcome = self.inner.interpolate(problem, ctx, evaluate)?;
        self.calls.lock().unwrap().push(EngineCall {
            tolerance: problem.absolute_tolerance(),
            termination: outcome.termination,
            error_estimate: outcome.error_estimate,
        });
        Ok(outcome)
    }
}

/// Ratio of the error estimate to the tolerance of one call.
fn decision_ratio(call: &EngineCall) -> f64 {
    if call.error_estimate == 0.0 {
        0.0
    } else {
        call.error_estimate / call.tolerance
    }
}

fn is_separated(call: &EngineCall) -> bool {
    let ratio = decision_ratio(call);
    !(1.0 / GOLDEN_DECISION_SEPARATION..=GOLDEN_DECISION_SEPARATION).contains(&ratio)
}

// ---------------------------------------------------------------------------
// Capturing a run
// ---------------------------------------------------------------------------

/// A sorted list of (site position, coordinate) pairs of a projector.
fn projector_key(
    projector: &tensor4all_partitionedtreetn::Projector,
    problem: &Problem,
) -> Vec<(usize, usize)> {
    let mut entries: Vec<(usize, usize)> = projector
        .iter()
        .map(|(site, &value)| (problem.position(site), value))
        .collect();
    entries.sort_unstable();
    entries
}

fn leg_label(leg: &Leg) -> String {
    match leg {
        Leg::Site(position) => format!("site:{position}"),
        Leg::Bond { to, dim } => format!("bond:{to}:{dim}"),
    }
}

/// Legs and flat data (real, or interleaved real and imaginary parts) of
/// every node of a stored patch, in node-name order.
fn capture_patch(
    result: &PatchedInterpolationResult<Name>,
    projector: &tensor4all_partitionedtreetn::Projector,
    problem: &Problem,
) -> (Vec<Value>, Vec<Value>) {
    let data = result.partition.get(projector).unwrap().data();
    let mut names = data.node_names();
    names.sort();
    let tensor_of = |name: &Name| data.tensor(data.node_index(name).unwrap()).unwrap();
    let mut layouts = Vec::new();
    let mut values = Vec::new();
    for name in &names {
        let tensor = tensor_of(name);
        let legs: Vec<String> = tensor
            .indices()
            .iter()
            .map(|index| {
                let leg = match problem.sites.iter().position(|s| s == index) {
                    Some(position) => Leg::Site(position),
                    None => Leg::Bond {
                        to: names
                            .iter()
                            .find(|other| {
                                *other != name && tensor_of(other).indices().contains(index)
                            })
                            .unwrap()
                            .clone(),
                        dim: index.dim(),
                    },
                };
                leg_label(&leg)
            })
            .collect();
        let (dtype, flat): (&str, Vec<f64>) = if tensor.is_f64() {
            ("f64", tensor.to_vec::<f64>().unwrap())
        } else {
            assert!(tensor.is_c64(), "node {name} is neither f64 nor c64");
            let complex = tensor.to_vec::<Complex64>().unwrap();
            ("c64", complex.iter().flat_map(|z| [z.re, z.im]).collect())
        };
        layouts.push(json!({"name": name, "legs": legs, "dtype": dtype}));
        values.push(json!(flat));
    }
    (layouts, values)
}

/// The golden record of one run.
fn capture(
    scenario: &Scenario,
    result: &PatchedInterpolationResult<Name>,
    calls: &[EngineCall],
) -> Value {
    let report = &result.report;
    let problem = &scenario.problem;
    let mut accepted_discrete = Vec::new();
    let mut accepted_floating = Vec::new();
    for record in &report.accepted {
        let (layouts, values) = capture_patch(result, &record.projector, problem);
        accepted_discrete.push(json!({
            "projector": projector_key(&record.projector, problem),
            "termination": format!("{:?}", record.termination),
            "max_bond_dim": record.max_bond_dim,
            "nodes": layouts,
        }));
        accepted_floating.push(json!({
            "error_estimate": record.engine_error_estimate,
            "max_sample_magnitude": record.max_sample_magnitude,
            "nodes": values,
        }));
    }
    let zero: Vec<Vec<(usize, usize)>> = zero_projectors(report)
        .iter()
        .map(|projector| projector_key(projector, problem))
        .collect();
    json!({
        "name": scenario.name,
        "discrete": {
            "splits": report.splits,
            "function_evaluations": report.function_evaluations,
            "cache_hits": report.cache_hits,
            "zero_projectors": zero,
            "accepted": accepted_discrete,
            "engine_terminations": calls
                .iter()
                .map(|call| format!("{:?}", call.termination))
                .collect::<Vec<_>>(),
        },
        "floating": {
            "reference_scale": max_reference(report),
            "accepted": accepted_floating,
            "engine_calls": calls
                .iter()
                .map(|call| json!({
                    "tolerance": call.tolerance,
                    "error_estimate": call.error_estimate,
                }))
                .collect::<Vec<_>>(),
        },
    })
}

fn run_typed<T, E>(
    engine: &E,
    scenario: &Scenario,
    f: &(dyn Fn(&[usize]) -> T + Sync),
) -> PatchedInterpolationResult<Name>
where
    T: CommonScalar + TensorElement,
    E: TreeInterpolator<T> + Sync,
{
    run(
        engine,
        &scenario.problem,
        f,
        &scenario.pivots,
        &scenario.options,
    )
    .unwrap_or_else(|error| panic!("scenario {} failed: {error}", scenario.name))
}

/// Run one scenario and capture its golden record; also return the engine
/// calls of a TreeTCI scenario.
fn run_scenario(scenario: &Scenario) -> (Value, Vec<EngineCall>) {
    match (scenario.engine, &scenario.function) {
        (Engine::Dense(fault), Function::Real(f)) => {
            let result = run_typed(&DenseEngine::with_fault(fault), scenario, f.as_ref());
            (capture(scenario, &result, &[]), Vec::new())
        }
        (Engine::Dense(fault), Function::Complex(f)) => {
            let result = run_typed(&DenseEngine::with_fault(fault), scenario, f.as_ref());
            (capture(scenario, &result, &[]), Vec::new())
        }
        (Engine::TreeTci, Function::Real(f)) => {
            let engine = RecordingTreeTci::default();
            let result = run_typed(&engine, scenario, f.as_ref());
            let calls = engine.calls.lock().unwrap().clone();
            (capture(scenario, &result, &calls), calls)
        }
        (Engine::TreeTci, Function::Complex(_)) => {
            panic!("scenario {}: no complex TreeTCI scenario", scenario.name)
        }
    }
}

fn all_scenarios() -> Vec<Scenario> {
    let mut scenarios = dense_scenarios();
    scenarios.extend(treetci_scenarios());
    scenarios
}

/// Panic unless every engine call of the scenario is well separated from
/// its threshold.
fn assert_screened(name: &str, calls: &[EngineCall]) {
    let close: Vec<(usize, f64)> = calls
        .iter()
        .enumerate()
        .filter(|(_, call)| !is_separated(call))
        .map(|(index, call)| (index, decision_ratio(call)))
        .collect();
    assert!(
        close.is_empty(),
        "scenario {name}: engine calls (index, error/tolerance) {close:?} lie within a factor \
         {GOLDEN_DECISION_SEPARATION} of their tolerance"
    );
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

fn as_floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry.as_f64().unwrap())
        .collect()
}

fn check_close_scalar(context: &str, expected: &Value, actual: &Value) -> Result<(), String> {
    let (expected, actual) = (expected.as_f64().unwrap(), actual.as_f64().unwrap());
    let scale = expected.abs().max(actual.abs());
    if (expected - actual).abs() <= M2_GOLDEN_RTOL * scale {
        Ok(())
    } else {
        Err(format!("{context}: expected {expected:e}, got {actual:e}"))
    }
}

/// Index of the entry that orients a stored node: its largest magnitude.
///
/// `None` for an all-zero node, whose gauge carries no information.
fn pivot_index(values: &[f64]) -> Option<usize> {
    let mut largest = 0.0_f64;
    let mut pivot = None;
    for (index, value) in values.iter().enumerate() {
        if value.abs() > largest {
            largest = value.abs();
            pivot = Some(index);
        }
    }
    pivot
}

/// Sign of `values[pivot]`, or `None` when the node is zero there.
fn pivot_sign(values: &[f64], pivot: usize) -> Option<f64> {
    let value = *values.get(pivot)?;
    if value < 0.0 {
        Some(-1.0)
    } else if value > 0.0 {
        Some(1.0)
    } else {
        None
    }
}

fn orient(values: &[f64], sign: f64) -> Vec<f64> {
    values.iter().map(|value| value * sign).collect()
}

/// Compare one stored node, and report the gauge sign each side contributed.
///
/// A tree network's stored data is defined only up to per-node signs, because a
/// factorization may move signs between a node and its bond legs without
/// changing the represented tensor. The `tenferro-rs` revision pinned here moved
/// exactly that convention: its coordinated strided-rs bump changed the
/// factorization's sign choices, so node data may be the exact negation of the
/// recorded one (`tensor4all-rs#930`, `tenferro-rs#2166`). Comparing raw node
/// data would therefore freeze a gauge rather than an output.
///
/// The node is oriented on the sign of its **largest-magnitude** entry, which is
/// the numerically robust choice: a tiny entry's sign can be noise. The pivot
/// index comes from the recorded node and orients both sides, so a rounding
/// difference inside a near-tie cannot move the pivot between them.
///
/// `Ok(None)` means the recorded node is identically zero. Its values are still
/// compared; the patch simply has no represented sign to compare, because a
/// contraction containing an all-zero node is the zero function.
///
/// # Errors
///
/// Returns the message naming the first difference: a length mismatch, or node
/// values that differ beyond [`M2_GOLDEN_RTOL`].
fn check_close_tensor(
    context: &str,
    expected: &Value,
    actual: &Value,
) -> Result<Option<(f64, f64)>, String> {
    let (expected_raw, actual_raw) = (as_floats(expected), as_floats(actual));
    if expected_raw.len() != actual_raw.len() {
        return Err(format!("{context}: length"));
    }
    // A recorded all-zero node has no gauge, so it contributes no sign; its
    // values are still compared, and the produced node must be zero too.
    let pivot = pivot_index(&expected_raw);
    let (expected_sign, actual_sign) = match pivot {
        Some(pivot) => (
            pivot_sign(&expected_raw, pivot).unwrap_or(1.0),
            pivot_sign(&actual_raw, pivot).unwrap_or(1.0),
        ),
        None => (1.0, 1.0),
    };
    let (expected, actual) = (
        orient(&expected_raw, expected_sign),
        orient(&actual_raw, actual_sign),
    );
    let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let difference = expected
        .iter()
        .zip(&actual)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    if difference <= M2_GOLDEN_RTOL * scale {
        Ok(pivot.map(|_| (expected_sign, actual_sign)))
    } else {
        Err(format!(
            "{context}: max difference {difference:e} exceeds {M2_GOLDEN_RTOL:e} * {scale:e}"
        ))
    }
}

/// Compare a recorded scenario against a produced one.
///
/// Each stored node may carry the sign gauge of the factorization, so nodes are
/// compared up to an exact negation; their product per accepted patch is what
/// keeps the represented function's sign (`tensor4all-rs#930`,
/// `tenferro-rs#2166`). A patch with a recorded all-zero node is the zero
/// function and has no sign to compare. The discrete outputs, the leg layout,
/// the split count and the gauge-invariant scalars remain exact comparisons.
///
/// # Errors
///
/// Returns the message naming the first difference. The golden test treats any
/// difference as fatal; the sign-gauge regression test asserts that a
/// compensated gauge passes and an uncompensated one does not.
fn compare_matches_golden(expected: &Value, actual: &Value) -> Result<(), String> {
    let name = expected["name"].as_str().unwrap();
    if expected["discrete"] != actual["discrete"] {
        return Err(format!("scenario {name}: discrete outputs differ"));
    }
    let (expected, actual) = (&expected["floating"], &actual["floating"]);
    check_close_scalar(
        &format!("{name}: reference_scale"),
        &expected["reference_scale"],
        &actual["reference_scale"],
    )?;
    let patches = expected["accepted"].as_array().unwrap();
    let actual_patches = actual["accepted"].as_array().unwrap();
    if patches.len() != actual_patches.len() {
        return Err(format!("{name}: accepted patch count"));
    }
    for (index, (e, a)) in patches.iter().zip(actual_patches).enumerate() {
        let context = format!("{name}: accepted patch {index}");
        check_close_scalar(
            &format!("{context} max_sample_magnitude"),
            &e["max_sample_magnitude"],
            &a["max_sample_magnitude"],
        )?;
        check_close_scalar(
            &format!("{context} error_estimate"),
            &e["error_estimate"],
            &a["error_estimate"],
        )?;
        let expected_nodes = e["nodes"].as_array().unwrap();
        let actual_nodes = a["nodes"].as_array().unwrap();
        if expected_nodes.len() != actual_nodes.len() {
            return Err(format!("{context}: node count"));
        }
        // The per-node gauge signs are taken at the recorded node's pivot, so
        // their product is the patch's represented sign: negating one node
        // negates every term of the contraction, and an odd number of extra
        // negations flips the represented function. A recorded all-zero node
        // makes the patch identically zero, where no sign is defined.
        let mut expected_sign = 1.0_f64;
        let mut actual_sign = 1.0_f64;
        let mut sign_is_defined = true;
        for (node, (e, a)) in expected_nodes.iter().zip(actual_nodes).enumerate() {
            match check_close_tensor(&format!("{context} node {node}"), e, a)? {
                Some((node_expected, node_actual)) => {
                    expected_sign *= node_expected;
                    actual_sign *= node_actual;
                }
                None => sign_is_defined = false,
            }
        }
        if sign_is_defined && expected_sign != actual_sign {
            return Err(format!(
                "{context}: the node sign gauge changes the represented function\
                 (recorded {expected_sign:+}, produced {actual_sign:+})"
            ));
        }
    }
    let engine_calls = expected["engine_calls"].as_array().unwrap();
    let actual_calls = actual["engine_calls"].as_array().unwrap();
    if engine_calls.len() != actual_calls.len() {
        return Err(format!("{name}: engine call count"));
    }
    for (index, (e, a)) in engine_calls.iter().zip(actual_calls).enumerate() {
        let context = format!("{name}: engine call {index}");
        check_close_scalar(
            &format!("{context} tolerance"),
            &e["tolerance"],
            &a["tolerance"],
        )?;
        check_close_scalar(
            &format!("{context} error_estimate"),
            &e["error_estimate"],
            &a["error_estimate"],
        )?;
    }
    Ok(())
}

fn assert_matches_golden(expected: &Value, actual: &Value) {
    if let Err(message) = compare_matches_golden(expected, actual) {
        panic!("{message}");
    }
}

fn load_golden() -> Vec<Value> {
    let text = std::fs::read_to_string(GOLDEN_PATH)
        .unwrap_or_else(|error| panic!("cannot read {GOLDEN_PATH}: {error}"));
    serde_json::from_str::<Value>(&text).unwrap()["scenarios"]
        .as_array()
        .unwrap()
        .clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn golden_trees_are_genuinely_branched() {
    for problem in [branched(), quantics_tree()] {
        let max_degree = problem
            .topology
            .graph()
            .node_indices()
            .map(|node| problem.topology.graph().neighbors(node).count())
            .max()
            .unwrap();
        assert!(max_degree >= 3);
    }
}

#[test]
fn m2_golden_outputs_are_reproduced() {
    let golden = load_golden();
    let scenarios = all_scenarios();
    assert_eq!(golden.len(), scenarios.len(), "scenario count");
    for (expected, scenario) in golden.iter().zip(&scenarios) {
        assert_eq!(expected["name"], scenario.name);
        let (actual, calls) = run_scenario(scenario);
        assert_screened(scenario.name, &calls);
        // Parse the actual record through the same text round trip as the
        // committed one, so that the JSON float parser cannot add a
        // difference of its own.
        let actual: Value = serde_json::from_str(&actual.to_string()).unwrap();
        assert_matches_golden(expected, &actual);
    }
}

/// The comparator keeps the represented function's sign: a per-node sign gauge
/// is accepted only when the product of the patch's per-node signs is unchanged
/// (`tensor4all-rs#930`, `tenferro-rs#2166`).
#[test]
fn golden_comparator_keeps_the_represented_function_sign() {
    let recorded = json!({
        "name": "sign-gauge",
        "discrete": {"node_count": 2, "bond_dim": 1},
        "floating": {
            "reference_scale": 6.0,
            "accepted": [{
                "max_sample_magnitude": 6.0,
                "error_estimate": 0.0,
                "nodes": [[2.0], [3.0]],
            }],
            "engine_calls": [],
        }
    });
    assert!(compare_matches_golden(&recorded, &recorded).is_ok());

    // One negated node of a two-node patch represents a different function
    // (2 * 3 != -2 * 3). Normalizing every node on its own would accept this.
    let mut single_flip = recorded.clone();
    single_flip["floating"]["accepted"][0]["nodes"][0][0] = json!(-2.0);
    assert!(compare_matches_golden(&recorded, &single_flip).is_err());

    // Negating both nodes preserves it: this is the gauge the pinned
    // tenferro-rs revision moved.
    let mut compensated = recorded.clone();
    compensated["floating"]["accepted"][0]["nodes"][0][0] = json!(-2.0);
    compensated["floating"]["accepted"][0]["nodes"][1][0] = json!(-3.0);
    assert!(compare_matches_golden(&recorded, &compensated).is_ok());

    // A rounding difference inside a near-tie must not move the pivot and
    // reject a node the tolerance accepts.
    let mut near_tie = recorded.clone();
    near_tie["floating"]["accepted"][0]["nodes"] = json!([[1.0, -1.0], [1.0]]);
    let mut near_tie_rounded = near_tie.clone();
    near_tie_rounded["floating"]["accepted"][0]["nodes"] = json!([[1.0 - 1e-12, -1.0], [1.0]]);
    assert!(compare_matches_golden(&near_tie, &near_tie_rounded).is_ok());

    // A patch with an all-zero node represents zero, so it has no sign to
    // compare, but its values are still compared.
    let mut zero = recorded.clone();
    zero["floating"]["accepted"][0]["nodes"] = json!([[0.0], [3.0]]);
    let mut zero_negated = zero.clone();
    zero_negated["floating"]["accepted"][0]["nodes"] = json!([[0.0], [-3.0]]);
    assert!(compare_matches_golden(&zero, &zero_negated).is_ok());
    let mut zero_replaced = zero.clone();
    zero_replaced["floating"]["accepted"][0]["nodes"] = json!([[1.0], [3.0]]);
    assert!(compare_matches_golden(&zero, &zero_replaced).is_err());

    // The discrete outputs stay exact, and a different node count is refused.
    let mut extra_node = recorded.clone();
    extra_node["floating"]["accepted"][0]["nodes"] = json!([[2.0], [3.0], [1.0]]);
    assert!(compare_matches_golden(&recorded, &extra_node).is_err());
}

/// Records the golden outputs. Run only on the reference implementation, and
/// review the diff: `T4A_RECORD_M2_GOLDEN=1 cargo test -p
/// tensor4all-partitionedtreetn --test integration -- --ignored adaptive_m2_golden::`.
#[test]
#[ignore = "re-records the committed golden outputs; run explicitly with T4A_RECORD_M2_GOLDEN=1"]
fn record_m2_golden() {
    assert!(
        std::env::var_os(RECORD_ENV).is_some(),
        "set {RECORD_ENV}=1 to overwrite {GOLDEN_PATH}"
    );
    let mut records = Vec::new();
    let mut screened = Vec::new();
    for scenario in all_scenarios() {
        let (record, calls) = run_scenario(&scenario);
        let ratios: Vec<f64> = calls.iter().map(decision_ratio).collect();
        eprintln!("{}: error/tolerance per call {ratios:?}", scenario.name);
        screened.push((scenario.name, calls));
        records.push(record);
    }
    for (name, calls) in &screened {
        assert_screened(name, calls);
    }
    let text = serde_json::to_string_pretty(&json!({
        "about": "Frozen M2 outputs of patched_interpolate; see tests/adaptive_m2_golden.rs",
        "scenarios": records,
    }))
    .unwrap();
    std::fs::create_dir_all(std::path::Path::new(GOLDEN_PATH).parent().unwrap()).unwrap();
    std::fs::write(GOLDEN_PATH, text + "\n").unwrap();
}
