//! The lane fan-out ([`patched_interpolate_on_lanes`]) of the adaptive patch
//! driver: parity with the serial entry, session use inside a lane, the
//! partial-wave `max_patches` contract, and typed error propagation.
//!
//! The fixtures use [`DenseEngine`], whose measurement path is the cached
//! evaluator's raw kernels on trees with one site per node, so the serial and
//! lane runs are expected to agree exactly; the generic-path rounding caveat
//! of the driver documentation is not exercised here.

use crate::adaptive_common;

use adaptive_common::*;
use hataori_tenferro::TenferroLanes;
use std::sync::Mutex;
use tenferro_cpu::CpuEngine;
use tensor4all_core::{
    contract_pair_on_session, ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor,
    PairwiseContractionOptions, TensorElement,
};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate_on_lanes, PatchedInterpolationError, PatchedInterpolationOptions,
    PatchedInterpolationResult,
};
use tensor4all_partitionedtreetn::ErrorNorm;
use tensor4all_treetn::interpolation::{EvalContext, Parallelism, TreeInterpolator};

/// Run the lanes entry on `threads` lanes, each with its own serial engine.
fn run_on_lanes<T, E, F>(
    threads: usize,
    engine: &E,
    problem: &Problem,
    evaluate: F,
    pivots: &[Vec<usize>],
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<Name>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement + Send,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>> + Sync,
{
    let initial = if pivots.is_empty() {
        ColMajorArray::new(vec![], vec![problem.sites.len(), 0]).unwrap()
    } else {
        problem.pivots(pivots)
    };
    let engines = (0..threads).map(|_| CpuEngine::serial()).collect();
    let mut lanes = TenferroLanes::new(engines).expect("one engine per lane");
    assert_eq!(
        lanes.lane_count(),
        threads,
        "the executor reports its lanes"
    );
    patched_interpolate_on_lanes(
        &mut lanes,
        engine,
        problem.topology.clone(),
        problem.node_sites.clone(),
        initial,
        evaluate,
        options,
    )
}

/// The plain evaluator of a host function, callable from several lanes.
fn evaluator<T>(
    f: &(dyn Fn(&[usize]) -> T + Sync),
    n_sites: usize,
) -> impl Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>> + Sync + '_
where
    T: CommonScalar,
{
    move |batch, _ctx| Ok(batch.data().chunks(n_sites).map(f).collect())
}

/// The canonical form of the accepted projectors: the (position in the site
/// order, coordinate) pairs of every record, in report order. A projector's
/// own debug form is a map and has no defined order.
fn projectors(
    problem: &Problem,
    result: &PatchedInterpolationResult<Name>,
) -> Vec<Vec<(usize, usize)>> {
    result
        .report
        .accepted
        .iter()
        .map(|record| {
            let mut pairs: Vec<(usize, usize)> = record
                .projector
                .iter()
                .map(|(index, &coordinate)| (problem.position(index), coordinate))
                .collect();
            pairs.sort_unstable();
            pairs
        })
        .collect()
}

/// A three-node chain of one binary site per node; `f(x) = 1 / (1 + x)` with
/// `x = b0 + 2 b1 + 4 b2`.
fn chain() -> Problem {
    Problem::new(
        &[("a", &[2]), ("b", &[2]), ("c", &[2])],
        &[("a", "b"), ("b", "c")],
    )
}

fn chain_f(point: &[usize]) -> f64 {
    1.0 / (1.0 + point.iter().sum::<usize>() as f64)
}

/// The chain with a cap of two: only rank-one patches are accepted, so the
/// domain splits twice and every leaf patch is exact.
fn chain_options() -> PatchedInterpolationOptions {
    let norm = (0..8)
        .map(|x| (1.0 / (1.0 + x as f64)).powi(2))
        .sum::<f64>()
        .sqrt();
    PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(
            tensor4all_partitionedtreetn::L2Reference::Given(norm),
        ))
        .with_tolerance(tol(1e-12))
}

/// Two runs agree when every report field and every stored patch agree.
fn same_report(
    problem: &Problem,
    left: &PatchedInterpolationResult<Name>,
    right: &PatchedInterpolationResult<Name>,
) {
    let (left_report, right_report) = (&left.report, &right.report);
    assert_eq!(left_report.tolerance, right_report.tolerance);
    assert_eq!(
        format!("{:?}", left_report.norm),
        format!("{:?}", right_report.norm),
        "the norm reports differ"
    );
    assert_eq!(left_report.splits, right_report.splits);
    assert_eq!(
        left_report.function_evaluations,
        right_report.function_evaluations
    );
    assert_eq!(left_report.cache_hits, right_report.cache_hits);
    assert_eq!(
        left_report.measurement_evaluations,
        right_report.measurement_evaluations
    );
    assert_eq!(
        left_report.audit_evaluations,
        right_report.audit_evaluations
    );
    assert_eq!(
        left_report.verification_failures,
        right_report.verification_failures
    );
    assert_eq!(left_report.engine_retries, right_report.engine_retries);
    assert_eq!(
        format!("{:?}", left_report.zero_patches),
        format!("{:?}", right_report.zero_patches)
    );
    assert_eq!(projectors(problem, left), projectors(problem, right));
    assert_eq!(left.partition.len(), right.partition.len());
    assert_eq!(
        left_report
            .accepted
            .iter()
            .map(|record| (
                record.termination,
                record.status,
                record.max_bond_dim,
                record.retries_used,
                record.engine_error_estimate,
                record.max_sample_magnitude,
                record.acceptance.clone(),
                record.audit.clone(),
            ))
            .collect::<Vec<_>>(),
        right_report
            .accepted
            .iter()
            .map(|record| (
                record.termination,
                record.status,
                record.max_bond_dim,
                record.retries_used,
                record.engine_error_estimate,
                record.max_sample_magnitude,
                record.acceptance.clone(),
                record.audit.clone(),
            ))
            .collect::<Vec<_>>()
    );
    // The stored patch networks agree up to the round-off of two sessions.
    let left_dense = left
        .partition
        .to_treetn()
        .unwrap()
        .contract_to_tensor()
        .unwrap();
    let right_dense = right
        .partition
        .to_treetn()
        .unwrap()
        .contract_to_tensor()
        .unwrap();
    let difference = left_dense.sub(&right_dense).unwrap().maxabs().unwrap();
    assert!(
        difference <= 1e-12,
        "the stored patches differ by {difference:e}"
    );
}

#[test]
fn lanes_and_serial_produce_the_same_report() {
    let problem = chain();
    let options = chain_options();
    let serial = run(&DenseEngine::new(), &problem, &chain_f, &[], &options).expect("serial run");
    // Root, two children, four leaves: three splits and four leaf patches.
    assert_eq!(serial.report.splits, 3);
    assert_eq!(serial.partition.len(), 4);

    for threads in [1, 2] {
        let lanes = run_on_lanes(
            threads,
            &DenseEngine::new(),
            &problem,
            evaluator(&chain_f, problem.sites.len()),
            &[],
            &options,
        )
        .expect("lane run");
        same_report(&problem, &serial, &lanes);
    }
}

/// Every patch is evaluated, measured and published in its lane's child
/// session: the evaluator issues a real indexed contraction there, and the
/// context reports serial application parallelism.
#[test]
fn the_evaluator_computes_in_the_lane_session() {
    let problem = chain();
    let options = chain_options();
    let n_sites = problem.sites.len();
    let calls = Mutex::new(0usize);
    let scale = 2.0_f64;
    let evaluate = |batch: ColMajorArrayRef<'_, usize>, ctx: &mut EvalContext<'_>| {
        assert!(
            matches!(ctx.parallelism(), Parallelism::Serial),
            "application code in a lane is serial"
        );
        *calls.lock().unwrap() += 1;
        let values: Vec<f64> = batch.data().chunks(n_sites).map(chain_f).collect();
        // Contract the batch with a scalar on the lane's session: an indexed
        // session operation, not host arithmetic.
        let column =
            IdxTensor::from_dense_compact(vec![DynIndex::new_dyn(values.len())], values.clone())?;
        let scalar = IdxTensor::from_dense_compact(Vec::new(), vec![scale])?;
        let product = contract_pair_on_session(
            ctx.session(),
            &column,
            &scalar,
            PairwiseContractionOptions::new(),
        )?;
        let doubled = product.dense_host_values::<f64>()?;
        anyhow::ensure!(doubled.len() == values.len());
        for (left, right) in doubled.iter().zip(&values) {
            anyhow::ensure!((left - scale * right).abs() <= 1e-15 * right.abs().max(1.0));
        }
        Ok::<Vec<f64>, anyhow::Error>(values)
    };
    let lanes =
        run_on_lanes(2, &DenseEngine::new(), &problem, evaluate, &[], &options).expect("lane run");
    assert_eq!(lanes.report.splits, 3);
    assert!(*calls.lock().unwrap() > 0, "the evaluator ran");
    // The lane run still equals the run of the plain evaluator.
    let serial = run(&DenseEngine::new(), &problem, &chain_f, &[], &options).expect("serial run");
    assert_eq!(projectors(&problem, &lanes), projectors(&problem, &serial));
    assert_eq!(
        lanes.report.function_evaluations,
        serial.report.function_evaluations
    );
}

/// A root and three children (`f` depends on the three-valued site `a0`): the
/// first wave holds one patch, the second three.
fn wave_boundary_problem() -> (Problem, PatchedInterpolationOptions, usize) {
    let problem = branched();
    let a0 = problem.site("a", 0);
    let position = problem.position(&a0);
    let options = sampled_max(2)
        .with_error_norm(ErrorNorm::sampled_max_with_reference(10.0))
        .with_patch_order(vec![a0]);
    (problem, options, position)
}

#[test]
fn max_patches_checks_the_allowance_before_a_wave() {
    let (problem, options, position) = wave_boundary_problem();
    let f = switch_on(position);
    let n_sites = problem.sites.len();

    // The whole run is one root and three children.
    let complete = run(
        &DenseEngine::new(),
        &problem,
        &f,
        &[],
        &options.clone().with_max_patches(4),
    )
    .expect("four patches fit");
    assert_eq!(complete.partition.len(), 3);
    assert_eq!(complete.report.splits, 1);
    let serial_calls = complete.report.function_evaluations;

    // At and above the boundary: the root plus all three children fit.
    for limit in [4, 5] {
        assert!(
            run(
                &DenseEngine::new(),
                &problem,
                &f,
                &[],
                &options.clone().with_max_patches(limit)
            )
            .is_ok(),
            "a limit of {limit} patches fits"
        );
    }

    // Below the boundary the driver runs the remaining allowance of the wave
    // in canonical order and then reports the same error as a per-patch check:
    // the root and two of the three children, so three engine runs.
    let engine = DenseEngine::new();
    let recorder = Recorder::default();
    let recorded = recording_evaluator(&f, n_sites, &recorder);
    let serial_partial = run_with(
        &engine,
        &problem,
        recorded,
        &[],
        &options.clone().with_max_patches(3),
    );
    match serial_partial {
        Err(PatchedInterpolationError::ResourceLimit { resource, limit }) => {
            assert_eq!((resource, limit), ("max_patches", 3));
        }
        other => panic!("expected ResourceLimit, got {other:?}"),
    }
    assert_eq!(
        engine.seen().len(),
        3,
        "the root and two of the three children ran"
    );
    let partial_points = recorder.points();
    assert!(partial_points > 0);

    // The lanes entry dispatches the same partial wave, so it evaluates the
    // same points and reports the same error.
    let lane_recorder = Recorder::default();
    let lane_evaluate = recording_lane_evaluator(&f, n_sites, &lane_recorder);
    let lane_engine = DenseEngine::new();
    let result = run_on_lanes(
        2,
        &lane_engine,
        &problem,
        lane_evaluate,
        &[],
        &options.clone().with_max_patches(3),
    );
    match result {
        Err(PatchedInterpolationError::ResourceLimit { resource, limit }) => {
            assert_eq!((resource, limit), ("max_patches", 3));
        }
        other => panic!("expected ResourceLimit, got {other:?}"),
    }
    assert_eq!(
        lane_engine.seen().len(),
        3,
        "the lane run dispatches the same partial wave"
    );
    assert_eq!(
        lane_recorder.points(),
        partial_points,
        "the partial wave evaluates the same points"
    );

    // At the boundary the lanes run completes like the serial one.
    let lanes = run_on_lanes(
        2,
        &DenseEngine::new(),
        &problem,
        evaluator(&f, n_sites),
        &[],
        &options.with_max_patches(4),
    )
    .expect("four patches fit on lanes");
    assert_eq!(lanes.partition.len(), 3);
    assert_eq!(lanes.report.function_evaluations, serial_calls);
}

/// A shared recording evaluator for the lanes entry: the serial helper is
/// `FnMut`, and a fan-out calls one evaluator from several lanes.
fn recording_lane_evaluator<'a>(
    f: &'a (dyn Fn(&[usize]) -> f64 + Sync),
    n_sites: usize,
    recorder: &'a Recorder,
) -> impl Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<f64>> + Sync + 'a
{
    move |batch, _ctx| {
        recorder
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut seen = recorder.seen.lock().unwrap();
        Ok(batch
            .data()
            .chunks(n_sites)
            .map(|point| {
                assert!(
                    seen.insert(point.to_vec()),
                    "point {point:?} evaluated twice"
                );
                f(point)
            })
            .collect())
    }
}

/// A patch failure keeps its typed detail, and the lowest canonical index
/// wins when a whole wave fails.
#[test]
fn a_failing_patch_keeps_its_typed_error() {
    let (problem, options, position) = wave_boundary_problem();
    let n_sites = problem.sites.len();
    let f = switch_on(position);
    let evaluate = |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| {
        let points: Vec<f64> = batch.data().chunks(n_sites).map(&f).collect();
        Err::<Vec<f64>, anyhow::Error>(anyhow::anyhow!(
            "the evaluator refuses this batch of {}",
            points.len()
        ))
    };
    let result = run_on_lanes(2, &DenseEngine::new(), &problem, evaluate, &[], &options);
    match result {
        Err(PatchedInterpolationError::Interpolation {
            projector,
            source: tensor4all_treetn::interpolation::InterpolationError::Evaluator { .. },
        }) => {
            // The root wave fails first, so its projector is the empty one.
            assert!(projector.is_empty(), "unexpected projector {projector:?}");
        }
        other => panic!("expected an evaluator error, got {other:?}"),
    }
}

/// The advertised lane construction: split `allowed_cpus()` into disjoint
/// lists, build one pinned backend per list, consume each backend into its
/// engine, and run the adaptive driver on the resulting lanes.
#[cfg(target_os = "linux")]
#[test]
fn allowed_cpus_split_into_pinned_backends_builds_lanes_and_runs() {
    use tensor4all_tensorbackend::CpuBackend;

    let allowed = CpuBackend::allowed_cpus().expect("the Linux affinity snapshot");
    let groups: Vec<&[usize]> = if allowed.len() <= 1 {
        vec![&allowed[..]]
    } else {
        let split = allowed.len().div_ceil(2);
        allowed.chunks(split).collect()
    };
    let engines = groups
        .iter()
        .map(|cpus| {
            CpuBackend::pinned(cpus)
                .expect("a disjoint pinned backend")
                .into_engine()
        })
        .collect();
    let mut lanes = TenferroLanes::new(engines).expect("one engine per lane");
    assert_eq!(lanes.lane_count(), groups.len());

    let problem = chain();
    let options = chain_options();
    let result = patched_interpolate_on_lanes(
        &mut lanes,
        &DenseEngine::new(),
        problem.topology.clone(),
        problem.node_sites.clone(),
        problem.pivots(&[]),
        evaluator(&chain_f, problem.sites.len()),
        &options,
    )
    .expect("the pinned lanes run the adaptive driver");
    assert_eq!(result.partition.len(), 4);
    assert_eq!(result.report.splits, 3);
}
