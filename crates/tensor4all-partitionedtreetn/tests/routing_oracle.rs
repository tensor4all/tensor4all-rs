//! Routing oracle: the patched tree driver runs inside held-session phase
//! lanes while another thread holds the process-global backend.
//!
//! This is the tree counterpart of
//! `tensor4all-partitionedtt/tests/phase_execution.rs`'s
//! `the_phase_entry_completes_while_the_process_global_backend_is_held`. The
//! hataori scheduler arrives in a later slice; here the test drives the
//! driver's work items directly on `HeldSession::phase` lanes, one patch
//! pipeline per work item.
//!
//! The test is its own binary (see the `routing_oracle` test target) so the
//! process-global eager runtime is genuinely cold when the phase starts: a
//! lane that published or evaluated through the process-global route would
//! block on the holder's reservation, and the timing assertion fails instead
//! of hanging the rest of the suite.
//!
//! Observability of the split-forcing fixture: `max_bond_dim = 2` with a
//! rank-three root makes the root reach the cap and split, while a child has
//! rank two and is accepted after its measurement through
//! `CappedPatches::AcceptUpTo { bits: 2 }`. The accepted patches therefore
//! exercise re-embedding of the fixed site (`contract_pair_on_session`), the
//! session measurement and readout of the stored network, and the per-patch
//! norm readout (`log_norm_on_session`). The evaluator reduces every batch on
//! the lane's session through `sum_on_session`, so a run cannot pass without
//! the caller's execution capability.

use std::collections::BTreeMap;
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

use tenferro_cpu::CpuBackend;
use tensor4all_core::{
    sum_on_session, ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor,
    MatrixLuciScalar, TensorElement,
};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate, CappedPatches, MeasurementMethod, PatchedInterpolationError,
    PatchedInterpolationOptions, PatchedInterpolationResult,
};
use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference};
use tensor4all_tensorbackend::explicit::{HeldSession, Session};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_tensorbackend::{
    default_cpu_execution_context, with_default_backend, CpuExecutionContext,
};
use tensor4all_treetci::globalpivot::ScalarParts;
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{EvalContext, Parallelism};
use tensor4all_treetn::NodeNameNetwork;

/// Long enough to separate "finished while the reservation was held" from
/// "waited for the reservation", and short enough to fail rather than hang an
/// instrumented run.
const HOLD: Duration = Duration::from_secs(20);

/// A run must finish well inside [`HOLD`]; a process-global entry blocks for
/// the whole hold instead.
const BUDGET: Duration = Duration::from_secs(10);

type Name = String;

/// One work item's outcome, as collected from the lanes.
type LaneOutcome = (
    usize,
    Result<PatchedInterpolationResult<Name>, PatchedInterpolationError>,
);

/// Work items: one coefficient variant each, so lanes process distinct runs.
const VARIANTS: [usize; 4] = [0, 1, 2, 3];

fn node(index: usize) -> Name {
    format!("n{index}")
}

/// The split-forcing fixture: `f(i, j, k) = (1 + [i = j]) (1 + (1 + variant) j k)`.
///
/// `1 + [i = j]` is the full-rank matrix `J + I`, so the root has rank three
/// across the `n0` edge (above the cap of two, so it splits), and a child has
/// rank two across the `n1` edge (at the cap, accepted after its measurement
/// through `CappedPatches::AcceptUpTo { bits: 2 }`).
fn value<T: CommonScalar>(variant: usize, point: &[usize]) -> T {
    let (i, j, k) = (point[0], point[1], point[2]);
    let full_rank = 1 + usize::from(i == j);
    let rank_two = 1 + (1 + variant) * j * k;
    T::from_f64((full_rank * rank_two) as f64)
}

/// The exact L2 norm of the fixture, computed on host values only.
fn reference_norm(variant: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let value: f64 = value(variant, &[i, j, k]);
                sum += value * value;
            }
        }
    }
    sum.sqrt()
}

/// One driver run of one work item on `ctx`.
///
/// Every stage is reached: the root splits, the children are accepted from the
/// engine after re-embedding and measurement, and the report readout takes the
/// session norm. The evaluator issues a real reduction on the caller's
/// session.
fn run_item<T>(
    variant: usize,
    rtol: f64,
    ctx: &mut EvalContext<'_>,
) -> Result<PatchedInterpolationResult<Name>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement + FullPivLuScalar + MatrixLuciScalar + ScalarParts,
{
    let mut topology = NodeNameNetwork::new();
    for index in 0..3 {
        topology.add_node(node(index)).unwrap();
    }
    topology.add_edge(&node(0), &node(1)).unwrap();
    topology.add_edge(&node(1), &node(2)).unwrap();
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(3)).collect();
    let node_sites: BTreeMap<Name, Vec<DynIndex>> = (0..3)
        .map(|index| (node(index), vec![sites[index].clone()]))
        .collect();
    let n_sites = sites.len();
    let options = PatchedInterpolationOptions::new(2)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(reference_norm(variant))))
        .with_tolerance(ErrorTolerance { rtol, atol: 0.0 })
        .with_n_initial_pivots(2)
        .with_capped_patches(CappedPatches::AcceptUpTo { bits: 2 })
        .with_seed(17);
    let evaluate = |batch: ColMajorArrayRef<'_, usize>,
                    ctx: &mut EvalContext<'_>|
     -> anyhow::Result<Vec<T>> {
        let values: Vec<T> = batch
            .data()
            .chunks(n_sites)
            .map(|point| value::<T>(variant, point))
            .collect();
        // Exercise the caller's capability: the batch is reduced on its
        // session, not on the process-global route.
        let host =
            IdxTensor::from_dense_compact(vec![DynIndex::new_dyn(values.len())], values.clone())?;
        let total = sum_on_session(ctx.session(), &host)?;
        anyhow::ensure!(total.real().is_finite(), "the batch sum is not finite");
        Ok(values)
    };
    // Two nonzero pivots in site order [n0, n1, n2].
    let pivots = ColMajorArray::new(vec![0, 0, 0, 1, 1, 1], vec![3, 2]).unwrap();
    patched_interpolate(
        &TreeTciInterpolator::default(),
        topology,
        node_sites,
        pivots,
        ctx,
        evaluate,
        &options,
    )
}

/// A positional, ID-free summary of one run for comparing runs.
#[derive(Debug, PartialEq)]
struct Digest {
    splits: usize,
    accepted: usize,
    zero_patches: usize,
    function_evaluations: usize,
    measurement_evaluations: usize,
    patches: Vec<(usize, usize, MeasurementMethod)>,
    /// Dense values of every accepted patch, in report order.
    // ponytail: per-patch materialization instead of `PartitionedTreeTN::to_treetn`,
    // whose direct sum serves only `f64`/`c64`; that is a core limitation, not
    // this oracle's, and the per-patch numbers still compare exactly what the
    // session published.
    values: Vec<Vec<f64>>,
}

fn digest<T>(result: &PatchedInterpolationResult<Name>) -> Digest
where
    T: CommonScalar + TensorElement,
{
    let patches = result
        .report
        .accepted
        .iter()
        .map(|record| {
            let acceptance = record
                .acceptance
                .as_ref()
                .expect("an L2 run measures every accepted patch");
            (record.max_bond_dim, acceptance.points, acceptance.method)
        })
        .collect();
    let values = result
        .report
        .accepted
        .iter()
        .map(|record| {
            let subdomain = result
                .partition
                .get(&record.projector)
                .expect("the partition holds every accepted projector");
            subdomain
                .data()
                .contract_to_tensor()
                .expect("patch materializes")
                .to_vec::<T>()
                .expect("dense values")
                .iter()
                .map(|value| value.abs_val())
                .collect()
        })
        .collect();
    Digest {
        splits: result.report.splits,
        accepted: result.report.accepted.len(),
        zero_patches: result.report.zero_patches.len(),
        function_evaluations: result.report.function_evaluations,
        measurement_evaluations: result.report.measurement_evaluations,
        patches,
        values,
    }
}

/// Run one work item serially on a caller-owned context, the baseline.
fn run_serial<T>(variant: usize, rtol: f64) -> Digest
where
    T: CommonScalar + TensorElement + FullPivLuScalar + MatrixLuciScalar + ScalarParts,
{
    let context =
        CpuExecutionContext::from_backend(CpuBackend::with_threads(1).expect("1T CPU backend"));
    let backend: CpuBackend = context.with_backend(|backend| backend.clone());
    let mut held = HeldSession::open(&backend).expect("held session");
    let result = held.with_session(|view| {
        let mut ctx = EvalContext::new(
            Session::from_backend_session(view.backend()),
            Parallelism::Serial,
        );
        run_item::<T>(variant, rtol, &mut ctx)
    });
    held.close().expect("affinity restores");
    digest::<T>(&result.expect("serial run"))
}

fn assert_same(label: &str, serial: &Digest, phase: &Digest, tolerance: f64) {
    assert_eq!(
        phase.splits, serial.splits,
        "{label}: splits: phase {phase:?} serial {serial:?}"
    );
    assert_eq!(phase.accepted, serial.accepted, "{label}: accepted patches");
    assert_eq!(
        phase.zero_patches, serial.zero_patches,
        "{label}: zero patches"
    );
    assert_eq!(
        phase.function_evaluations, serial.function_evaluations,
        "{label}: function evaluations"
    );
    assert_eq!(
        phase.measurement_evaluations, serial.measurement_evaluations,
        "{label}: measurement evaluations"
    );
    assert_eq!(phase.patches, serial.patches, "{label}: patch shape");
    assert_eq!(phase.values.len(), serial.values.len(), "{label}: patches");
    for (patch, (phase, serial)) in phase.values.iter().zip(&serial.values).enumerate() {
        assert_eq!(phase.len(), serial.len(), "{label}: patch {patch} size");
        for (index, (phase, serial)) in phase.iter().zip(serial).enumerate() {
            let scale = serial.abs().max(1.0);
            assert!(
                (phase - serial).abs() <= tolerance * scale,
                "{label}: patch {patch} value {index}: phase {phase} serial {serial}"
            );
        }
    }
}

/// The phase itself: the fixture's work items on the lane session, while the
/// holder owns the process-global backend.
fn run_phase<T>(rtol: f64) -> Vec<Digest>
where
    T: CommonScalar
        + TensorElement
        + FullPivLuScalar
        + MatrixLuciScalar
        + ScalarParts
        + Send
        + Sync
        + 'static,
{
    let context =
        CpuExecutionContext::from_backend(CpuBackend::with_threads(2).expect("2T CPU backend"));
    let backend: CpuBackend = context.with_backend(|backend| backend.clone());
    let mut held = HeldSession::open(&backend).expect("held session");

    // Holder-ready handshake: the phase starts only once the holder has the
    // process-global backend, so a lane that entered it would block.
    let (ready_tx, ready_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        with_default_backend(|_global| {
            ready_tx.send(()).expect("signal the hold");
            let _ = release_rx.recv_timeout(HOLD);
        });
    });
    ready_rx.recv().expect("the holder is ready");

    let queue: Mutex<Vec<usize>> = Mutex::new(VARIANTS.iter().rev().copied().collect());
    let results: Mutex<Vec<LaneOutcome>> = Mutex::new(Vec::new());
    let started = Instant::now();
    held.phase(|_lane, lane| -> Result<(), String> {
        loop {
            if lane.cancelled() {
                break;
            }
            let Some(variant) = queue.lock().expect("queue lock").pop() else {
                break;
            };
            let mut ctx = EvalContext::new(lane.session(), Parallelism::Serial);
            let outcome = run_item::<T>(variant, rtol, &mut ctx);
            results
                .lock()
                .expect("results lock")
                .push((variant, outcome));
        }
        Ok(())
    })
    .map_err(|error| format!("phase failed: {error:?}"))
    .expect("the phase completes while the process-global backend is held");
    let elapsed = started.elapsed();
    // Release the holder before touching any process-global route below.
    let _ = release_tx.send(());
    holder.join().expect("holder thread");
    held.close().expect("affinity restores");

    assert!(
        elapsed < BUDGET,
        "the phase took {elapsed:?} while the process-global backend was held for up to {HOLD:?}: \
         a pipeline stage still consulted the process-global route"
    );

    let mut results = results.into_inner().expect("results");
    results.sort_by_key(|(variant, _)| *variant);
    assert_eq!(results.len(), VARIANTS.len(), "every work item ran");
    results
        .into_iter()
        .map(|(variant, outcome)| {
            let result = outcome.unwrap_or_else(|error| panic!("variant {variant}: {error}"));
            assert!(result.report.splits >= 1, "variant {variant} must split");
            assert!(
                result
                    .report
                    .accepted
                    .iter()
                    .any(|record| record.max_bond_dim >= 2),
                "variant {variant}: an engine patch must be accepted after re-embedding"
            );
            digest::<T>(&result)
        })
        .collect()
}

fn run_case<T>(label: &str, rtol: f64, tolerance: f64)
where
    T: CommonScalar
        + TensorElement
        + FullPivLuScalar
        + MatrixLuciScalar
        + ScalarParts
        + Send
        + Sync
        + 'static,
{
    let phase = run_phase::<T>(rtol);
    for (variant, phase) in VARIANTS.iter().zip(&phase) {
        let serial = run_serial::<T>(*variant, rtol);
        assert_same(
            &format!("{label} variant {variant}"),
            &serial,
            phase,
            tolerance,
        );
    }
}

/// The process-global eager runtime must not have been initialized: a cold
/// run proves the pipeline does not depend on a warm-up.
fn assert_cold_default_eager_runtime() {
    let context = default_cpu_execution_context();
    let debug = format!("{context:?}");
    assert!(
        debug.contains("eager_initialized: false"),
        "the process-global eager runtime must be cold at this point: {debug}"
    );
}

#[test]
fn the_patch_pipeline_completes_in_phase_lanes_while_the_process_global_backend_is_held() {
    // The first case is the cold-runtime oracle; it also warms the
    // process-global eager runtime for the materialization of the serial
    // baseline, so this is the only case that can assert coldness.
    assert_cold_default_eager_runtime();
    run_case::<f64>("f64", 1e-10, 1e-9);
    // f32 runs the same fixture with a precision-appropriate tolerance; the
    // oracle (completion while held) is unchanged, the runtime is now warm.
    run_case::<f32>("f32", 1e-4, 1e-4);
}
