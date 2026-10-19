//! Local fan-out of the patch waves through scheduler lanes.
//!
//! [`patched_interpolate_on_lanes`] is the lanes counterpart of
//! [`patched_interpolate`](super::patched_interpolate): the same validation,
//! the same per-patch body and the same wave merge, with each wave of patches
//! spread over the lanes of a caller-owned
//! [`TenferroLanes`](hataori_tenferro::TenferroLanes). Every patch runs in the
//! child session of the lane that pulled it, so the patch body - engine,
//! re-embedding, measurement, readout, norm contribution and publication -
//! stays on one execution owner, and no stage enters the process-global
//! runtime.
//!
//! # Which waves fan out
//!
//! The root wave (one patch) and every wave with fewer patches than the
//! executor has lanes run one patch at a time; a larger wave spreads over the
//! lanes, one patch per lane. One patch at a time is the shape that gives a
//! single item the backend's whole team: the lane's child session selects
//! tenferro's inner execution mode with the full domain budget, so the item's
//! own kernels may use every worker of the pool. Application code - the
//! evaluator - always stays on its lane thread, which
//! [`Parallelism::Serial`](tensor4all_treetn::interpolation::Parallelism) states.
//!
//! The threshold is the lane count, recorded as a smoke-level default: the
//! patch fixtures here are too small to time a crossing, so no measurement
//! backs it yet.
//!
//! ponytail: the design put the root and small waves on the held session with
//! [`Parallelism::Pool`](tensor4all_treetn::interpolation::Parallelism), which
//! needs a session accessor on the lane executor; `hataori-tenferro` at
//! `e105c490` exposes only [`Lanes`](hataori::Lanes), and tenferro admits one
//! held session per backend. Small waves therefore run on a lane, one patch at
//! a time: their tenferro operations still use the whole pool through the
//! child session's inner mode, while the caller's own evaluator code is
//! `Serial` and gets no pool budget.
//!
//! # Determinism
//!
//! The driver's own contract does not change: every patch derives its
//! sub-seeds from the seed and its path, patches are merged in canonical path
//! order, and the report is a function of the problem, not of the schedule.
//! Counters are per patch and summed, so they do not depend on the number of
//! lanes. The measured network values of the generic cached-evaluator path can
//! still differ at rounding level between threads near `tau` (see "Randomness
//! and determinism" in the documentation of the parent module): serial and
//! lane runs agree on fixtures whose measurements use the raw kernels, and no
//! bitwise cross-thread or cross-process claim is made for the generic path.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Mutex;

use hataori::{map_in_lanes, Lanes, MapInError};
use hataori_tenferro::TenferroLanes;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, TensorElement};
use tensor4all_tensorbackend::explicit::Session;
use tensor4all_treetn::interpolation::{EvalContext, Parallelism, TreeInterpolator};
use tensor4all_treetn::NodeNameNetwork;

use super::options::PatchedInterpolationOptions;
use super::report::PatchedInterpolationResult;
use super::{
    assemble, patch_outcome, run_waves, Patch, PatchRun, PatchedInterpolationError, Pins, Plan,
    References, Shared, WaveControl, WavePins, Waves,
};

/// One failing patch, with its typed detail kept aside.
///
/// `hataori::map_in_lanes` reports the lowest failing input index and a
/// message, so the callback stores the typed error under the index of the patch
/// that produced it and returns this placeholder; the dispatcher then returns
/// that patch's own error.
#[derive(Debug)]
struct ItemFailure(String);

impl std::fmt::Display for ItemFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// The lanes driver: one caller-owned lane executor runs every wave.
struct LanesWaves<'a, 'lane, F> {
    lanes: &'a mut TenferroLanes<'lane>,
    /// The caller's evaluator, shared by the lanes. It is `Fn + Sync` because
    /// several lanes call it at the same time.
    evaluate: F,
}

impl<T, V, E, F> Waves<T, V, E> for LanesWaves<'_, '_, F>
where
    T: CommonScalar + TensorElement + Send,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>> + Sync,
{
    fn run_wave(
        &mut self,
        plan: &Plan<'_, T, V, E>,
        patches: Vec<Patch<T>>,
        pins: WavePins<'_>,
    ) -> Result<Vec<PatchRun<T, V>>, PatchedInterpolationError> {
        let start = pins.start();
        let root_wave = pins.writable();
        let lane_count = self.lanes.lane_count().max(1);
        let wave: Vec<(usize, Patch<T>)> = patches.into_iter().enumerate().collect();
        if root_wave || wave.len() < lane_count {
            // One patch at a time: a small wave would otherwise occupy fewer
            // lanes than the pool has workers while each of its items may
            // already use the whole team.
            let mut runs = Vec::with_capacity(wave.len());
            for item in wave {
                runs.extend(map_wave(
                    self.lanes,
                    plan,
                    &self.evaluate,
                    vec![item],
                    start,
                    root_wave,
                )?);
            }
            return Ok(runs);
        }
        map_wave(self.lanes, plan, &self.evaluate, wave, start, root_wave)
    }

    fn agree(
        &mut self,
        control: WaveControl,
        references: References,
    ) -> Result<(WaveControl, References), PatchedInterpolationError> {
        Ok((control, references))
    }
}

/// Adaptively interpolate a function on a tree, fanning every wave of patches
/// out over scheduler lanes.
///
/// This is the local-fan-out counterpart of
/// [`patched_interpolate`](super::patched_interpolate): it takes the same
/// engine, problem, pivots, evaluator and options, and produces the same
/// result. Instead of running every patch on the caller's session, each wave
/// runs on the lanes of `lanes`: a lane is one worker of tenferro's pool
/// together with its child session, so a patch body runs in the execution that
/// owns it and no stage enters the process-global runtime.
///
/// # Arguments
///
/// * `lanes` - The caller's lane executor
///   ([`TenferroLanes::open`](hataori_tenferro::TenferroLanes::open) over a
///   backend the caller owns and closes). Its lane count is the fan-out width;
///   a one-lane executor runs every patch on that lane.
/// * `engine` - Tree interpolation engine run on every patch with at least
///   two active sites, for example `tensor4all_treetci::TreeTciInterpolator`.
/// * `topology` - Tree topology with named nodes.
/// * `node_sites` - Site indices of every node, possibly none for a node.
/// * `initial_pivots` - Column-major `[n_sites, n_pivots]` full-domain points.
/// * `evaluate` - Batch evaluator. It receives the full-domain points and the
///   executing lane's [`EvalContext`], whose session is that lane's child
///   session and whose parallelism is
///   [`Parallelism::Serial`](tensor4all_treetn::interpolation::Parallelism):
///   the evaluator runs on the lane's thread and may not fan out, while the
///   tenferro operations it issues through the session follow the lane's own
///   policy. It must be `Fn + Sync`, because several lanes call it at once.
/// * `options` - See [`PatchedInterpolationOptions`].
///
/// # Returns
///
/// The same [`PatchedInterpolationResult`] as the serial entry: the partition
/// of accepted patches and the report, both published as compact dense host
/// storage, so both stay usable after the lane executor closes.
///
/// # Errors
///
/// The same errors as the serial entry for the inputs, the engine, the
/// evaluator and the resource limits, plus
/// [`PatchedInterpolationError::FanOut`] when the lane executor itself fails to
/// run, join, or clean up a lane.
///
/// # Examples
///
/// Three binary sites, one per node of a chain, on two lanes. The cap of two
/// only accepts rank-one patches, so the domain splits twice; each patch is
/// evaluated and measured in the child session of a lane.
///
/// ```
/// use std::collections::BTreeMap;
/// use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex};
/// use tensor4all_partitionedtreetn::adaptive_interpolation::{
///     patched_interpolate_on_lanes, GlobalL2Error, PatchedInterpolationOptions,
/// };
/// use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference};
/// use tensor4all_treetci::TreeTciInterpolator;
/// use tensor4all_treetn::interpolation::EvalContext;
/// use tensor4all_treetn::NodeNameNetwork;
/// use hataori_tenferro::TenferroLanes;
/// use tenferro_cpu::CpuBackend;
///
/// let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
/// let mut topology = NodeNameNetwork::new();
/// for node in 0..3usize {
///     topology.add_node(node)?;
/// }
/// topology.add_edge(&0, &1)?;
/// topology.add_edge(&1, &2)?;
/// let node_sites: BTreeMap<usize, Vec<DynIndex>> =
///     (0..3).map(|node| (node, vec![sites[node].clone()])).collect();
/// // f(x) = 1 / (1 + x) with x = b0 + 2 b1 + 4 b2.
/// let norm = (0..8)
///     .map(|x| (1.0 / (1.0 + x as f64)).powi(2))
///     .sum::<f64>()
///     .sqrt();
/// let evaluate = |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| {
///     Ok::<Vec<f64>, anyhow::Error>(
///         batch
///             .data()
///             .chunks(3)
///             .map(|point| 1.0 / (1.0 + point.iter().sum::<usize>() as f64))
///             .collect(),
///     )
/// };
/// let options = PatchedInterpolationOptions::new(2)
///     .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
///     .with_tolerance(ErrorTolerance { rtol: 1e-12, atol: 0.0 });
/// let pivots = ColMajorArray::new(vec![0, 0, 0], vec![3, 1])?;
///
/// let backend = CpuBackend::with_threads(2)?;
/// let mut lanes = TenferroLanes::open(&backend)?;
/// let result = patched_interpolate_on_lanes(
///     &mut lanes,
///     &TreeTciInterpolator::default(),
///     topology,
///     node_sites,
///     pivots,
///     evaluate,
///     &options,
/// )?;
/// lanes.close()?;
///
/// assert_eq!(result.report.splits, 3);
/// assert_eq!(result.partition.len(), 4);
/// assert_eq!(result.report.function_evaluations, 8);
/// let error = result.report.norm.l2_error().unwrap();
/// assert!(matches!(
///     error.global,
///     GlobalL2Error::Certified { rms_error, .. } if rms_error == 0.0
/// ));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn patched_interpolate_on_lanes<T, V, E, F>(
    lanes: &mut TenferroLanes<'_>,
    engine: &E,
    topology: NodeNameNetwork<V>,
    node_sites: BTreeMap<V, Vec<DynIndex>>,
    initial_pivots: ColMajorArray<usize>,
    evaluate: F,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement + Send,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>> + Sync,
{
    let plan = Plan::validated(engine, topology, node_sites, initial_pivots, options)?;
    let mut waves = LanesWaves { lanes, evaluate };
    let merged = run_waves(&plan, &mut waves)?;
    assemble(&plan, merged)
}

/// Run one patch per lane and keep each patch's typed error.
///
/// The evaluator stays behind a shared reference: every lane enters its own
/// child session and calls it on its own thread.
fn map_wave<T, V, E, F>(
    lanes: &mut TenferroLanes<'_>,
    plan: &Plan<'_, T, V, E>,
    evaluate: &F,
    wave: Vec<(usize, Patch<T>)>,
    start: References,
    root_wave: bool,
) -> Result<Vec<PatchRun<T, V>>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement + Send,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>> + Sync,
{
    let errors: Mutex<Vec<Option<PatchedInterpolationError>>> =
        Mutex::new((0..wave.len()).map(|_| None).collect());
    let shared = Shared(evaluate);
    let outcome = map_in_lanes(lanes, wave, |(index, patch), lane| {
        // One patch's view of the references: a copy per patch, written back
        // through the outcome, so a wave merge sees the pins of the root wave
        // in canonical order.
        let mut references = start;
        let pins = if root_wave {
            Pins::Root(&mut references)
        } else {
            Pins::Later(&references)
        };
        // The bridge of #859: the lane's child session *is* this frontend's
        // session, borrowed for the patch.
        let mut ctx = EvalContext::new(
            Session::from_backend_session(lane.session()),
            Parallelism::Serial,
        );
        match patch_outcome(plan, pins, &shared, patch, &mut ctx) {
            Ok(run) => Ok(run),
            Err(error) => {
                let message = error.to_string();
                if let Ok(mut slots) = errors.lock()
                    && let Some(slot) = slots.get_mut(index)
                {
                    *slot = Some(error);
                }
                Err(ItemFailure(message))
            }
        }
    });
    match outcome {
        Ok(runs) => Ok(runs),
        Err(MapInError::Callback(error)) => {
            let stored = errors
                .into_inner()
                .unwrap_or_default()
                .into_iter()
                .flatten()
                .next();
            match stored {
                Some(error) => Err(error),
                None => Err(PatchedInterpolationError::FanOut {
                    message: format!("a lane callback failed without a recorded error: {error}"),
                }),
            }
        }
        Err(error) => Err(PatchedInterpolationError::FanOut {
            message: format!("the lane executor failed: {error}"),
        }),
    }
}
