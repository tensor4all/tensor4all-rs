//! MPI fan-out of the patch waves.
//!
//! [`patched_interpolate_mpi`] runs the same wave loop as the serial and the
//! lanes entry, with the root orchestrating waves and every rank executing the
//! patches it is given through [`hataori::pmap_lanes`]. A patch travels as a
//! wire record of host data only - fixed coordinates, its evaluation cache with
//! the packing that cache was split with, the pivots and worst points it
//! inherits - and its outcome travels back as host arrays plus the metadata a
//! root needs to rebuild the same patch: the accepted network's node positions,
//! site positions in the root's site order, patch-local bond axes, and the
//! measurements, status, retry diagnostics, patch volume and norm contribution
//! of the patch.
//!
//! Every rank runs its own lane executor over its own backend, and the MPI
//! progress loop stays on the initialization thread, which is what
//! [`hataori::pmap_lanes`] requires.
//!
//! # Collective structure
//!
//! 1. Each rank validates the problem locally; the rank-local messages are
//!    gathered on the root and the root's control is broadcast, so a rank with
//!    invalid input fails the whole run instead of diverging.
//! 2. Each wave is one `pmap_lanes` call: the root supplies its canonical wave,
//!    every other rank supplies nothing, and the root receives the ordered
//!    outcomes.
//! 3. The root merges the outcomes in canonical path order and broadcasts the
//!    driver's control with the pinned references before the next wave, so the
//!    root's pins reach every rank.
//! 4. When the queue is empty the root assembles the partition and the report
//!    (arithmetic only: every patch measured its own norm contribution on the
//!    session that built it) and every other rank returns `None`.
//!
//! Root-only work converges before it returns. Decoding a wave's wire outcomes
//! and assembling the final result happen only on the root, so their failure is
//! broadcast as a failing control - exactly like the wave control - rather than
//! returning early: a peer must not be left waiting in the next broadcast, and
//! it must not report success for a run the root could not assemble.
//!
//! The root wave is one patch, so it is one `pmap_lanes` item: it runs on one
//! rank's lane, and the references it pins travel back to the root in its
//! outcome.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::Arc;

use hataori::{broadcast, gather, pmap_lanes, PmapOptions};
use hataori_tenferro::TenferroLanes;
use num_complex::{Complex32, Complex64};
use serde::{Deserialize, Serialize};
use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, TensorElement,
};
use tensor4all_tensorbackend::Storage;
use tensor4all_treetn::interpolation::{
    EvalContext, InterpolationTermination, Parallelism, TreeInterpolator,
};
use tensor4all_treetn::NodeNameNetwork;

use super::cache::CacheLayout;
use super::layout::SiteLayout;
use super::options::PatchedInterpolationOptions;
use super::report::{
    L2Measurement, L2ReferenceSource, MaxReferenceSource, MeasurementMethod, PatchRecord,
    PatchStatus, PatchedInterpolationResult, ZeroPatchRecord,
};
use super::{
    assemble, run_waves, Accepted, L2Pin, Patch, PatchRun, PatchedInterpolationError, Pins, Plan,
    References, Shared, Totals, Verdict, WaveControl, WavePins, Waves,
};
use crate::subdomain_tree_tn::SubDomainTreeTN;
use crate::{PartitionedTreeTNError, Projector};

#[cfg(all(feature = "hataori-mpi", feature = "hataori-mpi-rsmpi-rt"))]
compile_error!(
    "features `hataori-mpi` and `hataori-mpi-rsmpi-rt` select two mutually exclusive hataori MPI \
     backends"
);

#[cfg(all(feature = "hataori-mpi", not(feature = "hataori-mpi-rsmpi-rt")))]
use mpi as mpi_api;
#[cfg(feature = "hataori-mpi-rsmpi-rt")]
use mpi_rt as mpi_api;

use mpi_api::traits::Communicator;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// A host array with its scalar kind: the payload of one patch node tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum WireArray {
    F32(Vec<f32>),
    F64(Vec<f64>),
    C32(Vec<Complex32>),
    C64(Vec<Complex64>),
}

/// One leg of a patch node tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum WireAxis {
    /// A full site axis, by position in the run's site order.
    Site { position: usize },
    /// A patch-local bond axis; legs with the same id share one index.
    Bond { id: usize },
}

/// One node of a patch network.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireNode {
    /// Position of the node in the root's node order.
    node: usize,
    /// Legs in the stored tensor's positional order.
    axes: Vec<WireAxis>,
    /// Logical shape, column-major payload.
    shape: Vec<usize>,
    values: WireArray,
}

/// An accepted patch network: host arrays and the metadata a root rebuilds it
/// with.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireSubdomain {
    nodes: Vec<WireNode>,
}

/// The packing of a patch cache, holes of a split included.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireCacheLayout {
    dims: Vec<usize>,
    slots: Vec<(usize, u32, u32)>,
    n_words: usize,
}

/// A patch cache: its packing and its keyed values.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireCache<T> {
    layout: WireCacheLayout,
    entries: Vec<(Vec<u64>, T)>,
}

/// A patch waiting for its wave.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WirePatch<T> {
    path: Vec<(usize, usize)>,
    fixed: Vec<Option<usize>>,
    cache: WireCache<T>,
    recycled: Vec<Vec<usize>>,
    worst: Vec<Vec<usize>>,
}

/// How a residual was measured.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum WireMethod {
    Exact,
    Exhaustive,
    Sampled,
    ResidualTt,
}

/// One L2 measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireMeasurement {
    method: WireMethod,
    points: usize,
    patch_points: f64,
    rms: f64,
    mean_square_rel_std_error: f64,
    max_residual: f64,
    residual_rms: Option<f64>,
    residual_saturated: bool,
    sample_rms: Option<f64>,
}

/// The engine's verdict of one kept run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum WireTermination {
    Converged,
    BondCapReached,
    IterationLimit,
}

/// Whether a retained patch meets its allowance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum WireStatus {
    WithinTolerance,
    ToleranceNotMet,
}

/// Where the L2 reference came from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum WireL2Source {
    Given,
    ExactRoot,
    NotNeeded,
    MonteCarlo {
        samples: usize,
        mean_square_rel_std_error: f64,
    },
}

/// Where the max reference came from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum WireMaxSource {
    Given,
    ExactRoot,
    MaxOfRootCandidates,
}

/// The pinned L2 reference and allowance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct WireL2Pin {
    reference_rms: Option<f64>,
    source: WireL2Source,
    tau: f64,
}

/// The reference norms a run pinned.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct WireReferences {
    l2: Option<WireL2Pin>,
    max_reference: Option<(f64, WireMaxSource)>,
}

/// One accepted patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireAccepted {
    record: WirePatchRecord,
    subdomain: WireSubdomain,
    patch_points: f64,
    log_norm: Option<f64>,
}

/// The record of one stored patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WirePatchRecord {
    termination: WireTermination,
    engine_error_estimate: f64,
    max_sample_magnitude: f64,
    max_bond_dim: usize,
    retries_used: usize,
    acceptance: Option<WireMeasurement>,
    audit: Option<WireMeasurement>,
    status: WireStatus,
}

/// The record of one zero patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireZeroPatch {
    acceptance: Option<WireMeasurement>,
    audit: Option<WireMeasurement>,
    patch_points: f64,
}

/// The counter deltas of one patch.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct WireTotals {
    evaluations: usize,
    cache_hits: usize,
    measurement_evaluations: usize,
    audit_evaluations: usize,
    residual_evaluations: usize,
    residual_saturations: usize,
    verification_failures: usize,
    engine_retries: usize,
}

/// What processing one patch produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum WireVerdict<T> {
    Accepted(Box<WireAccepted>),
    Zero(WireZeroPatch),
    Split(Vec<WirePatch<T>>),
}

/// One patch's outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireOutcome<T> {
    path: Vec<(usize, usize)>,
    verdict: WireVerdict<T>,
    totals: WireTotals,
    /// The references after the patch; only the root wave pins.
    references: WireReferences,
}

/// The driver's control after one wave.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum WireControl {
    Continue,
    Stop,
    Fail(String),
}

// ---------------------------------------------------------------------------
// Conversion: report records
// ---------------------------------------------------------------------------

fn encode_method(method: MeasurementMethod) -> WireMethod {
    match method {
        MeasurementMethod::Exact => WireMethod::Exact,
        MeasurementMethod::Exhaustive => WireMethod::Exhaustive,
        MeasurementMethod::Sampled => WireMethod::Sampled,
        MeasurementMethod::ResidualTt => WireMethod::ResidualTt,
    }
}

fn decode_method(method: WireMethod) -> MeasurementMethod {
    match method {
        WireMethod::Exact => MeasurementMethod::Exact,
        WireMethod::Exhaustive => MeasurementMethod::Exhaustive,
        WireMethod::Sampled => MeasurementMethod::Sampled,
        WireMethod::ResidualTt => MeasurementMethod::ResidualTt,
    }
}

fn encode_measurement(measurement: &L2Measurement) -> WireMeasurement {
    WireMeasurement {
        method: encode_method(measurement.method),
        points: measurement.points,
        patch_points: measurement.patch_points,
        rms: measurement.rms,
        mean_square_rel_std_error: measurement.mean_square_rel_std_error,
        max_residual: measurement.max_residual,
        residual_rms: measurement.residual_rms,
        residual_saturated: measurement.residual_saturated,
        sample_rms: measurement.sample_rms,
    }
}

fn decode_measurement(measurement: WireMeasurement) -> L2Measurement {
    L2Measurement {
        method: decode_method(measurement.method),
        points: measurement.points,
        patch_points: measurement.patch_points,
        rms: measurement.rms,
        mean_square_rel_std_error: measurement.mean_square_rel_std_error,
        max_residual: measurement.max_residual,
        residual_rms: measurement.residual_rms,
        residual_saturated: measurement.residual_saturated,
        sample_rms: measurement.sample_rms,
    }
}

fn encode_termination(
    termination: InterpolationTermination,
) -> Result<WireTermination, PatchedInterpolationError> {
    match termination {
        InterpolationTermination::Converged => Ok(WireTermination::Converged),
        InterpolationTermination::BondCapReached => Ok(WireTermination::BondCapReached),
        InterpolationTermination::IterationLimit => Ok(WireTermination::IterationLimit),
        // `InterpolationTermination` is non-exhaustive; a verdict this build
        // does not know cannot be reported faithfully on another rank.
        _ => Err(wire_error(format!(
            "the engine returned the unknown termination {termination:?}"
        ))),
    }
}

fn decode_termination(termination: WireTermination) -> InterpolationTermination {
    match termination {
        WireTermination::Converged => InterpolationTermination::Converged,
        WireTermination::BondCapReached => InterpolationTermination::BondCapReached,
        WireTermination::IterationLimit => InterpolationTermination::IterationLimit,
    }
}

fn encode_status(status: PatchStatus) -> WireStatus {
    match status {
        PatchStatus::WithinTolerance => WireStatus::WithinTolerance,
        PatchStatus::ToleranceNotMet => WireStatus::ToleranceNotMet,
    }
}

fn decode_status(status: WireStatus) -> PatchStatus {
    match status {
        WireStatus::WithinTolerance => PatchStatus::WithinTolerance,
        WireStatus::ToleranceNotMet => PatchStatus::ToleranceNotMet,
    }
}

fn encode_l2_source(source: L2ReferenceSource) -> WireL2Source {
    match source {
        L2ReferenceSource::Given => WireL2Source::Given,
        L2ReferenceSource::ExactRoot => WireL2Source::ExactRoot,
        L2ReferenceSource::NotNeeded => WireL2Source::NotNeeded,
        L2ReferenceSource::MonteCarlo {
            samples,
            mean_square_rel_std_error,
        } => WireL2Source::MonteCarlo {
            samples,
            mean_square_rel_std_error,
        },
    }
}

fn decode_l2_source(source: WireL2Source) -> L2ReferenceSource {
    match source {
        WireL2Source::Given => L2ReferenceSource::Given,
        WireL2Source::ExactRoot => L2ReferenceSource::ExactRoot,
        WireL2Source::NotNeeded => L2ReferenceSource::NotNeeded,
        WireL2Source::MonteCarlo {
            samples,
            mean_square_rel_std_error,
        } => L2ReferenceSource::MonteCarlo {
            samples,
            mean_square_rel_std_error,
        },
    }
}

fn encode_max_source(source: MaxReferenceSource) -> WireMaxSource {
    match source {
        MaxReferenceSource::Given => WireMaxSource::Given,
        MaxReferenceSource::ExactRoot => WireMaxSource::ExactRoot,
        MaxReferenceSource::MaxOfRootCandidates => WireMaxSource::MaxOfRootCandidates,
    }
}

fn decode_max_source(source: WireMaxSource) -> MaxReferenceSource {
    match source {
        WireMaxSource::Given => MaxReferenceSource::Given,
        WireMaxSource::ExactRoot => MaxReferenceSource::ExactRoot,
        WireMaxSource::MaxOfRootCandidates => MaxReferenceSource::MaxOfRootCandidates,
    }
}

fn encode_patch_record(record: &PatchRecord) -> Result<WirePatchRecord, PatchedInterpolationError> {
    Ok(WirePatchRecord {
        termination: encode_termination(record.termination)?,
        engine_error_estimate: record.engine_error_estimate,
        max_sample_magnitude: record.max_sample_magnitude,
        max_bond_dim: record.max_bond_dim,
        retries_used: record.retries_used,
        acceptance: record.acceptance.as_ref().map(encode_measurement),
        audit: record.audit.as_ref().map(encode_measurement),
        status: encode_status(record.status),
    })
}

fn decode_patch_record(record: WirePatchRecord, projector: Projector) -> PatchRecord {
    PatchRecord {
        projector,
        termination: decode_termination(record.termination),
        engine_error_estimate: record.engine_error_estimate,
        max_sample_magnitude: record.max_sample_magnitude,
        max_bond_dim: record.max_bond_dim,
        retries_used: record.retries_used,
        acceptance: record.acceptance.map(decode_measurement),
        audit: record.audit.map(decode_measurement),
        status: decode_status(record.status),
    }
}

fn encode_references(references: References) -> WireReferences {
    WireReferences {
        l2: references.l2.map(|pin| WireL2Pin {
            reference_rms: pin.reference_rms,
            source: encode_l2_source(pin.source),
            tau: pin.tau,
        }),
        max_reference: references
            .max_reference
            .map(|(value, source)| (value, encode_max_source(source))),
    }
}

fn decode_references(references: &WireReferences) -> References {
    let mut decoded = References::default();
    if let Some(pin) = references.l2 {
        decoded.l2 = Some(L2Pin {
            reference_rms: pin.reference_rms,
            source: decode_l2_source(pin.source),
            tau: pin.tau,
        });
    }
    if let Some((value, source)) = references.max_reference {
        decoded.max_reference = Some((value, decode_max_source(source)));
    }
    decoded
}

fn encode_totals(totals: Totals) -> WireTotals {
    WireTotals {
        evaluations: totals.evaluations,
        cache_hits: totals.cache_hits,
        measurement_evaluations: totals.measurement_evaluations,
        audit_evaluations: totals.audit_evaluations,
        residual_evaluations: totals.residual_evaluations,
        residual_saturations: totals.residual_saturations,
        verification_failures: totals.verification_failures,
        engine_retries: totals.engine_retries,
    }
}

fn decode_totals(totals: WireTotals) -> Totals {
    Totals {
        evaluations: totals.evaluations,
        cache_hits: totals.cache_hits,
        measurement_evaluations: totals.measurement_evaluations,
        audit_evaluations: totals.audit_evaluations,
        residual_evaluations: totals.residual_evaluations,
        residual_saturations: totals.residual_saturations,
        verification_failures: totals.verification_failures,
        engine_retries: totals.engine_retries,
    }
}

fn wire_error(message: impl std::fmt::Display) -> PatchedInterpolationError {
    PatchedInterpolationError::FanOut {
        message: format!("the MPI wire rejected a record: {message}"),
    }
}

// ---------------------------------------------------------------------------
// Conversion: patch caches and patches
// ---------------------------------------------------------------------------

fn encode_cache<T: Copy>(cache: super::cache::PatchCache<T>) -> WireCache<T> {
    let layout = cache.layout_description();
    WireCache {
        layout: WireCacheLayout {
            dims: layout.dims,
            slots: layout.slots,
            n_words: layout.n_words,
        },
        entries: cache.into_entries(),
    }
}

fn decode_cache<T: Copy>(cache: WireCache<T>) -> Result<super::cache::PatchCache<T>, String> {
    super::cache::PatchCache::from_entries(
        CacheLayout {
            dims: cache.layout.dims,
            slots: cache.layout.slots,
            n_words: cache.layout.n_words,
        },
        cache.entries,
    )
}

fn encode_patch<T: Copy>(patch: Patch<T>) -> WirePatch<T> {
    let Patch {
        path,
        fixed,
        cache,
        recycled,
        worst,
    } = patch;
    WirePatch {
        path,
        fixed,
        cache: encode_cache(cache),
        recycled,
        worst,
    }
}

/// Rebuild a patch and check it against the receiver's site table, so a rank
/// whose problem differs from the root's fails instead of indexing out of range.
fn decode_patch<T: Copy, V>(layout: &SiteLayout<V>, patch: WirePatch<T>) -> Result<Patch<T>, String>
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    if patch.fixed.len() != layout.sites.len() {
        return Err(format!(
            "the wire fixes {} sites, this rank has {}",
            patch.fixed.len(),
            layout.sites.len()
        ));
    }
    for (position, coordinate) in patch.fixed.iter().enumerate() {
        if let Some(coordinate) = *coordinate
            && coordinate >= layout.dims[position]
        {
            return Err(format!(
                "the wire fixes site {position} to {coordinate}, outside {}",
                layout.dims[position]
            ));
        }
    }
    for &(position, _) in &patch.path {
        if position >= layout.sites.len() {
            return Err(format!(
                "the wire path names site position {position} of {}",
                layout.sites.len()
            ));
        }
    }
    for point in patch.recycled.iter().chain(&patch.worst) {
        if point.len() != layout.sites.len() {
            return Err(format!(
                "a wire point has {} coordinates, this rank has {} sites",
                point.len(),
                layout.sites.len()
            ));
        }
        for (position, &coordinate) in point.iter().enumerate() {
            if coordinate >= layout.dims[position] {
                return Err(format!(
                    "a wire point has coordinate {coordinate} outside dimension {} at site                      {position}",
                    layout.dims[position]
                ));
            }
        }
    }
    Ok(Patch {
        path: patch.path,
        fixed: patch.fixed,
        cache: decode_cache(patch.cache)?,
        recycled: patch.recycled,
        worst: patch.worst,
    })
}

// ---------------------------------------------------------------------------
// Conversion: patch networks
// ---------------------------------------------------------------------------

fn encode_values(tensor: &IdxTensor) -> Result<WireArray, PatchedInterpolationError> {
    let values = if tensor.is_f32() {
        WireArray::F32(
            tensor
                .dense_host_values::<f32>()
                .map_err(PartitionedTreeTNError::from)?,
        )
    } else if tensor.is_f64() {
        WireArray::F64(
            tensor
                .dense_host_values::<f64>()
                .map_err(PartitionedTreeTNError::from)?,
        )
    } else if tensor.is_c32() {
        WireArray::C32(
            tensor
                .dense_host_values::<Complex32>()
                .map_err(PartitionedTreeTNError::from)?,
        )
    } else if tensor.is_c64() {
        WireArray::C64(
            tensor
                .dense_host_values::<Complex64>()
                .map_err(PartitionedTreeTNError::from)?,
        )
    } else {
        return Err(PatchedInterpolationError::Partition {
            source: PartitionedTreeTNError::UnsupportedDType {
                dtype: format!("{:?}", tensor.storage_kind()),
            },
        });
    };
    Ok(values)
}

/// Encode a patch network against the run's site table and node order.
fn encode_subdomain<V>(
    layout: &SiteLayout<V>,
    subdomain: &SubDomainTreeTN<V>,
) -> Result<WireSubdomain, PatchedInterpolationError>
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    let data = subdomain.data();
    let mut bonds: Vec<(DynIndex, usize)> = Vec::new();
    let mut nodes = Vec::with_capacity(data.node_count());
    for name in data.node_names() {
        let position = layout
            .node_positions
            .iter()
            .position(|(node, _)| node == &name)
            .ok_or_else(|| wire_error(format!("the patch network has an unknown node {name:?}")))?;
        let index = data.node_index(&name).ok_or_else(|| {
            wire_error(format!("the patch network has no node index for {name:?}"))
        })?;
        let tensor = data
            .tensor(index)
            .ok_or_else(|| wire_error(format!("the patch network has no tensor for {name:?}")))?;
        let mut axes = Vec::with_capacity(tensor.indices().len());
        for leg in tensor.indices() {
            if let Some(site) = layout.sites.iter().position(|site| site == leg) {
                axes.push(WireAxis::Site { position: site });
                continue;
            }
            let id = match bonds.iter().position(|(bond, _)| bond == leg) {
                Some(id) => id,
                None => {
                    bonds.push((leg.clone(), bonds.len()));
                    bonds.len() - 1
                }
            };
            axes.push(WireAxis::Bond { id });
        }
        nodes.push(WireNode {
            node: position,
            axes,
            shape: tensor.dims(),
            values: encode_values(tensor)?,
        });
    }
    Ok(WireSubdomain { nodes })
}

fn decode_storage(
    values: WireArray,
    shape: &[usize],
) -> Result<Storage, PatchedInterpolationError> {
    let storage = match values {
        WireArray::F32(values) => Storage::from_dense_col_major(values, shape),
        WireArray::F64(values) => Storage::from_dense_col_major(values, shape),
        WireArray::C32(values) => Storage::from_dense_col_major(values, shape),
        WireArray::C64(values) => Storage::from_dense_col_major(values, shape),
    };
    storage.map_err(wire_error)
}

/// Rebuild a patch network from its host arrays and metadata.
fn decode_subdomain<V>(
    layout: &SiteLayout<V>,
    projector: Projector,
    subdomain: WireSubdomain,
) -> Result<SubDomainTreeTN<V>, PatchedInterpolationError>
where
    V: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let mut bonds: BTreeMap<usize, DynIndex> = BTreeMap::new();
    let mut names = Vec::with_capacity(subdomain.nodes.len());
    let mut tensors = Vec::with_capacity(subdomain.nodes.len());
    for node in subdomain.nodes {
        let (name, _) = layout
            .node_positions
            .get(node.node)
            .ok_or_else(|| wire_error(format!("the wire names node position {}", node.node)))?;
        if node.shape.len() != node.axes.len() {
            return Err(wire_error(format!(
                "a node of the wire has {} legs but {} dimensions",
                node.axes.len(),
                node.shape.len()
            )));
        }
        let mut legs = Vec::with_capacity(node.axes.len());
        for (axis, &dim) in node.axes.iter().zip(&node.shape) {
            let index = match *axis {
                WireAxis::Site { position } => layout
                    .sites
                    .get(position)
                    .ok_or_else(|| wire_error(format!("the wire names site position {position}")))?
                    .clone(),
                WireAxis::Bond { id } => match bonds.get(&id) {
                    Some(bond) => bond.clone(),
                    None => {
                        let bond = DynIndex::new_dyn(dim);
                        bonds.insert(id, bond.clone());
                        bond
                    }
                },
            };
            legs.push(index);
        }
        let storage = decode_storage(node.values, &node.shape)?;
        tensors.push(
            IdxTensor::from_storage(legs, Arc::new(storage))
                .map_err(PartitionedTreeTNError::from)?,
        );
        names.push(name.clone());
    }
    let data = tensor4all_treetn::TreeTN::from_tensors(tensors, names)
        .map_err(PartitionedTreeTNError::from)?;
    Ok(SubDomainTreeTN::from_masked_data(data, projector, None)?)
}

// ---------------------------------------------------------------------------
// Conversion: outcomes
// ---------------------------------------------------------------------------

fn encode_outcome<T, V, E>(
    plan: &Plan<'_, T, V, E>,
    run: PatchRun<T, V>,
) -> Result<WireOutcome<T>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement + Serialize + Copy,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
{
    let PatchRun {
        path,
        verdict,
        totals,
        references,
    } = run;
    let verdict = match verdict {
        Verdict::Accepted(entry) => {
            let Accepted {
                record,
                subdomain,
                patch_points,
                log_norm,
            } = *entry;
            WireVerdict::Accepted(Box::new(WireAccepted {
                record: encode_patch_record(&record)?,
                subdomain: encode_subdomain(&plan.layout, &subdomain)?,
                patch_points,
                log_norm,
            }))
        }
        Verdict::Zero(record, patch_points) => WireVerdict::Zero(WireZeroPatch {
            acceptance: record.acceptance.as_ref().map(encode_measurement),
            audit: record.audit.as_ref().map(encode_measurement),
            patch_points,
        }),
        Verdict::Split(children) => WireVerdict::Split(
            children
                .into_iter()
                .map(|child| encode_patch(child))
                .collect(),
        ),
    };
    Ok(WireOutcome {
        path,
        verdict,
        totals: encode_totals(totals),
        references: encode_references(references),
    })
}

fn decode_outcome<T, V, E>(
    plan: &Plan<'_, T, V, E>,
    outcome: WireOutcome<T>,
) -> Result<PatchRun<T, V>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement + Copy,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
{
    let WireOutcome {
        path,
        verdict,
        totals,
        references,
    } = outcome;
    let projector = plan.layout.projector(&path)?;
    let verdict = match verdict {
        WireVerdict::Accepted(entry) => {
            let WireAccepted {
                record,
                subdomain,
                patch_points,
                log_norm,
            } = *entry;
            let record = decode_patch_record(record, projector.clone());
            let subdomain = decode_subdomain(&plan.layout, projector, subdomain)?;
            Verdict::Accepted(Box::new(Accepted {
                record,
                subdomain,
                patch_points,
                log_norm,
            }))
        }
        WireVerdict::Zero(record) => Verdict::Zero(
            ZeroPatchRecord {
                projector,
                acceptance: record.acceptance.map(decode_measurement),
                audit: record.audit.map(decode_measurement),
            },
            record.patch_points,
        ),
        WireVerdict::Split(children) => Verdict::Split(
            children
                .into_iter()
                .map(|child| decode_patch(&plan.layout, child).map_err(wire_error))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    Ok(PatchRun {
        path,
        verdict,
        totals: decode_totals(totals),
        references: decode_references(&references),
    })
}

// ---------------------------------------------------------------------------
// The distributed driver
// ---------------------------------------------------------------------------

/// The MPI driver: the root orchestrates waves, every rank runs its share.
struct MpiWaves<'a, 'world, C, F> {
    world: &'world C,
    lanes: &'a mut TenferroLanes,
    evaluate: F,
    root: i32,
    rank: i32,
    options: PmapOptions,
}

impl<T, V, E, C, F> Waves<T, V, E> for MpiWaves<'_, '_, C, F>
where
    T: CommonScalar + TensorElement + Send + Serialize + for<'de> Deserialize<'de> + Copy,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    C: Communicator,
    F: Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>> + Sync,
{
    fn run_wave(
        &mut self,
        plan: &Plan<'_, T, V, E>,
        patches: Vec<Patch<T>>,
        pins: WavePins<'_>,
    ) -> Result<Vec<PatchRun<T, V>>, PatchedInterpolationError> {
        let world = self.world;
        let evaluate = &self.evaluate;
        let lanes = &mut *self.lanes;
        let options = self.options;
        let rank = self.rank;
        let root = self.root;
        let root_wave = pins.writable();
        let start = encode_references(pins.start());
        let items = if rank == root {
            Some(
                patches
                    .into_iter()
                    .map(encode_patch)
                    .collect::<Vec<WirePatch<T>>>(),
            )
        } else {
            None
        };
        let shared = Shared(evaluate);
        let outcome = pmap_lanes(world, lanes, options, items, |wire: WirePatch<T>, lane| {
            let patch = decode_patch(&plan.layout, wire).map_err(|error| error.to_string())?;
            let mut local = decode_references(&start);
            let pins = if root_wave {
                Pins::Root(&mut local)
            } else {
                Pins::Later(&local)
            };
            let mut ctx = EvalContext::new(
                tensor4all_tensorbackend::explicit::Session::new(lane),
                Parallelism::Serial,
            );
            match super::patch_outcome(plan, pins, &shared, patch, &mut ctx)
                .and_then(|run| encode_outcome(plan, run))
            {
                Ok(outcome) => Ok(outcome),
                // The typed detail stays on the rank that produced it; the scheduler carries the message.
                Err(error) => Err(error.to_string()),
            }
        });
        let outcomes = outcome.map_err(|error| PatchedInterpolationError::FanOut {
            message: format!("the MPI scheduler failed: {error}"),
        })?;
        match outcomes {
            Some(outcomes) => {
                let decoded: Result<Vec<_>, _> = outcomes
                    .into_iter()
                    .map(|outcome| decode_outcome(plan, outcome))
                    .collect();
                // Decoding is root-only work, so it converges with the wave control before returning:
                // a peer must not enter `agree` while the root bails out here.
                agree_on_root_outcome(world, root, rank, decoded)
            }
            // Every rank but the root owns no patches and no outcome.
            None => agree_on_root_outcome(world, root, rank, Ok(Vec::new())),
        }
    }

    fn agree(
        &mut self,
        control: WaveControl,
        references: References,
    ) -> Result<(WaveControl, References), PatchedInterpolationError> {
        let wire_control = match &control {
            WaveControl::Continue => WireControl::Continue,
            WaveControl::Stop => WireControl::Stop,
            WaveControl::Fail(message) => WireControl::Fail(message.clone()),
        };
        let agreed = broadcast(
            self.world,
            self.root,
            (self.rank == self.root).then_some(wire_control),
        )
        .map_err(|error| PatchedInterpolationError::FanOut {
            message: format!("the MPI control broadcast failed: {error}"),
        })?;
        let agreed_references = broadcast(
            self.world,
            self.root,
            (self.rank == self.root).then_some(encode_references(references)),
        )
        .map_err(|error| PatchedInterpolationError::FanOut {
            message: format!("the MPI reference broadcast failed: {error}"),
        })?;
        let control = match agreed {
            WireControl::Continue => WaveControl::Continue,
            WireControl::Stop => WaveControl::Stop,
            WireControl::Fail(message) => WaveControl::Fail(message),
        };
        Ok((control, decode_references(&agreed_references)))
    }
}

/// Converge a root-only result before the next collective step.
///
/// Only the root can decode the wire outcomes of a wave or assemble the final partition, so a failure it
/// observes there must fail every rank: a peer that kept going would either enter the next broadcast
/// without a partner (stranding the run) or report success while the root fails. The root's result is
/// reduced to a control and broadcast exactly as the wave control is, and every rank reads the agreed
/// control in [`converge_root_outcome`].
fn agree_on_root_outcome<C, R>(
    world: &C,
    root: i32,
    rank: i32,
    local: Result<R, PatchedInterpolationError>,
) -> Result<R, PatchedInterpolationError>
where
    C: Communicator,
{
    let control = (rank == root).then(|| match &local {
        Ok(_) => WireControl::Continue,
        Err(error) => WireControl::Fail(error.to_string()),
    });
    let control =
        broadcast(world, root, control).map_err(|error| PatchedInterpolationError::FanOut {
            message: format!("the MPI result broadcast failed: {error}"),
        })?;
    converge_root_outcome(local, control)
}

/// The rank-local reading of the agreed control for a root-only result.
///
/// `Continue` keeps the rank's own result - the root's decoded outcomes or assembled partition, and a
/// peer's empty placeholder - while `Fail` fails every rank with the converged message.
fn converge_root_outcome<R>(
    local: Result<R, PatchedInterpolationError>,
    control: WireControl,
) -> Result<R, PatchedInterpolationError> {
    match control {
        WireControl::Continue => local,
        WireControl::Fail(message) => Err(PatchedInterpolationError::FanOut { message }),
        WireControl::Stop => Err(PatchedInterpolationError::FanOut {
            message: "invalid stop control for a root-only result".to_string(),
        }),
    }
}

/// Agree on a root-side validation message before any wave runs.
fn agree_on_validation<C: Communicator>(
    world: &C,
    root: i32,
    rank: i32,
    local_error: Option<String>,
) -> Result<(), PatchedInterpolationError> {
    let gathered =
        gather(world, root, local_error).map_err(|error| PatchedInterpolationError::FanOut {
            message: format!("the MPI validation gather failed: {error}"),
        })?;
    let control = (rank == root).then(|| {
        match gathered
            .as_ref()
            .and_then(|errors| errors.iter().flatten().next())
        {
            Some(message) => WireControl::Fail(message.clone()),
            None => WireControl::Continue,
        }
    });
    let control =
        broadcast(world, root, control).map_err(|error| PatchedInterpolationError::FanOut {
            message: format!("the MPI validation broadcast failed: {error}"),
        })?;
    match control {
        WireControl::Continue => Ok(()),
        WireControl::Fail(message) => Err(PatchedInterpolationError::FanOut { message }),
        WireControl::Stop => Err(PatchedInterpolationError::FanOut {
            message: "invalid stop control during MPI validation".to_string(),
        }),
    }
}

/// Adaptively interpolate a function on a tree, distributing every wave of
/// patches over the ranks of an MPI world.
///
/// The distributed counterpart of
/// [`patched_interpolate`](super::patched_interpolate) and
/// [`patched_interpolate_on_lanes`](super::patched_interpolate_on_lanes): it
/// takes the same engine, problem, pivots, evaluator and options, distributes
/// each wave through [`hataori::pmap_lanes`], and returns the same result on the
/// root. Every rank opens its own lane executor over its own backend; only the
/// root owns the patch queue, the wave merge, the pinned references and the
/// result.
///
/// # Arguments
///
/// * `world` - The MPI communicator. Every rank must call this function once,
///   in the same order, with the same problem and options; MPI must provide
///   `MPI_THREAD_FUNNELED` or stronger, because the progress loop stays on the
///   initialization thread while the lanes compute.
/// * `lanes` - The rank's lane executor, built from one
///   [`CpuEngine`](tenferro_cpu::CpuEngine) per lane
///   ([`TenferroLanes::new`](hataori_tenferro::TenferroLanes::new)).
/// * `root` - The rank that owns the queue and receives the result. It must be
///   in `[0, world.size())`.
/// * `engine`, `topology`, `node_sites`, `initial_pivots`, `evaluate`,
///   `options` - As in [`patched_interpolate`](super::patched_interpolate).
///   The evaluator must be `Fn + Sync` and is only called on the rank and lane
///   that runs a patch.
///
/// # Returns
///
/// `Ok(Some(result))` on the root and `Ok(None)` on every other rank.
///
/// # Errors
///
/// A collective failure is reported on every rank as
/// [`PatchedInterpolationError::FanOut`] with the converged message. The root
/// keeps its typed error where it has one (the `max_patches` allowance, for
/// example). The problem and options are validated locally on every rank and
/// the rank-local messages are gathered first, so a rank with invalid input
/// fails the whole run instead of diverging.
///
/// # Examples
///
/// A runnable multi-rank example with numerical assertions is provided at
/// `examples/adaptive_mpi_smoke.rs` in this crate; it runs under a launcher:
///
/// ```sh
/// MPI_RT_LIB=/path/to/libmpiwrapper.so \
///     mpiexec -n 2 target/debug/examples/adaptive_mpi_smoke
/// ```
///
/// `scripts/check-mpi-smoke.sh` builds and runs the MPI integration test at
/// world sizes one and two. This entry needs an initialized MPI universe, which
/// a rustdoc example cannot provide, so the example is a binary instead of a
/// doc fence.
///
#[allow(clippy::too_many_arguments)]
pub fn patched_interpolate_mpi<T, V, E, F, C>(
    world: &C,
    lanes: &mut TenferroLanes,
    root: i32,
    engine: &E,
    topology: NodeNameNetwork<V>,
    node_sites: BTreeMap<V, Vec<DynIndex>>,
    initial_pivots: ColMajorArray<usize>,
    evaluate: F,
    options: &PatchedInterpolationOptions,
) -> Result<Option<PatchedInterpolationResult<V>>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement + Send + Serialize + for<'de> Deserialize<'de> + Copy,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>> + Sync,
    C: Communicator,
{
    let rank = world.rank();
    let size = world.size();
    if root < 0 || root >= size {
        return Err(PatchedInterpolationError::FanOut {
            message: format!("the MPI root {root} is outside the world of {size} ranks"),
        });
    }
    let validation = Plan::validated(engine, topology, node_sites, initial_pivots, options);
    let local_error = validation.as_ref().err().map(ToString::to_string);
    agree_on_validation(world, root, rank, local_error)?;
    let plan = validation?;
    let mut waves = MpiWaves {
        world,
        lanes,
        evaluate,
        root,
        rank,
        options: PmapOptions {
            root,
            batch_size: NonZeroUsize::MIN,
            local_mode: hataori::LocalMode::Outer,
            prefetch: false,
        },
    };
    let merged = run_waves(&plan, &mut waves)?;
    // Assembly is root-only, so its failure converges like the wave control; a peer must not report
    // success for a run the root could not assemble.
    let assembled = if rank == root {
        assemble(&plan, merged).map(Some)
    } else {
        Ok(None)
    };
    agree_on_root_outcome(world, root, rank, assembled)
}

#[cfg(test)]
mod tests {
    use super::super::cache::PatchCache;
    use super::super::layout::SiteLayout;
    use super::*;
    use std::collections::BTreeMap;
    use tensor4all_treetn::NodeNameNetwork;

    /// A layout of a two-node chain, each node with one binary site.
    fn chain_layout() -> SiteLayout<String> {
        let mut topology = NodeNameNetwork::new();
        topology.add_node("a".to_string()).unwrap();
        topology.add_node("b".to_string()).unwrap();
        topology
            .add_edge(&"a".to_string(), &"b".to_string())
            .unwrap();
        let node_sites = BTreeMap::from([
            ("a".to_string(), vec![DynIndex::new_dyn(2)]),
            ("b".to_string(), vec![DynIndex::new_dyn(2)]),
        ]);
        let pivots = ColMajorArray::new(vec![0, 0], vec![2, 1]).unwrap();
        SiteLayout::validated(
            topology,
            node_sites,
            &pivots,
            &PatchedInterpolationOptions::new(2)
                .with_error_norm(super::super::ErrorNorm::sampled_max()),
        )
        .unwrap()
    }

    /// The site layout of a one-node problem with three binary sites.
    fn chain_cache_layout() -> SiteLayout<String> {
        let mut topology = NodeNameNetwork::new();
        topology.add_node("a".to_string()).unwrap();
        let node_sites = BTreeMap::from([(
            "a".to_string(),
            vec![
                DynIndex::new_dyn(2),
                DynIndex::new_dyn(2),
                DynIndex::new_dyn(2),
            ],
        )]);
        let pivots = ColMajorArray::new(vec![0, 0, 0], vec![3, 1]).unwrap();
        SiteLayout::validated(
            topology,
            node_sites,
            &pivots,
            &PatchedInterpolationOptions::new(2)
                .with_error_norm(super::super::ErrorNorm::sampled_max()),
        )
        .unwrap()
    }

    /// A cache split at its first active slot keeps the parent's word offsets,
    /// so the child packs a coordinate at a different shift than a compact
    /// layout of its own dimensions would; the wire must carry the layout.
    #[test]
    fn a_split_cache_crosses_the_wire_with_its_holes() {
        let big = 1usize << 40;
        let mut cache = PatchCache::new(vec![2, big, 2]);
        let key = cache.layout().encode([1usize, 5, 1].into_iter());
        cache.insert(key, 1.5_f64);
        let children = cache.split(0).unwrap();
        let child = children.into_iter().nth(1).unwrap();
        let layout = child.layout_description();
        // The first slot of the parent occupied bit 0; the split clears it and
        // keeps the remaining slots at their shifts.
        assert_eq!(layout.dims, vec![big, 2]);
        assert_eq!(layout.slots, vec![(0, 1, 40), (0, 41, 1)]);
        let entries = child.into_entries();
        // The point (5, 1) keeps the parent's shifts: `5` still sits at bit 1
        // and the child's second slot at bit 41, with the split slot's bit 0
        // cleared.
        assert_eq!(entries, vec![(vec![(5u64 << 1) | (1u64 << 41)], 1.5_f64)]);

        let decoded = PatchCache::from_entries(layout.clone(), entries.clone()).unwrap();
        assert_eq!(decoded.layout_description(), layout);
        assert_eq!(decoded.get(&entries[0].0), Some(1.5));
        // A compact packing of the child's own dimensions places the same
        // coordinates at other bits, so a receiver that inferred the layout
        // from the dimensions would look up the wrong key.
        let wrong = PatchCache::<f64>::new(vec![big, 2])
            .layout()
            .encode([5usize, 1].into_iter());
        assert_ne!(wrong.as_ref(), entries[0].0.as_slice());
        assert_ne!(wrong.as_ref(), entries[0].0.as_slice());
        assert_eq!(decoded.get(&wrong), None);

        // A malformed description is rejected instead of misread.
        assert!(PatchCache::from_entries(
            CacheLayout {
                dims: vec![big, 2],
                slots: vec![(1, 1, 40), (0, 41, 1)],
                n_words: 1,
            },
            entries,
        )
        .is_err());
    }

    #[test]
    fn references_cross_the_wire_with_their_provenance() {
        let references = References {
            l2: Some(L2Pin {
                reference_rms: Some(0.25),
                source: L2ReferenceSource::MonteCarlo {
                    samples: 17,
                    mean_square_rel_std_error: 0.125,
                },
                tau: 0.5,
            }),
            max_reference: Some((3.5, MaxReferenceSource::MaxOfRootCandidates)),
        };
        let wire = encode_references(references);
        let decoded = decode_references(&wire);
        assert_eq!(encode_references(decoded), wire);
        assert_eq!(decoded.l2.map(|pin| pin.tau), Some(0.5));
        assert_eq!(decoded.l2.map(|pin| pin.reference_rms), Some(Some(0.25)));

        let empty = encode_references(References::default());
        assert!(decode_references(&empty).l2.is_none());
        assert!(decode_references(&empty).max_reference.is_none());
    }

    #[test]
    fn a_patch_crosses_the_wire_with_its_inherited_state() {
        let big = 1usize << 40;
        let mut cache = PatchCache::new(vec![2, big, 2]);
        cache.insert(cache.layout().encode([1usize, 7, 0].into_iter()), 2.5_f64);
        let children = cache.split(0).unwrap();
        let cache = children.into_iter().nth(1).unwrap();
        let patch = Patch {
            path: vec![(0, 1)],
            fixed: vec![Some(1), None, None],
            cache,
            recycled: vec![vec![1, 7 % 2, 0]],
            worst: vec![vec![1, 3 % 2, 1]],
        };
        let wire = encode_patch(patch);
        let layout = chain_cache_layout();
        let decoded = decode_patch(&layout, wire.clone()).unwrap();
        assert_eq!(decoded.path, vec![(0, 1)]);
        assert_eq!(decoded.fixed, vec![Some(1), None, None]);
        assert_eq!(decoded.recycled, vec![vec![1, 1, 0]]);
        assert_eq!(decoded.worst, vec![vec![1, 1, 1]]);
        // A point outside the receiver's site dimensions is rejected.
        let mut bad = wire.clone();
        bad.recycled = vec![vec![1, 7, 0]];
        assert!(decode_patch(&layout, bad).is_err());
        let expected = CacheLayout {
            dims: wire.cache.layout.dims.clone(),
            slots: wire.cache.layout.slots.clone(),
            n_words: wire.cache.layout.n_words,
        };
        assert_eq!(decoded.cache.layout_description(), expected);

        // The record survives a serde round trip, key packing included.
        let json = serde_json::to_string(&wire).unwrap();
        let back: WirePatch<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(format!("{back:?}"), format!("{wire:?}"));
    }

    #[test]
    fn a_patch_network_crosses_the_wire_with_its_axes() {
        let layout = chain_layout();
        let wire = WireSubdomain {
            nodes: vec![
                WireNode {
                    node: 0,
                    axes: vec![WireAxis::Site { position: 0 }, WireAxis::Bond { id: 0 }],
                    shape: vec![2, 3],
                    values: WireArray::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                },
                WireNode {
                    node: 1,
                    axes: vec![WireAxis::Bond { id: 0 }, WireAxis::Site { position: 1 }],
                    shape: vec![3, 2],
                    values: WireArray::F64(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
                },
            ],
        };
        let projector = layout.projector(&[]).unwrap();
        let subdomain = decode_subdomain(&layout, projector, wire.clone()).unwrap();
        assert_eq!(subdomain.node_count(), 2);
        assert_eq!(subdomain.max_bond_dim(), 3);
        let back = encode_subdomain(&layout, &subdomain).unwrap();
        assert_eq!(format!("{back:?}"), format!("{wire:?}"));
    }

    #[test]
    fn an_outcome_crosses_the_wire_as_a_whole() {
        let subdomain = WireSubdomain {
            nodes: vec![
                WireNode {
                    node: 0,
                    axes: vec![WireAxis::Site { position: 0 }],
                    shape: vec![2],
                    values: WireArray::F32(vec![1.0, 2.0]),
                },
                WireNode {
                    node: 1,
                    axes: vec![WireAxis::Site { position: 1 }],
                    shape: vec![2],
                    values: WireArray::C64(vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 1.0),
                    ]),
                },
            ],
        };
        let outcome = WireOutcome {
            path: Vec::new(),
            verdict: WireVerdict::<f64>::Accepted(Box::new(WireAccepted {
                record: WirePatchRecord {
                    termination: WireTermination::Converged,
                    engine_error_estimate: 1e-9,
                    max_sample_magnitude: 2.0,
                    max_bond_dim: 1,
                    retries_used: 1,
                    acceptance: Some(WireMeasurement {
                        method: WireMethod::Exhaustive,
                        points: 4,
                        patch_points: 4.0,
                        rms: 0.0,
                        mean_square_rel_std_error: 0.0,
                        max_residual: 0.0,
                        residual_rms: None,
                        residual_saturated: false,
                        sample_rms: None,
                    }),
                    audit: None,
                    status: WireStatus::WithinTolerance,
                },
                subdomain,
                patch_points: 4.0,
                log_norm: Some(1.5),
            })),
            totals: WireTotals {
                evaluations: 9,
                cache_hits: 2,
                measurement_evaluations: 7,
                audit_evaluations: 0,
                residual_evaluations: 3,
                residual_saturations: 1,
                verification_failures: 0,
                engine_retries: 1,
            },
            references: encode_references(References::default()),
        };
        let json = serde_json::to_string(&outcome).unwrap();
        let back: WireOutcome<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(format!("{back:?}"), format!("{outcome:?}"));
        // The verdict keeps the pieces a distributed report is assembled from.
        match &back.verdict {
            WireVerdict::Accepted(accepted) => {
                assert_eq!(
                    accepted.record.acceptance.as_ref().map(|m| m.points),
                    Some(4)
                );
                assert_eq!(accepted.record.retries_used, 1);
                assert_eq!(accepted.patch_points, 4.0);
                assert_eq!(accepted.log_norm, Some(1.5));
                assert_eq!(accepted.subdomain.nodes.len(), 2);
            }
            other => panic!("expected an accepted outcome, got {other:?}"),
        }
        assert_eq!(back.totals.evaluations, 9);
    }

    #[test]
    fn a_residual_tt_measurement_and_its_counters_cross_the_wire() {
        let measurement = WireMeasurement {
            method: WireMethod::ResidualTt,
            points: 8,
            patch_points: 64.0,
            rms: 0.25,
            mean_square_rel_std_error: 0.5,
            max_residual: 0.75,
            residual_rms: Some(0.25),
            residual_saturated: true,
            sample_rms: Some(0.125),
        };
        let json = serde_json::to_string(&measurement).unwrap();
        let back: WireMeasurement = serde_json::from_str(&json).unwrap();
        assert_eq!(format!("{back:?}"), format!("{measurement:?}"));
        assert!(matches!(
            decode_method(back.method),
            MeasurementMethod::ResidualTt
        ));

        let totals = WireTotals {
            evaluations: 21,
            cache_hits: 3,
            measurement_evaluations: 11,
            audit_evaluations: 2,
            residual_evaluations: 9,
            residual_saturations: 2,
            verification_failures: 1,
            engine_retries: 1,
        };
        let json = serde_json::to_string(&totals).unwrap();
        let back: WireTotals = serde_json::from_str(&json).unwrap();
        assert_eq!(format!("{back:?}"), format!("{totals:?}"));
        let decoded = decode_totals(back);
        assert_eq!(decoded.residual_evaluations, 9);
        assert_eq!(decoded.residual_saturations, 2);
        assert_eq!(decoded.measurement_evaluations, 11);
    }

    /// A root-only result converges through the agreed control: a failure the root alone observed fails
    /// every rank - including peers whose own placeholder result is `Ok` - and a success keeps each rank's
    /// own result.
    #[test]
    fn a_root_only_result_converges_through_the_agreed_control() {
        // The root succeeded: the control keeps its decoded outcomes.
        assert_eq!(
            converge_root_outcome(Ok::<u8, _>(7), WireControl::Continue).ok(),
            Some(7)
        );
        // A peer keeps its placeholder when the root succeeded.
        assert_eq!(
            converge_root_outcome(
                Ok::<Option<u8>, PatchedInterpolationError>(None),
                WireControl::Continue,
            )
            .ok(),
            Some(None)
        );
        // A root-only decode or assembly failure reaches every rank as the converged message, whatever
        // their local placeholder says.
        for local in [
            Ok::<u8, PatchedInterpolationError>(0),
            Err(PatchedInterpolationError::FanOut {
                message: "local placeholder".to_string(),
            }),
        ] {
            match converge_root_outcome(local, WireControl::Fail("root decode failed".to_string()))
            {
                Err(PatchedInterpolationError::FanOut { message }) => {
                    assert_eq!(message, "root decode failed");
                }
                other => panic!("expected the converged failure, got {other:?}"),
            }
        }
        // A local failure is never masked by an agreeing control.
        assert!(matches!(
            converge_root_outcome(
                Err::<u8, _>(PatchedInterpolationError::FanOut {
                    message: "local".to_string(),
                }),
                WireControl::Continue,
            ),
            Err(PatchedInterpolationError::FanOut { .. })
        ));
        // `Stop` is not a result control.
        assert!(matches!(
            converge_root_outcome(Ok(0_u8), WireControl::Stop),
            Err(PatchedInterpolationError::FanOut { .. })
        ));
    }
}
