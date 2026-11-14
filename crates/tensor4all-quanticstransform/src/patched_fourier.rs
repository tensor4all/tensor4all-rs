//! Fourier-specific preparation for the lazy merge-refine scheduler.

use crate::multiaxis_fourier::invalid;
use crate::{quantics_fourier_operator_multiaxis, AxisBit, FourierOptions, QuanticsTransformError};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};
use tensor4all_core::{DynIndex, IndexLike};
use tensor4all_partitionedtreetn::{
    reconstruction::{
        schedule_merge_refine, CoordinateGroup, MergeRefineOptions, MergeRefineResult,
        ReconstructionTarget, ReconstructionTolerance, SubsetOperatorOptions,
    },
    PartitionedTreeTN,
};
use tensor4all_treetn::SiteIndexNetwork;

/// Reconstruction accuracy and schedule policy for [`patched_fourier_transform`].
///
/// Unlike [`FourierOptions`], these options govern reconstruction of an already
/// constructed operator. Defaults use relative tolerance `1e-6`, complete input
/// coverage, and an exact full-depth schedule with no rank goal. Exact output
/// cells and intermediate bonds can grow substantially with bit count. Set
/// `merge_refine.target_bond_dim` to request adaptive compression; the goal is
/// soft and never overrides the error allowance or resource limits.
///
/// # Examples
/// ```
/// use tensor4all_quanticstransform::PatchedFourierOptions;
/// let options = PatchedFourierOptions::<usize>::default();
/// assert_eq!(options.tolerance.rtol, 1e-6);
/// assert_eq!(options.merge_refine.target_bond_dim, None);
/// assert!(options.merge_refine.coordinate_groups.is_none());
/// ```
#[derive(Debug, Clone)]
pub struct PatchedFourierOptions<V> {
    /// L2 reconstruction allowance `max(atol, rtol * reference_scale)`.
    /// Excludes operator construction error and backend roundoff.
    pub tolerance: ReconstructionTolerance,
    /// Scheduler controls, including coverage, output regions/depth, rank goal
    /// and work/term limits. Leave `coordinate_groups` unset: axes determine it.
    /// Without a rank goal and with the default exact `application`, no
    /// approximation is made. `merge_refine.application` also selects a
    /// measured or a certified truncating application of the operator.
    pub merge_refine: MergeRefineOptions<V>,
}

impl<V> Default for PatchedFourierOptions<V> {
    fn default() -> Self {
        Self {
            tolerance: ReconstructionTolerance::default(),
            merge_refine: MergeRefineOptions::default(),
        }
    }
}

/// Transform selected quantics axes while retaining lazy output-region terms.
///
/// `axes[a]` lists the full binary state indices of axis `a` in MSB-first input
/// order, with at least two bits. All indices must be distinct and lie on one
/// path of the state's topology; spectators, including those on selected nodes,
/// keep their identities. `center` must name an existing state node.
/// [`PartitionedTreeTN`] validates that all patches share the same named topology,
/// full site-index placement and dimensions at construction and mutation. The
/// front end therefore reads this common topology from any one patch.
///
/// The front end derives physical path order and coordinate groups, builds the
/// exact layout product of the per-axis operators specified by `options` (see
/// [`quantics_fourier_operator_multiaxis`]), and delegates to the merge-refine scheduler. It never globally sums patches.
/// A region can retain several terms, so `into_partition()` can fail. Requested
/// `output_regions` select intersecting schedule cells, which can be coarser
/// than the request. See [`MergeRefineOptions`] for limits and coverage contracts.
///
/// Each axis's output bits occupy its input indices in reversed significance.
/// To undo a normalized forward transform, reverse **each axis's index vector**
/// and use [`FourierOptions::inverse`]. Without normalization, the round trip
/// instead multiplies by the product of the axis lengths (`2^total_bits`).
///
/// `options.normalize` selects the scheduler's unitary scaling convention.
/// `report().error_bound` covers reconstruction of the constructed operator;
/// it excludes its Fourier approximation error and floating-point roundoff.
/// `options.max_bond_dim` limits per-axis construction, not exact layout
/// restructuring. No total exact-DFT error bound is promised.
///
/// # Input partition
///
/// The patches of `state` must form dyadic input leaves of the selected bits:
/// - each patch fixes a contiguous most-significant prefix of every axis and
///   leaves the remaining bits of that axis free;
/// - every patch fixes **the same coordinates on the spectator indices**, the
///   site indices outside `axes`, or none of them.
///
/// The second condition fails for most partitions from the patched producers
/// (patched interpolation, elementwise products, adaptive contraction) once
/// their patch order includes a spectator index. Partition such inputs on the
/// selected indices only. Otherwise refine every patch to a common spectator
/// layout with [`PartitionedTreeTN::refine_to`], split the patches by their
/// spectator constraint, and transform each group separately. Every output
/// term keeps its group's spectator constraint, and the group results sum to
/// the transform of the whole input.
///
/// # Errors
/// Returns [`QuanticsTransformError::InvalidConfiguration`] for empty/short
/// axes, duplicate/unknown/nonbinary indices, a missing center, an empty state,
/// branching selections, or caller-supplied coordinate groups. Construction and
/// restructuring failures retain their operation source; scheduler validation,
/// coverage, resource and backend failures are preserved by
/// [`QuanticsTransformError::Reconstruction`], including repair guidance. A
/// partition whose patches disagree on spectator constraints is reported there
/// as [`PartitionedTreeTNError::SpectatorMismatch`](tensor4all_partitionedtreetn::PartitionedTreeTNError::SpectatorMismatch).
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_partitionedtreetn::{PartitionedTreeTN, SubDomainTreeTN};
/// use tensor4all_quanticstransform::{patched_fourier_transform, FourierOptions, PatchedFourierOptions};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let bits = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
/// // Both selected bits share a named node. Flat payloads are column-major.
/// let tree = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(bits.clone(), vec![1.0; 4])?], vec!["site"])?;
/// let state = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(tree)?)?;
/// let mut policy = PatchedFourierOptions::default();
/// policy.merge_refine.target_bond_dim = Some(4);
/// policy.merge_refine.output_depth = Some(2);
/// let result = patched_fourier_transform(&state, &"site", &[bits],
///     FourierOptions::forward(), &policy)?;
/// // A normalized four-point DFT maps the constant function to 2 at zero.
/// let partition = result.into_partition()?;
/// assert!((partition.norm()? - 2.0).abs() < 1e-10);
/// # Ok(())
/// # }
/// ```
pub fn patched_fourier_transform<V>(
    state: &PartitionedTreeTN<V>,
    center: &V,
    axes: &[Vec<DynIndex>],
    options: FourierOptions,
    patched: &PatchedFourierOptions<V>,
) -> Result<MergeRefineResult<V>, QuanticsTransformError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if patched.merge_refine.coordinate_groups.is_some() {
        return Err(invalid(
            "coordinate_groups is derived from Fourier axes; leave it unset",
        ));
    }
    let patch = state.values().next().ok_or_else(|| {
        invalid(
        "patched Fourier transform needs a state with site indices; provide a nonempty partition")
    })?;
    let network = patch.data().site_index_network();
    if !network.has_node(center) {
        return Err(invalid(
            "Fourier center is absent from the state; choose an existing node",
        ));
    }
    let (selection, layout) = path_layout(network, axes)?;
    let operator = quantics_fourier_operator_multiaxis(&layout, options.clone())?;
    let preimage = ReconstructionTarget::from_partition(state)?;
    let mut schedule = patched.merge_refine.clone();
    schedule.coordinate_groups = Some(axes.iter().cloned().map(CoordinateGroup::new).collect());
    Ok(schedule_merge_refine(
        &preimage,
        center,
        &operator,
        &selection,
        &SubsetOperatorOptions {
            unitary: options.normalize,
        },
        patched.tolerance,
        &schedule,
    )?)
}

fn path_layout<V>(
    network: &SiteIndexNetwork<V, DynIndex>,
    axes: &[Vec<DynIndex>],
) -> Result<(Vec<DynIndex>, Vec<AxisBit>), QuanticsTransformError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if axes.is_empty() || axes.iter().any(|axis| axis.len() < 2) {
        return Err(invalid(
            "provide at least one Fourier axis, each with at least two bits",
        ));
    }
    let mut seen = HashSet::new();
    let mut by_owner: HashMap<V, Vec<(DynIndex, AxisBit)>> = HashMap::new();
    let mut terminals = HashSet::new();
    for (axis, indices) in axes.iter().enumerate() {
        for (bit, index) in indices.iter().enumerate() {
            if index.dim() != 2 {
                return Err(invalid(
                    "Fourier indices must be binary; select dimension-two indices",
                ));
            }
            if !seen.insert(index) {
                return Err(invalid(
                    "Fourier axes repeat a full index; select each index exactly once",
                ));
            }
            let owner = network.find_node_by_index(index).ok_or_else(|| {
                invalid("Fourier index is absent from the state; select full state site indices")
            })?;
            terminals.insert(
                network
                    .node_index(owner)
                    .ok_or_else(|| invalid("Fourier index owner is absent"))?,
            );
            by_owner
                .entry(owner.clone())
                .or_default()
                .push((index.clone(), AxisBit { axis, bit }));
        }
    }
    let span = network.steiner_tree_nodes(&terminals);
    let names: HashSet<V> = span
        .iter()
        .filter_map(|&node| network.node_name(node).cloned())
        .collect();
    let mut ends = Vec::new();
    for node in &names {
        let degree = network
            .neighbors(node)
            .filter(|next| names.contains(next))
            .count();
        if degree > 2 {
            return Err(invalid(
                "Fourier selected indices must lie on one tree path; split branching selections",
            ));
        }
        if degree <= 1 {
            ends.push(node.clone());
        }
    }
    ends.sort();
    let mut current = ends.into_iter().next();
    let mut previous = None;
    let mut selection = Vec::with_capacity(seen.len());
    let mut layout = Vec::with_capacity(seen.len());
    while let Some(node) = current {
        if let Some(entries) = by_owner.remove(&node) {
            for (index, bit) in entries {
                selection.push(index);
                layout.push(bit);
            }
        }
        current = network
            .neighbors(&node)
            .find(|next| names.contains(next) && Some(next) != previous.as_ref());
        previous = Some(node);
    }
    if !by_owner.is_empty() {
        return Err(invalid(
            "Fourier selected indices must share one connected tree path",
        ));
    }
    Ok((selection, layout))
}
