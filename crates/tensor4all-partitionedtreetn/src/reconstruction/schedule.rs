//! Level-coupled input merging and output refinement for patched subset QFT.
//!
//! The greedy [`reconstruct`](super::reconstruct) engine reduces a whole output
//! domain before it considers splitting it. The Fourier-specific schedule here
//! interleaves the two complementary directions instead: it merges input dyadic
//! siblings one bit at a time while it fixes output prefix bits, and it restricts
//! every contribution to its output region *before* adding, so a sum over the
//! whole output domain is never assembled.
//!
//! A work item represents `P_B F_selected P_A w` for one input dyadic region `A`
//! and one output prefix region `B`. For a uniform binary input partition of
//! depth `d`, level `t` merges the constraint on `k_(d-t+1)` and fixes `r_t`.
//! Input significance is `[k1, ..., kd]` in the operator's node order, while
//! frequency significance `r_j` is carried by the selected position that
//! previously carried `k_(d+1-j)`, matching the subset-operator output placement
//! (no bit-reversal permutation is applied). Spectator indices are unchanged.
//!
//! This module is the deterministic, serial reference trajectory: it applies the
//! complete transform once per input leaf and performs no compression, so the
//! reported [`MergeRefineReport::error_bound`] is zero and the result is exact
//! to backend roundoff. Adaptive rank control, inherited-error accounting, and
//! nonuniform input trees remain follow-up work; automatic padding is a separate
//! domain/embedding policy and is never applied silently.

use std::{
    collections::{btree_map::Entry, BTreeMap},
    fmt::Debug,
    hash::Hash,
};

use tensor4all_core::IdxTensor;
use tensor4all_treetn::LinearOperator;

use super::{
    absolute_allowance, invalid, single_term_partition, validate_tolerance, ReconstructionTarget,
    ReconstructionTolerance, SubsetOperatorOptions,
};
use crate::patching::logical_parameter_count;
use crate::{
    DynIndex, PartitionedTreeTN, PartitionedTreeTNError, Projector, Result, SubDomainTreeTN,
};

/// Output-refinement depth and resource policy for [`schedule_merge_refine`].
///
/// The schedule itself is exact, so this type selects no numerical accuracy:
/// [`ReconstructionTolerance`] carries the global L2 allowance.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::reconstruction::MergeRefineOptions;
/// let options = MergeRefineOptions::default();
/// assert_eq!(options.output_depth, None);
/// assert_eq!(options.max_work_items, 4096);
/// ```
#[derive(Debug, Clone)]
pub struct MergeRefineOptions {
    /// Number of most significant output bits to fix, at most the number of
    /// selected indices. `None` (the default) refines every selected bit and
    /// returns a strict partition of single-output-coordinate regions. A smaller
    /// depth stops after that many level transitions and returns disjoint output
    /// prefix regions whose terms overlap inside a region.
    pub output_depth: Option<usize>,
    /// Maximum live work items per level, default `4096`. Every level of the
    /// uniform trajectory holds exactly `2^selected_indices` items, so a limit
    /// below that is rejected before any operator is applied.
    pub max_work_items: usize,
}

impl Default for MergeRefineOptions {
    fn default() -> Self {
        Self {
            output_depth: None,
            max_work_items: 4096,
        }
    }
}

/// Structural counters of one merge-refine schedule run.
///
/// These are schedule-shape counters, not runtime costs: they let a caller or a
/// test verify the level count, the pairwise-addition count, and the live-item
/// trajectory of the uniform reference mode. Bond dimensions, term counts, and
/// stored parameters describe the returned representation. Numerically, the
/// schedule is exact, so [`Self::error_bound`] is zero; see the module
/// documentation.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::reconstruction::MergeRefineReport;
/// let report = MergeRefineReport {
///     reference_scale: 5.0,
///     absolute_tolerance: 0.0,
///     error_bound: 0.0,
///     level_count: 1,
///     applied_operator_count: 2,
///     additions: 2,
///     projections: 4,
///     work_items_per_level: vec![2, 2],
///     peak_work_items: 2,
///     region_count: 2,
///     term_count: 2,
///     max_bond_dim: 1,
///     logical_parameters: 4,
/// };
/// assert_eq!(report.work_items_per_level.len(), report.level_count + 1);
/// ```
#[derive(Debug, Clone)]
pub struct MergeRefineReport {
    /// Pinned reference scale: the preimage's reference scale times the
    /// operator amplification factor of [`SubsetOperatorOptions`].
    pub reference_scale: f64,
    /// `max(atol, rtol * reference_scale)`.
    pub absolute_tolerance: f64,
    /// Measured compression and drop bound. Zero while the schedule is exact.
    pub error_bound: f64,
    /// Number of executed level transitions.
    pub level_count: usize,
    /// Complete-transform applications, one per input leaf.
    pub applied_operator_count: usize,
    /// Pairwise additions performed by level transitions.
    pub additions: usize,
    /// Region restrictions performed by level transitions.
    pub projections: usize,
    /// Live work items after each level, starting with level zero.
    pub work_items_per_level: Vec<usize>,
    /// Largest number of live work items.
    pub peak_work_items: usize,
    /// Disjoint output regions in the result.
    pub region_count: usize,
    /// Terms across all regions; terms inside one region overlap and must not be
    /// combined by adding norm squares.
    pub term_count: usize,
    /// Largest retained bond dimension.
    pub max_bond_dim: usize,
    /// Stored logical parameters of the retained terms.
    pub logical_parameters: usize,
}

#[derive(Debug, Clone)]
struct MergeRefineRegion<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    output: Projector,
    inputs: Vec<Projector>,
    terms: Vec<SubDomainTreeTN<V>>,
}

impl<V> MergeRefineRegion<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    fn push(&mut self, input: Projector, term: SubDomainTreeTN<V>) {
        self.inputs.push(input);
        self.terms.push(term);
    }
}

/// Disjoint output regions of a merge-refine schedule run, with provenance.
///
/// Terms inside one region overlap: they are the contributions of different
/// input dyadic regions to the same output prefix region, so their norm squares
/// must not simply be added. [`Self::items`] exposes each term's input region;
/// [`Self::into_partition`] succeeds only when every region holds one term,
/// which the fully refined depth always produces.
///
/// # Examples
///
/// ```
/// # use std::collections::HashMap;
/// # use tensor4all_core::{DynIndex, IdxTensor};
/// # use tensor4all_partitionedtreetn::{
/// #     reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN,
/// # };
/// # use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let result = one_bit_identity_result()?;
/// assert_eq!(result.report().region_count, 2);
/// assert_eq!(result.regions().count(), 2);
/// assert_eq!(result.items().count(), 2);
/// assert!((result.into_partition()?.norm()? - 5.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// #
/// # fn one_bit_identity_result(
/// # ) -> Result<MergeRefineResult<usize>, Box<dyn std::error::Error>> {
/// # let site = DynIndex::new_dyn(2);
/// # let tree = TreeTN::from_tensors(
/// #     vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?],
/// #     vec![0usize],
/// # )?;
/// # let leaf = |value: usize| -> Result<SubDomainTreeTN, Box<dyn std::error::Error>> {
/// #     Ok(SubDomainTreeTN::from_treetn(tree.clone())?
/// #         .project(&Projector::from_pairs([(site.clone(), value)])?)?
/// #         .ok_or("zero leaf")?)
/// # };
/// # let preimage = ReconstructionTarget::from_partition(
/// #     &PartitionedTreeTN::from_subdomains(vec![leaf(0)?, leaf(1)?])?,
/// # )?;
/// # let (internal_input, internal_output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
/// # let mpo = TreeTN::from_tensors(
/// #     vec![IdxTensor::from_dense(
/// #         vec![internal_input.clone(), internal_output.clone()],
/// #         vec![1.0, 0.0, 0.0, 1.0],
/// #     )?],
/// #     vec![0usize],
/// # )?;
/// # let mut input = HashMap::new();
/// # input.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_input });
/// # let mut output = HashMap::new();
/// # output.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_output });
/// # Ok(schedule_merge_refine(
/// #     &preimage,
/// #     &0,
/// #     &LinearOperator::new(mpo, input, output),
/// #     std::slice::from_ref(&site),
/// #     &SubsetOperatorOptions { unitary: true },
/// #     ReconstructionTolerance { rtol: 1e-12, atol: 0.0 },
/// #     &MergeRefineOptions::default(),
/// # )?)
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MergeRefineResult<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    regions: Vec<MergeRefineRegion<V>>,
    report: MergeRefineReport,
}

impl<V> MergeRefineResult<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Borrow each output region's projector and its overlapping terms.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use tensor4all_core::{DynIndex, IdxTensor};
    /// # use tensor4all_partitionedtreetn::{
    /// #     reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN,
    /// # };
    /// # use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let result = one_bit_identity_result()?;
    /// assert_eq!(result.regions().count(), 2);
    /// # Ok(())
    /// # }
    /// #
    /// # fn one_bit_identity_result(
    /// # ) -> Result<MergeRefineResult<usize>, Box<dyn std::error::Error>> {
    /// # let site = DynIndex::new_dyn(2);
    /// # let tree = TreeTN::from_tensors(
    /// #     vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?],
    /// #     vec![0usize],
    /// # )?;
    /// # let leaf = |value: usize| -> Result<SubDomainTreeTN, Box<dyn std::error::Error>> {
    /// #     Ok(SubDomainTreeTN::from_treetn(tree.clone())?
    /// #         .project(&Projector::from_pairs([(site.clone(), value)])?)?
    /// #         .ok_or("zero leaf")?)
    /// # };
    /// # let preimage = ReconstructionTarget::from_partition(
    /// #     &PartitionedTreeTN::from_subdomains(vec![leaf(0)?, leaf(1)?])?,
    /// # )?;
    /// # let (internal_input, internal_output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// # let mpo = TreeTN::from_tensors(
    /// #     vec![IdxTensor::from_dense(
    /// #         vec![internal_input.clone(), internal_output.clone()],
    /// #         vec![1.0, 0.0, 0.0, 1.0],
    /// #     )?],
    /// #     vec![0usize],
    /// # )?;
    /// # let mut input = HashMap::new();
    /// # input.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_input });
    /// # let mut output = HashMap::new();
    /// # output.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_output });
    /// # Ok(schedule_merge_refine(
    /// #     &preimage,
    /// #     &0,
    /// #     &LinearOperator::new(mpo, input, output),
    /// #     std::slice::from_ref(&site),
    /// #     &SubsetOperatorOptions { unitary: true },
    /// #     ReconstructionTolerance { rtol: 1e-12, atol: 0.0 },
    /// #     &MergeRefineOptions::default(),
    /// # )?)
    /// # }
    /// ```
    pub fn regions(&self) -> impl Iterator<Item = (&Projector, &[SubDomainTreeTN<V>])> {
        self.regions
            .iter()
            .map(|region| (&region.output, region.terms.as_slice()))
    }

    /// Borrow every retained work item as its input region, output region, and term.
    ///
    /// The input region is the dyadic input prefix the term came from, so a
    /// caller can verify or recombine intermediate `P_B F P_A w` objects without
    /// re-deriving the geometry.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use tensor4all_core::{DynIndex, IdxTensor};
    /// # use tensor4all_partitionedtreetn::{
    /// #     reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN,
    /// # };
    /// # use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let result = one_bit_identity_result()?;
    /// let (inputs, outputs): (Vec<_>, Vec<_>) = result
    ///     .items()
    ///     .map(|(input, output, _)| (input.len(), output.len()))
    ///     .unzip();
    /// // The final level merged the whole input domain and fixed one output bit.
    /// assert_eq!(inputs, vec![0, 0]);
    /// assert_eq!(outputs, vec![1, 1]);
    /// # Ok(())
    /// # }
    /// #
    /// # fn one_bit_identity_result(
    /// # ) -> Result<MergeRefineResult<usize>, Box<dyn std::error::Error>> {
    /// # let site = DynIndex::new_dyn(2);
    /// # let tree = TreeTN::from_tensors(
    /// #     vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?],
    /// #     vec![0usize],
    /// # )?;
    /// # let leaf = |value: usize| -> Result<SubDomainTreeTN, Box<dyn std::error::Error>> {
    /// #     Ok(SubDomainTreeTN::from_treetn(tree.clone())?
    /// #         .project(&Projector::from_pairs([(site.clone(), value)])?)?
    /// #         .ok_or("zero leaf")?)
    /// # };
    /// # let preimage = ReconstructionTarget::from_partition(
    /// #     &PartitionedTreeTN::from_subdomains(vec![leaf(0)?, leaf(1)?])?,
    /// # )?;
    /// # let (internal_input, internal_output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// # let mpo = TreeTN::from_tensors(
    /// #     vec![IdxTensor::from_dense(
    /// #         vec![internal_input.clone(), internal_output.clone()],
    /// #         vec![1.0, 0.0, 0.0, 1.0],
    /// #     )?],
    /// #     vec![0usize],
    /// # )?;
    /// # let mut input = HashMap::new();
    /// # input.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_input });
    /// # let mut output = HashMap::new();
    /// # output.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_output });
    /// # Ok(schedule_merge_refine(
    /// #     &preimage,
    /// #     &0,
    /// #     &LinearOperator::new(mpo, input, output),
    /// #     std::slice::from_ref(&site),
    /// #     &SubsetOperatorOptions { unitary: true },
    /// #     ReconstructionTolerance { rtol: 1e-12, atol: 0.0 },
    /// #     &MergeRefineOptions::default(),
    /// # )?)
    /// # }
    /// ```
    pub fn items(&self) -> impl Iterator<Item = (&Projector, &Projector, &SubDomainTreeTN<V>)> {
        self.regions.iter().flat_map(|region| {
            region
                .inputs
                .iter()
                .zip(region.terms.iter())
                .map(move |(input, term)| (input, &region.output, term))
        })
    }

    /// Borrow the pinned scale, allowance, and structural counters.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use tensor4all_core::{DynIndex, IdxTensor};
    /// # use tensor4all_partitionedtreetn::{
    /// #     reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN,
    /// # };
    /// # use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let result = one_bit_identity_result()?;
    /// let report = result.report();
    /// assert_eq!(report.applied_operator_count, 2);
    /// assert_eq!(report.additions, 2);
    /// assert_eq!(report.error_bound, 0.0);
    /// # Ok(())
    /// # }
    /// #
    /// # fn one_bit_identity_result(
    /// # ) -> Result<MergeRefineResult<usize>, Box<dyn std::error::Error>> {
    /// # let site = DynIndex::new_dyn(2);
    /// # let tree = TreeTN::from_tensors(
    /// #     vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?],
    /// #     vec![0usize],
    /// # )?;
    /// # let leaf = |value: usize| -> Result<SubDomainTreeTN, Box<dyn std::error::Error>> {
    /// #     Ok(SubDomainTreeTN::from_treetn(tree.clone())?
    /// #         .project(&Projector::from_pairs([(site.clone(), value)])?)?
    /// #         .ok_or("zero leaf")?)
    /// # };
    /// # let preimage = ReconstructionTarget::from_partition(
    /// #     &PartitionedTreeTN::from_subdomains(vec![leaf(0)?, leaf(1)?])?,
    /// # )?;
    /// # let (internal_input, internal_output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// # let mpo = TreeTN::from_tensors(
    /// #     vec![IdxTensor::from_dense(
    /// #         vec![internal_input.clone(), internal_output.clone()],
    /// #         vec![1.0, 0.0, 0.0, 1.0],
    /// #     )?],
    /// #     vec![0usize],
    /// # )?;
    /// # let mut input = HashMap::new();
    /// # input.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_input });
    /// # let mut output = HashMap::new();
    /// # output.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_output });
    /// # Ok(schedule_merge_refine(
    /// #     &preimage,
    /// #     &0,
    /// #     &LinearOperator::new(mpo, input, output),
    /// #     std::slice::from_ref(&site),
    /// #     &SubsetOperatorOptions { unitary: true },
    /// #     ReconstructionTolerance { rtol: 1e-12, atol: 0.0 },
    /// #     &MergeRefineOptions::default(),
    /// # )?)
    /// # }
    /// ```
    pub fn report(&self) -> &MergeRefineReport {
        &self.report
    }

    /// Consume a fully refined result as a strict partition.
    ///
    /// # Errors
    /// Returns [`PartitionedTreeTNError::InvalidOptions`] when a region retains
    /// several overlapping terms; inspect [`Self::regions`] instead, or refine to
    /// the full output depth. Propagates partition site, dtype, and projector
    /// validation errors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use tensor4all_core::{DynIndex, IdxTensor};
    /// # use tensor4all_partitionedtreetn::{
    /// #     reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN,
    /// # };
    /// # use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let result = one_bit_identity_result()?;
    /// assert_eq!(result.into_partition()?.len(), 2);
    /// # Ok(())
    /// # }
    /// #
    /// # fn one_bit_identity_result(
    /// # ) -> Result<MergeRefineResult<usize>, Box<dyn std::error::Error>> {
    /// # let site = DynIndex::new_dyn(2);
    /// # let tree = TreeTN::from_tensors(
    /// #     vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?],
    /// #     vec![0usize],
    /// # )?;
    /// # let leaf = |value: usize| -> Result<SubDomainTreeTN, Box<dyn std::error::Error>> {
    /// #     Ok(SubDomainTreeTN::from_treetn(tree.clone())?
    /// #         .project(&Projector::from_pairs([(site.clone(), value)])?)?
    /// #         .ok_or("zero leaf")?)
    /// # };
    /// # let preimage = ReconstructionTarget::from_partition(
    /// #     &PartitionedTreeTN::from_subdomains(vec![leaf(0)?, leaf(1)?])?,
    /// # )?;
    /// # let (internal_input, internal_output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    /// # let mpo = TreeTN::from_tensors(
    /// #     vec![IdxTensor::from_dense(
    /// #         vec![internal_input.clone(), internal_output.clone()],
    /// #         vec![1.0, 0.0, 0.0, 1.0],
    /// #     )?],
    /// #     vec![0usize],
    /// # )?;
    /// # let mut input = HashMap::new();
    /// # input.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_input });
    /// # let mut output = HashMap::new();
    /// # output.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: internal_output });
    /// # Ok(schedule_merge_refine(
    /// #     &preimage,
    /// #     &0,
    /// #     &LinearOperator::new(mpo, input, output),
    /// #     std::slice::from_ref(&site),
    /// #     &SubsetOperatorOptions { unitary: true },
    /// #     ReconstructionTolerance { rtol: 1e-12, atol: 0.0 },
    /// #     &MergeRefineOptions::default(),
    /// # )?)
    /// # }
    /// ```
    pub fn into_partition(self) -> Result<PartitionedTreeTN<V>> {
        single_term_partition(
            self.regions
                .into_iter()
                .map(|region| (region.output, region.terms)),
        )
    }
}

/// Binary prefix geometry of the uniform merge-refine trajectory.
struct ScheduleGeometry {
    /// Selected indices in operator node order. Input significance `k_j` is
    /// `selected[j]`; frequency significance `r_j` sits on `selected[depth - j]`.
    selected: Vec<DynIndex>,
}

impl ScheduleGeometry {
    fn new(selection: &[DynIndex]) -> Result<Self> {
        if selection.is_empty() {
            return Err(invalid(
                "the merge-refine schedule needs at least one selected index",
            ));
        }
        if selection.iter().any(|index| index.dim != 2) {
            return Err(invalid(
                "the merge-refine schedule needs binary selected indices of dimension two",
            ));
        }
        Ok(Self {
            selected: selection.to_vec(),
        })
    }

    fn depth(&self) -> usize {
        self.selected.len()
    }

    /// The input region holding every leaf with the given `k1..k_(depth-level)`
    /// prefix.
    fn input_projector(&self, level: usize, region: usize) -> Result<Projector> {
        let fixed = self.depth() - level;
        let mut projector = Projector::new();
        for position in 0..fixed {
            let bit = (region >> (fixed - 1 - position)) & 1;
            projector.insert(self.selected[position].clone(), bit)?;
        }
        Ok(projector)
    }

    /// The output region fixing `r1..r_level`, mapped onto the selected positions
    /// that carry them.
    fn output_projector(&self, level: usize, region: usize) -> Result<Projector> {
        let mut projector = Projector::new();
        for significance in 1..=level {
            let bit = (region >> (level - significance)) & 1;
            projector.insert(self.selected[self.depth() - significance].clone(), bit)?;
        }
        Ok(projector)
    }

    /// Read the input leaf coordinate from a patch support, or `None` when the
    /// support leaves a selected index free.
    fn leaf_index(&self, projector: &Projector) -> Option<usize> {
        let mut leaf = 0usize;
        for index in &self.selected {
            leaf = (leaf << 1) | projector.get(index)?;
        }
        Some(leaf)
    }

    /// The spectator-only part of a patch support.
    fn spectator_part(&self, projector: &Projector) -> Projector {
        let mut spectator = projector.clone();
        for index in &self.selected {
            spectator.remove(index);
        }
        spectator
    }
}

/// Validate that `preimage` is exactly the dyadic input leaves of `geometry`.
///
/// Runs on patch supports only, so an unsupported partition is rejected before
/// the operator is applied to any patch.
fn validate_leaf_geometry<V>(
    preimage: &ReconstructionTarget<V>,
    geometry: &ScheduleGeometry,
    leaves: usize,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut covered = vec![false; leaves];
    let mut spectators: Option<Projector> = None;
    for projector in preimage.patch_projectors() {
        let leaf = geometry.leaf_index(projector).ok_or_else(|| {
            invalid("every preimage patch must fix all selected indices to one coordinate")
        })?;
        let patch_spectators = geometry.spectator_part(projector);
        match &spectators {
            None => spectators = Some(patch_spectators),
            Some(expected) if *expected != patch_spectators => {
                return Err(PartitionedTreeTNError::ProjectorMismatch);
            }
            Some(_) => {}
        }
        if std::mem::replace(&mut covered[leaf], true) {
            return Err(invalid(
                "input leaves must not repeat a selected-coordinate assignment",
            ));
        }
    }
    if covered.iter().any(|covered| !covered) {
        return Err(invalid(
            "input leaves must cover every selected-coordinate assignment of the binary selection",
        ));
    }
    Ok(())
}

/// Schedule `operator` on `selection` as a level-coupled merge-refine trajectory.
///
/// `preimage` must already be partitioned into the `2^d` dyadic input leaves of
/// the `d` selected binary indices, sharing identical constraints on every
/// spectator index. Level zero applies the complete transform once per leaf;
/// level `t` merges the input siblings by removing the constraint on
/// `k_(d-t+1)` and restricts the sum to each child output prefix fixing `r_t`.
/// Restriction happens before addition, so no sum over the whole output domain
/// is assembled and every computed object is reused level by level.
///
/// `center` is an existing node used by local operator application. `subset`
/// pins the reference scale exactly as in
/// [`ReconstructionTarget::from_subset_operator`]: `unitary = true` uses the
/// preimage scale, otherwise it is multiplied by the selected-space operator's
/// Frobenius norm. `tolerance` supplies `max(atol, rtol * reference_scale)`.
/// `options` selects the output depth and the work limit; no numerical
/// approximation is applied, so the reported bound is zero.
///
/// Output placement follows the subset-operator convention: `r_j` lands on the
/// selected index that carried `k_(d+1-j)`, so a contiguous output prefix fixes
/// the most significant frequency bits without an extra reversal.
///
/// # Errors
/// Returns [`PartitionedTreeTNError::InvalidOptions`] for a non-binary or empty
/// selection, an `output_depth` above the selection depth, a `max_work_items`
/// limit below the input-leaf count or exceeded by a level, a preimage that is
/// not exactly the dyadic input leaves, non-finite tolerances, and non-finite or
/// non-positive checked counts; [`PartitionedTreeTNError::ProjectorMismatch`]
/// when leaves disagree on spectator constraints; and the operator-mapping,
/// node-merging, application, projection, addition, and backend errors reported
/// by the underlying TreeTN operations.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let site = DynIndex::new_dyn(2);
/// // The preimage is split into the two dyadic input leaves x = 0 and x = 1.
/// let leaf = |value: usize| -> Result<SubDomainTreeTN, Box<dyn std::error::Error>> {
///     let tree = TreeTN::from_tensors(
///         vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?],
///         vec![0usize],
///     )?;
///     Ok(SubDomainTreeTN::new(tree, Projector::from_pairs([(site.clone(), value)])?)?)
/// };
/// let preimage = ReconstructionTarget::from_partition(
///     &PartitionedTreeTN::from_subdomains(vec![leaf(0)?, leaf(1)?])?)?;
///
/// // A 2x2 identity operator written as a one-node MPO.
/// let internal_input = DynIndex::new_dyn(2);
/// let internal_output = DynIndex::new_dyn(2);
/// let mpo = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(
///         vec![internal_input.clone(), internal_output.clone()],
///         vec![1.0, 0.0, 0.0, 1.0],
///     )?],
///     vec![0usize],
/// )?;
/// let mut input_mapping = HashMap::new();
/// input_mapping.insert(
///     0usize,
///     IndexMapping { true_index: site.clone(), internal_index: internal_input },
/// );
/// let mut output_mapping = HashMap::new();
/// output_mapping.insert(
///     0usize,
///     IndexMapping { true_index: site.clone(), internal_index: internal_output },
/// );
/// let operator = LinearOperator::new(mpo, input_mapping, output_mapping);
///
/// let result = schedule_merge_refine(
///     &preimage,
///     &0,
///     &operator,
///     std::slice::from_ref(&site),
///     &SubsetOperatorOptions { unitary: true },
///     ReconstructionTolerance { rtol: 1e-12, atol: 0.0 },
///     &MergeRefineOptions::default(),
/// )?;
///
/// // One complete transform per leaf, one level, one addition per output child.
/// assert_eq!(result.report().applied_operator_count, 2);
/// assert_eq!(result.report().additions, 2);
/// assert_eq!(result.report().work_items_per_level, vec![2, 2]);
/// assert_eq!(result.report().region_count, 2);
/// assert!((result.report().reference_scale - 5.0).abs() < 1e-12);
/// // The fully refined result is the original state, split by output coordinate.
/// assert!((result.into_partition()?.norm()? - 5.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
pub fn schedule_merge_refine<V>(
    preimage: &ReconstructionTarget<V>,
    center: &V,
    operator: &LinearOperator<IdxTensor, V>,
    selection: &[DynIndex],
    subset: &SubsetOperatorOptions,
    tolerance: ReconstructionTolerance,
    options: &MergeRefineOptions,
) -> Result<MergeRefineResult<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    validate_tolerance(tolerance)?;
    let geometry = ScheduleGeometry::new(selection)?;
    let depth = geometry.depth();
    let shift =
        u32::try_from(depth).map_err(|_| PartitionedTreeTNError::LogicalParameterCountOverflow)?;
    let leaves = 1usize
        .checked_shl(shift)
        .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)?;
    let output_depth = options.output_depth.unwrap_or(depth);
    if output_depth > depth {
        return Err(invalid(
            "output_depth must not exceed the number of selected indices",
        ));
    }
    if leaves > options.max_work_items {
        return Err(invalid(
            "max_work_items must cover every input leaf of the binary selection; raise the limit or reduce the selection",
        ));
    }
    // Validate the dyadic input geometry before applying any operator, so an
    // unsupported partition costs no transform work.
    validate_leaf_geometry(preimage, &geometry, leaves)?;

    let prepared =
        ReconstructionTarget::prepare_subset_images(preimage, center, operator, selection, subset)?;
    let super::target::PreparedImages { images, scale } = prepared;
    let mut level: BTreeMap<(usize, usize), SubDomainTreeTN<V>> = BTreeMap::new();
    for prepared in images {
        let leaf = geometry.leaf_index(&prepared.source).ok_or_else(|| {
            invalid("every preimage patch must fix all selected indices to one coordinate")
        })?;
        // Level zero: one complete transform per leaf, whole output domain.
        level.insert((leaf, 0), prepared.image);
    }

    let applied_operator_count = level.len();
    let mut additions = 0usize;
    let mut projections = 0usize;
    let mut peak_work_items = level.len();
    let mut work_items_per_level = Vec::with_capacity(output_depth + 1);
    work_items_per_level.push(level.len());
    for level_index in 1..=output_depth {
        let mut next: BTreeMap<(usize, usize), SubDomainTreeTN<V>> = BTreeMap::new();
        for ((input_region, output_region), value) in std::mem::take(&mut level) {
            for bit in 0..2usize {
                let child_region = (output_region << 1) | bit;
                // INVARIANT: restrict to the child output region before adding,
                // so a parent sum is never assembled over the whole output domain.
                let projector = geometry.output_projector(level_index, child_region)?;
                projections += 1;
                let Some(restricted) = value.project(&projector)? else {
                    continue;
                };
                match next.entry((input_region >> 1, child_region)) {
                    Entry::Vacant(slot) => {
                        slot.insert(restricted);
                    }
                    Entry::Occupied(mut slot) => {
                        // Both children carry the same prefix and spectator
                        // constraints, so strict subdomain addition applies.
                        let sum = slot.get().add(&restricted)?;
                        slot.insert(sum);
                        additions += 1;
                    }
                }
            }
        }
        let live = next.len();
        if live > options.max_work_items {
            return Err(invalid(
                "the merge-refine schedule exceeded max_work_items; raise the limit or reduce the requested depth",
            ));
        }
        peak_work_items = peak_work_items.max(live);
        work_items_per_level.push(live);
        level = next;
    }

    let mut regions: Vec<MergeRefineRegion<V>> = Vec::new();
    let mut region_positions: BTreeMap<usize, usize> = BTreeMap::new();
    let mut max_bond_dim = 0usize;
    let mut logical_parameters = 0usize;
    for ((input_region, output_region), value) in level {
        max_bond_dim = max_bond_dim.max(value.max_bond_dim());
        logical_parameters = logical_parameters
            .checked_add(logical_parameter_count(&value)?)
            .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)?;
        let input = geometry.input_projector(output_depth, input_region)?;
        match region_positions.get(&output_region) {
            Some(&position) => regions[position].push(input, value),
            None => {
                region_positions.insert(output_region, regions.len());
                // Every term of a region is masked to the same output prefix and
                // the same spectator constraints, so the region projector is the
                // shared term projector; `SubDomainTreeTN::add` enforces that
                // equality for later terms.
                let output = value.projector().clone();
                regions.push(MergeRefineRegion {
                    output,
                    inputs: vec![input],
                    terms: vec![value],
                });
            }
        }
    }
    let term_count = regions.iter().map(|region| region.terms.len()).sum();
    let reference_scale = scale;
    let report = MergeRefineReport {
        reference_scale,
        absolute_tolerance: absolute_allowance(tolerance, reference_scale)?,
        error_bound: 0.0,
        level_count: output_depth,
        applied_operator_count,
        additions,
        projections,
        work_items_per_level,
        peak_work_items,
        region_count: regions.len(),
        term_count,
        max_bond_dim,
        logical_parameters,
    };
    Ok(MergeRefineResult { regions, report })
}
