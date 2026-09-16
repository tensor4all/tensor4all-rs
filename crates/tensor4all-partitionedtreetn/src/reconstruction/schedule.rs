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
//! The trajectory is deterministic and serial. It applies the complete transform
//! once per input leaf, keeps each region's retained terms as an explicit
//! superposition, and compresses a merged item only against a measured residual
//! that fits that item's share of the global allowance.
//!
//! With [`MergeRefineOptions::target_bond_dim`] unset the trajectory is uniform
//! and exact: every region is refined down to the requested depth and
//! [`MergeRefineReport::error_bound`] stays zero. With a rank goal it becomes
//! adaptive: a region is refined only when its retained terms exceed the goal and
//! the refinement lowers its maximum retained rank, and a merged pair is combined
//! only when the combination pays off, so a rank-one object is not forced into a
//! preset output tiling. Dropping whole items, nonuniform input trees, and
//! multi-coordinate groups remain follow-up work; automatic padding is a separate
//! domain/embedding policy and is never applied silently.

use std::{
    collections::{btree_map::Entry, BTreeMap},
    fmt::Debug,
    hash::Hash,
};

use tensor4all_core::IdxTensor;
use tensor4all_treetn::LinearOperator;

use super::engine::{checked_add, truncate_toward_allowance};
use super::{
    absolute_allowance, finite, invalid, single_term_partition, validate_tolerance,
    ReconstructionTarget, ReconstructionTolerance, SubsetOperatorOptions,
};
use crate::patching::logical_parameter_count;
use crate::{
    DynIndex, PartitionedTreeTN, PartitionedTreeTNError, Projector, Result, SubDomainTreeTN,
};

/// Output-refinement depth, resource policy, and soft rank goal for
/// [`schedule_merge_refine`].
///
/// This type selects no numerical accuracy of its own: [`ReconstructionTolerance`]
/// carries the global L2 allowance, and the rank goal is soft, so it never
/// overrides that allowance.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::reconstruction::MergeRefineOptions;
/// let options = MergeRefineOptions::default();
/// assert_eq!(options.output_depth, None);
/// assert_eq!(options.max_work_items, 4096);
/// assert_eq!(options.max_terms, 1 << 20);
/// assert_eq!(options.target_bond_dim, None);
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
    /// Desired maximum bond dimension per retained term, default `None`.
    /// `None` disables truncation and keeps the trajectory uniform and exact.
    /// `Some(goal)` enables adaptive refinement for every output region whose
    /// retained terms exceed the goal: a region is refined only when that lowers
    /// its maximum retained term rank, and a merged pair is combined only when the
    /// combination pays off, otherwise its operands stay separate terms. A
    /// truncation candidate must also fit the item's share of the global
    /// allowance, so a soft rank goal never forces an accuracy violation.
    /// `Some(0)` is invalid.
    pub target_bond_dim: Option<usize>,
    /// Maximum retained terms across live and finalized regions, default
    /// `1 << 20`. The adaptive merge policy can keep operands separate, so this
    /// bounds that growth and returns a resource-limit error instead of
    /// exceeding the budget.
    pub max_terms: usize,
}

impl Default for MergeRefineOptions {
    fn default() -> Self {
        Self {
            output_depth: None,
            max_work_items: 4096,
            target_bond_dim: None,
            max_terms: 1 << 20,
        }
    }
}

/// Structural counters of one merge-refine schedule run.
///
/// These are schedule-shape counters, not runtime costs: they let a caller or a
/// test verify the level count, the pairwise-addition count, and the live-item
/// trajectory of the uniform reference mode. Bond dimensions, term counts, and
/// stored parameters describe the returned representation. Numerically the
/// trajectory starts from one exact transform per input leaf, so
/// [`Self::error_bound`] measures only truncation accepted against the global
/// allowance; see the module documentation.
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
///     compression_attempts: 0,
///     compressions: 0,
///     work_items_per_level: vec![2, 2],
///     peak_work_items: 2,
///     refined_regions: 1,
///     stopped_regions: 0,
///     region_count: 2,
///     term_count: 2,
///     max_bond_dim: 1,
///     max_transient_bond_dim: 1,
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
    /// Measured compression bound: per-item residuals accumulated by the
    /// triangle inequality within a region and combined by the Euclidean norm
    /// across the disjoint output regions. It is an a posteriori numerical bound
    /// and excludes backend roundoff. It is zero while `target_bond_dim` is
    /// `None`, and it excludes operator-construction error and the application
    /// error of an externally approximated operator.
    pub error_bound: f64,
    /// Number of executed level transitions.
    pub level_count: usize,
    /// Complete-transform applications, one per input leaf.
    pub applied_operator_count: usize,
    /// Pairwise additions performed by level transitions.
    pub additions: usize,
    /// Region restrictions performed by level transitions.
    pub projections: usize,
    /// Merged items that exceeded the soft rank goal and probed truncation.
    pub compression_attempts: usize,
    /// Probes that strictly lowered the bond dimension and fit their share.
    pub compressions: usize,
    /// Live work items after each level, starting with level zero.
    pub work_items_per_level: Vec<usize>,
    /// Largest number of live work items.
    pub peak_work_items: usize,
    /// Output regions refined into child prefix regions.
    pub refined_regions: usize,
    /// Output regions kept unsplit because refinement would not lower their
    /// retained rank. These regions report their terms through
    /// [`MergeRefineResult::regions`], and their terms cover the input prefixes
    /// that were still separate when they stopped.
    pub stopped_regions: usize,
    /// Disjoint output regions in the result.
    pub region_count: usize,
    /// Terms across all regions; terms inside one region overlap and must not be
    /// combined by adding norm squares.
    pub term_count: usize,
    /// Largest retained bond dimension.
    pub max_bond_dim: usize,
    /// Largest bond dimension of a merged sum before any compression.
    pub max_transient_bond_dim: usize,
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
    /// Sum of the terms' deviation bounds by the triangle inequality.
    bound: f64,
}

impl<V> MergeRefineRegion<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Start a region entry with its first retained term and input prefix.
    fn new(output: Projector, input: Projector, term: Term<V>) -> Self {
        Self {
            output,
            inputs: vec![input],
            terms: vec![term.value],
            bound: term.bound,
        }
    }

    /// Add another retained term of the same output region.
    fn push(&mut self, input: Projector, term: Term<V>) {
        self.inputs.push(input);
        self.terms.push(term.value);
        self.bound += term.bound;
    }
}

/// Disjoint output regions of a merge-refine schedule run, with provenance.
///
/// Terms inside one region overlap: they are the contributions of different
/// input dyadic regions, or the retained operands of a merge that did not pay
/// off, to the same output prefix region, so their norm squares must not simply
/// be added. [`Self::items`] exposes each term's input region.
/// [`Self::into_partition`] succeeds only when every region holds one term: the
/// uniform exact trajectory at the full output depth always does, while an
/// adaptive stop or a retained superposition does not.
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
/// `options` selects the output depth, the work limit, and the soft rank goal.
///
/// With [`MergeRefineOptions::target_bond_dim`] unset nothing is truncated and
/// the reported bound is exactly zero. With a goal the trajectory is adaptive: an
/// output region is refined only when its terms exceed the goal and refining
/// lowers its maximum retained rank, and a merged pair is combined into one term
/// only when that strictly lowers the bond dimension, otherwise both operands stay
/// as separate terms of the region's superposition. Each merged item is truncated
/// only toward the goal and only when its measured residual fits an equal share of
/// the allowance, so the reported bound sums measured residuals inside a region,
/// combines disjoint regions by the Euclidean norm, and never exceeds the
/// allowance. A rejected probe costs nothing, and a restriction is nonexpansive,
/// so an inherited bound carries over without consuming budget. Reaching
/// [`MergeRefineOptions::max_terms`] returns a resource-limit error instead of
/// silently relaxing accuracy.
///
/// Output placement follows the subset-operator convention: `r_j` lands on the
/// selected index that carried `k_(d+1-j)`, so a contiguous output prefix fixes
/// the most significant frequency bits without an extra reversal.
///
/// # Errors
/// Returns [`PartitionedTreeTNError::InvalidOptions`] for a non-binary or empty
/// selection, an `output_depth` above the selection depth, a `max_work_items`
/// limit below the input-leaf count, a zero `target_bond_dim`, a preimage that is
/// not exactly the dyadic input leaves, non-finite tolerances, and non-finite or
/// non-positive checked counts; [`PartitionedTreeTNError::ResourceLimit`] when the
/// retained term count exceeds `max_terms`; [`PartitionedTreeTNError::ProjectorMismatch`]
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
/// assert_eq!(result.report().error_bound, 0.0);
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
    let allowance = absolute_allowance(tolerance, scale)?;
    if options.target_bond_dim == Some(0) {
        return Err(invalid("target_bond_dim must be positive when specified"));
    }
    // Conservative l1 allocation: at most `leaves * output_depth` merges can
    // compress, so each receives an equal share of the global allowance. A
    // restriction is nonexpansive and therefore carries no new cost, and a
    // rejected probe costs nothing.
    let merge_slots = leaves
        .checked_mul(output_depth)
        .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)?;
    let share = if merge_slots == 0 {
        0.0
    } else {
        finite(allowance / merge_slots as f64)?
    };

    // Live output regions, keyed by their prefix value, each holding one item per
    // input prefix that has not been merged away yet.
    let mut level: BTreeMap<usize, BTreeMap<usize, Item<V>>> = BTreeMap::new();
    let mut first: BTreeMap<usize, Item<V>> = BTreeMap::new();
    for prepared in images {
        let leaf = geometry.leaf_index(&prepared.source).ok_or_else(|| {
            invalid("every preimage patch must fix all selected indices to one coordinate")
        })?;
        // Level zero: one complete transform per leaf, whole output domain, and
        // no application error beyond backend roundoff.
        first.insert(leaf, Item::single(prepared.image, 0.0));
    }

    let applied_operator_count: usize = first.len();
    let mut counters = MergeCounters {
        peak_work_items: first.len(),
        ..MergeCounters::default()
    };
    let mut work_items_per_level = Vec::with_capacity(output_depth + 1);
    work_items_per_level.push(first.len());
    level.insert(0, first);
    check_term_budget(&level, &[], options.max_terms)?;
    // `finalized` keeps the regions that stopped refining, so their terms count
    // against the same budget as the live ones.
    let mut finalized: Vec<FinalRegion<V>> = Vec::new();
    for level_index in 1..=output_depth {
        let mut next: BTreeMap<usize, BTreeMap<usize, Item<V>>> = BTreeMap::new();
        for (region, items) in std::mem::take(&mut level) {
            // Probe both child regions: restrict first, then merge input siblings.
            let mut children = Vec::with_capacity(2);
            for bit in 0..2usize {
                let child = (region << 1) | bit;
                let projector = geometry.output_projector(level_index, child)?;
                children.push((
                    child,
                    restrict_and_merge(&items, &projector, center, options, share, &mut counters)?,
                ));
            }
            let parent_rank = max_rank(&items);
            let child_rank = children
                .iter()
                .map(|(_, child_items)| max_rank(child_items))
                .max()
                .unwrap_or(0);
            // Refinement is worth its regions only when it lowers the retained
            // rank. Without a rank goal the trajectory stays uniform and exact.
            let refine = match options.target_bond_dim {
                None => true,
                Some(goal) => parent_rank > goal && child_rank < parent_rank,
            };
            if refine {
                counters.refined_regions = checked_add(counters.refined_regions, 1)?;
                for (child, child_items) in children {
                    next.insert(child, child_items);
                }
            } else {
                counters.stopped_regions = checked_add(counters.stopped_regions, 1)?;
                // The stopped region keeps the input prefixes it held before the
                // discarded probe, so its terms stay smaller than the probe sum.
                finalized.push((level_index - 1, region, level_index - 1, items));
            }
        }
        let live_items: usize = next.values().map(|items| items.len()).sum();
        // INVARIANT: a level holds at most `2^d` items (`#A * #B = 2^d`), which the
        // upfront work limit already covers, so no second item check is needed.
        counters.peak_work_items = counters.peak_work_items.max(live_items);
        work_items_per_level.push(live_items);
        level = next;
        check_term_budget(&level, &finalized, options.max_terms)?;
    }
    // Regions that survived the last level keep the input prefixes they hold at
    // that depth; a stopped region keeps the prefixes it had before its discarded
    // probe. Either way the caller consumes their terms through `regions()`.
    finalized.extend(
        level
            .into_iter()
            .map(|(region, items)| (output_depth, region, output_depth, items)),
    );

    let mut max_bond_dim = 0usize;
    let mut logical_parameters = 0usize;
    // Group retained terms by output region: terms of one output prefix region
    // overlap, so they stay in one region entry that `regions()` exposes as a
    // superposition and `into_partition()` rejects.
    let mut grouped: BTreeMap<(usize, usize), MergeRefineRegion<V>> = BTreeMap::new();
    for (output_level, region, input_level, items) in finalized {
        let output = geometry.output_projector(output_level, region)?;
        for (input_prefix, item) in items {
            let input = geometry.input_projector(input_level, input_prefix)?;
            for term in item.terms {
                max_bond_dim = max_bond_dim.max(term.value.max_bond_dim());
                logical_parameters = logical_parameters
                    .checked_add(logical_parameter_count(&term.value)?)
                    .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)?;
                match grouped.entry((output_level, region)) {
                    Entry::Vacant(slot) => {
                        slot.insert(MergeRefineRegion::new(output.clone(), input.clone(), term));
                    }
                    Entry::Occupied(mut slot) => slot.get_mut().push(input.clone(), term),
                }
            }
        }
    }
    let regions: Vec<MergeRefineRegion<V>> = grouped.into_values().collect();
    let term_count = regions.iter().map(|region| region.terms.len()).sum();
    // Region bounds add by the triangle inequality because terms inside a region
    // may overlap; disjoint regions combine by the Euclidean norm.
    let error_bound = regions
        .iter()
        .try_fold(0.0_f64, |total, region| finite(total.hypot(region.bound)))?;
    let reference_scale = scale;
    let report = MergeRefineReport {
        reference_scale,
        absolute_tolerance: allowance,
        error_bound,
        level_count: output_depth,
        applied_operator_count,
        additions: counters.additions,
        projections: counters.projections,
        compression_attempts: counters.compression_attempts,
        compressions: counters.compressions,
        work_items_per_level,
        peak_work_items: counters.peak_work_items,
        refined_regions: counters.refined_regions,
        stopped_regions: counters.stopped_regions,
        region_count: regions.len(),
        term_count,
        max_bond_dim,
        max_transient_bond_dim: counters.max_transient_bond_dim,
        logical_parameters,
    };
    Ok(MergeRefineResult { regions, report })
}

/// One retained term: `P_B F P_A w` up to a measured deviation bound.
#[derive(Debug, Clone)]
struct Term<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    value: SubDomainTreeTN<V>,
    /// Measured bound on the deviation of `value` from the exact object.
    bound: f64,
}

/// A finalized output region: its output level, output region key, the input
/// level of its items, and the items themselves.
type FinalRegion<V> = (usize, usize, usize, BTreeMap<usize, Item<V>>);

/// One work item: the superposition of its terms for one input prefix region.
#[derive(Debug, Clone)]
struct Item<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    terms: Vec<Term<V>>,
}

impl<V> Item<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    fn single(value: SubDomainTreeTN<V>, bound: f64) -> Self {
        Self {
            terms: vec![Term { value, bound }],
        }
    }

    /// Largest retained bond dimension over the item's terms.
    fn max_rank(&self) -> usize {
        self.terms
            .iter()
            .map(|term| term.value.max_bond_dim())
            .max()
            .unwrap_or(0)
    }

    /// Restrict every term to `projector`, dropping terms that vanish there.
    fn restrict(&self, projector: &Projector) -> Result<Self> {
        let mut terms = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            // Restriction is nonexpansive, so an inherited bound carries over
            // unchanged and consumes no budget.
            if let Some(value) = term.value.project(projector)? {
                terms.push(Term {
                    value,
                    bound: term.bound,
                });
            }
        }
        Ok(Self { terms })
    }
}

/// Largest retained rank over a region's items.
fn max_rank<V>(items: &BTreeMap<usize, Item<V>>) -> usize
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    items.values().map(Item::max_rank).max().unwrap_or(0)
}

/// Restrict a region's items to a child region and merge each input sibling pair.
fn restrict_and_merge<V>(
    items: &BTreeMap<usize, Item<V>>,
    projector: &Projector,
    center: &V,
    options: &MergeRefineOptions,
    share: f64,
    counters: &mut MergeCounters,
) -> Result<BTreeMap<usize, Item<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut children: BTreeMap<usize, Item<V>> = BTreeMap::new();
    for (input_prefix, item) in items {
        let restricted = item.restrict(projector)?;
        match children.entry(input_prefix >> 1) {
            Entry::Vacant(slot) => {
                slot.insert(restricted);
            }
            Entry::Occupied(mut slot) => {
                // Both siblings carry the same prefix and spectator constraints,
                // so strict subdomain addition applies.
                let combined =
                    merge_items(slot.get(), &restricted, center, options, share, counters)?;
                counters.additions = checked_add(counters.additions, 1)?;
                slot.insert(combined);
            }
        }
    }
    let projections = items
        .len()
        .checked_mul(2)
        .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)?;
    counters.projections = checked_add(counters.projections, projections)?;
    Ok(children)
}

/// Combine two sibling contributions to one region.
///
/// Without a rank goal the exact sum is kept as a single term. With a goal, each
/// paired sum is compressed toward the goal first; the single combined term is
/// kept only when it strictly lowers the bond dimension against the naive sum,
/// otherwise the operands stay separate terms so the retained ranks stay small.
fn merge_items<V>(
    left: &Item<V>,
    right: &Item<V>,
    center: &V,
    options: &MergeRefineOptions,
    share: f64,
    counters: &mut MergeCounters,
) -> Result<Item<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut terms = Vec::with_capacity(left.terms.len() + right.terms.len());
    for index in 0..left.terms.len().max(right.terms.len()) {
        match (left.terms.get(index), right.terms.get(index)) {
            (Some(left), Some(right)) => {
                let sum = left.value.add(&right.value)?;
                let inherited = finite(left.bound + right.bound)?;
                counters.max_transient_bond_dim =
                    counters.max_transient_bond_dim.max(sum.max_bond_dim());
                match options.target_bond_dim {
                    None => terms.push(Term {
                        value: sum,
                        bound: inherited,
                    }),
                    Some(goal) => {
                        let (candidate, residual) =
                            compress_item(&sum, center, goal, share, counters)?;
                        let naive_rank =
                            checked_add(left.value.max_bond_dim(), right.value.max_bond_dim())?;
                        if candidate.max_bond_dim() < naive_rank {
                            terms.push(Term {
                                value: candidate,
                                bound: finite(inherited + residual)?,
                            });
                        } else {
                            terms.push(left.clone());
                            terms.push(right.clone());
                        }
                    }
                }
            }
            (Some(only), None) | (None, Some(only)) => terms.push(only.clone()),
            (None, None) => {}
        }
    }
    Ok(Item { terms })
}

/// Fail before the retained term count exceeds the caller's budget.
fn check_term_budget<V>(
    live: &BTreeMap<usize, BTreeMap<usize, Item<V>>>,
    finalized: &[FinalRegion<V>],
    max_terms: usize,
) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    let mut retained = 0usize;
    for items in live
        .values()
        .chain(finalized.iter().map(|(_, _, _, items)| items))
    {
        for item in items.values() {
            retained = checked_add(retained, item.terms.len())?;
        }
    }
    if retained > max_terms {
        return Err(PartitionedTreeTNError::ResourceLimit {
            operation: "merge-refine scheduling",
            limit: "max_terms",
            value: retained,
        });
    }
    Ok(())
}

/// Live schedule counters, so the level loop stays readable.
#[derive(Debug, Default)]
struct MergeCounters {
    additions: usize,
    projections: usize,
    compression_attempts: usize,
    compressions: usize,
    peak_work_items: usize,
    max_transient_bond_dim: usize,
    refined_regions: usize,
    stopped_regions: usize,
}

/// Compress a merged sum toward the soft rank goal within `share`.
///
/// Returns the retained value and its measured residual against `sum`. A
/// truncation candidate is accepted only when it strictly lowers the bond
/// dimension and its measured residual fits `share`; an unaffordable or gainless
/// candidate is rejected at zero cost, so a soft rank goal never forces an
/// accuracy violation.
fn compress_item<V>(
    sum: &SubDomainTreeTN<V>,
    center: &V,
    goal: usize,
    share: f64,
    counters: &mut MergeCounters,
) -> Result<(SubDomainTreeTN<V>, f64)>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if sum.max_bond_dim() <= goal {
        return Ok((sum.clone(), 0.0));
    }
    counters.compression_attempts = checked_add(counters.compression_attempts, 1)?;
    let (candidate, residual) = truncate_toward_allowance(sum, center, share)?;
    if candidate.max_bond_dim() < sum.max_bond_dim() {
        counters.compressions = checked_add(counters.compressions, 1)?;
        Ok((candidate, residual))
    } else {
        Ok((sum.clone(), 0.0))
    }
}
