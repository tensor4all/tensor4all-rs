//! Collections of eagerly projected [`SubDomainTreeTN`] patches.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::{canonical_index_cmp, Projector};
use crate::subdomain_tree_tn::{
    contraction_output_site_indices, ensure_center, ensure_same_dtype, ensure_same_topology,
    ensure_same_tree_structure, sort_by_patch_order, validate_contraction_options,
    validate_contraction_site_assignment, validate_truncation_options, SubDomainTreeTN,
};
use tensor4all_core::DynIndex;
use tensor4all_treetn::{
    contraction::ContractionOptions, ExactSum, SumEngine, TreeTN, TruncationOptions,
};

/// Options of [`PartitionedTreeTN::contract`] and
/// [`PartitionedTreeTN::contract_selected`]: how each patch pair is
/// contracted, and how the contributions to one output projector are summed.
///
/// A plain [`ContractionOptions`] converts into this type with
/// [`SumEngine::Binary`], so `contract(&other, &center, contraction_options)`
/// keeps its meaning.
///
/// # Summing contributions
///
/// Several patch pairs can contribute to the same output projector. Their
/// exact sum is built with `sum_engine` and then truncated once with the
/// truncation settings of `contraction`. The default, [`SumEngine::Binary`],
/// adds the contributions one by one; it assumes few contributions per output
/// projector, where the add chain is cheap. [`SumEngine::Nary`] builds the sum
/// in one pass, which is much faster for many contributions, and
/// [`SumEngine::NaryWithin`] does so only while the estimated size of the
/// built sum fits `limit_bytes`. The estimate is
/// [`ExactSum::estimated_nary_bytes`](tensor4all_treetn::ExactSum::estimated_nary_bytes):
/// for every node, the product of its site dimensions and of its summed bond
/// dimensions, times 16 bytes.
///
/// A `NaryWithin` fallback to the add chain is not reported by the
/// contraction routines. To see which method a given sum takes, measure it
/// with [`TreeTN::exact_sum`] or [`PartitionedTreeTN::to_treetn_with`]. A
/// reporting variant of the routines may be added when a caller needs it.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::PartitionedContractionOptions;
/// use tensor4all_treetn::{contraction::ContractionOptions, SumEngine};
///
/// let default = PartitionedContractionOptions::default();
/// assert_eq!(default.sum_engine, SumEngine::Binary);
/// // A plain `ContractionOptions` means `Binary` as well.
/// let plain = PartitionedContractionOptions::from(ContractionOptions::default());
/// assert_eq!(plain.sum_engine, SumEngine::Binary);
///
/// let nary = PartitionedContractionOptions {
///     sum_engine: SumEngine::Nary,
///     ..ContractionOptions::default().into()
/// };
/// assert_eq!(nary.sum_engine, SumEngine::Nary);
/// ```
#[derive(Debug, Clone, Default)]
pub struct PartitionedContractionOptions {
    /// How each compatible patch pair is contracted, and the truncation of
    /// every summed output projector.
    pub contraction: ContractionOptions,
    /// How the contributions to one output projector are summed. Default:
    /// [`SumEngine::Binary`].
    pub sum_engine: SumEngine,
}

impl From<ContractionOptions> for PartitionedContractionOptions {
    fn from(contraction: ContractionOptions) -> Self {
        Self {
            contraction,
            sum_engine: SumEngine::Binary,
        }
    }
}

/// A patch pair of a partition contraction with its contracted contribution.
pub(crate) type ContractedPair<'a, V> = (
    &'a SubDomainTreeTN<V>,
    &'a SubDomainTreeTN<V>,
    SubDomainTreeTN<V>,
);

/// A compatible pair and its predicted output projector, without contraction.
pub(crate) type PlannedPair<'a, V> = (&'a SubDomainTreeTN<V>, &'a SubDomainTreeTN<V>, Projector);

/// The patch templates of a contraction with its output site space.
pub(crate) type ContractionPreparation<'a, V> = (
    &'a SubDomainTreeTN<V>,
    &'a SubDomainTreeTN<V>,
    Vec<DynIndex>,
);

#[cfg(test)]
thread_local! {
    /// Numeric pair attempts in partition contraction and adaptive groups;
    /// planning pairs alone does not advance this counter.
    pub(crate) static CONTRACTED_PAIRS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Largest number of atoms that [`PartitionedTreeTN::add`],
/// [`PartitionedTreeTN::add_truncated`], [`PartitionedTreeTN::contract`],
/// [`crate::contract_adaptive`], and [`crate::contract_adaptive_selected`]
/// refine onto automatically.
///
/// The non-adaptive methods refine operands whose patch layouts overlap without being
/// equal onto a layout of disjoint projectors, the atoms: the common
/// refinement for `add`, and for `contract` the grid of the uncontracted
/// indices that the patches fix, whose size is the product of their
/// dimensions. A refined operand holds at most its patch count times the
/// number of atoms. When the atoms would exceed this limit, those methods
/// return [`PartitionedTreeTNError::RefinementTooLarge`] before refining
/// anything.
///
/// Adaptive contraction checks the full candidate **output** grid after
/// selecting original pair keys, counting the product of all fixed output
/// dimensions before removing uncovered cells. This is more conservative than
/// `contract`, which checks each operand's grid separately. Disjoint selected
/// output layouts need no refinement and are not limited by this constant.
/// The bound limits layout expansion, not tensor bytes or the number of input
/// pairs; adaptive rank-triggered splitting has its separate patching policy.
///
/// The explicit [`PartitionedTreeTN::common_refinement`] and
/// [`PartitionedTreeTN::refine_to`] are not limited. To combine operands
/// that need more atoms, refine them explicitly: operands with equal layouts
/// are combined without a further refinement. For adaptive selected calls,
/// explicit operand refinement changes the original keys that `wanted` must
/// name; select the intended refined pair keys explicitly.
///
/// # Where the number comes from
///
/// 4096 is a policy bound, not a measured one. The design issue (#1095) asked for a
/// default small enough to be obviously safe ("thousands of atoms") and for the
/// error to name both numbers; 4096 is the smallest power of two above that
/// range, so the check stays a single comparison. The value says nothing about
/// what the crate can hold: an operand refined onto that many atoms holds 4096
/// patches per patch it started with, which is why the automatic path stops
/// there and the explicit [`PartitionedTreeTN::refine_to`] does not.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
///     AUTOMATIC_REFINEMENT_LIMIT,
/// };
/// use tensor4all_treetn::TreeTN;
///
/// // `{a=0}` against `{}` needs one atom per coordinate of `a`.
/// let a = DynIndex::new_dyn(AUTOMATIC_REFINEMENT_LIMIT + 1);
/// let tree = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![a.clone()], vec![1.0_f64; a.dim])?],
///     vec![0usize],
/// )?;
/// let whole = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(tree.clone())?)?;
/// let part = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(
///     tree,
///     Projector::from_pairs([(a.clone(), 0)])?,
/// )?)?;
/// assert!(matches!(
///     whole.add(&part),
///     Err(PartitionedTreeTNError::RefinementTooLarge { atoms, limit })
///         if atoms == AUTOMATIC_REFINEMENT_LIMIT + 1 && limit == AUTOMATIC_REFINEMENT_LIMIT
/// ));
///
/// // The explicit refinement is not limited.
/// let atoms = PartitionedTreeTN::common_refinement(&[&whole, &part])?;
/// assert_eq!(atoms.len(), AUTOMATIC_REFINEMENT_LIMIT + 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub const AUTOMATIC_REFINEMENT_LIMIT: usize = 4096;

/// A collection of pairwise-disjoint, eagerly masked TreeTN patches.
///
/// All patches in one collection have the same named topology, exact full
/// site-index assignment at every node, and one homogeneous scalar dtype.
/// Projector keys are immutable through this type; use [`Self::insert`] or
/// [`Self::append`] for validated transactional updates.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let data = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0])?],
///     vec![0usize],
/// )?;
/// let patch = SubDomainTreeTN::new(data, Projector::from_pairs([(site, 0)])?)?;
/// let partition = PartitionedTreeTN::from_subdomain(patch)?;
/// assert_eq!(partition.len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct PartitionedTreeTN<V = usize>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    data: HashMap<Projector, SubDomainTreeTN<V>>,
}

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Create an empty partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::PartitionedTreeTN;
    ///
    /// let partition = PartitionedTreeTN::<usize>::new();
    /// assert!(partition.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Return the number of stored patches.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return whether no patches are stored.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over projector keys in unspecified hash-map order.
    pub fn projectors(&self) -> impl Iterator<Item = &Projector> {
        self.data.keys()
    }

    /// Get the patch stored under an exact projector key.
    pub fn get(&self, projector: &Projector) -> Option<&SubDomainTreeTN<V>> {
        self.data.get(projector)
    }

    /// Return whether an exact projector key is present.
    pub fn contains(&self, projector: &Projector) -> bool {
        self.data.contains_key(projector)
    }

    /// Iterate over immutable `(projector, patch)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Projector, &SubDomainTreeTN<V>)> {
        self.data.iter()
    }

    /// Iterate over immutable patches.
    pub fn values(&self) -> impl Iterator<Item = &SubDomainTreeTN<V>> {
        self.data.values()
    }
}

impl<V> PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Construct a partition from validated patches.
    ///
    /// Equal projector keys replace earlier patches in input order. Different
    /// compatible projectors are rejected; all topology, site-space, dtype,
    /// projector, and overlap checks finish before the returned map is built.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`] for different
    /// overlapping keys, [`PartitionedTreeTNError::TopologyMismatch`] or
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] for inconsistent patch
    /// structure, [`PartitionedTreeTNError::DTypeMismatch`] for mixed scalar
    /// dtypes, or the construction/validation source errors from a patch.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let make = |value| -> Result<_, Box<dyn std::error::Error>> {
    ///     let data = TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], vec![value, value])?],
    ///         vec![0usize],
    ///     )?;
    ///     Ok(SubDomainTreeTN::new(
    ///         data,
    ///         Projector::from_pairs([(site.clone(), value as usize)])?,
    ///     )?)
    /// };
    /// let partition = PartitionedTreeTN::from_subdomains(vec![make(0.0)?, make(1.0)?])?;
    /// assert_eq!(partition.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_subdomains(subdomains: Vec<SubDomainTreeTN<V>>) -> Result<Self> {
        let references: Vec<_> = subdomains.iter().collect();
        validate_candidate_set(&references)?;

        let mut data = HashMap::with_capacity(subdomains.len());
        for subdomain in subdomains {
            let projector = subdomain.projector().clone();
            data.remove(&projector);
            data.insert(projector, subdomain);
        }
        Ok(Self { data })
    }

    /// Construct a partition from patches whose projectors are pairwise
    /// disjoint by construction, such as the leaves of one split tree.
    ///
    /// Each patch must already satisfy its own invariants (for example built
    /// with [`SubDomainTreeTN::from_masked_data`]). Topology, site-space, and
    /// dtype consistency are checked against the first patch, which is linear
    /// in the patch count; the quadratic pairwise overlap check of
    /// [`Self::from_subdomains`] is skipped and kept only as a debug assertion.
    pub(crate) fn from_disjoint_subdomains(subdomains: Vec<SubDomainTreeTN<V>>) -> Result<Self> {
        if let Some(template) = subdomains.first() {
            let dtype = template.scalar_kind()?;
            for candidate in subdomains.iter().skip(1) {
                ensure_same_tree_structure(template.data(), candidate.data())?;
                ensure_same_dtype(dtype, candidate.scalar_kind()?)?;
            }
        }
        debug_assert!(
            subdomains.iter().enumerate().all(|(position, left)| {
                subdomains[position + 1..]
                    .iter()
                    .all(|right| !left.projector().is_compatible_with(right.projector()))
            }),
            "from_disjoint_subdomains requires pairwise disjoint projectors"
        );
        let data = subdomains
            .into_iter()
            .map(|subdomain| (subdomain.projector().clone(), subdomain))
            .collect();
        Ok(Self { data })
    }

    /// Consume the partition into its patches, keyed by projector.
    pub(crate) fn into_patches(self) -> HashMap<Projector, SubDomainTreeTN<V>> {
        self.data
    }

    /// Construct a partition containing one patch.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::ProjectorIndexNotFound`] or
    /// [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] for an invalid
    /// stored projector, [`PartitionedTreeTNError::InvalidTopology`],
    /// [`PartitionedTreeTNError::DTypeMismatch`], or
    /// [`PartitionedTreeTNError::UnsupportedDType`] for invalid TreeTN data, and
    /// typed [`PartitionedTreeTNError::TreeTN`],
    /// [`PartitionedTreeTNError::TensorStorage`], or
    /// [`PartitionedTreeTNError::TensorConstruction`] source failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let data = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 0.0])?],
    ///     vec![0usize],
    /// )?;
    /// let partition = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(data)?)?;
    /// assert_eq!(partition.len(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_subdomain(subdomain: SubDomainTreeTN<V>) -> Result<Self> {
        Self::from_subdomains(vec![subdomain])
    }

    /// Insert or exactly replace one patch transactionally.
    ///
    /// An equal projector key replaces both the old key and value. A different
    /// compatible key is rejected. No mutation occurs until the candidate has
    /// passed all partition invariant checks.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`],
    /// [`PartitionedTreeTNError::DTypeMismatch`], or a typed patch validation
    /// error. On error, the partition is unchanged.
    pub fn insert(&mut self, subdomain: SubDomainTreeTN<V>) -> Result<()> {
        self.validate_contents()?;
        subdomain.validate_invariants()?;
        if let Some(template) = self.data.values().next() {
            ensure_same_tree_structure(template.data(), subdomain.data())?;
            ensure_same_dtype(template.scalar_kind()?, subdomain.scalar_kind()?)?;
        }
        self.insert_prevalidated(subdomain)
    }

    /// Insert a subdomain that has already satisfied every partition invariant.
    ///
    /// This private incremental path skips re-validating all existing patches
    /// and the candidate itself, which internal algebra already produces from
    /// validated operands, and checks only projector-key overlap. Repeatedly
    /// calling the public [`Self::insert`] while building a result is cubic in
    /// the patch count because each call rescans the full partition.
    fn insert_prevalidated(&mut self, subdomain: SubDomainTreeTN<V>) -> Result<()> {
        let projector = subdomain.projector().clone();
        for existing in self.data.keys() {
            if existing != &projector && existing.is_compatible_with(&projector) {
                return Err(PartitionedTreeTNError::OverlappingProjectors);
            }
        }
        self.data.remove(&projector);
        self.data.insert(projector, subdomain);
        Ok(())
    }

    /// Append another partition transactionally.
    ///
    /// Equal keys replace existing patches. Different overlapping keys are
    /// rejected, including overlaps between the two inputs. Every candidate is
    /// validated before this partition is mutated.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`] for an invalid
    /// layout, [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] for incompatible patch
    /// metadata. On error, `self` is unchanged.
    pub fn append(&mut self, other: Self) -> Result<()> {
        self.validate_contents()?;
        other.validate_contents()?;

        if let (Some(left), Some(right)) = (self.data.values().next(), other.data.values().next()) {
            ensure_same_tree_structure(left.data(), right.data())?;
            ensure_same_dtype(left.scalar_kind()?, right.scalar_kind()?)?;
        }

        for right_projector in other.data.keys() {
            for left_projector in self.data.keys() {
                if right_projector != left_projector
                    && right_projector.is_compatible_with(left_projector)
                {
                    return Err(PartitionedTreeTNError::OverlappingProjectors);
                }
            }
        }

        for (projector, subdomain) in other.data {
            self.data.remove(&projector);
            self.data.insert(projector, subdomain);
        }
        Ok(())
    }

    /// Append a vector of patches using the same transactional checks as
    /// [`Self::append`].
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] for incompatible patches, plus
    /// the typed projector, topology, dtype, tensor, and TreeTN validation errors
    /// documented by [`Self::from_subdomain`]. `self` is unchanged on failure.
    pub fn append_subdomains(&mut self, subdomains: Vec<SubDomainTreeTN<V>>) -> Result<()> {
        let other = Self::from_subdomains(subdomains)?;
        self.append(other)
    }

    /// Compute the squared Frobenius norm of all eager patches.
    ///
    /// This sums each stored patch's squared norm in canonical patch order, so
    /// the result does not depend on insertion or hash order, without
    /// remasking or dense materialization. The receiver is borrowed immutably;
    /// each patch's norm implementation clones its TreeTN wrapper for
    /// canonicalization. The square of a norm above about `1.3e154` overflows;
    /// use [`Self::norm`] for the norm itself.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] when stored partition metadata
    /// is inconsistent, the validation errors documented by
    /// [`Self::from_subdomain`], or [`PartitionedTreeTNError::TreeTN`] when a
    /// patch norm cannot be evaluated.
    pub fn norm_squared(&self) -> Result<f64> {
        self.validate_contents()?;
        self.ordered_patches()
            .into_iter()
            .try_fold(0.0, |sum, subdomain| Ok(sum + subdomain.norm_squared()?))
    }

    /// Compute the Frobenius norm of all eager patches.
    ///
    /// This combines patch norms with scaled `hypot` accumulation, preserving
    /// representable norms even when their squares overflow or underflow.
    /// Patches use canonical contribution order. The receiver is borrowed
    /// immutably and no full dense tensor is materialized.
    ///
    /// # Errors
    ///
    /// Returns the concrete validation variants documented by
    /// [`Self::norm_squared`], including
    /// [`PartitionedTreeTNError::OverlappingProjectors`],
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`],
    /// [`PartitionedTreeTNError::DTypeMismatch`], and
    /// [`PartitionedTreeTNError::TreeTN`].
    pub fn norm(&self) -> Result<f64> {
        self.validate_contents()?;
        self.ordered_patches()
            .into_iter()
            .try_fold(0.0_f64, |norm, patch| Ok(norm.hypot(patch.norm()?)))
    }

    /// Stored patches in canonical patch order, independent of insertion and
    /// hash order.
    fn ordered_patches(&self) -> Vec<&SubDomainTreeTN<V>> {
        let mut patches: Vec<_> = self.data.values().collect();
        sort_by_patch_order(
            &mut patches,
            |a, b| a.patch_order_key().cmp(b.patch_order_key()),
            |patch| patch.content_digest(),
        );
        patches
    }

    /// Refine every patch onto the layout projectors it is compatible with.
    ///
    /// Each patch is replaced by its exact projections onto the compatible layout
    /// projectors, merged with the restrictions the patch already carries, so a
    /// patch region is covered by the sub-patches of the layout projectors inside
    /// it and no value is truncated. This lets a caller combine partitions with
    /// different patch layouts without projecting patches by hand. Unlike the
    /// automatic refinement in [`Self::add`] and [`Self::contract`], this method
    /// is not limited by [`AUTOMATIC_REFINEMENT_LIMIT`]: the result can hold up
    /// to the patch count times `layout.len()` patches.
    ///
    /// # Arguments
    ///
    /// * `layout` - Pairwise-disjoint projectors to refine onto, normally a common
    ///   refinement of the operands' layouts as returned by
    ///   [`Self::common_refinement`]. Coordinates are zero-based.
    ///
    /// # Returns
    ///
    /// A new partition whose keys are the layout projectors merged with the
    /// restrictions of the patch they were projected from. A layout projector that
    /// conflicts with every patch contributes nothing, a patch region that no
    /// layout projector covers is dropped, and an empty `layout` returns an empty
    /// partition.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`] when `layout` is
    /// not pairwise disjoint (duplicate projectors included),
    /// [`PartitionedTreeTNError::ProjectorIndexNotFound`] when a layout projector
    /// fixes an index this partition's tree does not carry,
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] when a layout index matches a
    /// site identity with a different dimension,
    /// [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] for a coordinate
    /// beyond that index's dimension, and the patch, topology, dtype, and TreeTN
    /// validation errors documented by [`Self::from_subdomain`]. The receiver is
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// // One patch over (k1, k2), refined on k2 into its two sub-patches.
    /// let k1 = DynIndex::new_dyn(2);
    /// let k2 = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(
    ///         vec![k1.clone(), k2.clone()],
    ///         vec![1.0_f64, 2.0, 3.0, 4.0],
    ///     )?],
    ///     vec![0usize],
    /// )?;
    /// let coarse = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(
    ///     tree,
    ///     Projector::from_pairs([(k1.clone(), 0)])?,
    /// )?)?;
    /// let layout = vec![
    ///     Projector::from_pairs([(k1.clone(), 0), (k2.clone(), 0)])?,
    ///     Projector::from_pairs([(k1.clone(), 0), (k2, 1)])?,
    /// ];
    /// let refined = coarse.refine_to(&layout)?;
    /// assert_eq!(refined.len(), 2);
    /// assert!(refined.contains(&layout[1]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn refine_to(&self, layout: &[Projector]) -> Result<Self> {
        self.validate_contents()?;
        if !Projector::are_disjoint(layout) {
            return Err(PartitionedTreeTNError::OverlappingProjectors);
        }
        let mut patches = Vec::new();
        for patch in self.data.values() {
            for projector in layout {
                let Some(projected) = patch.project(projector)? else {
                    continue;
                };
                patches.push(projected);
            }
        }
        Self::from_subdomains(patches)
    }

    /// Return the common refinement of the operands' patch layouts.
    ///
    /// The result is a pairwise-disjoint projector list such that every patch
    /// projector of every operand is exactly a union of atoms, including the part
    /// of one operand's region that the other operands do not cover. Refining each
    /// operand onto these atoms with [`Self::refine_to`] gives operand layouts
    /// that [`Self::add`] can sum patch by patch. The atom count is not limited
    /// by [`AUTOMATIC_REFINEMENT_LIMIT`], so this is the way to combine operands
    /// whose common refinement exceeds it.
    ///
    /// # Arguments
    ///
    /// * `operands` - Partitions whose layouts are refined. They need not share a
    ///   site space; only their projector entries are combined.
    ///
    /// # Returns
    ///
    /// Atoms in canonical projector order, so the result depends neither on the
    /// operand order nor on hash-map iteration order.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::OverlappingProjectors`] when one
    /// operand's own stored layout is not pairwise disjoint,
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] when two operands' projectors
    /// fix the same index identity to coordinates of different dimensions, plus
    /// the patch, dtype, and topology validation errors documented by
    /// [`Self::from_subdomain`]. Only projector entries are combined, so an index
    /// that just one operand projects is not compared with the other operands'
    /// site spaces; [`Self::refine_to`] rejects a layout index that conflicts with
    /// a receiving tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let k1 = DynIndex::new_dyn(2);
    /// let k2 = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(
    ///         vec![k1.clone(), k2.clone()],
    ///         vec![1.0_f64; 4],
    ///     )?],
    ///     vec![0usize],
    /// )?;
    /// let patch = |pairs: Vec<(DynIndex, usize)>|
    ///     -> Result<_, Box<dyn std::error::Error>> {
    ///     Ok(SubDomainTreeTN::new(tree.clone(), Projector::from_pairs(pairs)?)?)
    /// };
    /// let coarse = PartitionedTreeTN::from_subdomains(vec![
    ///     patch(vec![(k1.clone(), 0)])?,
    ///     patch(vec![(k1.clone(), 1)])?,
    /// ])?;
    /// let fine = PartitionedTreeTN::from_subdomains(vec![
    ///     patch(vec![(k1.clone(), 0), (k2.clone(), 0)])?,
    ///     patch(vec![(k1.clone(), 0), (k2.clone(), 1)])?,
    ///     patch(vec![(k1.clone(), 1)])?,
    /// ])?;
    /// let atoms = PartitionedTreeTN::common_refinement(&[&coarse, &fine])?;
    /// // The fine layout already is the common refinement.
    /// assert_eq!(atoms.len(), 3);
    /// assert_eq!(fine.refine_to(&atoms)?.len(), 3);
    /// assert_eq!(coarse.refine_to(&atoms)?.len(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn common_refinement(operands: &[&Self]) -> Result<Vec<Projector>> {
        for operand in operands {
            operand.validate_contents()?;
        }
        // A fixed, canonical patch order makes the result a function of the
        // projector sets rather than of the operand or hash-map iteration order.
        let mut patches: Vec<Projector> = operands
            .iter()
            .flat_map(|operand| operand.data.keys().cloned())
            .collect();
        patches.sort_by(Projector::canonical_cmp);

        // `DynIndex` equality excludes the dimension, so an identity-matching
        // index with two dimensions would otherwise enumerate different
        // complements depending on which operand came first.
        let mut dimensions: HashMap<DynIndex, usize> = HashMap::new();
        for patch in &patches {
            for (index, _) in patch.iter() {
                match dimensions.get(index) {
                    Some(dim) if *dim != index.dim => {
                        return Err(PartitionedTreeTNError::SiteIndexMismatch);
                    }
                    Some(_) => {}
                    None => {
                        dimensions.insert(index.clone(), index.dim);
                    }
                }
            }
        }

        // Accumulate every patch's uncovered part, so the cells always hold a
        // disjoint cover of all operands' regions and no part is added twice.
        let mut cells: Vec<Projector> = Vec::new();
        for patch in &patches {
            let mut pending = vec![patch.clone()];
            for cell in &cells {
                pending = pending
                    .into_iter()
                    .flat_map(|piece| match piece.split_by(cell) {
                        Some(pieces) => pieces.into_iter().skip(1).collect(),
                        None => vec![piece],
                    })
                    .collect();
            }
            cells.extend(pending);
        }

        // Split until every cell lies inside or outside every patch. Splitting
        // fixes at least one more index per cell, so this terminates, and a split
        // keeps the cells disjoint.
        loop {
            let mut split = false;
            let mut next = Vec::with_capacity(cells.len());
            for cell in cells {
                // A split can also change the cell without producing a second
                // piece, when the newly fixed index has dimension one.
                let refinement = patches.iter().find_map(|patch| {
                    let pieces = cell.split_by(patch)?;
                    (pieces.first() != Some(&cell)).then_some(pieces)
                });
                match refinement {
                    Some(pieces) => {
                        split = true;
                        next.extend(pieces);
                    }
                    None => next.push(cell),
                }
            }
            cells = next;
            if !split {
                break;
            }
        }

        cells.sort_by(Projector::canonical_cmp);
        Ok(cells)
    }

    /// Add two partitions patch-wise, exactly.
    ///
    /// Identical projector keys are added with strict [`TreeTN::add`] semantics,
    /// so the bonds of a key present in both operands add and nothing is
    /// truncated. A key present on only one side is copied as the other side's
    /// zero contribution. To truncate the summed keys in the same call, use
    /// [`Self::add_truncated`].
    ///
    /// Every sum here has exactly two terms, one per operand, so this method
    /// takes no [`SumEngine`]: for two terms the binary add and the N-ary
    /// direct sum do the same work. To sum many partitions at once, collect
    /// their patches and use [`Self::to_treetn_with`] or the routines that take
    /// a [`SumEngine`].
    ///
    /// # Different patch layouts
    ///
    /// The operands may have different patch layouts. When a key of one side
    /// overlaps a different key of the other, both operands are first refined
    /// onto [`Self::common_refinement`], which keeps the part of every patch
    /// that the other operand does not cover, so the sum is exact rather than
    /// clipped to the overlap. The refinement masks sites and truncates nothing;
    /// it can increase the patch count. When the layouts are already
    /// equal or disjoint, no refinement runs and the result is unchanged. A
    /// common refinement with more than [`AUTOMATIC_REFINEMENT_LIMIT`] atoms is
    /// rejected before either operand is refined.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] for an empty operand,
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] before patch shortcuts,
    /// [`PartitionedTreeTNError::OverlappingProjectors`] for an operand whose
    /// own stored layout is invalid,
    /// [`PartitionedTreeTNError::RefinementTooLarge`] when the common refinement
    /// has more than [`AUTOMATIC_REFINEMENT_LIMIT`] atoms, or a typed TreeTN
    /// error for addition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let patch = |values: Vec<f64>, coordinate: usize|
    ///     -> Result<_, Box<dyn std::error::Error>> {
    ///     let tree = TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], values)?],
    ///         vec![0usize],
    ///     )?;
    ///     Ok(SubDomainTreeTN::new(
    ///         tree,
    ///         Projector::from_pairs([(site.clone(), coordinate)])?,
    ///     )?)
    /// };
    /// let left = PartitionedTreeTN::from_subdomains(vec![
    ///     patch(vec![1.0, 0.0], 0)?,
    ///     patch(vec![0.0, 2.0], 1)?,
    /// ])?;
    /// let right = PartitionedTreeTN::from_subdomain(patch(vec![10.0, 0.0], 0)?)?;
    ///
    /// let sum = left.add(&right)?;
    /// assert_eq!(sum.len(), 2);
    /// assert_eq!(sum.to_treetn()?.to_dense()?.to_vec::<f64>()?, vec![11.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.add_with(other, None, AUTOMATIC_REFINEMENT_LIMIT)
    }

    /// Add two partitions patch-wise and truncate every key present in both.
    ///
    /// Each key present in both operands is added exactly, as in [`Self::add`],
    /// and the sum is then truncated towards `center` with `options`. A key
    /// present on only one side is copied without truncation.
    ///
    /// The resulting error is not bounded by the cutoff in `options`: that
    /// threshold bounds the discarded weight of one local factorization, and
    /// each key's sum is truncated on its own, over several bonds. Use
    /// `max_bond_dim` when a hard retained-rank bound is required.
    ///
    /// Operands with different patch layouts are first refined onto
    /// [`Self::common_refinement`], as described for [`Self::add`]; the keys
    /// present in both refined operands are then the ones truncated.
    ///
    /// # Arguments
    ///
    /// * `center` - Node of the shared topology that each truncated key's
    ///   TreeTN is canonicalized towards.
    /// * `options` - Truncation applied to each key present in both operands.
    ///   `TruncationOptions::default()` uses the TreeTN default policy;
    ///   `max_bond_dim` caps every truncated key's bonds.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::InvalidOptions`] for an invalid
    /// truncation option before any other check, the errors documented by
    /// [`Self::add`], [`PartitionedTreeTNError::InvalidCenter`] for a center
    /// absent from either operand's topology, or a typed TreeTN error for
    /// truncation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    ///
    /// // A two-node chain whose bond has dimension 2.
    /// let k = DynIndex::new_dyn(2);
    /// let a = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![
    ///         IdxTensor::from_dense(vec![k.clone(), bond.clone()], vec![1.0_f64, 2.0, 3.0, 5.0])?,
    ///         IdxTensor::from_dense(vec![bond, a], vec![1.0_f64, 0.0, 0.0, 1.0])?,
    ///     ],
    ///     vec![0usize, 1],
    /// )?;
    /// let key = Projector::from_pairs([(k, 0)])?;
    /// let partition =
    ///     PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(tree, key.clone())?)?;
    ///
    /// // The exact sum's bond is 2 + 2; the truncated sum's is capped at 1.
    /// let exact = partition.add(&partition)?;
    /// let truncated = partition.add_truncated(
    ///     &partition,
    ///     &0,
    ///     TruncationOptions::default().with_max_bond_dim(1),
    /// )?;
    /// assert_eq!(exact.get(&key).ok_or("missing key")?.data().link_dims(), vec![4]);
    /// assert_eq!(truncated.get(&key).ok_or("missing key")?.data().link_dims(), vec![1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_truncated(
        &self,
        other: &Self,
        center: &V,
        options: TruncationOptions,
    ) -> Result<Self> {
        validate_truncation_options(&options)?;
        self.add_with(other, Some((center, options)), AUTOMATIC_REFINEMENT_LIMIT)
    }

    /// Shared body of [`Self::add`] and [`Self::add_truncated`]: validate the
    /// operands, refine differing layouts onto at most `limit` atoms, and sum
    /// patch-wise, truncating the keys present in both operands when
    /// `truncation` is given.
    fn add_with(
        &self,
        other: &Self,
        truncation: Option<(&V, TruncationOptions)>,
        limit: usize,
    ) -> Result<Self> {
        self.validate_contents()?;
        other.validate_contents()?;
        let (left_template, right_template) = nonempty_templates(self, other)?;
        ensure_same_tree_structure(left_template.data(), right_template.data())?;
        ensure_same_dtype(left_template.scalar_kind()?, right_template.scalar_kind()?)?;
        if let Some((center, _)) = truncation {
            ensure_center(left_template.data(), center)?;
            ensure_center(right_template.data(), center)?;
        }

        if ensure_union_disjoint(&self.data, &other.data).is_ok() {
            return self.add_patchwise(other, truncation);
        }
        let atoms = Self::common_refinement(&[self, other])?;
        ensure_refinement_within(atoms.len(), limit)?;
        let left = self.refine_to(&atoms)?;
        let right = other.refine_to(&atoms)?;
        left.add_patchwise(&right, truncation)
    }

    fn add_patchwise(
        &self,
        other: &Self,
        truncation: Option<(&V, TruncationOptions)>,
    ) -> Result<Self> {
        ensure_union_disjoint(&self.data, &other.data)?;

        let mut result = Self::new();
        for (projector, left) in &self.data {
            if let Some(right) = other.data.get(projector) {
                let mut sum = left.add(right)?;
                if let Some((center, options)) = truncation {
                    sum.truncate(center, options)?;
                }
                result.insert_prevalidated(sum)?;
            } else {
                result.insert_prevalidated(left.clone())?;
            }
        }
        for (projector, right) in &other.data {
            if !self.data.contains_key(projector) {
                result.insert_prevalidated(right.clone())?;
            }
        }
        Ok(result)
    }

    /// Contract two partitions at an explicit center.
    ///
    /// V1 requires the same named topology. Shared contractable site indices
    /// must remain assigned to the same named node; distinct site indices are
    /// retained as external output axes. Compatible patch pairs are contracted
    /// directly from their eager stored data. Projector entries absent from the
    /// output site space are pruned, and duplicate output keys are combined by
    /// strict subdomain addition followed by the requested bond truncation.
    /// To form only some output keys, use [`Self::contract_selected`], which
    /// skips the pairs of every other key.
    ///
    /// # Different patch layouts
    ///
    /// The operands may have different patch layouts. Overlapping output
    /// projectors without equal keys can arise from individually valid, internally
    /// disjoint input partitions, because regions such as `{a=0}` and `{b=0}` are
    /// distinct keys that still intersect in the full site space. When that
    /// happens, both operands are first refined on every uncontracted external
    /// index that any patch of either operand fixes, so every output projector
    /// fixes the same indices and any two are equal or disjoint. A patch count can
    /// grow with the product of those indices' dimensions, and the refinement is
    /// exact: it masks sites and truncates nothing. When the predicted output
    /// projectors are already equal or disjoint, no refinement runs and the result
    /// is unchanged. A grid with more than [`AUTOMATIC_REFINEMENT_LIMIT`] atoms
    /// is rejected before either operand is refined.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] for an empty operand,
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] before pair iteration,
    /// [`PartitionedTreeTNError::InvalidCenter`],
    /// [`PartitionedTreeTNError::InvalidOptions`],
    /// [`PartitionedTreeTNError::OverlappingProjectors`] for an operand whose own
    /// stored layout is invalid, [`PartitionedTreeTNError::RefinementTooLarge`]
    /// when the refinement grid has more than [`AUTOMATIC_REFINEMENT_LIMIT`]
    /// atoms, or a typed TreeTN/backend error. The operation returns a new
    /// partition, so failure cannot mutate either input.
    pub fn contract(
        &self,
        other: &Self,
        center: &V,
        options: impl Into<PartitionedContractionOptions>,
    ) -> Result<Self> {
        let options = options.into();
        validate_contraction_options(&options.contraction)?;
        self.validate_contents()?;
        other.validate_contents()?;
        if let Some((left, right)) =
            self.refined_for_contract(other, center, AUTOMATIC_REFINEMENT_LIMIT)?
        {
            return left.contract_with_selection(&right, center, options, None);
        }
        self.contract_with_selection(other, center, options, None)
    }

    /// Refine both operands if some pair output projector overlaps another
    /// without being equal, and return `None` when refining is unnecessary.
    ///
    /// The predicted output projectors are the patch intersections restricted to
    /// the contraction's output site space, so this decides exactly what the
    /// final output partition validation would reject, without contracting a
    /// pair. The operands are then refined on the uncontracted output indices that
    /// any patch of either operand fixes, each operand restricted to the indices
    /// its own tree carries, which makes every pair output projector fix that same
    /// index set. A grid with more than `limit` atoms is rejected before it is
    /// built.
    fn refined_for_contract(
        &self,
        other: &Self,
        center: &V,
        limit: usize,
    ) -> Result<Option<(Self, Self)>> {
        let Some((left_template, right_template, output_sites)) =
            prepare_contraction(self, other, center)?
        else {
            return Ok(None);
        };

        let mut outputs: Vec<Projector> = Vec::new();
        let mut overlaps = false;
        for left in self.data.values() {
            for right in other.data.values() {
                let Some(merged) = left.projector().intersection(right.projector()) else {
                    continue;
                };
                let output = merged.filter_indices(&output_sites);
                overlaps |= outputs
                    .iter()
                    .any(|seen| seen != &output && seen.is_compatible_with(&output));
                outputs.push(output);
            }
        }
        if !overlaps {
            return Ok(None);
        }

        let fixed = fixed_projector_indices(&outputs);
        let (left_sites, _) = left_template
            .data()
            .all_site_indices()
            .map_err(PartitionedTreeTNError::from)?;
        let (right_sites, _) = right_template
            .data()
            .all_site_indices()
            .map_err(PartitionedTreeTNError::from)?;
        let on = |sites: &[DynIndex]| -> Vec<DynIndex> {
            fixed
                .iter()
                .filter(|index| sites.contains(index))
                .cloned()
                .collect()
        };
        let left_fixed = on(&left_sites);
        let right_fixed = on(&right_sites);
        ensure_refinement_within(grid_len(&left_fixed).max(grid_len(&right_fixed)), limit)?;
        Ok(Some((
            self.refine_to(&value_grid(&left_fixed))?,
            other.refine_to(&value_grid(&right_fixed))?,
        )))
    }

    /// Contract two partitions, forming only the output patches whose
    /// projector is in `wanted`.
    ///
    /// The output projector of a patch pair is known before the pair is
    /// contracted: it is the intersection of the two projectors without the
    /// entries on contracted site indices. A pair whose output projector is not
    /// in `wanted` is skipped and never contracted, so the cost follows the
    /// number of pairs that reach a wanted output rather than all compatible
    /// pairs. Every returned patch equals the patch [`Self::contract`] returns
    /// under the same key when both operate on the same patch pairs: the same
    /// pairs are summed in the same order. Unlike [`Self::contract`], this method
    /// never refines the operands, so for different patch layouts its keys are the
    /// unrefined pair output keys.
    ///
    /// # Arguments
    ///
    /// * `other` - Second operand, with the same named topology.
    /// * `center` - Existing node used as the contraction and truncation center.
    /// * `options` - Non-dense TreeTN contraction method and rank policy.
    /// * `wanted` - Exact output projector keys to form. They are compared by
    ///   equality with the output keys, which carry only uncontracted site
    ///   indices. Duplicates are ignored. A key that no pair produces has no
    ///   output patch; an empty slice returns an empty partition.
    ///
    /// # Returns
    ///
    /// A new partition holding at most one patch per wanted key.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::contract`]:
    /// [`PartitionedTreeTNError::Empty`] for an empty operand,
    /// [`PartitionedTreeTNError::TopologyMismatch`],
    /// [`PartitionedTreeTNError::SiteIndexMismatch`], or
    /// [`PartitionedTreeTNError::DTypeMismatch`] before pair iteration,
    /// [`PartitionedTreeTNError::InvalidCenter`],
    /// [`PartitionedTreeTNError::InvalidOptions`],
    /// [`PartitionedTreeTNError::OverlappingProjectors`] for overlapping wanted
    /// output layouts, or a typed TreeTN/backend error. These checks run even
    /// when no pair is selected. Neither input is mutated.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};
    ///
    /// // Left is a 2x2 matrix over (x, k) split into its two rows; right is
    /// // the all-ones vector over k. Contracting k gives the row sums.
    /// let x = DynIndex::new_dyn(2);
    /// let k = DynIndex::new_dyn(2);
    /// let matrix = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![x.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])?],
    ///     vec![0usize],
    /// )?;
    /// let rows = PartitionedTreeTN::from_subdomains(vec![
    ///     SubDomainTreeTN::new(matrix.clone(), Projector::from_pairs([(x.clone(), 0)])?)?,
    ///     SubDomainTreeTN::new(matrix, Projector::from_pairs([(x.clone(), 1)])?)?,
    /// ])?;
    /// let ones = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(
    ///     TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![k], vec![1.0_f64, 1.0])?], vec![0usize])?,
    /// )?)?;
    ///
    /// let row1 = Projector::from_pairs([(x, 1)])?;
    /// let selected =
    ///     rows.contract_selected(&ones, &0, ContractionOptions::default(), &[row1.clone()])?;
    /// assert_eq!(selected.len(), 1);
    /// // Row 1 of the column-major matrix [[1, 3], [2, 4]] sums to 6; the
    /// // patch is masked to x = 1, so its x = 0 entry is zero.
    /// let patch = selected.get(&row1).ok_or("missing row 1")?;
    /// assert_eq!(patch.data().to_dense()?.to_vec::<f64>()?, vec![0.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contract_selected(
        &self,
        other: &Self,
        center: &V,
        options: impl Into<PartitionedContractionOptions>,
        wanted: &[Projector],
    ) -> Result<Self> {
        let wanted: HashSet<Projector> = wanted.iter().cloned().collect();
        self.contract_with_selection(other, center, options.into(), Some(&wanted))
    }

    fn contract_with_selection(
        &self,
        other: &Self,
        center: &V,
        options: PartitionedContractionOptions,
        wanted: Option<&HashSet<Projector>>,
    ) -> Result<Self> {
        let PartitionedContractionOptions {
            contraction: options,
            sum_engine,
        } = options;
        validate_contraction_options(&options)?;
        self.validate_contents()?;
        other.validate_contents()?;
        nonempty_templates(self, other)?;

        // Group the pairwise contributions by output projector. Pairs come in
        // logical patch-key order, with content digests breaking metadata ties,
        // so the order is independent of the input HashMap iteration order.
        let mut groups: Vec<(Projector, Vec<SubDomainTreeTN<V>>)> = Vec::new();
        let mut positions: HashMap<Projector, usize> = HashMap::new();
        for (_, _, contracted) in self.contract_pairs(other, center, &options, wanted)? {
            let projector = contracted.projector().clone();
            match positions.get(&projector) {
                Some(&position) => groups[position].1.push(contracted),
                None => {
                    positions.insert(projector.clone(), groups.len());
                    groups.push((projector, vec![contracted]));
                }
            }
        }

        // Exact-add each completed group with strict TreeTN addition, then
        // truncate the group once (a single contribution is already the
        // per-pair contraction result). This replaces repeated
        // add-and-truncate, which was order dependent and re-approximated the
        // sum at every contribution.
        sort_by_patch_order(
            &mut groups,
            |(_, left), (_, right)| {
                left.first()
                    .map(SubDomainTreeTN::patch_order_key)
                    .cmp(&right.first().map(SubDomainTreeTN::patch_order_key))
            },
            |(_, contributions)| {
                contributions
                    .first()
                    .and_then(SubDomainTreeTN::content_digest)
            },
        );
        let duplicate_options = truncation_options_from_contraction(&options);
        let mut subdomains = Vec::with_capacity(groups.len());
        for (_, mut contributions) in groups {
            if contributions.len() == 1 {
                subdomains.extend(contributions.pop());
                continue;
            }
            let terms: Vec<_> = contributions.iter().collect();
            let mut combined = SubDomainTreeTN::exact_sum(&terms, sum_engine)?;
            combined.truncate(center, duplicate_options)?;
            subdomains.push(combined);
        }

        // Build and validate the complete result partition once at the end. An
        // overlapping non-identical output projector is rejected here;
        // `contract` avoids that by refining its operands first, and
        // `contract_selected` reports it.
        PartitionedTreeTN::from_subdomains(subdomains)
    }

    /// Contract the compatible patch pairs of `self` and `other` whose output
    /// projector is in `wanted` (every compatible pair when `wanted` is
    /// `None`), in logical left-then-right patch-key order.
    ///
    /// Each pair's output projector is predicted before contraction from the
    /// projector intersection and the output site space, which is shared by
    /// every pair because all patches of a partition have the same site
    /// indices. Returns each contracted pair with its contribution; an empty
    /// operand gives no pairs. The operand checks of [`Self::contract`] run
    /// once on the patch templates, so a selection that skips every pair
    /// still reports an invalid center, topology, dtype, or site assignment.
    pub(crate) fn contract_pairs<'a>(
        &'a self,
        other: &'a Self,
        center: &V,
        options: &ContractionOptions,
        wanted: Option<&HashSet<Projector>>,
    ) -> Result<Vec<ContractedPair<'a, V>>> {
        let mut contracted = Vec::new();
        self.for_each_planned_pair(other, center, wanted, |(left, right, output)| {
            #[cfg(test)]
            CONTRACTED_PAIRS.with(|count| count.set(count.get() + 1));
            let Some(contribution) = left.contract(right, center, options.clone())? else {
                return Ok(());
            };
            debug_assert_eq!(
                contribution.projector(),
                &output,
                "predicted output projector differs from the contracted pair's"
            );
            contracted.push((left, right, contribution));
            Ok(())
        })?;
        Ok(contracted)
    }

    /// Plan compatible pairs in the same logical left-then-right order as
    /// `contract_pairs`, preserving template validation even for an empty selection.
    /// Each pair is passed to `visit` without retaining a second full pair list.
    pub(crate) fn for_each_planned_pair<'a>(
        &'a self,
        other: &'a Self,
        center: &V,
        wanted: Option<&HashSet<Projector>>,
        mut visit: impl FnMut(PlannedPair<'a, V>) -> Result<()>,
    ) -> Result<()> {
        let Some((_, _, output_sites)) = prepare_contraction(self, other, center)? else {
            return Ok(());
        };

        let left_patches = self.ordered_patches();
        let right_patches = other.ordered_patches();
        for left in left_patches {
            for &right in &right_patches {
                let Some(merged) = left.projector().intersection(right.projector()) else {
                    continue;
                };
                let output = merged.filter_indices(&output_sites);
                if wanted.is_some_and(|wanted| !wanted.contains(&output)) {
                    continue;
                }
                visit((left, right, output))?;
            }
        }
        Ok(())
    }

    /// Sum eager patches in logical patch-key order.
    ///
    /// This is [`Self::to_treetn_with`] with [`SumEngine::Binary`]: the patches
    /// are added one by one with strict [`TreeTN::add`]. The full tensor is
    /// never materialized densely.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] when there are no patches,
    /// typed invariant errors for invalid metadata, or [`PartitionedTreeTNError::TreeTN`]
    /// when strict TreeTN addition fails.
    pub fn to_treetn(&self) -> Result<TreeTN<tensor4all_core::IdxTensor, V>> {
        Ok(self.to_treetn_with(SumEngine::Binary)?.network)
    }

    /// Sum eager patches in logical patch-key order, building the sum with the
    /// method `engine` chooses, and report which method ran.
    ///
    /// The sum is exact; its bond dimensions are the sums of the patches' bond
    /// dimensions. Distinct patch keys have stable traversal and direct-sum
    /// construction order. Patches that share the logical index and projector
    /// key are then ordered by a content digest of their payloads, computed
    /// without materializing them and only for such a tie, so an input without
    /// metadata ties pays no linear payload pass. Patches that also tie on
    /// content have unspecified relative order: either their payloads are
    /// numerically identical, so any order gives the same bits, or a payload
    /// could not be fingerprinted without materializing it, or two payloads
    /// collided in the 64-bit digest; the last two leave reproducibility to the
    /// structural key components.
    ///
    /// # Arguments
    /// * `engine` - How the sum is built ([`SumEngine`]). [`SumEngine::Binary`]
    ///   adds the patches one by one, which [`Self::to_treetn`] uses; it assumes
    ///   few patches. [`SumEngine::Nary`] builds the sum in one pass, much
    ///   faster for many patches, and [`SumEngine::NaryWithin`] does so only
    ///   while the estimated size of the sum fits its limit.
    ///
    /// # Returns
    /// The sum ([`ExactSum::network`]), whether the N-ary method built it
    /// ([`ExactSum::used_nary`]), and the size estimate that a
    /// `NaryWithin` limit is compared with ([`ExactSum::estimated_nary_bytes`]).
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] when there are no patches,
    /// typed invariant errors for invalid metadata, or [`PartitionedTreeTNError::TreeTN`]
    /// when building the sum fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
    /// use tensor4all_treetn::{SumEngine, TreeTN};
    ///
    /// // A vector over x split into its two entries.
    /// let x = DynIndex::new_dyn(2);
    /// let vector = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![x.clone()], vec![3.0_f64, 4.0])?],
    ///     vec![0usize],
    /// )?;
    /// let parts = PartitionedTreeTN::from_subdomains(vec![
    ///     SubDomainTreeTN::new(vector.clone(), Projector::from_pairs([(x.clone(), 0)])?)?,
    ///     SubDomainTreeTN::new(vector, Projector::from_pairs([(x, 1)])?)?,
    /// ])?;
    ///
    /// let sum = parts.to_treetn_with(SumEngine::Nary)?;
    /// assert!(sum.used_nary);
    /// // One node of 2 elements, 16 bytes each.
    /// assert_eq!(sum.estimated_nary_bytes, 2 * 16);
    /// assert_eq!(sum.network.to_dense()?.to_vec::<f64>()?, vec![3.0, 4.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_treetn_with(
        &self,
        engine: SumEngine,
    ) -> Result<ExactSum<tensor4all_core::IdxTensor, V>> {
        self.validate_contents()?;
        if self.is_empty() {
            return Err(PartitionedTreeTNError::Empty);
        }

        let mut patches: Vec<_> = self.data.iter().collect();
        sort_by_patch_order(
            &mut patches,
            |(_, left), (_, right)| left.patch_order_key().cmp(right.patch_order_key()),
            |(_, patch)| patch.content_digest(),
        );
        let data: Vec<_> = patches.into_iter().map(|(_, patch)| patch.data()).collect();
        TreeTN::exact_sum(&data, engine).map_err(PartitionedTreeTNError::from)
    }

    pub(crate) fn validate_contents(&self) -> Result<()> {
        let references: Vec<_> = self.data.values().collect();
        validate_candidate_set(&references)
    }
}

fn validate_candidate_set<V>(candidates: &[&SubDomainTreeTN<V>]) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    for candidate in candidates {
        candidate.validate_invariants()?;
    }

    if let Some(template) = candidates.first() {
        for candidate in candidates.iter().skip(1) {
            ensure_same_tree_structure(template.data(), candidate.data())?;
            ensure_same_dtype(template.scalar_kind()?, candidate.scalar_kind()?)?;
        }
    }

    for (position, left) in candidates.iter().enumerate() {
        for right in candidates.iter().skip(position + 1) {
            if left.projector() != right.projector()
                && left.projector().is_compatible_with(right.projector())
            {
                return Err(PartitionedTreeTNError::OverlappingProjectors);
            }
        }
    }
    Ok(())
}

fn nonempty_templates<'a, V>(
    left: &'a PartitionedTreeTN<V>,
    right: &'a PartitionedTreeTN<V>,
) -> Result<(&'a SubDomainTreeTN<V>, &'a SubDomainTreeTN<V>)>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let left_template = left
        .data
        .values()
        .next()
        .ok_or(PartitionedTreeTNError::Empty)?;
    let right_template = right
        .data
        .values()
        .next()
        .ok_or(PartitionedTreeTNError::Empty)?;
    Ok((left_template, right_template))
}

/// Run the pair-independent contraction checks and return the patch templates
/// with the contraction's output site space, or `None` for an empty operand.
fn prepare_contraction<'a, V>(
    left: &'a PartitionedTreeTN<V>,
    right: &'a PartitionedTreeTN<V>,
    center: &V,
) -> Result<Option<ContractionPreparation<'a, V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (Some(left_template), Some(right_template)) =
        (left.data.values().next(), right.data.values().next())
    else {
        return Ok(None);
    };
    ensure_same_topology(left_template.data(), right_template.data())?;
    ensure_same_dtype(left_template.scalar_kind()?, right_template.scalar_kind()?)?;
    validate_contraction_site_assignment(left_template.data(), right_template.data())?;
    ensure_center(left_template.data(), center)?;
    ensure_center(right_template.data(), center)?;
    let output_sites =
        contraction_output_site_indices(left_template.data(), right_template.data())?;
    Ok(Some((left_template, right_template, output_sites)))
}

/// Every projector that fixes each index of `indices` to every coordinate.
/// Number of projectors [`value_grid`] returns for `indices`, saturating at
/// `usize::MAX`.
fn grid_len(indices: &[DynIndex]) -> usize {
    indices
        .iter()
        .fold(1usize, |len, index| len.saturating_mul(index.dim))
}

fn ensure_refinement_within(atoms: usize, limit: usize) -> Result<()> {
    if atoms > limit {
        return Err(PartitionedTreeTNError::RefinementTooLarge { atoms, limit });
    }
    Ok(())
}

/// Collect fixed full index identities in the shared refinement-grid order.
pub(crate) fn fixed_projector_indices(projectors: &[Projector]) -> Vec<DynIndex> {
    let mut fixed = Vec::new();
    for projector in projectors {
        for index in projector.projected_indices() {
            if !fixed.contains(index) {
                fixed.push(index.clone());
            }
        }
    }
    fixed.sort_by(canonical_index_cmp);
    fixed
}

/// Check the entire candidate grid before allocating any projector cells.
pub(crate) fn bounded_value_grid(indices: &[DynIndex], limit: usize) -> Result<Vec<Projector>> {
    ensure_refinement_within(grid_len(indices), limit)?;
    Ok(value_grid(indices))
}

fn value_grid(indices: &[DynIndex]) -> Vec<Projector> {
    let mut grid = vec![Projector::new()];
    for index in indices {
        let mut next = Vec::with_capacity(grid.len() * index.dim);
        for projector in &grid {
            for value in 0..index.dim {
                next.push(Projector::from_entries(
                    projector
                        .iter()
                        .map(|(fixed, &coordinate)| (fixed.clone(), coordinate))
                        .chain([(index.clone(), value)]),
                ));
            }
        }
        grid = next;
    }
    grid
}

fn ensure_union_disjoint<V>(
    left: &HashMap<Projector, SubDomainTreeTN<V>>,
    right: &HashMap<Projector, SubDomainTreeTN<V>>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    for left_projector in left.keys() {
        for right_projector in right.keys() {
            if left_projector != right_projector
                && left_projector.is_compatible_with(right_projector)
            {
                return Err(PartitionedTreeTNError::OverlappingProjectors);
            }
        }
    }
    Ok(())
}

fn truncation_options_from_contraction(options: &ContractionOptions) -> TruncationOptions {
    let mut truncation = TruncationOptions::default();
    if let Some(max_bond_dim) = options.max_bond_dim {
        truncation = truncation.with_max_bond_dim(max_bond_dim);
    }
    if let Some(policy) = options.svd_policy {
        truncation = truncation.with_svd_policy(policy);
    }
    truncation
}

impl<V> IntoIterator for PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    type Item = (Projector, SubDomainTreeTN<V>);
    type IntoIter = std::collections::hash_map::IntoIter<Projector, SubDomainTreeTN<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, V> IntoIterator for &'a PartitionedTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    type Item = (&'a Projector, &'a SubDomainTreeTN<V>);
    type IntoIter = std::collections::hash_map::Iter<'a, Projector, SubDomainTreeTN<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests;
