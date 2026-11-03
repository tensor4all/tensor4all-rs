//! Eagerly projected TreeTN subdomains.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{PartitionedTreeTNError, Result};
use crate::projector::Projector;
use tensor4all_core::{
    validate_svd_truncation_options, DynIndex, IdxTensor, IndexLike, SvdTruncationOptionsError,
    TagSetLike,
};
use tensor4all_treetn::{
    contraction::{self, ContractionMethod, ContractionOptions},
    SiteIndexNetwork, TreeTN, TruncationOptions,
};

/// A TreeTN together with the projector defining its subdomain.
///
/// The stored TreeTN is always eagerly masked. Every original site axis is
/// retained, and values outside the projector are zero. Rebuilding from local
/// tensors also clears canonical and orthogonality metadata; a subdomain does
/// not store a canonical center.
///
/// `V` names TreeTN nodes. The tensor storage is intentionally fixed to
/// [`IdxTensor`] because projection uses its differentiable `mask_index` API.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
///
/// let site = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0])?;
/// let tree = TreeTN::from_tensors(vec![tensor], vec![0usize])?;
/// let subdomain = SubDomainTreeTN::new(
///     tree,
///     Projector::from_pairs([(site.clone(), 1)])?,
/// )?;
///
/// assert_eq!(subdomain.node_count(), 1);
/// assert_eq!(subdomain.data().tensor(subdomain.data().node_index(&0).unwrap())
///     .unwrap().to_vec::<f64>()?, vec![0.0, 4.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct SubDomainTreeTN<V = usize>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    data: TreeTN<IdxTensor, V>,
    projector: Projector,
    budget_squared: Option<f64>,
    order_key: PatchOrderKey<V>,
}

// This type cannot contain a DynIndex or its allocation ID. Content is not part
// of it: the digest is only needed where two keys tie, and it is computed there.
pub(crate) type PatchOrderKey<V> = (
    Vec<(V, usize, Vec<String>, usize, i64)>,
    Vec<(V, usize)>,
    Vec<(V, usize, usize)>,
);

#[cfg(test)]
thread_local! {
    /// Content digests computed for ordering, so a test can assert that an input
    /// without metadata ties computes none.
    pub(crate) static CONTENT_DIGEST_CALLS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Fixed seed of the patch digest fold.
///
/// It is unrelated to `IdxTensor::content_digest`'s seed: the patch fold only
/// has to fold per-node digests reproducibly.
const PATCH_DIGEST_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// Fold one 64-bit word into the patch digest state.
///
/// The patch digest only has to be reproducible across processes and builds, so
/// it carries its own fixed mix instead of borrowing an unstable `Hasher`. The
/// word is added rather than XORed for the reason `IdxTensor::content_digest`
/// documents: an XOR step is linear over GF(2), so paired bit flips between two
/// inputs cancel.
fn mix_digest(state: u64, word: u64) -> u64 {
    let mixed = state.wrapping_add(word).wrapping_mul(0x0000_0100_0000_01b3);
    mixed ^ (mixed >> 32)
}

/// Content digest of every node tensor, folded in sorted node-name order.
///
/// `names` must be the patch's sorted node names, so two patches the signature
/// component already matches fold the same nodes in the same order.
///
/// Each node contributes its leg roles before its payload: which physical axis
/// is a site index and which is an internal bond, with the bond's labels and
/// dimension. Without that, two networks over the same site indices whose node
/// tensors carry the same buffer under a different leg order would share a
/// digest while being different networks. Bond allocation ids are not hashed.
fn tree_content_digest<V>(data: &TreeTN<IdxTensor, V>, names: &[V]) -> Option<u64>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut state = PATCH_DIGEST_SEED;
    for name in names {
        let node = data.node_index(name)?;
        let tensor = data.tensor(node)?;
        let site_indices = data.node_site_indices(name).unwrap_or_default();
        for index in tensor.indices() {
            let mut tags: Vec<_> = index.tags.iter().collect();
            tags.sort();
            let is_site = u64::from(site_indices.contains(index));
            state = mix_digest(state, is_site);
            state = mix_digest(state, index.dim as u64);
            state = mix_digest(state, index.plev as u64);
            state = mix_digest(state, tags.len() as u64);
            for tag in tags {
                for byte in tag.as_bytes() {
                    state = mix_digest(state, u64::from(*byte));
                }
            }
        }
        state = mix_digest(state, tensor.content_digest()?);
    }
    Some(state)
}

/// Sort patches by their metadata key, then re-order what ties on it by content.
///
/// The first pass sorts on the metadata comparator alone. Every run of two or
/// more items that compare equal is then re-sorted by content digest, so a
/// metadata tie is broken by payload content while a run of one item is left
/// alone.
///
/// The digest is a linear pass over a patch's payload, so this is what keeps
/// ordering free when nothing ties: an input without metadata ties never calls
/// `digest`. Items that also compare equal on content keep their relative order,
/// and that last case is numerically inert for a sum.
pub(crate) fn sort_by_patch_order<T>(
    items: &mut Vec<T>,
    compare: impl Fn(&T, &T) -> Ordering,
    digest: impl Fn(&T) -> Option<u64>,
) {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&left, &right| compare(&items[left], &items[right]));
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && compare(&items[order[end]], &items[order[start]]) == Ordering::Equal
        {
            end += 1;
        }
        if end - start > 1 {
            // One digest per item of the run, and stable, so content-identical
            // items keep the input order.
            order[start..end].sort_by_cached_key(|&index| digest(&items[index]));
        }
        start = end;
    }
    // `order` is a permutation of `0..items.len()`: sorting never drops or
    // duplicates an index.
    let mut slots: Vec<Option<T>> = std::mem::take(items).into_iter().map(Some).collect();
    for index in order {
        items.extend(slots[index].take());
    }
}

/// Build the patch ordering key from a patch's own data and projector.
fn build_order_key<V>(data: &TreeTN<IdxTensor, V>, projector: &Projector) -> PatchOrderKey<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut names = data.node_names();
    names.sort();
    let mut signature = Vec::new();
    let mut labels = Vec::new();
    let mut coordinates = Vec::new();
    for name in &names {
        for (axis, index) in data
            .node_site_indices(name)
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            let mut tags: Vec<_> = index.tags.iter().collect();
            tags.sort();
            signature.push((name.clone(), axis, tags, index.dim, index.plev));
            labels.push((name.clone(), axis));
            if let Some(value) = projector.get(index) {
                coordinates.push((name.clone(), axis, value));
            }
        }
    }
    (signature, labels, coordinates)
}

impl<V> SubDomainTreeTN<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Return the eagerly projected TreeTN by immutable reference.
    pub fn data(&self) -> &TreeTN<IdxTensor, V> {
        &self.data
    }

    /// Return the projector by immutable reference.
    pub fn projector(&self) -> &Projector {
        &self.projector
    }

    /// Consume the subdomain and return its eagerly projected TreeTN.
    pub fn into_data(self) -> TreeTN<IdxTensor, V> {
        self.data
    }

    /// Return the number of TreeTN nodes.
    pub fn node_count(&self) -> usize {
        self.data.node_count()
    }

    /// Return whether the TreeTN has no nodes.
    pub fn is_empty(&self) -> bool {
        self.data.node_count() == 0
    }

    /// Return all full site indices, including every axis on every node.
    ///
    /// Nodes appear in the stored TreeTN's node order and each node's axes in
    /// its tensor's leg order ([`TreeTN::node_site_indices`]), the same order
    /// as the TreeTN's `external_indices()`. Full index equality, rather than
    /// ID-only equality, is used by the stored network and projector.
    pub fn all_indices(&self) -> Vec<DynIndex> {
        self.data
            .node_names()
            .into_iter()
            .flat_map(|name| self.data.node_site_indices(&name).unwrap_or_default())
            .collect()
    }

    /// Return the TreeTN site-index network describing node topology and site axes.
    pub fn site_index_network(&self) -> &SiteIndexNetwork<V, DynIndex> {
        self.data.site_index_network()
    }

    /// Return whether `index` is fixed by this subdomain projector.
    pub fn is_projected_at(&self, index: &DynIndex) -> bool {
        self.projector.is_projected_at(index)
    }

    fn validate_projector(&self, projector: &Projector) -> Result<()>
    where
        V: Ord,
    {
        validate_projector_against_data(&self.data, projector)
    }
}

impl<V> SubDomainTreeTN<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    /// Return the largest internal bond dimension, or one for a bondless tree.
    pub fn max_bond_dim(&self) -> usize {
        self.data.link_dims().into_iter().fold(1, usize::max)
    }

    pub(crate) fn validate_invariants(&self) -> Result<()> {
        validate_data(&self.data)?;
        self.validate_projector(&self.projector)?;
        if let Some(budget_squared) = self.budget_squared {
            validate_budget_squared(budget_squared)?;
        }
        Ok(())
    }

    /// Order patches by index signature, then by their own projector
    /// coordinates in logical labels.
    ///
    /// This is the metadata key alone. Content is not part of it because
    /// fingerprinting a payload costs a linear pass; callers order by content
    /// only inside a metadata tie, through [`sort_by_patch_order`] and
    /// [`Self::content_digest`].
    pub(crate) fn patch_order_key(&self) -> &PatchOrderKey<V> {
        &self.order_key
    }

    /// Content digest of this patch's payloads, for breaking metadata ties.
    ///
    /// Covers every node tensor in sorted node-name order, each preceded by that
    /// node's leg roles and canonical bond labels, and each node's own payload
    /// bit patterns. `None` when a payload cannot be read without materializing
    /// it, in which case the patch keeps its metadata order.
    pub(crate) fn content_digest(&self) -> Option<u64> {
        #[cfg(test)]
        CONTENT_DIGEST_CALLS.with(|count| count.set(count.get() + 1));
        let mut names = self.data.node_names();
        names.sort();
        tree_content_digest(&self.data, &names)
    }

    /// Content digest of this patch followed by `other`, for ordering a patch
    /// pair as one item.
    pub(crate) fn paired_content_digest(&self, other: &Self) -> Option<u64> {
        Some(mix_digest(
            mix_digest(PATCH_DIGEST_SEED, self.content_digest()?),
            other.content_digest()?,
        ))
    }

    pub(crate) fn budget_squared(&self) -> Option<f64> {
        self.budget_squared
    }

    pub(crate) fn with_budget_squared(mut self, budget_squared: f64) -> Result<Self> {
        validate_budget_squared(budget_squared)?;
        self.budget_squared = Some(budget_squared);
        Ok(self)
    }

    /// Drop any inherited absolute budget metadata.
    ///
    /// Used when a stage truncates with a relative policy instead, so a later
    /// consumer cannot mistake a stale absolute budget for the active rule.
    pub(crate) fn without_budget_squared(mut self) -> Self {
        self.budget_squared = None;
        self
    }

    pub(crate) fn scalar_kind(&self) -> Result<Option<ScalarKind>> {
        validate_data(&self.data)
    }

    pub(crate) fn from_masked_data(
        data: TreeTN<IdxTensor, V>,
        projector: Projector,
        budget_squared: Option<f64>,
    ) -> Result<Self> {
        validate_data(&data)?;
        validate_projector_against_data(&data, &projector)?;
        if let Some(budget_squared) = budget_squared {
            validate_budget_squared(budget_squared)?;
        }
        let order_key = build_order_key(&data, &projector);
        debug_assert!(
            !std::any::type_name::<PatchOrderKey<()>>().contains(std::any::type_name::<DynIndex>())
        );
        debug_assert_eq!(
            order_key.2.len(),
            projector.len(),
            "key must use logical coordinates, never DynIndex IDs"
        );
        Ok(Self {
            data,
            projector,
            budget_squared,
            order_key,
        })
    }

    /// Construct an eagerly projected subdomain from a TreeTN and projector.
    ///
    /// The input topology must be one connected acyclic TreeTN. Every
    /// projector index must match a full site-index identity in the TreeTN;
    /// coordinates are checked against the TreeTN dimension, not merely the
    /// dimension carried by the projector key. All nodes must use one
    /// supported homogeneous `IdxTensor` scalar dtype.
    ///
    /// Construction masks each selected local tensor with
    /// [`IdxTensor::mask_index`], retains all site axes, and rebuilds the
    /// TreeTN. It therefore does not materialize the full network and does not
    /// retain canonical metadata from the input.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] for a zero-node input or
    /// [`PartitionedTreeTNError::InvalidTopology`] for a non-tree input,
    /// [`PartitionedTreeTNError::DTypeMismatch`] for mixed
    /// node dtypes, [`PartitionedTreeTNError::ProjectorIndexNotFound`] for an
    /// absent full site identity,
    /// [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] for an
    /// invalid coordinate, or a typed backend/TreeTN error when local masking
    /// or rebuilding fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0])?],
    ///     vec![0usize],
    /// )?;
    /// let subdomain = SubDomainTreeTN::new(
    ///     tree,
    ///     Projector::from_pairs([(site.clone(), 0)])?,
    /// )?;
    /// assert_eq!(subdomain.projector().get(&site), Some(0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(data: TreeTN<IdxTensor, V>, projector: Projector) -> Result<Self> {
        let _ = validate_data(&data)?;
        validate_projector_against_data(&data, &projector)?;
        let data = rebuild_masked_tree(&data, &projector)?;
        Self::from_masked_data(data, projector, None)
    }

    /// Construct an eagerly rebuilt subdomain with an empty projector.
    ///
    /// Rebuilding also clears canonical and orthogonality metadata present on
    /// the input TreeTN.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::Empty`] for a zero-node input or
    /// [`PartitionedTreeTNError::InvalidTopology`] for a non-tree input,
    /// [`PartitionedTreeTNError::DTypeMismatch`] or
    /// [`PartitionedTreeTNError::UnsupportedDType`] for invalid node dtypes,
    /// [`PartitionedTreeTNError::TensorStorage`] or
    /// [`PartitionedTreeTNError::TensorConstruction`] when rebuilding fails,
    /// and [`PartitionedTreeTNError::TreeTN`] for other TreeTN validation errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 3.0])?],
    ///     vec![0usize],
    /// )?;
    /// let subdomain = SubDomainTreeTN::from_treetn(tree)?;
    /// assert!(subdomain.projector().is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_treetn(data: TreeTN<IdxTensor, V>) -> Result<Self> {
        Self::new(data, Projector::new())
    }

    /// Apply compatible new restrictions and return another eagerly masked value.
    ///
    /// Existing restrictions are already represented in `self.data`, so only
    /// newly added projector entries are sent through `mask_index`. A
    /// conflicting projector returns `Ok(None)` without changing `self`.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::ProjectorIndexNotFound`] or
    /// [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] for invalid
    /// restrictions, [`PartitionedTreeTNError::TensorStorage`] or
    /// [`PartitionedTreeTNError::TensorConstruction`] when masking fails, and
    /// [`PartitionedTreeTNError::TreeTN`] when site-space validation or rebuilding
    /// the TreeTN fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
    ///
    /// let left = DynIndex::new_dyn(2);
    /// let right = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(
    ///         vec![left.clone(), right.clone()],
    ///         vec![1.0_f64, 2.0, 3.0, 4.0],
    ///     )?],
    ///     vec![0usize],
    /// )?;
    /// let source = SubDomainTreeTN::new(tree, Projector::from_pairs([(left.clone(), 0)])?)?;
    /// let result = source.project(&Projector::from_pairs([(right.clone(), 1)])?)?;
    /// let result = result.ok_or("conflicting projector")?;
    /// assert_eq!(result.projector().get(&left), Some(0));
    /// assert_eq!(result.projector().get(&right), Some(1));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn project(&self, projector: &Projector) -> Result<Option<Self>> {
        self.validate_projector(projector)?;
        let Some(merged_projector) = self.projector.intersection(projector) else {
            return Ok(None);
        };
        self.validate_projector(&merged_projector)?;

        let mut additions = Projector::new();
        for (index, &value) in projector.iter() {
            if self.projector.get(index) != Some(value) {
                additions.insert(index.clone(), value)?;
            }
        }

        let data = rebuild_masked_tree(&self.data, &additions)?;
        Self::from_masked_data(data, merged_projector, self.budget_squared).map(Some)
    }

    /// Add two subdomains with the same projector using strict TreeTN addition.
    ///
    /// The stored TreeTNs are already eagerly masked, so this method performs no
    /// projector pass. The result retains every site axis and the left budget
    /// metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::ProjectorMismatch`] when the projector
    /// keys differ, [`PartitionedTreeTNError::TopologyMismatch`] or
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] when the TreeTN structures
    /// differ, [`PartitionedTreeTNError::DTypeMismatch`] for mixed scalar dtypes,
    /// or [`PartitionedTreeTNError::TreeTN`] for strict addition failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let make = |values| {
    ///     TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], values)?],
    ///         vec![0usize],
    ///     )
    /// };
    /// let left = SubDomainTreeTN::from_treetn(make(vec![1.0_f64, 2.0])?)?;
    /// let right = SubDomainTreeTN::from_treetn(make(vec![3.0_f64, 4.0])?)?;
    /// let sum = left.add(&right)?;
    /// let node = sum.data().node_index(&0).ok_or("missing node")?;
    /// assert_eq!(sum.data().tensor(node).ok_or("missing tensor")?.to_vec::<f64>()?,
    ///            vec![4.0, 6.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.validate_invariants()?;
        other.validate_invariants()?;
        ensure_same_tree_structure(&self.data, &other.data)?;
        ensure_same_dtype(self.scalar_kind()?, other.scalar_kind()?)?;
        if self.projector != other.projector {
            return Err(PartitionedTreeTNError::ProjectorMismatch);
        }

        let data = self
            .data
            .add(&other.data)
            .map_err(PartitionedTreeTNError::from)?;
        ensure_same_tree_structure(&self.data, &data)?;
        Self::from_masked_data(data, self.projector.clone(), self.budget_squared)
    }

    /// Truncate this eagerly masked subdomain towards an explicit TreeTN center.
    ///
    /// The operation works on a clone and commits only after TreeTN truncation
    /// and invariant validation succeed. It preserves all site axes, the
    /// projector, and the internal squared budget; the center is not stored.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::InvalidCenter`] when `center` is absent,
    /// [`PartitionedTreeTNError::InvalidOptions`] when `options` requests an
    /// invalid bond dimension, [`PartitionedTreeTNError::TopologyMismatch`] or
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] if truncation changes the
    /// site structure, or [`PartitionedTreeTNError::TreeTN`] for canonicalization,
    /// SVD, or backend failures. The receiver is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![3.0_f64, 4.0])?],
    ///     vec!["center".to_string()],
    /// )?;
    /// let mut subdomain = SubDomainTreeTN::from_treetn(tree)?;
    /// subdomain.truncate(&"center".to_string(), TruncationOptions::default())?;
    /// assert_eq!(subdomain.node_count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn truncate(&mut self, center: &V, options: TruncationOptions) -> Result<()> {
        validate_truncation_options(&options)?;
        self.validate_invariants()?;
        ensure_center(&self.data, center)?;

        let original = self.data.clone();
        let mut data = original.clone();
        data.truncate_mut([center.clone()], options)
            .map_err(PartitionedTreeTNError::from)?;
        ensure_same_tree_structure(&original, &data)?;
        validate_data(&data)?;
        validate_projector_against_data(&data, &self.projector)?;
        self.data = data;
        Ok(())
    }

    /// Contract two compatible subdomains at an explicit TreeTN center.
    ///
    /// The operation contracts the already masked stored TreeTNs directly.
    /// Compatible projectors are merged, then entries for site indices absent
    /// from the contracted output are removed. The returned subdomain retains
    /// the left budget metadata and never stores `center`.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::TopologyMismatch`] when named node
    /// topologies differ, [`PartitionedTreeTNError::SiteIndexMismatch`] when a
    /// contractable site index is assigned to different nodes,
    /// [`PartitionedTreeTNError::DTypeMismatch`] for mixed dtypes,
    /// [`PartitionedTreeTNError::InvalidCenter`] for an absent center,
    /// [`PartitionedTreeTNError::InvalidOptions`] for forbidden dense/reference
    /// options, or [`PartitionedTreeTNError::TreeTN`] for contraction failures.
    /// Returns `Ok(None)` when the projector regions conflict.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    /// use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};
    ///
    /// let left_site = DynIndex::new_dyn(2);
    /// let right_site = DynIndex::new_dyn(2);
    /// let left = SubDomainTreeTN::from_treetn(TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![left_site], vec![1.0_f64, 2.0])?],
    ///     vec![0usize],
    /// )?)?;
    /// let right = SubDomainTreeTN::from_treetn(TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![right_site], vec![3.0_f64, 4.0])?],
    ///     vec![0usize],
    /// )?)?;
    /// let result = left.contract(&right, &0, ContractionOptions::default())?;
    /// assert!(result.is_some());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contract(
        &self,
        other: &Self,
        center: &V,
        options: ContractionOptions,
    ) -> Result<Option<Self>> {
        validate_contraction_options(&options)?;
        self.validate_invariants()?;
        other.validate_invariants()?;
        ensure_same_topology(&self.data, &other.data)?;
        ensure_same_dtype(self.scalar_kind()?, other.scalar_kind()?)?;
        validate_contraction_site_assignment(&self.data, &other.data)?;
        ensure_center(&self.data, center)?;
        if !self.projector.is_compatible_with(&other.projector) {
            return Ok(None);
        }

        let merged_projector = self
            .projector
            .intersection(&other.projector)
            .ok_or(PartitionedTreeTNError::ProjectorConflict)?;
        let data = contraction::contract(&self.data, &other.data, center, options)
            .map_err(PartitionedTreeTNError::from)?;
        let (site_indices, _) = data
            .all_site_indices()
            .map_err(PartitionedTreeTNError::from)?;
        let projector = merged_projector.filter_indices(&site_indices);
        Self::from_masked_data(data, projector, self.budget_squared).map(Some)
    }

    /// Compute the Frobenius norm without mutating the stored TreeTN.
    ///
    /// This clones the TreeTN wrapper and allows the clone to canonicalize
    /// internally. The cost is one TreeTN metadata/tensor clone plus the
    /// underlying canonicalization sweep; it does not materialize the full
    /// site tensor.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::TreeTN`] when TreeTN canonicalization
    /// or norm evaluation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![3.0_f64, 4.0])?],
    ///     vec![0usize],
    /// )?;
    /// let subdomain = SubDomainTreeTN::from_treetn(tree)?;
    /// assert!((subdomain.norm()? - 5.0).abs() < 1.0e-12);
    /// assert!(subdomain.data().canonical_region().is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn norm(&self) -> Result<f64> {
        let mut data = self.data.clone();
        data.norm().map_err(PartitionedTreeTNError::from)
    }

    /// Compute the squared Frobenius norm without mutating the stored TreeTN.
    ///
    /// This clones the TreeTN wrapper and lets the clone canonicalize, with the
    /// same linear-network clone and sweep cost as [`Self::norm`]. No full dense
    /// materialization is performed.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::TreeTN`] when TreeTN canonicalization
    /// or norm evaluation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tree = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site], vec![3.0_f64, 4.0])?],
    ///     vec![0usize],
    /// )?;
    /// let subdomain = SubDomainTreeTN::from_treetn(tree)?;
    /// assert!((subdomain.norm_squared()? - 25.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn norm_squared(&self) -> Result<f64> {
        let mut data = self.data.clone();
        data.norm_squared().map_err(PartitionedTreeTNError::from)
    }

    /// Compute the inner product using the already masked stored TreeTNs.
    ///
    /// No projector is reapplied and no dense materialization is performed.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::TopologyMismatch`] or
    /// [`PartitionedTreeTNError::SiteIndexMismatch`] when the stored trees
    /// differ, [`PartitionedTreeTNError::DTypeMismatch`] for mixed scalar
    /// dtypes, and [`PartitionedTreeTNError::TreeTN`] when the TreeTN inner
    /// product fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_partitionedtreetn::SubDomainTreeTN;
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let make = || {
    ///     TreeTN::from_tensors(
    ///         vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0_f64, 4.0])?],
    ///         vec![0usize],
    ///     )
    /// };
    /// let left = SubDomainTreeTN::from_treetn(make()?)?;
    /// let right = SubDomainTreeTN::from_treetn(make()?)?;
    /// assert!((left.inner(&right)?.real() - 25.0).abs() < 1.0e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn inner(&self, other: &Self) -> Result<tensor4all_core::AnyScalar> {
        self.validate_invariants()?;
        other.validate_invariants()?;
        ensure_same_tree_structure(&self.data, &other.data)?;
        ensure_same_dtype(self.scalar_kind()?, other.scalar_kind()?)?;
        self.data
            .inner(&other.data)
            .map_err(PartitionedTreeTNError::from)
    }
}

pub(crate) fn ensure_same_topology<V>(
    left: &TreeTN<IdxTensor, V>,
    right: &TreeTN<IdxTensor, V>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if left.same_topology(right) {
        Ok(())
    } else {
        Err(PartitionedTreeTNError::TopologyMismatch)
    }
}

pub(crate) fn ensure_same_tree_structure<V>(
    left: &TreeTN<IdxTensor, V>,
    right: &TreeTN<IdxTensor, V>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    ensure_same_topology(left, right)?;
    if left.share_equivalent_site_index_network(right) {
        ensure_compatible_site_dimensions(left, right)
    } else {
        Err(PartitionedTreeTNError::SiteIndexMismatch)
    }
}

/// Require equal dimensions for every site index identity shared by both trees.
///
/// [`DynIndex`] equality and hashing include ID, tags, and prime level but
/// exclude the dimension, so an identity-matching pair with different
/// dimensions must be rejected explicitly; otherwise two TreeTNs could be
/// treated as structurally identical while holding incompatible site spaces.
pub(crate) fn ensure_compatible_site_dimensions<V>(
    left: &TreeTN<IdxTensor, V>,
    right: &TreeTN<IdxTensor, V>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (left_indices, _) = left
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;
    let (right_indices, _) = right
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;
    for left_index in &left_indices {
        if let Some(right_index) = right_indices
            .iter()
            .find(|candidate| *candidate == left_index)
        {
            if left_index.dim != right_index.dim {
                return Err(PartitionedTreeTNError::SiteIndexMismatch);
            }
        }
    }
    Ok(())
}

pub(crate) fn ensure_same_dtype(
    expected: Option<ScalarKind>,
    actual: Option<ScalarKind>,
) -> Result<()> {
    if expected == actual {
        return Ok(());
    }
    Err(PartitionedTreeTNError::DTypeMismatch {
        expected: expected
            .map(|dtype| dtype.name().to_string())
            .unwrap_or_else(|| "empty".to_string()),
        actual: actual
            .map(|dtype| dtype.name().to_string())
            .unwrap_or_else(|| "empty".to_string()),
    })
}

pub(crate) fn ensure_center<V>(data: &TreeTN<IdxTensor, V>, center: &V) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    if data.node_index(center).is_some() {
        Ok(())
    } else {
        Err(PartitionedTreeTNError::InvalidCenter)
    }
}

fn validate_budget_squared(budget_squared: f64) -> Result<()> {
    if !budget_squared.is_finite() || budget_squared < 0.0 {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "squared patch budgets must be finite and non-negative",
        });
    }
    Ok(())
}

pub(crate) fn validate_truncation_options(options: &TruncationOptions) -> Result<()> {
    validate_svd_truncation_options(options.max_bond_dim(), options.svd_policy()).map_err(|error| {
        match error {
            SvdTruncationOptionsError::ZeroMaxBondDim => PartitionedTreeTNError::InvalidOptions {
                operation: "truncate",
                reason: "max_bond_dim must be greater than zero",
            },
            SvdTruncationOptionsError::InvalidThreshold(_) => {
                PartitionedTreeTNError::InvalidOptions {
                    operation: "truncate",
                    reason: "SVD truncation threshold must be finite and non-negative",
                }
            }
        }
    })
}

pub(crate) fn validate_contraction_options(options: &ContractionOptions) -> Result<()> {
    validate_svd_truncation_options(options.max_bond_dim, options.svd_policy).map_err(|error| {
        match error {
            SvdTruncationOptionsError::ZeroMaxBondDim => PartitionedTreeTNError::InvalidOptions {
                operation: "contract",
                reason: "max_bond_dim must be greater than zero",
            },
            SvdTruncationOptionsError::InvalidThreshold(_) => {
                PartitionedTreeTNError::InvalidOptions {
                    operation: "contract",
                    reason: "SVD truncation threshold must be finite and non-negative",
                }
            }
        }
    })?;
    if matches!(options.method, ContractionMethod::Naive) {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "dense/reference contraction is not allowed in partition algebra",
        });
    }
    if matches!(options.method, ContractionMethod::Fit) && options.nfullsweeps == 0 {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "Fit contraction requires at least one full sweep",
        });
    }
    if options
        .qr_rtol
        .is_some_and(|rtol| !rtol.is_finite() || rtol < 0.0)
    {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "qr_rtol must be finite and non-negative",
        });
    }
    if options
        .convergence_tol
        .is_some_and(|tol| !tol.is_finite() || tol < 0.0)
    {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "contract",
            reason: "convergence_tol must be finite and non-negative",
        });
    }
    Ok(())
}

pub(crate) fn validate_contraction_site_assignment<V>(
    left: &TreeTN<IdxTensor, V>,
    right: &TreeTN<IdxTensor, V>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (left_indices, left_nodes) = left
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;
    let (right_indices, right_nodes) = right
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;

    ensure_compatible_site_dimensions(left, right)?;
    for (left_index, left_node) in left_indices.iter().zip(&left_nodes) {
        for (right_index, right_node) in right_indices.iter().zip(&right_nodes) {
            if left_index.is_contractable(right_index) && left_node != right_node {
                return Err(PartitionedTreeTNError::SiteIndexMismatch);
            }
        }
    }
    Ok(())
}

/// Site indices that survive the contraction of `left` with `right`: those
/// without a contractable partner on the other side.
///
/// This is the output-index rule of TreeTN contraction, which pairs indices by
/// [`IndexLike::is_contractable`] rather than by equality. Site indices are
/// unique within one TreeTN, so testing each index against the whole other side
/// gives the same pairing as the contraction's one-to-one matching.
pub(crate) fn contraction_output_site_indices<V>(
    left: &TreeTN<IdxTensor, V>,
    right: &TreeTN<IdxTensor, V>,
) -> Result<Vec<DynIndex>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (left_indices, _) = left
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;
    let (right_indices, _) = right
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;
    let unpaired = |indices: &[DynIndex], others: &[DynIndex]| {
        indices
            .iter()
            .filter(|index| !others.iter().any(|other| index.is_contractable(other)))
            .cloned()
            .collect::<Vec<_>>()
    };
    let mut output = unpaired(&left_indices, &right_indices);
    output.extend(unpaired(&right_indices, &left_indices));
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarKind {
    F32,
    F64,
    C32,
    C64,
}

impl ScalarKind {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C32 => "Complex32",
            Self::C64 => "Complex64",
        }
    }

    /// Machine epsilon of the kind's real component type.
    pub(crate) fn epsilon(self) -> f64 {
        match self {
            Self::F32 | Self::C32 => f64::from(f32::EPSILON),
            Self::F64 | Self::C64 => f64::EPSILON,
        }
    }
}

fn scalar_kind(tensor: &IdxTensor) -> Result<ScalarKind> {
    if tensor.is_f32() {
        Ok(ScalarKind::F32)
    } else if tensor.is_f64() {
        Ok(ScalarKind::F64)
    } else if tensor.is_c32() {
        Ok(ScalarKind::C32)
    } else if tensor.is_c64() {
        Ok(ScalarKind::C64)
    } else {
        Err(PartitionedTreeTNError::UnsupportedDType {
            dtype: format!("{:?}", tensor.storage_kind()),
        })
    }
}

fn validate_data<V>(data: &TreeTN<IdxTensor, V>) -> Result<Option<ScalarKind>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    if data.node_count() == 0 {
        return Err(PartitionedTreeTNError::Empty);
    }
    data.validate_tree()
        .map_err(PartitionedTreeTNError::invalid_topology)?;

    let mut dtype: Option<ScalarKind> = None;
    for name in data.node_names() {
        let node = data
            .node_index(&name)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node name has no node index"))?;
        let tensor = data
            .tensor(node)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node has no tensor"))?;
        let actual = scalar_kind(tensor)?;
        if let Some(expected) = dtype {
            if expected != actual {
                return Err(PartitionedTreeTNError::DTypeMismatch {
                    expected: expected.name().to_string(),
                    actual: actual.name().to_string(),
                });
            }
        } else {
            dtype = Some(actual);
        }
    }
    Ok(dtype)
}

fn validate_projector_against_data<V>(
    data: &TreeTN<IdxTensor, V>,
    projector: &Projector,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let (site_indices, _) = data
        .all_site_indices()
        .map_err(PartitionedTreeTNError::from)?;
    for (index, &value) in projector.iter() {
        let Some(matched) = site_indices.iter().find(|candidate| *candidate == index) else {
            return Err(PartitionedTreeTNError::ProjectorIndexNotFound {
                index: index.clone(),
            });
        };
        // `DynIndex` equality excludes the dimension, so an identity-matching
        // alias with a different dimension must be rejected before masking;
        // otherwise the alias reaches `mask_index` and fails as an internal
        // label-shape mismatch instead of a public site mismatch.
        if index.dim != matched.dim {
            return Err(PartitionedTreeTNError::SiteIndexMismatch);
        }
        if value >= matched.dim {
            return Err(PartitionedTreeTNError::ProjectorCoordinateOutOfBounds {
                index: index.clone(),
                value,
                dim: matched.dim,
            });
        }
    }
    Ok(())
}

fn rebuild_masked_tree<V>(
    data: &TreeTN<IdxTensor, V>,
    projector: &Projector,
) -> Result<TreeTN<IdxTensor, V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let node_names = data.node_names();
    let mut tensors = Vec::with_capacity(node_names.len());

    for name in &node_names {
        let node = data
            .node_index(name)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node name has no node index"))?;
        let source = data
            .tensor(node)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node has no tensor"))?;
        let mut tensor = source.clone();
        for (index, &value) in projector.iter() {
            // Mask using the canonical site index resolved from this tensor,
            // never the projector key: an identity-equal alias with a
            // different dimension would otherwise fail below `mask_index` as
            // an internal label-shape mismatch.
            if let Some(canonical) = tensor
                .indices()
                .iter()
                .find(|candidate| *candidate == index)
            {
                tensor = tensor.mask_index(canonical, value)?;
            }
        }
        tensors.push(tensor);
    }

    TreeTN::from_tensors(tensors, node_names).map_err(PartitionedTreeTNError::from)
}

#[cfg(test)]
mod tests;
