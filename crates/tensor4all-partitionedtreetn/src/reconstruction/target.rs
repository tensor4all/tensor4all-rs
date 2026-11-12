use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};

use tensor4all_core::{IdxTensor, IndexLike, SvdTruncationPolicy};
use tensor4all_treetn::{
    apply_linear_operator_to_indices, contraction::ContractionOptions, partial_contract,
    ApplyOptions, LinearOperator, PartialContractionSpec, RestructureOptions, SiteIndexNetwork,
    TreeTN,
};

use super::{finite, invalid, SubsetOperatorOptions};
use crate::subdomain_tree_tn::{
    ensure_center, ensure_same_dtype, ensure_same_topology, ensure_same_tree_structure,
    sort_by_patch_order, PatchOrderKey,
};
use crate::{
    DynIndex, PartitionedTreeTN, PartitionedTreeTNError, Projector, Result, SubDomainTreeTN,
};

#[derive(Debug, Clone)]
enum Patch<V: Clone + Hash + Eq + Send + Sync + Debug> {
    Stored(Box<SubDomainTreeTN<V>>),
    Product(Box<(SubDomainTreeTN<V>, SubDomainTreeTN<V>)>),
}

impl<V: Clone + Hash + Eq + Ord + Send + Sync + Debug> Patch<V> {
    fn order_key(&self) -> (&PatchOrderKey<V>, Option<&PatchOrderKey<V>>) {
        match self {
            Self::Stored(patch) => (patch.patch_order_key(), None),
            Self::Product(pair) => (pair.0.patch_order_key(), Some(pair.1.patch_order_key())),
        }
    }

    /// Content digest of the patch, or of both factors of a tensor product.
    fn content_digest(&self) -> Option<u64> {
        match self {
            Self::Stored(patch) => patch.content_digest(),
            Self::Product(pair) => pair.0.paired_content_digest(&pair.1),
        }
    }
}

/// One transformed image together with the preimage support it came from.
#[derive(Debug, Clone)]
pub(super) struct PreparedImage<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Preimage patch support, including its selected-index constraints.
    pub(super) source: Projector,
    /// Image retaining only the patch's spectator constraints.
    pub(super) image: SubDomainTreeTN<V>,
    /// Measured deviation of `image` from the exact application, zero when the
    /// operator was applied exactly.
    pub(super) error: f64,
}

/// Images of every preimage patch, with their provenance and the pinned scale.
#[derive(Debug, Clone)]
pub(super) struct PreparedImages<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// One entry per preimage patch, in `preimage`'s canonical patch order.
    pub(super) images: Vec<PreparedImage<V>>,
    /// `amplification * preimage.reference_scale()`.
    pub(super) scale: f64,
}

/// Immutable orthogonal target and its pinned reference scale.
///
/// The scale is the exact L2 norm for [`Self::from_partition`] and
/// [`Self::from_tensor_products`]. [`Self::from_subset_operator`] instead pins
/// the preimage scale times an operator amplification factor, which is an upper
/// bound for a general operator; see its documentation.
///
/// Orthogonality is established through disjoint projector supports, never a
/// caller-supplied boolean. [`Self::from_partition`] snapshots an existing
/// partition. [`Self::from_tensor_products`] retains independent factors and
/// computes the norm without constructing their products. Reconstruction
/// borrows this value and never replaces it with an intermediate approximation.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{DynIndex, IdxTensor, TreeTN, SubDomainTreeTN,
///     PartitionedTreeTN, reconstruction::ReconstructionTarget};
/// let tensor = IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![3.0, 4.0])?;
/// let patch = SubDomainTreeTN::from_treetn(TreeTN::from_tensors(vec![tensor], vec![0usize])?)?;
/// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::from_subdomain(patch)?)?;
/// assert!((target.reference_scale() - 5.0).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ReconstructionTarget<V = usize>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    patches: Vec<(Projector, Patch<V>)>,
    pub(super) network: Option<SiteIndexNetwork<V, DynIndex>>,
    scale: f64,
}

impl<V: Clone + Hash + Eq + Ord + Send + Sync + Debug> ReconstructionTarget<V> {
    /// Snapshot `partition` and compute its L2 norm from disjoint patch norms.
    ///
    /// The empty partition is a valid zero target with no site topology.
    /// No full dense tensor or global direct sum is constructed.
    ///
    /// # Errors
    /// Propagates invalid topology, site identity, dtype, projector, and backend
    /// errors. Returns [`PartitionedTreeTNError::NonFiniteAdaptiveValue`] when a norm is non-finite.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::ReconstructionTarget};
    /// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
    /// assert_eq!(target.reference_scale(), 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_partition(partition: &PartitionedTreeTN<V>) -> Result<Self> {
        partition.validate_contents()?;
        let mut patches: Vec<_> = partition.values().collect();
        sort_by_patch_order(
            &mut patches,
            |a, b| a.patch_order_key().cmp(b.patch_order_key()),
            |patch| patch.content_digest(),
        );
        let network = patches.first().map(|p| p.site_index_network().clone());
        let mut scale = 0.0_f64;
        let mut stored = Vec::with_capacity(patches.len());
        for patch in patches {
            scale = finite(scale.hypot(finite(patch.norm()?)?))?;
            stored.push((
                patch.projector().clone(),
                Patch::Stored(Box::new(patch.clone())),
            ));
        }
        Ok(Self {
            patches: stored,
            network,
            scale,
        })
    }

    /// Prepare disjoint patches `left ⊗ right` on explicitly independent site spaces.
    ///
    /// Both factors must have the same named tree topology, which defines the
    /// output topology. A node owns the union of its factors' external indices.
    /// All pairs must use the same left and right site assignments and dtype.
    /// No external indices may coincide or contract across factors. Reindex or
    /// restructure factors explicitly before calling when needed.
    ///
    /// Products remain factorized here. Each patch norm is the product of its
    /// factor norms; the reference scale is their Euclidean norm. During
    /// reconstruction, the existing non-dense partial-contraction path forms
    /// local products without a rank cap or approximation threshold.
    /// Validation of arbitrary projector supports uses pairwise overlap checks.
    ///
    /// # Errors
    /// Returns topology, site, dtype, and projector errors for incompatible
    /// factors, `InvalidOptions` for non-independent spaces,
    /// `OverlappingProjectors` for overlapping (including duplicate) supports,
    /// or `NonFiniteAdaptiveValue` for a non-finite factor or product norm.
    /// Backend norm errors preserve their source.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_partitionedtreetn::{DynIndex, IdxTensor, TreeTN, SubDomainTreeTN,
    ///     reconstruction::ReconstructionTarget};
    /// let make = |values| -> Result<_, Box<dyn std::error::Error>> {
    ///     let tensor = IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], values)?;
    ///     Ok(SubDomainTreeTN::from_treetn(TreeTN::from_tensors(vec![tensor], vec![0usize])?)?)
    /// };
    /// let target = ReconstructionTarget::from_tensor_products(vec![
    ///     (make(vec![3.0, 4.0])?, make(vec![0.0, 2.0])?),
    /// ])?;
    /// assert!((target.reference_scale() - 10.0).abs() < 1e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_tensor_products(
        mut pairs: Vec<(SubDomainTreeTN<V>, SubDomainTreeTN<V>)>,
    ) -> Result<Self> {
        sort_by_patch_order(
            &mut pairs,
            |(a, aa), (b, bb)| {
                (a.patch_order_key(), aa.patch_order_key())
                    .cmp(&(b.patch_order_key(), bb.patch_order_key()))
            },
            |(left, right)| left.paired_content_digest(right),
        );
        let mut projectors = Vec::with_capacity(pairs.len());
        let mut network = None;
        for (left, right) in &pairs {
            left.validate_invariants()?;
            right.validate_invariants()?;
            ensure_same_topology(left.data(), right.data())?;
            ensure_same_dtype(left.scalar_kind()?, right.scalar_kind()?)?;
            if let Some((first_left, first_right)) = pairs.first() {
                ensure_same_tree_structure(first_left.data(), left.data())?;
                ensure_same_tree_structure(first_right.data(), right.data())?;
                ensure_same_dtype(first_left.scalar_kind()?, left.scalar_kind()?)?;
            }
            let left_indices = left.all_indices();
            let right_indices = right.all_indices();
            if left_indices
                .iter()
                .any(|a| right_indices.iter().any(|b| a == b || a.is_contractable(b)))
            {
                return Err(invalid("tensor-product factors must have independent site indices; reindex factors explicitly"));
            }
            if network.is_none() {
                let mut combined = left.site_index_network().clone();
                let (indices, owners) = right.data().all_site_indices()?;
                for (index, owner) in indices.into_iter().zip(owners) {
                    combined.add_site_index(&owner, index)?;
                }
                network = Some(combined);
            }
            let projector = left
                .projector()
                .intersection(right.projector())
                .ok_or(PartitionedTreeTNError::ProjectorConflict)?;
            projectors.push(projector);
        }
        // INVARIANT: arbitrary partial assignments require compatibility checks;
        // reuse Projector's existing O(M²) metadata-only validator, never form
        // product tensors to establish orthogonality.
        if !Projector::are_disjoint(&projectors) {
            return Err(PartitionedTreeTNError::OverlappingProjectors);
        }
        let mut scale = 0.0_f64;
        let mut patches = Vec::with_capacity(pairs.len());
        for ((left, right), projector) in pairs.into_iter().zip(projectors) {
            let patch_norm = finite(finite(left.norm()?)? * finite(right.norm()?)?)?;
            scale = finite(scale.hypot(patch_norm))?;
            patches.push((projector, Patch::Product(Box::new((left, right)))));
        }
        sort_by_patch_order(
            &mut patches,
            |(_, left), (_, right)| left.order_key().cmp(&right.order_key()),
            |(_, patch)| patch.content_digest(),
        );
        Ok(Self {
            patches,
            network,
            scale,
        })
    }

    /// Prepare the images of `preimage` under `operator` acting on an ordered subset.
    ///
    /// `selection` lists one full site index per operator node, in the operator's
    /// sorted node order. Operator node names may have a different type from
    /// the state node names; binding and exact owner grouping happen here.
    /// That node order defines the logical bit order: for a
    /// quantics Fourier transform its node 0 is the most significant input bit
    /// (`k1`), and the output at node `j` carries the bit-reversed frequency bit
    /// `r_(R+1-j)`, so no output bit-reversal permutation is applied here.
    /// Selection order is independent of reconstruction's `patch_order`.
    ///
    /// The selection may skip nodes (noncontiguous indices) and each selected
    /// index may share its tree node with spectator site indices, which keep
    /// their identity, dimension, and node assignment. Several selected indices
    /// may also share one tree node: the operator MPO nodes carrying them are
    /// fused into a single multi-site node, which is exact and adds no
    /// truncation error, so the operator and the state agree on the node
    /// grouping. That fusion requires those operator nodes to form one connected
    /// group inside the operator's own topology; a group separated by another
    /// owner's node is rejected rather than mis-bound.
    ///
    /// [`SubsetOperatorOptions::unitary`] selects how the reference scale is
    /// derived. `false` (the default) multiplies the preimage's
    /// [`Self::reference_scale`] by the selected-space operator's Frobenius
    /// (Hilbert--Schmidt) norm, which upper-bounds its induced amplification.
    /// `true` is a caller guarantee that the operator preserves the L2 norm, so
    /// the amplification factor is exactly `1`. For an `N`-dimensional unitary
    /// the Frobenius norm is `sqrt(N)` while the induced 2-norm is `1`; the
    /// option assumes the latter and does not assert a unit Frobenius norm.
    /// Spectator identity factors are neither materialized nor counted, because
    /// `||A ⊗ I||_2 = ||A||_2 <= ||A||_F`. Successive applications multiply
    /// these amplification factors instead of recomputing an output norm.
    ///
    /// The transformed images are never summed into one network and their
    /// output norm is never measured. For a general operator the returned scale
    /// is therefore a norm-based upper bound and `rtol` is relative to that
    /// scale, not to `||A x||_2`; the two coincide for a unitary acting on an
    /// exactly known input scale. Each image keeps only its spectator
    /// constraints.
    ///
    /// The operator is applied exactly, by zip-up with a zero SVD threshold
    /// centred at the node that balances the site bits, so the prepared target
    /// carries no application error beyond backend roundoff. On a chain every
    /// block the zip-up forms is bounded by the physical dimension; on other
    /// trees the largest block is estimated first and a term above `2^28`
    /// elements is refused (see `# Errors`). Truncating the
    /// application with an explicitly measured deviation is available through
    /// [`super::schedule_merge_refine`], whose item budget charges it, rather than
    /// here: this immutable target has no place to carry that error. The
    /// approximation made when *constructing* the operator, for example
    /// `FourierOptions::tolerance` and `max_bond_dim`, is not included in
    /// [`ReconstructionReport::error_bound`](super::ReconstructionReport), which
    /// only bounds the reconstruction of these images.
    ///
    /// # Errors
    /// Returns [`PartitionedTreeTNError::InvalidOptions`] when the selection does
    /// not match the operator's node count, repeats an index, or selects indices
    /// that share one tree node without forming one connected group of operator
    /// nodes (split the selection or restructure the operator),
    /// [`PartitionedTreeTNError::ResourceLimit`] before any application when the
    /// state is a tree that is not a chain and the exact zip-up of some patch is
    /// estimated to form a block above `2^28` elements (use a chain layout or
    /// fewer selected sites; the limit is fixed), and
    /// [`PartitionedTreeTNError::NonFiniteAdaptiveValue`] or backend errors from
    /// patch materialization, application, the operator's Frobenius norm, and
    /// scaling.
    ///
    /// # Examples
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_partitionedtreetn::{
    ///     reconstruction::{ReconstructionTarget, SubsetOperatorOptions},
    ///     PartitionedTreeTN, SubDomainTreeTN,
    /// };
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let state = TreeTN::from_tensors(
    ///     vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?], vec![0usize])?;
    /// let preimage = ReconstructionTarget::from_partition(
    ///     &PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(state)?)?)?;
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
    /// let operator_input = DynIndex::new_dyn(2);
    /// let operator_output = DynIndex::new_dyn(2);
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(
    ///     0usize,
    ///     IndexMapping { true_index: operator_input, internal_index: internal_input },
    /// );
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(
    ///     0usize,
    ///     IndexMapping { true_index: operator_output, internal_index: internal_output },
    /// );
    /// let operator = LinearOperator::new(mpo, input_mapping, output_mapping);
    ///
    /// // The default scales by the operator's Frobenius norm sqrt(2).
    /// let target = ReconstructionTarget::from_subset_operator(
    ///     &preimage, &0, &operator, &[site.clone()], &SubsetOperatorOptions::default())?;
    /// assert!((target.reference_scale() - 5.0 * 2.0_f64.sqrt()).abs() < 1e-12);
    ///
    /// // A caller that guarantees unitarity keeps the preimage scale.
    /// let unitary = ReconstructionTarget::from_subset_operator(
    ///     &preimage, &0, &operator, &[site], &SubsetOperatorOptions { unitary: true })?;
    /// assert!((unitary.reference_scale() - 5.0).abs() < 1e-12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// [`SubsetOperatorOptions::unitary`]: super::SubsetOperatorOptions::unitary
    pub fn from_subset_operator<OperatorV>(
        preimage: &Self,
        center: &V,
        operator: &LinearOperator<IdxTensor, OperatorV>,
        selection: &[DynIndex],
        options: &SubsetOperatorOptions,
    ) -> Result<Self>
    where
        OperatorV: Clone + Hash + Eq + Ord + Send + Sync + Debug,
    {
        if preimage.patches.is_empty() {
            if selection.is_empty() {
                return Ok(Self {
                    patches: Vec::new(),
                    network: None,
                    scale: 0.0,
                });
            }
            return Err(invalid(
                "cannot apply a subset operator to an empty target; build a target with sites first",
            ));
        }

        let prepared =
            Self::prepare_subset_images(preimage, center, operator, selection, options, None)?;
        let mut patches: Vec<_> = prepared
            .images
            .into_iter()
            .map(|prepared| {
                let projector = prepared.image.projector().clone();
                (projector, Patch::Stored(Box::new(prepared.image)))
            })
            .collect();
        sort_by_patch_order(
            &mut patches,
            |(_, left), (_, right)| left.order_key().cmp(&right.order_key()),
            |(_, patch)| patch.content_digest(),
        );

        Ok(Self {
            patches,
            network: preimage.network.clone(),
            scale: prepared.scale,
        })
    }

    /// Apply `operator` to `selection` on every patch of `preimage`.
    ///
    /// This is the preparation seam shared by [`Self::from_subset_operator`] and
    /// [`super::schedule_merge_refine`]. Every result keeps the preimage patch
    /// support it came from, so a caller that schedules the transform over dyadic
    /// input leaves can recover that leaf geometry instead of inferring it from
    /// the spectator-only image projector. Application is exact, so an image
    /// carries only backend roundoff.
    ///
    /// # Errors
    /// Propagates the selection, operator-mapping, node-merging, and application
    /// errors documented on [`Self::from_subset_operator`].
    pub(super) fn prepare_subset_images<OperatorV>(
        preimage: &Self,
        center: &V,
        operator: &LinearOperator<IdxTensor, OperatorV>,
        selection: &[DynIndex],
        options: &SubsetOperatorOptions,
        apply: Option<&ApplyOptions<V>>,
    ) -> Result<PreparedImages<V>>
    where
        OperatorV: Clone + Hash + Eq + Ord + Send + Sync + Debug,
    {
        let mut operator_nodes = operator.mpo().node_names();
        operator_nodes.sort();
        if operator_nodes.len() != selection.len() {
            return Err(invalid(
                "selection must contain exactly one full index per operator node",
            ));
        }
        let mut seen = HashSet::new();
        for index in selection {
            if !seen.insert(index) {
                return Err(invalid("selection must not repeat a full site index"));
            }
        }

        let network = preimage
            .network
            .as_ref()
            .ok_or_else(|| invalid("preimage target has no site topology"))?;
        // Several selected indices may share one tree node. The operator MPO
        // nodes carrying them are then fused into one multi-site node, so the
        // operator and the state agree on the node grouping. Fusion is exact:
        // `LinearOperator::restructure_to` contracts the group locally without
        // truncation and moves every mapping to the node owning its internal
        // index. Singleton groups make this a pure node rename.
        let mpo_network = operator.mpo().site_index_network();
        let mut target = SiteIndexNetwork::<V, DynIndex>::new();
        let mut owners: HashMap<OperatorV, V> = HashMap::with_capacity(operator_nodes.len());
        let mut input_pairs = Vec::with_capacity(selection.len());
        let mut output_pairs = Vec::with_capacity(selection.len());
        for (node, target_index) in operator_nodes.iter().zip(selection) {
            let owner = network
                .find_node_by_index(target_index)
                .ok_or_else(|| {
                    invalid("selection index must belong to the preimage target site space")
                })?
                .clone();
            let inputs = operator
                .get_input_mappings(node)
                .ok_or_else(|| invalid("every operator node needs exactly one input mapping"))?;
            let outputs = operator
                .get_output_mappings(node)
                .ok_or_else(|| invalid("every operator node needs exactly one output mapping"))?;
            let ([input], [output]) = (inputs, outputs) else {
                return Err(invalid(
                    "subset operators must carry one input and one output mapping per node",
                ));
            };
            if !target.has_node(&owner) {
                target
                    .add_node(owner.clone(), HashSet::new())
                    .map_err(merging)?;
            }
            for index in [&input.internal_index, &output.internal_index] {
                target
                    .add_site_index(&owner, index.clone())
                    .map_err(merging)?;
            }
            input_pairs.push((input.true_index.clone(), target_index.clone()));
            output_pairs.push((output.true_index.clone(), target_index.clone()));
            owners.insert(node.clone(), owner);
        }
        // INVARIANT: a fused node must be reachable inside its own group. A
        // group threaded through another owner's node cannot be fused without
        // absorbing sites that belong elsewhere, so reject it with guidance
        // instead of mis-binding a multi-site operator node.
        for owner in target.node_names().into_iter().cloned().collect::<Vec<V>>() {
            let members: HashSet<OperatorV> = owners
                .iter()
                .filter(|(_, candidate)| **candidate == owner)
                .map(|(node, _)| node.clone())
                .collect();
            if !is_connected_group(mpo_network, &members) {
                return Err(invalid(
                    "selected indices that share one tree node need operator MPO nodes forming one connected group; split the selection or restructure the operator",
                ));
            }
        }
        // The quotient topology: keep one edge per pair of distinct owners.
        let mut linked = HashSet::new();
        for (left, right) in mpo_network.edges() {
            let (Some(owner_left), Some(owner_right)) = (owners.get(&left), owners.get(&right))
            else {
                continue;
            };
            if owner_left == owner_right {
                continue;
            }
            let pair = if owner_left <= owner_right {
                (owner_left.clone(), owner_right.clone())
            } else {
                (owner_right.clone(), owner_left.clone())
            };
            if linked.insert(pair.clone()) {
                target.add_edge(&pair.0, &pair.1).map_err(merging)?;
            }
        }
        let operator = operator
            .restructure_to(&target, &RestructureOptions::default())
            .map_err(merging)?;

        // INVARIANT: the exact application is always performed. It is the
        // prepared image when the caller requests no truncation, and the measured
        // reference when the caller does, so an approximate application never
        // becomes a silent error that the report would mis-attribute.
        //
        // It is a zip-up with a zero SVD threshold and no cap, centred at the
        // balanced node, not the naive local product: the naive product makes
        // every bond input bond × operator bond wide (7200 for a two-axis
        // interleaved Fourier transform at R = 10), while the zip-up keeps each
        // factorized block bounded by the physical dimension of the side it has
        // processed (#1279; see the design note on exact application in
        // `orthogonal-target-reconstruction.md`). The threshold must be explicit:
        // `svd_policy = None` resolves to the default relative cutoff, which is
        // not exact.
        //
        // On a chain every block is bounded on both sides by the physical
        // dimension. On other trees only the processed side is, so the largest
        // block is estimated for every term before any application, and a term
        // above `EXACT_TREE_BLOCK_LIMIT` is refused instead of allocated.
        let terms = preimage.materialize_terms(center)?;
        let operator_nodes: HashSet<V> = operator.mpo().node_names().into_iter().collect();
        let operator_bond = operator.mpo().link_dims().into_iter().max().unwrap_or(1);
        let mut centers = Vec::with_capacity(terms.len());
        for term in &terms {
            let state = term.data();
            let term_center = balanced_center(state.site_index_network());
            if let Some(term_center) = &term_center {
                let peak = tree_zipup_peak(
                    state.site_index_network(),
                    |left, right| {
                        state
                            .edge_between(left, right)
                            .and_then(|edge| state.bond_index(edge))
                            .map_or(1, IndexLike::dim)
                    },
                    &operator_nodes,
                    operator_bond,
                    term_center,
                );
                refuse_large_tree_block(peak, EXACT_TREE_BLOCK_LIMIT)?;
            }
            centers.push(term_center);
        }
        let mut images = Vec::with_capacity(preimage.patches.len());
        for (term, term_center) in terms.into_iter().zip(centers) {
            let exact_options = ApplyOptions {
                svd_policy: Some(SvdTruncationPolicy::new(0.0)),
                max_bond_dim: None,
                center: term_center,
                ..ApplyOptions::zipup()
            };
            let apply_with = |options: ApplyOptions<V>| -> Result<TreeTN<IdxTensor, V>> {
                apply_linear_operator_to_indices(
                    &operator,
                    term.data(),
                    &input_pairs,
                    &output_pairs,
                    options,
                )
                .map_err(|error| {
                    PartitionedTreeTNError::tree(format!("subset operator apply: {error}"))
                })
            };
            let exact = apply_with(exact_options)?;
            let (data, error) = match apply {
                None => (exact, 0.0),
                Some(requested) => {
                    let approximate = apply_with(requested.clone())?;
                    let mut difference = exact.axpby(1.0, &approximate, -1.0)?;
                    (approximate, finite(difference.norm()?)?)
                }
            };
            // Images of disjoint patches generally overlap and the selected
            // constraints no longer hold, so keep only the spectator ones. The
            // original support stays as input provenance.
            let source = term.projector().clone();
            let mut projector = source.clone();
            for index in selection {
                projector.remove(index);
            }
            images.push(PreparedImage {
                source,
                image: SubDomainTreeTN::new(data, projector)?,
                error,
            });
        }

        // The output norm is deliberately not measured: doing so would need
        // either a global direct sum of the images (TreeTN addition adds bond
        // dimensions and restores the global-rank bottleneck) or a
        // pairwise-overlap Gram sum (O(M^2) and cancellation-prone). Scale the
        // preimage's reference scale by the operator's amplification factor
        // instead. `unitary = true` is the caller's guarantee of factor 1;
        // otherwise the selected-space operator's Frobenius norm is a norm-based
        // upper bound. Successive applications multiply these factors.
        let amplification = if options.unitary {
            1.0
        } else {
            // `TreeTN::norm` canonicalizes a clone; the MPO is gauge-invariant.
            finite(operator.mpo().clone().norm()?)?
        };
        let scale = finite(amplification * preimage.scale)?;
        Ok(PreparedImages { images, scale })
    }

    /// Return the target's reference scale, fixed at construction.
    ///
    /// It is the exact L2 norm for [`Self::from_partition`] and
    /// [`Self::from_tensor_products`], and a norm-based upper bound for
    /// [`Self::from_subset_operator`] with a general operator.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::ReconstructionTarget};
    /// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
    /// assert_eq!(target.reference_scale(), 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reference_scale(&self) -> f64 {
        self.scale
    }

    /// Borrow the canonical patch supports, in the order [`Self::materialize_terms`] uses.
    ///
    /// Used by callers that must validate patch geometry before paying for
    /// operator application.
    pub(super) fn patch_projectors(&self) -> impl Iterator<Item = &Projector> {
        self.patches.iter().map(|(projector, _)| projector)
    }

    pub(super) fn logical_site_indices(&self) -> Vec<DynIndex> {
        let Some((_, patch)) = self.patches.first() else {
            return Vec::new();
        };
        let factors: Vec<&SubDomainTreeTN<V>> = match patch {
            Patch::Stored(patch) => vec![patch],
            Patch::Product(pair) => vec![&pair.0, &pair.1],
        };
        let mut nodes = factors[0].data().node_names();
        nodes.sort();
        nodes
            .into_iter()
            .flat_map(|node| {
                factors.iter().flat_map(move |patch| {
                    patch.data().node_site_indices(&node).unwrap_or_default()
                })
            })
            .collect()
    }

    pub(super) fn materialize_terms(&self, center: &V) -> Result<Vec<SubDomainTreeTN<V>>> {
        let mut terms = Vec::with_capacity(self.patches.len());
        for (projector, patch) in &self.patches {
            let data = match patch {
                Patch::Stored(patch) => {
                    ensure_center(patch.data(), center)?;
                    patch.data().clone()
                }
                Patch::Product(pair) => {
                    let (left, right) = pair.as_ref();
                    ensure_center(left.data(), center)?;
                    let spec = PartialContractionSpec {
                        contract_pairs: Vec::new(),
                        diagonal_pairs: Vec::new(),
                        output_order: None,
                    };
                    let options = ContractionOptions::default()
                        .with_svd_policy(SvdTruncationPolicy::new(0.0))
                        .with_qr_rtol(0.0);
                    partial_contract(left.data(), right.data(), &spec, center, options)?
                }
            };
            // Products and stored patches are zero outside their original
            // support. Retain that metadata until a requested output region
            // deliberately groups several original supports.
            terms.push(SubDomainTreeTN::new(data, projector.clone())?);
        }
        Ok(terms)
    }
}

/// Wrap a node-merging failure with the operation that raised it.
fn merging(error: tensor4all_treetn::TreeTNOperationError) -> PartitionedTreeTNError {
    PartitionedTreeTNError::tree(format!("subset operator node merging: {error}"))
}

/// True when `members` form one connected group using group-internal edges only.
///
/// Connectivity inside the group, not merely inside the whole operator, is what
/// makes local fusion of one owner's operator nodes possible.
fn is_connected_group<V>(network: &SiteIndexNetwork<V, DynIndex>, members: &HashSet<V>) -> bool
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if members.len() <= 1 {
        return true;
    }
    let Some(start) = members.iter().next() else {
        return true;
    };
    let mut seen = HashSet::new();
    let mut stack = vec![start.clone()];
    seen.insert(start.clone());
    while let Some(node) = stack.pop() {
        for neighbor in network.neighbors(&node) {
            if members.contains(&neighbor) && seen.insert(neighbor.clone()) {
                stack.push(neighbor);
            }
        }
    }
    seen.len() == members.len()
}

/// Largest block, in elements, that the exact application may form on a tree
/// that is not a chain: `2^28`, 4 GiB in complex `f64` (#1279).
///
/// A chain needs no limit, since the chain zip-up bounds every block on both
/// sides by the physical dimension; on other trees only the processed side is
/// bounded. The largest block measured on a chain, two interleaved axes at
/// R = 10, is about `2^24.3` elements.
const EXACT_TREE_BLOCK_LIMIT: usize = 1 << 28;

/// The nodes of `network` in depth-first order from `root`, each after its
/// parent, and every node's parent; empty when `root` is absent.
fn rooted_order<V>(network: &SiteIndexNetwork<V, DynIndex>, root: &V) -> (Vec<V>, HashMap<V, V>)
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut parent: HashMap<V, V> = HashMap::new();
    let mut order = Vec::new();
    if !network.has_node(root) {
        return (order, parent);
    }
    let mut stack = vec![root.clone()];
    let mut visited = HashSet::from([root.clone()]);
    while let Some(node) = stack.pop() {
        for next in network.neighbors(&node) {
            if visited.insert(next.clone()) {
                parent.insert(next.clone(), node.clone());
                stack.push(next);
            }
        }
        order.push(node);
    }
    (order, parent)
}

/// The children of `node` under the rooting recorded in `parent`.
fn children<'a, V>(
    network: &'a SiteIndexNetwork<V, DynIndex>,
    parent: &'a HashMap<V, V>,
    node: &'a V,
) -> impl Iterator<Item = V> + 'a
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    network
        .neighbors(node)
        .filter(move |next| parent.get(next) == Some(node))
}

/// The node whose largest neighbouring subtree carries the fewest site bits,
/// ties broken by node order; `None` for an empty network.
///
/// A site's bits are `log2` of its dimension. The exact zip-up of
/// `prepare_subset_images` factorizes blocks bounded by the physical dimension
/// of the side already processed, which grows towards the centre; centring here
/// keeps that side at most about half of the total (#1279). On a chain this is
/// the middle by bits rather than the default endpoint.
fn balanced_center<V>(network: &SiteIndexNetwork<V, DynIndex>) -> Option<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut names: Vec<V> = network.node_names().into_iter().cloned().collect();
    names.sort();
    let root = names.first()?.clone();
    let bits = |node: &V| -> f64 {
        network.site_space(node).map_or(0.0, |sites| {
            sites.iter().map(|site| (site.dim() as f64).log2()).sum()
        })
    };
    let (order, parent) = rooted_order(network, &root);

    // Bits of the subtree below each node, children before parents.
    let mut below: HashMap<V, f64> = order
        .iter()
        .map(|node| (node.clone(), bits(node)))
        .collect();
    for node in order.iter().rev() {
        let own = below.get(node).copied().unwrap_or(0.0);
        if let Some(up) = parent.get(node) {
            if let Some(total) = below.get_mut(up) {
                *total += own;
            }
        }
    }
    let total = below.get(&root).copied().unwrap_or(0.0);

    names
        .into_iter()
        .map(|node| {
            let above = total - below.get(&node).copied().unwrap_or(0.0);
            let largest = children(network, &parent, &node)
                .map(|child| below.get(&child).copied().unwrap_or(0.0))
                .fold(above, f64::max);
            (largest, node)
        })
        // `min_by` keeps the first of equal minima, so ties go to the smaller name.
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, node)| node)
}

/// An upper estimate, in elements, of the largest block the exact zip-up forms
/// when it applies an operator on `operator_nodes` to a state on `network`
/// centred at `center`; `None` on a chain, where the chain schedule bounds every
/// block (see [`EXACT_TREE_BLOCK_LIMIT`]).
///
/// It follows the tree zip-up from the leaves to `center`. A node outside the
/// centre whose subtree holds no operator node is emitted unchanged, so its
/// parent keeps only the state bond to it. Every other node contracts its
/// children's remainders with its own state and operator tensors into a block of
/// `(remainders × site dimensions) × (state bond × operator bond)` towards its
/// parent, and passes on a remainder no wider than the smaller of the two
/// factors; the centre's block has no parent side. `state_bond` gives the state's
/// bond dimension on an edge; every operator bond is taken as `operator_bond`,
/// the operator's largest, which overestimates bonds the operator only passes
/// through. The estimate is a float, so it does not overflow.
fn tree_zipup_peak<V>(
    network: &SiteIndexNetwork<V, DynIndex>,
    state_bond: impl Fn(&V, &V) -> usize,
    operator_nodes: &HashSet<V>,
    operator_bond: usize,
    center: &V,
) -> Option<f64>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if network
        .node_names()
        .into_iter()
        .all(|node| network.neighbors(node).count() <= 2)
    {
        return None;
    }
    let site_dimension = |node: &V| -> f64 {
        network.site_space(node).map_or(1.0, |sites| {
            sites.iter().map(|site| site.dim() as f64).product()
        })
    };
    let (order, parent) = rooted_order(network, center);
    let mut acts: HashMap<V, bool> = HashMap::with_capacity(order.len());
    let mut remainder: HashMap<V, f64> = HashMap::with_capacity(order.len());
    let mut peak = 0.0_f64;
    for node in order.iter().rev() {
        let mut processed = site_dimension(node);
        let mut acts_below = operator_nodes.contains(node);
        for child in children(network, &parent, node) {
            processed *= remainder.get(&child).copied().unwrap_or(1.0);
            acts_below |= acts.get(&child).copied().unwrap_or(false);
        }
        acts.insert(node.clone(), acts_below);
        let Some(up) = parent.get(node) else {
            peak = peak.max(processed);
            continue;
        };
        let bond = state_bond(node, up) as f64;
        if acts_below {
            let outward = bond * operator_bond as f64;
            peak = peak.max(processed * outward);
            remainder.insert(node.clone(), processed.min(outward));
        } else {
            remainder.insert(node.clone(), bond);
        }
    }
    Some(peak)
}

/// Refuse an exact application whose estimated largest block exceeds `limit`
/// elements; `peak` is [`tree_zipup_peak`]'s estimate, `None` on a chain.
///
/// # Errors
/// Returns [`PartitionedTreeTNError::ResourceLimit`] when `peak` exceeds `limit`.
fn refuse_large_tree_block(peak: Option<f64>, limit: usize) -> Result<()> {
    match peak {
        Some(peak) if peak > limit as f64 => Err(PartitionedTreeTNError::ResourceLimit {
            operation: "exact subset operator application on a tree that is not a chain",
            limit: "zip-up block (elements; use a chain layout)",
            // `as` saturates a float above `usize::MAX`.
            value: peak as usize,
        }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests;
