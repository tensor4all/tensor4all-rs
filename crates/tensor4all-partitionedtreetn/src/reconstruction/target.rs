use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};

use tensor4all_core::{IdxTensor, IndexLike, SvdTruncationPolicy};
use tensor4all_treetn::{
    apply_linear_operator_to_indices, contraction::ContractionOptions, partial_contract,
    ApplyOptions, LinearOperator, PartialContractionSpec, RestructureOptions, SiteIndexNetwork,
};

use super::{finite, invalid, SubsetOperatorOptions};
use crate::subdomain_tree_tn::{
    ensure_center, ensure_same_dtype, ensure_same_topology, ensure_same_tree_structure,
};
use crate::{
    DynIndex, PartitionedTreeTN, PartitionedTreeTNError, Projector, Result, SubDomainTreeTN,
};

#[derive(Debug, Clone)]
enum Patch<V: Clone + Hash + Eq + Send + Sync + Debug> {
    Stored(Box<SubDomainTreeTN<V>>),
    Product(Box<(SubDomainTreeTN<V>, SubDomainTreeTN<V>)>),
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
        patches.sort_by(|a, b| a.projector().canonical_cmp(b.projector()));
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
        pairs: Vec<(SubDomainTreeTN<V>, SubDomainTreeTN<V>)>,
    ) -> Result<Self> {
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
        patches.sort_by(|a, b| a.0.canonical_cmp(&b.0));
        Ok(Self {
            patches,
            network,
            scale,
        })
    }

    /// Prepare the images of `preimage` under `operator` acting on an ordered subset.
    ///
    /// `selection` lists one full site index per operator node, in the operator's
    /// own node order. That node order defines the logical bit order: for a
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
    /// The operator is applied exactly with the local naive path; this entry
    /// point accepts no truncating apply options, so the prepared target carries
    /// no application error beyond backend roundoff. The approximation made when
    /// *constructing* the operator, for example `FourierOptions::tolerance` and
    /// `max_bond_dim`, is not included in
    /// [`ReconstructionReport::error_bound`](super::ReconstructionReport), which
    /// only bounds the reconstruction of these images; approximate application
    /// and its separate error accounting are follow-up work.
    ///
    /// # Errors
    /// Returns [`PartitionedTreeTNError::InvalidOptions`] when the selection does
    /// not match the operator's node count, repeats an index, or selects indices
    /// that share one tree node without forming one connected group of operator
    /// nodes (split the selection or restructure the operator), and
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
    pub fn from_subset_operator(
        preimage: &Self,
        center: &V,
        operator: &LinearOperator<IdxTensor, V>,
        selection: &[DynIndex],
        options: &SubsetOperatorOptions,
    ) -> Result<Self> {
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
        let mut owners: HashMap<V, V> = HashMap::with_capacity(operator_nodes.len());
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
            let members: HashSet<V> = owners
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

        // INVARIANT: application is exact (no truncation is exposed), so the
        // prepared target carries only backend roundoff, never silent
        // application error that the reconstruction report would mis-attribute.
        let apply_options = ApplyOptions::naive();
        let mut images = Vec::with_capacity(preimage.patches.len());
        for term in preimage.materialize_terms(center)? {
            let data = apply_linear_operator_to_indices(
                &operator,
                term.data(),
                &input_pairs,
                &output_pairs,
                apply_options.clone(),
            )
            .map_err(|error| {
                PartitionedTreeTNError::tree(format!("subset operator apply: {error}"))
            })?;
            // Images of disjoint patches generally overlap and the selected
            // constraints no longer hold, so keep only the spectator ones.
            let mut projector = term.projector().clone();
            for index in selection {
                projector.remove(index);
            }
            images.push(SubDomainTreeTN::new(data, projector)?);
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
        let mut patches: Vec<_> = images
            .into_iter()
            .map(|image| {
                let projector = image.projector().clone();
                (projector, Patch::Stored(Box::new(image)))
            })
            .collect();
        patches.sort_by(|a, b| a.0.canonical_cmp(&b.0));

        Ok(Self {
            patches,
            network: preimage.network.clone(),
            scale,
        })
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
