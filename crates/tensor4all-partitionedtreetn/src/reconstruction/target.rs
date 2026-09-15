use std::{fmt::Debug, hash::Hash};

use tensor4all_core::{IndexLike, SvdTruncationPolicy};
use tensor4all_treetn::{
    contraction::ContractionOptions, partial_contract, PartialContractionSpec, SiteIndexNetwork,
};

use super::{finite, invalid};
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

/// Immutable orthogonal target and its pinned global L2 norm.
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
/// assert!((target.reference_norm() - 5.0).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ReconstructionTarget<V = usize>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    patches: Vec<(Projector, Patch<V>)>,
    pub(super) network: Option<SiteIndexNetwork<V, DynIndex>>,
    norm: f64,
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
    /// assert_eq!(target.reference_norm(), 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_partition(partition: &PartitionedTreeTN<V>) -> Result<Self> {
        partition.validate_contents()?;
        let mut patches: Vec<_> = partition.values().collect();
        patches.sort_by(|a, b| a.projector().canonical_cmp(b.projector()));
        let network = patches.first().map(|p| p.site_index_network().clone());
        let mut norm = 0.0_f64;
        let mut stored = Vec::with_capacity(patches.len());
        for patch in patches {
            norm = finite(norm.hypot(finite(patch.norm()?)?))?;
            stored.push((
                patch.projector().clone(),
                Patch::Stored(Box::new(patch.clone())),
            ));
        }
        Ok(Self {
            patches: stored,
            network,
            norm,
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
    /// factor norms; the target norm is their Euclidean norm. During
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
    /// assert!((target.reference_norm() - 10.0).abs() < 1e-12);
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
        let mut norm = 0.0_f64;
        let mut patches = Vec::with_capacity(pairs.len());
        for ((left, right), projector) in pairs.into_iter().zip(projectors) {
            let patch_norm = finite(finite(left.norm()?)? * finite(right.norm()?)?)?;
            norm = finite(norm.hypot(patch_norm))?;
            patches.push((projector, Patch::Product(Box::new((left, right)))));
        }
        patches.sort_by(|a, b| a.0.canonical_cmp(&b.0));
        Ok(Self {
            patches,
            network,
            norm,
        })
    }

    /// Return the original target L2 norm, fixed at construction.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_partitionedtreetn::{PartitionedTreeTN, reconstruction::ReconstructionTarget};
    /// let target = ReconstructionTarget::from_partition(&PartitionedTreeTN::<usize>::new())?;
    /// assert_eq!(target.reference_norm(), 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reference_norm(&self) -> f64 {
        self.norm
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
