//! Partial site contraction for TreeTN.
//!
//! Provides [`partial_contract`] for selecting which site indices should be
//! contracted and which should be aligned before calling the existing TreeTN
//! contraction pipeline.

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{bail, Context, Result};

use super::contraction::{contract, ContractionOptions};
use super::TreeTN;
use tensor4all_core::{IndexLike, TensorIndex, TensorLike};

/// Specification for partial site contraction between two TreeTNs.
///
/// - `contract_pairs`: Site index pairs to sum over (removed from result).
/// - `multiply_pairs`: Site index pairs to identify/multiply (kept in result).
/// - Remaining (unmentioned) site indices pass through as external legs.
///
/// Uses `T::Index` objects directly (not `Index::Id`), following Julia ITensor
/// conventions.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_core::DynIndex;
/// use tensor4all_treetn::PartialContractionSpec;
///
/// let idx_a = DynIndex::new_dyn(4);
/// let idx_b = DynIndex::new_dyn(4);
/// let idx_c = DynIndex::new_dyn(3);
/// let idx_d = DynIndex::new_dyn(3);
///
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
///     multiply_pairs: vec![(idx_c.clone(), idx_d.clone())],
/// };
///
/// assert_eq!(spec.contract_pairs.len(), 1);
/// assert_eq!(spec.multiply_pairs.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PartialContractionSpec<I: IndexLike> {
    /// Site index pairs to contract (summed over, removed from result).
    pub contract_pairs: Vec<(I, I)>,
    /// Site index pairs to multiply (identified, kept in result).
    pub multiply_pairs: Vec<(I, I)>,
}

fn validate_partial_contraction_spec<T, V>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    spec: &PartialContractionSpec<T::Index>,
) -> Result<()>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Debug + Send + Sync + Ord,
{
    if !a.same_topology(b) {
        bail!("partial_contract: networks have incompatible topologies");
    }

    let a_external_ids: HashSet<_> = a
        .external_indices()
        .into_iter()
        .map(|idx| idx.id().clone())
        .collect();
    let b_external_ids: HashSet<_> = b
        .external_indices()
        .into_iter()
        .map(|idx| idx.id().clone())
        .collect();

    let mut seen_a_ids = HashSet::new();
    let mut seen_b_ids = HashSet::new();

    for (kind, pairs) in [
        ("contract_pairs", &spec.contract_pairs),
        ("multiply_pairs", &spec.multiply_pairs),
    ] {
        for (idx_a, idx_b) in pairs {
            if idx_a.dim() != idx_b.dim() {
                bail!(
                    "partial_contract: {} index dimension mismatch: {} != {}",
                    kind,
                    idx_a.dim(),
                    idx_b.dim()
                );
            }

            if !a_external_ids.contains(idx_a.id()) {
                bail!(
                    "partial_contract: {:?} from {} not found in first TreeTN external indices",
                    idx_a.id(),
                    kind
                );
            }
            if !b_external_ids.contains(idx_b.id()) {
                bail!(
                    "partial_contract: {:?} from {} not found in second TreeTN external indices",
                    idx_b.id(),
                    kind
                );
            }

            if !seen_a_ids.insert(idx_a.id().clone()) {
                bail!(
                    "partial_contract: first TreeTN index {:?} appears in multiple pairs",
                    idx_a.id()
                );
            }
            if !seen_b_ids.insert(idx_b.id().clone()) {
                bail!(
                    "partial_contract: second TreeTN index {:?} appears in multiple pairs",
                    idx_b.id()
                );
            }
        }
    }

    // Validate: contract_pairs and multiply_pairs must not target the same node
    // in either network. When both share a node, replaceind makes all shared
    // indices identical and contract() cannot distinguish "sum over" from "keep".
    let sin = a.site_index_network();
    for (c_a, _c_b) in &spec.contract_pairs {
        if let Some(node_a) = sin.find_node_by_index(c_a) {
            for (m_a, _m_b) in &spec.multiply_pairs {
                if let Some(node_m) = sin.find_node_by_index(m_a) {
                    if node_a == node_m {
                        bail!(
                            "partial_contract: contract index {:?} and multiply index {:?} \
                             both belong to the same node {:?} in the first network. \
                             contract_pairs and multiply_pairs must target different nodes.",
                            c_a.id(),
                            m_a.id(),
                            node_a
                        );
                    }
                }
            }
        }
    }

    let sin_b = b.site_index_network();
    for (_, c_b) in &spec.contract_pairs {
        if let Some(node_b) = sin_b.find_node_by_index(c_b) {
            for (_, m_b) in &spec.multiply_pairs {
                if let Some(node_m) = sin_b.find_node_by_index(m_b) {
                    if node_b == node_m {
                        bail!(
                            "partial_contract: contract index {:?} and multiply index {:?} \
                             both belong to the same node {:?} in the second network. \
                             contract_pairs and multiply_pairs must target different nodes.",
                            c_b.id(),
                            m_b.id(),
                            node_b
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

/// Partially contract two TreeTNs according to the given specification.
///
/// # Arguments
/// * `a` - First tensor network
/// * `b` - Second tensor network
/// * `spec` - Which site indices to contract vs multiply
/// * `center` - Canonical center node for the result
/// * `options` - Contraction algorithm options
///
/// # Index handling
///
/// - **contract_pairs**: Both indices are traced over (inner product).
///   Neither appears in the result.
/// - **multiply_pairs**: The two indices are identified (same physical quantity).
///   One index from each pair appears in the result.
/// - **Unmentioned indices**: Pass through unchanged as external legs.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_core::DynIndex;
/// use tensor4all_treetn::{
///     contraction::ContractionOptions,
///     partial_contract,
///     PartialContractionSpec,
/// };
///
/// let idx_a = DynIndex::new_dyn(2);
/// let idx_b = DynIndex::new_dyn(2);
///
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
///     multiply_pairs: vec![],
/// };
///
/// assert_eq!(spec.contract_pairs.len(), 1);
/// let _ = (partial_contract, ContractionOptions::default());
/// ```
pub fn partial_contract<T, V>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    spec: &PartialContractionSpec<T::Index>,
    center: &V,
    options: ContractionOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    validate_partial_contraction_spec(a, b, spec)?;

    let mut b_modified = b.clone();

    for (idx_a, idx_b) in &spec.multiply_pairs {
        b_modified = b_modified.replaceind(idx_b, idx_a).with_context(|| {
            format!(
                "partial_contract: failed to align multiply pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;
    }

    for (idx_a, idx_b) in &spec.contract_pairs {
        b_modified = b_modified.replaceind(idx_b, idx_a).with_context(|| {
            format!(
                "partial_contract: failed to align contract pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;
    }

    contract(a, &b_modified, center, options).context("partial_contract: contraction failed")
}

#[cfg(test)]
mod tests;
