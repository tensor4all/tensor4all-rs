//! Partial site contraction for TreeTN.
//!
//! Provides [`partial_contract`] for selecting which site indices should be
//! contracted and which should be linked through explicit diagonal/copy
//! structure before calling the existing TreeTN contraction pipeline.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{anyhow, bail, Context, Result};

use super::contraction::{contract, ContractionOptions};
use super::decompose::{factorize_tensor_to_treetn_with, TreeTopology};
use super::TreeTN;
use tensor4all_core::{
    AllowedPairs, AnyScalar, DynIndex, FactorizeAlg, FactorizeOptions, IndexLike, TensorDynLen,
    TensorIndex, TensorLike,
};

type DiagonalPairApplication<V> = (
    TreeTN<TensorDynLen, V>,
    TreeTN<TensorDynLen, V>,
    Vec<DynIndex>,
    Vec<DynIndex>,
);

/// Specification for partial site contraction between two TreeTNs.
///
/// - `contract_pairs`: Site index pairs to sum over and remove from the result.
/// - `diagonal_pairs`: Site index pairs to identify through diagonal/copy
///   structure while keeping the left-hand site leg in the result.
/// - Remaining (unmentioned) site indices pass through as external legs.
///
/// Uses `Index` objects directly (not raw IDs), following Julia ITensor-style
/// conventions.
///
/// # Examples
///
/// ```
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
///     diagonal_pairs: vec![(idx_c.clone(), idx_d.clone())],
///     output_order: None,
/// };
///
/// assert_eq!(spec.contract_pairs.len(), 1);
/// assert_eq!(spec.diagonal_pairs.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PartialContractionSpec<I: IndexLike> {
    /// Site index pairs to contract (summed over, removed from result).
    pub contract_pairs: Vec<(I, I)>,
    /// Site index pairs to link through diagonal/copy structure while keeping
    /// the left-hand site index in the result.
    pub diagonal_pairs: Vec<(I, I)>,
    /// Optional order for the surviving external site indices in the result.
    ///
    /// The indices must refer to the final result indices after applying
    /// `contract_pairs` and `diagonal_pairs`. When provided, the result is
    /// post-processed so that these indices appear in the requested order.
    ///
    /// Current implementation requires that each surviving site index occupies a
    /// distinct node in the result.
    pub output_order: Option<Vec<I>>,
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
        ("diagonal_pairs", &spec.diagonal_pairs),
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

    Ok(())
}

fn canonical_edge<V>(left: &V, right: &V) -> (V, V)
where
    V: Clone + Ord,
{
    if left <= right {
        (left.clone(), right.clone())
    } else {
        (right.clone(), left.clone())
    }
}

fn sorted_edge_set<V>(tn: &TreeTN<TensorDynLen, V>) -> Vec<(V, V)>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
{
    let mut edges: Vec<_> = tn
        .site_index_network()
        .edges()
        .map(|(u, v)| canonical_edge(&u, &v))
        .collect();
    edges.sort();
    edges.dedup();
    edges
}

fn compatible_union_node_names<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
) -> Vec<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
{
    let mut names: Vec<_> = a.node_names();
    names.extend(b.node_names());
    names.sort();
    names.dedup();
    names
}

fn validate_union_topology<V>(node_names: &[V], edges: &[(V, V)]) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
{
    if node_names.is_empty() {
        bail!("partial_contract: networks must contain at least one node");
    }

    if edges.len() + 1 != node_names.len() {
        bail!("partial_contract: networks have incompatible topologies");
    }

    let mut adjacency: HashMap<V, Vec<V>> = node_names
        .iter()
        .cloned()
        .map(|name| (name, Vec::new()))
        .collect();
    for (u, v) in edges {
        let Some(neighbors_u) = adjacency.get_mut(u) else {
            bail!("partial_contract: union topology references unknown node");
        };
        neighbors_u.push(v.clone());
        let Some(neighbors_v) = adjacency.get_mut(v) else {
            bail!("partial_contract: union topology references unknown node");
        };
        neighbors_v.push(u.clone());
    }

    let mut seen = HashSet::new();
    let mut stack = vec![node_names[0].clone()];
    while let Some(node) = stack.pop() {
        if !seen.insert(node.clone()) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&node) {
            stack.extend(neighbors.iter().cloned());
        }
    }

    if seen.len() != node_names.len() {
        bail!("partial_contract: networks have incompatible topologies");
    }

    Ok(())
}

fn factorize_options_from_contraction_options(
    options: &ContractionOptions,
) -> Result<FactorizeOptions> {
    let mut factorize_options = match options.factorize_alg {
        FactorizeAlg::SVD => FactorizeOptions::svd(),
        FactorizeAlg::QR => FactorizeOptions::qr(),
        FactorizeAlg::LU => FactorizeOptions::lu(),
        FactorizeAlg::CI => FactorizeOptions::ci(),
    };
    if let Some(policy) = options.svd_policy {
        factorize_options = factorize_options.with_svd_policy(policy);
    }
    if let Some(rtol) = options.qr_rtol {
        factorize_options = factorize_options.with_qr_rtol(rtol);
    }
    if let Some(max_rank) = options.max_rank {
        factorize_options = factorize_options.with_max_rank(max_rank);
    }
    factorize_options.validate().map_err(|err| {
        anyhow!("partial_contract: invalid contraction factorization options: {err}")
    })?;
    Ok(factorize_options)
}

fn union_result_topology<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    contracted_tensor: &TensorDynLen,
) -> Result<TreeTopology<V, <DynIndex as IndexLike>::Id>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let node_names = compatible_union_node_names(a, b);
    let mut union_edges = sorted_edge_set(a);
    union_edges.extend(sorted_edge_set(b));
    union_edges.sort();
    union_edges.dedup();
    validate_union_topology(&node_names, &union_edges)?;

    let surviving_ids: HashSet<_> = contracted_tensor
        .external_indices()
        .into_iter()
        .map(|idx| *idx.id())
        .collect();

    let mut nodes = HashMap::new();
    for node_name in &node_names {
        let mut ids = Vec::new();

        if let Some(site_space_a) = a.site_index_network().site_space(node_name) {
            for site_idx in site_space_a {
                if surviving_ids.contains(site_idx.id()) {
                    ids.push(*site_idx.id());
                }
            }
        }

        if let Some(site_space_b) = b.site_index_network().site_space(node_name) {
            for site_idx in site_space_b {
                if surviving_ids.contains(site_idx.id()) && !ids.contains(site_idx.id()) {
                    ids.push(*site_idx.id());
                }
            }
        }

        nodes.insert(node_name.clone(), ids);
    }

    Ok(TreeTopology::new(nodes, union_edges))
}

fn contract_mismatched_topologies<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    center: &V,
    options: ContractionOptions,
) -> Result<TreeTN<TensorDynLen, V>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let a_dense = a
        .sim_internal_inds()
        .contract_to_tensor()
        .context("partial_contract: failed to contract first mismatched-topology TreeTN")?;
    let b_dense = b
        .sim_internal_inds()
        .contract_to_tensor()
        .context("partial_contract: failed to contract second mismatched-topology TreeTN")?;
    let contracted_tensor =
        <TensorDynLen as TensorLike>::contract(&[&a_dense, &b_dense], AllowedPairs::All)
            .context("partial_contract: failed dense contraction for mismatched topologies")?;

    if contracted_tensor.external_indices().is_empty() {
        let mut result = TreeTN::<TensorDynLen, V>::new();
        result
            .add_tensor(center.clone(), contracted_tensor)
            .context("partial_contract: failed to wrap scalar mismatched-topology result")?;
        result
            .set_canonical_region([center.clone()])
            .context("partial_contract: failed to set canonical region for scalar result")?;
        return Ok(result);
    }

    let topology = union_result_topology(a, b, &contracted_tensor)?;
    let factorize_options = factorize_options_from_contraction_options(&options)?;
    factorize_tensor_to_treetn_with(&contracted_tensor, &topology, factorize_options, center)
        .context("partial_contract: failed to factorize mismatched-topology dense result")
}

fn apply_output_order<T, V>(result: TreeTN<T, V>, output_order: &[T::Index]) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    T::Index: Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let (current_indices, _) = result.all_site_indices()?;
    if output_order.len() != current_indices.len() {
        bail!(
            "partial_contract: output_order length {} does not match surviving external index count {}",
            output_order.len(),
            current_indices.len()
        );
    }

    let current_ids: HashSet<_> = current_indices.iter().map(|idx| idx.id().clone()).collect();
    let requested_ids: HashSet<_> = output_order.iter().map(|idx| idx.id().clone()).collect();
    if current_ids != requested_ids {
        bail!("partial_contract: output_order must contain exactly the surviving external indices");
    }

    let mut current_nodes = Vec::with_capacity(current_indices.len());
    for index in &current_indices {
        let node = result.site_index_network().find_node_by_index(index).ok_or_else(|| {
            anyhow!(
                "partial_contract: current result index {:?} is not present in the site index network",
                index.id()
            )
        })?;
        current_nodes.push(node.clone());
    }

    let unique_current_nodes: HashSet<_> = current_nodes.iter().cloned().collect();
    if unique_current_nodes.len() != current_nodes.len() {
        bail!(
            "partial_contract: output_order currently requires at most one surviving site index per node"
        );
    }

    let mut seen_requested = HashSet::new();
    let mut ordered_nodes = Vec::with_capacity(result.node_count());
    let mut ordered_node_set = HashSet::new();

    for index in output_order {
        if !seen_requested.insert(index.id().clone()) {
            bail!("partial_contract: output_order contains duplicate indices");
        }
        let current_node = result
            .site_index_network()
            .find_node_by_index(index)
            .ok_or_else(|| {
                anyhow!(
                    "partial_contract: output_order index {:?} is not present in the result",
                    index.id()
                )
            })?;
        if !ordered_node_set.insert(current_node.clone()) {
            bail!(
                "partial_contract: output_order currently requires each requested index to occupy a distinct node"
            );
        }
        ordered_nodes.push(current_node.clone());
    }

    for node_name in result.node_names() {
        if ordered_node_set.insert(node_name.clone()) {
            ordered_nodes.push(node_name);
        }
    }

    let tensors = ordered_nodes
        .iter()
        .map(|node_name| {
            let node_idx = result.node_index(node_name).ok_or_else(|| {
                anyhow!(
                    "partial_contract: output_order node {:?} is not present in the result",
                    node_name
                )
            })?;
            result.tensor(node_idx).cloned().ok_or_else(|| {
                anyhow!(
                    "partial_contract: tensor for output_order node {:?} is missing",
                    node_name
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut reordered = TreeTN::from_tensors(tensors, ordered_nodes)
        .context("partial_contract: failed to rebuild result in requested output order")?;
    reordered.canonical_region = result.canonical_region.clone();
    reordered.canonical_form = result.canonical_form;
    reordered.ortho_towards = result.ortho_towards.clone();
    Ok(reordered)
}

fn diagonal_copy_value(tensor: &TensorDynLen) -> AnyScalar {
    if tensor.is_complex() {
        AnyScalar::new_complex(1.0, 0.0)
    } else {
        AnyScalar::new_real(1.0)
    }
}

fn apply_diagonal_pairs<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    diagonal_pairs: &[(DynIndex, DynIndex)],
) -> Result<DiagonalPairApplication<V>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let mut a_modified = a.clone();
    let mut b_modified = b.clone();
    let mut restore_from = Vec::with_capacity(diagonal_pairs.len());
    let mut restore_to = Vec::with_capacity(diagonal_pairs.len());

    for (idx_a, idx_b) in diagonal_pairs {
        let node_name = a_modified
            .site_index_network()
            .find_node_by_index(idx_a)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "partial_contract: diagonal pair left index {:?} is not a site index of the first TreeTN",
                    idx_a.id()
                )
            })?;
        let node_idx = a_modified.node_index(&node_name).ok_or_else(|| {
            anyhow!(
                "partial_contract: node {:?} for left diagonal index {:?} not found",
                node_name,
                idx_a.id()
            )
        })?;
        let local_tensor = a_modified.tensor(node_idx).cloned().ok_or_else(|| {
            anyhow!(
                "partial_contract: tensor for node {:?} not found while processing diagonal pair {:?}",
                node_name,
                idx_a.id()
            )
        })?;

        let aux_index = idx_a.sim();
        let kept_index = idx_a.sim();
        let copy_tensor = TensorDynLen::copy_tensor(
            vec![idx_a.clone(), aux_index.clone(), kept_index.clone()],
            diagonal_copy_value(&local_tensor),
        )
        .with_context(|| {
            format!(
                "partial_contract: failed to build copy tensor for diagonal pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;
        let expanded_tensor = local_tensor
            .tensordot(&copy_tensor, &[(idx_a.clone(), idx_a.clone())])
            .with_context(|| {
                format!(
                    "partial_contract: failed to apply diagonal structure for pair {:?} <- {:?}",
                    idx_a.id(),
                    idx_b.id()
                )
            })?;
        a_modified
            .replace_tensor(node_idx, expanded_tensor)
            .with_context(|| {
                format!(
                    "partial_contract: failed to replace tensor at node {:?} for diagonal pair {:?}",
                    node_name,
                    idx_a.id()
                )
            })?
            .ok_or_else(|| {
                anyhow!(
                    "partial_contract: node {:?} disappeared while processing diagonal pair {:?}",
                    node_name,
                    idx_a.id()
                )
            })?;

        b_modified = b_modified.replaceind(idx_b, &aux_index).with_context(|| {
            format!(
                "partial_contract: failed to align diagonal pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;

        restore_from.push(kept_index);
        restore_to.push(idx_a.clone());
    }

    Ok((a_modified, b_modified, restore_from, restore_to))
}

/// Partially contract two TreeTNs according to the given specification.
///
/// # Arguments
/// * `a` - First tensor network
/// * `b` - Second tensor network
/// * `spec` - Which site indices to contract versus link through diagonal
///   structure
/// * `center` - Canonical center node for the result
/// * `options` - Contraction algorithm options
///
/// # Index handling
///
/// - **contract_pairs**: Both indices are traced over (inner product).
///   Neither appears in the result.
/// - **diagonal_pairs**: The two indices are linked through explicit diagonal
///   structure so that only matching values contribute, while the left-hand site
///   index remains in the result.
/// - **Unmentioned indices**: Pass through unchanged as external legs.
///
/// # Examples
///
/// ```no_run
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{
///     contraction::ContractionOptions,
///     partial_contract,
///     PartialContractionSpec,
///     TreeTN,
/// };
///
/// let idx_a = DynIndex::new_dyn(2);
/// let idx_b = DynIndex::new_dyn(2);
/// let a = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![idx_a.clone()], vec![1.0, 2.0]).unwrap()],
///     vec![0],
/// ).unwrap();
/// let b = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![idx_b.clone()], vec![3.0, 4.0]).unwrap()],
///     vec![0],
/// ).unwrap();
///
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
///     diagonal_pairs: vec![],
///     output_order: None,
/// };
///
/// let result = partial_contract(&a, &b, &spec, &0usize, ContractionOptions::default()).unwrap();
/// assert_eq!(result.node_count(), 1);
/// ```
pub fn partial_contract<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    spec: &PartialContractionSpec<DynIndex>,
    center: &V,
    options: ContractionOptions,
) -> Result<TreeTN<TensorDynLen, V>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    validate_partial_contraction_spec(a, b, spec)?;

    let (a_modified, mut b_modified, restore_from, restore_to) =
        apply_diagonal_pairs(a, b, &spec.diagonal_pairs)?;

    for (idx_a, idx_b) in &spec.contract_pairs {
        b_modified = b_modified.replaceind(idx_b, idx_a).with_context(|| {
            format!(
                "partial_contract: failed to align contract pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;
    }

    let mut result = if a_modified.same_topology(&b_modified) {
        contract(&a_modified, &b_modified, center, options)
            .context("partial_contract: contraction failed")?
    } else {
        contract_mismatched_topologies(&a_modified, &b_modified, center, options)?
    };

    if !restore_from.is_empty() {
        result = result.replaceinds(&restore_from, &restore_to).context(
            "partial_contract: failed to restore surviving left-hand indices after diagonal pairing",
        )?;
    }

    if let Some(output_order) = &spec.output_order {
        apply_output_order(result, output_order)
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
