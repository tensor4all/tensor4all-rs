//! Exact local pair contraction and shared contraction topology planning.

use super::*;
use tensor4all_core::LinearizationOrder;

/// Node-retention policy for exact local contraction and zip-up contraction.
///
/// Use [`Self::PreserveInputTopology`] to retain every named node, or
/// [`Self::PruneScalarSubtrees`] to match ordinary zip-up's emitted topology.
/// Neither policy changes the factorization or truncation settings of zip-up.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{TreeTN, contraction::{contract_local_exact, ContractionTopology}};
/// let bond = DynIndex::new_dyn(2);
/// let tree = TreeTN::from_tensors(vec![
///     IdxTensor::from_dense(vec![bond.clone()], vec![1.0, 2.0])?,
///     IdxTensor::from_dense(vec![bond], vec![3.0, 4.0])?], vec![0usize, 1])?;
/// for (topology, nodes) in [(ContractionTopology::PruneScalarSubtrees, 1),
///                           (ContractionTopology::PreserveInputTopology, 2)] {
///     let result = contract_local_exact(&tree, &tree, &0, topology)?;
///     assert_eq!(result.node_count(), nodes);
///     assert_eq!(result.to_dense()?.to_vec::<f64>()?, vec![121.0]);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractionTopology {
    /// Match zip-up's scalar-subtree pruning, retaining scalar bridge nodes.
    ///
    /// The chain schedule also follows its final two-node block convention:
    /// a scalar endpoint can disappear while an adjacent scalar bridge stays.
    /// The rooted-tree schedule retains `center` for a fully contracted scalar.
    /// The chain schedule follows legacy routing by full site-space inequality,
    /// including directed bra/ket spaces that contract to a scalar. Node
    /// retention is planned from the tree, center, and full
    /// external index identities, without sampled values.
    PruneScalarSubtrees,
    /// Keep all input nodes and edges, as Fit, SRC, and naive operator apply do.
    PreserveInputTopology,
}

/// Contract corresponding site tensors exactly and fuse each pair of links.
///
/// `a` and `b` must have the same nonempty named tree topology, and matching
/// contractable site indices must belong to the same node. `center` is an
/// existing node used to orient topology cleanup; it is not a canonical-gauge
/// certificate. `topology` chooses whether to retain all nodes or match
/// ordinary zip-up's pruning convention.
///
/// This performs no factorization or truncation and never materializes a full
/// network tensor. Retained bonds have the product of the corresponding input
/// dimensions. Site-free nodes omitted by `topology` are absorbed through local
/// pair contractions only. The result has no canonical region or form.
///
/// # Returns
///
/// A new exact TreeTN. If either operand has dimension one on every bond, each
/// retained bond has the other operand's dimension; arbitrary local matrices,
/// signs, singular maps, and copy tensors satisfy this rule.
///
/// # Errors
///
/// Returns [`TreeTNOperationError::InvalidGraph`] for an empty, invalid, or
/// mismatched tree, an absent center, or shared site indices on different nodes;
/// [`TreeTNOperationError::InvalidConfiguration`] when a product bond dimension
/// overflows; and a source-preserving tensor/backend error if local contraction
/// or link fusion fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{TreeTN, contraction::{contract_local_exact, ContractionTopology}};
/// let x = DynIndex::new_dyn(2);
/// let left = TreeTN::from_tensors(vec![
///     IdxTensor::from_dense(vec![x.clone()], vec![1.0, 2.0])?], vec![0usize])?;
/// let right = TreeTN::from_tensors(vec![
///     IdxTensor::from_dense(vec![x], vec![3.0, 4.0])?], vec![0usize])?;
/// let result = contract_local_exact(&left, &right, &0,
///     ContractionTopology::PruneScalarSubtrees)?;
/// assert_eq!(result.to_dense()?.to_vec::<f64>()?, vec![11.0]);
/// assert!(result.canonical_region().is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_local_exact<T, V>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    center: &V,
    topology: ContractionTopology,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    a.validate_tree()?;
    b.validate_tree()?;
    if a.node_count() == 0 || !a.same_topology(b) || a.node_index(center).is_none() {
        return Err(TreeTNOperationError::invalid_graph(anyhow::anyhow!(
            "exact local contraction needs matching nonempty trees and an existing center"
        )));
    }
    let (b_indices, b_nodes) = b.all_site_indices()?;
    let b_sites: HashMap<_, _> = b_indices.into_iter().zip(b_nodes).collect();
    let (a_indices, a_nodes) = a.all_site_indices()?;
    for (index, node) in a_indices.into_iter().zip(a_nodes) {
        // Full conjugate identity pairs directed bra/ket indices; undirected
        // indices conjugate to themselves. Prime levels and tags stay distinct.
        if let Some(other_node) = b_sites.get(&index.conj()) {
            if other_node != &node {
                return Err(TreeTNOperationError::invalid_graph(anyhow::anyhow!(
                    "exact local contraction: shared site index is assigned to different nodes"
                )));
            }
        }
    }
    for (left, right) in a.site_index_network().edges() {
        let dimension = |tree: &TreeTN<T, V>| {
            tree.edge_between(&left, &right)
                .and_then(|edge| tree.bond_index(edge))
                .map(IndexLike::dim)
        };
        let (left_dim, right_dim) = dimension(a).zip(dimension(b)).ok_or_else(|| {
            TreeTNOperationError::invalid_graph(anyhow::anyhow!(
                "exact local contraction: missing edge bond"
            ))
        })?;
        checked_product_bond_dimension(left_dim, right_dim)?;
    }
    let has_outputs = output_site_nodes(a, b);
    let chain = zipup_chain_order(a, b, center);
    let retained = output_nodes(a, center, topology, chain.as_deref(), has_outputs)?;
    let result = contract_local_exact_impl(a, b)?;
    prune_to_nodes(result, center, &retained).map_err(TreeTNOperationError::from)
}

/// Preserve zip-up's established chain-vs-rooted dispatch, including directed
/// bra/ket site spaces whose full sets differ despite having no output axes.
pub(super) fn zipup_chain_order<T, V>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    center: &V,
) -> Option<Vec<V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    a.node_names()
        .iter()
        .any(|node| distinct_site_spaces(a.site_space(node), b.site_space(node)))
        .then(|| a.chain_order(center))
        .flatten()
}

pub(super) fn distinct_site_spaces<I: IndexLike>(
    a: Option<&HashSet<I>>,
    b: Option<&HashSet<I>>,
) -> bool {
    match (a, b) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some(a), Some(b)) => a != b,
    }
}

/// Validate the fused dimension before any local contraction allocates output.
pub(super) fn checked_product_bond_dimension(
    left: usize,
    right: usize,
) -> std::result::Result<usize, TreeTNOperationError> {
    left.checked_mul(right).ok_or_else(|| {
        TreeTNOperationError::invalid_configuration(anyhow::anyhow!(
            "exact local contraction: product bond dimension overflows"
        ))
    })
}

/// Nodes with surviving local axes, using full contractable index identities.
pub(super) fn output_site_nodes<T, V>(a: &TreeTN<T, V>, b: &TreeTN<T, V>) -> HashSet<V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    a.node_names()
        .into_iter()
        .filter(|node| {
            let a_sites = a.site_space(node);
            let b_sites = b.site_space(node);
            a_sites.into_iter().flatten().any(|index| {
                !b_sites
                    .and_then(|sites| sites.get(&index.conj()))
                    .is_some_and(|other| index.is_contractable(other))
            }) || b_sites.into_iter().flatten().any(|index| {
                !a_sites
                    .and_then(|sites| sites.get(&index.conj()))
                    .is_some_and(|other| index.is_contractable(other))
            })
        })
        .collect()
}

/// Metadata-only plan shared by exact local cleanup and zip-up emission.
pub(super) fn output_nodes<T, V>(
    a: &TreeTN<T, V>,
    center: &V,
    topology: ContractionTopology,
    chain: Option<&[V]>,
    mut retained: HashSet<V>,
) -> Result<HashSet<V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    if topology == ContractionTopology::PreserveInputTopology {
        return Ok(a.node_names().into_iter().collect());
    }
    if let Some(chain) = chain.filter(|chain| chain.len() >= 2) {
        let n = chain.len();
        let mut emitted = HashSet::with_capacity(n);
        let mut has_left_bond = false;
        for node in &chain[..n - 2] {
            has_left_bond |= retained.contains(node);
            if has_left_bond {
                emitted.insert(node.clone());
            }
        }
        let left = has_left_bond || retained.contains(&chain[n - 2]);
        let right = retained.contains(&chain[n - 1]);
        if left {
            emitted.insert(chain[n - 2].clone());
        }
        if !left || right {
            emitted.insert(chain[n - 1].clone());
        }
        return Ok(emitted);
    }
    let edges = a
        .edges_to_canonicalize_by_names(center)
        .ok_or_else(|| anyhow::anyhow!("contraction topology plan: center not found"))?;
    for (source, destination) in edges {
        if retained.contains(&source) {
            retained.insert(destination);
        }
    }
    retained.insert(center.clone());
    Ok(retained)
}

fn prune_to_nodes<T, V>(
    mut tree: TreeTN<T, V>,
    center: &V,
    retained: &HashSet<V>,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    if retained.len() == tree.node_count() {
        return Ok(tree);
    }
    let root = if retained.contains(center) {
        center
    } else {
        retained
            .iter()
            .min()
            .ok_or_else(|| anyhow::anyhow!("exact cleanup has no retained node"))?
    };
    let edges = tree
        .edges_to_canonicalize_by_names(root)
        .ok_or_else(|| anyhow::anyhow!("exact cleanup root is absent"))?;
    let names = tree.node_names();
    let mut tensors: HashMap<_, _> = names
        .iter()
        .map(|name| {
            // INVARIANT: traversal and node names were captured above; this
            // consumed tree is never queried again after taking its tensors.
            let tensor = tree
                .graph
                .remove_node(name)
                .ok_or_else(|| anyhow::anyhow!("exact cleanup tensor is absent"))?;
            Ok((name.clone(), tensor))
        })
        .collect::<Result<_>>()?;
    for (source, destination) in edges {
        if retained.contains(&source) {
            continue;
        }
        let source_tensor = tensors
            .remove(&source)
            .ok_or_else(|| anyhow::anyhow!("exact cleanup source is absent"))?;
        let destination_tensor = tensors
            .get_mut(&destination)
            .ok_or_else(|| anyhow::anyhow!("exact cleanup destination is absent"))?;
        *destination_tensor = destination_tensor.contract_pair(&source_tensor)?;
    }
    let names: Vec<_> = names
        .into_iter()
        .filter(|node| retained.contains(node))
        .collect();
    let tensors = names
        .iter()
        .map(|node| {
            tensors
                .remove(node)
                .ok_or_else(|| anyhow::anyhow!("exact cleanup retained tensor is absent"))
        })
        .collect::<Result<_>>()?;
    TreeTN::from_tensors(tensors, names).map_err(anyhow::Error::from)
}

fn contract_local_exact_impl<T, V>(
    transformed_state: &TreeTN<T, V>,
    mpo: &TreeTN<T, V>,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let state = transformed_state.sim_internal_inds();
    let mpo = mpo.sim_internal_inds();

    let mut node_names = state.node_names();
    node_names.sort();

    let mut tensors_by_node: HashMap<V, T> = HashMap::with_capacity(node_names.len());
    for node in &node_names {
        let state_node = state.node_index(node).ok_or_else(|| {
            anyhow::anyhow!("contract_local_exact: missing state node {:?}", node)
        })?;
        let mpo_node = mpo.node_index(node).ok_or_else(|| {
            anyhow::anyhow!("contract_local_exact: missing operator node {:?}", node)
        })?;
        let state_tensor = state.tensor(state_node).ok_or_else(|| {
            anyhow::anyhow!("contract_local_exact: missing state tensor at {:?}", node)
        })?;
        let mpo_tensor = mpo.tensor(mpo_node).ok_or_else(|| {
            anyhow::anyhow!(
                "contract_local_exact: missing operator tensor at {:?}",
                node
            )
        })?;

        // A node may carry only distinct external axes on the two operands,
        // even when both site spaces are nonempty. The binary primitive
        // handles that outer product as well as shared-index contraction.
        let contracted = state_tensor
            .contract_pair(mpo_tensor)
            .with_context(|| format!("contract_local_exact: failed at {:?}", node))?;
        tensors_by_node.insert(node.clone(), contracted);
    }

    let mut edges: Vec<(V, V)> = state.site_index_network().edges().collect();
    edges.sort();
    for (node_a, node_b) in edges {
        let state_edge = state.edge_between(&node_a, &node_b).ok_or_else(|| {
            anyhow::anyhow!(
                "contract_local_exact: missing state edge {:?}-{:?}",
                node_a,
                node_b
            )
        })?;
        let mpo_edge = mpo.edge_between(&node_a, &node_b).ok_or_else(|| {
            anyhow::anyhow!(
                "contract_local_exact: missing operator edge {:?}-{:?}",
                node_a,
                node_b
            )
        })?;
        let state_bond = state.bond_index(state_edge).ok_or_else(|| {
            anyhow::anyhow!(
                "contract_local_exact: missing state bond {:?}-{:?}",
                node_a,
                node_b
            )
        })?;
        let mpo_bond = mpo.bond_index(mpo_edge).ok_or_else(|| {
            anyhow::anyhow!(
                "contract_local_exact: missing operator bond {:?}-{:?}",
                node_a,
                node_b
            )
        })?;
        let old_indices = [state_bond.clone(), mpo_bond.clone()];
        let product_link = T::Index::product_link(&old_indices).with_context(|| {
            format!(
                "contract_local_exact: failed to create product link for {:?}-{:?}",
                node_a, node_b
            )
        })?;

        for node in [&node_a, &node_b] {
            let tensor = tensors_by_node.get_mut(node).ok_or_else(|| {
                anyhow::anyhow!(
                    "contract_local_exact: missing contracted tensor at {:?}",
                    node
                )
            })?;
            *tensor = tensor
                .fuse_indices(
                    &old_indices,
                    product_link.clone(),
                    LinearizationOrder::ColumnMajor,
                )
                .with_context(|| {
                    format!(
                        "contract_local_exact: failed to fuse product link at {:?}",
                        node
                    )
                })?;
        }
    }

    let tensors = node_names
        .iter()
        .map(|node| {
            tensors_by_node.remove(node).ok_or_else(|| {
                anyhow::anyhow!("contract_local_exact: missing result tensor for {:?}", node)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let result = TreeTN::from_tensors(tensors, node_names)
        .context("contract_local_exact: failed to build result TreeTN")?;

    Ok(result)
}
