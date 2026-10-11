//! Patch networks: exact small patches, the layout check of engine outcomes,
//! and the re-embedding of fixed sites shared by every engine.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{outer_product, CommonScalar, DynIndex, IdxTensor, TensorElement};
use tensor4all_treetn::TreeTN;

use crate::error::PartitionedTreeTNError;

use super::layout::SiteLayout;

type Result<T> = std::result::Result<T, PartitionedTreeTNError>;

/// A one-hot vector over `site` with its entry at `coordinate`, built from the
/// patch scalar type so that every tensor of a patch has one dtype.
fn one_hot<T>(site: &DynIndex, coordinate: usize) -> Result<IdxTensor>
where
    T: CommonScalar + TensorElement,
{
    let mut data = vec![T::from_f64(0.0); site.dim];
    data[coordinate] = T::from_f64(1.0);
    Ok(IdxTensor::from_dense(vec![site.clone()], data)?)
}

/// Re-attach every fixed site to its node with a one-hot factor.
///
/// `network` carries the problem's node names and only the active sites. The
/// result carries every site of the layout and is already masked by the
/// fixed coordinates.
pub(super) fn embed_fixed_sites<T, V>(
    network: &TreeTN<IdxTensor, V>,
    layout: &SiteLayout<V>,
    fixed: &[Option<usize>],
) -> Result<TreeTN<IdxTensor, V>>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let mut names = Vec::with_capacity(layout.node_sites.len());
    let mut tensors = Vec::with_capacity(layout.node_sites.len());
    for (node, positions) in &layout.node_positions {
        let tensor = network
            .node_index(node)
            .and_then(|index| network.tensor(index))
            .ok_or_else(|| PartitionedTreeTNError::tree(format!("node {node:?} has no tensor")))?;
        let mut tensor = tensor.clone();
        for &position in positions {
            if let Some(coordinate) = fixed[position] {
                let factor = one_hot::<T>(&layout.sites[position], coordinate)?;
                tensor = outer_product(&tensor, &factor)?;
            }
        }
        names.push(node.clone());
        tensors.push(tensor);
    }
    Ok(TreeTN::from_tensors(tensors, names)?)
}

/// Build the network of a patch with at most one active site from its exact
/// values, over the active sites only, with dimension-one links.
///
/// The values sit on the node that carries the active site, or on the
/// smallest node name when no site is active; every other node holds a one.
pub(super) fn exact_active_network<T, V>(
    layout: &SiteLayout<V>,
    active: &[usize],
    mut values: Vec<T>,
) -> Result<TreeTN<IdxTensor, V>>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let value_node = match active {
        [] => layout.node_sites.keys().next(),
        [position] => Some(&layout.site_nodes[*position]),
        _ => None,
    }
    .ok_or_else(|| PartitionedTreeTNError::tree("an exact patch has at most one active site"))?;

    let mut links: BTreeMap<&V, Vec<DynIndex>> = BTreeMap::new();
    for (left, right) in &layout.edges {
        let link = DynIndex::new_dyn(1);
        links.entry(left).or_default().push(link.clone());
        links.entry(right).or_default().push(link);
    }

    let mut names = Vec::with_capacity(layout.node_sites.len());
    let mut tensors = Vec::with_capacity(layout.node_sites.len());
    for node in layout.node_sites.keys() {
        let mut indices = Vec::new();
        let data = if node == value_node {
            indices.extend(
                active
                    .iter()
                    .map(|&position| layout.sites[position].clone()),
            );
            std::mem::take(&mut values)
        } else {
            vec![T::from_f64(1.0)]
        };
        indices.extend(links.remove(node).unwrap_or_default());
        names.push(node.clone());
        tensors.push(IdxTensor::from_dense(indices, data)?);
    }
    Ok(TreeTN::from_tensors(tensors, names)?)
}

/// Check that an engine outcome has the problem's topology and exactly the
/// active sites of every node (full identity and dimension).
pub(super) fn check_outcome_layout<V>(
    network: &TreeTN<IdxTensor, V>,
    layout: &SiteLayout<V>,
    fixed: &[Option<usize>],
) -> std::result::Result<(), String>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    if network.node_count() != layout.node_sites.len() {
        return Err(format!(
            "the outcome network has {} nodes, the topology {}",
            network.node_count(),
            layout.node_sites.len()
        ));
    }
    for (node, positions) in &layout.node_positions {
        let Some(space) = network.site_space(node) else {
            return Err(format!("the outcome network has no node {node:?}"));
        };
        let active: Vec<&DynIndex> = positions
            .iter()
            .filter(|&&position| fixed[position].is_none())
            .map(|&position| &layout.sites[position])
            .collect();
        let matches = space.len() == active.len()
            && active
                .iter()
                .all(|site| space.get(*site).is_some_and(|found| found.dim == site.dim));
        if !matches {
            return Err(format!(
                "node {node:?} of the outcome network carries {space:?}, expected the active sites {active:?}"
            ));
        }
    }
    for (left, right) in &layout.edges {
        if network.edge_between(left, right).is_none() {
            return Err(format!(
                "the outcome network lacks the edge {left:?} - {right:?}"
            ));
        }
    }
    Ok(())
}
