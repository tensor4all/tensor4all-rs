//! Site-fixing operations for general-tree `TreeTN`s.
//!
//! [`fix_and_remove_site`] restricts one node to a fixed site coordinate and
//! then removes that node, preserving the names of the surviving nodes. It is
//! the general-tree, name-addressed replacement for the former
//! position-addressed chain helper, so it works for combs and other non-path
//! topologies without flattening to a `SimpleTensorTrain`.

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{IndexLike, TensorLike};

use super::TreeTN;
use crate::error::TreeTNOperationError;

/// Fix the site at `node` to `value` and remove `node` from the tree.
///
/// The node is contracted into one surviving neighbour, so the represented
/// tensor is unchanged apart from the removed site. Surviving node names are
/// preserved: no renumbering happens, and callers can keep addressing the
/// result with the names they already have.
///
/// # Arguments
///
/// * `treetn` - The tree to restrict.
/// * `node` - Name of the node to fix and remove. It must carry exactly one
///   site index; site-free nodes and multi-site nodes are rejected.
/// * `value` - Local coordinate kept at the removed site, in
///   `0..site_dimension`.
///
/// # Returns
///
/// A new `TreeTN` with `node` removed and the fixed site dropped from the
/// external index list. All other nodes keep their names.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] when `node` is not part of `treetn`, when
/// `node` does not carry exactly one site index, when `value` is outside the
/// removed site's dimension, when `node` is isolated (removing it would leave
/// an empty network), or when the site projection or the neighbour contraction
/// fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::{fix_and_remove_site, TreeTN};
///
/// # fn main() -> anyhow::Result<()> {
/// let site_a = DynIndex::new_dyn(2);
/// let site_b = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(2);
/// let a = IdxTensor::from_dense(
///     vec![site_a, bond.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0],
/// )?;
/// let b = IdxTensor::from_dense(
///     vec![bond, site_b],
///     vec![1.0_f64, 2.0, 3.0, 4.0],
/// )?;
/// let tn = TreeTN::<IdxTensor, String>::from_tensors(
///     vec![a, b],
///     vec!["a".to_string(), "b".to_string()],
/// )?;
///
/// let reduced = fix_and_remove_site(tn, &"b".to_string(), 1)?;
///
/// assert_eq!(reduced.node_names(), vec!["a".to_string()]);
/// assert_eq!(reduced.contract_to_tensor()?.to_vec::<f64>()?, vec![3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub fn fix_and_remove_site<T, V>(
    treetn: TreeTN<T, V>,
    node: &V,
    value: usize,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    T::Index: IndexLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let node_index = treetn.node_index(node).ok_or_else(|| {
        TreeTNOperationError::from(anyhow::anyhow!(
            "fix_and_remove_site: node {node:?} is not part of the TreeTN"
        ))
    })?;

    let site_space = treetn.site_space(node).ok_or_else(|| {
        TreeTNOperationError::from(anyhow::anyhow!(
            "fix_and_remove_site: node {node:?} has no site space"
        ))
    })?;
    if site_space.len() != 1 {
        return Err(TreeTNOperationError::from(anyhow::anyhow!(
            "fix_and_remove_site: node {node:?} must carry exactly one site index, got {}",
            site_space.len()
        )));
    }
    let site_index = site_space
        .iter()
        .next()
        .ok_or_else(|| {
            TreeTNOperationError::from(anyhow::anyhow!(
                "fix_and_remove_site: node {node:?} has no site index"
            ))
        })?
        .clone();
    if value >= site_index.dim() {
        return Err(TreeTNOperationError::from(anyhow::anyhow!(
            "fix_and_remove_site: value {value} is out of range for site dimension {}",
            site_index.dim()
        )));
    }

    let neighbors: Vec<V> = treetn.site_index_network().neighbors(node).collect();
    if neighbors.is_empty() {
        return Err(TreeTNOperationError::from(anyhow::anyhow!(
            "fix_and_remove_site: node {node:?} is isolated; removing it would leave an empty network"
        )));
    }

    // Deterministic merge target: the neighbour with the smallest node index.
    // Any neighbour is correct for value preservation; fixing the choice keeps
    // output names and bond layout reproducible.
    let parent = neighbors
        .iter()
        .min_by_key(|neighbor| treetn.node_index(neighbor).map(|index| index.index()))
        .cloned()
        .ok_or_else(|| {
            TreeTNOperationError::from(anyhow::anyhow!(
                "fix_and_remove_site: node {node:?} has no neighbour to absorb it"
            ))
        })?;

    let node_tensor = treetn.tensor(node_index).ok_or_else(|| {
        TreeTNOperationError::from(anyhow::anyhow!(
            "fix_and_remove_site: tensor for node {node:?} is missing"
        ))
    })?;
    let sliced = node_tensor
        .select_indices(std::slice::from_ref(&site_index), &[value])
        .map_err(|error| TreeTNOperationError::from(anyhow::Error::new(error)))?;

    let parent_index = treetn.node_index(&parent).ok_or_else(|| {
        TreeTNOperationError::from(anyhow::anyhow!(
            "fix_and_remove_site: neighbour {parent:?} is missing from the TreeTN"
        ))
    })?;
    let parent_tensor = treetn.tensor(parent_index).ok_or_else(|| {
        TreeTNOperationError::from(anyhow::anyhow!(
            "fix_and_remove_site: tensor for neighbour {parent:?} is missing"
        ))
    })?;
    let merged = T::contract(&[parent_tensor, &sliced])
        .map_err(|error| TreeTNOperationError::from(anyhow::Error::new(error)))?;

    // Rebuild from the surviving tensors. `from_tensors` reconnects nodes that
    // share a bond index, so the neighbours of the removed node attach to
    // `parent` while every surviving name is preserved.
    let mut tensors = Vec::with_capacity(treetn.node_count() - 1);
    let mut names = Vec::with_capacity(treetn.node_count() - 1);
    for name in treetn.node_names() {
        if &name == node {
            continue;
        }
        let tensor = if name == parent {
            merged.clone()
        } else {
            let index = treetn.node_index(&name).ok_or_else(|| {
                TreeTNOperationError::from(anyhow::anyhow!(
                    "fix_and_remove_site: node {name:?} is missing from the TreeTN"
                ))
            })?;
            treetn
                .tensor(index)
                .ok_or_else(|| {
                    TreeTNOperationError::from(anyhow::anyhow!(
                        "fix_and_remove_site: tensor for node {name:?} is missing"
                    ))
                })?
                .clone()
        };
        tensors.push(tensor);
        names.push(name);
    }

    TreeTN::from_tensors(tensors, names)
}
