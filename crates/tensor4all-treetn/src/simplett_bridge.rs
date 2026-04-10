use anyhow::{ensure, Result};
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorElement};
use tensor4all_simplett::{AbstractTensorTrain, TTScalar, Tensor3Ops, TensorTrain};

use crate::TreeTN;

/// Convert a linear-chain simple tensor train into a `TreeTN` with node names `0..n-1`.
///
/// The returned site indices are ordered by tensor-train site position, which is
/// convenient for downstream state/layout bookkeeping.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{IndexLike, TensorIndex, TensorLike};
/// use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, TensorTrain};
/// use tensor4all_treetn::tensor_train_to_treetn;
///
/// let tt = TensorTrain::new(vec![
///     tensor3_from_data(vec![1.0_f64, 2.0], 1, 2, 1),
/// ]).unwrap();
///
/// let (treetn, site_indices) = tensor_train_to_treetn(&tt).unwrap();
/// let dense = treetn.contract_to_tensor().unwrap();
///
/// assert_eq!(treetn.node_names(), vec![0]);
/// assert_eq!(site_indices.len(), 1);
/// assert_eq!(dense.external_indices()[0].id(), site_indices[0].id());
/// assert_eq!(dense.dims(), vec![2]);
/// ```
pub fn tensor_train_to_treetn<T>(
    tt: &TensorTrain<T>,
) -> Result<(TreeTN<TensorDynLen, usize>, Vec<DynIndex>)>
where
    T: TTScalar + TensorElement + Clone,
{
    tensor_train_to_treetn_with_names(tt, (0..tt.len()).collect())
}

/// Convert a linear-chain simple tensor train into a `TreeTN` with explicit node names.
///
/// The returned site indices are ordered by tensor-train site position, not by
/// sorted node-name order.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{tensor3_from_data, TensorTrain};
/// use tensor4all_treetn::tensor_train_to_treetn_with_names;
///
/// let tt = TensorTrain::new(vec![
///     tensor3_from_data(vec![1.0_f64, 2.0], 1, 2, 1),
/// ]).unwrap();
///
/// let (treetn, site_indices) =
///     tensor_train_to_treetn_with_names(&tt, vec!["site0".to_string()]).unwrap();
///
/// assert_eq!(treetn.node_names(), vec!["site0".to_string()]);
/// assert_eq!(site_indices.len(), 1);
/// ```
pub fn tensor_train_to_treetn_with_names<T, V>(
    tt: &TensorTrain<T>,
    node_names: Vec<V>,
) -> Result<(TreeTN<TensorDynLen, V>, Vec<DynIndex>)>
where
    T: TTScalar + TensorElement + Clone,
    V: Clone + std::hash::Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    tensor_train_to_treetn_impl(tt, node_names, None)
}

/// Convert a linear-chain simple tensor train into a `TreeTN` with explicit node names
/// and caller-provided site indices.
///
/// This is useful when downstream code must preserve external site-index identities
/// across a conversion boundary while still allowing internal bond indices to be
/// created fresh.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_simplett::{tensor3_from_data, TensorTrain};
/// use tensor4all_treetn::tensor_train_to_treetn_with_names_and_site_indices;
///
/// let tt = TensorTrain::new(vec![
///     tensor3_from_data(vec![1.0_f64, 2.0], 1, 2, 1),
/// ]).unwrap();
/// let site = DynIndex::new_dyn(2);
///
/// let treetn = tensor_train_to_treetn_with_names_and_site_indices(
///     &tt,
///     vec!["site0".to_string()],
///     vec![site],
/// ).unwrap();
///
/// assert_eq!(treetn.node_names(), vec!["site0".to_string()]);
/// ```
pub fn tensor_train_to_treetn_with_names_and_site_indices<T, V>(
    tt: &TensorTrain<T>,
    node_names: Vec<V>,
    site_indices: Vec<DynIndex>,
) -> Result<TreeTN<TensorDynLen, V>>
where
    T: TTScalar + TensorElement + Clone,
    V: Clone + std::hash::Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let (treetn, _) = tensor_train_to_treetn_impl(tt, node_names, Some(site_indices))?;
    Ok(treetn)
}

fn tensor_train_to_treetn_impl<T, V>(
    tt: &TensorTrain<T>,
    node_names: Vec<V>,
    site_indices: Option<Vec<DynIndex>>,
) -> Result<(TreeTN<TensorDynLen, V>, Vec<DynIndex>)>
where
    T: TTScalar + TensorElement + Clone,
    V: Clone + std::hash::Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    ensure!(
        tt.len() == node_names.len(),
        "tensor_train_to_treetn: node_names length {} must match tensor-train length {}",
        node_names.len(),
        tt.len()
    );

    if tt.is_empty() {
        let site_indices = site_indices.unwrap_or_default();
        ensure!(
            site_indices.is_empty(),
            "tensor_train_to_treetn: empty tensor train requires zero site indices"
        );
        return Ok((TreeTN::new(), Vec::new()));
    }

    let site_indices = match site_indices {
        Some(indices) => {
            ensure!(
                indices.len() == tt.len(),
                "tensor_train_to_treetn: site_indices length {} must match tensor-train length {}",
                indices.len(),
                tt.len()
            );
            for (site, index) in indices.iter().enumerate() {
                ensure!(
                    index.dim() == tt.site_dim(site),
                    "tensor_train_to_treetn: site index {} has dim {} but tensor-train site {} has dim {}",
                    site,
                    index.dim(),
                    site,
                    tt.site_dim(site)
                );
            }
            indices
        }
        None => tt.site_dims().into_iter().map(DynIndex::new_dyn).collect(),
    };
    let bond_indices: Vec<DynIndex> = tt
        .link_dims()
        .into_iter()
        .map(DynIndex::new_bond)
        .collect::<Result<_>>()?;
    let nsites = tt.len();

    let mut tensors = Vec::with_capacity(nsites);
    for site in 0..nsites {
        let site_tensor = tt.site_tensor(site);
        let tensor = if nsites == 1 {
            let data = single_site_data(site_tensor);
            TensorDynLen::from_dense(vec![site_indices[site].clone()], data)?
        } else if site == 0 {
            let data = left_boundary_data(site_tensor);
            TensorDynLen::from_dense(
                vec![site_indices[site].clone(), bond_indices[site].clone()],
                data,
            )?
        } else if site + 1 == nsites {
            let data = right_boundary_data(site_tensor);
            TensorDynLen::from_dense(
                vec![bond_indices[site - 1].clone(), site_indices[site].clone()],
                data,
            )?
        } else {
            let data = middle_site_data(site_tensor);
            TensorDynLen::from_dense(
                vec![
                    bond_indices[site - 1].clone(),
                    site_indices[site].clone(),
                    bond_indices[site].clone(),
                ],
                data,
            )?
        };
        tensors.push(tensor);
    }

    let treetn = TreeTN::from_tensors(tensors, node_names)?;
    Ok((treetn, site_indices))
}

fn single_site_data<T>(tensor: &tensor4all_simplett::Tensor3<T>) -> Vec<T>
where
    T: TTScalar + Clone,
{
    let mut data = Vec::with_capacity(tensor.site_dim());
    for s in 0..tensor.site_dim() {
        data.push(*tensor.get3(0, s, 0));
    }
    data
}

fn left_boundary_data<T>(tensor: &tensor4all_simplett::Tensor3<T>) -> Vec<T>
where
    T: TTScalar + Clone,
{
    let mut data = Vec::with_capacity(tensor.site_dim() * tensor.right_dim());
    for r in 0..tensor.right_dim() {
        for s in 0..tensor.site_dim() {
            data.push(*tensor.get3(0, s, r));
        }
    }
    data
}

fn right_boundary_data<T>(tensor: &tensor4all_simplett::Tensor3<T>) -> Vec<T>
where
    T: TTScalar + Clone,
{
    let mut data = Vec::with_capacity(tensor.left_dim() * tensor.site_dim());
    for s in 0..tensor.site_dim() {
        for l in 0..tensor.left_dim() {
            data.push(*tensor.get3(l, s, 0));
        }
    }
    data
}

fn middle_site_data<T>(tensor: &tensor4all_simplett::Tensor3<T>) -> Vec<T>
where
    T: TTScalar + Clone,
{
    let mut data = Vec::with_capacity(tensor.left_dim() * tensor.site_dim() * tensor.right_dim());
    for r in 0..tensor.right_dim() {
        for s in 0..tensor.site_dim() {
            for l in 0..tensor.left_dim() {
                data.push(*tensor.get3(l, s, r));
            }
        }
    }
    data
}
