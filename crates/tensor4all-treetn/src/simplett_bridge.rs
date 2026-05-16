use anyhow::{ensure, Result};
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorElement};
use tensor4all_simplett::{
    tensor3_from_data, AbstractTensorTrain, TTScalar, Tensor3Ops, TensorTrain,
};

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
///     tensor3_from_data(vec![1.0_f64, 2.0], 1, 2, 1).unwrap(),
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
///     tensor3_from_data(vec![1.0_f64, 2.0], 1, 2, 1).unwrap(),
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
///     tensor3_from_data(vec![1.0_f64, 2.0], 1, 2, 1).unwrap(),
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

/// Convert a linear-chain `TreeTN<TensorDynLen, usize>` back into a simple tensor train.
///
/// The TreeTN must use node names `0..n-1`, contain exactly one site index per
/// node, and have edges only between adjacent node names. This is the inverse of
/// [`tensor_train_to_treetn`] for linear-chain TreeTNs.
///
/// # Arguments
/// * `treetn` - A consumed linear-chain tree tensor network whose node names
///   identify tensor-train site positions. Each node must contain one site
///   index and zero, one, or two adjacent bond indices.
///
/// # Returns
/// A simple [`TensorTrain`] with one core per TreeTN node. Site and bond index
/// identities are used only to validate and order local axes; `TensorTrain`
/// cores do not retain index metadata.
///
/// # Errors
/// Returns an error if node names are not `0..n-1`, the topology is not a
/// nearest-neighbor chain, a node has the wrong site/bond indices, a site
/// tensor tracks autodiff state, scalar extraction fails, or a resulting core
/// has invalid dimensions.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, TensorTrain};
/// use tensor4all_treetn::{tensor_train_to_treetn, treetn_to_tensor_train};
///
/// let tt = TensorTrain::new(vec![
///     tensor3_from_data(vec![1.0_f64, 2.0, 3.0, 4.0], 1, 2, 2).unwrap(),
///     tensor3_from_data(vec![0.5, 1.5, -1.0, 2.0], 2, 2, 1).unwrap(),
/// ]).unwrap();
///
/// let (tree, _) = tensor_train_to_treetn(&tt).unwrap();
/// let roundtrip = treetn_to_tensor_train::<f64>(tree).unwrap();
///
/// assert_eq!(roundtrip.site_dims(), tt.site_dims());
/// assert_eq!(roundtrip.link_dims(), tt.link_dims());
/// assert_eq!(roundtrip.fulltensor(), tt.fulltensor());
/// ```
pub fn treetn_to_tensor_train<T>(mut treetn: TreeTN<TensorDynLen, usize>) -> Result<TensorTrain<T>>
where
    T: TTScalar + TensorElement + Clone,
{
    let nsites = treetn.node_count();
    if nsites == 0 {
        return Ok(TensorTrain::new(Vec::new())?);
    }

    let mut node_names = treetn.node_names();
    node_names.sort_unstable();
    let expected_names: Vec<_> = (0..nsites).collect();
    ensure!(
        node_names == expected_names,
        "treetn_to_tensor_train: expected node names 0..{}, got {:?}",
        nsites,
        node_names
    );
    ensure!(
        treetn.edge_count() == nsites - 1,
        "treetn_to_tensor_train: expected a chain with {} edges, got {}",
        nsites - 1,
        treetn.edge_count()
    );

    let mut site_metadata = Vec::with_capacity(nsites);
    for site in 0..nsites {
        let site_space = treetn.site_space(&site).ok_or_else(|| {
            anyhow::anyhow!("treetn_to_tensor_train: missing site space at node {site}")
        })?;
        ensure!(
            site_space.len() == 1,
            "treetn_to_tensor_train: node {site} must have exactly one site index, got {}",
            site_space.len()
        );
        let site_index = site_space.iter().next().ok_or_else(|| {
            anyhow::anyhow!("treetn_to_tensor_train: node {site} has no site index")
        })?;

        let left_bond = if site == 0 {
            None
        } else {
            let edge = treetn.edge_between(&(site - 1), &site).ok_or_else(|| {
                anyhow::anyhow!(
                    "treetn_to_tensor_train: missing chain edge between nodes {} and {}",
                    site - 1,
                    site
                )
            })?;
            Some(
                treetn
                    .bond_index(edge)
                    .ok_or_else(|| {
                        anyhow::anyhow!("treetn_to_tensor_train: missing left bond at node {site}")
                    })?
                    .clone(),
            )
        };

        let right_bond = if site + 1 == nsites {
            None
        } else {
            let edge = treetn.edge_between(&site, &(site + 1)).ok_or_else(|| {
                anyhow::anyhow!(
                    "treetn_to_tensor_train: missing chain edge between nodes {} and {}",
                    site,
                    site + 1
                )
            })?;
            Some(
                treetn
                    .bond_index(edge)
                    .ok_or_else(|| {
                        anyhow::anyhow!("treetn_to_tensor_train: missing right bond at node {site}")
                    })?
                    .clone(),
            )
        };

        site_metadata.push(ChainSiteMetadata {
            site_index: site_index.clone(),
            left_bond,
            right_bond,
        });
    }

    let mut tensors = Vec::with_capacity(nsites);
    for (site, metadata) in site_metadata.into_iter().enumerate() {
        let tensor = treetn.remove_tensor_by_name(&site).ok_or_else(|| {
            anyhow::anyhow!("treetn_to_tensor_train: missing tensor at node {site}")
        })?;

        tensors.push(treetn_site_to_tensor3::<T>(tensor, metadata, site)?);
    }

    Ok(TensorTrain::new(tensors)?)
}

struct ChainSiteMetadata {
    site_index: DynIndex,
    left_bond: Option<DynIndex>,
    right_bond: Option<DynIndex>,
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

fn treetn_site_to_tensor3<T>(
    tensor: TensorDynLen,
    metadata: ChainSiteMetadata,
    site: usize,
) -> Result<tensor4all_simplett::Tensor3<T>>
where
    T: TTScalar + TensorElement + Clone,
{
    let (tensor_indices, source) = tensor.into_dense_col_major_parts::<T>()?;
    let expected_rank =
        1 + usize::from(metadata.left_bond.is_some()) + usize::from(metadata.right_bond.is_some());
    ensure!(
        tensor_indices.len() == expected_rank,
        "treetn_to_tensor_train: node {site} has rank {}, expected {expected_rank}",
        tensor_indices.len()
    );

    let dims: Vec<_> = tensor_indices.iter().map(IndexLike::dim).collect();
    let site_axis = index_axis(&tensor_indices, &metadata.site_index).ok_or_else(|| {
        anyhow::anyhow!("treetn_to_tensor_train: node {site} tensor is missing its site index")
    })?;
    let left_axis = metadata
        .left_bond
        .as_ref()
        .map(|index| {
            index_axis(&tensor_indices, index).ok_or_else(|| {
                anyhow::anyhow!(
                    "treetn_to_tensor_train: node {site} tensor is missing its left bond"
                )
            })
        })
        .transpose()?;
    let right_axis = metadata
        .right_bond
        .as_ref()
        .map(|index| {
            index_axis(&tensor_indices, index).ok_or_else(|| {
                anyhow::anyhow!(
                    "treetn_to_tensor_train: node {site} tensor is missing its right bond"
                )
            })
        })
        .transpose()?;

    let left_dim = metadata.left_bond.as_ref().map(IndexLike::dim).unwrap_or(1);
    let site_dim = metadata.site_index.dim();
    let right_dim = metadata
        .right_bond
        .as_ref()
        .map(IndexLike::dim)
        .unwrap_or(1);

    let expected_axes: Vec<_> = left_axis
        .into_iter()
        .chain(std::iter::once(site_axis))
        .chain(right_axis)
        .collect();
    let data = if expected_axes == (0..tensor_indices.len()).collect::<Vec<_>>() {
        source
    } else {
        let mut data = Vec::with_capacity(left_dim * site_dim * right_dim);
        for r in 0..right_dim {
            for s in 0..site_dim {
                for l in 0..left_dim {
                    let mut multi = vec![0usize; tensor_indices.len()];
                    if let Some(axis) = left_axis {
                        multi[axis] = l;
                    }
                    multi[site_axis] = s;
                    if let Some(axis) = right_axis {
                        multi[axis] = r;
                    }
                    data.push(source[col_major_offset(&multi, &dims)]);
                }
            }
        }
        data
    };

    Ok(tensor3_from_data(data, left_dim, site_dim, right_dim)?)
}

fn index_axis(indices: &[DynIndex], target: &DynIndex) -> Option<usize> {
    indices.iter().position(|index| index == target)
}

fn col_major_offset(multi: &[usize], dims: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&coord, &dim) in multi.iter().zip(dims) {
        offset += coord * stride;
        stride *= dim;
    }
    offset
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
