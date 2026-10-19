use crate::error::TreeTNOperationError;
use anyhow::Result;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike, TensorElement};
use tensor4all_simplett::{
    tensor3_from_data, tensor3_zeros, AbstractTensorTrain, SimpleTensorTrain, TTScalar, Tensor3Ops,
};
use tensor4all_tensorbackend::{
    dense_native_tensor_from_col_major, dense_native_tensor_from_col_major_owned,
    einsum_native_subscripts, native_tensor_primal_to_dense_col_major,
};

use crate::TreeTN;

/// Convert a linear-chain simple tensor train into a `TreeTN` with node names `0..n-1`.
///
/// The returned site indices are ordered by tensor-train site position, which is
/// convenient for downstream state/layout bookkeeping.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] when a fresh bond index cannot be allocated, or when
/// [`TreeTN::from_tensors`] rejects the generated node names.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{IndexLike, TensorIndex, TensorLike};
/// use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};
/// use tensor4all_treetn::tensor_train_to_treetn;
///
/// let tt = SimpleTensorTrain::new(vec![
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
    tt: &SimpleTensorTrain<T>,
) -> std::result::Result<(TreeTN<IdxTensor, usize>, Vec<DynIndex>), TreeTNOperationError>
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
/// # Errors
///
/// Returns [`TreeTNOperationError`] when `node_names` has a different length than the train,
/// when a fresh bond index cannot be allocated, or when
/// [`TreeTN::from_tensors`] rejects the names.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};
/// use tensor4all_treetn::tensor_train_to_treetn_with_names;
///
/// let tt = SimpleTensorTrain::new(vec![
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
    tt: &SimpleTensorTrain<T>,
    node_names: Vec<V>,
) -> std::result::Result<(TreeTN<IdxTensor, V>, Vec<DynIndex>), TreeTNOperationError>
where
    T: TTScalar + TensorElement + Clone,
    V: Clone + std::hash::Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    tensor_train_to_treetn_impl(tt, node_names, None).map_err(TreeTNOperationError::from)
}

/// Convert a linear-chain simple tensor train into a `TreeTN` with explicit node names
/// and caller-provided site indices.
///
/// This is useful when downstream code must preserve external site-index identities
/// across a conversion boundary while still allowing internal bond indices to be
/// created fresh.
///
/// # Errors
///
/// Returns an error when `node_names` or `site_indices` has a different length
/// than the train, when a provided site index dimension does not match the
/// corresponding train site, when a fresh bond index cannot be allocated, or
/// when [`TreeTN::from_tensors`] rejects the names.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_simplett::{tensor3_from_data, SimpleTensorTrain};
/// use tensor4all_treetn::tensor_train_to_treetn_with_names_and_site_indices;
///
/// let tt = SimpleTensorTrain::new(vec![
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
    tt: &SimpleTensorTrain<T>,
    node_names: Vec<V>,
    site_indices: Vec<DynIndex>,
) -> std::result::Result<TreeTN<IdxTensor, V>, TreeTNOperationError>
where
    T: TTScalar + TensorElement + Clone,
    V: Clone + std::hash::Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let (treetn, _) = tensor_train_to_treetn_impl(tt, node_names, Some(site_indices))?;
    Ok(treetn)
}

/// Convert a linear-chain `TreeTN<IdxTensor, usize>` back into a simple tensor train.
///
/// The TreeTN must use node names `0..n-1`, contain exactly one site index per
/// node, and have edges only between adjacent node names. This is the inverse of
/// [`tensor_train_to_treetn`] for linear-chain TreeTNs.
///
/// # Arguments
/// * `treetn` - A consumed linear-chain tree tensor network whose node names
///
///   identify tensor-train site positions. Each node must contain one site
///   index and zero, one, or two adjacent bond indices.
///
/// # Returns
/// A simple [`SimpleTensorTrain`] with one core per TreeTN node. Site and bond index
/// identities are used only to validate and order local axes; `SimpleTensorTrain`
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
/// use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};
/// use tensor4all_treetn::{tensor_train_to_treetn, treetn_to_tensor_train};
///
/// let tt = SimpleTensorTrain::new(vec![
///     tensor3_from_data(vec![1.0_f64, 2.0, 3.0, 4.0], 1, 2, 2).unwrap(),
///     tensor3_from_data(vec![0.5, 1.5, -1.0, 2.0], 2, 2, 1).unwrap(),
/// ]).unwrap();
///
/// let (tree, _) = tensor_train_to_treetn(&tt).unwrap();
/// let roundtrip = treetn_to_tensor_train::<f64>(tree).unwrap();
///
/// assert_eq!(roundtrip.site_dims(), tt.site_dims());
/// assert_eq!(roundtrip.link_dims(), tt.link_dims());
/// assert_eq!(roundtrip.full_tensor().unwrap(), tt.full_tensor().unwrap());
/// ```
pub fn treetn_to_tensor_train<T>(
    mut treetn: TreeTN<IdxTensor, usize>,
) -> std::result::Result<SimpleTensorTrain<T>, TreeTNOperationError>
where
    T: TTScalar + TensorElement + Clone,
{
    let nsites = treetn.node_count();
    if nsites == 0 {
        return SimpleTensorTrain::new(Vec::new())
            .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)));
    }

    let mut node_names = treetn.node_names();
    node_names.sort_unstable();
    let expected_names: Vec<_> = (0..nsites).collect();
    if !(node_names == expected_names) {
        return Err(anyhow::anyhow!(
            "treetn_to_tensor_train: expected node names 0..{}, got {:?}",
            nsites,
            node_names
        )
        .into());
    };
    if !(treetn.edge_count() == nsites - 1) {
        return Err(anyhow::anyhow!(
            "treetn_to_tensor_train: expected a chain with {} edges, got {}",
            nsites - 1,
            treetn.edge_count()
        )
        .into());
    };

    let mut site_metadata = Vec::with_capacity(nsites);
    for site in 0..nsites {
        let site_space = treetn.site_space(&site).ok_or_else(|| {
            anyhow::anyhow!("treetn_to_tensor_train: missing site space at node {site}")
        })?;
        if !(site_space.len() == 1) {
            return Err(anyhow::anyhow!(
                "treetn_to_tensor_train: node {site} must have exactly one site index, got {}",
                site_space.len()
            )
            .into());
        };
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

    SimpleTensorTrain::new(tensors).map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))
}

/// Insert a one-hot site into a linear-chain `TreeTN<IdxTensor, usize>`.
///
/// The input and output chains use node names `0..n-1` / `0..n`. `position`
/// chooses the insertion point:
/// - `0` inserts before the first site;
/// - `len` inserts after the last site;
/// - any `1..len` inserts on the chain edge between `position - 1` and
///
///   `position`.
///
/// The inserted site is fixed to `value`, so the new tensor evaluates to the
/// old tensor when the inserted coordinate equals `value` and to zero
/// otherwise. Existing site indices are preserved in order.
///
/// # Arguments
///
/// - `treetn`: a linear-chain `TreeTN` whose node names are exactly `0..n`.
///
///   Each node must carry exactly one site index.
/// - `position`: insertion point in `0..=n`; `0` prepends and `n` appends.
/// - `site_index`: site index to attach to the inserted node.
/// - `value`: fixed coordinate for the inserted one-hot site.
///
/// # Returns
///
/// A new linear-chain `TreeTN` with node names `0..=n`. Existing site indices
/// keep their relative order and the inserted `site_index` appears at
/// `position`.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] when the chain is not named `0..n` with exactly one site
/// index per node, when `position` is greater than the site count, when `value`
/// is outside the inserted site's dimension, or when a conversion to or from
/// [`SimpleTensorTrain`] fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};
/// use tensor4all_treetn::{
///     insert_onehot_site_in_treetn_chain, tensor_train_to_treetn, treetn_to_tensor_train,
/// };
///
/// # fn main() -> anyhow::Result<()> {
/// let tt = SimpleTensorTrain::new(vec![
///     tensor3_from_data(vec![1.0_f64, 2.0], 1, 2, 1)?,
/// ])?;
/// let (tree, _) = tensor_train_to_treetn(&tt)?;
/// let fixed = DynIndex::new_dyn(2);
///
/// let extended = insert_onehot_site_in_treetn_chain::<f64>(tree, 0, fixed, 0)?;
/// let roundtrip = treetn_to_tensor_train::<f64>(extended)?;
///
/// assert_eq!(roundtrip.site_dims(), vec![2, 2]);
/// # Ok(())
/// # }
/// ```
pub fn insert_onehot_site_in_treetn_chain<T>(
    treetn: TreeTN<IdxTensor, usize>,
    position: usize,
    site_index: DynIndex,
    value: usize,
) -> std::result::Result<TreeTN<IdxTensor, usize>, TreeTNOperationError>
where
    T: TTScalar + TensorElement + Clone + Default,
{
    let old_site_indices = chain_site_indices(&treetn, "insert_onehot_site_in_treetn_chain")?;
    if !(position <= old_site_indices.len()) {
        return Err(anyhow::anyhow!(
            "insert_onehot_site_in_treetn_chain: position {} is out of range 0..={}",
            position,
            old_site_indices.len()
        )
        .into());
    };
    if !(value < site_index.dim()) {
        return Err(anyhow::anyhow!(
            "insert_onehot_site_in_treetn_chain: fixed value {} exceeds site dimension {}",
            value,
            site_index.dim()
        )
        .into());
    };

    let tt = treetn_to_tensor_train::<T>(treetn)?;
    let mut tensors = Vec::with_capacity(tt.len() + 1);
    for site in 0..position {
        tensors.push(tt.site_tensor(site).clone());
    }

    let bond_dim = if tt.is_empty() || position == 0 {
        1
    } else {
        tt.site_tensor(position - 1).right_dim()
    };
    let mut inserted =
        tensor3_zeros::<T>(bond_dim, site_index.dim(), bond_dim).map_err(anyhow::Error::from)?;
    for bond in 0..bond_dim {
        inserted.set3(bond, value, bond, T::one());
    }
    tensors.push(inserted);

    for site in position..tt.len() {
        tensors.push(tt.site_tensor(site).clone());
    }

    let mut site_indices = old_site_indices;
    site_indices.insert(position, site_index);
    let tt = SimpleTensorTrain::new(tensors)
        .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?;
    tensor_train_to_treetn_with_names_and_site_indices(&tt, (0..tt.len()).collect(), site_indices)
}

/// Contract a site of a linear-chain `TreeTN<IdxTensor, usize>` with weights and remove it.
///
/// The removed site is summed as `sum_s weights[s] * tensor[..., s, ...]`.
/// Pass already scaled weights, such as `1/d` averaging weights, when a
/// normalized reduction is desired. The returned chain has one fewer site, node
/// names `0..n-2`, and all remaining site indices preserved in their original
/// order.
///
/// # Arguments
///
/// - `treetn`: a linear-chain `TreeTN` whose node names are exactly `0..n`.
///
///   Each node must carry exactly one site index.
/// - `position`: chain site to contract and remove, in `0..n`.
/// - `weights`: one weight per local coordinate of the removed site.
///
/// # Returns
///
/// A new linear-chain `TreeTN` representing the weighted local reduction with
/// the selected site removed from the external index list.
///
/// # Errors
///
/// Returns [`TreeTNOperationError`] when the chain is not named `0..n` with exactly one site
/// index per node, when `position` is out of range, when removing the site
/// would leave a zero-site chain, when `weights` does not match the site's
/// dimension, or when a conversion to or from [`SimpleTensorTrain`] fails.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};
/// use tensor4all_treetn::{
///     tensor_train_to_treetn, treetn_to_tensor_train, weighted_remove_site_from_treetn_chain,
/// };
///
/// # fn main() -> anyhow::Result<()> {
/// let tt = SimpleTensorTrain::new(vec![
///     tensor3_from_data(vec![1.0_f64, 3.0], 1, 2, 1)?,
///     tensor3_from_data(vec![2.0_f64, 4.0], 1, 2, 1)?,
/// ])?;
/// let (tree, _) = tensor_train_to_treetn(&tt)?;
///
/// let reduced = weighted_remove_site_from_treetn_chain::<f64>(tree, 0, &[0.25, 0.75])?;
/// let roundtrip = treetn_to_tensor_train::<f64>(reduced)?;
///
/// assert_eq!(roundtrip.site_dims(), vec![2]);
/// assert!((roundtrip.evaluate(&[0])? - 5.0).abs() < 1.0e-12);
/// assert!((roundtrip.evaluate(&[1])? - 10.0).abs() < 1.0e-12);
/// # Ok(())
/// # }
/// ```
pub fn weighted_remove_site_from_treetn_chain<T>(
    treetn: TreeTN<IdxTensor, usize>,
    position: usize,
    weights: &[T],
) -> std::result::Result<TreeTN<IdxTensor, usize>, TreeTNOperationError>
where
    T: TTScalar + TensorElement + Clone + Default,
{
    let site_indices = chain_site_indices(&treetn, "weighted_remove_site_from_treetn_chain")?;
    if !(position < site_indices.len()) {
        return Err(anyhow::anyhow!(
            "weighted_remove_site_from_treetn_chain: position {} is out of range 0..{}",
            position,
            site_indices.len()
        )
        .into());
    };
    if !(site_indices.len() > 1) {
        return Err(anyhow::anyhow!("weighted_remove_site_from_treetn_chain: cannot remove the only site because scalar zero-site TreeTN chains are not supported").into());
    };

    let tt = treetn_to_tensor_train::<T>(treetn)?;
    if !(weights.len() == tt.site_dim(position)) {
        return Err(anyhow::anyhow!("weighted_remove_site_from_treetn_chain: weights length {} must match site dimension {}",
        weights.len(),
        tt.site_dim(position)).into());
    };

    let reduced_site = weighted_site_matrix(tt.site_tensor(position), weights)?;
    remove_site_with_reduced_matrix(tt, site_indices, position, &reduced_site)
        .map_err(TreeTNOperationError::from)
}

fn chain_site_indices(treetn: &TreeTN<IdxTensor, usize>, context: &str) -> Result<Vec<DynIndex>> {
    let nsites = treetn.node_count();
    let mut node_names = treetn.node_names();
    node_names.sort_unstable();
    if !(node_names == (0..nsites).collect::<Vec<_>>()) {
        return Err(anyhow::anyhow!(
            "{context}: expected node names 0..{}, got {:?}",
            nsites,
            node_names
        ));
    };

    let mut site_indices = Vec::with_capacity(nsites);
    for site in 0..nsites {
        let site_space = treetn
            .site_space(&site)
            .ok_or_else(|| anyhow::anyhow!("{context}: missing site space at node {site}"))?;
        if !(site_space.len() == 1) {
            return Err(anyhow::anyhow!(
                "{context}: node {site} must have exactly one site index, got {}",
                site_space.len()
            ));
        };
        let site_index = site_space
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("{context}: node {site} has no site index"))?;
        site_indices.push(site_index.clone());
    }
    Ok(site_indices)
}

fn weighted_site_matrix<T>(
    tensor: &tensor4all_simplett::Tensor3<T>,
    weights: &[T],
) -> Result<Vec<T>>
where
    T: TTScalar + TensorElement,
{
    // `l s r, s -> l r`: contract the site axis against the weights and keep the
    // `(left, right)` matrix. Cold path -- one graph dispatch per removed site.
    let site = dense_native_tensor_from_col_major_owned(tensor.to_col_major_vec(), tensor.dims())?;
    let weights = dense_native_tensor_from_col_major(weights, &[weights.len()])?;
    let matrix = einsum_native_subscripts("lsr,s->lr", &[&site, &weights])?;
    Ok(native_tensor_primal_to_dense_col_major::<T>(&matrix)?)
}

fn remove_site_with_reduced_matrix<T>(
    tt: SimpleTensorTrain<T>,
    mut site_indices: Vec<DynIndex>,
    position: usize,
    reduced_site: &[T],
) -> Result<TreeTN<IdxTensor, usize>>
where
    T: TTScalar + TensorElement + Clone + Default,
{
    let mut tensors = Vec::with_capacity(tt.len().saturating_sub(1));
    for site in 0..position {
        if site + 1 == position && position + 1 == tt.len() {
            tensors.push(absorb_reduced_site_into_left(
                tt.site_tensor(site),
                reduced_site,
            )?);
        } else {
            tensors.push(tt.site_tensor(site).clone());
        }
    }

    if position + 1 < tt.len() {
        tensors.push(absorb_reduced_site_into_right(
            reduced_site,
            tt.site_tensor(position),
            tt.site_tensor(position + 1),
        )?);
        for site in position + 2..tt.len() {
            tensors.push(tt.site_tensor(site).clone());
        }
    }

    site_indices.remove(position);
    let tt = SimpleTensorTrain::new(tensors)?;
    tensor_train_to_treetn_with_names_and_site_indices(&tt, (0..tt.len()).collect(), site_indices)
        .map_err(anyhow::Error::from)
}

fn absorb_reduced_site_into_right<T>(
    reduced_site: &[T],
    removed: &tensor4all_simplett::Tensor3<T>,
    right: &tensor4all_simplett::Tensor3<T>,
) -> Result<tensor4all_simplett::Tensor3<T>>
where
    T: TTScalar + TensorElement,
{
    // `l k, k s r -> l s r`: contract the removed site's right bond (already
    // reduced into `reduced_site` as an `(l, k)` matrix) into the next site.
    let left_dim = removed.left_dim();
    let removed_right_dim = removed.right_dim();
    let site_dim = right.site_dim();
    let right_dim = right.right_dim();
    let reduced = dense_native_tensor_from_col_major(reduced_site, &[left_dim, removed_right_dim])?;
    let right_native =
        dense_native_tensor_from_col_major_owned(right.to_col_major_vec(), right.dims())?;
    let absorbed = einsum_native_subscripts("lk,ksr->lsr", &[&reduced, &right_native])?;
    let data = native_tensor_primal_to_dense_col_major::<T>(&absorbed)?;
    Ok(tensor3_from_data(data, left_dim, site_dim, right_dim)?)
}

fn absorb_reduced_site_into_left<T>(
    left: &tensor4all_simplett::Tensor3<T>,
    reduced_site: &[T],
) -> Result<tensor4all_simplett::Tensor3<T>>
where
    T: TTScalar + TensorElement,
{
    // `l s k, k -> l s`, then keep a trailing unit right bond.
    let left_dim = left.left_dim();
    let site_dim = left.site_dim();
    let left_native =
        dense_native_tensor_from_col_major_owned(left.to_col_major_vec(), left.dims())?;
    let weights_native = dense_native_tensor_from_col_major(reduced_site, &[reduced_site.len()])?;
    let absorbed = einsum_native_subscripts("lsk,k->ls", &[&left_native, &weights_native])?;
    let data = native_tensor_primal_to_dense_col_major::<T>(&absorbed)?;
    Ok(tensor3_from_data(data, left_dim, site_dim, 1)?)
}

struct ChainSiteMetadata {
    site_index: DynIndex,
    left_bond: Option<DynIndex>,
    right_bond: Option<DynIndex>,
}

fn tensor_train_to_treetn_impl<T, V>(
    tt: &SimpleTensorTrain<T>,
    node_names: Vec<V>,
    site_indices: Option<Vec<DynIndex>>,
) -> Result<(TreeTN<IdxTensor, V>, Vec<DynIndex>)>
where
    T: TTScalar + TensorElement + Clone,
    V: Clone + std::hash::Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    if !(tt.len() == node_names.len()) {
        return Err(anyhow::anyhow!(
            "tensor_train_to_treetn: node_names length {} must match tensor-train length {}",
            node_names.len(),
            tt.len()
        ));
    };

    if tt.is_empty() {
        let site_indices = site_indices.unwrap_or_default();
        if !(site_indices.is_empty()) {
            return Err(anyhow::anyhow!(
                "tensor_train_to_treetn: empty tensor train requires zero site indices"
            ));
        };
        return Ok((TreeTN::new(), Vec::new()));
    }

    let site_indices = match site_indices {
        Some(indices) => {
            if !(indices.len() == tt.len()) {
                return Err(anyhow::anyhow!("tensor_train_to_treetn: site_indices length {} must match tensor-train length {}",
                indices.len(),
                tt.len()));
            };
            for (site, index) in indices.iter().enumerate() {
                if !(index.dim() == tt.site_dim(site)) {
                    return Err(anyhow::anyhow!("tensor_train_to_treetn: site index {} has dim {} but tensor-train site {} has dim {}",
                    site,
                    index.dim(),
                    site,
                    tt.site_dim(site)));
                };
            }
            indices
        }
        None => tt.site_dims().into_iter().map(DynIndex::new_dyn).collect(),
    };
    let bond_indices: Vec<DynIndex> = tt
        .link_dims()
        .into_iter()
        .map(DynIndex::new_bond)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let nsites = tt.len();

    let mut tensors = Vec::with_capacity(nsites);
    for site in 0..nsites {
        let site_tensor = tt.site_tensor(site);
        let tensor = if nsites == 1 {
            let data = single_site_data(site_tensor);
            IdxTensor::from_dense(vec![site_indices[site].clone()], data)?
        } else if site == 0 {
            let data = left_boundary_data(site_tensor);
            IdxTensor::from_dense(
                vec![site_indices[site].clone(), bond_indices[site].clone()],
                data,
            )?
        } else if site + 1 == nsites {
            let data = right_boundary_data(site_tensor);
            IdxTensor::from_dense(
                vec![bond_indices[site - 1].clone(), site_indices[site].clone()],
                data,
            )?
        } else {
            let data = middle_site_data(site_tensor);
            IdxTensor::from_dense(
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
    tensor: IdxTensor,
    metadata: ChainSiteMetadata,
    site: usize,
) -> Result<tensor4all_simplett::Tensor3<T>>
where
    T: TTScalar + TensorElement + Clone,
{
    let (tensor_indices, source) = tensor.into_dense_col_major_parts::<T>()?;
    let expected_rank =
        1 + usize::from(metadata.left_bond.is_some()) + usize::from(metadata.right_bond.is_some());
    if !(tensor_indices.len() == expected_rank) {
        return Err(anyhow::anyhow!(
            "treetn_to_tensor_train: node {site} has rank {}, expected {expected_rank}",
            tensor_indices.len()
        ));
    };

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

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    /// `t[l, s, r] = 1 + l + 2 s + 4 r`, column-major (r slowest, l fastest).
    fn real_site_tensor() -> tensor4all_simplett::Tensor3<f64> {
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 2, 2)
            .expect("valid 2x2x2 tensor")
    }

    /// The same values as [`real_site_tensor`], embedded in `Complex64`.
    fn complex_site_tensor() -> tensor4all_simplett::Tensor3<Complex64> {
        tensor3_from_data(
            (1..=8)
                .map(|value| Complex64::new(f64::from(value), 0.0))
                .collect(),
            2,
            2,
            2,
        )
        .expect("valid complex 2x2x2 tensor")
    }

    #[test]
    fn weighted_site_matrix_matches_hand_computed_reduction() {
        // matrix[l, r] = sum_s t[l, s, r] * w[s] with w = [2, 3]:
        //   r = 0: l = 0 -> 1*2 + 3*3 = 11, l = 1 -> 2*2 + 4*3 = 16
        //   r = 1: l = 0 -> 5*2 + 7*3 = 31, l = 1 -> 6*2 + 8*3 = 36
        let matrix = weighted_site_matrix(&real_site_tensor(), &[2.0, 3.0]).unwrap();
        assert_eq!(matrix, vec![11.0, 16.0, 31.0, 36.0]);
    }

    #[test]
    fn weighted_site_matrix_matches_hand_computed_complex_reduction() {
        // Same reduction with w = [i, 1]:
        //   r = 0: l = 0 -> 1*i + 3 = 3 + i,  l = 1 -> 2*i + 4 = 4 + 2i
        //   r = 1: l = 0 -> 5*i + 7 = 7 + 5i, l = 1 -> 6*i + 8 = 8 + 6i
        let matrix = weighted_site_matrix(
            &complex_site_tensor(),
            &[Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap();
        assert_eq!(
            matrix,
            vec![
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, 2.0),
                Complex64::new(7.0, 5.0),
                Complex64::new(8.0, 6.0),
            ]
        );
    }

    #[test]
    fn absorb_reduced_site_into_right_matches_hand_computed_contraction() {
        // reduced = [[1, 3], [2, 4]] column-major, so
        // result[l, s, r] = sum_k reduced[l, k] * t[k, s, r]:
        //   s = 0, r = 0: l = 0 -> 1*1 + 3*2 = 7,  l = 1 -> 2*1 + 4*2 = 10
        //   s = 1, r = 0: l = 0 -> 1*3 + 3*4 = 15, l = 1 -> 2*3 + 4*4 = 22
        //   s = 0, r = 1: l = 0 -> 1*5 + 3*6 = 23, l = 1 -> 2*5 + 4*6 = 34
        //   s = 1, r = 1: l = 0 -> 1*7 + 3*8 = 31, l = 1 -> 2*7 + 4*8 = 46
        let removed = tensor3_from_data(vec![0.0; 4], 2, 1, 2).expect("valid removed site");
        let absorbed =
            absorb_reduced_site_into_right(&[1.0, 2.0, 3.0, 4.0], &removed, &real_site_tensor())
                .unwrap();

        assert_eq!((absorbed.left_dim(), absorbed.site_dim()), (2, 2));
        assert_eq!(
            absorbed.to_col_major_vec(),
            vec![7.0, 10.0, 15.0, 22.0, 23.0, 34.0, 31.0, 46.0]
        );
    }

    #[test]
    fn absorb_reduced_site_into_left_matches_hand_computed_contraction() {
        // result[l, s] = sum_k t[l, s, k] * w[k] with w = [2, 3], so
        //   (l = 0, s = 0) -> 1*2 + 5*3 = 17, (l = 1, s = 0) -> 2*2 + 6*3 = 22
        //   (l = 0, s = 1) -> 3*2 + 7*3 = 27, (l = 1, s = 1) -> 4*2 + 8*3 = 32
        let absorbed = absorb_reduced_site_into_left(&real_site_tensor(), &[2.0, 3.0]).unwrap();
        assert_eq!(absorbed.left_dim(), 2);
        assert_eq!(absorbed.site_dim(), 2);
        assert_eq!(absorbed.right_dim(), 1);
        assert_eq!(absorbed.to_col_major_vec(), vec![17.0, 22.0, 27.0, 32.0]);
    }
}
