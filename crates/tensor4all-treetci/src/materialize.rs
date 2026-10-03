use crate::error::{Result as TreeTciResult, TreeTciError};
use crate::{
    batch::{evaluate_points_chunked, EVALUATION_CHUNK_POINTS},
    ncols_2d, GlobalIndexBatch, SubtreeKey, TreeTCI2, TreeTciEdge,
};
use anyhow::{ensure, Result};

use std::borrow::Cow;
use std::collections::HashMap;
use tensor4all_core::MatrixLuciScalar as Scalar;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::TreeTN;

/// Materialize a converged TreeTCI state as a `TreeTN`.
///
/// Converts the pivot sets stored in a [`TreeTCI2`] into site tensors
/// of a [`TreeTN`]. The `evaluate` closure is called to fill tensor
/// entries at the selected pivot points, in calls of at most 65,536 points
/// (see [`GlobalIndexBatch`](crate::GlobalIndexBatch#batch-sizes)).
///
/// `center_site` selects the BFS root for the tree decomposition
/// (default: site 0).
///
/// This function is called internally by [`crossinterpolate2`](crate::crossinterpolate2).
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
pub fn to_treetn<T, F>(
    state: &TreeTCI2<T>,
    evaluate: F,
    center_site: Option<usize>,
) -> TreeTciResult<TreeTN<IdxTensor, usize>>
where
    T: FullPivLuScalar + tensor4all_core::MatrixLuciScalar + tensor4all_core::TensorElement,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    to_treetn_chunked(state, &evaluate, center_site, EVALUATION_CHUNK_POINTS)
}

/// [`to_treetn`] with an explicit number of points per evaluator call.
///
/// The result does not depend on `chunk_points`; tests use it to check that.
pub(crate) fn to_treetn_chunked<T, F>(
    state: &TreeTCI2<T>,
    evaluate: &F,
    center_site: Option<usize>,
    chunk_points: usize,
) -> TreeTciResult<TreeTN<IdxTensor, usize>>
where
    T: FullPivLuScalar + tensor4all_core::MatrixLuciScalar + tensor4all_core::TensorElement,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    let node_names = (0..state.graph.n_sites()).collect::<Vec<_>>();
    let site_indices = state
        .local_dims
        .iter()
        .map(|&dim| vec![DynIndex::new_dyn(dim)])
        .collect::<Vec<_>>();
    to_named_treetn_chunked(
        state,
        evaluate,
        center_site,
        &node_names,
        &site_indices,
        chunk_points,
    )
}

/// Materialize a swept TreeTCI state with caller-chosen node names and site
/// indices.
///
/// Vertex `v` becomes the node `node_names[v]` whose site legs are
/// `site_indices[v]`, listed before the bond legs. The product of their
/// dimensions must equal the vertex's local dimension, and the vertex
/// coordinate is split column-major over them (the first index varies
/// fastest). An empty list is allowed only for a dimension-one vertex, whose
/// unit leg is then omitted. Both splits are pure reshapes of the column-major
/// site tensor.
pub(crate) fn to_named_treetn<T, F, V>(
    state: &TreeTCI2<T>,
    evaluate: F,
    center_site: Option<usize>,
    node_names: &[V],
    site_indices: &[Vec<DynIndex>],
) -> TreeTciResult<TreeTN<IdxTensor, V>>
where
    T: FullPivLuScalar + tensor4all_core::MatrixLuciScalar + tensor4all_core::TensorElement,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    V: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    to_named_treetn_chunked(
        state,
        &evaluate,
        center_site,
        node_names,
        site_indices,
        EVALUATION_CHUNK_POINTS,
    )
}

fn to_named_treetn_chunked<T, F, V>(
    state: &TreeTCI2<T>,
    evaluate: &F,
    center_site: Option<usize>,
    node_names: &[V],
    site_indices: &[Vec<DynIndex>],
    chunk_points: usize,
) -> TreeTciResult<TreeTN<IdxTensor, V>>
where
    T: FullPivLuScalar + tensor4all_core::MatrixLuciScalar + tensor4all_core::TensorElement,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    V: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let n_vertices = state.graph.n_sites();
    if node_names.len() != n_vertices || site_indices.len() != n_vertices {
        return Err(anyhow::anyhow!(
            "materialization needs one node name and one site-index list per vertex: \
             {n_vertices} vertices, {} names, {} site-index lists",
            node_names.len(),
            site_indices.len()
        )
        .into());
    }
    for (vertex, indices) in site_indices.iter().enumerate() {
        let product = indices
            .iter()
            .try_fold(1usize, |product, index| product.checked_mul(index.dim()))
            .ok_or_else(|| {
                anyhow::anyhow!("site dimension product of vertex {vertex} overflowed usize")
            })?;
        if product != state.local_dims[vertex] {
            return Err(anyhow::anyhow!(
                "site indices of vertex {vertex} have dimension product {product}, \
                 but the vertex has local dimension {}",
                state.local_dims[vertex]
            )
            .into());
        }
    }

    let root = center_site.unwrap_or(0);
    let (parents, distances) = state.graph.bfs_tree(root)?;

    let mut bond_indices = HashMap::new();
    for edge in state.graph.edges() {
        let (left_key, right_key) = state.graph.subregion_vertices(edge)?;
        let left_rank = state
            .ijset
            .get(&left_key)
            .map(ncols_2d)
            .transpose()?
            .unwrap_or(0);
        let right_rank = state
            .ijset
            .get(&right_key)
            .map(ncols_2d)
            .transpose()?
            .unwrap_or(0);
        if !(left_rank == right_rank) {
            return Err(anyhow::anyhow!(
                "bond ranks disagree across edge {:?}: left {}, right {}",
                edge,
                left_rank,
                right_rank
            )
            .into());
        };
        bond_indices.insert(edge, DynIndex::new_dyn(left_rank.max(1)));
    }

    let mut sites = (0..state.graph.n_sites()).collect::<Vec<_>>();
    sites.sort_by_key(|&site| (distances[site], site));

    let mut tensors = Vec::with_capacity(sites.len());
    let mut names = Vec::with_capacity(sites.len());
    for site in sites {
        let parent_edge = parents[site]
            .map(|parent| state.graph.edge_between(site, parent))
            .transpose()?;
        let incoming_edges = match parent_edge {
            Some(edge) => state.graph.adjacent_edges(site, &[edge]),
            None => state.graph.adjacent_edges(site, &[]),
        };
        let in_keys = state.graph.edge_in_ij_keys(site, &incoming_edges)?;
        let out_edges = parent_edge.into_iter().collect::<Vec<_>>();
        let out_keys = state.graph.edge_in_ij_keys(site, &out_edges)?;

        let data = if out_edges.is_empty() {
            fill_tensor_values(state, &in_keys, &out_keys, &[site], evaluate, chunk_points)?
        } else {
            site_tensor_with_parent(
                state,
                site,
                out_edges[0],
                &in_keys,
                &out_keys,
                evaluate,
                chunk_points,
            )?
        };

        let index_count = incoming_edges
            .len()
            .checked_add(out_edges.len())
            .and_then(|count| count.checked_add(site_indices[site].len()))
            .ok_or_else(|| anyhow::anyhow!("materialized site index count overflowed usize"))?;
        let mut indices = Vec::with_capacity(index_count);
        indices.extend(site_indices[site].iter().cloned());
        for edge in &incoming_edges {
            indices.push(
                bond_indices
                    .get(edge)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing bond index for edge {:?}", edge))?,
            );
        }
        for edge in &out_edges {
            indices.push(
                bond_indices
                    .get(edge)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing bond index for edge {:?}", edge))?,
            );
        }

        tensors.push(IdxTensor::from_dense(indices, data)?);
        names.push(node_names[site].clone());
    }

    TreeTN::from_tensors(tensors, names).map_err(TreeTciError::from)
}

fn site_tensor_with_parent<T, F>(
    state: &TreeTCI2<T>,
    site: usize,
    parent_edge: TreeTciEdge,
    in_keys: &[SubtreeKey],
    out_keys: &[SubtreeKey],
    evaluate: &F,
    chunk_points: usize,
) -> Result<Vec<T>>
where
    T: FullPivLuScalar + tensor4all_core::MatrixLuciScalar + tensor4all_core::TensorElement,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    if !(out_keys.len() == 1) {
        return Err(anyhow::anyhow!(
            "MVP TreeTCI materialization expects exactly one outgoing key per non-root site"
        ));
    };

    let pi1_values = fill_tensor_values(state, in_keys, out_keys, &[site], evaluate, chunk_points)?;
    let rows = state.local_dims[site]
        .checked_mul(product_pivot_dims(state, in_keys)?)
        .ok_or_else(|| anyhow::anyhow!("materialized site row count overflowed usize"))?;
    let cols = product_pivot_dims(state, out_keys)?;

    let site_side_key = site_side_key(state, site, parent_edge)?;
    let p_values = fill_tensor_values(
        state,
        std::slice::from_ref(&site_side_key),
        out_keys,
        &[],
        evaluate,
        chunk_points,
    )?;
    let p_rows = state
        .ijset
        .get(&site_side_key)
        .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", site_side_key))
        .and_then(ncols_2d)?;
    if !(p_rows == cols) {
        return Err(anyhow::anyhow!(
            "pivot matrix for site {} is not square: {} x {}",
            site,
            p_rows,
            cols
        ));
    };

    // A numerically zero pivot matrix (the function underflows in this
    // subdomain) cannot be solved; emit a zero site tensor of the same shape
    // instead of failing the solve. Mirrors the guard in
    // `tensor4all-tensorci`'s `fill_site_tensors`.
    if p_values
        .iter()
        .all(|value| tensor4all_core::Scalar::abs_val(*value) < f64::EPSILON)
    {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow::anyhow!("materialized zero site size overflowed usize"))?;
        return Ok(vec![T::zero(); len]);
    }

    T::solve_right_full_piv_lu(&pi1_values, rows, cols, &p_values, p_rows, cols)
        .map_err(anyhow::Error::from)
}

fn site_side_key<T>(state: &TreeTCI2<T>, site: usize, edge: TreeTciEdge) -> Result<SubtreeKey> {
    let (left_key, right_key) = state.graph.subregion_vertices(edge)?;
    if left_key.as_slice().contains(&site) {
        Ok(left_key)
    } else if right_key.as_slice().contains(&site) {
        Ok(right_key)
    } else {
        Err(anyhow::anyhow!(
            "site {} does not appear in either side of edge {:?}",
            site,
            edge
        ))
    }
}

fn product_pivot_dims<T>(state: &TreeTCI2<T>, keys: &[SubtreeKey]) -> Result<usize> {
    let mut product = 1usize;
    for key in keys {
        let dim = state
            .ijset
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", key))
            .and_then(ncols_2d)?;
        product = product
            .checked_mul(dim.max(1))
            .ok_or_else(|| anyhow::anyhow!("pivot dimension product overflowed usize"))?;
    }
    Ok(product)
}

/// A site-partition failure of [`fill_tensor_values`], typed as
/// [`TreeTciError::IndexOutOfBounds`] like the per-point assembly it replaces.
/// It only occurs for an inconsistent internal state.
fn index_error(message: String) -> anyhow::Error {
    TreeTciError::IndexOutOfBounds { message }.into()
}

/// One factor of the point product in [`fill_tensor_values`]: a set of
/// `count` columns over `sites`, stored column-major.
struct PointFactor<'a> {
    sites: &'a [usize],
    columns: Cow<'a, [usize]>,
    count: usize,
}

/// Evaluate the function on every point of the product
/// `out_keys x in_keys x central_sites` and return the values in that
/// column-major order: central sites vary fastest (the last one first), then
/// the pivots of `in_keys[0]`, `in_keys[1]`, ..., then those of `out_keys`.
///
/// The points are written straight into a reused, chunked batch buffer
/// instead of one `Vec` per point and per pivot combination. The bipartition
/// of the sites is checked once up front instead of once per point.
fn fill_tensor_values<T, F>(
    state: &TreeTCI2<T>,
    in_keys: &[SubtreeKey],
    out_keys: &[SubtreeKey],
    central_sites: &[usize],
    evaluate: &F,
    chunk_points: usize,
) -> Result<Vec<T>>
where
    T: Scalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    let n_sites = state.local_dims.len();
    let mut factors = Vec::with_capacity(central_sites.len() + in_keys.len() + out_keys.len());
    for site in central_sites.iter().rev() {
        let dim = *state.local_dims.get(*site).ok_or_else(|| {
            index_error(format!("site {site} is out of bounds for {n_sites} sites"))
        })?;
        factors.push(PointFactor {
            sites: std::slice::from_ref(site),
            columns: Cow::Owned((0..dim).collect()),
            count: dim,
        });
    }
    for key in in_keys.iter().chain(out_keys) {
        let pivots = state
            .ijset
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", key))?;
        ensure!(
            pivots.shape().first().copied() == Some(key.as_slice().len()),
            "pivot set of shape {:?} does not match subtree key {:?}",
            pivots.shape(),
            key
        );
        factors.push(PointFactor {
            sites: key.as_slice(),
            columns: Cow::Borrowed(pivots.data()),
            count: ncols_2d(pivots)?,
        });
    }

    let mut assigned = vec![false; n_sites];
    for &site in factors.iter().flat_map(|factor| factor.sites) {
        // Defensive: central sites were bounds-checked above and an
        // out-of-range key has no pivot set, so this cannot fail today.
        let slot = assigned.get_mut(site).ok_or_else(|| {
            index_error(format!("site {site} is out of bounds for {n_sites} sites"))
        })?;
        if *slot {
            return Err(index_error(format!(
                "site {site} was assigned more than once"
            )));
        }
        *slot = true;
    }
    if !assigned.iter().all(|&seen| seen) {
        return Err(index_error(
            "global point assembly left some sites unassigned".to_string(),
        ));
    }

    let n_points = factors.iter().try_fold(1usize, |count, factor| {
        count
            .checked_mul(factor.count)
            .ok_or_else(|| anyhow::anyhow!("materialization point count overflowed usize"))
    })?;

    // Mixed-radix counter over the factors, first factor fastest.
    let mut digits = vec![0usize; factors.len()];
    evaluate_points_chunked(
        n_sites,
        n_points,
        chunk_points,
        |point| {
            for (factor, &digit) in factors.iter().zip(&digits) {
                let width = factor.sites.len();
                let column = &factor.columns[digit * width..(digit + 1) * width];
                for (&site, &value) in factor.sites.iter().zip(column) {
                    point[site] = value;
                }
            }
            for (digit, factor) in digits.iter_mut().zip(&factors) {
                *digit += 1;
                if *digit < factor.count {
                    break;
                }
                *digit = 0;
            }
        },
        evaluate,
        "fill-tensor points",
    )
}

#[cfg(test)]
mod tests;
