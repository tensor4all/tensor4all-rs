//! Addition operations for TreeTN using direct-sum (block) construction.
//!
//! This module provides helper functions and types for adding two TreeTNs:
//! - [`MergedBondInfo`]: Information about merged bond indices
//! - [`compute_merged_bond_indices`]: Compute merged bond index information from two networks
//! - [`direct_sum_tensors`]: Compute the direct sum of two tensors with merged bond indices

use num_complex::Complex64;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use anyhow::Result;

use tensor4all_core::index::{DynId, Index, NoSymmSpace, Symmetry, TagSet};
use tensor4all_core::storage::{DenseStorageC64, DenseStorageF64, Storage};
use tensor4all_core::TensorDynLen;

use super::{compute_strides, linear_to_multi_index, multi_to_linear_index, TreeTN};

/// Information about a merged bond index for direct-sum addition.
///
/// When adding two TreeTNs, each bond index in the result has dimension
/// `dim_a + dim_b`, where `dim_a` and `dim_b` are the original bond dimensions.
#[derive(Debug, Clone)]
pub struct MergedBondInfo<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    /// Bond dimension from the first TreeTN
    pub dim_a: usize,
    /// Bond dimension from the second TreeTN
    pub dim_b: usize,
    /// The new merged bond index (with dimension dim_a + dim_b)
    pub merged_index: Index<Id, Symm>,
}

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Compute merged bond indices for direct-sum addition.
    ///
    /// For each edge in the network, compute the merged bond information
    /// containing dimensions from both networks and a new merged index.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to compute merged bonds with
    ///
    /// # Returns
    /// A HashMap mapping edge keys (node_name_pair in canonical order) to MergedBondInfo.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Networks have incompatible topologies
    /// - Bond indices cannot be found
    pub fn compute_merged_bond_indices(
        &self,
        other: &Self,
    ) -> Result<HashMap<(V, V), MergedBondInfo<Id, Symm>>>
    where
        Id: From<DynId>,
        Symm: From<NoSymmSpace>,
        V: Ord,
    {
        let mut result = HashMap::new();

        for edge in self.graph.graph().edge_indices() {
            let (src, tgt) = self
                .graph
                .graph()
                .edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge has no endpoints"))?;

            let bond_index_a = self
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found in self"))?;
            let dim_a = bond_index_a.size();

            let src_name = self
                .graph
                .node_name(src)
                .ok_or_else(|| anyhow::anyhow!("Source node name not found"))?
                .clone();
            let tgt_name = self
                .graph
                .node_name(tgt)
                .ok_or_else(|| anyhow::anyhow!("Target node name not found"))?
                .clone();

            // Find corresponding edge in other
            let src_idx_other = other
                .graph
                .node_index(&src_name)
                .ok_or_else(|| anyhow::anyhow!("Source node not found in other"))?;
            let tgt_idx_other = other
                .graph
                .node_index(&tgt_name)
                .ok_or_else(|| anyhow::anyhow!("Target node not found in other"))?;

            // Find edge between these nodes in other
            let edge_other = other
                .graph
                .graph()
                .edges_connecting(src_idx_other, tgt_idx_other)
                .next()
                .or_else(|| {
                    other
                        .graph
                        .graph()
                        .edges_connecting(tgt_idx_other, src_idx_other)
                        .next()
                })
                .ok_or_else(|| anyhow::anyhow!("Edge not found in other"))?;

            let bond_index_b = other
                .bond_index(edge_other.id())
                .ok_or_else(|| anyhow::anyhow!("Bond index not found in other"))?;
            let dim_b = bond_index_b.size();

            // Create ONE shared bond index for both endpoints
            let new_dim = dim_a + dim_b;
            let dyn_bond_index = Index::new_link(new_dim)
                .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;
            let merged_index: Index<Id, Symm, TagSet> = Index {
                id: dyn_bond_index.id.into(),
                symm: dyn_bond_index.symm.into(),
                tags: dyn_bond_index.tags,
            };

            // Store in canonical order (smaller name first)
            let key = if src_name < tgt_name {
                (src_name, tgt_name)
            } else {
                (tgt_name, src_name)
            };

            result.insert(
                key,
                MergedBondInfo {
                    dim_a,
                    dim_b,
                    merged_index,
                },
            );
        }

        Ok(result)
    }
}

/// Compute the direct sum of two tensors with merged bond indices.
///
/// This function creates a new tensor that embeds tensor_a in the "A block"
/// and tensor_b in the "B block" of each bond dimension.
///
/// # Arguments
/// * `tensor_a` - First tensor
/// * `tensor_b` - Second tensor
/// * `site_indices` - Physical (site) indices for this node
/// * `bond_info_by_neighbor` - Map from neighbor node name to MergedBondInfo
/// * `neighbor_names_a` - Map from bond index ID in tensor_a to neighbor name
/// * `neighbor_names_b` - Map from bond index ID in tensor_b to neighbor name
///
/// # Returns
/// A new tensor representing the direct sum, with merged bond indices.
pub fn direct_sum_tensors<Id, Symm, V>(
    tensor_a: &TensorDynLen<Id, Symm>,
    tensor_b: &TensorDynLen<Id, Symm>,
    site_indices: &HashSet<Id>,
    bond_info_by_neighbor: &HashMap<V, &MergedBondInfo<Id, Symm>>,
    neighbor_names_a: &HashMap<Id, V>,
    neighbor_names_b: &HashMap<Id, V>,
) -> Result<TensorDynLen<Id, Symm>>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Ord + std::fmt::Debug,
{
    // Separate physical and bond indices for tensor_a, preserving order
    let mut physical_inds_a: Vec<(usize, Index<Id, Symm>)> = Vec::new();
    let mut bond_inds_a: Vec<(usize, Index<Id, Symm>, V)> = Vec::new(); // (pos, index, neighbor_name)

    for (pos, idx) in tensor_a.indices.iter().enumerate() {
        if site_indices.contains(&idx.id) {
            physical_inds_a.push((pos, idx.clone()));
        } else if let Some(neighbor) = neighbor_names_a.get(&idx.id) {
            bond_inds_a.push((pos, idx.clone(), neighbor.clone()));
        }
    }

    // Do the same for tensor_b
    let mut bond_inds_b: Vec<(usize, Index<Id, Symm>, V)> = Vec::new();
    for (pos, idx) in tensor_b.indices.iter().enumerate() {
        if !site_indices.contains(&idx.id) {
            if let Some(neighbor) = neighbor_names_b.get(&idx.id) {
                bond_inds_b.push((pos, idx.clone(), neighbor.clone()));
            }
        }
    }

    // Build canonical index order: physical indices first (in original order), then bonds (sorted by neighbor)
    let mut canonical_indices: Vec<Index<Id, Symm>> =
        physical_inds_a.iter().map(|(_, idx)| idx.clone()).collect();
    let mut canonical_dims: Vec<usize> =
        physical_inds_a.iter().map(|(_, idx)| idx.size()).collect();

    // Sort bonds by neighbor name for deterministic ordering
    let mut sorted_bond_neighbors: Vec<V> = bond_info_by_neighbor.keys().cloned().collect();
    sorted_bond_neighbors.sort();

    // Track bond axis info for embedding: (canonical_axis, dim_a, dim_b)
    let mut bond_axis_info: Vec<(usize, usize, usize)> = Vec::new();

    for neighbor_name in &sorted_bond_neighbors {
        let info = bond_info_by_neighbor.get(neighbor_name).ok_or_else(|| {
            anyhow::anyhow!("Bond info not found for neighbor {:?}", neighbor_name)
        })?;

        bond_axis_info.push((canonical_indices.len(), info.dim_a, info.dim_b));
        canonical_indices.push(info.merged_index.clone());
        canonical_dims.push(info.dim_a + info.dim_b);
    }

    // Build permutation for tensor_a
    let mut perm_a: Vec<usize> = Vec::new();
    // Physical indices in order
    for (pos, _) in &physical_inds_a {
        perm_a.push(*pos);
    }
    // Bond indices in sorted neighbor order
    for neighbor_name in &sorted_bond_neighbors {
        for (pos, _, bond_neighbor) in &bond_inds_a {
            if bond_neighbor == neighbor_name {
                perm_a.push(*pos);
                break;
            }
        }
    }

    let permuted_dims_a: Vec<usize> = perm_a.iter().map(|&i| tensor_a.dims[i]).collect();

    // Build permutation for tensor_b
    let mut perm_b: Vec<usize> = Vec::new();
    // Physical indices - match by position in physical_inds_a
    for (_, idx_a) in &physical_inds_a {
        // Find corresponding physical index in tensor_b by ID
        let pos_b = tensor_b
            .indices
            .iter()
            .position(|idx_b| site_indices.contains(&idx_b.id) && idx_b.id == idx_a.id)
            .ok_or_else(|| {
                anyhow::anyhow!("Physical index {:?} not found in tensor_b", idx_a.id)
            })?;
        perm_b.push(pos_b);
    }
    // Bond indices in sorted neighbor order
    for neighbor_name in &sorted_bond_neighbors {
        for (pos, _, bond_neighbor) in &bond_inds_b {
            if bond_neighbor == neighbor_name {
                perm_b.push(*pos);
                break;
            }
        }
    }

    let permuted_dims_b: Vec<usize> = perm_b.iter().map(|&i| tensor_b.dims[i]).collect();

    // Create the new tensor storage
    let total_size: usize = canonical_dims.iter().product();
    let is_complex = matches!(tensor_a.storage.as_ref(), Storage::DenseC64(_))
        || matches!(tensor_b.storage.as_ref(), Storage::DenseC64(_));

    // If there are no bonds for this node, add tensors element-wise
    let has_bonds = !bond_axis_info.is_empty();

    let new_storage = if is_complex {
        let data_a = match tensor_a.storage.as_ref() {
            Storage::DenseF64(d) => d
                .as_slice()
                .iter()
                .map(|&x| Complex64::new(x, 0.0))
                .collect::<Vec<_>>(),
            Storage::DenseC64(d) => d.as_slice().to_vec(),
            _ => {
                return Err(anyhow::anyhow!(
                    "Only dense storage is supported for TTN addition"
                ))
            }
        };
        let data_b = match tensor_b.storage.as_ref() {
            Storage::DenseF64(d) => d
                .as_slice()
                .iter()
                .map(|&x| Complex64::new(x, 0.0))
                .collect::<Vec<_>>(),
            Storage::DenseC64(d) => d.as_slice().to_vec(),
            _ => {
                return Err(anyhow::anyhow!(
                    "Only dense storage is supported for TTN addition"
                ))
            }
        };

        let permuted_data_a = permute_data_c64(&data_a, &tensor_a.dims, &perm_a);
        let permuted_data_b = permute_data_c64(&data_b, &tensor_b.dims, &perm_b);

        if has_bonds {
            let mut result_data = vec![Complex64::new(0.0, 0.0); total_size];
            embed_block_c64(
                &mut result_data,
                &canonical_dims,
                &permuted_data_a,
                &permuted_dims_a,
                &bond_axis_info,
                true,
            )?;
            embed_block_c64(
                &mut result_data,
                &canonical_dims,
                &permuted_data_b,
                &permuted_dims_b,
                &bond_axis_info,
                false,
            )?;
            Storage::DenseC64(DenseStorageC64::from_vec(result_data))
        } else {
            let result_data: Vec<Complex64> = permuted_data_a
                .iter()
                .zip(permuted_data_b.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            Storage::DenseC64(DenseStorageC64::from_vec(result_data))
        }
    } else {
        let data_a = match tensor_a.storage.as_ref() {
            Storage::DenseF64(d) => d.as_slice().to_vec(),
            _ => {
                return Err(anyhow::anyhow!(
                    "Only dense storage is supported for TTN addition"
                ))
            }
        };
        let data_b = match tensor_b.storage.as_ref() {
            Storage::DenseF64(d) => d.as_slice().to_vec(),
            _ => {
                return Err(anyhow::anyhow!(
                    "Only dense storage is supported for TTN addition"
                ))
            }
        };

        let permuted_data_a = permute_data_f64(&data_a, &tensor_a.dims, &perm_a);
        let permuted_data_b = permute_data_f64(&data_b, &tensor_b.dims, &perm_b);

        if has_bonds {
            let mut result_data = vec![0.0_f64; total_size];
            embed_block_f64(
                &mut result_data,
                &canonical_dims,
                &permuted_data_a,
                &permuted_dims_a,
                &bond_axis_info,
                true,
            )?;
            embed_block_f64(
                &mut result_data,
                &canonical_dims,
                &permuted_data_b,
                &permuted_dims_b,
                &bond_axis_info,
                false,
            )?;
            Storage::DenseF64(DenseStorageF64::from_vec(result_data))
        } else {
            let result_data: Vec<f64> = permuted_data_a
                .iter()
                .zip(permuted_data_b.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            Storage::DenseF64(DenseStorageF64::from_vec(result_data))
        }
    };

    Ok(TensorDynLen::new(
        canonical_indices,
        canonical_dims,
        Arc::new(new_storage),
    ))
}

// ============================================================================
// Helper functions for block embedding
// ============================================================================

/// Embed a source tensor block into a larger destination tensor for f64 data.
fn embed_block_f64(
    dest: &mut [f64],
    dest_dims: &[usize],
    src: &[f64],
    src_dims: &[usize],
    bond_positions: &[(usize, usize, usize)],
    is_a_block: bool,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }

    let src_total: usize = src_dims.iter().product();
    if src.len() != src_total {
        return Err(anyhow::anyhow!(
            "Source data length {} doesn't match dims product {}",
            src.len(),
            src_total
        ));
    }

    let src_strides = compute_strides(src_dims);
    let dest_strides = compute_strides(dest_dims);

    let bond_map: HashMap<usize, (usize, usize)> = bond_positions
        .iter()
        .map(|&(pos, dim_a, dim_b)| (pos, (dim_a, dim_b)))
        .collect();

    for src_linear in 0..src_total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, src_dims.len());

        let mut dest_multi = Vec::with_capacity(dest_dims.len());
        let mut src_idx = 0;

        for (dest_axis, _dest_dim) in dest_dims.iter().enumerate() {
            if let Some(&(dim_a, _dim_b)) = bond_map.get(&dest_axis) {
                let src_bond_idx = src_multi[src_idx];
                let dest_bond_idx = if is_a_block {
                    src_bond_idx
                } else {
                    src_bond_idx + dim_a
                };
                dest_multi.push(dest_bond_idx);
                src_idx += 1;
            } else {
                dest_multi.push(src_multi[src_idx]);
                src_idx += 1;
            }
        }

        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);
        dest[dest_linear] = src[src_linear];
    }

    Ok(())
}

/// Embed a source tensor block into a larger destination tensor for Complex64 data.
fn embed_block_c64(
    dest: &mut [Complex64],
    dest_dims: &[usize],
    src: &[Complex64],
    src_dims: &[usize],
    bond_positions: &[(usize, usize, usize)],
    is_a_block: bool,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }

    let src_total: usize = src_dims.iter().product();
    if src.len() != src_total {
        return Err(anyhow::anyhow!(
            "Source data length {} doesn't match dims product {}",
            src.len(),
            src_total
        ));
    }

    let src_strides = compute_strides(src_dims);
    let dest_strides = compute_strides(dest_dims);

    let bond_map: HashMap<usize, (usize, usize)> = bond_positions
        .iter()
        .map(|&(pos, dim_a, dim_b)| (pos, (dim_a, dim_b)))
        .collect();

    for src_linear in 0..src_total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, src_dims.len());

        let mut dest_multi = Vec::with_capacity(dest_dims.len());
        let mut src_idx = 0;

        for (dest_axis, _dest_dim) in dest_dims.iter().enumerate() {
            if let Some(&(dim_a, _dim_b)) = bond_map.get(&dest_axis) {
                let src_bond_idx = src_multi[src_idx];
                let dest_bond_idx = if is_a_block {
                    src_bond_idx
                } else {
                    src_bond_idx + dim_a
                };
                dest_multi.push(dest_bond_idx);
                src_idx += 1;
            } else {
                dest_multi.push(src_multi[src_idx]);
                src_idx += 1;
            }
        }

        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);
        dest[dest_linear] = src[src_linear];
    }

    Ok(())
}

/// Permute tensor data according to axis permutation (f64 version).
fn permute_data_f64(data: &[f64], dims: &[usize], perm: &[usize]) -> Vec<f64> {
    if perm.is_empty() || data.is_empty() {
        return data.to_vec();
    }

    let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return data.to_vec();
    }

    let rank = dims.len();
    let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
    let total: usize = dims.iter().product();

    let src_strides = compute_strides(dims);
    let dest_strides = compute_strides(&new_dims);

    let mut result = vec![0.0_f64; total];

    for src_linear in 0..total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, rank);
        let dest_multi: Vec<usize> = (0..rank).map(|i| src_multi[perm[i]]).collect();
        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);
        result[dest_linear] = data[src_linear];
    }

    result
}

/// Permute tensor data according to axis permutation (Complex64 version).
fn permute_data_c64(data: &[Complex64], dims: &[usize], perm: &[usize]) -> Vec<Complex64> {
    if perm.is_empty() || data.is_empty() {
        return data.to_vec();
    }

    let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return data.to_vec();
    }

    let rank = dims.len();
    let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
    let total: usize = dims.iter().product();

    let src_strides = compute_strides(dims);
    let dest_strides = compute_strides(&new_dims);

    let mut result = vec![Complex64::new(0.0, 0.0); total];

    for src_linear in 0..total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, rank);
        let dest_multi: Vec<usize> = (0..rank).map(|i| src_multi[perm[i]]).collect();
        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);
        result[dest_linear] = data[src_linear];
    }

    result
}
