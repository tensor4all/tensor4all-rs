//! Direct sum operations for tensors.
//!
//! This module provides functionality to compute the direct sum of tensors
//! along specified index pairs. The direct sum concatenates tensor data along
//! the paired indices, creating new indices with combined dimensions.
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.

use crate::defaults::{DynIndex, Index};
use crate::index_like::IndexLike;
use crate::storage::{DenseStorageC64, DenseStorageF64, Storage};
use crate::tensor::TensorDynLen;
use anyhow::Result;
use num_complex::Complex64;
use std::sync::Arc;

/// Compute the direct sum of two tensors along specified index pairs.
///
/// For tensors A and B with indices to be summed specified as pairs,
/// creates a new tensor C where each paired index has dimension = dim_A + dim_B.
/// Non-paired indices must match exactly between A and B (same ID).
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `pairs` - Pairs of (a_index, b_index) to be summed. Each pair creates
///   a new index in the result with dimension = dim(a_index) + dim(b_index).
///
/// # Returns
///
/// A tuple of:
/// - The direct sum tensor
/// - The new indices created for the summed dimensions (one per pair)
///
/// # Example
///
/// ```ignore
/// // A has indices [i, j] with dims [2, 3]
/// // B has indices [i, k] with dims [2, 4]
/// // If we pair (j, k), result has indices [i, m] with dims [2, 7]
/// // where m is a new index with dim = 3 + 4 = 7
/// let (result, new_indices) = direct_sum(&a, &b, &[(j, k)])?;
/// ```
pub fn direct_sum(
    a: &TensorDynLen,
    b: &TensorDynLen,
    pairs: &[(DynIndex, DynIndex)],
) -> Result<(TensorDynLen, Vec<DynIndex>)> {
    // Dispatch based on storage type
    match (a.storage.as_ref(), b.storage.as_ref()) {
        (Storage::DenseF64(_), Storage::DenseF64(_)) => direct_sum_f64(a, b, pairs),
        (Storage::DenseC64(_), Storage::DenseC64(_)) => direct_sum_c64(a, b, pairs),
        _ => Err(anyhow::anyhow!(
            "direct_sum requires both tensors to have the same dense storage type (DenseF64 or DenseC64)"
        )),
    }
}

/// Setup data for direct sum computation
struct DirectSumSetup {
    common_a_positions: Vec<usize>,
    common_b_positions: Vec<usize>,
    paired_a_positions: Vec<usize>,
    paired_b_positions: Vec<usize>,
    paired_dims_a: Vec<usize>,
    paired_dims_b: Vec<usize>,
    result_indices: Vec<DynIndex>,
    result_dims: Vec<usize>,
    result_strides: Vec<usize>,
    result_total: usize,
    a_strides: Vec<usize>,
    b_strides: Vec<usize>,
    new_indices: Vec<DynIndex>,
    n_common: usize,
}

fn setup_direct_sum(
    a: &TensorDynLen,
    b: &TensorDynLen,
    pairs: &[(DynIndex, DynIndex)],
) -> Result<DirectSumSetup> {
    use crate::defaults::DynId;
    use std::collections::HashMap;

    if pairs.is_empty() {
        return Err(anyhow::anyhow!(
            "direct_sum requires at least one index pair"
        ));
    }

    // Build maps from index IDs to positions
    let a_idx_map: HashMap<&DynId, usize> = a
        .indices
        .iter()
        .enumerate()
        .map(|(i, idx)| (idx.id(), i))
        .collect();
    let b_idx_map: HashMap<&DynId, usize> = b
        .indices
        .iter()
        .enumerate()
        .map(|(i, idx)| (idx.id(), i))
        .collect();

    // Build set of paired index IDs
    let mut a_paired_ids: std::collections::HashSet<&DynId> = std::collections::HashSet::new();
    let mut b_paired_ids: std::collections::HashSet<&DynId> = std::collections::HashSet::new();

    for (a_idx, b_idx) in pairs {
        if !a_idx_map.contains_key(a_idx.id()) {
            return Err(anyhow::anyhow!(
                "Index not found in first tensor (direct_sum)"
            ));
        }
        if !b_idx_map.contains_key(b_idx.id()) {
            return Err(anyhow::anyhow!(
                "Index not found in second tensor (direct_sum)"
            ));
        }
        a_paired_ids.insert(a_idx.id());
        b_paired_ids.insert(b_idx.id());
    }

    // Identify common (non-paired) indices
    let mut common_a_positions: Vec<usize> = Vec::new();
    let mut common_b_positions: Vec<usize> = Vec::new();
    for (a_id, &a_pos) in &a_idx_map {
        if a_paired_ids.contains(a_id) {
            continue;
        }
        if let Some(&b_pos) = b_idx_map.get(a_id) {
            if b_paired_ids.contains(a_id) {
                continue;
            }
            if a.dims[a_pos] != b.dims[b_pos] {
                return Err(anyhow::anyhow!(
                    "Dimension mismatch for common index: {} vs {}",
                    a.dims[a_pos],
                    b.dims[b_pos]
                ));
            }
            common_a_positions.push(a_pos);
            common_b_positions.push(b_pos);
        }
    }

    // Build paired positions and dimensions
    let paired_a_positions: Vec<usize> = pairs
        .iter()
        .map(|(a_idx, _)| a_idx_map[a_idx.id()])
        .collect();
    let paired_b_positions: Vec<usize> = pairs
        .iter()
        .map(|(_, b_idx)| b_idx_map[b_idx.id()])
        .collect();
    let paired_dims_a: Vec<usize> = paired_a_positions.iter().map(|&p| a.dims[p]).collect();
    let paired_dims_b: Vec<usize> = paired_b_positions.iter().map(|&p| b.dims[p]).collect();

    // Create new indices for paired dimensions
    let mut new_indices: Vec<DynIndex> = Vec::new();
    for (&dim_a, &dim_b) in paired_dims_a.iter().zip(&paired_dims_b) {
        let new_dim = dim_a + dim_b;
        let new_index =
            DynIndex::new_link(new_dim).map_err(|e| anyhow::anyhow!("Failed to create index: {:?}", e))?;
        new_indices.push(new_index);
    }

    // Build result indices and dimensions
    let mut result_indices: Vec<DynIndex> = Vec::new();
    let mut result_dims: Vec<usize> = Vec::new();

    // Common indices first
    for &a_pos in &common_a_positions {
        result_indices.push(a.indices[a_pos].clone());
        result_dims.push(a.dims[a_pos]);
    }

    // New indices from pairs
    for new_idx in &new_indices {
        result_indices.push(Index::new(new_idx.id().clone(), new_idx.symm.clone()));
        result_dims.push(new_idx.dim());
    }

    // Compute strides
    let result_total: usize = result_dims.iter().product();
    let mut result_strides: Vec<usize> = vec![1; result_dims.len()];
    for i in (0..result_dims.len().saturating_sub(1)).rev() {
        result_strides[i] = result_strides[i + 1] * result_dims[i + 1];
    }

    let mut a_strides: Vec<usize> = vec![1; a.dims.len()];
    for i in (0..a.dims.len().saturating_sub(1)).rev() {
        a_strides[i] = a_strides[i + 1] * a.dims[i + 1];
    }

    let mut b_strides: Vec<usize> = vec![1; b.dims.len()];
    for i in (0..b.dims.len().saturating_sub(1)).rev() {
        b_strides[i] = b_strides[i + 1] * b.dims[i + 1];
    }

    let n_common = common_a_positions.len();

    Ok(DirectSumSetup {
        common_a_positions,
        common_b_positions,
        paired_a_positions,
        paired_b_positions,
        paired_dims_a,
        paired_dims_b,
        result_indices,
        result_dims,
        result_strides,
        result_total,
        a_strides,
        b_strides,
        new_indices,
        n_common,
    })
}

fn linear_to_multi(linear: usize, dims: &[usize]) -> Vec<usize> {
    let mut multi = vec![0; dims.len()];
    let mut remaining = linear;
    for i in (0..dims.len()).rev() {
        multi[i] = remaining % dims[i];
        remaining /= dims[i];
    }
    multi
}

fn multi_to_linear(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides).map(|(&m, &s)| m * s).sum()
}

fn direct_sum_f64(
    a: &TensorDynLen,
    b: &TensorDynLen,
    pairs: &[(DynIndex, DynIndex)],
) -> Result<(TensorDynLen, Vec<DynIndex>)> {
    let setup = setup_direct_sum(a, b, pairs)?;

    let a_data = match a.storage.as_ref() {
        Storage::DenseF64(s) => s.as_slice(),
        _ => return Err(anyhow::anyhow!("Expected DenseF64 storage")),
    };
    let b_data = match b.storage.as_ref() {
        Storage::DenseF64(s) => s.as_slice(),
        _ => return Err(anyhow::anyhow!("Expected DenseF64 storage")),
    };

    let mut result_data: Vec<f64> = vec![0.0; setup.result_total];

    for result_linear in 0..setup.result_total {
        let result_multi = linear_to_multi(result_linear, &setup.result_dims);
        let common_multi: Vec<usize> = result_multi[..setup.n_common].to_vec();
        let paired_multi: Vec<usize> = result_multi[setup.n_common..].to_vec();

        let all_from_a = paired_multi
            .iter()
            .enumerate()
            .all(|(i, &pm)| pm < setup.paired_dims_a[i]);
        let all_from_b = paired_multi
            .iter()
            .enumerate()
            .all(|(i, &pm)| pm >= setup.paired_dims_a[i]);

        if all_from_a {
            let mut a_multi = vec![0usize; a.dims.len()];
            for (i, &cp) in setup.common_a_positions.iter().enumerate() {
                a_multi[cp] = common_multi[i];
            }
            for (i, &pp) in setup.paired_a_positions.iter().enumerate() {
                a_multi[pp] = paired_multi[i];
            }
            let a_linear = multi_to_linear(&a_multi, &setup.a_strides);
            result_data[result_linear] = a_data[a_linear];
        } else if all_from_b {
            let mut b_multi = vec![0usize; b.dims.len()];
            for (i, &cp) in setup.common_b_positions.iter().enumerate() {
                b_multi[cp] = common_multi[i];
            }
            for (i, &pp) in setup.paired_b_positions.iter().enumerate() {
                b_multi[pp] = paired_multi[i] - setup.paired_dims_a[i];
            }
            let b_linear = multi_to_linear(&b_multi, &setup.b_strides);
            result_data[result_linear] = b_data[b_linear];
        }
        // else: mixed case stays 0.0
    }

    let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(result_data)));
    let result = TensorDynLen::new(setup.result_indices, setup.result_dims, storage);
    Ok((result, setup.new_indices))
}

fn direct_sum_c64(
    a: &TensorDynLen,
    b: &TensorDynLen,
    pairs: &[(DynIndex, DynIndex)],
) -> Result<(TensorDynLen, Vec<DynIndex>)> {
    let setup = setup_direct_sum(a, b, pairs)?;

    let a_data = match a.storage.as_ref() {
        Storage::DenseC64(s) => s.as_slice(),
        _ => return Err(anyhow::anyhow!("Expected DenseC64 storage")),
    };
    let b_data = match b.storage.as_ref() {
        Storage::DenseC64(s) => s.as_slice(),
        _ => return Err(anyhow::anyhow!("Expected DenseC64 storage")),
    };

    let mut result_data: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); setup.result_total];

    for result_linear in 0..setup.result_total {
        let result_multi = linear_to_multi(result_linear, &setup.result_dims);
        let common_multi: Vec<usize> = result_multi[..setup.n_common].to_vec();
        let paired_multi: Vec<usize> = result_multi[setup.n_common..].to_vec();

        let all_from_a = paired_multi
            .iter()
            .enumerate()
            .all(|(i, &pm)| pm < setup.paired_dims_a[i]);
        let all_from_b = paired_multi
            .iter()
            .enumerate()
            .all(|(i, &pm)| pm >= setup.paired_dims_a[i]);

        if all_from_a {
            let mut a_multi = vec![0usize; a.dims.len()];
            for (i, &cp) in setup.common_a_positions.iter().enumerate() {
                a_multi[cp] = common_multi[i];
            }
            for (i, &pp) in setup.paired_a_positions.iter().enumerate() {
                a_multi[pp] = paired_multi[i];
            }
            let a_linear = multi_to_linear(&a_multi, &setup.a_strides);
            result_data[result_linear] = a_data[a_linear];
        } else if all_from_b {
            let mut b_multi = vec![0usize; b.dims.len()];
            for (i, &cp) in setup.common_b_positions.iter().enumerate() {
                b_multi[cp] = common_multi[i];
            }
            for (i, &pp) in setup.paired_b_positions.iter().enumerate() {
                b_multi[pp] = paired_multi[i] - setup.paired_dims_a[i];
            }
            let b_linear = multi_to_linear(&b_multi, &setup.b_strides);
            result_data[result_linear] = b_data[b_linear];
        }
        // else: mixed case stays 0.0
    }

    let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(result_data)));
    let result = TensorDynLen::new(setup.result_indices, setup.result_dims, storage);
    Ok((result, setup.new_indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::{DynId, NoSymmSpace, TagSet};

    type TestIndex = Index<DynId, NoSymmSpace, TagSet>;

    #[test]
    fn test_direct_sum_simple() {
        // Create two tensors with one common index
        let i = TestIndex::new_dyn(2);
        let j = TestIndex::new_dyn(3);
        let k = TestIndex::new_dyn(4);

        // A[i, j] - 2x3 tensor
        let a = TensorDynLen::new(
            vec![
                Index::new(i.id.clone(), i.symm.clone()),
                Index::new(j.id.clone(), j.symm.clone()),
            ],
            vec![2, 3],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![
                1.0, 2.0, 3.0, // i=0
                4.0, 5.0, 6.0, // i=1
            ]))),
        );

        // B[i, k] - 2x4 tensor
        let b = TensorDynLen::new(
            vec![
                Index::new(i.id.clone(), i.symm.clone()),
                Index::new(k.id.clone(), k.symm.clone()),
            ],
            vec![2, 4],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![
                10.0, 20.0, 30.0, 40.0, // i=0
                50.0, 60.0, 70.0, 80.0, // i=1
            ]))),
        );

        // Direct sum along (j, k)
        let j_idx: DynIndex = Index::new(j.id.clone(), j.symm.clone());
        let k_idx: DynIndex = Index::new(k.id.clone(), k.symm.clone());
        let (result, new_indices) = direct_sum(&a, &b, &[(j_idx, k_idx)]).unwrap();

        // Result should be 2x7 (common i, merged j+k)
        assert_eq!(result.dims.len(), 2);
        assert_eq!(result.dims[0], 2); // i dimension
        assert_eq!(result.dims[1], 7); // j+k dimension (3+4)
        assert_eq!(new_indices.len(), 1);
        assert_eq!(new_indices[0].dim(), 7);

        // Check data
        let data = match result.storage.as_ref() {
            Storage::DenseF64(s) => s.as_slice(),
            _ => panic!("Expected DenseF64"),
        };

        // i=0: [1, 2, 3, 10, 20, 30, 40]
        // i=1: [4, 5, 6, 50, 60, 70, 80]
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
        assert_eq!(data[3], 10.0);
        assert_eq!(data[4], 20.0);
        assert_eq!(data[5], 30.0);
        assert_eq!(data[6], 40.0);
        assert_eq!(data[7], 4.0);
        assert_eq!(data[8], 5.0);
        assert_eq!(data[9], 6.0);
        assert_eq!(data[10], 50.0);
        assert_eq!(data[11], 60.0);
        assert_eq!(data[12], 70.0);
        assert_eq!(data[13], 80.0);
    }

    #[test]
    fn test_direct_sum_multiple_pairs() {
        // Create tensors with two indices to be paired
        let i = TestIndex::new_dyn(2);
        let j = TestIndex::new_dyn(2);
        let k = TestIndex::new_dyn(3);
        let l = TestIndex::new_dyn(3);

        // A[i, j] - 2x2 tensor
        let a = TensorDynLen::new(
            vec![
                Index::new(i.id.clone(), i.symm.clone()),
                Index::new(j.id.clone(), j.symm.clone()),
            ],
            vec![2, 2],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![
                1.0, 2.0, 3.0, 4.0,
            ]))),
        );

        // B[k, l] - 3x3 tensor (no common indices)
        let b = TensorDynLen::new(
            vec![
                Index::new(k.id.clone(), k.symm.clone()),
                Index::new(l.id.clone(), l.symm.clone()),
            ],
            vec![3, 3],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
            ]))),
        );

        // Direct sum along (i, k) and (j, l)
        let i_idx: DynIndex = Index::new(i.id.clone(), i.symm.clone());
        let j_idx: DynIndex = Index::new(j.id.clone(), j.symm.clone());
        let k_idx: DynIndex = Index::new(k.id.clone(), k.symm.clone());
        let l_idx: DynIndex = Index::new(l.id.clone(), l.symm.clone());
        let (result, new_indices) = direct_sum(&a, &b, &[(i_idx, k_idx), (j_idx, l_idx)]).unwrap();

        // Result should be 5x5 (2+3, 2+3)
        assert_eq!(result.dims.len(), 2);
        assert_eq!(result.dims[0], 5);
        assert_eq!(result.dims[1], 5);
        assert_eq!(new_indices.len(), 2);
    }
}
