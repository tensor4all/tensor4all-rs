//! Direct sum operations for tensors.
//!
//! This module provides functionality to compute the direct sum of tensors
//! along specified index pairs. The direct sum concatenates tensor data along
//! the paired indices, creating new indices with combined dimensions.
//!
//! This module works with concrete types (`DynIndex`, `IdxTensor`) only.

use crate::defaults::DynIndex;
use crate::defaults::IdxTensorError;
use crate::index_like::IndexLike;
use crate::tensor::IdxTensor;
use anyhow::Result;
use num_traits::Zero;
use tensor4all_tensorbackend::TensorElement;

/// Compute the direct sum of two tensors along specified index pairs.
///
/// For tensors A and B with indices to be summed specified as pairs,
/// creates a new tensor C where each paired index has dimension = dim_A + dim_B.
/// Non-paired indices must match exactly between A and B.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `pairs` - Pairs of (a_index, b_index) to be summed. Each pair creates
///
///   a new index in the result with dimension = dim(a_index) + dim(b_index).
///
/// # Returns
///
/// A tuple of:
/// - The direct sum tensor
/// - The new indices created for the summed dimensions (one per pair)
///
/// # Errors
///
/// Returns [`IdxTensorError::ShapeMismatch`] when a non-paired index of either
/// operand is not a non-paired index of the other one (an index present on one
/// side only, or matching only in id but not in prime level or tags), and
/// [`IdxTensorError::Operation`] when `pairs` is empty, when a paired index is
/// missing or a common index differs in dimension, when the operands have
/// different dense scalar types, or when the backend construction fails.
///
/// # Example
///
/// ```
/// use tensor4all_core::{direct_sum, DynIndex, IdxTensor};
///
/// # fn main() -> anyhow::Result<()> {
/// let j = DynIndex::new_dyn(2);
/// let k = DynIndex::new_dyn(3);
///
/// let a = IdxTensor::from_dense(vec![j.clone()], vec![1.0, 2.0])?;
/// let b = IdxTensor::from_dense(vec![k.clone()], vec![3.0, 4.0, 5.0])?;
/// let (result, new_indices) = direct_sum(&a, &b, &[(j.clone(), k.clone())])?;
///
/// assert_eq!(new_indices.len(), 1);
/// assert_eq!(result.to_vec::<f64>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// # Ok(())
/// # }
/// ```
pub fn direct_sum(
    a: &IdxTensor,
    b: &IdxTensor,
    pairs: &[(DynIndex, DynIndex)],
) -> std::result::Result<(IdxTensor, Vec<DynIndex>), IdxTensorError> {
    if a.is_f64() && b.is_f64() {
        direct_sum_typed::<f64>(a, b, pairs).map_err(IdxTensorError::from)
    } else if a.is_complex() && b.is_complex() {
        direct_sum_typed::<num_complex::Complex64>(a, b, pairs).map_err(IdxTensorError::from)
    } else {
        Err(IdxTensorError::Operation {
            operation: "direct_sum",
            source: std::sync::Arc::new(std::io::Error::other(
                "direct_sum requires both tensors to have the same dense scalar type (f64 or Complex64)",
            )),
        })
    }
}

/// Setup data for direct sum computation
struct DirectSumSetup {
    common_a_positions: Vec<usize>,
    common_b_positions: Vec<usize>,
    paired_a_positions: Vec<usize>,
    paired_b_positions: Vec<usize>,
    paired_dims_a: Vec<usize>,
    result_indices: Vec<DynIndex>,
    result_dims: Vec<usize>,
    result_total: usize,
    a_strides: Vec<usize>,
    b_strides: Vec<usize>,
    new_indices: Vec<DynIndex>,
    n_common: usize,
}

fn setup_direct_sum(
    a: &IdxTensor,
    b: &IdxTensor,
    pairs: &[(DynIndex, DynIndex)],
) -> Result<DirectSumSetup> {
    use std::collections::HashMap;

    if pairs.is_empty() {
        return Err(anyhow::anyhow!(
            "direct_sum requires at least one index pair"
        ));
    }

    // Build maps from full index metadata to positions. Same-id indices that
    // differ by prime level or tags are distinct axes.
    let a_idx_map: HashMap<&DynIndex, usize> = a
        .indices
        .iter()
        .enumerate()
        .map(|(i, idx)| (idx, i))
        .collect();
    let b_idx_map: HashMap<&DynIndex, usize> = b
        .indices
        .iter()
        .enumerate()
        .map(|(i, idx)| (idx, i))
        .collect();

    // Build set of paired indices.
    let mut a_paired_indices: std::collections::HashSet<&DynIndex> =
        std::collections::HashSet::new();
    let mut b_paired_indices: std::collections::HashSet<&DynIndex> =
        std::collections::HashSet::new();

    for (a_idx, b_idx) in pairs {
        if !a_idx_map.contains_key(a_idx) {
            return Err(anyhow::anyhow!(
                "Index not found in first tensor (direct_sum)"
            ));
        }
        if !b_idx_map.contains_key(b_idx) {
            return Err(anyhow::anyhow!(
                "Index not found in second tensor (direct_sum)"
            ));
        }
        a_paired_indices.insert(a_idx);
        b_paired_indices.insert(b_idx);
    }

    // Identify common (non-paired) indices. Every non-paired index of either
    // operand must be a non-paired index of the other one: an index left on
    // one side only has no axis in the result, so accepting it would silently
    // read that operand at coordinate 0 (tensor4all/tensor4all-rs#1018).
    // Iterate over a.indices (Vec) to preserve deterministic order, using
    // b_idx_map only for lookups.
    let unpaired_mismatch = |side: &str, idx: &DynIndex| {
        anyhow::Error::new(IdxTensorError::ShapeMismatch {
            operation: "direct_sum",
            expected: format!(
                "every non-paired index of the {side} operand to be a non-paired index of the other operand"
            ),
            actual: format!("{idx:?} has no non-paired counterpart"),
        })
    };
    let mut common_a_positions: Vec<usize> = Vec::new();
    let mut common_b_positions: Vec<usize> = Vec::new();
    for (a_pos, a_idx) in a.indices.iter().enumerate() {
        if a_paired_indices.contains(a_idx) {
            continue;
        }
        let b_pos = match b_idx_map.get(a_idx) {
            Some(&b_pos) if !b_paired_indices.contains(a_idx) => b_pos,
            _ => return Err(unpaired_mismatch("first", a_idx)),
        };
        let a_dims = a.dims();
        let b_dims = b.dims();
        if a_dims[a_pos] != b_dims[b_pos] {
            return Err(anyhow::anyhow!(
                "Dimension mismatch for common index: {} vs {}",
                a_dims[a_pos],
                b_dims[b_pos]
            ));
        }
        common_a_positions.push(a_pos);
        common_b_positions.push(b_pos);
    }
    if let Some(b_idx) = b.indices.iter().find(|&b_idx| {
        !b_paired_indices.contains(b_idx)
            && !(a_idx_map.contains_key(b_idx) && !a_paired_indices.contains(b_idx))
    }) {
        return Err(unpaired_mismatch("second", b_idx));
    }

    // Build paired positions and dimensions
    let paired_a_positions: Vec<usize> = pairs.iter().map(|(a_idx, _)| a_idx_map[a_idx]).collect();
    let paired_b_positions: Vec<usize> = pairs.iter().map(|(_, b_idx)| b_idx_map[b_idx]).collect();
    let a_dims = a.dims();
    let b_dims = b.dims();
    let paired_dims_a: Vec<usize> = paired_a_positions.iter().map(|&p| a_dims[p]).collect();
    let paired_dims_b: Vec<usize> = paired_b_positions.iter().map(|&p| b_dims[p]).collect();

    // Create new indices for paired dimensions
    let mut new_indices: Vec<DynIndex> = Vec::new();
    for (&dim_a, &dim_b) in paired_dims_a.iter().zip(&paired_dims_b) {
        let new_dim = dim_a
            .checked_add(dim_b)
            .ok_or_else(|| anyhow::anyhow!("direct_sum index dimension overflowed usize"))?;
        let new_index = DynIndex::new_link(new_dim)
            .map_err(|e| anyhow::anyhow!("Failed to create index: {:?}", e))?;
        new_indices.push(new_index);
    }

    // Build result indices and dimensions
    let mut result_indices: Vec<DynIndex> = Vec::new();
    let mut result_dims: Vec<usize> = Vec::new();

    // Common indices first.
    let a_dims = a.dims();
    for &a_pos in &common_a_positions {
        result_indices.push(a.indices[a_pos].clone());
        result_dims.push(a_dims[a_pos]);
    }

    // New indices from pairs preserve the full index identity semantics (id + plev + tags).
    for new_idx in &new_indices {
        result_indices.push(new_idx.clone());
        result_dims.push(new_idx.dim());
    }

    // Compute column-major strides (leftmost index fastest).
    let result_total = checked_product(&result_dims, "direct_sum result")?;

    let a_dims = a.dims();
    let b_dims = b.dims();
    let a_strides = checked_strides(&a_dims, "direct_sum lhs")?;
    let b_strides = checked_strides(&b_dims, "direct_sum rhs")?;

    let n_common = common_a_positions.len();

    Ok(DirectSumSetup {
        common_a_positions,
        common_b_positions,
        paired_a_positions,
        paired_b_positions,
        paired_dims_a,
        result_indices,
        result_dims,
        result_total,
        a_strides,
        b_strides,
        new_indices,
        n_common,
    })
}

fn checked_product(dims: &[usize], name: &str) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("{name} shape product overflowed usize"))
    })
}

fn checked_strides(dims: &[usize], name: &str) -> Result<Vec<usize>> {
    let mut strides = vec![1usize; dims.len()];
    for i in 1..dims.len() {
        strides[i] = strides[i - 1]
            .checked_mul(dims[i - 1])
            .ok_or_else(|| anyhow::anyhow!("{name} stride overflowed usize"))?;
    }
    Ok(strides)
}

fn direct_sum_typed<T: TensorElement + Zero>(
    a: &IdxTensor,
    b: &IdxTensor,
    pairs: &[(DynIndex, DynIndex)],
) -> Result<(IdxTensor, Vec<DynIndex>)> {
    if a.tracks_grad() || b.tracks_grad() {
        return Err(anyhow::anyhow!(
            "direct_sum does not support tracked tensors until an AD-preserving block path is available"
        ));
    }
    let setup = setup_direct_sum(a, b, pairs)?;
    let result_strides = checked_strides(&setup.result_dims, "direct_sum result")?;
    // INVARIANT: the zero fill is semantic: every entry outside the two
    // diagonal blocks is zero.
    let mut result_data: Vec<T> = vec![T::zero(); setup.result_total];
    let mut cursor = vec![0usize; setup.result_dims.len()];

    // `a` fills the leading block of every paired axis; `b` starts after it.
    let b_start = setup
        .paired_dims_a
        .iter()
        .zip(&result_strides[setup.n_common..])
        .map(|(dim, stride)| dim * stride)
        .sum();
    for (operand, common, paired, strides, dst_start) in [
        (
            a,
            &setup.common_a_positions,
            &setup.paired_a_positions,
            &setup.a_strides,
            0,
        ),
        (
            b,
            &setup.common_b_positions,
            &setup.paired_b_positions,
            &setup.b_strides,
            b_start,
        ),
    ] {
        let dims = operand.dims();
        let axes = common.iter().chain(paired);
        let layout = BlockLayout {
            dims: axes.clone().map(|&axis| dims[axis]).collect(),
            src_strides: axes.map(|&axis| strides[axis]).collect(),
            dst_strides: &result_strides,
            dst_start,
        };
        operand.with_dense_slice::<T, _>(|values| {
            copy_block(values, &mut result_data, &layout, &mut cursor)
        })?;
    }

    let result = IdxTensor::from_dense(setup.result_indices, result_data)?;
    Ok((result, setup.new_indices))
}

/// Where one operand's block goes: its extents in result axis order, the
/// operand's own (column-major) stride of each result axis, the result's
/// strides, and the result offset of the block's first element.
struct BlockLayout<'a> {
    dims: Vec<usize>,
    src_strides: Vec<usize>,
    dst_strides: &'a [usize],
    dst_start: usize,
}

/// Copy one operand, read in its own layout, into its diagonal block of the
/// result.
///
/// The leading result axes along which both layouts are contiguous are merged
/// into one run moved with `copy_from_slice`. When not even the first axis is
/// contiguous in the operand, the inner loop walks that axis with its operand
/// stride. An odometer steps the remaining axes once per run, not once per
/// element, so no multi-index is decoded per element.
fn copy_block<T: Copy>(
    values: &[T],
    result: &mut [T],
    layout: &BlockLayout<'_>,
    cursor: &mut [usize],
) {
    let rank = layout.dims.len();
    let mut merged = 0;
    let mut run = 1usize;
    while merged < rank && layout.src_strides[merged] == run && layout.dst_strides[merged] == run {
        run *= layout.dims[merged];
        merged += 1;
    }
    // With no merged axis, axis 0 is walked element by element (its result
    // stride is 1, its operand stride `src_strides[0]`).
    let (inner_len, inner_stride, first_outer) = if merged == 0 {
        (layout.dims[0], layout.src_strides[0], 1)
    } else {
        (run, 1, merged)
    };
    cursor.fill(0);
    let mut src = 0usize;
    let mut dst = layout.dst_start;
    // INVARIANT: `cursor[a] < dims[a]` on every axis, so `src` stays inside the
    // operand's `prod(dims)` elements and `dst` inside the operand's block,
    // which `setup_direct_sum` sized to lie inside the result. `rank >= 1`
    // (at least one pair) and `inner_len > 0`, because an `IdxTensor` has no
    // zero dimension (`checked_total_size` rejects one at construction).
    loop {
        if inner_stride == 1 {
            result[dst..dst + inner_len].copy_from_slice(&values[src..src + inner_len]);
        } else {
            for (k, slot) in result[dst..dst + inner_len].iter_mut().enumerate() {
                *slot = values[src + k * inner_stride];
            }
        }
        let mut axis = first_outer;
        loop {
            if axis == rank {
                return;
            }
            cursor[axis] += 1;
            src += layout.src_strides[axis];
            dst += layout.dst_strides[axis];
            if cursor[axis] < layout.dims[axis] {
                break;
            }
            src -= layout.src_strides[axis] * layout.dims[axis];
            dst -= layout.dst_strides[axis] * layout.dims[axis];
            cursor[axis] = 0;
            axis += 1;
        }
    }
}

#[cfg(test)]
mod tests;
