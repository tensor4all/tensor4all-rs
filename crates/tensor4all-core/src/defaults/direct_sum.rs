//! Direct sum operations for tensors.
//!
//! This module provides functionality to compute the direct sum of tensors
//! along specified index pairs. The direct sum concatenates tensor data along
//! the paired indices, creating new indices with combined dimensions.
//! [`direct_sum`] combines two tensors; [`direct_sum_many`] combines any
//! number in one pass.
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
    if !(a.is_f64() && b.is_f64() || a.is_complex() && b.is_complex()) {
        return Err(IdxTensorError::Operation {
            operation: "direct_sum",
            source: std::sync::Arc::new(std::io::Error::other(
                "direct_sum requires both tensors to have the same dense scalar type (f64 or Complex64)",
            )),
        });
    }
    if pairs.is_empty() {
        return Err(IdxTensorError::from(anyhow::anyhow!(
            "direct_sum requires at least one index pair"
        )));
    }
    direct_sum_dispatch(&[a, b], Groups::Pairs(pairs))
}

/// Compute the direct sum of any number of tensors in one pass.
///
/// Each entry of `groups` lists one index per operand, in operand order, and
/// becomes one new index whose dimension is the sum of the group's
/// dimensions. Operand `i` occupies its own diagonal block, after the blocks
/// of operands `0..i`, and every other entry is zero. The indices of an
/// operand that belong to no group (its common indices) must be the same
/// set, with the same dimensions, for every operand. With two operands this
/// is [`direct_sum`] with `pairs[g] = (groups[g][0], groups[g][1])`.
///
/// Every result element is written once. A chain of binary [`direct_sum`]
/// calls copies the growing partial result at every step instead.
///
/// # Arguments
///
/// * `operands` - The tensors to combine; at least one. They must share one
///   dense scalar type, `f64` or `Complex64`.
/// * `groups` - The grouped indices; at least one group, each with exactly
///   `operands.len()` entries, entry `i` an index of operand `i`. An index of
///   an operand may appear in at most one group.
///
/// # Returns
///
/// A tuple of the direct-sum tensor and the new indices (one per group, in
/// `groups` order). The result's indices are `operands[0]`'s common indices in
/// its order, followed by the new indices; the payload is column-major.
///
/// # Errors
///
/// Returns [`IdxTensorError::ShapeMismatch`] when a common index of one
/// operand is not a common index of another one (an index present on one side
/// only, or matching only in id but not in prime level or tags), and
/// [`IdxTensorError::Operation`] when `operands` or `groups` is empty, when a
/// group does not have one index per operand, when a grouped index is missing
/// from its operand or appears in two groups, when a common index differs in
/// dimension, when the operands do not share one dense scalar type, when an
/// operand tracks gradients, or when the backend construction fails.
///
/// # Example
///
/// ```
/// use tensor4all_core::{direct_sum_many, DynIndex, IdxTensor, IndexLike};
///
/// # fn main() -> anyhow::Result<()> {
/// let site = DynIndex::new_dyn(2);
/// let (b0, b1, b2) = (DynIndex::new_dyn(1), DynIndex::new_dyn(2), DynIndex::new_dyn(1));
///
/// let t0 = IdxTensor::from_dense(vec![site.clone(), b0.clone()], vec![1.0, 2.0])?;
/// let t1 = IdxTensor::from_dense(vec![site.clone(), b1.clone()], vec![3.0, 4.0, 5.0, 6.0])?;
/// let t2 = IdxTensor::from_dense(vec![b2.clone(), site.clone()], vec![7.0, 8.0])?;
/// let (result, new_indices) = direct_sum_many(&[&t0, &t1, &t2], &[vec![b0, b1, b2]])?;
///
/// // The common `site` index comes first, then the new bond of dimension 1 + 2 + 1.
/// assert_eq!(result.indices(), &[site, new_indices[0].clone()]);
/// assert_eq!(new_indices[0].dim(), 4);
/// assert_eq!(result.to_vec::<f64>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// # Ok(())
/// # }
/// ```
pub fn direct_sum_many(
    operands: &[&IdxTensor],
    groups: &[Vec<DynIndex>],
) -> std::result::Result<(IdxTensor, Vec<DynIndex>), IdxTensorError> {
    if operands.is_empty() {
        return Err(IdxTensorError::from(anyhow::anyhow!(
            "direct_sum requires at least one operand"
        )));
    }
    let all = |is: fn(&IdxTensor) -> bool| operands.iter().all(|operand| is(operand));
    if !(all(IdxTensor::is_f64) || all(IdxTensor::is_complex)) {
        return Err(IdxTensorError::from(anyhow::anyhow!(
            "direct_sum requires every operand to have the same dense scalar type (f64 or Complex64)"
        )));
    }
    if groups.is_empty() {
        return Err(IdxTensorError::from(anyhow::anyhow!(
            "direct_sum requires at least one index group"
        )));
    }
    direct_sum_dispatch(operands, Groups::Lists(groups))
}

/// Run the typed kernel for operands already checked to share `f64` or
/// `Complex64` storage, with non-empty `operands` and `groups`.
fn direct_sum_dispatch(
    operands: &[&IdxTensor],
    groups: Groups<'_>,
) -> std::result::Result<(IdxTensor, Vec<DynIndex>), IdxTensorError> {
    if operands[0].is_f64() {
        direct_sum_typed::<f64>(operands, groups).map_err(IdxTensorError::from)
    } else {
        direct_sum_typed::<num_complex::Complex64>(operands, groups).map_err(IdxTensorError::from)
    }
}

/// "first", "second", ... for operand `i` in error messages.
fn ordinal(i: usize) -> String {
    match i {
        0 => "first".to_string(),
        1 => "second".to_string(),
        2 => "third".to_string(),
        _ => format!("{}-th", i + 1),
    }
}

/// The grouped indices of a direct sum, read in place: the binary path's
/// pairs or the N-ary path's lists, without converting one into the other.
#[derive(Clone, Copy)]
enum Groups<'a> {
    Pairs(&'a [(DynIndex, DynIndex)]),
    Lists(&'a [Vec<DynIndex>]),
}

impl<'a> Groups<'a> {
    fn len(self) -> usize {
        match self {
            Groups::Pairs(pairs) => pairs.len(),
            Groups::Lists(lists) => lists.len(),
        }
    }

    /// Number of indices listed in group `g`.
    fn width(self, g: usize) -> usize {
        match self {
            Groups::Pairs(_) => 2,
            Groups::Lists(lists) => lists[g].len(),
        }
    }

    /// Operand `i`'s index of group `g`; `i < self.width(g)`.
    fn index(self, g: usize, i: usize) -> &'a DynIndex {
        match self {
            Groups::Pairs(pairs) if i == 0 => &pairs[g].0,
            Groups::Pairs(pairs) => &pairs[g].1,
            Groups::Lists(lists) => &lists[g][i],
        }
    }
}

/// Setup data for direct sum computation.
struct DirectSumSetup {
    /// Per operand: its dims, computed once.
    operand_dims: Vec<Vec<usize>>,
    /// Per operand: the position, in the operand's own index order, of each
    /// result axis (common indices, then the operand's index of each group).
    operand_axes: Vec<Vec<usize>>,
    /// Per operand: its own column-major strides.
    operand_strides: Vec<Vec<usize>>,
    result_indices: Vec<DynIndex>,
    result_dims: Vec<usize>,
    result_total: usize,
    new_indices: Vec<DynIndex>,
    n_common: usize,
}

fn setup_direct_sum(operands: &[&IdxTensor], groups: Groups<'_>) -> Result<DirectSumSetup> {
    use std::collections::{HashMap, HashSet};

    let n = operands.len();
    let operand_dims: Vec<Vec<usize>> = operands.iter().map(|operand| operand.dims()).collect();
    // Build maps from full index metadata to positions. Same-id indices that
    // differ by prime level or tags are distinct axes.
    let index_maps: Vec<HashMap<&DynIndex, usize>> = operands
        .iter()
        .map(|operand| {
            operand
                .indices
                .iter()
                .enumerate()
                .map(|(i, idx)| (idx, i))
                .collect()
        })
        .collect();

    // Build the set of grouped indices of every operand.
    let mut grouped: Vec<HashSet<&DynIndex>> = vec![HashSet::new(); n];
    for g in 0..groups.len() {
        if groups.width(g) != n {
            return Err(anyhow::anyhow!(
                "index group {g} has {} entries, expected one per operand ({n})",
                groups.width(g)
            ));
        }
        for (i, operand_grouped) in grouped.iter_mut().enumerate() {
            let idx = groups.index(g, i);
            if !index_maps[i].contains_key(idx) {
                return Err(anyhow::anyhow!(
                    "Index not found in {} tensor (direct_sum)",
                    ordinal(i)
                ));
            }
            if !operand_grouped.insert(idx) {
                return Err(anyhow::anyhow!(
                    "index {idx:?} of the {} tensor appears in two groups (direct_sum)",
                    ordinal(i)
                ));
            }
        }
    }

    // Identify common (non-grouped) indices. Every non-grouped index of an
    // operand must be a non-grouped index of every other one: an index left on
    // one side only has no axis in the result, so accepting it would silently
    // read that operand at coordinate 0 (tensor4all/tensor4all-rs#1018).
    // Iterate over the first operand's indices (Vec) to preserve
    // deterministic order, using the other maps only for lookups.
    let unpaired_mismatch = |i: usize, idx: &DynIndex| {
        let other = if n == 2 {
            "the other operand"
        } else {
            "every other operand"
        };
        anyhow::Error::new(IdxTensorError::ShapeMismatch {
            operation: "direct_sum",
            expected: format!(
                "every non-paired index of the {} operand to be a non-paired index of {other}",
                ordinal(i)
            ),
            actual: format!("{idx:?} has no non-paired counterpart"),
        })
    };
    let first = operands[0];
    let first_dims = &operand_dims[0];
    let mut operand_axes: Vec<Vec<usize>> = Vec::with_capacity(n);
    operand_axes.push(
        (0..first.indices.len())
            .filter(|&pos| !grouped[0].contains(&first.indices[pos]))
            .collect(),
    );
    let n_common = operand_axes[0].len();
    for (i, operand) in operands.iter().enumerate().skip(1) {
        let dims = &operand_dims[i];
        let mut axes = Vec::with_capacity(operand.indices.len());
        for &first_pos in &operand_axes[0][..n_common] {
            let idx = &first.indices[first_pos];
            let pos = match index_maps[i].get(idx) {
                Some(&pos) if !grouped[i].contains(idx) => pos,
                _ => return Err(unpaired_mismatch(0, idx)),
            };
            if first_dims[first_pos] != dims[pos] {
                return Err(anyhow::anyhow!(
                    "Dimension mismatch for common index: {} vs {}",
                    first_dims[first_pos],
                    dims[pos]
                ));
            }
            axes.push(pos);
        }
        if let Some(idx) = operand.indices.iter().find(|&idx| {
            !grouped[i].contains(idx)
                && !(index_maps[0].contains_key(idx) && !grouped[0].contains(idx))
        }) {
            return Err(unpaired_mismatch(i, idx));
        }
        operand_axes.push(axes);
    }
    for (i, axes) in operand_axes.iter_mut().enumerate() {
        axes.extend((0..groups.len()).map(|g| index_maps[i][groups.index(g, i)]));
    }

    // Create new indices for grouped dimensions.
    let mut new_indices: Vec<DynIndex> = Vec::with_capacity(groups.len());
    for g in 0..groups.len() {
        let new_dim = (0..n).try_fold(0usize, |acc, i| {
            acc.checked_add(operand_dims[i][operand_axes[i][n_common + g]])
                .ok_or_else(|| anyhow::anyhow!("direct_sum index dimension overflowed usize"))
        })?;
        let new_index = DynIndex::new_link(new_dim)
            .map_err(|e| anyhow::anyhow!("Failed to create index: {:?}", e))?;
        new_indices.push(new_index);
    }

    // Common indices first, then the new indices; the new indices preserve
    // the full index identity semantics (id + plev + tags).
    let result_rank = n_common + new_indices.len();
    let mut result_indices: Vec<DynIndex> = Vec::with_capacity(result_rank);
    let mut result_dims: Vec<usize> = Vec::with_capacity(result_rank);
    for &pos in &operand_axes[0][..n_common] {
        result_indices.push(first.indices[pos].clone());
        result_dims.push(first_dims[pos]);
    }
    for new_idx in &new_indices {
        result_indices.push(new_idx.clone());
        result_dims.push(new_idx.dim());
    }

    let result_total = checked_product(&result_dims, "direct_sum result")?;
    let operand_strides = operand_dims
        .iter()
        .map(|dims| checked_strides(dims, "direct_sum operand"))
        .collect::<Result<Vec<_>>>()?;

    Ok(DirectSumSetup {
        operand_dims,
        operand_axes,
        operand_strides,
        result_indices,
        result_dims,
        result_total,
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
    operands: &[&IdxTensor],
    groups: Groups<'_>,
) -> Result<(IdxTensor, Vec<DynIndex>)> {
    if operands.iter().any(|operand| operand.tracks_grad()) {
        return Err(anyhow::anyhow!(
            "direct_sum does not support tracked tensors until an AD-preserving block path is available"
        ));
    }
    let setup = setup_direct_sum(operands, groups)?;
    let result_strides = checked_strides(&setup.result_dims, "direct_sum result")?;
    // INVARIANT: the zero fill is semantic: every entry outside the diagonal
    // blocks is zero.
    let mut result_data: Vec<T> = vec![T::zero(); setup.result_total];
    let mut cursor = vec![0usize; setup.result_dims.len()];

    // Operand `i` starts, along every new axis, after the blocks of operands
    // `0..i`.
    let mut offsets = vec![0usize; groups.len()];
    for (i, operand) in operands.iter().enumerate() {
        let dims = &setup.operand_dims[i];
        let axes = &setup.operand_axes[i];
        let layout = BlockLayout {
            dims: axes.iter().map(|&axis| dims[axis]).collect(),
            src_strides: axes
                .iter()
                .map(|&axis| setup.operand_strides[i][axis])
                .collect(),
            dst_strides: &result_strides,
            dst_start: offsets
                .iter()
                .zip(&result_strides[setup.n_common..])
                .map(|(offset, stride)| offset * stride)
                .sum(),
        };
        operand.with_dense_slice::<T, _>(|values| {
            copy_block(values, &mut result_data, &layout, &mut cursor)
        })?;
        for (offset, &axis) in offsets.iter_mut().zip(&axes[setup.n_common..]) {
            *offset += dims[axis];
        }
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
