//! Direct sum operations for tensors.
//!
//! This module provides functionality to compute the direct sum of tensors
//! along specified index groups. The direct sum concatenates tensor data along
//! the grouped indices, creating new indices with combined dimensions, and
//! fills every off-diagonal block with zeros.
//!
//! [`direct_sum_many`] stacks any number of operands in one pass;
//! [`direct_sum`] is its two-operand form.
//!
//! This module works with concrete types (`DynIndex`, `IdxTensor`) only.

use std::collections::HashMap;

use crate::defaults::DynIndex;
use crate::defaults::IdxTensorError;
use crate::tensor::IdxTensor;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tensor4all_tensorbackend::TensorElement;

const OP: &str = "direct_sum";

/// Compute the direct sum of two tensors along specified index pairs.
///
/// For tensors A and B with indices to be summed specified as pairs,
/// creates a new tensor C where each paired index has dimension = dim_A + dim_B.
/// Non-paired indices must match exactly between A and B. This is the
/// two-operand form of [`direct_sum_many`].
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
/// - The direct sum tensor: the non-paired indices of `a` in `a`'s order,
///   followed by the new indices in `pairs` order (column-major payload).
/// - The new indices created for the summed dimensions (one per pair)
///
/// # Errors
///
/// Same as [`direct_sum_many`] with the operands `[a, b]`:
/// [`IdxTensorError::ShapeMismatch`] when a non-paired index of either operand
/// is not a non-paired index of the other one (an index present on one side
/// only, or matching only in id but not in prime level or tags), and
/// [`IdxTensorError::Operation`] when `pairs` is empty, when a paired index is
/// missing or used twice, when a common index differs in dimension, when the
/// operands have different dense scalar types, when an operand tracks
/// gradients, or when the backend construction fails.
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
    let groups: Vec<Vec<DynIndex>> = pairs
        .iter()
        .map(|(a_index, b_index)| vec![a_index.clone(), b_index.clone()])
        .collect();
    direct_sum_many(&[a, b], &groups)
}

/// Compute the direct sum of any number of tensors in one pass.
///
/// Each entry of `groups` lists one index per operand, in operand order. The
/// result replaces each group by one new index whose dimension is the sum of
/// the group's dimensions, and places operand `i` in its own diagonal block:
/// along every new index, operand `i` occupies the coordinates after those of
/// operands `0..i`. All entries outside the diagonal blocks are zero. The
/// indices of an operand that belong to no group (its common indices) must be
/// the same set, with the same dimensions, for every operand.
///
/// Summing `N` operands this way writes every result element once. Chaining
/// [`direct_sum`] instead copies the growing partial result at every step.
///
/// # Arguments
///
/// * `operands` - The tensors to stack; at least one. They must share one
///   dense scalar type (`f32`, `f64`, `Complex32` or `Complex64`).
/// * `groups` - The grouped indices; at least one group, each with exactly
///   `operands.len()` entries, entry `i` an index of operand `i`. An index of
///   an operand may appear in at most one group.
///
/// # Returns
///
/// A tuple of the direct-sum tensor and the new indices (one per group, in
/// `groups` order). The result's indices are the common indices in
/// `operands[0]`'s order, followed by the new indices; its payload is
/// column-major.
///
/// # Errors
///
/// Returns [`IdxTensorError::ShapeMismatch`] when a common index of one
/// operand is not a common index of another one (an index left on one side
/// only, or matching only in id but not in prime level or tags), and
/// [`IdxTensorError::Operation`] when `operands` or `groups` is empty, when a
/// group does not have one entry per operand, when a grouped index is missing
/// from its operand or used twice, when a common index differs in dimension,
/// when the operands have different dense scalar types, when an operand
/// tracks gradients, or when the backend construction fails.
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
/// assert_eq!(
///     result.to_vec::<f64>()?,
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
/// );
/// # Ok(())
/// # }
/// ```
pub fn direct_sum_many(
    operands: &[&IdxTensor],
    groups: &[Vec<DynIndex>],
) -> std::result::Result<(IdxTensor, Vec<DynIndex>), IdxTensorError> {
    if operands.is_empty() {
        return Err(operation_error("direct_sum requires at least one operand"));
    }
    if groups.is_empty() {
        return Err(operation_error(
            "direct_sum requires at least one index group",
        ));
    }
    if operands.iter().any(|operand| operand.tracks_grad()) {
        return Err(operation_error(
            "direct_sum does not support tracked tensors until an AD-preserving block path is available",
        ));
    }
    let plan = plan_direct_sum(operands, groups)?;
    let all = |is: fn(&IdxTensor) -> bool| operands.iter().all(|operand| is(operand));
    if all(IdxTensor::is_f64) {
        direct_sum_typed::<f64>(operands, plan)
    } else if all(IdxTensor::is_c64) {
        direct_sum_typed::<Complex64>(operands, plan)
    } else if all(IdxTensor::is_f32) {
        direct_sum_typed::<f32>(operands, plan)
    } else if all(IdxTensor::is_c32) {
        direct_sum_typed::<Complex32>(operands, plan)
    } else {
        Err(operation_error(
            "direct_sum requires every operand to have the same dense scalar type (f32, f64, Complex32 or Complex64)",
        ))
    }
}

fn operation_error(message: &'static str) -> IdxTensorError {
    IdxTensorError::operation(OP, anyhow::anyhow!(message))
}

/// Validated layout of an N-ary direct sum.
struct DirectSumPlan {
    /// Per operand: its indices in result order (common indices, then its
    /// grouped index of each group), the permutation applied before copying.
    operand_orders: Vec<Vec<DynIndex>>,
    /// `group_dims[g][i]`: dimension of operand `i`'s index in group `g`.
    group_dims: Vec<Vec<usize>>,
    /// Product of the common dimensions.
    common_len: usize,
    result_indices: Vec<DynIndex>,
    result_dims: Vec<usize>,
    new_indices: Vec<DynIndex>,
}

fn plan_direct_sum(
    operands: &[&IdxTensor],
    groups: &[Vec<DynIndex>],
) -> std::result::Result<DirectSumPlan, IdxTensorError> {
    let n = operands.len();
    let mut grouped: Vec<HashMap<&DynIndex, usize>> = vec![HashMap::new(); n];
    let mut group_dims = Vec::with_capacity(groups.len());
    for (g, group) in groups.iter().enumerate() {
        if group.len() != n {
            return Err(IdxTensorError::operation(
                OP,
                anyhow::anyhow!(
                    "index group {g} has {} entries, expected one per operand ({n})",
                    group.len()
                ),
            ));
        }
        let mut dims = Vec::with_capacity(n);
        for (i, index) in group.iter().enumerate() {
            let Some(position) = operands[i].indices.iter().position(|own| own == index) else {
                return Err(IdxTensorError::operation(
                    OP,
                    anyhow::anyhow!(
                        "index {index:?} of group {g} not found in operand {i} (direct_sum)"
                    ),
                ));
            };
            if grouped[i].insert(index, g).is_some() {
                return Err(IdxTensorError::operation(
                    OP,
                    anyhow::anyhow!("index {index:?} of operand {i} appears in two groups"),
                ));
            }
            dims.push(operands[i].dims()[position]);
        }
        group_dims.push(dims);
    }

    // Common (non-grouped) indices, in operands[0]'s order. Every common index
    // of each operand must be a common index of every other one: an index left
    // on one side only has no axis in the result, so accepting it would
    // silently read that operand at coordinate 0
    // (tensor4all/tensor4all-rs#1018).
    let common_mismatch = |i: usize, index: &DynIndex| IdxTensorError::ShapeMismatch {
        operation: OP,
        expected: format!(
            "every non-paired index of operand {i} to be a non-paired index of every other operand"
        ),
        actual: format!("{index:?} has no non-paired counterpart"),
    };
    let first = operands[0];
    let common: Vec<(&DynIndex, usize)> = first
        .indices
        .iter()
        .zip(first.dims())
        .filter(|(index, _)| !grouped[0].contains_key(index))
        .collect();
    let mut operand_orders = Vec::with_capacity(n);
    for (i, operand) in operands.iter().enumerate() {
        let dims = operand.dims();
        let mut order = Vec::with_capacity(operand.indices.len());
        for &(index, dim) in &common {
            let position = operand
                .indices
                .iter()
                .position(|own| own == index)
                .filter(|_| !grouped[i].contains_key(index))
                .ok_or_else(|| common_mismatch(0, index))?;
            if dims[position] != dim {
                return Err(IdxTensorError::operation(
                    OP,
                    anyhow::anyhow!(
                        "Dimension mismatch for common index: {dim} vs {}",
                        dims[position]
                    ),
                ));
            }
            order.push(operand.indices[position].clone());
        }
        if let Some(extra) = operand.indices.iter().find(|index| {
            !grouped[i].contains_key(index) && !common.iter().any(|(c, _)| c == index)
        }) {
            return Err(common_mismatch(i, extra));
        }
        order.extend(groups.iter().map(|group| group[i].clone()));
        operand_orders.push(order);
    }

    let mut result_indices: Vec<DynIndex> =
        common.iter().map(|(index, _)| (*index).clone()).collect();
    let mut result_dims: Vec<usize> = common.iter().map(|&(_, dim)| dim).collect();
    let common_len = checked_product(&result_dims)?;
    let mut new_indices = Vec::with_capacity(groups.len());
    for dims in &group_dims {
        let total = dims.iter().try_fold(0usize, |acc, &dim| {
            acc.checked_add(dim)
                .ok_or_else(|| operation_error("direct_sum index dimension overflowed usize"))
        })?;
        let index = DynIndex::new_link(total)
            .map_err(|error| IdxTensorError::operation(OP, anyhow::Error::new(error)))?;
        result_indices.push(index.clone());
        result_dims.push(total);
        new_indices.push(index);
    }
    checked_product(&result_dims)?;

    Ok(DirectSumPlan {
        operand_orders,
        group_dims,
        common_len,
        result_indices,
        result_dims,
        new_indices,
    })
}

fn checked_product(dims: &[usize]) -> std::result::Result<usize, IdxTensorError> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| operation_error("direct_sum shape product overflowed usize"))
    })
}

fn direct_sum_typed<T: TensorElement + Zero>(
    operands: &[&IdxTensor],
    plan: DirectSumPlan,
) -> std::result::Result<(IdxTensor, Vec<DynIndex>), IdxTensorError> {
    let n_common = plan.result_dims.len() - plan.group_dims.len();
    // Column-major stride of each new axis, in units of one common block.
    // INVARIANT: the products are bounded by the result size that
    // `plan_direct_sum` checked for overflow.
    let mut group_strides = Vec::with_capacity(plan.group_dims.len());
    let mut stride = 1usize;
    for &extent in &plan.result_dims[n_common..] {
        group_strides.push(stride);
        stride *= extent;
    }
    // INVARIANT: the zero fill is semantic: every entry outside the diagonal
    // blocks is zero.
    let mut result: Vec<T> = vec![T::zero(); plan.common_len * stride];

    let mut offsets = vec![0usize; plan.group_dims.len()];
    let mut coordinates = vec![0usize; plan.group_dims.len()];
    let mut dims = vec![0usize; plan.group_dims.len()];
    for (i, operand) in operands.iter().enumerate() {
        for (dim, group) in dims.iter_mut().zip(&plan.group_dims) {
            *dim = group[i];
        }
        let aligned = operand.permute_indices(&plan.operand_orders[i])?;
        aligned.with_dense_slice::<T, _>(|values| {
            copy_block(
                values,
                &mut result,
                plan.common_len,
                &dims,
                &offsets,
                &group_strides,
                &mut coordinates,
            )
        })?;
        for (offset, dim) in offsets.iter_mut().zip(&dims) {
            *offset += dim;
        }
    }

    let tensor = IdxTensor::from_dense(plan.result_indices, result)?;
    Ok((tensor, plan.new_indices))
}

/// Copy one operand, laid out as `[common..., group axes...]` column-major,
/// into its diagonal block of the result.
///
/// The common axes and the first group axis are contiguous in both layouts,
/// so the copy moves runs of `common_len * dims[0]` elements and steps an
/// odometer over the remaining group axes once per run, not once per element.
fn copy_block<T: Copy>(
    values: &[T],
    result: &mut [T],
    common_len: usize,
    dims: &[usize],
    offsets: &[usize],
    group_strides: &[usize],
    coordinates: &mut [usize],
) {
    // INVARIANT: `run > 0`, as `chunks_exact` requires: an `IdxTensor` has no
    // zero dimension (`checked_total_size` rejects one at construction).
    let run = common_len * dims[0];
    coordinates.fill(0);
    // INVARIANT: `values` holds `common_len * prod(dims)` elements (the
    // operand's own shape), and every destination run lies inside the result
    // because `offsets[g] + dims[g]` is at most the extent of new axis `g`.
    for chunk in values.chunks_exact(run) {
        let block: usize = offsets
            .iter()
            .zip(coordinates.iter())
            .zip(group_strides)
            .map(|((&offset, &coordinate), &stride)| (offset + coordinate) * stride)
            .sum();
        let start = block * common_len;
        result[start..start + run].copy_from_slice(chunk);
        for axis in 1..dims.len() {
            coordinates[axis] += 1;
            if coordinates[axis] < dims[axis] {
                break;
            }
            coordinates[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests;
