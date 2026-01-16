//! Pairwise tensor contraction using batched matrix multiplication.
//!
//! This module implements the core contraction operation, similar to PyTorch's
//! `sumproduct_pair` in aten/src/ATen/native/Linear.cpp.
//!
//! The key idea is to categorize dimensions into:
//! - **lro**: dimensions present in left, right, AND output (batch dimensions)
//! - **lo**: dimensions present in left and output only
//! - **ro**: dimensions present in right and output only
//! - **sum**: dimensions to be summed over (contracted between left and right)
//! - **left_trace**: dimensions only in left, not in output (traced/summed within left)
//! - **right_trace**: dimensions only in right, not in output (traced/summed within right)
//!
//! Then reshape to 3D tensors and use batched matrix multiplication:
//! - left:  [lro_size, lo_size, sum_size]
//! - right: [lro_size, sum_size, ro_size]
//! - result: [lro_size, lo_size, ro_size]

use crate::AxisId;
use mdarray::{DynRank, Layout, Slice, Tensor};
use mdarray_linalg::matmul::{MatMul, MatMulBuilder};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};
use std::collections::HashSet;

/// Perform pairwise tensor contraction.
///
/// # Arguments
/// * `left` - (axis_ids, tensor) for left operand
/// * `right` - (axis_ids, tensor) for right operand
/// * `output_ids` - IDs that should appear in the output
///
/// # Returns
/// Contracted tensor with axes in the order specified by output_ids
pub fn sumproduct_pair<T, ID, L>(
    backend: &impl MatMul<T>,
    left: (&[ID], &Slice<T, DynRank, L>),
    right: (&[ID], &Slice<T, DynRank, L>),
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    let (left_ids, left_tensor) = left;
    let (right_ids, right_tensor) = right;

    let left_set: HashSet<&ID> = left_ids.iter().collect();
    let right_set: HashSet<&ID> = right_ids.iter().collect();
    let output_set: HashSet<&ID> = output_ids.iter().collect();

    // Categorize dimensions
    let mut lro: Vec<&ID> = Vec::new(); // in left, right, and output
    let mut lo: Vec<&ID> = Vec::new(); // in left and output only
    let mut ro: Vec<&ID> = Vec::new(); // in right and output only
    let mut sum_ids: Vec<&ID> = Vec::new(); // to be summed (contracted between left and right)
    let mut left_trace: Vec<&ID> = Vec::new(); // only in left, not in output (trace)
    let mut right_trace: Vec<&ID> = Vec::new(); // only in right, not in output (trace)

    // Process left IDs
    for id in left_ids {
        if right_set.contains(id) {
            if output_set.contains(id) {
                if !lro.contains(&id) {
                    lro.push(id);
                }
            } else if !sum_ids.contains(&id) {
                sum_ids.push(id);
            }
        } else if output_set.contains(id) {
            if !lo.contains(&id) {
                lo.push(id);
            }
        } else {
            // Only in left, not in right, not in output -> trace
            if !left_trace.contains(&id) {
                left_trace.push(id);
            }
        }
    }

    // Process right IDs
    for id in right_ids {
        if !left_set.contains(id) {
            if output_set.contains(id) {
                if !ro.contains(&id) {
                    ro.push(id);
                }
            } else {
                // Only in right, not in left, not in output -> trace
                if !right_trace.contains(&id) {
                    right_trace.push(id);
                }
            }
        }
    }

    // Handle traces before contraction if needed
    let (left_ids_after_trace, left_tensor_traced) = if left_trace.is_empty() {
        (left_ids.to_vec(), left_tensor.to_tensor())
    } else {
        trace_tensor(left_ids, left_tensor, &left_trace)
    };

    let (right_ids_after_trace, right_tensor_traced) = if right_trace.is_empty() {
        (right_ids.to_vec(), right_tensor.to_tensor())
    } else {
        trace_tensor(right_ids, right_tensor, &right_trace)
    };

    // Re-reference the traced tensors
    let left_ids = &left_ids_after_trace[..];
    let left_tensor = &left_tensor_traced;
    let right_ids = &right_ids_after_trace[..];
    let right_tensor = &right_tensor_traced;

    // Build dimension maps
    let left_dim_map: std::collections::HashMap<&ID, usize> = left_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id, left_tensor.dim(i)))
        .collect();
    let right_dim_map: std::collections::HashMap<&ID, usize> = right_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id, right_tensor.dim(i)))
        .collect();

    // Compute sizes
    let lro_size: usize = lro.iter().map(|id| left_dim_map[id]).product();
    let lo_size: usize = lo.iter().map(|id| left_dim_map[id]).product();
    let ro_size: usize = ro.iter().map(|id| right_dim_map[id]).product();
    let sum_size: usize = sum_ids
        .iter()
        .map(|id| left_dim_map.get(id).copied().unwrap_or(1))
        .product();

    // Handle edge cases
    let lro_size = if lro_size == 0 { 1 } else { lro_size };
    let lo_size = if lo_size == 0 { 1 } else { lo_size };
    let ro_size = if ro_size == 0 { 1 } else { ro_size };
    let sum_size = if sum_size == 0 { 1 } else { sum_size };

    // Build permutation for left: [lro..., lo..., sum...]
    let left_perm: Vec<usize> = lro
        .iter()
        .chain(lo.iter())
        .chain(sum_ids.iter())
        .filter_map(|id| left_ids.iter().position(|x| x == *id))
        .collect();

    // Build permutation for right: [lro..., sum..., ro...]
    let right_perm: Vec<usize> = lro
        .iter()
        .chain(sum_ids.iter())
        .chain(ro.iter())
        .filter_map(|id| right_ids.iter().position(|x| x == *id))
        .collect();

    // Permute and reshape
    let left_permuted = if left_perm.len() == left_ids.len() && !left_perm.is_empty() {
        left_tensor.permute(left_perm.clone()).to_tensor()
    } else if left_ids.is_empty() {
        left_tensor.to_tensor()
    } else {
        // Handle case where some dimensions are missing
        left_tensor.to_tensor()
    };

    let right_permuted = if right_perm.len() == right_ids.len() && !right_perm.is_empty() {
        right_tensor.permute(right_perm.clone()).to_tensor()
    } else if right_ids.is_empty() {
        right_tensor.to_tensor()
    } else {
        right_tensor.to_tensor()
    };

    // Reshape to 3D for batched matmul
    let left_dyn = left_permuted.into_dyn();
    let left_reshaped = left_dyn.reshape([lro_size, lo_size, sum_size]).to_tensor().into_dyn();
    let right_dyn = right_permuted.into_dyn();
    let right_reshaped = right_dyn.reshape([lro_size, sum_size, ro_size]).to_tensor().into_dyn();

    // Batched matrix multiplication (loop for now)
    let result_3d = batched_matmul(backend, &left_reshaped, &right_reshaped);

    // Build output shape
    let mut output_shape: Vec<usize> = Vec::new();
    for id in output_ids {
        let dim = left_dim_map
            .get(id)
            .or_else(|| right_dim_map.get(id))
            .copied()
            .unwrap_or(1);
        output_shape.push(dim);
    }

    if output_shape.is_empty() {
        // Scalar output - result_3d is [1,1,1], reshape to scalar [1]
        return result_3d.reshape([1]).to_tensor().into_dyn();
    }

    // Reshape result to output shape
    // First, reshape to [lro_dims..., lo_dims..., ro_dims...]
    let mut intermediate_shape: Vec<usize> = Vec::new();
    for id in &lro {
        intermediate_shape.push(left_dim_map[id]);
    }
    for id in &lo {
        intermediate_shape.push(left_dim_map[id]);
    }
    for id in &ro {
        intermediate_shape.push(right_dim_map[id]);
    }

    if intermediate_shape.is_empty() {
        intermediate_shape.push(1);
    }

    let result_expanded = result_3d.reshape(intermediate_shape);

    // Build permutation to match output order
    let intermediate_ids: Vec<&ID> = lro.iter().chain(lo.iter()).chain(ro.iter()).copied().collect();

    let output_perm: Vec<usize> = output_ids
        .iter()
        .map(|id| {
            intermediate_ids
                .iter()
                .position(|x| *x == id)
                .unwrap_or_else(|| panic!("Output ID {:?} not found in intermediate result", id))
        })
        .collect();

    if output_perm.iter().enumerate().all(|(i, &p)| i == p) {
        result_expanded.to_tensor().into_dyn()
    } else {
        result_expanded
            .as_ref()
            .permute(output_perm)
            .to_tensor()
            .into_dyn()
    }
}

/// Sum over (trace) the specified axes in a tensor.
///
/// Returns the new axis IDs (with traced axes removed) and the traced tensor.
fn trace_tensor<T, ID, L>(
    ids: &[ID],
    tensor: &Slice<T, DynRank, L>,
    trace_ids: &[&ID],
) -> (Vec<ID>, Tensor<T, DynRank>)
where
    T: Clone + Zero + std::ops::Add<Output = T>,
    ID: AxisId,
    L: Layout,
{
    // Find axes to sum over
    let trace_axes: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, id)| trace_ids.iter().any(|t| *t == *id))
        .map(|(i, _)| i)
        .collect();

    // Compute remaining IDs
    let remaining_ids: Vec<ID> = ids
        .iter()
        .enumerate()
        .filter(|(i, _)| !trace_axes.contains(i))
        .map(|(_, id)| id.clone())
        .collect();

    // Sum over trace axes from back to front
    let mut result = tensor.to_tensor();
    for &axis in trace_axes.iter().rev() {
        result = sum_axis_dyn(&result, axis);
    }

    (remaining_ids, result)
}

/// Sum a DynRank tensor along a given axis.
fn sum_axis_dyn<T>(tensor: &Tensor<T, DynRank>, axis: usize) -> Tensor<T, DynRank>
where
    T: Clone + Zero + std::ops::Add<Output = T>,
{
    let shape: Vec<usize> = (0..tensor.rank()).map(|i| tensor.dim(i)).collect();

    // Build output shape (remove the summed axis)
    let mut out_shape: Vec<usize> = shape.clone();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_len: usize = out_shape.iter().product();
    let mut out_data = vec![T::zero(); out_len];

    // Iterate over all elements and accumulate
    let total_len: usize = shape.iter().product();
    for flat_idx in 0..total_len {
        // Convert flat index to multi-index
        let mut multi_idx: Vec<usize> = Vec::with_capacity(shape.len());
        let mut remaining = flat_idx;
        for &dim in shape.iter().rev() {
            multi_idx.push(remaining % dim);
            remaining /= dim;
        }
        multi_idx.reverse();

        // Compute output flat index (skip the summed axis)
        let mut out_flat_idx = 0;
        let mut stride = 1;
        for i in (0..out_shape.len()).rev() {
            let src_i = if i >= axis { i + 1 } else { i };
            out_flat_idx += multi_idx[src_i] * stride;
            stride *= out_shape[i];
        }

        out_data[out_flat_idx] = out_data[out_flat_idx].clone() + tensor[&multi_idx[..]].clone();
    }

    Tensor::from(out_data).into_shape(out_shape).into_dyn()
}

/// Batched matrix multiplication using a simple loop.
///
/// Input tensors must have shape [batch, m, k] and [batch, k, n].
/// Returns tensor with shape [batch, m, n].
fn batched_matmul<T, L: Layout>(
    backend: &impl MatMul<T>,
    a: &Slice<T, DynRank, L>,
    b: &Slice<T, DynRank, L>,
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
{
    assert_eq!(a.rank(), 3, "Left tensor must be 3D");
    assert_eq!(b.rank(), 3, "Right tensor must be 3D");

    let batch = a.dim(0);
    let m = a.dim(1);
    let k = a.dim(2);
    let n = b.dim(2);

    assert_eq!(b.dim(0), batch, "Batch dimensions must match");
    assert_eq!(b.dim(1), k, "Inner dimensions must match");

    // Collect data into flat vector, then reshape
    let mut result_data = vec![T::zero(); batch * m * n];

    for bi in 0..batch {
        // Extract 2D slice by creating a view at batch index bi
        let a_2d = extract_2d_slice(a, bi, m, k);
        let b_2d = extract_2d_slice(b, bi, k, n);
        let c_2d = backend.matmul(&a_2d, &b_2d).eval();

        for i in 0..m {
            for j in 0..n {
                result_data[bi * m * n + i * n + j] = c_2d[[i, j]];
            }
        }
    }

    Tensor::from(result_data).into_shape([batch, m, n]).into_dyn()
}

/// Extract a 2D slice from a 3D tensor at batch index `bi`.
fn extract_2d_slice<T: Clone, L: Layout>(
    tensor: &Slice<T, DynRank, L>,
    bi: usize,
    rows: usize,
    cols: usize,
) -> Tensor<T, (usize, usize)> {
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(tensor[[bi, i, j]].clone());
        }
    }
    Tensor::from(data).into_shape([rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use mdarray::tensor;
    use mdarray_linalg::Naive;

    #[test]
    fn test_sumproduct_pair_matmul() {
        // ij,jk->ik
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = tensor![[5.0, 6.0], [7.0, 8.0]].into_dyn();

        let result = sumproduct_pair(
            &Naive,
            (&[0u32, 1], &a),
            (&[1u32, 2], &b),
            &[0, 2],
        );

        assert_relative_eq!(result[[0, 0]], 19.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1]], 22.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0]], 43.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 1]], 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sumproduct_pair_outer() {
        // i,j->ij (outer product)
        let a = tensor![1.0, 2.0].into_dyn();
        let b = tensor![3.0, 4.0, 5.0].into_dyn();

        let result = sumproduct_pair(&Naive, (&[0u32], &a), (&[1u32], &b), &[0, 1]);

        assert_eq!(result.rank(), 2);
        assert_eq!(result.dim(0), 2);
        assert_eq!(result.dim(1), 3);
        assert_relative_eq!(result[[0, 0]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 2]], 5.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 1]], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sumproduct_pair_dot() {
        // i,i-> (dot product)
        let a = tensor![1.0, 2.0, 3.0].into_dyn();
        let b = tensor![4.0, 5.0, 6.0].into_dyn();

        let result = sumproduct_pair(&Naive, (&[0u32], &a), (&[0u32], &b), &[]);

        // 1*4 + 2*5 + 3*6 = 32
        // Scalar result is [1] shaped
        assert_eq!(result.shape().dims(), &[1]);
        assert_relative_eq!(result[[0]].re(), 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sumproduct_pair_batch() {
        // bij,bjk->bik (batched matmul)
        let a = tensor![[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]].into_dyn();
        let b = tensor![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]].into_dyn();

        let result = sumproduct_pair(
            &Naive,
            (&[0u32, 1, 2], &a),
            (&[0u32, 2, 3], &b),
            &[0, 1, 3],
        );

        // Batch 0: I @ [[1,2],[3,4]] = [[1,2],[3,4]]
        // Batch 1: 2I @ [[5,6],[7,8]] = [[10,12],[14,16]]
        assert_relative_eq!(result[[0, 0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1, 1]], 4.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0, 0]], 10.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 1, 1]], 16.0, epsilon = 1e-10);
    }
}
