//! Multi-tensor contraction for hyperedges.
//!
//! When an index appears in 3+ tensors (hyperedge), we need to contract
//! all those tensors simultaneously to compute the correct sum:
//! `sum_j(A[i,j] * B[j,k] * C[j,l])` (correct)
//! vs `(sum_j A[i,j]*B[j,k]) * C[j,l]` (incorrect pairwise approach)
//!
//! This module provides functions to contract multiple tensors at once,
//! handling the hyperedge index as a batch dimension and using pairwise
//! contractions within each batch.

use crate::sumproduct_pair::sumproduct_pair;
use crate::AxisId;
use mdarray::{DynRank, Layout, Slice, Tensor};
use mdarray_linalg::matmul::MatMul;
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};
use std::collections::{HashMap, HashSet};

/// Contract multiple tensors sharing a common index (hyperedge contraction).
///
/// This uses batched GEMM by treating the hyperedge index as a batch dimension.
/// For each value of the hyperedge index, pairwise contractions are performed,
/// then results are summed across the batch dimension.
///
/// # Arguments
/// * `backend` - Linear algebra backend for matrix multiplication
/// * `inputs` - List of (axis_ids, tensor) pairs for all tensors to contract
/// * `contracted_index` - The index being summed over (hyperedge)
/// * `output_ids` - Desired output axis IDs (must not contain contracted_index)
///
/// # Returns
/// Contracted tensor with axes ordered according to output_ids
pub fn multi_contract<T, ID, L>(
    backend: &impl MatMul<T>,
    inputs: &[(&[ID], &Slice<T, DynRank, L>)],
    contracted_index: &ID,
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    assert!(
        inputs.len() >= 2,
        "multi_contract requires at least 2 tensors"
    );

    // Build dimension maps
    let mut dim_map: HashMap<ID, usize> = HashMap::new();
    for (ids, tensor) in inputs {
        for (axis, id) in ids.iter().enumerate() {
            dim_map.entry(id.clone()).or_insert(tensor.dim(axis));
        }
    }

    let contracted_dim = dim_map.get(contracted_index).copied().unwrap_or(1);

    // Verify contracted_index appears in multiple tensors
    let sharing_count = inputs
        .iter()
        .filter(|(ids, _)| ids.contains(contracted_index))
        .count();
    assert!(
        sharing_count >= 2,
        "contracted_index must appear in at least 2 tensors"
    );

    // For each slice along the contracted dimension, perform pairwise contractions
    // Then sum the results

    // Extract slices and contract
    let mut batch_results: Vec<Tensor<T, DynRank>> = Vec::with_capacity(contracted_dim);

    for j in 0..contracted_dim {
        // Extract slice at j for each tensor that contains the contracted index
        // For tensors not containing the index, use the full tensor
        let sliced_tensors: Vec<(Vec<ID>, Tensor<T, DynRank>)> = inputs
            .iter()
            .map(|(ids, tensor)| {
                if let Some(axis) = ids.iter().position(|id| id == contracted_index) {
                    // Extract slice at axis=j
                    let (new_ids, sliced) = extract_slice_at_axis(ids, tensor, axis, j);
                    (new_ids, sliced)
                } else {
                    // No contracted index in this tensor - use as is
                    (ids.to_vec(), tensor.to_tensor())
                }
            })
            .collect();

        // Contract all sliced tensors pairwise
        // Output includes all non-contracted indices
        let slice_output_ids: Vec<ID> = output_ids.to_vec();

        let contracted = contract_pairwise(backend, &sliced_tensors, &slice_output_ids);
        batch_results.push(contracted);
    }

    // Sum all batch results
    sum_tensors(&batch_results, output_ids)
}

/// Contract multiple tensors sharing a common index (hyperedge contraction).
///
/// This is a simpler interface that doesn't require a MatMul backend,
/// but uses naive element-wise computation. Primarily for testing.
#[allow(dead_code)]
pub fn multi_contract_naive<T, ID, L>(
    inputs: &[(&[ID], &Slice<T, DynRank, L>)],
    contracted_index: &ID,
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: Clone + Zero + One + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    ID: AxisId,
    L: Layout,
{
    assert!(
        inputs.len() >= 2,
        "multi_contract requires at least 2 tensors"
    );

    // Build dimension maps
    let mut dim_map: HashMap<ID, usize> = HashMap::new();
    for (ids, tensor) in inputs {
        for (axis, id) in ids.iter().enumerate() {
            dim_map.entry(id.clone()).or_insert(tensor.dim(axis));
        }
    }

    // Build output shape
    let output_shape: Vec<usize> = output_ids
        .iter()
        .map(|id| dim_map.get(id).copied().unwrap_or(1))
        .collect();

    let contracted_dim = dim_map.get(contracted_index).copied().unwrap_or(1);

    // Allocate output
    let output_len: usize = output_shape.iter().product::<usize>().max(1);
    let mut output_data = vec![T::zero(); output_len];

    let output_strides = compute_strides(&output_shape);

    // Iterate over contracted index
    for j in 0..contracted_dim {
        iterate_multi_index(&output_shape, |out_multi_idx| {
            let mut product = T::one();

            for (ids, tensor) in inputs {
                let tensor_idx: Vec<usize> = ids
                    .iter()
                    .map(|id| {
                        if id == contracted_index {
                            j
                        } else if let Some(pos) = output_ids.iter().position(|oid| oid == id) {
                            out_multi_idx[pos]
                        } else {
                            0
                        }
                    })
                    .collect();

                product = product * tensor[&tensor_idx[..]].clone();
            }

            let out_flat_idx: usize = out_multi_idx
                .iter()
                .zip(output_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum();

            output_data[out_flat_idx] = output_data[out_flat_idx].clone() + product;
        });
    }

    if output_shape.is_empty() {
        Tensor::from(output_data).into_shape([1]).into_dyn()
    } else {
        Tensor::from(output_data)
            .into_shape(output_shape)
            .into_dyn()
    }
}

/// Contract multiple tensors sharing multiple common indices.
///
/// This uses batched GEMM by processing one contracted index at a time.
/// For each contracted index, we use `multi_contract` to handle it efficiently.
pub fn multi_contract_general<T, ID, L>(
    backend: &impl MatMul<T>,
    inputs: &[(&[ID], &Slice<T, DynRank, L>)],
    contracted_indices: &[ID],
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    if inputs.is_empty() {
        return Tensor::from(vec![T::zero()]).into_shape([1]).into_dyn();
    }

    if inputs.len() == 1 {
        return trace_and_permute(inputs[0].0, inputs[0].1, contracted_indices, output_ids);
    }

    if contracted_indices.is_empty() {
        // No indices to contract - just outer product via pairwise
        let tensors: Vec<(Vec<ID>, Tensor<T, DynRank>)> = inputs
            .iter()
            .map(|(ids, t)| (ids.to_vec(), t.to_tensor()))
            .collect();
        return contract_pairwise(backend, &tensors, output_ids);
    }

    // Process one contracted index at a time
    // Find the first contracted index that appears in 2+ tensors
    let mut current_tensors: Vec<(Vec<ID>, Tensor<T, DynRank>)> = inputs
        .iter()
        .map(|(ids, t)| (ids.to_vec(), t.to_tensor()))
        .collect();

    for contracted_idx in contracted_indices {
        // Find tensors sharing this index
        let sharing: Vec<usize> = current_tensors
            .iter()
            .enumerate()
            .filter(|(_, (ids, _))| ids.contains(contracted_idx))
            .map(|(i, _)| i)
            .collect();

        if sharing.len() < 2 {
            // Index appears in 0-1 tensors, skip (will be handled later or is already gone)
            continue;
        }

        // Collect tensors to contract (remove in reverse order)
        let mut indices_to_remove = sharing.clone();
        indices_to_remove.sort_by(|a, b| b.cmp(a));

        let contracted_tensors: Vec<(Vec<ID>, Tensor<T, DynRank>)> = indices_to_remove
            .iter()
            .map(|&i| current_tensors.remove(i))
            .collect();

        // Compute intermediate output IDs (keep all non-contracted indices)
        let output_set: HashSet<&ID> = output_ids.iter().collect();
        let remaining_ids: HashSet<&ID> = current_tensors
            .iter()
            .flat_map(|(ids, _)| ids.iter())
            .collect();

        let mut intermediate_output: Vec<ID> = Vec::new();
        let mut seen: HashSet<&ID> = HashSet::new();
        for (ids, _) in &contracted_tensors {
            for id in ids {
                if id == contracted_idx {
                    continue;
                }
                let needed = output_set.contains(id) || remaining_ids.contains(id);
                if needed && !seen.contains(id) {
                    intermediate_output.push(id.clone());
                    seen.insert(id);
                }
            }
        }

        // Build refs for multi_contract
        let tensor_refs: Vec<(&[ID], &Slice<T, DynRank, mdarray::Dense>)> = contracted_tensors
            .iter()
            .map(|(ids, t)| (ids.as_slice(), t.as_ref()))
            .collect();

        let result = multi_contract(backend, &tensor_refs, contracted_idx, &intermediate_output);
        current_tensors.push((intermediate_output, result));
    }

    // Contract remaining tensors pairwise
    if current_tensors.len() == 1 {
        let (final_ids, final_tensor) = current_tensors.pop().unwrap();
        return permute_to_output(&final_ids, final_tensor, output_ids);
    }

    contract_pairwise(backend, &current_tensors, output_ids)
}

/// Extract a slice of a tensor at a specific index along a given axis.
/// Returns (remaining_ids, sliced_tensor).
fn extract_slice_at_axis<T, ID, L>(
    ids: &[ID],
    tensor: &Slice<T, DynRank, L>,
    axis: usize,
    idx: usize,
) -> (Vec<ID>, Tensor<T, DynRank>)
where
    T: Clone,
    ID: AxisId,
    L: Layout,
{
    let shape: Vec<usize> = (0..tensor.rank()).map(|i| tensor.dim(i)).collect();

    // Build new shape without the sliced axis
    let mut new_shape: Vec<usize> = shape.clone();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    // Build new ids without the sliced axis
    let mut new_ids: Vec<ID> = ids.to_vec();
    new_ids.remove(axis);

    // Extract data
    let new_len: usize = new_shape.iter().product();
    let mut new_data = Vec::with_capacity(new_len);

    // Iterate over all indices of the result
    iterate_multi_index(&new_shape, |out_idx| {
        // Build full index by inserting `idx` at `axis`
        let mut full_idx = out_idx.to_vec();
        full_idx.insert(axis, idx);
        new_data.push(tensor[&full_idx[..]].clone());
    });

    (
        new_ids,
        Tensor::from(new_data).into_shape(new_shape).into_dyn(),
    )
}

/// Contract a list of tensors pairwise until one remains.
fn contract_pairwise<T, ID>(
    backend: &impl MatMul<T>,
    tensors: &[(Vec<ID>, Tensor<T, DynRank>)],
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
{
    if tensors.is_empty() {
        return Tensor::from(vec![T::zero()]).into_shape([1]).into_dyn();
    }

    if tensors.len() == 1 {
        // Permute to output order
        return permute_to_output(&tensors[0].0, tensors[0].1.clone(), output_ids);
    }

    let output_set: HashSet<&ID> = output_ids.iter().collect();

    // Clone tensors into working list
    let mut operands: Vec<(Vec<ID>, Tensor<T, DynRank>)> = tensors.to_vec();

    // Contract pairwise until one remains
    while operands.len() > 1 {
        let (ids_b, tensor_b) = operands.pop().unwrap();
        let (ids_a, tensor_a) = operands.pop().unwrap();

        // Compute intermediate output
        let remaining_ids: HashSet<&ID> = operands.iter().flat_map(|(ids, _)| ids.iter()).collect();
        let intermediate_output =
            compute_intermediate_output(&ids_a, &ids_b, &output_set, &remaining_ids);

        let result = sumproduct_pair(
            backend,
            (&ids_a, &tensor_a),
            (&ids_b, &tensor_b),
            &intermediate_output,
        );

        operands.push((intermediate_output, result));
    }

    let (final_ids, final_tensor) = operands.pop().unwrap();
    permute_to_output(&final_ids, final_tensor, output_ids)
}

/// Compute intermediate output IDs after contracting two tensors.
fn compute_intermediate_output<ID: AxisId>(
    ids_a: &[ID],
    ids_b: &[ID],
    final_output: &HashSet<&ID>,
    remaining_ids: &HashSet<&ID>,
) -> Vec<ID> {
    let mut result = Vec::new();

    for id in ids_a {
        let needed = final_output.contains(id) || remaining_ids.contains(id);
        if needed && !result.contains(id) {
            result.push(id.clone());
        }
    }

    for id in ids_b {
        let needed = final_output.contains(id) || remaining_ids.contains(id);
        if needed && !result.contains(id) {
            result.push(id.clone());
        }
    }

    result
}

/// Sum a list of tensors element-wise.
fn sum_tensors<T, ID>(tensors: &[Tensor<T, DynRank>], output_ids: &[ID]) -> Tensor<T, DynRank>
where
    T: Clone + Zero + std::ops::Add<Output = T>,
    ID: AxisId,
{
    if tensors.is_empty() {
        let shape: Vec<usize> = vec![1];
        return Tensor::from(vec![T::zero()]).into_shape(shape).into_dyn();
    }

    let shape: Vec<usize> = (0..tensors[0].rank()).map(|i| tensors[0].dim(i)).collect();
    let len: usize = shape.iter().product::<usize>().max(1);
    let mut result_data = vec![T::zero(); len];

    for tensor in tensors {
        for (i, val) in tensor.iter().enumerate() {
            result_data[i] = result_data[i].clone() + val.clone();
        }
    }

    if output_ids.is_empty() || shape.is_empty() {
        Tensor::from(result_data).into_shape([1]).into_dyn()
    } else {
        Tensor::from(result_data).into_shape(shape).into_dyn()
    }
}

/// Permute tensor to match output order.
fn permute_to_output<T, ID>(
    current_ids: &[ID],
    tensor: Tensor<T, DynRank>,
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: Clone,
    ID: AxisId,
{
    if output_ids.is_empty() {
        return tensor;
    }

    if current_ids == output_ids {
        return tensor;
    }

    // Handle case where current has fewer dims (scalar-ish result)
    if current_ids.is_empty() {
        return tensor;
    }

    let perm: Vec<usize> = output_ids
        .iter()
        .filter_map(|id| current_ids.iter().position(|x| x == id))
        .collect();

    if perm.len() != output_ids.len() {
        // Some output IDs not in current - just return as is
        return tensor;
    }

    tensor.as_ref().permute(perm).to_tensor()
}

/// Helper: trace over specified indices and permute to output order.
fn trace_and_permute<T, ID, L>(
    ids: &[ID],
    tensor: &Slice<T, DynRank, L>,
    trace_indices: &[ID],
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: Clone + Zero + std::ops::Add<Output = T>,
    ID: AxisId,
    L: Layout,
{
    let trace_set: HashSet<&ID> = trace_indices.iter().collect();

    let trace_axes: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, id)| trace_set.contains(id))
        .map(|(i, _)| i)
        .collect();

    if trace_axes.is_empty() {
        return permute_tensor(ids, tensor, output_ids);
    }

    let mut result = tensor.to_tensor();
    let mut current_ids = ids.to_vec();

    for &axis in trace_axes.iter().rev() {
        result = sum_axis(&result, axis);
        current_ids.remove(axis);
    }

    permute_tensor(&current_ids, &result, output_ids)
}

/// Helper: permute tensor to match output order.
fn permute_tensor<T, ID, L>(
    current_ids: &[ID],
    tensor: &Slice<T, DynRank, L>,
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: Clone,
    ID: AxisId,
    L: Layout,
{
    if current_ids == output_ids {
        return tensor.to_tensor();
    }

    let perm: Vec<usize> = output_ids
        .iter()
        .map(|id| {
            current_ids
                .iter()
                .position(|x| x == id)
                .expect("Output ID not found in current IDs")
        })
        .collect();

    tensor.permute(perm).to_tensor()
}

/// Helper: sum tensor along axis.
fn sum_axis<T>(tensor: &Tensor<T, DynRank>, axis: usize) -> Tensor<T, DynRank>
where
    T: Clone + Zero + std::ops::Add<Output = T>,
{
    let shape: Vec<usize> = (0..tensor.rank()).map(|i| tensor.dim(i)).collect();

    let mut out_shape: Vec<usize> = shape.clone();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_len = out_shape.iter().product();
    let mut out_data = vec![T::zero(); out_len];

    let total_len: usize = shape.iter().product();
    for flat_idx in 0..total_len {
        let multi_idx = flat_to_multi_vec(flat_idx, &shape);

        let mut out_multi = multi_idx.clone();
        out_multi.remove(axis);
        if out_multi.is_empty() {
            out_multi.push(0);
        }

        let out_flat = multi_to_flat(&out_multi, &out_shape);
        out_data[out_flat] = out_data[out_flat].clone() + tensor[&multi_idx[..]].clone();
    }

    Tensor::from(out_data).into_shape(out_shape).into_dyn()
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn flat_to_multi_vec(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    let mut remaining = flat;
    for i in (0..shape.len()).rev() {
        result[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    result
}

fn multi_to_flat(multi: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for i in (0..multi.len()).rev() {
        flat += multi[i] * stride;
        stride *= shape[i];
    }
    flat
}

fn iterate_multi_index<F>(shape: &[usize], mut f: F)
where
    F: FnMut(&[usize]),
{
    if shape.is_empty() {
        f(&[]);
        return;
    }

    let total: usize = shape.iter().product();
    let mut multi_idx = vec![0; shape.len()];

    for _ in 0..total {
        f(&multi_idx);

        for i in (0..shape.len()).rev() {
            multi_idx[i] += 1;
            if multi_idx[i] < shape[i] {
                break;
            }
            multi_idx[i] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use mdarray::tensor;
    use mdarray_linalg::Naive;

    #[test]
    fn test_multi_contract_three_tensors() {
        // ij,jk,jl->ikl with j contracted (hyperedge)
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2
        let c = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].into_dyn(); // 2x3

        let result = multi_contract(
            &Naive,
            &[(&[0u32, 1], &a), (&[1u32, 2], &b), (&[1u32, 3], &c)],
            &1u32,      // Contract over j
            &[0, 2, 3], // Output: i, k, l
        );

        // Manual calculation:
        // result[i,k,l] = sum_j A[i,j] * B[j,k] * C[j,l]
        // result[0,0,0] = A[0,0]*B[0,0]*C[0,0] + A[0,1]*B[1,0]*C[1,0]
        //               = 1*1*5 + 2*3*8 = 5 + 48 = 53
        assert_relative_eq!(result[[0, 0, 0]], 53.0, epsilon = 1e-10);

        // result[0,1,0] = A[0,0]*B[0,1]*C[0,0] + A[0,1]*B[1,1]*C[1,0]
        //               = 1*2*5 + 2*4*8 = 10 + 64 = 74
        assert_relative_eq!(result[[0, 1, 0]], 74.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_contract_four_tensors() {
        // Four tensors sharing index j: ij,jk,jl,jm->iklm
        let a = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2
        let b = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2
        let c = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2
        let d = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2

        let result = multi_contract(
            &Naive,
            &[
                (&[0u32, 1], &a),
                (&[1u32, 2], &b),
                (&[1u32, 3], &c),
                (&[1u32, 4], &d),
            ],
            &1u32,
            &[0, 2, 3, 4],
        );

        // All ones, so result[i,k,l,m] = sum_j 1*1*1*1 = 2 (j has dim 2)
        for i in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    for m in 0..2 {
                        assert_relative_eq!(result[[i, k, l, m]], 2.0, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_multi_contract_general_two_hyperedges() {
        // Two hyperedges: j appears in A,B,C and k appears in B,C,D
        // ij,jkl,jk,km->ilm
        let a = tensor![[1.0, 1.0]].into_dyn(); // 1x2 (i,j)
        let b = tensor![[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]].into_dyn(); // 2x2x2 (j,k,l)
        let c = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2 (j,k)
        let d = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2 (k,m)

        let result = multi_contract_general(
            &Naive,
            &[
                (&[0u32, 1], &a),
                (&[1u32, 2, 3], &b),
                (&[1u32, 2], &c),
                (&[2u32, 4], &d),
            ],
            &[1, 2],    // Contract j and k
            &[0, 3, 4], // Output: i, l, m
        );

        // result[i,l,m] = sum_{j,k} A[i,j] * B[j,k,l] * C[j,k] * D[k,m]
        // With all ones: sum_{j,k} 1 = 2*2 = 4
        assert_relative_eq!(result[[0, 0, 0]], 4.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1, 1]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_contract_matches_einsum() {
        // Verify multi_contract gives same result as naive einsum
        // ij,jk,jl->ikl
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let c = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].into_dyn();

        let result = multi_contract(
            &Naive,
            &[(&[0u32, 1], &a), (&[1u32, 2], &b), (&[1u32, 3], &c)],
            &1u32,
            &[0, 2, 3],
        );

        // Compute expected result manually
        let mut expected = vec![0.0; 2 * 2 * 3];
        for i in 0..2 {
            for k in 0..2 {
                for l in 0..3 {
                    let mut sum = 0.0;
                    for j in 0..2 {
                        sum += a[[i, j]] * b[[j, k]] * c[[j, l]];
                    }
                    expected[i * 6 + k * 3 + l] = sum;
                }
            }
        }

        for i in 0..2 {
            for k in 0..2 {
                for l in 0..3 {
                    assert_relative_eq!(
                        result[[i, k, l]],
                        expected[i * 6 + k * 3 + l],
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    #[test]
    fn test_multi_contract_naive_matches() {
        // Verify naive version gives same result
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let c = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].into_dyn();

        let result_batched = multi_contract(
            &Naive,
            &[(&[0u32, 1], &a), (&[1u32, 2], &b), (&[1u32, 3], &c)],
            &1u32,
            &[0, 2, 3],
        );

        let result_naive = multi_contract_naive(
            &[(&[0u32, 1], &a), (&[1u32, 2], &b), (&[1u32, 3], &c)],
            &1u32,
            &[0, 2, 3],
        );

        for i in 0..2 {
            for k in 0..2 {
                for l in 0..3 {
                    assert_relative_eq!(
                        result_batched[[i, k, l]],
                        result_naive[[i, k, l]],
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    #[test]
    fn test_multi_contract_hyperedge_on_right() {
        // ia,ja,ka->ijk (hyperedge 'a' is on the right side of all tensors)
        // This tests the case where the contracted index is not at position 0
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2 (i,a)
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2 (j,a)
        let c = tensor![[5.0, 6.0], [7.0, 8.0]].into_dyn(); // 2x2 (k,a)

        let result = multi_contract(
            &Naive,
            &[
                (&[0u32, 3], &a), // i=0, a=3
                (&[1u32, 3], &b), // j=1, a=3
                (&[2u32, 3], &c), // k=2, a=3
            ],
            &3u32,      // Contract over a
            &[0, 1, 2], // Output: i, j, k
        );

        // Manual calculation:
        // result[i,j,k] = sum_a A[i,a] * B[j,a] * C[k,a]
        // result[0,0,0] = A[0,0]*B[0,0]*C[0,0] + A[0,1]*B[0,1]*C[0,1]
        //               = 1*1*5 + 2*2*6 = 5 + 24 = 29
        assert_relative_eq!(result[[0, 0, 0]], 29.0, epsilon = 1e-10);

        // result[1,1,1] = A[1,0]*B[1,0]*C[1,0] + A[1,1]*B[1,1]*C[1,1]
        //               = 3*3*7 + 4*4*8 = 63 + 128 = 191
        assert_relative_eq!(result[[1, 1, 1]], 191.0, epsilon = 1e-10);

        // result[0,1,0] = A[0,0]*B[1,0]*C[0,0] + A[0,1]*B[1,1]*C[0,1]
        //               = 1*3*5 + 2*4*6 = 15 + 48 = 63
        assert_relative_eq!(result[[0, 1, 0]], 63.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_contract_hyperedge_mixed_positions() {
        // ai,ja,ak->ijk (hyperedge 'a' at different positions in each tensor)
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2 (a,i)
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2 (j,a)
        let c = tensor![[5.0, 6.0], [7.0, 8.0]].into_dyn(); // 2x2 (a,k)

        let result = multi_contract(
            &Naive,
            &[
                (&[3u32, 0], &a), // a=3, i=0
                (&[1u32, 3], &b), // j=1, a=3
                (&[3u32, 2], &c), // a=3, k=2
            ],
            &3u32,      // Contract over a
            &[0, 1, 2], // Output: i, j, k
        );

        // Manual calculation:
        // result[i,j,k] = sum_a A[a,i] * B[j,a] * C[a,k]
        // result[0,0,0] = A[0,0]*B[0,0]*C[0,0] + A[1,0]*B[0,1]*C[1,0]
        //               = 1*1*5 + 3*2*7 = 5 + 42 = 47
        assert_relative_eq!(result[[0, 0, 0]], 47.0, epsilon = 1e-10);

        // result[1,1,1] = A[0,1]*B[1,0]*C[0,1] + A[1,1]*B[1,1]*C[1,1]
        //               = 2*3*6 + 4*4*8 = 36 + 128 = 164
        assert_relative_eq!(result[[1, 1, 1]], 164.0, epsilon = 1e-10);
    }
}
