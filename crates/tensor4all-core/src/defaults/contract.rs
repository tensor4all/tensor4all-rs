//! Multi-tensor contraction with optimal contraction order.
//!
//! This module provides functions to contract multiple tensors efficiently
//! by using omeco's GreedyMethod to find the optimal contraction order.
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.

use std::collections::HashMap;

use anyhow::Result;
use omeco::{optimize_code, EinCode, GreedyMethod, NestedEinsum};

use crate::index_like::IndexLike;
use crate::defaults::TensorDynLen;

/// Contract multiple tensors into a single tensor.
///
/// Uses omeco's GreedyMethod to find the optimal contraction order for N>=3 tensors.
///
/// # Arguments
/// * `tensors` - Slice of tensors to contract
///
/// # Returns
/// The result of contracting all tensors over common indices
/// (indices appearing in exactly two tensors are contracted).
///
/// # Behavior by N
/// - N=0: Error
/// - N=1: Clone of input
/// - N=2: Direct tensordot
/// - N>=3: Optimal order via omeco's GreedyMethod
///
/// # Example
/// ```ignore
/// use tensor4all_core::contract_multi;
///
/// let tensors = vec![tensor_a, tensor_b, tensor_c];
/// let result = contract_multi(&tensors)?;
/// ```
pub fn contract_multi(tensors: &[TensorDynLen]) -> Result<TensorDynLen> {
    match tensors.len() {
        0 => Err(anyhow::anyhow!("No tensors to contract")),
        1 => Ok(tensors[0].clone()),
        2 => contract_pair(&tensors[0], &tensors[1]),
        _ => contract_multi_optimized(tensors),
    }
}

/// Contract two tensors over their common indices.
/// If there are no common indices, performs outer product.
fn contract_pair(a: &TensorDynLen, b: &TensorDynLen) -> Result<TensorDynLen> {
    let common: Vec<_> = a
        .indices
        .iter()
        .filter_map(|idx_a| {
            b.indices
                .iter()
                .find(|idx_b| idx_a.id() == idx_b.id())
                .map(|idx_b| (idx_a.clone(), idx_b.clone()))
        })
        .collect();

    if common.is_empty() {
        // No common indices - perform outer product
        a.outer_product(b)
    } else {
        a.tensordot(b, &common)
    }
}

/// Contract multiple tensors using omeco's GreedyMethod for optimal ordering.
fn contract_multi_optimized(tensors: &[TensorDynLen]) -> Result<TensorDynLen> {
    use crate::defaults::DynId;

    // Build mapping from DynId -> usize for omeco
    let mut id_to_idx: HashMap<DynId, usize> = HashMap::new();
    let mut next_idx = 0usize;

    for tensor in tensors {
        for idx in &tensor.indices {
            id_to_idx
                .entry(idx.id().clone())
                .or_insert_with(|| {
                    let i = next_idx;
                    next_idx += 1;
                    i
                });
        }
    }

    // Build EinCode with usize labels
    let ixs: Vec<Vec<usize>> = tensors
        .iter()
        .map(|t| t.indices.iter().map(|idx| id_to_idx[idx.id()]).collect())
        .collect();

    // Determine output indices (appear exactly once across all tensors)
    // Keep them sorted by their first occurrence in the input tensors
    // (which is their assigned usize value in id_to_idx)
    let mut idx_count: HashMap<usize, usize> = HashMap::new();
    for ix in &ixs {
        for &i in ix {
            *idx_count.entry(i).or_insert(0) += 1;
        }
    }
    let mut output: Vec<usize> = idx_count
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(&idx, _)| idx)
        .collect();
    // Sort to maintain deterministic ordering based on first occurrence
    output.sort();

    // Build size dictionary
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for tensor in tensors {
        for (idx, &dim) in tensor.indices.iter().zip(tensor.dims.iter()) {
            let i = id_to_idx[idx.id()];
            sizes.entry(i).or_insert(dim);
        }
    }

    // Optimize contraction order using GreedyMethod
    let code = EinCode::new(ixs, output);
    let tree = optimize_code(&code, &sizes, &GreedyMethod::default())
        .ok_or_else(|| anyhow::anyhow!("Failed to optimize contraction order"))?;

    // Execute the contraction tree
    execute_contraction_tree(tensors, &tree)
}

/// Execute a contraction tree by recursively contracting tensors.
fn execute_contraction_tree(
    tensors: &[TensorDynLen],
    tree: &NestedEinsum<usize>,
) -> Result<TensorDynLen> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => Ok(tensors[*tensor_index].clone()),
        NestedEinsum::Node { args, .. } => {
            // Recursively evaluate children
            let children: Vec<TensorDynLen> = args
                .iter()
                .map(|arg| execute_contraction_tree(tensors, arg))
                .collect::<Result<_>>()?;

            // Contract all children (typically 2 for binary tree)
            let mut result = children[0].clone();
            for child in children.iter().skip(1) {
                result = contract_pair(&result, child)?;
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::{DynId, DynIndex, Index, NoSymmSpace};
    use crate::storage::{DenseStorageC64, Storage};
    use num_complex::Complex64;
    use std::sync::Arc;

    fn make_test_tensor(shape: &[usize], ids: &[u128]) -> TensorDynLen {
        let indices: Vec<DynIndex> = ids
            .iter()
            .zip(shape.iter())
            .map(|(&id, &dim)| Index::new(DynId(id), NoSymmSpace::new(dim)))
            .collect();
        let dims = shape.to_vec();
        let total_size: usize = shape.iter().product();
        let data: Vec<Complex64> = (0..total_size)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect();
        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
        TensorDynLen::new(indices, dims, storage)
    }

    #[test]
    fn test_contract_multi_empty() {
        let tensors: Vec<TensorDynLen> = vec![];
        let result = contract_multi(&tensors);
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_multi_single() {
        let tensor = make_test_tensor(&[2, 3], &[1, 2]);
        let tensors = vec![tensor.clone()];
        let result = contract_multi(&tensors).unwrap();
        assert_eq!(result.dims, tensor.dims);
    }

    #[test]
    fn test_contract_multi_pair() {
        // A[i,j] * B[j,k] -> C[i,k]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let tensors = vec![a, b];
        let result = contract_multi(&tensors).unwrap();
        assert_eq!(result.dims, vec![2, 4]); // i, k
    }

    #[test]
    fn test_contract_multi_three() {
        // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
        let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
        let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
        let c = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4
        let tensors = vec![a, b, c];
        let result = contract_multi(&tensors).unwrap();
        // Output has dimensions for i and l (order may vary due to tree structure)
        let mut sorted_dims = result.dims.clone();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 5]); // i=2, l=5
    }

    #[test]
    fn test_contract_multi_four() {
        // A[i,j] * B[j,k] * C[k,l] * D[l,m] -> E[i,m]
        let a = make_test_tensor(&[2, 3], &[1, 2]);
        let b = make_test_tensor(&[3, 4], &[2, 3]);
        let c = make_test_tensor(&[4, 5], &[3, 4]);
        let d = make_test_tensor(&[5, 6], &[4, 5]);
        let tensors = vec![a, b, c, d];
        let result = contract_multi(&tensors).unwrap();
        // Output has dimensions for i and m (order may vary due to tree structure)
        let mut sorted_dims = result.dims.clone();
        sorted_dims.sort();
        assert_eq!(sorted_dims, vec![2, 6]); // i=2, m=6
    }

    #[test]
    fn test_contract_multi_no_contraction() {
        // A[i,j] * B[k,l] -> C[i,j,k,l] (no common indices)
        let a = make_test_tensor(&[2, 3], &[1, 2]);
        let b = make_test_tensor(&[4, 5], &[3, 4]);
        let tensors = vec![a, b];
        let result = contract_multi(&tensors).unwrap();
        // Total elements should be 2*3*4*5 = 120
        let total_elements: usize = result.dims.iter().product();
        assert_eq!(total_elements, 2 * 3 * 4 * 5);
        assert_eq!(result.dims.len(), 4);
    }
}
