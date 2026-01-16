//! Einstein summation (einsum) for mdarray with ID-based axis specification.
//!
//! This crate provides einsum functionality similar to PyTorch's `torch.einsum`,
//! but using generic axis IDs instead of string-based notation.
//!
//! # Example
//! ```rust
//! use mdarray::tensor;
//! use mdarray_einsum::einsum;
//! use mdarray_linalg::Naive;
//!
//! // Matrix multiplication: ij,jk->ik
//! let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
//! let b = tensor![[5.0, 6.0], [7.0, 8.0]].into_dyn();
//!
//! let result = einsum(
//!     &Naive,
//!     &[
//!         (&[0u32, 1][..], &a),  // axes: i=0, j=1
//!         (&[1u32, 2][..], &b),  // axes: j=1, k=2
//!     ],
//!     &[0, 2],  // output: i=0, k=2
//! );
//! // result is a 2x2 tensor with values [[19, 22], [43, 50]]
//! ```

mod error;
pub mod hyperedge_optimizer;
mod multi_contract;
pub mod optimizer;
mod sumproduct_pair;
pub mod typed_tensor;

pub use error::EinsumError;
pub use hyperedge_optimizer::{optimize_hyperedge_greedy, HyperedgePath, HyperedgeStep};
pub use multi_contract::{multi_contract, multi_contract_general};
pub use optimizer::{optimize_greedy, ContractionStep};
pub use typed_tensor::{einsum_typed, einsum_typed_simple, needs_c64_promotion, TypedTensor};

// Re-export omeco's Label trait for convenience
pub use omeco::Label;

// Re-export matmul trait and backends for convenience
pub use mdarray_linalg::matmul::MatMul;
pub use mdarray_linalg::Naive;

use mdarray::{DynRank, Layout, Slice, Tensor};
use num_complex::ComplexFloat;
use num_traits::{MulAdd, One, Zero};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Either a borrowed view or an owned tensor.
/// Used to avoid unnecessary copies during contraction.
enum TensorRef<'a, T, L: Layout> {
    Borrowed(&'a Slice<T, DynRank, L>),
    Owned(Tensor<T, DynRank>),
}

/// Call sumproduct_pair with two TensorRefs, avoiding unnecessary copies when possible.
/// When both are Borrowed (same Layout), no copy is needed.
/// When mixed, Borrowed is converted to Dense.
fn sumproduct_pair_refs<T, ID, L>(
    backend: &impl MatMul<T>,
    left: (&[ID], &TensorRef<'_, T, L>),
    right: (&[ID], &TensorRef<'_, T, L>),
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    let (ids_a, ref_a) = left;
    let (ids_b, ref_b) = right;
    match (ref_a, ref_b) {
        (TensorRef::Borrowed(a), TensorRef::Borrowed(b)) => {
            // Both have same Layout L
            sumproduct_pair::sumproduct_pair(backend, (ids_a, *a), (ids_b, *b), output_ids)
        }
        (TensorRef::Borrowed(a), TensorRef::Owned(b)) => {
            // Mixed: convert Borrowed to Dense
            let a_dense = a.to_tensor();
            sumproduct_pair::sumproduct_pair(
                backend,
                (ids_a, a_dense.as_ref()),
                (ids_b, b.as_ref()),
                output_ids,
            )
        }
        (TensorRef::Owned(a), TensorRef::Borrowed(b)) => {
            // Mixed: convert Borrowed to Dense
            let b_dense = b.to_tensor();
            sumproduct_pair::sumproduct_pair(
                backend,
                (ids_a, a.as_ref()),
                (ids_b, b_dense.as_ref()),
                output_ids,
            )
        }
        (TensorRef::Owned(a), TensorRef::Owned(b)) => {
            // Both Dense
            sumproduct_pair::sumproduct_pair(
                backend,
                (ids_a, a.as_ref()),
                (ids_b, b.as_ref()),
                output_ids,
            )
        }
    }
}

/// Call multi_contract with TensorRefs.
/// For hyperedge contraction (3+ tensors), we need uniform Layout.
/// Borrowed tensors are converted to Dense only when mixed with Owned.
fn multi_contract_refs<T, ID, L>(
    backend: &impl MatMul<T>,
    inputs: &[(Vec<ID>, TensorRef<'_, T, L>)],
    contracted_index: &ID,
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    // Check if all are Borrowed (same Layout L)
    let all_borrowed = inputs
        .iter()
        .all(|(_, r)| matches!(r, TensorRef::Borrowed(_)));

    if all_borrowed {
        // All have same Layout L, can call multi_contract directly
        let refs: Vec<(&[ID], &Slice<T, DynRank, L>)> = inputs
            .iter()
            .map(|(ids, r)| match r {
                TensorRef::Borrowed(s) => (ids.as_slice(), *s),
                TensorRef::Owned(_) => unreachable!(),
            })
            .collect();
        multi_contract::multi_contract(backend, &refs, contracted_index, output_ids)
    } else {
        // Mixed layouts - convert all to Dense
        let owned: Vec<(Vec<ID>, Tensor<T, DynRank>)> = inputs
            .iter()
            .map(|(ids, r)| {
                let t = match r {
                    TensorRef::Borrowed(s) => s.to_tensor(),
                    TensorRef::Owned(t) => t.clone(),
                };
                (ids.clone(), t)
            })
            .collect();
        let refs: Vec<(&[ID], &Slice<T, DynRank, mdarray::Dense>)> = owned
            .iter()
            .map(|(ids, t)| (ids.as_slice(), t.as_ref()))
            .collect();
        multi_contract::multi_contract(backend, &refs, contracted_index, output_ids)
    }
}

/// Axis ID trait - any type that can be used as an axis identifier.
pub trait AxisId: Clone + Eq + Hash + std::fmt::Debug {}
impl<T: Clone + Eq + Hash + std::fmt::Debug> AxisId for T {}

/// Input specification for einsum: axis IDs and tensor reference.
pub type EinsumInput<'a, ID, T, L> = (&'a [ID], &'a Slice<T, DynRank, L>);

/// Perform Einstein summation on multiple tensors.
///
/// # Arguments
/// * `inputs` - Slice of (axis_ids, tensor) pairs
/// * `output_ids` - Axis IDs for the output tensor
///
/// # Returns
/// Result tensor with axes ordered according to `output_ids`
pub fn einsum<T, ID, L>(
    backend: &impl MatMul<T>,
    inputs: &[EinsumInput<ID, T, L>],
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    einsum_with_path(backend, inputs, output_ids, None)
}

/// Perform Einstein summation with optimized contraction order.
///
/// Uses omeco's greedy algorithm to find an efficient contraction path.
///
/// # Arguments
/// * `inputs` - Slice of (axis_ids, tensor) pairs
/// * `output_ids` - Axis IDs for the output tensor
/// * `sizes` - Dimension sizes for each axis ID
///
/// # Returns
/// Result tensor with axes ordered according to `output_ids`
pub fn einsum_optimized<T, ID, L>(
    backend: &impl MatMul<T>,
    inputs: &[EinsumInput<ID, T, L>],
    output_ids: &[ID],
    sizes: &HashMap<ID, usize>,
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId + omeco::Label,
    L: Layout,
{
    use omeco::{CodeOptimizer, EinCode as OmecoEinCode, GreedyMethod};

    let input_ids: Vec<Vec<ID>> = inputs.iter().map(|(ids, _)| ids.to_vec()).collect();

    if inputs.len() <= 1 {
        return einsum(backend, inputs, output_ids);
    }

    // Check for hyperedge contractions (index appears in 3+ tensors but not in output)
    // For these cases, omeco's optimization can give incorrect results.
    let output_set: HashSet<&ID> = output_ids.iter().collect();
    let mut id_counts: HashMap<&ID, usize> = HashMap::new();
    for ids in &input_ids {
        for id in ids {
            *id_counts.entry(id).or_insert(0) += 1;
        }
    }
    let has_contracted_hyperedge = id_counts
        .iter()
        .any(|(id, &count)| count >= 3 && !output_set.contains(*id));

    if has_contracted_hyperedge {
        // Use hyperedge-aware optimizer for correct handling
        return einsum_with_hyperedge_optimizer(backend, inputs, output_ids, sizes);
    }

    let code = OmecoEinCode::new(input_ids.clone(), output_ids.to_vec());
    let optimizer = GreedyMethod::default();

    if let Some(nested) = optimizer.optimize(&code, sizes) {
        // Execute the nested einsum tree directly
        let tensors: Vec<Tensor<T, DynRank>> = inputs.iter().map(|(_, t)| t.to_tensor()).collect();
        let (result_ids, result) =
            execute_nested(backend, &nested, &input_ids, &tensors, output_ids);

        // Check if omeco's output matches user's expected output
        if result_ids == output_ids.to_vec() {
            result
        } else if result_ids.len() == output_ids.len() {
            // Same number of dimensions, just need to permute
            permute_to_output(&result_ids, result, output_ids)
        } else {
            // omeco kept extra dimensions - need to do a final trace/sum
            // This can happen with hyperedge contractions where extra indices
            // need to be summed over
            trace_to_output(&result_ids, result, output_ids)
        }
    } else {
        // Fallback to left-to-right if optimization fails
        einsum(backend, inputs, output_ids)
    }
}

/// Einsum with hyperedge-aware optimizer.
///
/// This uses the hyperedge optimizer which correctly handles indices that appear
/// in 3+ tensors by contracting all sharing tensors simultaneously.
fn einsum_with_hyperedge_optimizer<T, ID, L>(
    backend: &impl MatMul<T>,
    inputs: &[EinsumInput<ID, T, L>],
    output_ids: &[ID],
    sizes: &HashMap<ID, usize>,
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    let input_ids: Vec<Vec<ID>> = inputs.iter().map(|(ids, _)| ids.to_vec()).collect();
    let path = hyperedge_optimizer::optimize_hyperedge_greedy(&input_ids, output_ids, sizes);

    // Execute the path using TensorRef to avoid unnecessary copies of input tensors
    // Borrowed = original input view, Owned = intermediate result
    let mut operands: Vec<(Vec<ID>, TensorRef<'_, T, L>)> = inputs
        .iter()
        .map(|(ids, t)| (ids.to_vec(), TensorRef::Borrowed(*t)))
        .collect();

    for step in &path.steps {
        if step.operand_indices.len() == 2 {
            // Pairwise contraction (use sumproduct_pair)
            let (i, j) = (step.operand_indices[0], step.operand_indices[1]);
            let (i, j) = if i < j { (i, j) } else { (j, i) };

            let (ids_b, tensor_b) = operands.remove(j);
            let (ids_a, tensor_a) = operands.remove(i);

            let result = sumproduct_pair_refs(
                backend,
                (&ids_a, &tensor_a),
                (&ids_b, &tensor_b),
                &step.output_ids,
            );

            operands.push((step.output_ids.clone(), TensorRef::Owned(result)));
        } else {
            // Multi-tensor contraction (hyperedge case)
            // Collect all tensors being contracted
            let mut indices: Vec<usize> = step.operand_indices.clone();
            indices.sort_by(|a, b| b.cmp(a)); // Sort descending for safe removal

            let contracted_tensors: Vec<(Vec<ID>, TensorRef<'_, T, L>)> =
                indices.iter().map(|&i| operands.remove(i)).collect();

            if let Some(contracted_idx) = &step.contracted_index {
                // For multi_contract, we need to build a uniform slice vector.
                // Convert Borrowed to Dense via to_tensor only when needed.
                let result = multi_contract_refs(
                    backend,
                    &contracted_tensors,
                    contracted_idx,
                    &step.output_ids,
                );
                operands.push((step.output_ids.clone(), TensorRef::Owned(result)));
            } else {
                // Pure outer product case - chain pairwise
                // This shouldn't happen for hyperedge case, but handle it
                let (first_ids, first_tensor) = &contracted_tensors[0];
                let mut result_ids = first_ids.clone();
                let mut result: Tensor<T, DynRank> = match first_tensor {
                    TensorRef::Borrowed(s) => s.to_tensor(),
                    TensorRef::Owned(t) => t.clone(),
                };

                for (ids, tensor) in contracted_tensors.iter().skip(1) {
                    let mut new_output: Vec<ID> = result_ids.clone();
                    for id in ids {
                        if !new_output.contains(id) {
                            new_output.push(id.clone());
                        }
                    }
                    let result_ref = TensorRef::Owned(result);
                    result = sumproduct_pair_refs(
                        backend,
                        (&result_ids, &result_ref),
                        (ids, tensor),
                        &new_output,
                    );
                    result_ids = new_output;
                }
                operands.push((step.output_ids.clone(), TensorRef::Owned(result)));
            }
        }
    }

    assert_eq!(operands.len(), 1, "Contraction incomplete");
    let (final_ids, final_tensor) = operands.pop().unwrap();
    let final_owned = match final_tensor {
        TensorRef::Borrowed(s) => s.to_tensor(),
        TensorRef::Owned(t) => t,
    };
    permute_to_output(&final_ids, final_owned, output_ids)
}

/// Execute a NestedEinsum tree recursively.
fn execute_nested<T, ID>(
    backend: &impl MatMul<T>,
    nested: &omeco::NestedEinsum<ID>,
    input_ids: &[Vec<ID>],
    tensors: &[Tensor<T, DynRank>],
    _final_output_ids: &[ID],
) -> (Vec<ID>, Tensor<T, DynRank>)
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId + omeco::Label,
{
    match nested {
        omeco::NestedEinsum::Leaf { tensor_index } => {
            let ids = input_ids[*tensor_index].clone();
            let tensor = tensors[*tensor_index].clone();
            (ids, tensor)
        }
        omeco::NestedEinsum::Node { args, eins } => {
            assert_eq!(args.len(), 2, "Only binary contraction trees supported");

            let (ids_a, tensor_a) =
                execute_nested(backend, &args[0], input_ids, tensors, _final_output_ids);
            let (ids_b, tensor_b) =
                execute_nested(backend, &args[1], input_ids, tensors, _final_output_ids);

            // Always use omeco's computed output for proper intermediate contractions
            let result = sumproduct_pair::sumproduct_pair(
                backend,
                (&ids_a, &tensor_a),
                (&ids_b, &tensor_b),
                &eins.iy,
            );

            (eins.iy.clone(), result)
        }
    }
}

/// Perform Einstein summation with an optional contraction path.
///
/// # Arguments
/// * `inputs` - Slice of (axis_ids, tensor) pairs
/// * `output_ids` - Axis IDs for the output tensor
/// * `path` - Optional contraction path as pairs of operand indices
///
/// # Returns
/// Result tensor with axes ordered according to `output_ids`
pub fn einsum_with_path<T, ID, L>(
    backend: &impl MatMul<T>,
    inputs: &[EinsumInput<ID, T, L>],
    output_ids: &[ID],
    path: Option<&[(usize, usize)]>,
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    assert!(
        !inputs.is_empty(),
        "einsum: must provide at least one operand"
    );

    if inputs.len() == 1 {
        return einsum_single(backend, inputs[0], output_ids);
    }

    // Collect all operands into owned tensors with their axis IDs
    let mut ops: Vec<(Vec<ID>, Tensor<T, DynRank>)> = inputs
        .iter()
        .map(|(ids, tensor)| (ids.to_vec(), tensor.to_tensor()))
        .collect();

    // Build output ID set for determining contraction dimensions
    let output_set: HashSet<&ID> = output_ids.iter().collect();

    // Contract according to path or left-to-right
    let path_iter: Box<dyn Iterator<Item = (usize, usize)>> = if let Some(p) = path {
        Box::new(p.iter().copied())
    } else {
        // Default: contract left to right
        Box::new((0..inputs.len() - 1).map(|_| (0, 1)))
    };

    for (i, j) in path_iter {
        if ops.len() <= 1 {
            break;
        }

        let (i, j) = if i < j { (i, j) } else { (j, i) };

        // Remove operands (j first since j > i)
        let (ids_b, tensor_b) = ops.remove(j);
        let (ids_a, tensor_a) = ops.remove(i);

        // Collect IDs from remaining operands (needed for later contractions)
        let remaining_ids: HashSet<&ID> = ops.iter().flat_map(|(ids, _)| ids.iter()).collect();

        // Compute intermediate output IDs
        let intermediate_output =
            compute_intermediate_output(&ids_a, &ids_b, &output_set, &remaining_ids);

        // Contract the pair
        let result = sumproduct_pair::sumproduct_pair(
            backend,
            (&ids_a, &tensor_a),
            (&ids_b, &tensor_b),
            &intermediate_output,
        );

        // Add result back to operands
        ops.push((intermediate_output, result));
    }

    assert_eq!(ops.len(), 1, "einsum: contraction incomplete");

    let (final_ids, final_tensor) = ops.pop().unwrap();

    // Permute to match output order if necessary
    permute_to_output(&final_ids, final_tensor, output_ids)
}

/// Compute intermediate output IDs after contracting two tensors.
///
/// An ID appears in the intermediate output if:
/// - It appears in the final output, OR
/// - It appears in one of the remaining operands (needed for later contractions)
///
/// IDs shared between both tensors but not needed for output or later contractions are contracted.
/// IDs in only one tensor and not needed are traced (summed within that tensor).
fn compute_intermediate_output<ID: AxisId>(
    ids_a: &[ID],
    ids_b: &[ID],
    final_output: &HashSet<&ID>,
    remaining_ids: &HashSet<&ID>,
) -> Vec<ID> {
    let _set_a: HashSet<&ID> = ids_a.iter().collect();
    let _set_b: HashSet<&ID> = ids_b.iter().collect();

    let mut result = Vec::new();

    // Keep IDs that are in final output OR in remaining operands (for later contractions)
    // But only if they appear in at least one of the current operands
    for id in ids_a {
        let needed = final_output.contains(id) || remaining_ids.contains(id);
        if needed && !result.contains(id) {
            result.push(id.clone());
        }
    }

    for id in ids_b {
        let needed = final_output.contains(id) || remaining_ids.contains(id);
        // For hyperedges: if the ID is shared between a and b and NOT needed,
        // it should be contracted. If only in b and not needed, it should be traced.
        // If needed, keep it (but avoid duplicates from the case where it's in both).
        if needed && !result.contains(id) {
            result.push(id.clone());
        }
    }

    result
}

/// Handle single-tensor einsum (trace, diagonal, permutation).
fn einsum_single<T, ID, L>(
    _backend: &impl MatMul<T>,
    input: EinsumInput<ID, T, L>,
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: ComplexFloat + Zero + One + MulAdd<Output = T> + Default + 'static,
    ID: AxisId,
    L: Layout,
{
    let (ids, tensor) = input;

    // Check for repeated indices (trace/diagonal)
    let mut id_counts: HashMap<&ID, usize> = HashMap::new();
    for id in ids {
        *id_counts.entry(id).or_insert(0) += 1;
    }

    // For now, only handle simple permutation (no repeated indices)
    for (id, count) in &id_counts {
        if *count > 1 {
            panic!("einsum_single: repeated index {:?} not yet supported", id);
        }
    }

    // Simple permutation case
    permute_to_output(ids, tensor.to_tensor(), output_ids)
}

/// Sum over indices that are not in output (trace operation) and permute to match output order.
fn trace_to_output<T, ID>(
    current_ids: &[ID],
    tensor: Tensor<T, DynRank>,
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: Clone + Zero + std::ops::Add<Output = T>,
    ID: AxisId,
{
    // Find indices to sum over (in current but not in output)
    let output_set: HashSet<&ID> = output_ids.iter().collect();
    let sum_axes: Vec<usize> = current_ids
        .iter()
        .enumerate()
        .filter(|(_, id)| !output_set.contains(id))
        .map(|(i, _)| i)
        .collect();

    if sum_axes.is_empty() {
        return permute_to_output(current_ids, tensor, output_ids);
    }

    // Sum over the extra axes (from back to front to preserve indices)
    let mut result = tensor;
    let mut remaining_ids = current_ids.to_vec();
    for &axis in sum_axes.iter().rev() {
        result = sum_axis(&result, axis);
        remaining_ids.remove(axis);
    }

    permute_to_output(&remaining_ids, result, output_ids)
}

/// Sum a tensor along a given axis.
fn sum_axis<T>(tensor: &Tensor<T, DynRank>, axis: usize) -> Tensor<T, DynRank>
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
        let mut out_idx = 0;
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            if i != axis {
                out_idx += multi_idx[i] * stride;
                stride *= shape[i];
            }
        }

        out_data[out_idx] = out_data[out_idx].clone() + tensor[&multi_idx[..]].clone();
    }

    Tensor::from(out_data).into_shape(out_shape).into_dyn()
}

/// Permute tensor axes to match output order.
fn permute_to_output<T, ID>(
    current_ids: &[ID],
    tensor: Tensor<T, DynRank>,
    output_ids: &[ID],
) -> Tensor<T, DynRank>
where
    T: Clone,
    ID: AxisId,
{
    // Handle empty output (scalar result)
    if output_ids.is_empty() {
        return tensor;
    }

    if current_ids == output_ids {
        return tensor;
    }

    // Verify lengths match
    assert_eq!(
        current_ids.len(),
        output_ids.len(),
        "permute_to_output: current_ids {:?} and output_ids {:?} have different lengths",
        current_ids,
        output_ids
    );

    // Build permutation
    let perm: Vec<usize> = output_ids
        .iter()
        .map(|id| {
            current_ids.iter().position(|x| x == id).unwrap_or_else(|| {
                panic!(
                    "Output ID {:?} not found in result (current: {:?})",
                    id, current_ids
                )
            })
        })
        .collect();

    tensor.as_ref().permute(perm).to_tensor()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use mdarray::tensor;
    use mdarray_linalg::Naive;

    #[test]
    fn test_matrix_multiplication() {
        // ij,jk->ik
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = tensor![[5.0, 6.0], [7.0, 8.0]].into_dyn();

        let result = einsum(&Naive, &[(&[0u32, 1], &a), (&[1u32, 2], &b)], &[0, 2]);

        // Expected: [[19, 22], [43, 50]]
        let expected = tensor![[19.0, 22.0], [43.0, 50.0]].into_dyn();

        assert_eq!(result.shape(), expected.shape());
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_inner_product() {
        // i,i-> (dot product)
        let a = tensor![1.0, 2.0, 3.0].into_dyn();
        let b = tensor![4.0, 5.0, 6.0].into_dyn();

        let result = einsum(&Naive, &[(&[0u32], &a), (&[0u32], &b)], &[]);

        // Expected: 1*4 + 2*5 + 3*6 = 32
        // Result is a scalar wrapped in a tensor; current impl returns [1,1,1] shape
        // We check the single element
        let total: f64 = result.iter().copied().sum();
        assert_relative_eq!(total, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_outer_product() {
        // i,j->ij
        let a = tensor![1.0, 2.0].into_dyn();
        let b = tensor![3.0, 4.0, 5.0].into_dyn();

        let result = einsum(&Naive, &[(&[0u32], &a), (&[1u32], &b)], &[0, 1]);

        // Expected: [[3, 4, 5], [6, 8, 10]]
        let expected = tensor![[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]].into_dyn();

        assert_eq!(result.shape(), expected.shape());
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_batch_matmul() {
        // bij,bjk->bik (batched matrix multiplication)
        let a = tensor![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]].into_dyn();
        let b = tensor![[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]].into_dyn();

        let result = einsum(
            &Naive,
            &[(&[0u32, 1, 2], &a), (&[0u32, 2, 3], &b)],
            &[0, 1, 3],
        );

        // Batch 0: [[1,2],[3,4]] @ I = [[1,2],[3,4]]
        // Batch 1: [[5,6],[7,8]] @ 2I = [[10,12],[14,16]]
        assert_eq!(result.rank(), 3);
        assert_eq!(result.dim(0), 2);
        assert_eq!(result.dim(1), 2);
        assert_eq!(result.dim(2), 2);
        assert_relative_eq!(result[[0, 0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 0, 1]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0, 0]], 10.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 1, 1]], 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_tensor_contraction() {
        // ij,jk,kl->il
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = tensor![[1.0, 0.0], [0.0, 1.0]].into_dyn(); // Identity
        let c = tensor![[5.0, 6.0], [7.0, 8.0]].into_dyn();

        let result = einsum(
            &Naive,
            &[(&[0u32, 1], &a), (&[1u32, 2], &b), (&[2u32, 3], &c)],
            &[0, 3],
        );

        // A @ I @ C = A @ C = [[19,22],[43,50]]
        let expected = tensor![[19.0, 22.0], [43.0, 50.0]].into_dyn();

        assert_eq!(result.shape(), expected.shape());
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_hyperedge_output() {
        // ij,jk,jl->ijkl (j appears in 3 tensors and output)
        let a = tensor![[1.0, 0.0], [0.0, 1.0]].into_dyn(); // 2x2 identity
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2
        let c = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].into_dyn(); // 2x3

        let result = einsum(
            &Naive,
            &[
                (&[0u32, 1], &a), // i,j
                (&[1u32, 2], &b), // j,k
                (&[1u32, 3], &c), // j,l
            ],
            &[0, 1, 2, 3], // i,j,k,l
        );

        // Result should have shape [2, 2, 2, 3]
        assert_eq!(result.rank(), 4);
        assert_eq!(result.dim(0), 2);
        assert_eq!(result.dim(1), 2);
        assert_eq!(result.dim(2), 2);
        assert_eq!(result.dim(3), 3);

        // Since A is identity:
        // result[i,j,k,l] = A[i,j] * B[j,k] * C[j,l]
        // result[0,0,k,l] = 1 * B[0,k] * C[0,l]
        // result[1,1,k,l] = 1 * B[1,k] * C[1,l]
        // result[0,1,k,l] = 0
        // result[1,0,k,l] = 0

        assert_relative_eq!(result[[0, 0, 0, 0]], 1.0 * 5.0, epsilon = 1e-10); // B[0,0]*C[0,0]
        assert_relative_eq!(result[[0, 0, 1, 2]], 2.0 * 7.0, epsilon = 1e-10); // B[0,1]*C[0,2]
        assert_relative_eq!(result[[1, 1, 0, 0]], 3.0 * 8.0, epsilon = 1e-10); // B[1,0]*C[1,0]
        assert_relative_eq!(result[[0, 1, 0, 0]], 0.0, epsilon = 1e-10); // A[0,1]=0
        assert_relative_eq!(result[[1, 0, 0, 0]], 0.0, epsilon = 1e-10); // A[1,0]=0
    }

    #[test]
    fn test_einsum_optimized_chain() {
        // ij,jk,kl->il with optimized contraction order
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let b = tensor![[1.0, 0.0], [0.0, 1.0]].into_dyn(); // Identity
        let c = tensor![[5.0, 6.0], [7.0, 8.0]].into_dyn();

        let sizes: HashMap<u32, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into_iter().collect();

        let result = einsum_optimized(
            &Naive,
            &[(&[0u32, 1], &a), (&[1u32, 2], &b), (&[2u32, 3], &c)],
            &[0, 3],
            &sizes,
        );

        // A @ I @ C = A @ C = [[19,22],[43,50]]
        let expected = tensor![[19.0, 22.0], [43.0, 50.0]].into_dyn();

        assert_eq!(result.shape(), expected.shape());
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_einsum_optimized_hyperedge() {
        // ij,jk,jl->ijkl (j appears in 3 tensors and output - hyperedge case)
        let a = tensor![[1.0, 0.0], [0.0, 1.0]].into_dyn(); // 2x2 identity
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2
        let c = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].into_dyn(); // 2x3

        let sizes: HashMap<u32, usize> = [(0, 2), (1, 2), (2, 2), (3, 3)].into_iter().collect();

        let result = einsum_optimized(
            &Naive,
            &[
                (&[0u32, 1], &a), // i,j
                (&[1u32, 2], &b), // j,k
                (&[1u32, 3], &c), // j,l
            ],
            &[0, 1, 2, 3], // i,j,k,l
            &sizes,
        );

        // Compare with non-optimized version
        let expected = einsum(
            &Naive,
            &[
                (&[0u32, 1][..], &a),
                (&[1u32, 2][..], &b),
                (&[1u32, 3][..], &c),
            ],
            &[0, 1, 2, 3],
        );

        // Result should have shape [2, 2, 2, 3]
        assert_eq!(result.rank(), 4);
        assert_eq!(result.dim(0), 2);
        assert_eq!(result.dim(1), 2);
        assert_eq!(result.dim(2), 2);
        assert_eq!(result.dim(3), 3);

        // Check all elements match the non-optimized version
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..3 {
                        assert_relative_eq!(
                            result[[i, j, k, l]],
                            expected[[i, j, k, l]],
                            epsilon = 1e-10
                        );
                    }
                }
            }
        }

        // Also verify specific values
        // result[i,j,k,l] = A[i,j] * B[j,k] * C[j,l]
        // Since A is identity, result[0,0,k,l] = B[0,k]*C[0,l], result[1,1,k,l] = B[1,k]*C[1,l]
        assert_relative_eq!(result[[0, 0, 0, 0]], 1.0 * 5.0, epsilon = 1e-10); // B[0,0]*C[0,0]
        assert_relative_eq!(result[[0, 0, 1, 2]], 2.0 * 7.0, epsilon = 1e-10); // B[0,1]*C[0,2]
        assert_relative_eq!(result[[1, 1, 0, 0]], 3.0 * 8.0, epsilon = 1e-10); // B[1,0]*C[1,0]
        assert_relative_eq!(result[[0, 1, 0, 0]], 0.0, epsilon = 1e-10); // A[0,1]=0
        assert_relative_eq!(result[[1, 0, 0, 0]], 0.0, epsilon = 1e-10); // A[1,0]=0
    }

    #[test]
    fn test_einsum_optimized_hyperedge_contracted() {
        // ij,jk,jl->ikl (j appears in 3 tensors but is contracted - hyperedge)
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2
        let c = tensor![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]].into_dyn(); // 2x3

        let sizes: HashMap<u32, usize> = [(0, 2), (1, 2), (2, 2), (3, 3)].into_iter().collect();

        let result = einsum_optimized(
            &Naive,
            &[
                (&[0u32, 1], &a), // i,j
                (&[1u32, 2], &b), // j,k
                (&[1u32, 3], &c), // j,l
            ],
            &[0, 2, 3], // i,k,l (j is contracted)
            &sizes,
        );

        // Compare with non-optimized version
        let expected = einsum(
            &Naive,
            &[
                (&[0u32, 1][..], &a),
                (&[1u32, 2][..], &b),
                (&[1u32, 3][..], &c),
            ],
            &[0, 2, 3],
        );

        // Result should have shape [2, 2, 3]
        assert_eq!(result.rank(), 3);
        assert_eq!(result.dim(0), 2);
        assert_eq!(result.dim(1), 2);
        assert_eq!(result.dim(2), 3);

        // Check all elements match
        for i in 0..2 {
            for k in 0..2 {
                for l in 0..3 {
                    assert_relative_eq!(result[[i, k, l]], expected[[i, k, l]], epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_einsum_optimized_two_hyperedges() {
        // Two hyperedges: j (in tensors 0,1,2) and k (in tensors 1,3,4)
        // ij,jk,jl,km,kn->ilmn
        // j is hyperedge contracted, k is hyperedge contracted
        let a = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2 (i,j)
        let b = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2 (j,k)
        let c = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2 (j,l)
        let d = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2 (k,m)
        let e = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2 (k,n)

        let sizes: HashMap<u32, usize> = [
            (0, 2), // i
            (1, 2), // j
            (2, 2), // k
            (3, 2), // l
            (4, 2), // m
            (5, 2), // n
        ]
        .into_iter()
        .collect();

        let result = einsum_optimized(
            &Naive,
            &[
                (&[0u32, 1], &a), // i,j
                (&[1u32, 2], &b), // j,k
                (&[1u32, 3], &c), // j,l
                (&[2u32, 4], &d), // k,m
                (&[2u32, 5], &e), // k,n
            ],
            &[0, 3, 4, 5], // i,l,m,n (j and k contracted)
            &sizes,
        );

        // Compare with non-optimized version
        let expected = einsum(
            &Naive,
            &[
                (&[0u32, 1][..], &a),
                (&[1u32, 2][..], &b),
                (&[1u32, 3][..], &c),
                (&[2u32, 4][..], &d),
                (&[2u32, 5][..], &e),
            ],
            &[0, 3, 4, 5],
        );

        // Result should have shape [2, 2, 2, 2]
        assert_eq!(result.rank(), 4);
        assert_eq!(result.dim(0), 2);
        assert_eq!(result.dim(1), 2);
        assert_eq!(result.dim(2), 2);
        assert_eq!(result.dim(3), 2);

        // Check all elements match
        for i in 0..2 {
            for l in 0..2 {
                for m in 0..2 {
                    for n in 0..2 {
                        assert_relative_eq!(
                            result[[i, l, m, n]],
                            expected[[i, l, m, n]],
                            epsilon = 1e-10
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_einsum_optimized_four_tensor_hyperedge() {
        // Four tensors sharing index j
        // ij,jk,jl,jm->iklm (j contracted)
        let a = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2 (i,j)
        let b = tensor![[1.0, 2.0], [3.0, 4.0]].into_dyn(); // 2x2 (j,k)
        let c = tensor![[5.0, 6.0], [7.0, 8.0]].into_dyn(); // 2x2 (j,l)
        let d = tensor![[1.0, 1.0], [1.0, 1.0]].into_dyn(); // 2x2 (j,m)

        let sizes: HashMap<u32, usize> = [
            (0, 2), // i
            (1, 2), // j
            (2, 2), // k
            (3, 2), // l
            (4, 2), // m
        ]
        .into_iter()
        .collect();

        let result = einsum_optimized(
            &Naive,
            &[
                (&[0u32, 1], &a), // i,j
                (&[1u32, 2], &b), // j,k
                (&[1u32, 3], &c), // j,l
                (&[1u32, 4], &d), // j,m
            ],
            &[0, 2, 3, 4], // i,k,l,m (j contracted)
            &sizes,
        );

        // Compare with non-optimized version
        let expected = einsum(
            &Naive,
            &[
                (&[0u32, 1][..], &a),
                (&[1u32, 2][..], &b),
                (&[1u32, 3][..], &c),
                (&[1u32, 4][..], &d),
            ],
            &[0, 2, 3, 4],
        );

        // Result should have shape [2, 2, 2, 2]
        assert_eq!(result.rank(), 4);

        // Check all elements match
        for i in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    for m in 0..2 {
                        assert_relative_eq!(
                            result[[i, k, l, m]],
                            expected[[i, k, l, m]],
                            epsilon = 1e-10
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod scalar_output_tests {
    use super::*;
    use mdarray::{DynRank, Tensor};
    use std::collections::HashMap;

    #[test]
    fn test_einsum_scalar_output() {
        // Two 1D tensors with the same index - should produce scalar
        let a: Tensor<f64, DynRank> = Tensor::from(vec![1.0, 2.0])
            .into_shape([2].as_slice())
            .into_dyn();
        let b: Tensor<f64, DynRank> = Tensor::from(vec![3.0, 4.0])
            .into_shape([2].as_slice())
            .into_dyn();

        let mut sizes = HashMap::new();
        sizes.insert(0_usize, 2);

        // Contract i,i -> (scalar)
        let result = einsum_optimized(
            &Naive,
            &[(&[0_usize][..], a.as_ref()), (&[0_usize][..], b.as_ref())],
            &[], // Empty output = scalar
            &sizes,
        );

        println!("Result shape: {:?}", result.shape().dims());
        println!("Result rank: {:?}", result.rank());

        // Scalar should have empty dims or [1]
        // Expected: 1*3 + 2*4 = 11
        assert!(result.rank() == 0 || (result.rank() == 1 && result.dim(0) == 1));
    }
}

#[cfg(test)]
mod omeco_hyperedge_tests {
    use super::*;
    use approx::assert_relative_eq;
    use mdarray::tensor;
    use omeco::{CodeOptimizer, EinCode as OmecoEinCode, GreedyMethod};
    use std::collections::HashMap;

    /// Test case from omeco author: ixs = [[1, 2], [2], [2, 3]], out = [1, 3]
    /// This is: A[i,j], B[j], C[j,k] -> R[i,k] where j is a hyperedge (in 3 tensors)
    ///
    /// This test demonstrates that omeco's optimizer produces incorrect intermediate
    /// outputs for hyperedge contractions. The first contraction contracts j too early.
    #[test]
    fn test_omeco_hyperedge_optimization_shows_bug() {
        // Check that omeco's optimizer handles this case
        let ixs = vec![
            vec![1usize, 2], // tensor 0: indices i, j
            vec![2usize],    // tensor 1: index j only
            vec![2usize, 3], // tensor 2: indices j, k
        ];
        let out = vec![1usize, 3]; // output: i, k

        let code = OmecoEinCode::new(ixs.clone(), out.clone());

        let mut sizes = HashMap::new();
        sizes.insert(1usize, 2); // i
        sizes.insert(2usize, 3); // j
        sizes.insert(3usize, 2); // k

        let optimizer = GreedyMethod::default();

        // Optimization succeeds (returns Some)
        let nested = optimizer.optimize(&code, &sizes);
        assert!(nested.is_some(), "omeco optimization should succeed");

        let nested = nested.unwrap();
        println!("NestedEinsum from omeco: {:#?}", nested);

        // The tree is binary
        assert!(nested.is_binary(), "Result should be a binary tree");
        assert_eq!(nested.leaf_count(), 3, "Should have 3 leaves");

        // BUG DEMONSTRATION:
        // omeco produces: ((A[i,j] * B[j]) -> [i]) * C[j,k] -> [j,k,i]
        // The first contraction has iy = [1] meaning j is contracted away!
        // But j should be preserved because C[j,k] still needs it.
        //
        // Expected correct behavior would be:
        // ((A[i,j] * B[j]) -> [i,j]) * C[j,k] -> [i,k]
        //
        // This is why we use hyperedge_optimizer instead of omeco for these cases.
    }

    /// Test actual computation with hyperedge case using our einsum
    /// A[i,j] * B[j,k] * C[j,l] -> R[i,k,l] = sum_j (A[i,j] * B[j,k] * C[j,l])
    ///
    /// This uses einsum_optimized which correctly detects the hyperedge and
    /// uses hyperedge_optimizer instead of omeco.
    #[test]
    fn test_hyperedge_2d_tensors() {
        // A is 2x3 (i=2, j=3)
        let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        // B is 3x2 (j=3, k=2)
        let b = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
        // C is 3x2 (j=3, l=2)
        let c = tensor![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].into_dyn();

        let sizes: HashMap<usize, usize> = [(1, 2), (2, 3), (3, 2), (4, 2)].into_iter().collect();

        // Use einsum_optimized (which will use hyperedge_optimizer for this case)
        let result = einsum_optimized(
            &Naive,
            &[
                (&[1usize, 2][..], a.as_ref()), // i, j
                (&[2usize, 3][..], b.as_ref()), // j, k
                (&[2usize, 4][..], c.as_ref()), // j, l
            ],
            &[1, 3, 4], // output: i, k, l
            &sizes,
        );

        // Expected: R[i,k,l] = sum_j (A[i,j] * B[j,k] * C[j,l])
        // R[0,0,0] = A[0,0]*B[0,0]*C[0,0] + A[0,1]*B[1,0]*C[1,0] + A[0,2]*B[2,0]*C[2,0]
        //          = 1*1*10 + 2*3*30 + 3*5*50 = 10 + 180 + 750 = 940

        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        assert_relative_eq!(result[[0, 0, 0]], 940.0, epsilon = 1e-10);
    }

    /// Compare optimized vs non-optimized for hyperedge case
    #[test]
    fn test_hyperedge_optimized_vs_naive() {
        let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let b = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
        let c = tensor![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]].into_dyn();

        let sizes: HashMap<usize, usize> = [(1, 2), (2, 3), (3, 2), (4, 2)].into_iter().collect();

        // Non-optimized (left-to-right)
        let naive_result = einsum(
            &Naive,
            &[
                (&[1usize, 2][..], a.as_ref()),
                (&[2usize, 3][..], b.as_ref()),
                (&[2usize, 4][..], c.as_ref()),
            ],
            &[1, 3, 4],
        );

        // Optimized (uses hyperedge_optimizer for this case)
        let opt_result = einsum_optimized(
            &Naive,
            &[
                (&[1usize, 2][..], a.as_ref()),
                (&[2usize, 3][..], b.as_ref()),
                (&[2usize, 4][..], c.as_ref()),
            ],
            &[1, 3, 4],
            &sizes,
        );

        // Both should give the same result
        assert_eq!(naive_result.shape().dims(), opt_result.shape().dims());
        for i in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    assert_relative_eq!(
                        naive_result[[i, k, l]],
                        opt_result[[i, k, l]],
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    /// Test the standard 2D hyperedge case: A[i,j], B[j,k], C[j,l] -> R[i,k,l]
    /// This is the classic example where j appears in 3 tensors.
    #[test]
    fn test_standard_hyperedge_case() {
        // A is 2x3 (i=2, j=3)
        let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        // B is 3x2 (j=3, k=2)
        let b = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
        // C is 3x2 (j=3, l=2)
        let c = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 2), (3, 2)].into_iter().collect();

        // Use einsum_optimized
        let result = einsum_optimized(
            &Naive,
            &[
                (&[0usize, 1][..], a.as_ref()), // i, j
                (&[1usize, 2][..], b.as_ref()), // j, k
                (&[1usize, 3][..], c.as_ref()), // j, l
            ],
            &[0, 2, 3], // output: i, k, l
            &sizes,
        );

        // Expected: R[i,k,l] = sum_j (A[i,j] * B[j,k] * C[j,l])
        assert_eq!(result.shape().dims(), &[2, 2, 2]);

        // R[0,0,0] = sum_j A[0,j]*B[j,0]*C[j,0]
        //          = 1*1*1 + 2*3*3 + 3*5*5 = 1 + 18 + 75 = 94
        assert_relative_eq!(result[[0, 0, 0]], 94.0, epsilon = 1e-10);
    }
}
