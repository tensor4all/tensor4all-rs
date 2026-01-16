//! Integration with omeco for contraction path optimization.
//!
//! This module provides functions to optimize contraction order using
//! omeco's greedy and TreeSA algorithms.

use crate::AxisId;
use omeco::{CodeOptimizer, EinCode as OmecoEinCode, GreedyMethod, Label, NestedEinsum};
use std::collections::HashMap;

/// A contraction step specifying which two operands to contract.
#[derive(Debug, Clone)]
pub struct ContractionStep<ID> {
    /// Index of first operand to contract
    pub left: usize,
    /// Index of second operand to contract
    pub right: usize,
    /// Output axis IDs for the intermediate result
    pub output_ids: Vec<ID>,
}

/// Convert a NestedEinsum tree to a sequence of contraction steps.
///
/// The steps are returned in execution order (post-order traversal).
fn nested_to_steps<L: Label + Clone>(
    nested: &NestedEinsum<L>,
    _input_labels: &[Vec<L>],
    current_idx: &mut usize,
    steps: &mut Vec<ContractionStep<L>>,
    tensor_map: &mut HashMap<usize, usize>, // original tensor index -> current operand index
) -> usize {
    match nested {
        NestedEinsum::Leaf { tensor_index } => {
            let idx = *current_idx;
            tensor_map.insert(*tensor_index, idx);
            *current_idx += 1;
            idx
        }
        NestedEinsum::Node { args, eins } => {
            // Process children first
            let mut child_indices = Vec::with_capacity(args.len());
            for arg in args {
                let idx = nested_to_steps(arg, _input_labels, current_idx, steps, tensor_map);
                child_indices.push(idx);
            }

            // For binary contractions
            if args.len() == 2 {
                let left = child_indices[0];
                let right = child_indices[1];

                // The output of this contraction becomes a new operand
                let result_idx = *current_idx;
                *current_idx += 1;

                steps.push(ContractionStep {
                    left,
                    right,
                    output_ids: eins.iy.clone(),
                });

                result_idx
            } else {
                // For non-binary nodes, we'd need to handle differently
                // For now, assume the optimizer always produces binary trees
                panic!("Non-binary contraction node not supported");
            }
        }
    }
}

/// Optimize contraction order using omeco's greedy algorithm.
///
/// # Arguments
/// * `input_ids` - Axis IDs for each input tensor
/// * `output_ids` - Axis IDs for the output
/// * `sizes` - Dimension sizes for each axis ID
///
/// # Returns
/// A vector of contraction steps in execution order, or None if optimization fails.
pub fn optimize_greedy<ID: AxisId + Label>(
    input_ids: &[Vec<ID>],
    output_ids: &[ID],
    sizes: &HashMap<ID, usize>,
) -> Option<Vec<ContractionStep<ID>>> {
    if input_ids.len() <= 1 {
        return Some(vec![]);
    }

    let code = OmecoEinCode::new(input_ids.to_vec(), output_ids.to_vec());

    let optimizer = GreedyMethod::default();
    let nested = optimizer.optimize(&code, sizes)?;

    // Convert NestedEinsum to contraction steps
    let mut steps = Vec::new();
    let mut current_idx = 0;
    let mut tensor_map = HashMap::new();

    nested_to_steps(
        &nested,
        input_ids,
        &mut current_idx,
        &mut steps,
        &mut tensor_map,
    );

    Some(steps)
}

/// Convert contraction steps to a flat path format (pairs of indices).
///
/// The path format matches PyTorch's einsum path convention:
/// - For n operands, the path has (n-1) pairs
/// - Each pair (i, j) specifies which operands to contract
/// - After each contraction, the result is appended to the operand list
///   and the contracted operands are removed
pub fn steps_to_path<ID>(steps: &[ContractionStep<ID>]) -> Vec<(usize, usize)> {
    steps.iter().map(|s| (s.left, s.right)).collect()
}

/// Get the intermediate output IDs for each contraction step.
pub fn steps_output_ids<ID: Clone>(steps: &[ContractionStep<ID>]) -> Vec<Vec<ID>> {
    steps.iter().map(|s| s.output_ids.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_greedy_matmul() {
        // ij,jk->ik
        let input_ids = vec![vec!['i', 'j'], vec!['j', 'k']];
        let output_ids = vec!['i', 'k'];
        let sizes: HashMap<char, usize> = [('i', 10), ('j', 20), ('k', 10)].into_iter().collect();

        let steps = optimize_greedy(&input_ids, &output_ids, &sizes);
        assert!(steps.is_some());

        let steps = steps.unwrap();
        assert_eq!(steps.len(), 1); // One contraction for two tensors
        assert_eq!(steps[0].output_ids, vec!['i', 'k']);
    }

    #[test]
    fn test_optimize_greedy_chain() {
        // ij,jk,kl->il (matrix chain)
        let input_ids = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']];
        let output_ids = vec!['i', 'l'];
        let sizes: HashMap<char, usize> = [('i', 4), ('j', 8), ('k', 8), ('l', 4)]
            .into_iter()
            .collect();

        let steps = optimize_greedy(&input_ids, &output_ids, &sizes);
        assert!(steps.is_some());

        let steps = steps.unwrap();
        assert_eq!(steps.len(), 2); // Two contractions for three tensors
    }

    #[test]
    fn test_steps_to_path() {
        let steps = vec![
            ContractionStep {
                left: 0,
                right: 1,
                output_ids: vec!['i', 'k'],
            },
            ContractionStep {
                left: 2,
                right: 3,
                output_ids: vec!['i', 'l'],
            },
        ];

        let path = steps_to_path(&steps);
        assert_eq!(path, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn test_single_tensor() {
        let input_ids = vec![vec!['i', 'j']];
        let output_ids = vec!['i', 'j'];
        let sizes: HashMap<char, usize> = [('i', 10), ('j', 10)].into_iter().collect();

        let steps = optimize_greedy(&input_ids, &output_ids, &sizes);
        assert!(steps.is_some());
        assert!(steps.unwrap().is_empty()); // No contractions needed
    }
}
