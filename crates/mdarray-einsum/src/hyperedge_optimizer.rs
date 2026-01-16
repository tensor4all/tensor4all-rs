//! Hyperedge-aware greedy contraction path optimizer.
//!
//! Unlike standard tensor network optimizers (like omeco), this optimizer
//! correctly handles hyperedges - indices that appear in 3 or more tensors.
//!
//! # Algorithm
//!
//! Instead of selecting pairs of tensors to contract, this optimizer selects
//! indices to contract. When an index is selected:
//!
//! 1. Find all tensors sharing that index
//! 2. Contract ALL those tensors simultaneously
//! 3. If the index is not in the output, take trace (sum over it)
//!
//! This ensures correct semantics: `sum_j(A*B*C)` instead of `(sum_j A*B) * C`.

use crate::AxisId;
use std::collections::{HashMap, HashSet};

/// A contraction step in the hyperedge-aware path.
#[derive(Debug, Clone)]
pub struct HyperedgeStep<ID> {
    /// Indices of operands to contract (may be 2 or more)
    pub operand_indices: Vec<usize>,
    /// The index being contracted (summed over), None for pure outer product
    pub contracted_index: Option<ID>,
    /// Output axis IDs for the intermediate result
    pub output_ids: Vec<ID>,
}

/// Result of hyperedge-aware optimization.
#[derive(Debug, Clone)]
pub struct HyperedgePath<ID> {
    /// Sequence of contraction steps
    pub steps: Vec<HyperedgeStep<ID>>,
}

/// Estimate the cost of contracting tensors sharing a given index.
///
/// Cost = product of all dimensions involved in the contraction.
fn estimate_contraction_cost<ID: AxisId>(
    index: &ID,
    operand_ids: &[Vec<ID>],
    sizes: &HashMap<ID, usize>,
) -> f64 {
    // Find all tensors sharing this index
    let sharing: Vec<usize> = operand_ids
        .iter()
        .enumerate()
        .filter(|(_, ids)| ids.contains(index))
        .map(|(i, _)| i)
        .collect();

    if sharing.len() < 2 {
        return f64::INFINITY; // Can't contract an index in fewer than 2 tensors
    }

    // Collect all unique indices involved
    let mut all_indices: HashSet<&ID> = HashSet::new();
    for &i in &sharing {
        for id in &operand_ids[i] {
            all_indices.insert(id);
        }
    }

    // Cost = product of all dimensions
    let mut cost: f64 = 1.0;
    for id in all_indices {
        let size = sizes.get(id).copied().unwrap_or(1);
        cost *= size as f64;
    }

    cost
}

/// Optimize contraction order using hyperedge-aware greedy algorithm.
///
/// # Arguments
/// * `input_ids` - Axis IDs for each input tensor
/// * `output_ids` - Axis IDs for the output
/// * `sizes` - Dimension sizes for each axis ID
///
/// # Returns
/// A HyperedgePath containing the contraction steps.
pub fn optimize_hyperedge_greedy<ID: AxisId>(
    input_ids: &[Vec<ID>],
    output_ids: &[ID],
    sizes: &HashMap<ID, usize>,
) -> HyperedgePath<ID> {
    if input_ids.len() <= 1 {
        return HyperedgePath { steps: vec![] };
    }

    let output_set: HashSet<&ID> = output_ids.iter().collect();
    let mut steps = Vec::new();

    // Working copies of operand IDs (will be modified as we contract)
    let mut operand_ids: Vec<Vec<ID>> = input_ids.to_vec();

    // Keep contracting until we have one operand left
    while operand_ids.len() > 1 {
        // Find all contractable indices (appear in 2+ operands, not in output)
        let mut index_counts: HashMap<&ID, usize> = HashMap::new();
        for ids in &operand_ids {
            for id in ids {
                *index_counts.entry(id).or_insert(0) += 1;
            }
        }

        // Indices that can be contracted: appear in 2+ tensors and not in output
        let contractable: Vec<&ID> = index_counts
            .iter()
            .filter(|(id, &count)| count >= 2 && !output_set.contains(*id))
            .map(|(id, _)| *id)
            .collect();

        if contractable.is_empty() {
            // No more indices to contract - need to do outer products
            // Contract the two smallest operands
            let (i, j) = find_smallest_pair(&operand_ids, sizes);
            let step = create_outer_product_step(&operand_ids, i, j, &output_set, sizes);
            apply_step(&mut operand_ids, &step);
            steps.push(step);
            continue;
        }

        // Find the index with minimum contraction cost
        let best_index = contractable
            .into_iter()
            .min_by(|a, b| {
                let cost_a = estimate_contraction_cost(*a, &operand_ids, sizes);
                let cost_b = estimate_contraction_cost(*b, &operand_ids, sizes);
                cost_a.partial_cmp(&cost_b).unwrap()
            })
            .unwrap()
            .clone();

        // Create and apply the contraction step
        let step = create_contraction_step(&operand_ids, &best_index, &output_set, sizes);
        apply_step(&mut operand_ids, &step);
        steps.push(step);
    }

    HyperedgePath { steps }
}

/// Find the two smallest operands (for outer product when no shared indices).
fn find_smallest_pair<ID: AxisId>(
    operand_ids: &[Vec<ID>],
    sizes: &HashMap<ID, usize>,
) -> (usize, usize) {
    let mut operand_sizes: Vec<(usize, usize)> = operand_ids
        .iter()
        .enumerate()
        .map(|(i, ids)| {
            let size: usize = ids
                .iter()
                .map(|id| sizes.get(id).copied().unwrap_or(1))
                .product();
            (i, size)
        })
        .collect();

    operand_sizes.sort_by_key(|(_, size)| *size);

    let i = operand_sizes[0].0;
    let j = operand_sizes[1].0;
    if i < j {
        (i, j)
    } else {
        (j, i)
    }
}

/// Create a step for contracting all tensors sharing a given index.
fn create_contraction_step<ID: AxisId>(
    operand_ids: &[Vec<ID>],
    contracted_index: &ID,
    output_set: &HashSet<&ID>,
    _sizes: &HashMap<ID, usize>,
) -> HyperedgeStep<ID> {
    // Find all operands sharing this index
    let sharing: Vec<usize> = operand_ids
        .iter()
        .enumerate()
        .filter(|(_, ids)| ids.contains(contracted_index))
        .map(|(i, _)| i)
        .collect();

    // Collect remaining operand indices (not being contracted)
    let remaining_operands: HashSet<usize> = (0..operand_ids.len())
        .filter(|i| !sharing.contains(i))
        .collect();

    // Collect IDs from remaining operands
    let remaining_ids: HashSet<&ID> = remaining_operands
        .iter()
        .flat_map(|&i| operand_ids[i].iter())
        .collect();

    // Compute output IDs for this step
    // Keep IDs that are: in final output OR in remaining operands
    // The contracted_index is NOT kept (it's being summed over)
    let mut step_output_ids: Vec<ID> = Vec::new();
    let mut seen: HashSet<&ID> = HashSet::new();

    for &i in &sharing {
        for id in &operand_ids[i] {
            if id == contracted_index {
                continue; // This is being contracted
            }
            let needed = output_set.contains(id) || remaining_ids.contains(id);
            if needed && !seen.contains(id) {
                step_output_ids.push(id.clone());
                seen.insert(id);
            }
        }
    }

    HyperedgeStep {
        operand_indices: sharing,
        contracted_index: Some(contracted_index.clone()),
        output_ids: step_output_ids,
    }
}

/// Create a step for outer product (no shared indices to contract).
fn create_outer_product_step<ID: AxisId>(
    operand_ids: &[Vec<ID>],
    i: usize,
    j: usize,
    output_set: &HashSet<&ID>,
    _sizes: &HashMap<ID, usize>,
) -> HyperedgeStep<ID> {
    // For outer product, we need a "dummy" contracted index
    // We'll use the first ID from operand i that's shared with j but not in output
    // If none exists, this is a pure outer product

    let ids_i: HashSet<&ID> = operand_ids[i].iter().collect();
    let ids_j: HashSet<&ID> = operand_ids[j].iter().collect();

    // Find shared index not in output
    let shared_not_output: Option<&ID> = ids_i
        .intersection(&ids_j)
        .find(|id| !output_set.contains(*id))
        .copied();

    // Collect remaining operand indices
    let remaining_operands: HashSet<usize> = (0..operand_ids.len())
        .filter(|&k| k != i && k != j)
        .collect();

    let remaining_ids: HashSet<&ID> = remaining_operands
        .iter()
        .flat_map(|&k| operand_ids[k].iter())
        .collect();

    // Compute output IDs
    let mut step_output_ids: Vec<ID> = Vec::new();
    let mut seen: HashSet<&ID> = HashSet::new();

    for id in &operand_ids[i] {
        if shared_not_output.is_some() && Some(id) == shared_not_output {
            continue;
        }
        let needed = output_set.contains(id) || remaining_ids.contains(id);
        if needed && !seen.contains(id) {
            step_output_ids.push(id.clone());
            seen.insert(id);
        }
    }

    for id in &operand_ids[j] {
        if shared_not_output.is_some() && Some(id) == shared_not_output {
            continue;
        }
        let needed = output_set.contains(id) || remaining_ids.contains(id);
        if needed && !seen.contains(id) {
            step_output_ids.push(id.clone());
            seen.insert(id);
        }
    }

    // contracted_index is None for pure outer product (no index to sum over)
    let contracted = shared_not_output.cloned();

    HyperedgeStep {
        operand_indices: vec![i, j],
        contracted_index: contracted,
        output_ids: step_output_ids,
    }
}

/// Apply a contraction step to the operand list.
fn apply_step<ID: AxisId>(operand_ids: &mut Vec<Vec<ID>>, step: &HyperedgeStep<ID>) {
    // Remove operands in reverse order to preserve indices
    let mut indices = step.operand_indices.clone();
    indices.sort_by(|a, b| b.cmp(a)); // Sort descending

    for i in indices {
        operand_ids.remove(i);
    }

    // Add the result
    operand_ids.push(step.output_ids.clone());
}

/// Check if a path contains any hyperedge contractions (3+ tensors at once).
pub fn has_hyperedge_contraction<ID>(path: &HyperedgePath<ID>) -> bool {
    path.steps.iter().any(|s| s.operand_indices.len() >= 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_chain() {
        // ij,jk,kl->il
        let input_ids = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']];
        let output_ids = vec!['i', 'l'];
        let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)]
            .into_iter()
            .collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // Should have 2 steps for 3 tensors
        assert_eq!(path.steps.len(), 2);

        // All steps should be pairwise (no hyperedges in this case)
        assert!(!has_hyperedge_contraction(&path));
    }

    #[test]
    fn test_hyperedge_contracted() {
        // ij,jk,jl->ikl (j is a hyperedge, contracted)
        let input_ids = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['j', 'l']];
        let output_ids = vec!['i', 'k', 'l'];
        let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 3)]
            .into_iter()
            .collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // Should contract all 3 tensors sharing j in one step
        assert!(has_hyperedge_contraction(&path));

        // First step should contract j (appears in all 3 tensors)
        let first_step = &path.steps[0];
        assert_eq!(first_step.operand_indices.len(), 3);
        assert_eq!(first_step.contracted_index, Some('j'));

        // Output should not contain j
        assert!(!first_step.output_ids.contains(&'j'));
    }

    #[test]
    fn test_hyperedge_output() {
        // ij,jk,jl->ijkl (j is a hyperedge, kept in output)
        let input_ids = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['j', 'l']];
        let output_ids = vec!['i', 'j', 'k', 'l'];
        let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 3)]
            .into_iter()
            .collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // j is in output, so it should NOT be contracted as a hyperedge
        // Instead, we should have pairwise contractions
        // No step should have 3+ operands because j must be preserved
        for step in &path.steps {
            // j should never be the contracted index (it's in output)
            assert_ne!(step.contracted_index, Some('j'));
        }
    }

    #[test]
    fn test_mixed_hyperedge() {
        // ij,jk,kl,km->ilm (j and k are shared, k is hyperedge)
        let input_ids = vec![
            vec!['i', 'j'],
            vec!['j', 'k'],
            vec!['k', 'l'],
            vec!['k', 'm'],
        ];
        let output_ids = vec!['i', 'l', 'm'];
        let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2), ('m', 2)]
            .into_iter()
            .collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // Should have steps contracting the hyperedge k (3 tensors share it)
        assert!(!path.steps.is_empty());
    }

    #[test]
    fn test_single_tensor() {
        let input_ids = vec![vec!['i', 'j']];
        let output_ids = vec!['i', 'j'];
        let sizes: HashMap<char, usize> = [('i', 2), ('j', 2)].into_iter().collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // No steps needed for single tensor
        assert!(path.steps.is_empty());
    }

    #[test]
    fn test_two_hyperedges() {
        // ij,jk,jl,km,lm->im
        // j is hyperedge (in tensors 0,1,2), contracted
        // k is normal edge (in tensors 1,3), contracted
        // l is normal edge (in tensors 2,4), contracted
        // m is normal edge (in tensors 3,4), in output
        let input_ids = vec![
            vec!['i', 'j'], // tensor 0
            vec!['j', 'k'], // tensor 1
            vec!['j', 'l'], // tensor 2
            vec!['k', 'm'], // tensor 3
            vec!['l', 'm'], // tensor 4
        ];
        let output_ids = vec!['i', 'm'];
        let sizes: HashMap<char, usize> = [('i', 2), ('j', 3), ('k', 2), ('l', 2), ('m', 2)]
            .into_iter()
            .collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // j is a hyperedge (appears in 3 tensors)
        // The optimizer should contract all j-sharing tensors together
        assert!(!path.steps.is_empty());

        // Verify the hyperedge j is contracted at some point
        let contracts_j = path.steps.iter().any(|s| s.contracted_index == Some('j'));
        assert!(contracts_j, "Should contract hyperedge j");
    }

    #[test]
    fn test_two_independent_hyperedges() {
        // Two separate hyperedges that don't share tensors
        // ij,jk,jl,mn,no,np->iklop
        // j is hyperedge (tensors 0,1,2)
        // n is hyperedge (tensors 3,4,5)
        let input_ids = vec![
            vec!['i', 'j'], // tensor 0
            vec!['j', 'k'], // tensor 1
            vec!['j', 'l'], // tensor 2
            vec!['m', 'n'], // tensor 3
            vec!['n', 'o'], // tensor 4
            vec!['n', 'p'], // tensor 5
        ];
        let output_ids = vec!['i', 'k', 'l', 'm', 'o', 'p'];
        let sizes: HashMap<char, usize> = [
            ('i', 2),
            ('j', 2),
            ('k', 2),
            ('l', 2),
            ('m', 2),
            ('n', 2),
            ('o', 2),
            ('p', 2),
        ]
        .into_iter()
        .collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // Both j and n should be contracted
        let contracts_j = path.steps.iter().any(|s| s.contracted_index == Some('j'));
        let contracts_n = path.steps.iter().any(|s| s.contracted_index == Some('n'));
        assert!(contracts_j, "Should contract hyperedge j");
        assert!(contracts_n, "Should contract hyperedge n");
    }

    #[test]
    fn test_nested_hyperedges() {
        // Hyperedges that share a tensor
        // ij,jk,jkl->il
        // j appears in tensors 0,1,2 (hyperedge)
        // k appears in tensors 1,2 (normal edge)
        let input_ids = vec![
            vec!['i', 'j'],      // tensor 0
            vec!['j', 'k'],      // tensor 1
            vec!['j', 'k', 'l'], // tensor 2
        ];
        let output_ids = vec!['i', 'l'];
        let sizes: HashMap<char, usize> = [('i', 2), ('j', 2), ('k', 2), ('l', 2)]
            .into_iter()
            .collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // j is a hyperedge, k is shared between 2 tensors
        // Both should be contracted
        assert!(!path.steps.is_empty());
    }

    #[test]
    fn test_hyperedge_with_different_sizes() {
        // Test that optimizer considers sizes when choosing contraction order
        // ij,jk,jl->ikl where j has large dimension
        let input_ids = vec![vec!['i', 'j'], vec!['j', 'k'], vec!['j', 'l']];
        let output_ids = vec!['i', 'k', 'l'];
        let sizes: HashMap<char, usize> = [
            ('i', 10),
            ('j', 100), // Large dimension
            ('k', 10),
            ('l', 10),
        ]
        .into_iter()
        .collect();

        let path = optimize_hyperedge_greedy(&input_ids, &output_ids, &sizes);

        // The optimizer should still contract j since it's the only contractable index
        assert!(has_hyperedge_contraction(&path));
    }
}
