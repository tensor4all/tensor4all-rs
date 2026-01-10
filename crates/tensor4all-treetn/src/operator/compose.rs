//! Composition of exclusive (non-overlapping) operators.
//!
//! This module provides functions to compose multiple operators that act on
//! non-overlapping regions into a single operator on the full target space.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use anyhow::{Context, Result};
use petgraph::stable_graph::NodeIndex;

use tensor4all_core::index::{DynId, Index, NoSymmSpace, Symmetry, TagSet};
use tensor4all_core::storage::{DenseStorageF64, Storage};
use tensor4all_core::{IndexLike, TensorDynLen};

use super::identity::build_identity_operator_tensor;
use super::Operator;
use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;
// TODO: Re-enable after operator module is refactored
// use crate::treetn::linsolve::{IndexMapping, LinearOperator};

/// Check if a set of operators are exclusive (non-overlapping) on the target network.
///
/// Operators are exclusive if:
/// 1. **Vertex-disjoint**: No two operators share a node
/// 2. **Connected subtrees**: Each operator's nodes form a connected subtree
/// 3. **Path-exclusive**: Paths between different operators don't cross other operators
///
/// # Arguments
///
/// * `target` - The target site index network (full space)
/// * `operators` - The operators to check
///
/// # Returns
///
/// `true` if operators are exclusive, `false` otherwise.
pub fn are_exclusive_operators<I, V, O>(
    target: &SiteIndexNetwork<V, I>,
    operators: &[&O],
) -> bool
where
    I: IndexLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
    O: Operator<I, V>,
{
    // Collect node sets for each operator
    let node_sets: Vec<HashSet<V>> = operators.iter().map(|op| op.node_names()).collect();

    // 1. Check vertex-disjoint
    for i in 0..node_sets.len() {
        for j in (i + 1)..node_sets.len() {
            if !node_sets[i].is_disjoint(&node_sets[j]) {
                return false;
            }
        }
    }

    // 2. Check each operator's nodes form a connected subtree in target
    for node_set in &node_sets {
        if node_set.is_empty() {
            continue;
        }

        // Convert to NodeIndex set
        let node_indices: HashSet<NodeIndex> = node_set
            .iter()
            .filter_map(|name| target.node_index(name))
            .collect();

        if node_indices.len() != node_set.len() {
            // Some nodes don't exist in target
            return false;
        }

        if !target.is_connected_subset(&node_indices) {
            return false;
        }
    }

    // 3. Path-exclusive check: paths between operators should not cross other operators
    for i in 0..node_sets.len() {
        for j in (i + 1)..node_sets.len() {
            if !check_path_exclusive(target, &node_sets[i], &node_sets[j], &node_sets) {
                return false;
            }
        }
    }

    true
}

/// Check if paths between two operator regions don't cross other operators.
fn check_path_exclusive<I, V>(
    target: &SiteIndexNetwork<V, I>,
    set_a: &HashSet<V>,
    set_b: &HashSet<V>,
    all_sets: &[HashSet<V>],
) -> bool
where
    I: IndexLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // Find a node from each set
    let node_a = match set_a.iter().next() {
        Some(n) => n,
        None => return true, // Empty set
    };
    let node_b = match set_b.iter().next() {
        Some(n) => n,
        None => return true,
    };

    // Get path between them
    let idx_a = match target.node_index(node_a) {
        Some(idx) => idx,
        None => return false,
    };
    let idx_b = match target.node_index(node_b) {
        Some(idx) => idx,
        None => return false,
    };

    let path = match target.path_between(idx_a, idx_b) {
        Some(p) => p,
        None => return false, // No path means disconnected, which is fine for exclusivity
    };

    // Check that path nodes (excluding endpoints) don't belong to other operators
    let other_operator_nodes: HashSet<&V> = all_sets
        .iter()
        .filter(|s| *s != set_a && *s != set_b)
        .flat_map(|s| s.iter())
        .collect();

    for node_idx in &path[1..path.len().saturating_sub(1)] {
        if let Some(name) = target.node_name(*node_idx) {
            if other_operator_nodes.contains(name) {
                return false;
            }
        }
    }

    true
}

/// Compose exclusive LinearOperators into a single LinearOperator.
///
/// This function takes multiple non-overlapping operators and combines them into
/// a single operator that acts on the full target space. Gap positions (nodes not
/// covered by any operator) are filled with identity operators.
///
/// # Arguments
///
/// * `target` - The full site index network (defines the output structure)
/// * `operators` - Non-overlapping LinearOperators to compose
/// * `gap_site_indices` - Site indices for gap nodes: node_name -> (input_index, output_index)
///
/// # Returns
///
/// A LinearOperator representing the composed operator on the full target space.
///
/// # Errors
///
/// Returns an error if:
/// - Operators are not exclusive (overlapping)
/// - Operator nodes don't exist in target
/// - Gap node site indices not provided
pub fn compose_exclusive_linear_operators<I, V>(
    target: &SiteIndexNetwork<V, I>,
    operators: &[&LinearOperator<I, V>],
    gap_site_indices: &HashMap<V, Vec<(I, I)>>,
) -> Result<LinearOperator<I, V>>
where
    I: IndexLike,
    I::Id: Clone + Hash + Eq + Ord + Debug + From<DynId> + Send + Sync,
    I::Symm: Clone + Symmetry + Debug + From<NoSymmSpace> + PartialEq + Send + Sync,
    I::Tags: Default,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // 1. Validate exclusivity
    if !are_exclusive_operators(target, operators) {
        return Err(anyhow::anyhow!(
            "Operators are not exclusive: they may overlap or not form connected subtrees"
        ))
        .context("compose_exclusive_linear_operators: operators must be exclusive");
    }

    // 2. Collect covered nodes
    let covered: HashSet<V> = operators.iter().flat_map(|op| op.node_names()).collect();

    // 3. Identify gap nodes
    let all_target_nodes: HashSet<V> = target.node_names().into_iter().cloned().collect();
    let gaps: Vec<V> = all_target_nodes.difference(&covered).cloned().collect();

    // 4. Build tensors and mappings
    let mut tensors: Vec<TensorDynLen<I::Id, I::Symm>> = Vec::new();
    let mut result_node_names: Vec<V> = Vec::new();
    let mut combined_input_mapping: HashMap<V, IndexMapping<I>> = HashMap::new();
    let mut combined_output_mapping: HashMap<V, IndexMapping<I>> = HashMap::new();

    // 4a. Add tensors and mappings from operators
    for op in operators {
        // Copy tensors
        for name in op.node_names() {
            let node_idx = op
                .mpo()
                .node_index(&name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", name))?;
            let tensor = op
                .mpo()
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", name))?;

            tensors.push(tensor.clone());
            result_node_names.push(name.clone());

            // Copy mappings
            if let Some(input_map) = op.get_input_mapping(&name) {
                combined_input_mapping.insert(name.clone(), input_map.clone());
            }
            if let Some(output_map) = op.get_output_mapping(&name) {
                combined_output_mapping.insert(name.clone(), output_map.clone());
            }
        }
    }

    // 4b. Add identity tensors at gaps
    for gap_name in gaps {
        let index_pairs = gap_site_indices.get(&gap_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Site indices not provided for gap node {:?}",
                gap_name
            )
        })?;

        if index_pairs.is_empty() {
            // No site indices at this gap - create scalar identity
            let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0])));
            let scalar_tensor = TensorDynLen::new(vec![], vec![], storage);
            tensors.push(scalar_tensor);
            result_node_names.push(gap_name);
            continue;
        }

        // Create internal indices for the identity operator
        // (different IDs from true indices)
        let mut internal_inputs: Vec<Index<I::Id, I::Symm, I::Tags>> = Vec::new();
        let mut internal_outputs: Vec<Index<I::Id, I::Symm, I::Tags>> = Vec::new();

        for (true_input, true_output) in index_pairs {
            // Create new internal indices with fresh IDs
            // Use Index::new_dyn to get unique IDs
            let dim_in = true_input.dim();
            let dim_out = true_output.dim();

            // Create internal indices with matching dimensions
            let internal_in_base: Index<DynId, NoSymmSpace, TagSet> = Index::new_dyn(dim_in);
            let internal_out_base: Index<DynId, NoSymmSpace, TagSet> = Index::new_dyn(dim_out);

            // Convert to the target type
            let internal_in: Index<I::Id, I::Symm, I::Tags> = Index::new_with_tags(
                I::Id::from(internal_in_base.id),
                I::Symm::from(internal_in_base.symm),
                I::Tags::default(),
            );
            let internal_out: Index<I::Id, I::Symm, I::Tags> = Index::new_with_tags(
                I::Id::from(internal_out_base.id),
                I::Symm::from(internal_out_base.symm),
                I::Tags::default(),
            );

            internal_inputs.push(internal_in.clone());
            internal_outputs.push(internal_out.clone());

            // Store mappings (first pair only for now - single site index per node)
            if combined_input_mapping.get(&gap_name).is_none() {
                combined_input_mapping.insert(
                    gap_name.clone(),
                    IndexMapping {
                        true_index: true_input.clone(),
                        internal_index: internal_in,
                    },
                );
            }
            if combined_output_mapping.get(&gap_name).is_none() {
                combined_output_mapping.insert(
                    gap_name.clone(),
                    IndexMapping {
                        true_index: true_output.clone(),
                        internal_index: internal_out,
                    },
                );
            }
        }

        // Build identity tensor
        let identity_tensor = build_identity_operator_tensor(&internal_inputs, &internal_outputs)
            .with_context(|| format!("Failed to build identity tensor for gap {:?}", gap_name))?;

        tensors.push(identity_tensor);
        result_node_names.push(gap_name);
    }

    // 5. Create TreeTN from tensors
    let mpo = TreeTN::from_tensors(tensors, result_node_names)
        .context("compose_exclusive_linear_operators: failed to create TreeTN")?;

    Ok(LinearOperator::new(
        mpo,
        combined_input_mapping,
        combined_output_mapping,
    ))
}

/// Compose exclusive operators into a single operator (convenience wrapper).
///
/// This is a generic version that accepts any type implementing the Operator trait.
/// For actual composition, use [`compose_exclusive_linear_operators`] with LinearOperator inputs.
pub fn compose_exclusive_operators<I, V, O>(
    _target: &SiteIndexNetwork<V, I>,
    _operators: &[&O],
    _gap_site_indices: &HashMap<V, Vec<(I, I)>>,
) -> Result<LinearOperator<I, V>>
where
    I: IndexLike,
    I::Id: Clone + Hash + Eq + Ord + Debug + From<DynId> + Send + Sync,
    I::Symm: Clone + Symmetry + Debug + From<NoSymmSpace> + PartialEq + Send + Sync,
    I::Tags: Default,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
    O: Operator<I, V>,
{
    // This function requires operators to be LinearOperator
    // Use compose_exclusive_linear_operators directly for LinearOperator inputs
    Err(anyhow::anyhow!(
        "Generic compose_exclusive_operators requires LinearOperator inputs. \
         Use compose_exclusive_linear_operators directly."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{random_treetn_f64, LinkSpace};
    use tensor4all_core::TensorAccess;

    type DynIndex = Index<DynId, NoSymmSpace, TagSet>;

    fn make_index(dim: usize) -> DynIndex {
        Index::new_dyn(dim)
    }

    fn create_chain_site_network(n: usize) -> SiteIndexNetwork<String, DynIndex> {
        let mut net = SiteIndexNetwork::new();
        for i in 0..n {
            let name = format!("N{}", i);
            let site_idx = make_index(2);
            net.add_node(name, [site_idx].into_iter().collect::<HashSet<_>>())
                .unwrap();
        }
        for i in 0..(n - 1) {
            net.add_edge(&format!("N{}", i), &format!("N{}", i + 1))
                .unwrap();
        }
        net
    }

    /// Create a simple LinearOperator from a TreeTN with explicit index mappings.
    fn create_linear_operator_from_treetn(
        mpo: TreeTN<DynIndex, String>,
        input_indices: &[(String, DynIndex, DynIndex)], // (node, true_input, internal_input)
        output_indices: &[(String, DynIndex, DynIndex)], // (node, true_output, internal_output)
    ) -> LinearOperator<DynIndex, String> {
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for (node, true_idx, internal_idx) in input_indices {
            input_mapping.insert(
                node.clone(),
                IndexMapping {
                    true_index: true_idx.clone(),
                    internal_index: internal_idx.clone(),
                },
            );
        }

        for (node, true_idx, internal_idx) in output_indices {
            output_mapping.insert(
                node.clone(),
                IndexMapping {
                    true_index: true_idx.clone(),
                    internal_index: internal_idx.clone(),
                },
            );
        }

        LinearOperator::new(mpo, input_mapping, output_mapping)
    }

    #[test]
    fn test_are_exclusive_disjoint() {
        // Target: N0 -- N1 -- N2 -- N3 -- N4
        let target = create_chain_site_network(5);

        // Create two non-overlapping operators
        let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        op1_net.add_node("N0".to_string(), HashSet::new()).unwrap();
        op1_net.add_node("N1".to_string(), HashSet::new()).unwrap();
        op1_net
            .add_edge(&"N0".to_string(), &"N1".to_string())
            .unwrap();

        let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        op2_net.add_node("N3".to_string(), HashSet::new()).unwrap();
        op2_net.add_node("N4".to_string(), HashSet::new()).unwrap();
        op2_net
            .add_edge(&"N3".to_string(), &"N4".to_string())
            .unwrap();

        // Create TreeTNs with these networks
        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let op1 = random_treetn_f64(&mut rng, &op1_net, link_space.clone());
        let op2 = random_treetn_f64(&mut rng, &op2_net, link_space.clone());

        // Test exclusivity
        let result = are_exclusive_operators(&target, &[&op1, &op2]);
        assert!(result, "Disjoint operators should be exclusive");
    }

    #[test]
    fn test_are_exclusive_overlapping() {
        // Target: N0 -- N1 -- N2 -- N3
        let target = create_chain_site_network(4);

        // Create overlapping operators (both include N1)
        let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        op1_net.add_node("N0".to_string(), HashSet::new()).unwrap();
        op1_net.add_node("N1".to_string(), HashSet::new()).unwrap();
        op1_net
            .add_edge(&"N0".to_string(), &"N1".to_string())
            .unwrap();

        let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        op2_net.add_node("N1".to_string(), HashSet::new()).unwrap();
        op2_net.add_node("N2".to_string(), HashSet::new()).unwrap();
        op2_net
            .add_edge(&"N1".to_string(), &"N2".to_string())
            .unwrap();

        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let op1 = random_treetn_f64(&mut rng, &op1_net, link_space.clone());
        let op2 = random_treetn_f64(&mut rng, &op2_net, link_space.clone());

        let result = are_exclusive_operators(&target, &[&op1, &op2]);
        assert!(!result, "Overlapping operators should not be exclusive");
    }

    #[test]
    fn test_are_exclusive_single_node_operators() {
        // Target: N0 -- N1 -- N2 -- N3
        let target = create_chain_site_network(4);

        // Create single-node operators
        let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        op1_net.add_node("N0".to_string(), HashSet::new()).unwrap();

        let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        op2_net.add_node("N2".to_string(), HashSet::new()).unwrap();

        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let op1 = random_treetn_f64(&mut rng, &op1_net, link_space.clone());
        let op2 = random_treetn_f64(&mut rng, &op2_net, link_space.clone());

        let result = are_exclusive_operators(&target, &[&op1, &op2]);
        assert!(result, "Single-node disjoint operators should be exclusive");
    }

    // =========================================================================
    // Integration tests for compose_exclusive_linear_operators
    // =========================================================================

    #[test]
    fn test_compose_exclusive_linear_operators_basic() {
        // Target: N0 -- N1 -- N2 -- N3 -- N4
        // Op1 covers N0, N1
        // Op2 covers N3, N4
        // Gap: N2 (needs identity)
        let target = create_chain_site_network(5);

        // Create site networks for operators
        let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s0_in = make_index(2);
        let s0_out = make_index(2);
        let s1_in = make_index(2);
        let s1_out = make_index(2);
        op1_net
            .add_node(
                "N0".to_string(),
                [s0_in.clone(), s0_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();
        op1_net
            .add_node(
                "N1".to_string(),
                [s1_in.clone(), s1_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();
        op1_net
            .add_edge(&"N0".to_string(), &"N1".to_string())
            .unwrap();

        let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s3_in = make_index(2);
        let s3_out = make_index(2);
        let s4_in = make_index(2);
        let s4_out = make_index(2);
        op2_net
            .add_node(
                "N3".to_string(),
                [s3_in.clone(), s3_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();
        op2_net
            .add_node(
                "N4".to_string(),
                [s4_in.clone(), s4_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();
        op2_net
            .add_edge(&"N3".to_string(), &"N4".to_string())
            .unwrap();

        // Create TreeTNs for the operators
        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let mpo1 = random_treetn_f64(&mut rng, &op1_net, link_space.clone());
        let mpo2 = random_treetn_f64(&mut rng, &op2_net, link_space.clone());

        // True site indices (what the composed operator maps from/to)
        let true_s0 = make_index(2);
        let true_s1 = make_index(2);
        let true_s3 = make_index(2);
        let true_s4 = make_index(2);

        // Create LinearOperators with explicit mappings
        let lin_op1 = create_linear_operator_from_treetn(
            mpo1,
            &[
                ("N0".to_string(), true_s0.clone(), s0_in.clone()),
                ("N1".to_string(), true_s1.clone(), s1_in.clone()),
            ],
            &[
                ("N0".to_string(), true_s0.clone(), s0_out.clone()),
                ("N1".to_string(), true_s1.clone(), s1_out.clone()),
            ],
        );

        let lin_op2 = create_linear_operator_from_treetn(
            mpo2,
            &[
                ("N3".to_string(), true_s3.clone(), s3_in.clone()),
                ("N4".to_string(), true_s4.clone(), s4_in.clone()),
            ],
            &[
                ("N3".to_string(), true_s3.clone(), s3_out.clone()),
                ("N4".to_string(), true_s4.clone(), s4_out.clone()),
            ],
        );

        // Gap site indices for N2
        let true_s2_in = make_index(2);
        let true_s2_out = make_index(2);
        let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
        gap_site_indices.insert(
            "N2".to_string(),
            vec![(true_s2_in.clone(), true_s2_out.clone())],
        );

        // Compose the operators
        let composed =
            compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices)
                .expect("Composition should succeed");

        // Verify the composed operator
        let node_names = composed.node_names();
        assert_eq!(node_names.len(), 5, "Composed operator should have 5 nodes");
        assert!(node_names.contains(&"N0".to_string()));
        assert!(node_names.contains(&"N1".to_string()));
        assert!(node_names.contains(&"N2".to_string())); // Gap node
        assert!(node_names.contains(&"N3".to_string()));
        assert!(node_names.contains(&"N4".to_string()));

        // Verify mappings exist for all nodes
        assert!(composed.get_input_mapping(&"N0".to_string()).is_some());
        assert!(composed.get_input_mapping(&"N1".to_string()).is_some());
        assert!(composed.get_input_mapping(&"N2".to_string()).is_some()); // Gap
        assert!(composed.get_input_mapping(&"N3".to_string()).is_some());
        assert!(composed.get_input_mapping(&"N4".to_string()).is_some());

        assert!(composed.get_output_mapping(&"N0".to_string()).is_some());
        assert!(composed.get_output_mapping(&"N1".to_string()).is_some());
        assert!(composed.get_output_mapping(&"N2".to_string()).is_some()); // Gap
        assert!(composed.get_output_mapping(&"N3".to_string()).is_some());
        assert!(composed.get_output_mapping(&"N4".to_string()).is_some());
    }

    #[test]
    fn test_compose_exclusive_linear_operators_single_operators() {
        // Target: N0 -- N1 -- N2
        // Op1 covers N0
        // Op2 covers N2
        // Gap: N1 (needs identity)
        let target = create_chain_site_network(3);

        // Create single-node site networks
        let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s0_in = make_index(2);
        let s0_out = make_index(2);
        op1_net
            .add_node(
                "N0".to_string(),
                [s0_in.clone(), s0_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();

        let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s2_in = make_index(2);
        let s2_out = make_index(2);
        op2_net
            .add_node(
                "N2".to_string(),
                [s2_in.clone(), s2_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();

        // Create TreeTNs
        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let mpo1 = random_treetn_f64(&mut rng, &op1_net, link_space.clone());
        let mpo2 = random_treetn_f64(&mut rng, &op2_net, link_space.clone());

        // True site indices
        let true_s0 = make_index(2);
        let true_s2 = make_index(2);

        // Create LinearOperators
        let lin_op1 = create_linear_operator_from_treetn(
            mpo1,
            &[("N0".to_string(), true_s0.clone(), s0_in.clone())],
            &[("N0".to_string(), true_s0.clone(), s0_out.clone())],
        );

        let lin_op2 = create_linear_operator_from_treetn(
            mpo2,
            &[("N2".to_string(), true_s2.clone(), s2_in.clone())],
            &[("N2".to_string(), true_s2.clone(), s2_out.clone())],
        );

        // Gap for N1
        let true_s1_in = make_index(2);
        let true_s1_out = make_index(2);
        let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
        gap_site_indices.insert(
            "N1".to_string(),
            vec![(true_s1_in.clone(), true_s1_out.clone())],
        );

        // Compose
        let composed =
            compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices)
                .expect("Composition should succeed");

        // Verify
        assert_eq!(composed.node_names().len(), 3);
    }

    #[test]
    fn test_compose_exclusive_linear_operators_no_gap() {
        // Target: N0 -- N1
        // Op1 covers N0
        // Op2 covers N1
        // No gap
        let target = create_chain_site_network(2);

        // Create single-node site networks
        let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s0_in = make_index(2);
        let s0_out = make_index(2);
        op1_net
            .add_node(
                "N0".to_string(),
                [s0_in.clone(), s0_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();

        let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s1_in = make_index(2);
        let s1_out = make_index(2);
        op2_net
            .add_node(
                "N1".to_string(),
                [s1_in.clone(), s1_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();

        // Create TreeTNs
        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let mpo1 = random_treetn_f64(&mut rng, &op1_net, link_space.clone());
        let mpo2 = random_treetn_f64(&mut rng, &op2_net, link_space.clone());

        // True site indices
        let true_s0 = make_index(2);
        let true_s1 = make_index(2);

        // Create LinearOperators
        let lin_op1 = create_linear_operator_from_treetn(
            mpo1,
            &[("N0".to_string(), true_s0.clone(), s0_in.clone())],
            &[("N0".to_string(), true_s0.clone(), s0_out.clone())],
        );

        let lin_op2 = create_linear_operator_from_treetn(
            mpo2,
            &[("N1".to_string(), true_s1.clone(), s1_in.clone())],
            &[("N1".to_string(), true_s1.clone(), s1_out.clone())],
        );

        // No gaps
        let gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();

        // Compose
        let composed =
            compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices)
                .expect("Composition should succeed");

        // Verify
        assert_eq!(composed.node_names().len(), 2);
        assert!(composed.get_input_mapping(&"N0".to_string()).is_some());
        assert!(composed.get_input_mapping(&"N1".to_string()).is_some());
    }

    #[test]
    fn test_compose_exclusive_linear_operators_overlap_error() {
        // Target: N0 -- N1 -- N2
        // Op1 covers N0, N1
        // Op2 covers N1, N2 (overlaps at N1!)
        let target = create_chain_site_network(3);

        // Create overlapping networks
        let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s0_in = make_index(2);
        let s1_in = make_index(2);
        op1_net
            .add_node(
                "N0".to_string(),
                [s0_in.clone()].into_iter().collect::<HashSet<_>>(),
            )
            .unwrap();
        op1_net
            .add_node(
                "N1".to_string(),
                [s1_in.clone()].into_iter().collect::<HashSet<_>>(),
            )
            .unwrap();
        op1_net
            .add_edge(&"N0".to_string(), &"N1".to_string())
            .unwrap();

        let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s1_in2 = make_index(2);
        let s2_in = make_index(2);
        op2_net
            .add_node(
                "N1".to_string(),
                [s1_in2.clone()].into_iter().collect::<HashSet<_>>(),
            )
            .unwrap();
        op2_net
            .add_node(
                "N2".to_string(),
                [s2_in.clone()].into_iter().collect::<HashSet<_>>(),
            )
            .unwrap();
        op2_net
            .add_edge(&"N1".to_string(), &"N2".to_string())
            .unwrap();

        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let mpo1 = random_treetn_f64(&mut rng, &op1_net, link_space.clone());
        let mpo2 = random_treetn_f64(&mut rng, &op2_net, link_space.clone());

        let lin_op1 = LinearOperator::new(mpo1, HashMap::new(), HashMap::new());
        let lin_op2 = LinearOperator::new(mpo2, HashMap::new(), HashMap::new());

        let gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();

        // Composition should fail due to overlap
        let result =
            compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices);
        assert!(result.is_err(), "Should fail for overlapping operators");
    }

    #[test]
    fn test_compose_gap_identity_tensor_is_diagonal() {
        // Test that gap nodes get proper identity tensors
        // Target: N0 -- N1 -- N2
        // Op covers N0, N2
        // Gap: N1
        let target = create_chain_site_network(3);

        // Create a two-node operator (non-contiguous in target, but we handle this separately)
        // Actually, for exclusivity check, we need connected subtrees.
        // Let's use single-node operators instead.

        let mut op1_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s0_in = make_index(2);
        let s0_out = make_index(2);
        op1_net
            .add_node(
                "N0".to_string(),
                [s0_in.clone(), s0_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();

        let mut op2_net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        let s2_in = make_index(2);
        let s2_out = make_index(2);
        op2_net
            .add_node(
                "N2".to_string(),
                [s2_in.clone(), s2_out.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .unwrap();

        let link_space = LinkSpace::uniform(2);
        let mut rng = rand::thread_rng();
        let mpo1 = random_treetn_f64(&mut rng, &op1_net, link_space.clone());
        let mpo2 = random_treetn_f64(&mut rng, &op2_net, link_space.clone());

        let true_s0 = make_index(2);
        let true_s2 = make_index(2);

        let lin_op1 = create_linear_operator_from_treetn(
            mpo1,
            &[("N0".to_string(), true_s0.clone(), s0_in.clone())],
            &[("N0".to_string(), true_s0.clone(), s0_out.clone())],
        );

        let lin_op2 = create_linear_operator_from_treetn(
            mpo2,
            &[("N2".to_string(), true_s2.clone(), s2_in.clone())],
            &[("N2".to_string(), true_s2.clone(), s2_out.clone())],
        );

        // Gap for N1 with dimension 3 (to distinguish from operator sites)
        let true_s1_in = make_index(3);
        let true_s1_out = make_index(3);
        let mut gap_site_indices: HashMap<String, Vec<(DynIndex, DynIndex)>> = HashMap::new();
        gap_site_indices.insert(
            "N1".to_string(),
            vec![(true_s1_in.clone(), true_s1_out.clone())],
        );

        let composed =
            compose_exclusive_linear_operators(&target, &[&lin_op1, &lin_op2], &gap_site_indices)
                .expect("Composition should succeed");

        // Get the tensor at N1 (should be identity)
        let n1_idx = composed.mpo().node_index(&"N1".to_string()).unwrap();
        let n1_tensor = composed.mpo().tensor(n1_idx).unwrap();

        // Identity tensor for dim 3 should have shape [3, 3] with indices [in, out]
        assert_eq!(n1_tensor.dims, vec![3, 3]);

        // Check it's diagonal (only diagonal elements are 1.0)
        let data = match n1_tensor.storage() {
            Storage::DenseF64(d) => d.as_slice(),
            _ => panic!("Expected DenseF64"),
        };

        // For 3x3 identity in row-major: [1,0,0, 0,1,0, 0,0,1]
        let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (i, (got, want)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "Identity tensor element {} mismatch: {} vs {}",
                i,
                got,
                want
            );
        }
    }
}
