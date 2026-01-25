//! Composition of exclusive (non-overlapping) operators.
//!
//! This module provides functions to compose multiple operators that act on
//! non-overlapping regions into a single operator on the full target space.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{Context, Result};
use petgraph::stable_graph::NodeIndex;

use tensor4all_core::{IndexLike, TensorLike};

use super::index_mapping::IndexMapping;
use super::linear_operator::LinearOperator;
use super::Operator;
use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;

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
pub fn are_exclusive_operators<T, V, O>(
    target: &SiteIndexNetwork<V, T::Index>,
    operators: &[&O],
) -> bool
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
    O: Operator<T, V>,
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
            if !check_path_exclusive::<T, V>(target, &node_sets[i], &node_sets[j], &node_sets) {
                return false;
            }
        }
    }

    true
}

/// Check if paths between two operator regions don't cross other operators.
fn check_path_exclusive<T, V>(
    target: &SiteIndexNetwork<V, T::Index>,
    set_a: &HashSet<V>,
    set_b: &HashSet<V>,
    all_sets: &[HashSet<V>],
) -> bool
where
    T: TensorLike,
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
/// covered by any operator) are filled with identity operators using `T::delta()`.
///
/// # Arguments
///
/// * `target` - The full site index network (defines the output structure)
/// * `operators` - Non-overlapping LinearOperators to compose
/// * `gap_site_indices` - Site indices for gap nodes: node_name -> [(input_index, output_index), ...]
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
#[allow(clippy::type_complexity)]
pub fn compose_exclusive_linear_operators<T, V>(
    target: &SiteIndexNetwork<V, T::Index>,
    operators: &[&LinearOperator<T, V>],
    gap_site_indices: &HashMap<V, Vec<(T::Index, T::Index)>>,
) -> Result<LinearOperator<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // 1. Validate exclusivity
    if !are_exclusive_operators::<T, V, _>(target, operators) {
        return Err(anyhow::anyhow!(
            "Operators are not exclusive: they may overlap or not form connected subtrees"
        ))
        .context("compose_exclusive_linear_operators: operators must be exclusive");
    }

    // 2. Collect covered nodes and build node-to-operator map
    let covered: HashSet<V> = operators.iter().flat_map(|op| op.node_names()).collect();

    let mut node_to_operator: HashMap<V, usize> = HashMap::new();
    for (op_idx, op) in operators.iter().enumerate() {
        for name in op.node_names() {
            node_to_operator.insert(name, op_idx);
        }
    }

    // 3. Identify gap nodes
    let all_target_nodes: HashSet<V> = target.node_names().into_iter().cloned().collect();
    let gaps: Vec<V> = all_target_nodes.difference(&covered).cloned().collect();
    let gap_set: HashSet<V> = gaps.iter().cloned().collect();

    // 4. Identify cross-component edges and create dummy link pairs
    // An edge is cross-component if endpoints are in different operators or one is a gap
    let mut dummy_links_for_node: HashMap<V, Vec<T::Index>> = HashMap::new();

    for (node_a, node_b) in target.edges() {
        let comp_a = node_to_operator.get(&node_a);
        let comp_b = node_to_operator.get(&node_b);
        let is_gap_a = gap_set.contains(&node_a);
        let is_gap_b = gap_set.contains(&node_b);

        let is_cross = match (comp_a, comp_b, is_gap_a, is_gap_b) {
            (Some(a), Some(b), false, false) => a != b,
            _ => true,
        };

        if is_cross {
            let (link_a, link_b) = T::Index::create_dummy_link_pair();
            dummy_links_for_node
                .entry(node_a.clone())
                .or_default()
                .push(link_a);
            dummy_links_for_node
                .entry(node_b.clone())
                .or_default()
                .push(link_b);
        }
    }

    // 5. Build tensors and mappings
    let mut tensors: Vec<T> = Vec::new();
    let mut result_node_names: Vec<V> = Vec::new();
    let mut combined_input_mapping: HashMap<V, IndexMapping<T::Index>> = HashMap::new();
    let mut combined_output_mapping: HashMap<V, IndexMapping<T::Index>> = HashMap::new();

    // 5a. Add tensors from operators (with dummy links added via outer product)
    for op in operators {
        for name in op.node_names() {
            let node_idx = op
                .mpo()
                .node_index(&name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", name))?;
            let mut tensor = op
                .mpo()
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", name))?
                .clone();

            // Add dummy links via outer product with ones tensor
            if let Some(links) = dummy_links_for_node.get(&name) {
                for link in links {
                    let ones = T::ones(std::slice::from_ref(link)).with_context(|| {
                        format!("Failed to create ones tensor for dummy link at {:?}", name)
                    })?;
                    tensor = tensor.outer_product(&ones).with_context(|| {
                        format!("Failed to add dummy link to tensor at {:?}", name)
                    })?;
                }
            }

            tensors.push(tensor);
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

    // 5b. Add identity tensors at gaps (with dummy links added via outer product)
    for gap_name in gaps {
        let index_pairs = gap_site_indices.get(&gap_name).ok_or_else(|| {
            anyhow::anyhow!("Site indices not provided for gap node {:?}", gap_name)
        })?;

        // Collect site indices for delta
        let input_indices: Vec<T::Index> = index_pairs.iter().map(|(i, _)| i.clone()).collect();
        let output_indices: Vec<T::Index> = index_pairs.iter().map(|(_, o)| o.clone()).collect();

        // Store mappings for the first site pair (if any)
        if let Some((true_input, true_output)) = index_pairs.first() {
            if !combined_input_mapping.contains_key(&gap_name) {
                combined_input_mapping.insert(
                    gap_name.clone(),
                    IndexMapping {
                        true_index: true_input.clone(),
                        internal_index: input_indices[0].clone(),
                    },
                );
            }
            if !combined_output_mapping.contains_key(&gap_name) {
                combined_output_mapping.insert(
                    gap_name.clone(),
                    IndexMapping {
                        true_index: true_output.clone(),
                        internal_index: output_indices[0].clone(),
                    },
                );
            }
        }

        // Create delta tensor for site indices
        let mut identity_tensor = if input_indices.is_empty() {
            T::delta(&[], &[]).context("Failed to create scalar identity tensor")?
        } else {
            T::delta(&input_indices, &output_indices).with_context(|| {
                format!("Failed to build identity tensor for gap {:?}", gap_name)
            })?
        };

        // Add dummy links via outer product (bond indices, not site indices)
        if let Some(links) = dummy_links_for_node.get(&gap_name) {
            for link in links {
                let ones = T::ones(std::slice::from_ref(link)).with_context(|| {
                    format!(
                        "Failed to create ones tensor for dummy link at gap {:?}",
                        gap_name
                    )
                })?;
                identity_tensor = identity_tensor.outer_product(&ones).with_context(|| {
                    format!("Failed to add dummy link to gap tensor {:?}", gap_name)
                })?;
            }
        }

        tensors.push(identity_tensor);
        result_node_names.push(gap_name);
    }

    // 6. Create TreeTN from tensors
    let mpo = TreeTN::from_tensors(tensors, result_node_names)
        .context("compose_exclusive_linear_operators: failed to create TreeTN")?;

    Ok(LinearOperator::new(
        mpo,
        combined_input_mapping,
        combined_output_mapping,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{random_treetn_f64, LinkSpace};
    use tensor4all_core::index::{DynId, Index, TagSet};
    use tensor4all_core::TensorDynLen;

    type DynIndex = Index<DynId, TagSet>;

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
        mpo: TreeTN<TensorDynLen, String>,
        input_indices: &[(String, DynIndex, DynIndex)], // (node, true_input, internal_input)
        output_indices: &[(String, DynIndex, DynIndex)], // (node, true_output, internal_output)
    ) -> LinearOperator<TensorDynLen, String> {
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
        let result = are_exclusive_operators::<TensorDynLen, _, _>(&target, &[&op1, &op2]);
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

        let result = are_exclusive_operators::<TensorDynLen, _, _>(&target, &[&op1, &op2]);
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

        let result = are_exclusive_operators::<TensorDynLen, _, _>(&target, &[&op1, &op2]);
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
        assert!(node_names.contains("N0"));
        assert!(node_names.contains("N1"));
        assert!(node_names.contains("N2")); // Gap node
        assert!(node_names.contains("N3"));
        assert!(node_names.contains("N4"));

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

        // Get the tensor at N1 (should be identity with dummy links)
        let n1_idx = composed.mpo().node_index(&"N1".to_string()).unwrap();
        let n1_tensor = composed.mpo().tensor(n1_idx).unwrap();

        // N1 has 2 neighbors (N0 and N2), so 2 dummy links of dim 1 are added
        // Shape should be [3, 3, 1, 1] (site_in, site_out, dummy_link1, dummy_link2)
        // The order may vary, but total size should be 3*3*1*1 = 9
        let n1_dims = n1_tensor.dims();
        let total_size: usize = n1_dims.iter().product();
        assert_eq!(
            total_size, 9,
            "Total tensor size should be 9 (3x3 identity with dim-1 links)"
        );

        // Check that site dimensions 3 appear exactly twice
        let dim3_count = n1_dims.iter().filter(|&&d| d == 3).count();
        assert_eq!(
            dim3_count, 2,
            "Should have exactly 2 site dimensions of size 3"
        );

        // Check it's effectively an identity (only diagonal elements are non-zero)
        let data = n1_tensor.as_slice_f64().expect("Expected DenseF64");

        // For identity with dim-1 dummy links, data is still [1,0,0, 0,1,0, 0,0,1]
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
