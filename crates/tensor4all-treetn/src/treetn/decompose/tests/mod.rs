use super::*;
use std::collections::HashMap;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};

#[test]
fn test_tree_topology_validate() {
    // Test empty topology (use usize as a dummy ID type)
    let empty: TreeTopology<String, usize> = TreeTopology::new(HashMap::new(), Vec::new());
    assert!(empty.validate().is_err());

    // Test single node (valid)
    let mut nodes = HashMap::new();
    nodes.insert("node0".to_string(), vec![0usize]);
    let single = TreeTopology::new(nodes, Vec::new());
    assert!(single.validate().is_ok());

    // Test valid tree (2 nodes, 1 edge)
    let mut nodes = HashMap::new();
    nodes.insert("node0".to_string(), vec![0usize]);
    nodes.insert("node1".to_string(), vec![1usize]);
    let edges = vec![("node0".to_string(), "node1".to_string())];
    let tree = TreeTopology::new(nodes, edges);
    assert!(tree.validate().is_ok());

    // Test invalid: wrong number of edges
    let mut nodes = HashMap::new();
    nodes.insert("node0".to_string(), vec![0usize]);
    nodes.insert("node1".to_string(), vec![1usize]);
    let edges = vec![
        ("node0".to_string(), "node1".to_string()),
        ("node0".to_string(), "node1".to_string()),
    ];
    let invalid = TreeTopology::new(nodes, edges);
    assert!(invalid.validate().is_err());

    // Test invalid: edge references unknown node
    let mut nodes = HashMap::new();
    nodes.insert("node0".to_string(), vec![0usize]);
    let edges = vec![("node0".to_string(), "node2".to_string())];
    let invalid = TreeTopology::new(nodes, edges);
    assert!(invalid.validate().is_err());
}

#[test]
fn test_factorize_tensor_to_treetn_rejects_duplicate_index_ids() {
    let i0 = DynIndex::new_dyn(2);
    let i1 = DynIndex::new_dyn(2);
    let tensor =
        TensorDynLen::from_dense(vec![i0.clone(), i1.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let mut nodes: HashMap<String, Vec<<DynIndex as IndexLike>::Id>> = HashMap::new();
    nodes.insert("node0".to_string(), vec![*i0.id()]);
    nodes.insert("node1".to_string(), vec![*i0.id()]); // duplicate on purpose

    let topo = TreeTopology::new(nodes, vec![("node0".to_string(), "node1".to_string())]);

    let result = factorize_tensor_to_treetn_with(
        &tensor,
        &topo,
        FactorizeOptions::qr(),
        &"node0".to_string(),
    );
    assert!(result.is_err());
}

#[test]
fn test_factorize_tensor_to_treetn_preserves_requested_chain_topology_for_nonroot_leaf() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(
        vec![s0.clone(), s1.clone(), s2.clone()],
        vec![10.0, 30.0, 20.0, 40.0, 11.0, 31.0, 21.0, 41.0],
    )
    .unwrap();

    let mut nodes = HashMap::new();
    nodes.insert(0usize, vec![*s0.id()]);
    nodes.insert(1usize, vec![*s1.id()]);
    nodes.insert(2usize, vec![*s2.id()]);
    let topology = TreeTopology::new(nodes, vec![(0usize, 1usize), (1usize, 2usize)]);

    let tn = factorize_tensor_to_treetn(&tensor, &topology, &2usize).unwrap();
    let mut edges = tn.site_index_network().edges().collect::<Vec<_>>();
    edges.sort();

    assert_eq!(edges, vec![(0usize, 1usize), (1usize, 2usize)]);
}
