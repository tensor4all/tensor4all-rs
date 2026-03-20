
use super::*;
use tensor4all_core::DynIndex;

#[test]
fn test_site_index_network_basic() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    let site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
    net.add_node("A".to_string(), site1).unwrap();
    net.add_node("B".to_string(), site2).unwrap();

    assert_eq!(net.node_count(), 2);
    assert!(net.has_node(&"A".to_string()));

    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    assert_eq!(net.edge_count(), 1);
}

#[test]
fn test_post_order_dfs_chain() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    net.add_node("A".to_string(), empty.clone()).unwrap();
    net.add_node("B".to_string(), empty.clone()).unwrap();
    net.add_node("C".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let result = net.post_order_dfs(&"B".to_string()).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.last().unwrap(), "B");
    assert!(result.contains(&"A".to_string()));
    assert!(result.contains(&"C".to_string()));
}

#[test]
fn test_path_between_chain() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    let a = net.add_node("A".to_string(), empty.clone()).unwrap();
    let b = net.add_node("B".to_string(), empty.clone()).unwrap();
    let c = net.add_node("C".to_string(), empty.clone()).unwrap();
    let d = net.add_node("D".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    let path = net.path_between(a, d).unwrap();
    assert_eq!(path, vec![a, b, c, d]);
}

#[test]
fn test_edges_to_canonicalize_full() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    let a = net.add_node("A".to_string(), empty.clone()).unwrap();
    let b = net.add_node("B".to_string(), empty.clone()).unwrap();
    let c = net.add_node("C".to_string(), empty.clone()).unwrap();
    let d = net.add_node("D".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    let edges = net.edges_to_canonicalize(None, d);
    assert_eq!(edges.len(), 3);
    let edge_vec: Vec<_> = edges.iter().cloned().collect();
    assert_eq!(edge_vec, vec![(a, b), (b, c), (c, d)]);
}

#[test]
fn test_is_connected_subset() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    let a = net.add_node("A".to_string(), empty.clone()).unwrap();
    let b = net.add_node("B".to_string(), empty.clone()).unwrap();
    let c = net.add_node("C".to_string(), empty.clone()).unwrap();
    let _d = net.add_node("D".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    assert!(net.is_connected_subset(&HashSet::new()));
    assert!(net.is_connected_subset(&[a].into()));
    assert!(net.is_connected_subset(&[b, c].into()));
    assert!(!net.is_connected_subset(&[a, c].into()));
}

#[test]
fn test_share_equivalent_site_index_network() {
    let mut net1: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    net1.add_node("A".to_string(), site1.clone()).unwrap();
    net1.add_node("B".to_string(), HashSet::new()).unwrap();
    net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let mut net2: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net2.add_node("A".to_string(), site1.clone()).unwrap();
    net2.add_node("B".to_string(), HashSet::new()).unwrap();
    net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(net1.share_equivalent_site_index_network(&net2));

    // Different site space
    let mut net3: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site3: HashSet<_> = [DynIndex::new_dyn(5)].into();
    net3.add_node("A".to_string(), site3).unwrap();
    net3.add_node("B".to_string(), HashSet::new()).unwrap();
    net3.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(!net1.share_equivalent_site_index_network(&net3));
}

#[test]
fn test_apply_operator_topology() {
    // Create state network
    let mut state: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    let site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
    state.add_node("A".to_string(), site1.clone()).unwrap();
    state.add_node("B".to_string(), site2.clone()).unwrap();
    state.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    // Create operator with same topology
    let mut operator: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    // Operator has different index IDs but same dimensions
    let op_site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    let op_site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
    operator.add_node("A".to_string(), op_site1).unwrap();
    operator.add_node("B".to_string(), op_site2).unwrap();
    operator
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    // Should succeed
    let result = state.apply_operator_topology(&operator);
    assert!(result.is_ok());

    // Create operator with different topology
    let mut bad_operator: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    bad_operator
        .add_node("A".to_string(), HashSet::new())
        .unwrap();
    bad_operator
        .add_node("C".to_string(), HashSet::new())
        .unwrap(); // Different node name
    bad_operator
        .add_edge(&"A".to_string(), &"C".to_string())
        .unwrap();

    // Should fail
    let result = state.apply_operator_topology(&bad_operator);
    assert!(result.is_err());
}

#[test]
fn test_compatible_site_dimensions() {
    // Create two networks with same dimensions
    let mut net1: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    net1.add_node("A".to_string(), site1).unwrap();
    net1.add_node("B".to_string(), HashSet::new()).unwrap();
    net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let mut net2: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    // Different ID but same dimension
    let site2: HashSet<_> = [DynIndex::new_dyn(2)].into();
    net2.add_node("A".to_string(), site2).unwrap();
    net2.add_node("B".to_string(), HashSet::new()).unwrap();
    net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(net1.compatible_site_dimensions(&net2));

    // Create network with different dimension
    let mut net3: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site3: HashSet<_> = [DynIndex::new_dyn(5)].into(); // Different dim
    net3.add_node("A".to_string(), site3).unwrap();
    net3.add_node("B".to_string(), HashSet::new()).unwrap();
    net3.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(!net1.compatible_site_dimensions(&net3));
}
