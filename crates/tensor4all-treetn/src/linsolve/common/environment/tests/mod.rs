use std::collections::HashSet;

use tensor4all_core::DynIndex;

use super::*;

#[test]
fn test_environment_cache_creation() {
    // Compile-time test
}

#[test]
fn cached_topology_returns_neighbors_without_graph_lookup() {
    let mut network = SiteIndexNetwork::<String, DynIndex>::new();
    network
        .add_node("a".to_string(), HashSet::from([DynIndex::new_dyn(2)]))
        .unwrap();
    network
        .add_node("b".to_string(), HashSet::from([DynIndex::new_dyn(2)]))
        .unwrap();
    network
        .add_node("c".to_string(), HashSet::from([DynIndex::new_dyn(2)]))
        .unwrap();
    network
        .add_edge(&"a".to_string(), &"b".to_string())
        .unwrap();
    network
        .add_edge(&"a".to_string(), &"c".to_string())
        .unwrap();

    let topology = CachedTopology::from_site_index_network(&network);
    let mut neighbors = topology.neighbors(&"a".to_string()).collect::<Vec<_>>();
    neighbors.sort();

    assert_eq!(neighbors, vec!["b".to_string(), "c".to_string()]);
    assert!(topology.neighbors(&"missing".to_string()).next().is_none());
}
