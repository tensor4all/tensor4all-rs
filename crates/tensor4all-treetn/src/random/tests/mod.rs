use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

#[test]
fn test_random_treetn_f64_two_nodes() {
    // Use String as node name to avoid lifetime issues with &str
    let mut site_network = SiteIndexNetwork::<String, Index<DynId, TagSet>>::new();
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    site_network
        .add_node("A".to_string(), HashSet::from([i.clone()]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([j.clone()]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let treetn = random_treetn_f64(&mut rng, &site_network, LinkSpace::uniform(4));

    assert_eq!(treetn.node_count(), 2);
    assert_eq!(treetn.edge_count(), 1);
}

#[test]
fn test_random_treetn_c64_chain() {
    let mut site_network = SiteIndexNetwork::<i32, Index<DynId, TagSet>>::new();
    // Chain: 0 -- 1 -- 2
    for i in 0..3 {
        let idx = Index::new_dyn(2);
        site_network.add_node(i, HashSet::from([idx])).unwrap();
    }
    site_network.add_edge(&0, &1).unwrap();
    site_network.add_edge(&1, &2).unwrap();

    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let treetn = random_treetn_c64(&mut rng, &site_network, LinkSpace::uniform(3));

    assert_eq!(treetn.node_count(), 3);
    assert_eq!(treetn.edge_count(), 2);
}

#[test]
fn test_link_space_per_edge() {
    let mut site_network = SiteIndexNetwork::<String, Index<DynId, TagSet>>::new();
    site_network
        .add_node("A".to_string(), HashSet::new())
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::new())
        .unwrap();
    site_network
        .add_node("C".to_string(), HashSet::new())
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();

    let mut dims = HashMap::new();
    dims.insert(("A".to_string(), "B".to_string()), 5);
    dims.insert(("B".to_string(), "C".to_string()), 10);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let treetn = random_treetn_f64(&mut rng, &site_network, LinkSpace::per_edge(dims));

    assert_eq!(treetn.node_count(), 3);
    assert_eq!(treetn.edge_count(), 2);
}
