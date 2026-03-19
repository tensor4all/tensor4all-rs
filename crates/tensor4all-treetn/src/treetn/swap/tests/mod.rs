
use super::*;

use tensor4all_core::DynIndex;

fn build_chain_topology() -> NodeNameNetwork<String> {
    let mut net = NodeNameNetwork::new();
    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();
    net.add_node("C".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net
}

fn build_chain_topology_with_d() -> NodeNameNetwork<String> {
    let mut net = NodeNameNetwork::new();
    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();
    net.add_node("C".to_string()).unwrap();
    net.add_node("D".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();
    net
}

#[test]
fn test_subtree_oracle_chain() {
    // Chain A-B-C, rooted at A.
    // DFS from A: in_A=0, in_B=1, in_C=2, out_C=3, out_B=4, out_A=5
    let topo = build_chain_topology();
    let oracle = SubtreeOracle::new(&topo, &"A".to_string()).unwrap();

    // Edge (A, B): A-side = {A}, B-side = {B, C}
    assert!(oracle.is_target_on_a_side(&"A".to_string(), &"B".to_string(), &"A".to_string()));
    assert!(!oracle.is_target_on_a_side(&"A".to_string(), &"B".to_string(), &"B".to_string()));
    assert!(!oracle.is_target_on_a_side(&"A".to_string(), &"B".to_string(), &"C".to_string()));

    // Edge (B, C): B-side = {A, B}, C-side = {C}
    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"A".to_string()));
    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"B".to_string()));
    assert!(!oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"C".to_string()));

    // Reversed edge (B, A): B-side = {B, C}, A-side = {A}
    assert!(!oracle.is_target_on_a_side(&"B".to_string(), &"A".to_string(), &"A".to_string()));
    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"A".to_string(), &"B".to_string()));
    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"A".to_string(), &"C".to_string()));
}

#[test]
fn test_subtree_oracle_longer_chain() {
    // Chain A-B-C-D, rooted at B.
    let topo = build_chain_topology_with_d();
    let oracle = SubtreeOracle::new(&topo, &"B".to_string()).unwrap();

    // Edge (B, C): B-side = {A, B}, C-side = {C, D}
    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"A".to_string()));
    assert!(!oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"D".to_string()));
}

#[test]
fn test_subtree_oracle_unknown_root() {
    let topo = build_chain_topology();
    let res = SubtreeOracle::new(&topo, &"Z".to_string());
    assert!(res.is_err());
}

#[test]
fn test_swap_plan_has_swaps_at() {
    let topo = build_chain_topology();
    let ix = DynIndex::new_dyn(2);
    let mut current = HashMap::new();
    current.insert(ix.id().to_owned(), "A".to_string());
    let mut target = HashMap::new();
    target.insert(ix.id().to_owned(), "B".to_string());

    let plan = SwapPlan::<String, DynIndex>::new(&current, &target, &topo).unwrap();
    assert!(plan.has_swaps_at(&("A".to_string(), "B".to_string())));
    assert!(!plan.has_swaps_at(&("B".to_string(), "C".to_string())));
}

#[test]
fn test_swap_plan_no_move() {
    let topo = build_chain_topology();
    let mut current = HashMap::new();
    let ix = DynIndex::new_dyn(2);
    let iy = DynIndex::new_dyn(2);
    current.insert(ix.id().to_owned(), "A".to_string());
    current.insert(iy.id().to_owned(), "B".to_string());
    let target = HashMap::new();

    let plan = SwapPlan::<String, DynIndex>::new(&current, &target, &topo).unwrap();
    assert!(plan.edges_with_swaps().is_empty());
}

#[test]
fn test_swap_plan_two_node_swap() {
    let topo = build_chain_topology();
    let mut current = HashMap::new();
    let ix = DynIndex::new_dyn(2);
    let iy = DynIndex::new_dyn(2);
    current.insert(ix.id().to_owned(), "A".to_string());
    current.insert(iy.id().to_owned(), "B".to_string());
    let mut target = HashMap::new();
    target.insert(ix.id().to_owned(), "B".to_string());
    target.insert(iy.id().to_owned(), "A".to_string());

    let plan = SwapPlan::<String, DynIndex>::new(&current, &target, &topo).unwrap();
    let edges = plan.edges_with_swaps();
    assert_eq!(edges.len(), 1);
    assert!(edges.contains(&("A".to_string(), "B".to_string())));
}

#[test]
fn test_swap_plan_invalid_target_node() {
    let topo = build_chain_topology();
    let ix = DynIndex::new_dyn(2);
    let mut current = HashMap::new();
    current.insert(ix.id().to_owned(), "A".to_string());
    let mut target = HashMap::new();
    target.insert(ix.id().to_owned(), "Z".to_string()); // Z not in topology

    let res = SwapPlan::<String, DynIndex>::new(&current, &target, &topo);
    assert!(res.is_err());
}

#[test]
fn test_swap_plan_unknown_index_in_target() {
    let topo = build_chain_topology();
    let ix = DynIndex::new_dyn(2);
    let iy = DynIndex::new_dyn(2);
    let mut current = HashMap::new();
    current.insert(ix.id().to_owned(), "A".to_string());
    let mut target = HashMap::new();
    target.insert(iy.id().to_owned(), "B".to_string()); // iy not in current (different id)

    let res = SwapPlan::<String, DynIndex>::new(&current, &target, &topo);
    assert!(res.is_err());
}
