use super::*;

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

fn build_two_node_topology() -> NodeNameNetwork<String> {
    let mut net = NodeNameNetwork::new();
    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net
}

fn build_single_node_topology() -> NodeNameNetwork<String> {
    let mut net = NodeNameNetwork::new();
    net.add_node("A".to_string()).unwrap();
    net
}

fn build_y_topology() -> NodeNameNetwork<String> {
    let mut net = NodeNameNetwork::new();
    net.add_node("C".to_string()).unwrap();
    net.add_node("L0".to_string()).unwrap();
    net.add_node("L1".to_string()).unwrap();
    net.add_node("L2".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"L0".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"L1".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"L2".to_string()).unwrap();
    net
}

#[test]
fn test_subtree_oracle_chain() {
    let topo = build_chain_topology();
    let oracle = SubtreeOracle::new(&topo, &"A".to_string()).unwrap();

    assert!(oracle.is_target_on_a_side(&"A".to_string(), &"B".to_string(), &"A".to_string()));
    assert!(!oracle.is_target_on_a_side(&"A".to_string(), &"B".to_string(), &"B".to_string()));
    assert!(!oracle.is_target_on_a_side(&"A".to_string(), &"B".to_string(), &"C".to_string()));

    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"A".to_string()));
    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"B".to_string()));
    assert!(!oracle.is_target_on_a_side(&"B".to_string(), &"C".to_string(), &"C".to_string()));

    assert!(!oracle.is_target_on_a_side(&"B".to_string(), &"A".to_string(), &"A".to_string()));
    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"A".to_string(), &"B".to_string()));
    assert!(oracle.is_target_on_a_side(&"B".to_string(), &"A".to_string(), &"C".to_string()));
}

#[test]
fn test_subtree_oracle_longer_chain() {
    let topo = build_chain_topology_with_d();
    let oracle = SubtreeOracle::new(&topo, &"B".to_string()).unwrap();

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
fn test_swap_schedule_chain_swap_endpoints() {
    let topo = build_chain_topology();
    let current = HashMap::from([
        ("s_a".to_string(), "A".to_string()),
        ("s_c".to_string(), "C".to_string()),
    ]);
    let target = HashMap::from([
        ("s_a".to_string(), "C".to_string()),
        ("s_c".to_string(), "A".to_string()),
    ]);
    let root = "A".to_string();

    let schedule = SwapSchedule::build(&topo, &current, &target, &root).unwrap();

    assert_eq!(schedule.root, root);
    assert_eq!(schedule.steps.len(), 3);

    assert_eq!(schedule.steps[0].transport_path, Vec::<String>::new());
    assert_eq!(schedule.steps[0].node_a, "A");
    assert_eq!(schedule.steps[0].node_b, "B");
    assert_eq!(schedule.steps[0].a_side_sites, HashSet::new());
    assert_eq!(
        schedule.steps[0].b_side_sites,
        HashSet::from(["s_a".to_string()])
    );

    assert_eq!(schedule.steps[1].transport_path, Vec::<String>::new());
    assert_eq!(schedule.steps[1].node_a, "B");
    assert_eq!(schedule.steps[1].node_b, "C");
    assert_eq!(
        schedule.steps[1].a_side_sites,
        HashSet::from(["s_c".to_string()])
    );
    assert_eq!(
        schedule.steps[1].b_side_sites,
        HashSet::from(["s_a".to_string()])
    );

    assert_eq!(
        schedule.steps[2].transport_path,
        vec!["C".to_string(), "B".to_string()]
    );
    assert_eq!(schedule.steps[2].node_a, "B");
    assert_eq!(schedule.steps[2].node_b, "A");
    assert_eq!(schedule.steps[2].a_side_sites, HashSet::new());
    assert_eq!(
        schedule.steps[2].b_side_sites,
        HashSet::from(["s_c".to_string()])
    );
}

#[test]
fn test_swap_schedule_y_shape() {
    let topo = build_y_topology();
    let current = HashMap::from([
        ("s0".to_string(), "L0".to_string()),
        ("s1".to_string(), "L1".to_string()),
        ("s2".to_string(), "L2".to_string()),
    ]);
    let target = HashMap::from([
        ("s0".to_string(), "L1".to_string()),
        ("s1".to_string(), "L0".to_string()),
    ]);
    let root = "C".to_string();

    let schedule = SwapSchedule::build(&topo, &current, &target, &root).unwrap();

    assert_eq!(schedule.steps.len(), 3);

    assert_eq!(schedule.steps[0].transport_path, Vec::<String>::new());
    assert_eq!(schedule.steps[0].node_a, "C");
    assert_eq!(schedule.steps[0].node_b, "L1");
    assert_eq!(
        schedule.steps[0].a_side_sites,
        HashSet::from(["s1".to_string()])
    );
    assert_eq!(schedule.steps[0].b_side_sites, HashSet::new());

    assert_eq!(
        schedule.steps[1].transport_path,
        vec!["L1".to_string(), "C".to_string()]
    );
    assert_eq!(schedule.steps[1].node_a, "C");
    assert_eq!(schedule.steps[1].node_b, "L0");
    assert_eq!(
        schedule.steps[1].a_side_sites,
        HashSet::from(["s0".to_string()])
    );
    assert_eq!(
        schedule.steps[1].b_side_sites,
        HashSet::from(["s1".to_string()])
    );

    assert_eq!(
        schedule.steps[2].transport_path,
        vec!["L0".to_string(), "C".to_string()]
    );
    assert_eq!(schedule.steps[2].node_a, "C");
    assert_eq!(schedule.steps[2].node_b, "L1");
    assert_eq!(schedule.steps[2].a_side_sites, HashSet::new());
    assert_eq!(
        schedule.steps[2].b_side_sites,
        HashSet::from(["s0".to_string()])
    );

    assert!(schedule.steps.iter().all(|step| {
        !(step.node_a == "C" && step.node_b == "L2"
            || step.node_a == "L2" && step.node_b == "C")
    }));
}

#[test]
fn test_swap_schedule_no_op() {
    let topo = build_chain_topology();
    let current = HashMap::from([
        ("s0".to_string(), "A".to_string()),
        ("s1".to_string(), "B".to_string()),
    ]);
    let target = HashMap::from([
        ("s0".to_string(), "A".to_string()),
        ("s1".to_string(), "B".to_string()),
    ]);
    let root = "A".to_string();

    let schedule = SwapSchedule::build(&topo, &current, &target, &root).unwrap();
    assert!(schedule.steps.is_empty());
}

#[test]
fn test_swap_schedule_partial_assignment_keeps_unassigned_on_current_side() {
    let topo = build_two_node_topology();
    let current = HashMap::from([
        ("s0".to_string(), "A".to_string()),
        ("s1".to_string(), "B".to_string()),
    ]);
    let target = HashMap::from([("s0".to_string(), "B".to_string())]);
    let root = "A".to_string();

    let schedule = SwapSchedule::build(&topo, &current, &target, &root).unwrap();

    assert_eq!(schedule.steps.len(), 1);
    assert_eq!(schedule.steps[0].node_a, "A");
    assert_eq!(schedule.steps[0].node_b, "B");
    assert_eq!(schedule.steps[0].a_side_sites, HashSet::new());
    assert_eq!(
        schedule.steps[0].b_side_sites,
        HashSet::from(["s0".to_string(), "s1".to_string()])
    );
}

#[test]
fn test_swap_schedule_single_node() {
    let topo = build_single_node_topology();
    let current = HashMap::from([("s0".to_string(), "A".to_string())]);
    let target = HashMap::from([("s0".to_string(), "A".to_string())]);
    let root = "A".to_string();

    let schedule = SwapSchedule::build(&topo, &current, &target, &root).unwrap();
    assert!(schedule.steps.is_empty());
}

#[test]
fn test_swap_schedule_multi_site_node() {
    let topo = build_two_node_topology();
    let current = HashMap::from([
        ("s0".to_string(), "A".to_string()),
        ("s1".to_string(), "A".to_string()),
        ("s2".to_string(), "B".to_string()),
    ]);
    let target = HashMap::from([
        ("s0".to_string(), "B".to_string()),
        ("s2".to_string(), "A".to_string()),
    ]);
    let root = "A".to_string();

    let schedule = SwapSchedule::build(&topo, &current, &target, &root).unwrap();

    assert_eq!(schedule.steps.len(), 1);
    assert_eq!(
        schedule.steps[0].a_side_sites,
        HashSet::from(["s1".to_string(), "s2".to_string()])
    );
    assert_eq!(
        schedule.steps[0].b_side_sites,
        HashSet::from(["s0".to_string()])
    );
}

#[test]
fn test_swap_schedule_transit_node_zero_to_n_to_zero() {
    let topo = build_chain_topology();
    let current = HashMap::from([
        ("s0".to_string(), "A".to_string()),
        ("s1".to_string(), "A".to_string()),
    ]);
    let target = HashMap::from([
        ("s0".to_string(), "C".to_string()),
        ("s1".to_string(), "C".to_string()),
    ]);
    let root = "A".to_string();

    let schedule = SwapSchedule::build(&topo, &current, &target, &root).unwrap();

    assert_eq!(schedule.steps.len(), 2);
    assert_eq!(schedule.steps[0].node_a, "A");
    assert_eq!(schedule.steps[0].node_b, "B");
    assert_eq!(schedule.steps[0].a_side_sites, HashSet::new());
    assert_eq!(
        schedule.steps[0].b_side_sites,
        HashSet::from(["s0".to_string(), "s1".to_string()])
    );

    assert_eq!(schedule.steps[1].node_a, "B");
    assert_eq!(schedule.steps[1].node_b, "C");
    assert_eq!(schedule.steps[1].a_side_sites, HashSet::new());
    assert_eq!(
        schedule.steps[1].b_side_sites,
        HashSet::from(["s0".to_string(), "s1".to_string()])
    );
}

#[test]
fn test_swap_schedule_invalid_target_node() {
    let topo = build_chain_topology();
    let current = HashMap::from([("s0".to_string(), "A".to_string())]);
    let target = HashMap::from([("s0".to_string(), "Z".to_string())]);

    let res = SwapSchedule::build(&topo, &current, &target, &"A".to_string());
    assert!(res.is_err());
}

#[test]
fn test_swap_schedule_unknown_index_in_target() {
    let topo = build_chain_topology();
    let current = HashMap::from([("s0".to_string(), "A".to_string())]);
    let target = HashMap::from([("s1".to_string(), "B".to_string())]);

    let res = SwapSchedule::build(&topo, &current, &target, &"A".to_string());
    assert!(res.is_err());
}
