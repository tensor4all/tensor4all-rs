use super::*;

fn chain_abc() -> NodeNameNetwork<&'static str> {
    let mut network = NodeNameNetwork::new();
    network.add_node("A").unwrap();
    network.add_node("B").unwrap();
    network.add_node("C").unwrap();
    network.add_edge(&"A", &"B").unwrap();
    network.add_edge(&"B", &"C").unwrap();
    network
}

fn star_abcd() -> NodeNameNetwork<&'static str> {
    let mut network = NodeNameNetwork::new();
    for node in ["A", "B", "C", "D"] {
        network.add_node(node).unwrap();
    }
    network.add_edge(&"A", &"B").unwrap();
    network.add_edge(&"A", &"C").unwrap();
    network.add_edge(&"A", &"D").unwrap();
    network
}

#[test]
fn reverse_regions_rejects_empty_nodes() {
    let step = TdvpRegionStep {
        nodes: Vec::<&str>::new(),
        new_center: "unused",
        exponent_step: Complex64::new(0.0, -0.2),
        kind: TdvpRegionKind::OneSite,
    };

    assert!(reverse_regions(&[step]).is_none());
}

#[test]
fn applyexp_sub_steps_match_itensornetworks_orders() {
    assert_eq!(applyexp_sub_steps(1).unwrap(), vec![1.0]);
    assert_eq!(applyexp_sub_steps(2).unwrap(), vec![0.5, 0.5]);

    let fourth = applyexp_sub_steps(4).unwrap();
    assert_eq!(fourth.len(), 6);
    let total: f64 = fourth.iter().sum();
    assert!((total - 1.0).abs() < 1.0e-14);
    assert!(applyexp_sub_steps(3).is_none());
}

#[test]
fn two_site_plan_starts_at_root_edge_with_corrections_except_last_edge() {
    let network = chain_abc();
    let exponent = Complex64::new(0.0, -0.2);
    let plan = TdvpRegionPlan::new(&network, &"B", 2, 1, exponent).unwrap();

    let regions: Vec<_> = plan
        .steps
        .iter()
        .map(|step| (step.kind, step.nodes.clone(), step.exponent_step))
        .collect();
    // The first region contains the sweep root B; the walk moves outward.
    assert_eq!(
        regions,
        vec![
            (TdvpRegionKind::TwoSite, vec!["C", "B"], exponent),
            (TdvpRegionKind::SiteCorrection, vec!["B"], -exponent),
            (TdvpRegionKind::TwoSite, vec!["B", "A"], exponent),
        ]
    );
}

#[test]
fn second_order_plan_reverses_even_substep_regions() {
    let network = chain_abc();
    let exponent = Complex64::new(0.0, -0.2);
    let plan = TdvpRegionPlan::new(&network, &"B", 2, 2, exponent).unwrap();

    let regions: Vec<_> = plan
        .steps
        .iter()
        .map(|step| (step.kind, step.nodes.clone(), step.exponent_step))
        .collect();
    let half = exponent * 0.5;
    assert_eq!(
        regions,
        vec![
            (TdvpRegionKind::TwoSite, vec!["C", "B"], half),
            (TdvpRegionKind::SiteCorrection, vec!["B"], -half),
            (TdvpRegionKind::TwoSite, vec!["B", "A"], half),
            (TdvpRegionKind::TwoSite, vec!["A", "B"], half),
            (TdvpRegionKind::SiteCorrection, vec!["B"], -half),
            (TdvpRegionKind::TwoSite, vec!["B", "C"], half),
        ]
    );
}

#[test]
fn two_site_star_plan_walks_outward_from_root_leaf() {
    // Regression test for the root-edge-first ordering (issue #560): on a
    // branching tree the half-sweep must start at the edge containing the
    // sweep root and keep every subsequent region attached to an
    // already-visited vertex, with the site correction on the overlap.
    let network = star_abcd();
    let exponent = Complex64::new(0.0, -0.2);
    let plan = TdvpRegionPlan::new(&network, &"B", 2, 1, exponent).unwrap();

    let regions: Vec<_> = plan
        .steps
        .iter()
        .map(|step| (step.kind, step.nodes.clone(), step.exponent_step))
        .collect();
    assert_eq!(
        regions,
        vec![
            (TdvpRegionKind::TwoSite, vec!["B", "A"], exponent),
            (TdvpRegionKind::SiteCorrection, vec!["A"], -exponent),
            (TdvpRegionKind::TwoSite, vec!["D", "A"], exponent),
            (TdvpRegionKind::SiteCorrection, vec!["A"], -exponent),
            (TdvpRegionKind::TwoSite, vec!["A", "C"], exponent),
        ]
    );
}

#[test]
fn two_site_plan_keeps_every_region_attached_to_visited_vertices() {
    // Two branches of depth 2 force a jump between subtrees; even then the
    // walk must stay contiguous: the first region contains the sweep root
    // and every later region shares a vertex with the already-visited set.
    let mut network = NodeNameNetwork::new();
    for node in ["A", "B", "C", "D", "E"] {
        network.add_node(node).unwrap();
    }
    network.add_edge(&"A", &"B").unwrap();
    network.add_edge(&"B", &"C").unwrap();
    network.add_edge(&"A", &"D").unwrap();
    network.add_edge(&"D", &"E").unwrap();

    let exponent = Complex64::new(0.0, -0.2);
    let plan = TdvpRegionPlan::new(&network, &"A", 2, 1, exponent).unwrap();

    let two_site_regions: Vec<Vec<&str>> = plan
        .steps
        .iter()
        .filter(|step| step.kind == TdvpRegionKind::TwoSite)
        .map(|step| step.nodes.clone())
        .collect();
    assert_eq!(two_site_regions.len(), 4);
    assert!(two_site_regions[0].contains(&"A"));
    let mut visited: std::collections::HashSet<&str> =
        two_site_regions[0].iter().copied().collect();
    for region in &two_site_regions[1..] {
        assert!(
            region.iter().any(|node| visited.contains(node)),
            "region {region:?} is detached from visited set {visited:?}"
        );
        visited.extend(region.iter().copied());
    }
}

#[test]
fn one_site_plan_uses_post_order_vertices() {
    let network = star_abcd();
    let exponent = Complex64::new(0.0, -0.2);
    let plan = TdvpRegionPlan::new(&network, &"A", 1, 1, exponent).unwrap();

    assert_eq!(plan.nsite, 1);
    assert_eq!(plan.order, 1);
    assert_eq!(
        plan.steps
            .iter()
            .map(|step| (step.kind, step.nodes.clone()))
            .collect::<Vec<_>>(),
        vec![
            (TdvpRegionKind::OneSite, vec!["B"]),
            (TdvpRegionKind::OneSite, vec!["C"]),
            (TdvpRegionKind::OneSite, vec!["D"]),
            (TdvpRegionKind::OneSite, vec!["A"]),
        ]
    );
}
