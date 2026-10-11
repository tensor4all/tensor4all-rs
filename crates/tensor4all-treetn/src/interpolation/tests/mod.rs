use super::*;
use tensor4all_core::{Index, TagSet};

fn network(nodes: &[usize], edges: &[(usize, usize)]) -> NodeNameNetwork<usize> {
    let mut topology = NodeNameNetwork::new();
    for &node in nodes {
        topology.add_node(node).unwrap();
    }
    for (a, b) in edges {
        topology.add_edge(a, b).unwrap();
    }
    topology
}

/// Chain 0 - 1 - 2 (max degree 2); node 1 has two sites, node 2 none.
fn chain_parts() -> (
    NodeNameNetwork<usize>,
    BTreeMap<usize, Vec<DynIndex>>,
    Vec<DynIndex>,
) {
    let sites = vec![
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
    ];
    let node_sites = BTreeMap::from([
        (0, vec![sites[0].clone()]),
        (1, vec![sites[1].clone(), sites[2].clone()]),
        (2, vec![]),
    ]);
    (network(&[0, 1, 2], &[(0, 1), (1, 2)]), node_sites, sites)
}

fn pivots(data: Vec<usize>, shape: Vec<usize>) -> ColMajorArray<usize> {
    ColMajorArray::new(data, shape).unwrap()
}

fn expect_invalid<V>(result: Result<InterpolationProblem<V>, InterpolationError>, needle: &str)
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    match result {
        Err(InterpolationError::InvalidProblem { message }) => assert!(
            message.contains(needle),
            "message {message:?} does not mention {needle:?}"
        ),
        other => panic!("expected InvalidProblem mentioning {needle:?}, got {other:?}"),
    }
}

#[test]
fn new_accepts_valid_chain_and_exposes_fields() {
    let (topology, node_sites, sites) = chain_parts();
    let problem = InterpolationProblem::new(
        topology,
        node_sites.clone(),
        pivots(vec![1, 2, 0, 0, 1, 1], vec![3, 2]),
        1e-9,
        NonZeroUsize::new(5),
        11,
    )
    .unwrap();
    assert_eq!(problem.site_order(), &sites[..]);
    assert_eq!(problem.node_sites(), &node_sites);
    assert_eq!(problem.topology().node_count(), 3);
    assert_eq!(problem.initial_pivots().shape(), &[3, 2]);
    assert_eq!(problem.absolute_tolerance(), 1e-9);
    assert_eq!(problem.max_bond_dim(), NonZeroUsize::new(5));
    assert_eq!(problem.seed(), 11);
}

#[test]
fn derive_site_order_uses_ascending_node_names_and_given_site_order() {
    let (x, y, z) = (
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    );
    let node_sites = BTreeMap::from([
        (9usize, vec![x.clone()]),
        (1, vec![z.clone(), y.clone()]),
        (4, vec![]),
    ]);
    assert_eq!(
        InterpolationProblem::derive_site_order(&node_sites),
        vec![z, y, x]
    );
}

#[test]
fn new_accepts_same_id_sites_that_differ_by_prime_level_or_tags() {
    let base = DynIndex::new_dyn(2);
    let primed = base.prime();
    let tagged = Index::new_with_tags(base.id, 2, TagSet::from_str("Site").unwrap());
    let node_sites = BTreeMap::from([
        (0usize, vec![base.clone(), primed.clone()]),
        (1, vec![tagged.clone()]),
    ]);
    let problem = InterpolationProblem::new(
        network(&[0, 1], &[(0, 1)]),
        node_sites,
        pivots(vec![0, 1, 1], vec![3, 1]),
        0.0,
        None,
        0,
    )
    .unwrap();
    assert_eq!(problem.site_order(), &[base, primed, tagged][..]);
}

#[test]
fn new_rejects_topology_node_set_mismatch() {
    let (_, node_sites, _) = chain_parts();
    // Fewer topology nodes than node_sites entries.
    expect_invalid(
        InterpolationProblem::new(
            network(&[0, 1], &[(0, 1)]),
            node_sites.clone(),
            pivots(vec![0, 0, 0], vec![3, 1]),
            0.0,
            None,
            0,
        ),
        "node_sites has 3 entries",
    );
    // Same count, different names.
    expect_invalid(
        InterpolationProblem::new(
            network(&[0, 1, 7], &[(0, 1), (1, 7)]),
            node_sites,
            pivots(vec![0, 0, 0], vec![3, 1]),
            0.0,
            None,
            0,
        ),
        "not in the topology",
    );
}

#[test]
fn new_rejects_empty_topology() {
    expect_invalid(
        InterpolationProblem::<usize>::new(
            NodeNameNetwork::new(),
            BTreeMap::new(),
            pivots(vec![], vec![0, 1]),
            0.0,
            None,
            0,
        ),
        "no nodes",
    );
}

#[test]
fn new_rejects_non_tree_topologies() {
    let (_, node_sites, _) = chain_parts();
    // Too few edges.
    expect_invalid(
        InterpolationProblem::new(
            network(&[0, 1, 2], &[(0, 1)]),
            node_sites.clone(),
            pivots(vec![0, 0, 0], vec![3, 1]),
            0.0,
            None,
            0,
        ),
        "has 2 edges, got 1",
    );
    // Too many edges (a cycle).
    expect_invalid(
        InterpolationProblem::new(
            network(&[0, 1, 2], &[(0, 1), (1, 2), (2, 0)]),
            node_sites.clone(),
            pivots(vec![0, 0, 0], vec![3, 1]),
            0.0,
            None,
            0,
        ),
        "has 2 edges, got 3",
    );
    // Right edge count but disconnected: a self-loop on node 0.
    expect_invalid(
        InterpolationProblem::new(
            network(&[0, 1, 2], &[(0, 0), (1, 2)]),
            node_sites,
            pivots(vec![0, 0, 0], vec![3, 1]),
            0.0,
            None,
            0,
        ),
        "not connected",
    );
}

#[test]
fn new_rejects_invalid_sites() {
    // No active site at all.
    expect_invalid(
        InterpolationProblem::new(
            network(&[0, 1], &[(0, 1)]),
            BTreeMap::from([(0usize, vec![]), (1, vec![])]),
            pivots(vec![], vec![0, 1]),
            0.0,
            None,
            0,
        ),
        "no active site",
    );
    // Dimension zero.
    expect_invalid(
        InterpolationProblem::new(
            network(&[0], &[]),
            BTreeMap::from([(0usize, vec![DynIndex::new_dyn(0)])]),
            pivots(vec![0], vec![1, 1]),
            0.0,
            None,
            0,
        ),
        "dimension zero",
    );
    // The same full index on two nodes.
    let shared = DynIndex::new_dyn(2);
    expect_invalid(
        InterpolationProblem::new(
            network(&[0, 1], &[(0, 1)]),
            BTreeMap::from([(0usize, vec![shared.clone()]), (1, vec![shared])]),
            pivots(vec![0, 0], vec![2, 1]),
            0.0,
            None,
            0,
        ),
        "more than once",
    );
}

#[test]
fn validate_layout_matches_the_layout_checks_of_new() {
    let (chain, chain_sites, _) = chain_parts();
    let shared = DynIndex::new_dyn(2);
    let cases = vec![
        (
            network(&[0, 1], &[(0, 1)]),
            chain_sites.clone(),
            "node_sites has 3 entries",
        ),
        (
            network(&[0, 1, 7], &[(0, 1), (1, 7)]),
            chain_sites.clone(),
            "not in the topology",
        ),
        (NodeNameNetwork::new(), BTreeMap::new(), "no nodes"),
        (
            network(&[0, 1, 2], &[(0, 1)]),
            chain_sites.clone(),
            "has 2 edges, got 1",
        ),
        (
            network(&[0, 1, 2], &[(0, 0), (1, 2)]),
            chain_sites.clone(),
            "not connected",
        ),
        (
            network(&[0, 1], &[(0, 1)]),
            BTreeMap::from([(0usize, vec![]), (1, vec![])]),
            "no active site",
        ),
        (
            network(&[0], &[]),
            BTreeMap::from([(0usize, vec![DynIndex::new_dyn(0)])]),
            "dimension zero",
        ),
        (
            network(&[0, 1], &[(0, 1)]),
            BTreeMap::from([(0usize, vec![shared.clone()]), (1, vec![shared])]),
            "more than once",
        ),
    ];
    for (topology, node_sites, needle) in cases {
        let layout = validate_layout(&topology, &node_sites);
        let Err(InterpolationError::InvalidProblem { message }) = layout else {
            panic!("validate_layout accepted a layout that must fail with {needle:?}");
        };
        assert!(message.contains(needle), "{message:?} lacks {needle:?}");
        let n_sites = InterpolationProblem::derive_site_order(&node_sites).len();
        match InterpolationProblem::new(
            topology,
            node_sites,
            pivots(vec![0; n_sites], vec![n_sites, 1]),
            0.0,
            None,
            0,
        ) {
            Err(InterpolationError::InvalidProblem { message: from_new }) => {
                assert_eq!(from_new, message)
            }
            other => panic!("new must fail like validate_layout, got {other:?}"),
        }
    }
    assert!(validate_layout(&chain, &chain_sites).is_ok());
}

#[test]
fn new_rejects_invalid_initial_pivots() {
    let cases = [
        (pivots(vec![0, 0, 0], vec![3]), "must be a 2D array"),
        (pivots(vec![0, 0], vec![2, 1]), "has 2 rows"),
        (pivots(vec![], vec![3, 0]), "at least one pivot"),
        // Row 1 has dimension 3; coordinate 3 is out of range.
        (
            pivots(vec![0, 0, 0, 1, 3, 1], vec![3, 2]),
            "coordinate 3 at row 1",
        ),
    ];
    for (initial_pivots, needle) in cases {
        let (topology, node_sites, _) = chain_parts();
        expect_invalid(
            InterpolationProblem::new(topology, node_sites, initial_pivots, 0.0, None, 0),
            needle,
        );
    }
}

#[test]
fn new_rejects_invalid_tolerance() {
    for tolerance in [-1e-12, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (topology, node_sites, _) = chain_parts();
        expect_invalid(
            InterpolationProblem::new(
                topology,
                node_sites,
                pivots(vec![0, 0, 0], vec![3, 1]),
                tolerance,
                None,
                0,
            ),
            "absolute_tolerance",
        );
    }
}

#[test]
fn error_displays_and_sources_are_stable() {
    let invalid = InterpolationError::InvalidProblem {
        message: "bad".to_string(),
    };
    assert_eq!(invalid.to_string(), "invalid interpolation problem: bad");
    assert!(std::error::Error::source(&invalid).is_none());

    let evaluator = InterpolationError::Evaluator {
        source: anyhow::anyhow!("root"),
    };
    assert_eq!(evaluator.to_string(), "evaluator failed: root");
    assert_eq!(
        std::error::Error::source(&evaluator).unwrap().to_string(),
        "root"
    );

    assert_eq!(
        InterpolationError::AllSamplesZero.to_string(),
        "every initial pivot evaluates to zero"
    );

    let engine = InterpolationError::Engine {
        source: anyhow::anyhow!("inner"),
    };
    assert_eq!(engine.to_string(), "interpolation engine failed: inner");
    assert_eq!(
        std::error::Error::source(&engine).unwrap().to_string(),
        "inner"
    );
}
