//! Unit tests of the error classification and the outcome mapping.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use tensor4all_core::ColMajorArray;
use tensor4all_treetn::NodeNameNetwork;

use super::*;
use crate::{TreeRsiDiagnostics, TreeRsiEdgeReport};

fn problem(cap: Option<usize>) -> InterpolationProblem<usize> {
    let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let mut topology = NodeNameNetwork::new();
    topology.add_node(0usize).unwrap();
    topology.add_node(1).unwrap();
    topology.add_edge(&0, &1).unwrap();
    InterpolationProblem::new(
        topology,
        BTreeMap::from([(0usize, vec![a]), (1, vec![b])]),
        ColMajorArray::new(vec![0, 0], vec![2, 1]).unwrap(),
        1e-8,
        cap.map(|cap| NonZeroUsize::new(cap).unwrap()),
        0,
    )
    .unwrap()
}

fn result(ranks: &[usize]) -> TreeRsiResult<usize> {
    TreeRsiResult {
        tree: TreeTN::new(),
        diagnostics: TreeRsiDiagnostics {
            root: 1,
            sketch_dim: 4,
            edges: ranks
                .iter()
                .map(|&rank| TreeRsiEdgeReport {
                    child: 0,
                    parent: 1,
                    rank,
                    rows: 4,
                    columns: 4,
                    exact_columns: false,
                    rank_limit_reached: false,
                    relative_pivot: 0.0,
                })
                .collect(),
            sketch_messages_per_input: 0,
            exact_messages_per_input: 0,
        },
    }
}

#[test]
fn only_a_rank_at_the_cap_is_bond_cap_reached() {
    use InterpolationTermination as I;
    let cases: [(&[usize], Option<usize>, I); 5] = [
        (&[3], None, I::ErrorNotMeasured),
        (&[3], Some(4), I::ErrorNotMeasured),
        (&[4], Some(4), I::BondCapReached),
        (&[2, 4], Some(4), I::BondCapReached),
        // A single node has no edge: never at the cap, even a cap of one.
        (&[], Some(1), I::ErrorNotMeasured),
    ];
    for (ranks, cap, expected) in cases {
        let outcome = outcome(result(ranks), &problem(cap));
        assert_eq!(outcome.termination, expected, "{ranks:?} {cap:?}");
        assert_eq!(outcome.error_estimate, f64::INFINITY);
        assert_eq!(outcome.max_sample_magnitude, 0.0);
        assert!(outcome.pivots.is_none());
    }
}

#[test]
fn errors_are_classified_by_their_cause() {
    for invalid in [
        TreeRsiError::NoInputs,
        TreeRsiError::EmptyTree { input: 0 },
        TreeRsiError::TopologyMismatch { input: 1 },
        TreeRsiError::PhysicalIndexMismatch {
            input: 1,
            node: "0".to_string(),
        },
        TreeRsiError::InvalidOption {
            option: "sketch_dim",
            message: "set sketch_dim when max_bond_dim is unbounded",
        },
        TreeRsiError::SizeOverflow { context: "points" },
    ] {
        assert!(matches!(
            classify(invalid),
            InterpolationError::InvalidProblem { .. }
        ));
    }
    let InterpolationError::Engine { source } = classify(TreeRsiError::Context {
        message: "refused".to_string(),
    }) else {
        panic!("a context failure is an engine error");
    };
    assert!(matches!(
        source.downcast_ref::<TreeRsiError>(),
        Some(TreeRsiError::Context { .. })
    ));
}
