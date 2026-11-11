//! Unit tests of the error classification and the outcome mapping.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use tensor4all_core::ColMajorArray;
use tensor4all_treetn::NodeNameNetwork;

use super::*;
use crate::TreeAciDiagnostics;

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

fn result(termination: TreeAciTermination, rank: usize) -> TreeAciResult<usize> {
    TreeAciResult {
        tree: TreeTN::new(),
        max_ranks: vec![rank],
        max_errors: vec![3e-9, 2e-9],
        max_sample_magnitude: 5.0,
        global_pivots_found: vec![0],
        termination,
        diagnostics: TreeAciDiagnostics {
            edge_ranks: vec![(0, 1, rank)],
            ..TreeAciDiagnostics::default()
        },
    }
}

#[test]
fn terminations_map_to_the_engine_contract() {
    use InterpolationTermination as I;
    use TreeAciTermination as T;
    let cases = [
        (T::Converged, 3, None, I::Converged),
        (T::Converged, 3, Some(4), I::Converged),
        (T::Converged, 4, Some(4), I::BondCapReached),
        (T::RankLimited, 2, Some(4), I::BondCapReached),
        (T::MaxSweeps, 2, Some(4), I::IterationLimit),
        (T::MaxSweeps, 4, Some(4), I::IterationLimit),
    ];
    for (termination, rank, cap, expected) in cases {
        let outcome = outcome(result(termination, rank), &problem(cap));
        assert_eq!(
            outcome.termination, expected,
            "{termination:?} {rank} {cap:?}"
        );
        assert_eq!(outcome.error_estimate, 2e-9);
        assert_eq!(outcome.max_sample_magnitude, 5.0);
        assert!(outcome.pivots.is_none());
    }
}

#[test]
fn a_run_without_sweeps_reports_a_zero_error_estimate() {
    let mut exact = result(TreeAciTermination::Converged, 0);
    exact.max_errors.clear();
    exact.diagnostics.edge_ranks.clear();
    let outcome = outcome(exact, &problem(Some(1)));
    assert_eq!(outcome.error_estimate, 0.0);
    assert_eq!(outcome.termination, InterpolationTermination::Converged);
}

#[test]
fn errors_are_classified_by_their_cause() {
    let operator = classify(
        TreeAciError::Callback {
            message: "boom".to_string(),
        },
        Some(anyhow::anyhow!("boom")),
    );
    assert!(matches!(operator, InterpolationError::Evaluator { .. }));
    // A callback error that the adapter did not record is tree ACI's own.
    let unrecorded = classify(
        TreeAciError::Callback {
            message: "boom".to_string(),
        },
        None,
    );
    assert!(matches!(unrecorded, InterpolationError::Engine { .. }));
    for invalid in [
        TreeAciError::NoInputs,
        TreeAciError::EmptyTree { input: 0 },
        TreeAciError::TopologyMismatch { input: 1 },
        TreeAciError::PhysicalIndexMismatch {
            input: 1,
            node: "0".to_string(),
        },
        TreeAciError::SizeOverflow { context: "points" },
    ] {
        assert!(matches!(
            classify(invalid, None),
            InterpolationError::InvalidProblem { .. }
        ));
    }
    let numerical = classify(
        TreeAciError::Numerical {
            message: "singular".to_string(),
        },
        None,
    );
    let InterpolationError::Engine { source } = numerical else {
        panic!("a numerical failure is an engine error");
    };
    assert!(matches!(
        source.downcast_ref::<TreeAciError>(),
        Some(TreeAciError::Numerical { .. })
    ));
}

#[test]
fn the_tolerance_factor_divides_the_problems_tolerance() {
    let problem = problem(Some(4));
    let plain = TreeAciInterpolator::default();
    assert_eq!(plain.run_options(&problem).tolerance, 1e-8 / 3.0);
    let untightened = TreeAciInterpolator::default()
        .with_tolerance_factor(1.0)
        .unwrap();
    assert_eq!(untightened.run_options(&problem).tolerance, 1e-8);
    let tightened = TreeAciInterpolator::default()
        .with_tolerance_factor(4.0)
        .unwrap();
    let options = tightened.run_options(&problem);
    assert_eq!(options.tolerance, 1e-8 / 4.0);
    assert!(!options.scale_tolerance);
    assert_eq!(options.max_bond_dim, Some(4));
}

#[test]
fn a_tolerance_factor_below_one_or_not_finite_is_rejected() {
    for factor in [0.0, 0.5, -1.0, f64::NAN, f64::INFINITY] {
        let error = TreeAciInterpolator::default()
            .with_tolerance_factor(factor)
            .unwrap_err();
        assert!(
            matches!(
                error,
                TreeAciError::InvalidOption {
                    option: "tolerance_factor",
                    ..
                }
            ),
            "{factor}"
        );
    }
    assert!(TreeAciInterpolator::default()
        .with_tolerance_factor(1.0)
        .is_ok());
}
