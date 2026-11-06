use super::*;
use tensor4all_core::{SingularValueMeasure, ThresholdScale, TruncationRule};

#[test]
fn tdvp_benchmark_uses_itensors_cutoff_strategy() {
    let policy = itensor_cutoff_policy();
    assert_eq!(policy.threshold, ITENSOR_CUTOFF);
    assert_eq!(policy.scale, ThresholdScale::Relative);
    assert_eq!(policy.measure, SingularValueMeasure::SquaredValue);
    assert_eq!(policy.rule, TruncationRule::DiscardedTailSum);
}

#[test]
fn star_benchmark_uses_itensornetworks_leaf_root() {
    assert_eq!(tdvp_root_name(Topology::Chain), node_name(0));
    assert_eq!(tdvp_root_name(Topology::Star), node_name(1));
}

#[test]
fn topology_selection_parses_all_chain_and_star() {
    assert!(matches!(
        parse_topology_selection(None).unwrap(),
        TopologySelection::All
    ));
    assert!(matches!(
        parse_topology_selection(Some("chain")).unwrap(),
        TopologySelection::One(Topology::Chain)
    ));
    assert!(matches!(
        parse_topology_selection(Some("star")).unwrap(),
        TopologySelection::One(Topology::Star)
    ));
    assert!(parse_topology_selection(Some("line")).is_err());
}

#[test]
fn tdvp_benchmark_options_set_requested_solver_caps() {
    let options = benchmark_tdvp_options(4, 0.02);
    assert_eq!(options.nsite, 2);
    assert_eq!(options.order, 2);
    assert_eq!(options.nsweeps, 4);
    assert_eq!(options.max_bond_dim, Some(32));
    assert_eq!(options.krylov.max_iter, 30);
    assert_eq!(options.krylov.max_time_splits, 100);
    assert_eq!(options.krylov.tol, 1.0e-12);
}
