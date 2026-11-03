use super::*;
use tensor4all_core::{SingularValueMeasure, ThresholdScale, TruncationRule};

#[test]
fn dmrg_benchmark_uses_itensors_cutoff_strategy() {
    let policy = itensor_cutoff_policy();
    assert_eq!(policy.threshold, ITENSOR_CUTOFF);
    assert_eq!(policy.scale, ThresholdScale::Relative);
    assert_eq!(policy.measure, SingularValueMeasure::SquaredValue);
    assert_eq!(policy.rule, TruncationRule::DiscardedTailSum);
}

#[test]
fn star_benchmark_uses_itensornetworks_leaf_root() {
    assert_eq!(dmrg_root_name(Topology::Chain), node_name(0));
    assert_eq!(dmrg_root_name(Topology::Star), node_name(1));
}

#[test]
fn dmrg_benchmark_runs_all_requested_sweeps() {
    let options = benchmark_dmrg_options(4);
    assert_eq!(options.nsweeps, 4);
    assert_eq!(options.energy_tol, None);
}
