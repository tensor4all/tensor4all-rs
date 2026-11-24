//! Heavy #1371 regressions for the adaptive local threshold on the sampled
//! stop path, scheduled instead of running on every push.
//!
//! The full 18-run reproducer (six seeds at three tolerances) and the 60-site
//! slow-decay comb together need about 30 million oracle points, too many for
//! the per-push suite in the unoptimized profile. Three of the 18 runs stay in
//! `issue1371_local_threshold.rs`. CONTRIBUTING.md, "CI time budget", has the
//! selection command and the budgets this file stays outside of.
//!
//! All runs select `StopRule::Sampled`; under the default `PerBond` the
//! threshold does not adapt (#1405).
//!
//! Before the adaptive threshold (main at 050eb4ba, the behaviour `Sampled`
//! kept until this change), the 18 runs took 5 to 30
//! iterations: two ended `MaxIterations`, ten of the sixteen `Converged` runs
//! had a dense error above the tolerance (up to 2.04x), and the oracle points
//! summed to 1,362,136 / 2,135,123 / 1,641,678 at 1e-4 / 1e-6 / 1e-8. On the
//! comb, all three seeds ran into `MaxIterations` at 30 iterations with 59 to
//! 62 million points and a sampled error of 1.26 to 2.28x the tolerance.

use crate::common::issue1371::{
    comb60_graph, comb60_target, dense_relative_error, gaussian_ridge, run_tree,
    sampled_relative_error,
};
use tensor4all_treetci::{StopRule, TreeTciGraph, TreeTciOptions, TreeTciTermination};

/// The reproducer's options under `StopRule::Sampled`, the only rule that
/// adapts the local threshold (#1405).
fn options(tolerance: f64, seed: u64) -> TreeTciOptions {
    TreeTciOptions {
        tolerance,
        max_iter: 30,
        seed: Some(seed),
        stop_rule: StopRule::Sampled,
        ..Default::default()
    }
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_issue1371_all_eighteen_runs_converge_within_tolerance() {
    // Measured with the adaptive threshold: every run `Converged` in 4 to 6
    // iterations, dense error 0.22 to 0.50x the tolerance, final factor 1/4
    // (1/2 for 1e-6 seed 4), points summing to 540,563 / 647,757 / 892,792.
    // Oracle points summed over the six seeds before the change.
    for (tolerance, before) in [(1e-4, 1_362_136), (1e-6, 2_135_123), (1e-8, 1_641_678)] {
        let mut points = 0;
        for seed in 1..=6 {
            let result = run_tree(
                gaussian_ridge,
                vec![2; 16],
                TreeTciGraph::linear_chain(16).unwrap(),
                options(tolerance, seed),
            );
            let context = format!("tolerance {tolerance:e}, seed {seed}");
            // Every run, without exception: `Converged`, the threshold
            // tightened but not down to the floor, and a dense error within
            // the tolerance.
            assert_eq!(
                result.termination,
                TreeTciTermination::Converged,
                "{context}"
            );
            assert!(result.ranks.len() <= 10, "{context}: {:?}", result.ranks);
            assert!(
                result.local_threshold_factor < 1.0 && result.local_threshold_factor > 1.0 / 15.0,
                "{context}: factor {}",
                result.local_threshold_factor
            );
            let error = dense_relative_error(&result, gaussian_ridge, 16, 2);
            assert!(error <= tolerance, "{context}: dense error {error:e}");
            points += result.evaluation.evaluated_points;
        }
        assert!(points < before, "tolerance {tolerance:e}: {points} points");
    }
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_issue1371_comb60_converges_within_tolerance() {
    // Measured with the adaptive threshold at 1e-8: `Converged` in 5 / 5 / 8
    // iterations, 7,581,824 / 7,271,185 / 14,910,349 points, sampled error
    // 0.57 / 0.39 / 0.45x the tolerance, final factor 1/4.
    let tolerance = 1e-8;
    for seed in 1..=3 {
        let result = run_tree(
            comb60_target,
            vec![2; 60],
            comb60_graph(),
            options(tolerance, seed),
        );
        let context = format!("seed {seed}");
        assert_eq!(
            result.termination,
            TreeTciTermination::Converged,
            "{context}"
        );
        assert!(result.ranks.len() <= 10, "{context}: {:?}", result.ranks);
        assert!(result.local_threshold_factor < 1.0, "{context}");
        let error = sampled_relative_error(&result, comb60_target, 60);
        assert!(error <= tolerance, "{context}: sampled error {error:e}");
    }
}
