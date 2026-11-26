//! Point-cost and accuracy acceptance for streaming leaf-side enumeration.

use crate::common::comb::{run_with_stop_rule, Target, TOLERANCE};
use anyhow::Result;
use tensor4all_treetci::{DefaultProposer, StopRule, TreeTciTermination};

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_comb_point_costs() -> Result<()> {
    for rule in [StopRule::Sampled, StopRule::PerBond] {
        for delta in [1.0, 4.0] {
            let run =
                run_with_stop_rule(Target::Oscillating { delta }, true, &DefaultProposer, rule)?;
            println!(
                "issue1441 comb delta={delta} seed=11 rule={rule:?} points={} iterations={} ranks={:?} termination={:?} error={:e}",
                run.result.evaluation.evaluated_points, run.result.errors.len(),
                run.result.ranks, run.result.termination, run.max_rel_error,
            );
            assert_eq!(run.result.termination, TreeTciTermination::Converged);
            assert!(run.max_rel_error < 10.0 * TOLERANCE);
        }
    }
    Ok(())
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_reopened_table_point_costs() -> Result<()> {
    crate::issue1352_rank_deficient::check_reopened_seeds(
        1.0_f64,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    )
}
