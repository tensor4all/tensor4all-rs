//! Heavy regression witness, scheduled instead of running on every push.
//!
//! The r=9 low-temperature G0 case takes ~23 minutes in the `ci` profile
//! (~95 minutes under coverage instrumentation) on the 4-vCPU CI runner, which is
//! why it lives in `heavy_*.rs` and is `#[ignore]`d. The fast half of the same
//! regression (r=3,4,5 over three topologies) stays in `g0_convergence.rs` and runs
//! on every push. CONTRIBUTING.md, "CI time budget", has the selection command and
//! the budgets this file stays outside of.

#[path = "support/g0.rs"]
mod g0;

use num_complex::Complex64;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treeaci::{tree_elementwise_batched, TreeAciTermination};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn low_temperature_branch_convergence_does_not_hide_growth_on_smaller_cuts() -> g0::TestResult<()> {
    let r = 9;
    let (sites, inputs) = g0::fixture(r, "cttn")?;
    let options = g0::options();
    let result = tree_elementwise_batched(g0::operator, &inputs, &options)?;
    assert_eq!(result.termination, TreeAciTermination::Converged);
    // Independent witnesses from #741 and their reflection partners. Before
    // the fix the first point has absolute error 0.247, despite Converged.
    let points = [
        (292, 70, 256),
        (292, 442, 256),
        (242, 70, 256),
        (230, 442, 256),
        (243, 454, 256),
    ];
    let mut coordinates = Vec::with_capacity(points.len() * 3 * r);
    for &(x, y, n) in &points {
        for bit in 0..r {
            for value in [x, y, n] {
                coordinates.push((value >> (r - 1 - bit)) & 1);
            }
        }
    }
    let mut evaluator =
        TreeTNCachedEvaluator::new(&result.tree, &sites, CachedEvaluatorOptions::default())?;
    let actual = evaluator.evaluate_batched_typed::<Complex64>(
        ColMajorArrayRef::new(&coordinates, &[3 * r, points.len()])?,
        EvaluationHint::default(),
    )?;
    let error = actual
        .iter()
        .zip(&points)
        .map(|(&value, &(x, y, n))| (value - g0::exact(r, x, y, n)).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        error <= options.tolerance * options.global_tolerance_margin,
        "max absolute witness residual={error:e}"
    );
    Ok(())
}
