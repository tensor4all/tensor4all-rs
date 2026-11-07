//! Regression test for issue #1153: bonds into a degree-3 vertex of local
//! dimension 1 must not stay pinned at rank 1 while the run reports
//! `Converged`.
//!
//! The tree is the comb of [`crate::common::comb`]. Global pivots are
//! disabled so that only the local updates and the initial pivots can grow
//! the junction bonds.
//!
//! Before the fix, every junction bond stayed at rank 1: the junction-side
//! candidate set was (pivots of the other two bonds) x (local dimension 1) =
//! 1 candidate, the bond error was computed on a 4 x 1 matrix and was zero,
//! and the run returned `Converged` with a maximum relative error of 1.68
//! (`delta = 4`) and 1.32 (`delta = 1`) on random points. A run that cannot
//! measure a bond error now reports `DegenerateCandidateSet`, and the
//! initial pivots are enriched around the junction as in xfac, so the bonds
//! grow and the run converges.

use crate::common::comb::{run, Target, TOLERANCE};
use anyhow::Result;
use tensor4all_treetci::{DefaultProposer, TreeTciTermination};

/// `expected_junction_ranks` and `expected_iterations` pin the values
/// measured on macOS after #1159 (unchanged from #1155): the permanent
/// enrichment candidates must not change the non-separable result.
fn check(
    delta: f64,
    expected_junction_ranks: [usize; 3],
    expected_iterations: usize,
) -> Result<()> {
    let run = run(Target::Oscillating { delta }, false, &DefaultProposer)?;
    assert_eq!(run.junction_ranks, expected_junction_ranks);
    assert_eq!(run.result.ranks.len(), expected_iterations);
    // The junction bonds must leave rank 1.
    assert!(
        run.junction_ranks.iter().all(|&rank| rank > 1),
        "junction bonds stayed at rank 1: {:?}",
        run.junction_ranks
    );
    // With the enriched initial pivots the local updates grow the junction
    // bonds and the run converges near the tolerance (measured: 1.62e-6 for
    // `delta = 4`, 1.20e-6 for `delta = 1`; the issue reported 1.68 / 1.32,
    // which the backward second pass of #1154 alone does not change).
    assert_eq!(run.result.termination, TreeTciTermination::Converged);
    assert!(
        run.max_rel_error < 10.0 * TOLERANCE,
        "Converged with max relative error {:.3e}",
        run.max_rel_error
    );
    Ok(())
}

#[test]
fn comb_junction_of_local_dimension_one_delta4() -> Result<()> {
    check(4.0, [27, 27, 23], 6)
}

#[test]
fn comb_junction_of_local_dimension_one_delta1() -> Result<()> {
    check(1.0, [12, 12, 11], 4)
}
