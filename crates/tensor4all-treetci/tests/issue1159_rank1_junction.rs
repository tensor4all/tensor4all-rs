//! Regression test for issue #1159: a target that is exactly rank 1 across
//! the bonds of a vertex of local dimension 1 must be able to end in
//! `Converged`.
//!
//! The tree is the comb of [`crate::common::comb`]; the target of the
//! reproducer is its separable window factor
//! `sinc(kx/16) sinc(ky/16) sinc(kz/4)`.
//!
//! Before the fix the enrichment pivots of #1155 lived only in the pivot sets.
//! The first update of a junction bond kept one of them, and from then on the
//! junction side had a single candidate (rank 1 x rank 1 x local dimension 1),
//! so the bond was recorded as unmeasured and every run ended in
//! `DegenerateCandidateSet` at a maximum relative error of 7.7e-7 (1362
//! distinct points with the global search off, 3240 with it on). The enriched
//! pivots are now permanent candidates of the junction side, as in xfac's
//! pivot history `I0`.

use crate::common::comb::{comb_graph, comb_local_dims, run, Target, JUNCTION, N_SITES, TOLERANCE};
use anyhow::Result;
use rand::Rng;
use tensor4all_treetci::{
    optimize_with_proposer, AllEdges, DefaultProposer, GlobalIndexBatch, MultiIndex,
    PivotCandidateProposer, TreeTCI2, TreeTciEdge, TreeTciOptions, TreeTciResult,
    TreeTciTermination,
};

fn check_separable(enable_global_pivots: bool) -> Result<()> {
    let run = run(Target::Separable, enable_global_pivots, &DefaultProposer)?;
    // Rank 1 is the exact answer across the junction.
    assert_eq!(run.junction_ranks, vec![1; 3]);
    assert_eq!(run.result.termination, TreeTciTermination::Converged);
    assert!(
        run.max_rel_error < 10.0 * TOLERANCE,
        "Converged with max relative error {:.3e}",
        run.max_rel_error
    );
    Ok(())
}

#[test]
fn separable_target_converges_global_off() -> Result<()> {
    check_separable(false)
}

#[test]
fn separable_target_converges_global_on() -> Result<()> {
    check_separable(true)
}

/// `prod_s (1 + x_s)` is exactly rank 1 on every bond.
fn product(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
    let (s, p) = (batch.n_sites(), batch.n_points());
    Ok((0..p)
        .map(|j| {
            batch.data()[j * s..(j + 1) * s]
                .iter()
                .map(|&x| 1.0 + x as f64)
                .product()
        })
        .collect())
}

fn optimize_product<P: PivotCandidateProposer>(
    initial_pivots: &[MultiIndex],
    enable_global_pivots: bool,
    proposer: &P,
) -> Result<tensor4all_treetci::TreeTciOptimizationResult> {
    let mut state = TreeTCI2::<f64>::new(comb_local_dims(), comb_graph()?)?;
    state.add_global_pivots(initial_pivots)?;
    let options = TreeTciOptions {
        tolerance: TOLERANCE,
        enable_global_pivots,
        seed: Some(1),
        ..Default::default()
    };
    Ok(optimize_with_proposer(
        &mut state,
        product,
        &options,
        proposer,
        &AllEdges,
        &mut (),
    )?)
}

/// From a single initial pivot (the enrichment adds pivots) and from two
/// initial pivots that already give every junction bond rank 2 (the
/// enrichment adds none but still records the candidates), the rank-1 bonds
/// of a product function are measured and the run converges, with or
/// without the global search.
#[test]
fn product_function_converges_with_measured_rank_one_bonds() -> Result<()> {
    let mut alternate = vec![1; N_SITES];
    alternate[JUNCTION] = 0;
    for initial in [vec![vec![0; N_SITES]], vec![vec![0; N_SITES], alternate]] {
        for enable_global_pivots in [false, true] {
            let result = optimize_product(&initial, enable_global_pivots, &DefaultProposer)?;
            assert_eq!(result.termination, TreeTciTermination::Converged);
            assert_eq!(result.ranks.last(), Some(&1));
            assert_eq!(result.errors.last(), Some(&0.0));
        }
    }
    Ok(())
}

/// [`DefaultProposer`] restricted to the first candidate on the junction side
/// of every junction bond: the side can never offer a second candidate, as
/// before the enrichment of #1155.
struct SingleJunctionCandidate;

impl PivotCandidateProposer for SingleJunctionCandidate {
    fn candidates_with_rng<T, R: Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
        rng: &mut R,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (mut left, mut right) = DefaultProposer.candidates_with_rng(state, edge, rng)?;
        if edge.u() == JUNCTION {
            left.truncate(1);
        } else if edge.v() == JUNCTION {
            right.truncate(1);
        }
        Ok((left, right))
    }
}

/// A real stall stays reported: when the junction side cannot offer a second
/// candidate, the zero bond errors are not measurements and the run must not
/// end in `Converged`, for a rank-1 and for a non-separable target.
/// The enrichment candidates are a property of [`DefaultProposer`]; this
/// proposer drops them on purpose.
#[test]
fn single_candidate_junction_side_is_not_reported_converged() -> Result<()> {
    let result = optimize_product(&[vec![0; N_SITES]], false, &SingleJunctionCandidate)?;
    assert_eq!(
        result.termination,
        TreeTciTermination::DegenerateCandidateSet
    );

    let run = run(
        Target::Oscillating { delta: 4.0 },
        false,
        &SingleJunctionCandidate,
    )?;
    // Measured: rank 1 junction bonds at a maximum relative error of 1.68,
    // the stall of #1153.
    assert_eq!(
        run.result.termination,
        TreeTciTermination::DegenerateCandidateSet
    );
    assert_eq!(run.junction_ranks, vec![1; 3]);
    assert!(run.max_rel_error > 0.1, "{:.3e}", run.max_rel_error);
    Ok(())
}
