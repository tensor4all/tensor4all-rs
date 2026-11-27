//! A large opt-in rank slack can terminate before a newly discovered mode settles.
use tensor4all_core::IdxTensor;
use tensor4all_treetci::{
    crossinterpolate2, GlobalIndexBatch, PivotCandidateProposer, TreeTCI2, TreeTciEdge,
    TreeTciGraph, TreeTciOptions, TreeTciTermination,
};

/// Grow the sampled diagonal one coordinate per inner pass. With two passes
/// per iteration the recorded ranks are 3, 5, 7, 8, 8. Exhausted local samples
/// have zero residual, although a new diagonal mode remains outside the cross.
struct GrowingDiagonal;

impl PivotCandidateProposer for GrowingDiagonal {
    fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        _edge: TreeTciEdge,
        _rng: &mut R,
    ) -> tensor4all_treetci::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        let n = (state.max_bond_dim() + 1).min(8);
        let side: Vec<_> = (0..n).map(|i| vec![i]).collect();
        Ok((side.clone(), side))
    }
}

#[test]
fn opt_in_rank_slack_can_trade_accuracy_for_an_earlier_stop() {
    let run = |rank_slack| {
        crossinterpolate2(
            |batch: GlobalIndexBatch<'_>| {
                Ok::<_, anyhow::Error>(
                    (0..batch.n_points())
                        .map(|p| {
                            if batch.get(0, p) == batch.get(1, p) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<f64>>(),
                )
            },
            vec![8, 8],
            TreeTciGraph::linear_chain(2).unwrap(),
            vec![vec![0, 0]],
            TreeTciOptions {
                ncheck_history: 2,
                rank_slack,
                enable_global_pivots: false,
                seed: Some(0),
                ..Default::default()
            },
            None,
            &GrowingDiagonal,
            &mut (),
        )
        .unwrap()
    };
    let default = run(0.0);
    let slack = run(0.4);
    assert_eq!(default.termination, TreeTciTermination::Converged);
    assert_eq!(slack.termination, TreeTciTermination::Converged);
    assert_eq!(default.ranks, vec![3, 5, 7, 8, 8]);
    assert_eq!(slack.ranks, vec![3, 5]);
    for (result, expected_error) in [(&default, 0.0), (&slack, 1.0)] {
        assert_eq!(result.errors, vec![0.0; result.ranks.len()]);
        assert_eq!(result.nglobal_pivots, vec![0; result.ranks.len()]);
        let actual = result.treetn.contract_to_tensor().unwrap();
        let expected = IdxTensor::from_dense(
            actual.indices().to_vec(),
            (0..64)
                .map(|i| if i % 8 == i / 8 { 1.0 } else { 0.0 })
                .collect(),
        )
        .unwrap();
        assert_eq!(
            actual.sub(&expected).unwrap().maxabs().unwrap(),
            expected_error
        );
    }
}
