use super::*;
use crate::{
    crossinterpolate2, PivotCandidateProposer, TreeTciEvaluationStats, TreeTciIteration,
    TreeTciObserver, TreeTciRunResult,
};
use std::cell::Cell;

struct LimitedCross {
    retain_injected: bool,
    rotate: bool,
    calls: Cell<usize>,
}
impl PivotCandidateProposer for LimitedCross {
    fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        _: TreeTciEdge,
        _: &mut R,
    ) -> crate::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        let candidates = if self.retain_injected && state.max_bond_dim() > 1 {
            vec![vec![0], vec![1], vec![2]]
        } else if self.rotate && (call / 6) % 2 == 1 {
            vec![vec![1], vec![0]]
        } else {
            vec![vec![0], vec![1]]
        };
        Ok((candidates.clone(), candidates))
    }
}

#[derive(Default)]
struct Confirmations(Vec<(usize, f64, usize, usize)>);
impl TreeTciObserver for Confirmations {
    fn iteration(&mut self, it: &TreeTciIteration<'_>) {
        self.0.push((
            it.iteration,
            it.error,
            it.evaluation.confirmation_checks,
            it.evaluation.confirmation_pivots,
        ));
    }
}

fn peak_run(
    options: TreeTciOptions,
    retain_injected: bool,
    rotate: bool,
) -> (TreeTciRunResult, Confirmations) {
    let mut observed = Confirmations::default();
    let proposer = LimitedCross {
        retain_injected,
        rotate,
        calls: Cell::new(0),
    };
    let result = crossinterpolate2(
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            Ok(batch
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| if *p == [2, 2] { 2.0 } else { 1.0 })
                .collect())
        },
        vec![3, 3],
        two_site_graph(),
        vec![vec![2, 2]],
        options,
        Some(0),
        &proposer,
        &mut observed,
    )
    .unwrap();
    (result, observed)
}

fn enabled(max_iter: usize) -> TreeTciOptions {
    TreeTciOptions {
        confirm_convergence: true,
        enable_global_pivots: false,
        nsearch: 0,
        max_iter,
        seed: Some(7),
        ..Default::default()
    }
}

fn peak_dense_error(result: &TreeTciRunResult) -> f64 {
    let indices = (0..2)
        .map(|site| {
            result
                .treetn
                .site_space(&site)
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .clone()
        })
        .collect();
    let reference =
        IdxTensor::from_dense(indices, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).unwrap();
    result
        .treetn
        .to_dense()
        .unwrap()
        .sub(&reference)
        .unwrap()
        .maxabs()
        .unwrap()
        / reference.maxabs().unwrap()
}

#[test]
fn disabled_confirmation_matches_main35174_bit_for_bit() {
    // Captured by confirmation_off_baseline.rs against immutable main 35174fdd.
    // This existing restricted-cross fixture records global residual 1, whereas
    // 501ebb4a recorded zero. The final sweep skips ordinary global discovery.
    struct InitialCrossOnly;
    impl PivotCandidateProposer for InitialCrossOnly {
        fn candidates_with_rng<T, R: rand::Rng + ?Sized>(
            &self,
            _: &TreeTCI2<T>,
            _: TreeTciEdge,
            _: &mut R,
        ) -> crate::TreeTciResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
            Ok((vec![vec![0]], vec![vec![0]]))
        }
    }
    let options = TreeTciOptions {
        tolerance: 1e-12,
        max_iter: 4,
        nsearch: 20,
        seed: Some(7),
        ..Default::default()
    };
    assert!(!options.confirm_convergence);
    let mut state = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    state.max_sample_value = 1.0;
    let result = crate::optimize_with_proposer(
        &mut state,
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            Ok((0..batch.n_points())
                .map(|p| (1 + batch.get(0, p).unwrap() + batch.get(1, p).unwrap()) as f64)
                .collect())
        },
        &options,
        &InitialCrossOnly,
        &crate::AllEdges,
        &mut (),
    )
    .unwrap();
    assert_eq!(
        result
            .errors
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        [1.0_f64, 1.0, 1.0, 0.0].map(f64::to_bits)
    );
    assert_eq!(result.ranks, vec![1; 4]);
    assert_eq!(result.termination, TreeTciTermination::MaxIterations);
    assert_eq!(
        result.evaluation,
        TreeTciEvaluationStats {
            requested_points: 203,
            evaluated_points: 203,
            ..Default::default()
        }
    );
}

#[test]
fn failed_confirmation_injects_then_requires_three_fresh_sweeps() {
    for memo in [None, Some(4096)] {
        let (result, observed) = peak_run(
            TreeTciOptions {
                evaluation_cache_bytes: memo,
                ..enabled(12)
            },
            true,
            false,
        );
        assert_eq!(result.termination, TreeTciTermination::Converged);
        assert_eq!(result.ranks, vec![1, 1, 1, 2, 2, 2]);
        assert_eq!(result.errors, vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(result.evaluation.confirmation_checks, 2);
        assert_eq!(result.evaluation.confirmation_failures, 1);
        assert_eq!(result.evaluation.confirmation_pivots, 1);
        assert_eq!(
            observed.0.iter().map(|p| p.2).collect::<Vec<_>>(),
            vec![0, 0, 1, 1, 1, 2]
        );
        assert_eq!(observed.0[2], (2, 0.5, 1, 1));
        assert_eq!(peak_dense_error(&result), 0.0);
        if memo.is_some() {
            assert!(result.evaluation.cache_hits > 0);
        }
    }
}

#[test]
fn final_failed_confirmation_returns_measured_max_iterations_without_injection() {
    let (result, observed) = peak_run(enabled(3), false, false);
    assert_eq!(result.termination, TreeTciTermination::MaxIterations);
    assert_eq!(result.errors, vec![0.0, 0.0, 0.5]);
    assert_eq!(result.evaluation.confirmation_checks, 1);
    assert_eq!(result.evaluation.confirmation_failures, 1);
    assert_eq!(result.evaluation.confirmation_pivots, 0);
    assert_eq!(observed.0[2], (2, 0.5, 1, 0));
    assert_eq!(peak_dense_error(&result), 0.5);
}

#[test]
fn final_iteration_precedes_an_otherwise_detected_floor() {
    let (result, _) = peak_run(enabled(9), false, false);
    assert_eq!(result.termination, TreeTciTermination::MaxIterations);
    assert_eq!(result.errors, vec![0.0, 0.0, 0.5].repeat(3));
    assert_eq!(result.evaluation.confirmation_checks, 3);
    assert_eq!(result.evaluation.confirmation_failures, 3);
    assert_eq!(result.evaluation.confirmation_pivots, 2);
    assert_eq!(peak_dense_error(&result), 0.5);
}

#[test]
fn only_repeated_failed_confirmation_snapshots_report_a_floor() {
    for insertion_limit in [0, 5] {
        let (result, observed) = peak_run(
            TreeTciOptions {
                max_nglobal_pivot: insertion_limit,
                ..enabled(12)
            },
            false,
            false,
        );
        assert_eq!(result.termination, TreeTciTermination::SampledErrorFloor);
        assert_eq!(
            result.errors,
            vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
        );
        assert_eq!(result.evaluation.confirmation_checks, 3);
        assert_eq!(result.evaluation.confirmation_failures, 3);
        assert_eq!(
            result.evaluation.confirmation_pivots,
            if insertion_limit == 0 { 0 } else { 2 }
        );
        assert_eq!(observed.0.len(), 9);
        assert_eq!(peak_dense_error(&result), 0.5);
    }
}

#[test]
fn changing_failed_snapshots_at_equal_rank_do_not_report_a_floor() {
    let (result, _) = peak_run(enabled(10), false, true);
    assert_eq!(result.termination, TreeTciTermination::MaxIterations);
    assert_eq!(result.ranks, vec![1; 10]);
    assert_eq!(result.evaluation.confirmation_checks, 3);
    assert_eq!(result.evaluation.confirmation_failures, 3);
    assert_eq!(peak_dense_error(&result), 0.5);
}

#[test]
fn caps_limits_and_zero_tolerance_do_not_run_confirmation() {
    for options in [
        enabled(2),
        TreeTciOptions {
            max_bond_dim: Some(1),
            ..enabled(12)
        },
        TreeTciOptions {
            tolerance: 0.0,
            ..enabled(4)
        },
    ] {
        let cap = options.max_bond_dim;
        let max_iter = options.max_iter;
        let (result, _) = peak_run(options, false, false);
        assert_eq!(result.evaluation.confirmation_checks, 0);
        assert_eq!(
            result.termination,
            if cap.is_some() {
                TreeTciTermination::MaxBondDimension
            } else {
                TreeTciTermination::MaxIterations
            }
        );
        assert_eq!(
            result.errors,
            vec![0.0; if cap.is_some() { 3 } else { max_iter }]
        );
    }
}

#[test]
fn confirmation_witness_callback_errors_are_preserved() {
    for error in [false, true] {
        let peak_calls = Cell::new(0);
        let proposer = LimitedCross {
            retain_injected: false,
            rotate: false,
            calls: Cell::new(0),
        };
        let result = crossinterpolate2(
            |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
                if batch.data() == [2, 2] {
                    peak_calls.set(peak_calls.get() + 1);
                    if peak_calls.get() > 1 {
                        if error {
                            anyhow::bail!("confirmation witness failed");
                        }
                        return Ok(Vec::new());
                    }
                }
                Ok(batch
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| if *p == [2, 2] { 2.0 } else { 1.0 })
                    .collect())
            },
            vec![3, 3],
            two_site_graph(),
            vec![vec![2, 2]],
            enabled(12),
            Some(0),
            &proposer,
            &mut (),
        )
        .unwrap_err();
        assert!(result.to_string().contains(if error {
            "confirmation witness failed"
        } else {
            "confirmation witness evaluator returned wrong length"
        }));
    }
}
