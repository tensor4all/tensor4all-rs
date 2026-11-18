//! Bounded validation must preserve honest verdicts without preventing convergence.
//!
//! https://github.com/tensor4all/tensor4all-rs/pull/1372

use anyhow::Result;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciGraph, TreeTciOptions,
    TreeTciTermination,
};

fn value(point: &[usize]) -> f64 {
    let bits = point.len() / 2;
    let a = point[..bits].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
    let c = point[bits..].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
    let width = 40.0 * (1usize << (bits - 8)) as f64;
    (-((a - c) / width).powi(2)).exp()
}

#[test]
fn a_tiny_history_budget_does_not_prevent_constant_convergence() -> Result<()> {
    for budget in [0, 1] {
        let run = crossinterpolate2(
            |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![2.0; batch.n_points()]) },
            vec![2; 3],
            TreeTciGraph::linear_chain(3)?,
            vec![vec![0; 3]],
            TreeTciOptions {
                validation_cache_bytes: budget,
                seed: Some(1),
                ..Default::default()
            },
            Some(0),
            &DefaultProposer,
            &mut (),
        )?;
        assert_eq!(run.termination, TreeTciTermination::Converged);
        assert!(run.evaluation.validation_limit_reached);
        assert_eq!(run.evaluation.validation_samples, 0);
        assert_eq!(run.treetn.to_dense()?.maxabs()?, 2.0);
    }
    Ok(())
}

#[test]
fn the_32_site_seed_two_run_outlives_a_bounded_history() -> Result<()> {
    for budget in [32 * 1024 * 1024, 64 * 1024] {
        let run = crossinterpolate2(
            |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
                Ok(batch
                    .data()
                    .chunks_exact(batch.n_sites())
                    .map(value)
                    .collect())
            },
            vec![2; 32],
            TreeTciGraph::linear_chain(32)?,
            vec![vec![0; 32]],
            TreeTciOptions {
                tolerance: 1e-6,
                max_iter: 30,
                seed: Some(2),
                validation_cache_bytes: budget,
                ..Default::default()
            },
            Some(0),
            &DefaultProposer,
            &mut (),
        )?;
        let reported = *run.errors.last().unwrap();
        if budget == 64 * 1024 {
            assert!(run.evaluation.validation_limit_reached);
            assert!(run.evaluation.validation_evictions > 0);
        }
        assert!(run.evaluation.validation_bytes <= budget);
        assert!(run.errors.len() >= 3);
        match run.termination {
            TreeTciTermination::Converged => assert!(reported < 1e-6),
            TreeTciTermination::SampledErrorFloor { .. } => assert!(reported >= 1e-6),
            TreeTciTermination::MaxIterations => {
                // Changing pivot sets are not a proven stationary floor. The
                // run must reach its requested limit rather than a cache stop.
                assert_eq!(run.errors.len(), 30);
                assert!(reported >= 1e-6);
            }
            other => panic!("unexpected stop {other:?}, residual {reported:e}"),
        }
        // This is a budget/verdict regression, not an exhaustive accuracy
        // check over the 2^32-point domain. The constant-budget case above
        // separately checks an exact dense value and successful convergence.
        eprintln!("32-site seed 2, budget {budget}: {:?}, {} sweeps, {} evaluations, {} history evictions, reported {reported:e}",
            run.termination, run.errors.len(), run.evaluation.evaluated_points, run.evaluation.validation_evictions);
    }
    Ok(())
}

#[test]
fn disabling_global_checks_disables_validation_history() -> Result<()> {
    let run = crossinterpolate2(
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![2.0; batch.n_points()]) },
        vec![2; 3],
        TreeTciGraph::linear_chain(3)?,
        vec![vec![0; 3]],
        TreeTciOptions {
            enable_global_pivots: false,
            validation_cache_bytes: 0,
            seed: Some(1),
            ..Default::default()
        },
        Some(0),
        &DefaultProposer,
        &mut (),
    )?;
    assert_eq!(run.termination, TreeTciTermination::Converged);
    assert_eq!(run.evaluation.validation_samples, 0);
    assert_eq!(run.evaluation.validation_bytes, 0);
    assert!(!run.evaluation.validation_limit_reached);
    assert_eq!(run.treetn.to_dense()?.maxabs()?, 2.0);
    Ok(())
}
