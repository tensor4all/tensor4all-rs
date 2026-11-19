//! Larger opt-in confirmation evidence; excluded from the ordinary suite.
use anyhow::Result;
use num_complex::Complex64;
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
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
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
                confirm_convergence: true,
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
        assert_eq!(run.termination, TreeTciTermination::MaxIterations);
        assert_eq!(run.errors.len(), 30);
        // This is a budget/verdict regression, not an exhaustive accuracy
        // check over the 2^32-point domain. The small ordinary budget test
        // separately checks exact dense values and successful convergence.
        eprintln!("32-site seed 2, budget {budget}: {:?}, {} sweeps, {} evaluations, {} history evictions, reported {reported:e}",
            run.termination, run.errors.len(), run.evaluation.evaluated_points, run.evaluation.validation_evictions);
    }
    Ok(())
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn frozen_plane_seed_survey() -> Result<()> {
    let seeds = (1..=16).collect::<Vec<_>>();
    super::issue1352_rank_deficient::check_reopened(&seeds, 1.0_f64)?;
    super::issue1352_rank_deficient::check_reopened(&seeds, Complex64::new(1.0, 0.5))
}
