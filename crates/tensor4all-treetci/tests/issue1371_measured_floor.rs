//! A measured pointwise violation must be resolved or reported as a floor.
//!
//! https://github.com/tensor4all/tensor4all-rs/issues/1371

use anyhow::Result;
use tensor4all_core::IdxTensor;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciGraph, TreeTciOptions,
    TreeTciTermination,
};

fn value(point: &[usize]) -> f64 {
    let a = point[..8].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
    let c = point[8..].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
    (-((a - c) / 40.0).powi(2)).exp()
}

#[test]
fn gaussian_resolves_the_measured_floor_or_names_it() -> Result<()> {
    for tolerance in [1e-6, 1e-4, 1e-8] {
        for seed in 1..=6 {
            let run = crossinterpolate2(
                |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
                    Ok(batch
                        .data()
                        .chunks_exact(batch.n_sites())
                        .map(value)
                        .collect())
                },
                vec![2; 16],
                TreeTciGraph::linear_chain(16)?,
                vec![vec![0; 16]],
                TreeTciOptions {
                    tolerance,
                    max_iter: 30,
                    seed: Some(seed),
                    ..Default::default()
                },
                Some(0),
                &DefaultProposer,
                &mut (),
            )?;
            let indices: Vec<_> = (0..16)
                .map(|site| {
                    run.treetn
                        .site_space(&site)
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap()
                        .clone()
                })
                .collect();
            let reference = IdxTensor::from_dense(
                indices,
                (0..1 << 16)
                    .map(|flat| {
                        let point: Vec<_> = (0..16).map(|site| (flat >> site) & 1).collect();
                        value(&point)
                    })
                    .collect::<Vec<_>>(),
            )?;
            let dense_error = run.treetn.to_dense()?.sub(&reference)?.maxabs()?;
            let reported = run.errors.last().copied().unwrap();
            eprintln!("tol {tolerance:e} seed {seed}: {:?}, {} iterations, {} evaluated points, recorded {reported:e}, dense {dense_error:e}", run.termination, run.errors.len(), run.evaluation.evaluated_points);
            match run.termination {
                TreeTciTermination::Converged => {
                    assert!(
                        dense_error <= tolerance,
                        "seed {seed}: recorded {reported:e}, dense {dense_error:e}"
                    );
                    assert!(
                        run.errors.len() <= 10,
                        "seed {seed}: {} iterations",
                        run.errors.len()
                    );
                }
                TreeTciTermination::SampledErrorFloor { .. } => assert!(reported >= tolerance),
                other => {
                    panic!("seed {seed}: {other:?}, recorded {reported:e}, dense {dense_error:e}")
                }
            }
        }
    }
    Ok(())
}

#[test]
fn incomplete_validation_cannot_report_convergence() -> Result<()> {
    let run = crossinterpolate2(
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![2.0; batch.n_points()]) },
        vec![2; 3],
        TreeTciGraph::linear_chain(3)?,
        vec![vec![0; 3]],
        TreeTciOptions {
            validation_cache_bytes: 1,
            seed: Some(1),
            ..Default::default()
        },
        Some(0),
        &DefaultProposer,
        &mut (),
    )?;
    assert_eq!(
        run.termination,
        TreeTciTermination::ValidationBudgetExceeded
    );
    assert!(run.evaluation.validation_limit_reached);
    assert_eq!(run.evaluation.validation_samples, 0);
    assert_eq!(run.treetn.to_dense()?.maxabs()?, 2.0);
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
