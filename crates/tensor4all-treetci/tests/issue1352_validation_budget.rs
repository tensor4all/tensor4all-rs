//! Minimal fixed-seed history pressure and explicit opt-in boundaries.
use anyhow::Result;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciGraph, TreeTciOptions,
    TreeTciTermination,
};

#[test]
fn eviction_does_not_stop_convergence_confirmation() -> Result<()> {
    let run = crossinterpolate2(
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            Ok(batch
                .data()
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| 1.0 + p.iter().sum::<usize>() as f64)
                .collect())
        },
        vec![2; 3],
        TreeTciGraph::linear_chain(3)?,
        vec![vec![0; 3]],
        TreeTciOptions {
            confirm_convergence: true,
            validation_cache_bytes: 512,
            nsearch: 0,
            seed: Some(7),
            ..Default::default()
        },
        Some(0),
        &DefaultProposer,
        &mut (),
    )?;
    assert_eq!(run.termination, TreeTciTermination::Converged);
    assert_eq!(run.errors.len(), 3);
    assert_eq!(run.evaluation.confirmation_checks, 1);
    assert_eq!(run.evaluation.confirmation_failures, 0);
    assert!(run.evaluation.validation_limit_reached);
    assert!(run.evaluation.validation_evictions > 0);
    assert!(run.evaluation.validation_bytes <= 512);
    let indices = (0..3)
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
    let reference = tensor4all_core::IdxTensor::from_dense(
        indices,
        vec![1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0],
    )?;
    assert_eq!(run.treetn.to_dense()?.sub(&reference)?.maxabs()?, 0.0);
    Ok(())
}

#[test]
fn zero_history_still_confirms_current_pairs() -> Result<()> {
    for budget in [0, 1] {
        let run = crossinterpolate2(
            |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![2.0; batch.n_points()]) },
            vec![2; 3],
            TreeTciGraph::linear_chain(3)?,
            vec![vec![0; 3]],
            TreeTciOptions {
                confirm_convergence: true,
                enable_global_pivots: false,
                validation_cache_bytes: budget,
                seed: Some(7),
                ..Default::default()
            },
            Some(0),
            &DefaultProposer,
            &mut (),
        )?;
        assert_eq!(run.termination, TreeTciTermination::Converged);
        assert_eq!(run.errors.len(), 3);
        assert_eq!(run.evaluation.confirmation_checks, 1);
        assert_eq!(run.evaluation.validation_samples, 0);
        assert!(run.evaluation.validation_limit_reached);
        assert_eq!(run.treetn.to_dense()?.maxabs()?, 2.0);
    }
    Ok(())
}
