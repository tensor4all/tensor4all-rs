//! Reproduce #1371's evaluation counts without measuring elapsed time.
//!
//! Run this identical source against each revision with the default CPU provider.
//! Dense verification happens after interpolation and is excluded from the
//! reported oracle count. No oracle memoization is enabled.

use anyhow::Result;
use tensor4all_core::IdxTensor;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciGraph, TreeTciOptions,
};

fn value(point: &[usize]) -> f64 {
    let a = point[..8].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
    let c = point[8..].iter().fold(0, |n, &bit| 2 * n + bit) as f64;
    (-((a - c) / 40.0).powi(2)).exp()
}

fn main() -> Result<()> {
    println!("tolerance,seed,termination,iterations,max_rank,evaluated_points,reported_error,dense_error");
    for tolerance in [1e-4, 1e-6, 1e-8] {
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
            println!(
                "{tolerance:e},{seed},{:?},{},{},{},{:e},{dense_error:e}",
                run.termination,
                run.errors.len(),
                run.ranks.iter().copied().max().unwrap_or(0),
                run.evaluation.evaluated_points,
                run.errors.last().copied().unwrap_or(f64::NAN),
            );
        }
    }
    Ok(())
}
