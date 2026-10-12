//! A caller-owned context for the standalone optimizer (#859 F7).
//!
//! `optimize_with_proposer_in` exists so a consumer that drives the TCI itself - `tensor4all-quanticstci` and
//! `tensor4all-partitionedtreetn` do - can take the same caller-owned route that `crossinterpolate2_in` gives a
//! whole run. The test pins parity with the plain entry and records the figure the F7 row asks for. It does not
//! claim whole-run independence: the evaluator and the pivot search's readout are unchanged.

use std::time::Instant;

use tenferro_cpu::CpuBackend;
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treetci::{
    optimize_with_proposer, optimize_with_proposer_in, DefaultProposer, GlobalIndexBatch, TreeTCI2,
    TreeTciGraph, TreeTciOptions,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn state() -> TestResult<TreeTCI2<f64>> {
    let graph = TreeTciGraph::linear_chain(4)?;
    let mut state = TreeTCI2::<f64>::new(vec![3, 2, 2, 3], graph)?;
    state.add_global_pivots(&[vec![0, 0, 0, 0]])?;
    Ok(state)
}

fn evaluate(batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>> {
    Ok((0..batch.n_points())
        .map(|point| {
            let mut total = 1.0;
            for site in 0..4 {
                let value = batch.get(site, point).expect("site in batch") as f64;
                total += value * 0.5 + (site as f64) * value * value;
            }
            total
        })
        .collect())
}

fn options() -> TreeTciOptions {
    TreeTciOptions {
        seed: Some(11),
        tolerance: 1e-10,
        max_iter: 4,
        max_bond_dim: Some(8),
        normalize_error: true,
        enable_global_pivots: true,
        ..TreeTciOptions::default()
    }
}

fn run(context: Option<&CpuExecutionContext>) -> TestResult<(Vec<usize>, Vec<f64>)> {
    let mut state = state()?;
    let options = options();
    let result = match context {
        Some(context) => {
            optimize_with_proposer_in(&mut state, evaluate, &options, &DefaultProposer, context)?
        }
        None => optimize_with_proposer(&mut state, evaluate, &options, &DefaultProposer)?,
    };
    Ok((result.ranks, result.errors))
}

#[test]
fn the_context_optimizer_computes_what_the_plain_one_computes() -> TestResult {
    let plain = run(None)?;
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    let with_context = run(Some(&context))?;
    assert_eq!(with_context.0, plain.0, "ranks");
    assert_eq!(with_context.1.len(), plain.1.len(), "error history length");
    for (with, without) in with_context.1.iter().zip(&plain.1) {
        assert!(
            (with - without).abs() <= 1e-12 * without.abs().max(1.0),
            "{with} != {without}"
        );
    }
    Ok(())
}

#[test]
fn the_context_optimizer_is_measured_against_the_plain_one() -> TestResult {
    // A measurement, not a timing assertion: whichever way it falls, the figure is the deliverable.
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    let reps = 5;

    let started = Instant::now();
    for _ in 0..reps {
        let _ = run(None)?;
    }
    let plain = started.elapsed();

    let started = Instant::now();
    for _ in 0..reps {
        let _ = run(Some(&context))?;
    }
    let with_context = started.elapsed();

    println!(
        "treetci_optimizer_context: reps={reps} plain_ms_per_run={:.3} context_ms_per_run={:.3}",
        plain.as_secs_f64() * 1e3 / reps as f64,
        with_context.as_secs_f64() * 1e3 / reps as f64
    );
    Ok(())
}
