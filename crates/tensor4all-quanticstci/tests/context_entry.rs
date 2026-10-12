//! A caller-owned context for the quantics TCI entry (#859 F7).
//!
//! The pass-through this pins: `quanticstci` drives the TCI itself through `tensor4all-treetci`'s optimizer, so its
//! context entry forwards to `optimize_with_proposer_in`. Parity is the assertion; the timing is a measurement, and
//! the final materialization of the returned tensor train stays on its existing route (recorded in the inventory).

use std::time::Instant;

use tenferro_cpu::CpuBackend;
use tensor4all_quanticstci::{
    pointwise_coordinate_batch, pointwise_index_batch, quanticscrossinterpolate_batch,
    quanticscrossinterpolate_batch_in, quanticscrossinterpolate_discrete_batch,
    quanticscrossinterpolate_discrete_batch_in, DiscretizedGrid, QtciOptions,
};
use tensor4all_tensorbackend::CpuExecutionContext;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn options() -> QtciOptions {
    QtciOptions::default().with_tolerance(1e-10)
}

fn run(context: Option<&CpuExecutionContext>) -> TestResult<(Vec<usize>, Vec<f64>, f64)> {
    let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
    let sizes = vec![16, 16];
    let (qtci, ranks, errors) = match context {
        Some(context) => quanticscrossinterpolate_discrete_batch_in(
            &sizes,
            pointwise_index_batch(f),
            None,
            options(),
            context,
        )?,
        None => quanticscrossinterpolate_discrete_batch(
            &sizes,
            pointwise_index_batch(f),
            None,
            options(),
        )?,
    };
    let value = qtci.evaluate(&[4, 9])?;
    Ok((ranks, errors, value))
}

#[test]
fn the_context_entry_computes_what_the_plain_entry_computes() -> TestResult {
    let (plain_ranks, plain_errors, plain_value) = run(None)?;
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    let (context_ranks, context_errors, context_value) = run(Some(&context))?;

    assert_eq!(context_ranks, plain_ranks, "ranks");
    assert_eq!(
        context_errors.len(),
        plain_errors.len(),
        "error history length"
    );
    for (with, without) in context_errors.iter().zip(&plain_errors) {
        assert!(
            (with - without).abs() <= 1e-9 * without.abs().max(1.0),
            "{with} != {without}"
        );
    }
    assert!((context_value - plain_value).abs() < 1e-10);
    assert!((context_value - 13.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn the_context_entry_is_measured_against_the_plain_one() -> TestResult {
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
        "quanticstci_context: reps={reps} plain_ms_per_run={:.3} context_ms_per_run={:.3}",
        plain.as_secs_f64() * 1e3 / reps as f64,
        with_context.as_secs_f64() * 1e3 / reps as f64
    );
    Ok(())
}

fn grid() -> DiscretizedGrid {
    DiscretizedGrid::builder(&[4])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .include_endpoint(true)
        .build()
        .expect("grid")
}

fn run_grid(context: Option<&CpuExecutionContext>) -> TestResult<(Vec<usize>, Vec<f64>)> {
    let f = |coords: &[f64]| coords[0].sin();
    let (_, ranks, errors) = match context {
        Some(context) => quanticscrossinterpolate_batch_in::<f64, _>(
            &grid(),
            pointwise_coordinate_batch(f),
            None,
            options(),
            context,
        )?,
        None => quanticscrossinterpolate_batch::<f64, _>(
            &grid(),
            pointwise_coordinate_batch(f),
            None,
            options(),
        )?,
    };
    Ok((ranks, errors))
}

#[test]
fn the_grid_context_entry_computes_what_the_plain_entry_computes() -> TestResult {
    let (plain_ranks, plain_errors) = run_grid(None)?;
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    let (context_ranks, context_errors) = run_grid(Some(&context))?;
    assert_eq!(context_ranks, plain_ranks, "ranks");
    assert_eq!(
        context_errors.len(),
        plain_errors.len(),
        "error history length"
    );
    for (with, without) in context_errors.iter().zip(&plain_errors) {
        assert!(
            (with - without).abs() <= 1e-9 * without.abs().max(1.0),
            "{with} != {without}"
        );
    }
    Ok(())
}

#[test]
fn the_grid_context_entry_is_measured_against_the_plain_one() -> TestResult {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    let reps = 5;
    let started = Instant::now();
    for _ in 0..reps {
        let _ = run_grid(None)?;
    }
    let plain = started.elapsed();
    let started = Instant::now();
    for _ in 0..reps {
        let _ = run_grid(Some(&context))?;
    }
    let with_context = started.elapsed();
    println!(
        "quanticstci_grid_context: reps={reps} plain_ms_per_run={:.3} context_ms_per_run={:.3}",
        plain.as_secs_f64() * 1e3 / reps as f64,
        with_context.as_secs_f64() * 1e3 / reps as f64
    );
    Ok(())
}
