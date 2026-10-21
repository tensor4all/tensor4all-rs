//! ACI's factorizations in a caller-owned context (#859 F7).
//!
//! The context entry must compute what the plain one computes, and the share it moves must be measurable. What it
//! does *not* claim: that the whole workload runs in that context - the batched frame updates and the caller's own
//! operator keep their existing routes, so no held-global dispatch proof is available here (recorded in
//! `docs/design/859-f7-consumer-inventory.md`).

use std::time::Instant;

use rand::SeedableRng;

use tensor4all_aci::{
    elementwise_batched_with_rng, elementwise_batched_with_rng_in, AciOptions, ElementwiseBatch,
};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::CpuExecutionContext;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// A three-site product tensor train with unit bonds, so its rank cannot hide a wrong factorization.
fn product_tt(values: &[f64; 3]) -> SimpleTensorTrain<f64> {
    SimpleTensorTrain::new(vec![
        tensor3_from_data(vec![values[0], 1.0], 1, 2, 1).unwrap(),
        tensor3_from_data(vec![values[1], 1.0], 1, 2, 1).unwrap(),
        tensor3_from_data(vec![values[2], 1.0], 1, 2, 1).unwrap(),
    ])
    .unwrap()
}

fn options() -> AciOptions<f64> {
    AciOptions {
        max_iters: 8,
        min_iters: 2,
        max_bond_dim: Some(8),
        tolerance: 1e-12,
        rng_seed: 7,
        ..AciOptions::default()
    }
}

fn run(context: Option<&CpuExecutionContext>) -> TestResult<SimpleTensorTrain<f64>> {
    let left = product_tt(&[0.5, -1.25, 2.0]);
    let right = product_tt(&[1.5, 0.75, -0.5]);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let op = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
            *value = batch.get(0, point)? * batch.get(1, point)?;
        }
        Ok(())
    };
    let result = match context {
        Some(context) => {
            elementwise_batched_with_rng_in(op, &[left, right], &options(), &mut rng, context)?
        }
        None => elementwise_batched_with_rng(op, &[left, right], &options(), &mut rng)?,
    };
    Ok(result.tensor_train)
}

#[test]
fn the_context_entry_computes_what_the_plain_entry_computes() -> TestResult {
    let plain = run(None)?;
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    let with_context = run(Some(&context))?;

    assert_eq!(with_context.site_dims(), plain.site_dims());
    let (plain_data, plain_shape) = plain.full_tensor()?;
    let (context_data, context_shape) = with_context.full_tensor()?;
    assert_eq!(context_shape, plain_shape);
    assert_eq!(context_data.len(), plain_data.len());
    for (with, without) in context_data.iter().zip(&plain_data) {
        assert!(
            (with - without).abs() <= 1e-12 * without.abs().max(1.0),
            "{with} != {without}"
        );
    }
    Ok(())
}

#[test]
fn the_context_route_is_measured_against_the_plain_one() -> TestResult {
    // A measurement, not an assertion about speed: whichever way it falls, the figure is the point, and a timing
    // assertion on a shared machine would be a flake.
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
        "aci_elementwise_context: reps={reps} plain_ms_per_run={:.3} context_ms_per_run={:.3} \
         (the context entry moves the workload's MatrixLUCI factorizations and frame products; the batched frame updates keep their existing route)",
        plain.as_secs_f64() * 1e3 / reps as f64,
        with_context.as_secs_f64() * 1e3 / reps as f64
    );
    Ok(())
}
