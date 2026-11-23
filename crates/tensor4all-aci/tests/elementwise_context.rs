//! ACI's numerics in a caller-owned context (#859 F7, #1382).
//!
//! The context entry must compute what the plain one computes, and the share it moves must be measurable. Since
//! #1382 the whole ACI workload (every MatrixLUCI factorization and every matrix product, batched or not) runs in the
//! caller's context, so the dispatch test holds the process-global backend while the context entry runs on a worker
//! thread: a step that still took the process-global route would block on that lock, and the timeout turns it into a
//! failure. Only the caller's own operator is outside the claim.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use rand::SeedableRng;

use tensor4all_aci::{
    elementwise_batched_with_rng, elementwise_batched_with_rng_in, AciOptions, ElementwiseBatch,
};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};
use tensor4all_tensorbackend::{with_default_backend, CpuBackend, CpuExecutionContext};

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
    run_inputs(context, left, right)
}

fn run_inputs(
    context: Option<&CpuExecutionContext>,
    left: SimpleTensorTrain<f64>,
    right: SimpleTensorTrain<f64>,
) -> TestResult<SimpleTensorTrain<f64>> {
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
         (the context entry moves every MatrixLUCI factorization and matrix product of the workload)",
        plain.as_secs_f64() * 1e3 / reps as f64,
        with_context.as_secs_f64() * 1e3 / reps as f64
    );
    Ok(())
}

/// A three-site tensor train with bond dimension two, so the sweep takes nontrivial local blocks, frame updates
/// and solution core updates rather than rank-one shortcuts.
fn bond_two_tt(shift: f64) -> SimpleTensorTrain<f64> {
    SimpleTensorTrain::new(vec![
        tensor3_from_data(vec![1.0 + shift, 0.5, -0.25, 2.0 - shift], 1, 2, 2).unwrap(),
        tensor3_from_data(
            vec![
                0.75,
                -1.0 + shift,
                0.5,
                1.25,
                -0.5,
                0.25 + shift,
                1.5,
                -0.75,
            ],
            2,
            2,
            2,
        )
        .unwrap(),
        tensor3_from_data(vec![1.0, -0.5 + shift, 0.25, 1.5], 2, 2, 1).unwrap(),
    ])
    .unwrap()
}

/// #1382: the context entry must never take the process-global route. Before the fix the local block setup,
/// local block materialization and batched frame updates multiplied through the process-global `mat_mul` even when
/// a context was supplied; with the global backend held, that call blocks and this test times out.
#[test]
fn context_entry_runs_while_the_process_global_backend_is_held() -> TestResult {
    let (left, right) = (bond_two_tt(0.0), bond_two_tt(0.375));
    // The reference runs before the lock is taken, on the plain (process-global) entry, which also warms the
    // process-global runtime so its lazy initialization cannot be what blocks below.
    let reference = run_inputs(None, left.clone(), right.clone())?;

    let worker_result = with_default_backend(|_held_global_backend| {
        let (done_tx, done_rx) = mpsc::channel();
        let (left, right) = (left.clone(), right.clone());
        let worker = std::thread::spawn(move || {
            let result = CpuBackend::with_threads(1)
                .map_err(|error| error.to_string())
                .and_then(|backend| {
                    let context = CpuExecutionContext::from_backend(backend);
                    run_inputs(Some(&context), left, right).map_err(|error| error.to_string())
                });
            let _ = done_tx.send(());
            result
        });
        match done_rx.recv_timeout(Duration::from_secs(120)) {
            Ok(()) => worker.join().expect("ACI worker thread panicked"),
            Err(_) => panic!(
                "elementwise_batched_with_rng_in did not finish while the process-global backend was held: a \
                 factorization or matrix product of the ACI sweep still took the process-global route \
                 instead of the caller's context (tensor4all-rs#1382)"
            ),
        }
    });
    let with_context = worker_result?;

    // Same seed, same algorithm, different engine: the interpolants agree to rounding.
    let (reference_data, reference_shape) = reference.full_tensor()?;
    let (context_data, context_shape) = with_context.full_tensor()?;
    assert_eq!(context_shape, reference_shape);
    assert!(
        with_context.link_dims().iter().any(|&dim| dim > 1),
        "the fixture must exercise a nontrivial rank, got {:?}",
        with_context.link_dims()
    );
    for (with, without) in context_data.iter().zip(&reference_data) {
        assert!(
            (with - without).abs() <= 1e-10 * without.abs().max(1.0),
            "{with} != {without}"
        );
    }
    Ok(())
}
