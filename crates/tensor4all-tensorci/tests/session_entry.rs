//! F4 slice 1: the TCI2 path can run its dense factor assembly and site solves on a caller-entered session.
//!
//! Value parity alone would also hold with the compatibility route, so the dispatch test holds the
//! process-global backend while the session entry runs: a run that consulted the global session would block
//! there. The lazy-block pivot path is deliberately not threaded yet (the design records the exclusion), and the
//! fixtures below take the dense path (`PivotSearchStrategy::Full`, the default).

use std::time::Duration;

use rand::SeedableRng;

use tensor4all_core::MultiIndex;
use tensor4all_tensorbackend::{with_default_backend, CpuExecutionContext};
use tensor4all_tensorci::{
    crossinterpolate2_with_rng, crossinterpolate2_with_rng_on_session, TCI2Options,
};

fn options() -> TCI2Options {
    TCI2Options {
        tolerance: 1e-10,
        max_bond_dim: Some(8),
        seed: Some(7),
        ..TCI2Options::default()
    }
}

/// A smooth, non-separable integrand with a moving sharp feature: it forces real sweeps rather than a rank-1
/// short cut.
fn oracle(index: &MultiIndex) -> f64 {
    let distance: f64 = index
        .iter()
        .map(|&point| (point as f64 / 4.0 - 0.3).powi(2))
        .sum();
    1.0 / (0.05 + distance)
}

fn dense_values(result: &tensor4all_tensorci::TCI2OptimizationResult<f64>) -> Vec<f64> {
    let tt = result.tci.to_tensor_train().expect("tensor train");
    let (values, _shape) = tt.full_tensor().expect("dense");
    values
}

fn run_plain() -> tensor4all_tensorci::TCI2OptimizationResult<f64> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    crossinterpolate2_with_rng::<f64, _, fn(&[MultiIndex]) -> Vec<f64>, _>(
        oracle,
        None,
        vec![4; 6],
        vec![vec![0usize; 6]],
        options(),
        &mut rng,
    )
    .expect("plain run")
}

fn run_on_session(
    context: &CpuExecutionContext,
) -> tensor4all_tensorci::TCI2OptimizationResult<f64> {
    context
        .with_concrete_session(|session| {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
            crossinterpolate2_with_rng_on_session::<f64, _, fn(&[MultiIndex]) -> Vec<f64>, _>(
                oracle,
                None,
                vec![4; 6],
                vec![vec![0usize; 6]],
                options(),
                &mut rng,
                session,
            )
        })
        .expect("session entry")
        .expect("run on session")
}

fn context() -> CpuExecutionContext {
    CpuExecutionContext::from_backend(
        tenferro_cpu::CpuBackend::with_threads(1).expect("1T CPU backend"),
    )
}

#[test]
fn the_session_entry_computes_what_the_plain_entry_computes() {
    let plain = run_plain();
    let on_session = run_on_session(&context());

    assert_eq!(on_session.termination, plain.termination);
    assert_eq!(on_session.ranks, plain.ranks);
    assert_eq!(on_session.errors, plain.errors);

    let plain_values = dense_values(&plain);
    let session_values = dense_values(&on_session);
    assert_eq!(session_values.len(), plain_values.len());
    for (index, (left, right)) in session_values.iter().zip(&plain_values).enumerate() {
        assert!(
            (left - right).abs() <= 1e-12 * right.abs().max(1.0),
            "entry {index}: {left} != {right}"
        );
    }
}

#[test]
fn the_session_entry_runs_while_the_process_global_backend_is_held() {
    // Warm the process-global eager runtime first: cold initialization takes the global context's backend
    // lock, which this test holds.
    let warm = run_plain();
    let reference = dense_values(&warm);

    // The worker thread lets a wrong route fail this test instead of hanging the suite; the panic releases the
    // lock. Both the dense factor assembly inside the sweeps and the site solves are inside the run.
    let values = with_default_backend(|_global| {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = run_on_session(&context());
            let _ = done_tx.send(());
            result
        });
        match done_rx.recv_timeout(Duration::from_secs(120)) {
            Ok(()) => dense_values(&worker.join().expect("worker thread")),
            Err(_) => panic!(
                "the session entry did not finish while the process-global backend was held: a numerical path \
                 on this route still consulted the global session"
            ),
        }
    });

    assert_eq!(values.len(), reference.len());
    for (index, (left, right)) in values.iter().zip(&reference).enumerate() {
        assert!(
            (left - right).abs() <= 1e-12 * right.abs().max(1.0),
            "entry {index}: {left} != {right}"
        );
    }
}
