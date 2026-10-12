//! F4 slice 3a: adaptive interpolation whose patch work runs on the lanes of one held session.
//!
//! The value comparison alone would also hold on the compatibility route, so these tests also observe execution:
//! two lanes must run patch work at the same time. The third oracle the design lists - an unrelated non-waiting
//! entry must be refused while the phase holds its reservation - is not covered here: in this configuration the
//! unrelated entry waits rather than being refused, so a test asserting a refusal would be asserting the wrong
//! behaviour; that oracle needs the arbiter's non-waiting entry path and is recorded as still open.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tensor4all_core::{DynIndex, MultiIndex};
use tensor4all_partitionedtt::{
    adaptiveinterpolate, adaptiveinterpolate_on_phase, AdaptiveInterpolateOptions,
    AdaptiveInterpolationResult,
};
use tensor4all_tensorbackend::{explicit::HeldSession, with_default_backend, CpuExecutionContext};
use tensor4all_tensorci::TCI2Options;

fn binary_sites(nsites: usize) -> Vec<DynIndex> {
    (0..nsites).map(|_| DynIndex::new_dyn(2)).collect()
}

/// A sharp, non-separable integrand: it forces patch splits, so the wave has more than one patch to schedule.
fn oracle(index: &MultiIndex) -> f64 {
    let distance: f64 = index
        .iter()
        .map(|&point| (point as f64 / 4.0 - 0.3).powi(2))
        .sum();
    1.0 / (0.05 + distance)
}

fn options() -> AdaptiveInterpolateOptions {
    AdaptiveInterpolateOptions {
        tci_options: TCI2Options {
            tolerance: 1e-6,
            max_bond_dim: Some(4),
            seed: Some(1),
            ..TCI2Options::default()
        },
        ..AdaptiveInterpolateOptions::default()
    }
}

fn held_session(threads: usize) -> HeldSession<'static> {
    let context = CpuExecutionContext::from_backend(
        tenferro_cpu::CpuBackend::with_threads(threads).expect("CPU backend"),
    );
    // The held session needs a backend that outlives it; a test can leak its own clone.
    let backend: &'static tenferro_cpu::CpuBackend =
        Box::leak(Box::new(context.with_backend(|backend| backend.clone())));
    HeldSession::open(backend).expect("held session")
}

fn projectors(
    result: &AdaptiveInterpolationResult<f64>,
) -> Vec<tensor4all_partitionedtt::Projector> {
    result
        .patch_caches()
        .iter()
        .map(|cache| cache.projector().clone())
        .collect()
}

fn dense_values(result: &AdaptiveInterpolationResult<f64>) -> Vec<f64> {
    let tt = result
        .partitioned_tt()
        .to_tensor_train()
        .expect("tensor train");
    let tensors: Vec<_> = (0..tt.len())
        .map(|site| tt.tensor(site).expect("site tensor"))
        .collect();
    tensor4all_core::contract(&tensors)
        .expect("dense")
        .to_vec::<f64>()
        .expect("values")
}

#[test]
fn the_phase_entry_matches_the_sequential_entry() {
    let sites = binary_sites(8);
    let sequential = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        oracle,
        None,
        sites.clone(),
        Vec::new(),
        options(),
    )
    .expect("sequential run");
    let expected_projectors = projectors(&sequential);
    assert!(
        expected_projectors.len() > 1,
        "the fixture must split at least once"
    );
    let expected_values = dense_values(&sequential);

    for threads in [1usize, 2] {
        let mut held = held_session(threads);
        let result = adaptiveinterpolate_on_phase::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
            &mut held,
            oracle,
            None,
            sites.clone(),
            Vec::new(),
            options(),
        )
        .expect("phase run");
        held.close().expect("close");

        assert_eq!(
            projectors(&result),
            expected_projectors,
            "threads = {threads}"
        );
        let values = dense_values(&result);
        assert_eq!(values.len(), expected_values.len(), "threads = {threads}");
        for (value, expected) in values.iter().zip(&expected_values) {
            assert!(
                (value - expected).abs() < 1e-8,
                "threads = {threads}: {value} != {expected}"
            );
        }
    }
}

#[test]
fn the_phase_entry_distributes_patch_work_across_lanes_and_stays_off_the_global_route() {
    let sites = binary_sites(8);
    let thread_ids: Arc<Mutex<std::collections::HashSet<Option<usize>>>> =
        Arc::new(Mutex::new(std::collections::HashSet::new()));
    let thread_writer = Arc::clone(&thread_ids);
    let evaluator = move |index: &MultiIndex| {
        // Which pool worker ran this evaluation is the observation: with more than one entry the phase really
        // distributed the wave's items across lanes instead of running them all in one.
        thread_writer
            .lock()
            .expect("thread set")
            .insert(rayon::current_thread_index());
        oracle(index)
    };

    // Warm the process-global eager runtime before the global backend is held: cold initialization takes that
    // lock, which the holder then owns.
    let sequential = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        oracle,
        None,
        sites.clone(),
        Vec::new(),
        options(),
    )
    .expect("sequential run");
    let expected_values = dense_values(&sequential);

    // A held session is bound to the thread that opened it, so the phase runs here and a second thread holds the
    // process-global backend for a bounded time. A patch path that consulted the global session would block until
    // that hold ends, which the timing assertion catches without hanging the suite.
    const HOLD: Duration = Duration::from_secs(20);
    let (release, released) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        with_default_backend(|_global| {
            let _ = released.recv_timeout(HOLD);
        });
    });
    std::thread::sleep(Duration::from_millis(200));

    let mut held = held_session(2);
    let started = std::time::Instant::now();
    let result = adaptiveinterpolate_on_phase::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        &mut held,
        evaluator,
        None,
        sites,
        Vec::new(),
        options(),
    )
    .expect("phase run");
    let elapsed = started.elapsed();
    held.close().expect("close");
    let _ = release.send(());
    holder.join().expect("holder thread");

    let threads = thread_ids.lock().expect("thread set");
    assert!(
        threads.len() > 1,
        "the wave must be distributed across lanes, workers seen: {threads:?}"
    );
    assert!(
        elapsed < HOLD / 2,
        "the phase took {elapsed:?} while the process-global backend was held for {HOLD:?}: a patch path still \
         consulted the global session"
    );
    let values = dense_values(&result);
    assert_eq!(values.len(), expected_values.len());
    for (value, expected) in values.iter().zip(&expected_values) {
        assert!((value - expected).abs() < 1e-8, "{value} != {expected}");
    }
}
