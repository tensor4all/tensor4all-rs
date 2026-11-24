//! Adaptive local truncation threshold on the sampled stop path (#1371).
//!
//! The local rrLU threshold is halved after each iteration whose recorded
//! error fails `tolerance`, down to `1 / E`; for E >= 2, a failing window at
//! that floor ends the run with `LocalThresholdFloor`. These runs check the
//! untightened and tightened-then-converged paths on the public entry points;
//! the floor stop, its precedence below the bond cap and the unarmed floor of
//! a single-edge tree are covered in `src/optimize/tests.rs`. Only
//! `StopRule::Sampled` adapts the threshold, so every run here selects it,
//! except the check that the default `PerBond` leaves it inert (#1405).

use crate::common::issue1371::{dense_relative_error, gaussian_ridge, run_tree};
use tensor4all_treetci::{
    StopRule, TreeTciGraph, TreeTciOptions, TreeTciRunResult, TreeTciTermination,
};

fn run_chain(
    value: impl Fn(&[usize]) -> f64,
    n_sites: usize,
    dim: usize,
    options: TreeTciOptions,
) -> TreeTciRunResult {
    let graph = TreeTciGraph::linear_chain(n_sites).unwrap();
    run_tree(value, vec![dim; n_sites], graph, options)
}

/// Runs from the #1371 reproducer that stalled on the untightened threshold:
/// (tolerance, seed, iterations and termination before the change).
/// - 1e-4, seed 3: 30 iterations, `MaxIterations`.
/// - 1e-6, seed 1: 30 iterations, `MaxIterations`.
/// - 1e-8, seed 4: 22 iterations, `Converged` with dense error 1.27 tol.
const STALLED_RUNS: [(f64, u64); 3] = [(1e-4, 3), (1e-6, 1), (1e-8, 4)];

#[test]
fn tightened_threshold_converges_the_stalled_issue1371_runs() {
    for (tolerance, seed) in STALLED_RUNS {
        let result = run_chain(
            gaussian_ridge,
            16,
            2,
            TreeTciOptions {
                tolerance,
                max_iter: 30,
                seed: Some(seed),
                stop_rule: StopRule::Sampled,
                ..Default::default()
            },
        );
        let context = format!("tolerance {tolerance:e}, seed {seed}");
        assert_eq!(
            result.termination,
            TreeTciTermination::Converged,
            "{context}"
        );
        assert!(result.ranks.len() <= 10, "{context}: {:?}", result.ranks);
        // Early iterations exceed the tolerance, so the threshold tightened.
        assert!(
            result.local_threshold_factor < 1.0,
            "{context}: factor {}",
            result.local_threshold_factor
        );
        assert!(result.local_threshold_factor >= 1.0 / 15.0, "{context}");
        let error = dense_relative_error(&result, gaussian_ridge, 16, 2);
        assert!(error <= tolerance, "{context}: dense error {error:e}");
    }
}

/// A product of two affine sites plus another product: exact rank 2 on every
/// bond, captured in the first sweep, so no recorded error reaches the
/// tolerance.
fn rank_two(point: &[usize]) -> f64 {
    let first: f64 = point
        .iter()
        .enumerate()
        .map(|(site, &x)| 1.0 + 0.25 * (site + x) as f64)
        .product();
    let second: f64 = point
        .iter()
        .enumerate()
        .map(|(site, &x)| 0.5 + 0.125 * ((site * x) % 3) as f64)
        .product();
    first + second
}

#[test]
fn untightened_run_keeps_its_pre_change_trajectory() {
    let options = || TreeTciOptions {
        tolerance: 1e-10,
        max_iter: 20,
        seed: Some(11),
        stop_rule: StopRule::Sampled,
        ..Default::default()
    };
    let result = run_chain(rank_two, 6, 3, options());
    assert!(result.errors.iter().all(|&error| error < 1e-10));
    assert_eq!(result.local_threshold_factor.to_bits(), 1.0_f64.to_bits());
    // Pinned from main before the adaptive threshold (050eb4ba): a run that
    // never fails the tolerance takes exactly the same path.
    assert_eq!(result.termination, TreeTciTermination::Converged);
    assert_eq!(result.ranks, PINNED_RANKS.to_vec());
    assert_eq!(result.evaluation.evaluated_points, PINNED_POINTS);
    let error = dense_relative_error(&result, rank_two, 6, 3);
    assert!(error <= 1e-12, "dense error {error:e}");
    // A second identical run is bit-identical, errors included.
    let again = run_chain(rank_two, 6, 3, options());
    let bits = |r: &TreeTciRunResult| r.errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&again), bits(&result));
}

const PINNED_RANKS: [usize; 3] = [2, 2, 2];
const PINNED_POINTS: usize = 1608;

#[test]
fn per_bond_leaves_the_local_threshold_inert() {
    // The adaptive threshold belongs to `StopRule::Sampled` (#1405). Under the
    // default `PerBond` the factor stays exactly 1.0, `LocalThresholdFloor` is
    // never reported, and the trajectory is that of `main` before this change
    // (recorded at 0d238951). Termination, ranks and points are compared
    // exactly; the errors within 1e-9 relative, as #1446's goldens do, since
    // their last bits can differ across CI platforms (the macOS nightly).
    // The capped run is the case the gate exists for: the cap keeps the bond
    // error far above the tolerance from the second iteration on, which under
    // `Sampled` tightens the threshold.
    let options = |max_bond_dim, stop_rule| TreeTciOptions {
        tolerance: 1e-6,
        max_iter: 30,
        max_bond_dim,
        seed: Some(1),
        stop_rule,
        ..Default::default()
    };
    struct Recorded {
        cap: Option<usize>,
        termination: TreeTciTermination,
        ranks: &'static [usize],
        points: usize,
        errors: &'static [f64],
    }
    let recorded = [
        Recorded {
            cap: None,
            termination: TreeTciTermination::Converged,
            ranks: &[4, 19, 20, 20, 20],
            points: 108_800,
            errors: &[
                6.940018927714176e-7,
                8.498025160454665e-7,
                9.26631012648649e-7,
                9.315569476650136e-7,
                9.315569476650136e-7,
            ],
        },
        Recorded {
            cap: Some(10),
            termination: TreeTciTermination::MaxBondDimension,
            ranks: &[4, 10, 10, 10],
            points: 38_320,
            errors: &[
                6.940018927714176e-7,
                0.02479891298879443,
                0.02732447437795533,
                0.014820605117377041,
            ],
        },
    ];
    for Recorded {
        cap,
        termination,
        ranks,
        points,
        errors,
    } in recorded
    {
        let result = run_chain(gaussian_ridge, 16, 2, options(cap, StopRule::PerBond));
        let context = format!("cap {cap:?}");
        assert_eq!(
            result.local_threshold_factor.to_bits(),
            1.0_f64.to_bits(),
            "{context}"
        );
        assert_eq!(result.termination, termination, "{context}");
        assert_eq!(result.ranks, ranks, "{context}");
        assert_eq!(result.evaluation.evaluated_points, points, "{context}");
        assert_eq!(result.errors.len(), errors.len(), "{context}");
        for (error, expected) in result.errors.iter().zip(errors) {
            assert!(
                (error - expected).abs() <= 1e-9 * expected,
                "{context}: errors {:?}, recorded {errors:?}",
                result.errors
            );
        }
    }
    // The same capped run under `Sampled` does tighten, so the PerBond run
    // above is the gate at work, not a run (B) would leave alone anyway.
    let sampled = run_chain(gaussian_ridge, 16, 2, options(Some(10), StopRule::Sampled));
    assert_eq!(sampled.termination, TreeTciTermination::MaxBondDimension);
    assert!(sampled.local_threshold_factor < 1.0);
}
