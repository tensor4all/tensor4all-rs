//! `Converged` must never carry a result above the tolerance (tensor4all-rs#1257).
//!
//! Two sites: the candidate matrix is the whole tensor, so the interpolant's
//! true maximum error over the grid is what the verdict has to bound.
//! The property asserted for both strategies and both `max_nglobal_pivot`
//! settings is `!Converged || true_error <= tolerance`. The reported bond error
//! is exact only up to the final 1-site cleanup sweep, which may add up to
//! `tolerance`; the matrices below stay inside the stricter bound with a margin
//! of 24-35 %, and `smooth_ridge_tolerance.rs` states the allowance for `Full`.
//!
//! Why `Rook` needs a check of its own: its bond error is the last rook pivot,
//! the largest entry of that pivot's own residual row and column, not of the
//! residual. On the ridges below that stopped a bond one pivot early (rank 49
//! instead of 50, 129 instead of 131) with the true maximum 1.2-1.3x above the
//! tolerance while the reported error was below it. A `Rook` run that meets the
//! convergence criterion therefore ends with one exact `Full` sweep, and only
//! that sweep may report `Converged`.
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, PivotSearchStrategy, TCI2Options, TCI2Termination};

const TOL: f64 = 1e-8;

type Entry = fn(usize, usize) -> f64;

/// Smooth ridge from the issue, 200 x 200.
fn ridge200(i: usize, j: usize) -> f64 {
    let (x, y) = (i as f64 / 200.0, (j as f64 + 0.5) / 200.0);
    (-(x - y) * (x - y) / 0.002).exp() + 0.3 * (3.0 * x * y).cos()
}

/// Rectangular ridge with a different bandwidth and an oscillating background.
fn ridge150x260(i: usize, j: usize) -> f64 {
    let (x, y) = (i as f64 / 150.0, (j as f64 + 0.5) / 260.0);
    (-(x - y) * (x - y) / 0.004).exp() + 0.2 * (2.0 * x + y).sin()
}

/// Two crossing bands: the diagonal and the anti-diagonal.
fn twoband300(i: usize, j: usize) -> f64 {
    let (x, y) = (i as f64 / 300.0, j as f64 / 300.0);
    (-(x - y).powi(2) / 0.001).exp() + 0.5 * (-(x + y - 1.0).powi(2) / 0.0005).exp()
}

/// `(termination, sweeps, rank, true max error / max |f|)`.
fn run(
    entry: Entry,
    n: usize,
    m: usize,
    strategy: PivotSearchStrategy,
    max_nglobal_pivot: usize,
    seed: u64,
) -> (TCI2Termination, usize, usize, f64) {
    let amax = (0..n)
        .flat_map(|i| (0..m).map(move |j| entry(i, j).abs()))
        .fold(0.0, f64::max);
    let options = TCI2Options {
        tolerance: TOL,
        max_iter: 30,
        pivot_search: strategy,
        max_nglobal_pivot,
        seed: Some(seed),
        ..TCI2Options::default()
    };
    let res = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
        |v: &Vec<usize>| entry(v[0], v[1]),
        None,
        vec![n, m],
        vec![vec![0, 0]],
        options,
    )
    .unwrap();
    let tt = res.tci.to_tensor_train().unwrap();
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..m {
            let err = (tt.evaluate(&[i, j]).unwrap() - entry(i, j)).abs();
            assert!(err.is_finite(), "non-finite error at ({i}, {j})");
            worst = worst.max(err);
        }
    }
    (
        res.termination,
        res.errors.len(),
        tt.link_dims()[0],
        worst / amax,
    )
}

#[test]
fn converged_implies_true_error_within_tolerance_for_both_strategies() {
    let cases: [(&str, Entry, usize, usize); 3] = [
        ("ridge200", ridge200, 200, 200),
        ("ridge150x260", ridge150x260, 150, 260),
        ("twoband300", twoband300, 300, 300),
    ];
    for (name, entry, n, m) in cases {
        for strategy in [PivotSearchStrategy::Full, PivotSearchStrategy::Rook] {
            for (max_nglobal_pivot, seed) in [(0, 1), (5, 1), (0, 2), (5, 2), (0, 3), (5, 3)] {
                let (termination, sweeps, rank, true_rel) =
                    run(entry, n, m, strategy, max_nglobal_pivot, seed);
                println!(
                    "{name} {strategy:?} g={max_nglobal_pivot} seed={seed}: {termination:?} \
                     sweeps={sweeps} rank={rank} true={true_rel:.2e}"
                );
                assert!(
                    termination != TCI2Termination::Converged || true_rel <= TOL,
                    "{name} {strategy:?} g={max_nglobal_pivot}: Converged with true error \
                     {true_rel:.2e} above tolerance {TOL:.0e}"
                );
            }
        }
    }
}

#[test]
fn rook_still_converges_on_the_ridges() {
    // The verdict is stricter, not unreachable: on these smooth matrices the
    // exact sweep confirms the Rook pivots and the run converges.
    for (entry, n, m) in [
        (ridge200 as Entry, 200, 200),
        (ridge150x260 as Entry, 150, 260),
        (twoband300 as Entry, 300, 300),
    ] {
        for max_nglobal_pivot in [0, 5] {
            let (termination, _, _, true_rel) =
                run(entry, n, m, PivotSearchStrategy::Rook, max_nglobal_pivot, 1);
            assert_eq!(termination, TCI2Termination::Converged);
            assert!(true_rel <= TOL, "true error {true_rel:.2e}");
        }
    }
}
