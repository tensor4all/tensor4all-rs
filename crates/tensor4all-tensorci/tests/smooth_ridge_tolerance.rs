//! The smooth ridge from tensor4all-rs#1256 under both pivot strategies.
//!
//! `Full` is the certified strategy: its reported bond error is the maximum
//! remaining residual, so the returned interpolant's true error stays within
//! the reported error plus a `tolerance` allowance for the final 1-site cleanup
//! sweep, which `errors` does not record.
//!
//! `Rook` stops when the current rook pivot is below tolerance. On this ridge
//! that pivot is a local maximum far below the remaining residual, so `Rook`
//! terminates `Converged` with a true error above the tolerance. The second test
//! pins that limitation of the heuristic contract.
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, PivotSearchStrategy, TCI2Options, TCI2Termination};

const N: usize = 200;
const TOL: f64 = 1e-8;

fn entry(i: usize, j: usize) -> f64 {
    let (x, y) = (i as f64 / N as f64, (j as f64 + 0.5) / N as f64);
    (-(x - y) * (x - y) / 0.002).exp() + 0.3 * (3.0 * x * y).cos()
}

/// Run TCI2 on the ridge and return `(termination, reported, true_rel)`: the
/// last sweep error and the interpolant's true maximum error, both relative to
/// the tensor's maximum magnitude.
fn run(strategy: PivotSearchStrategy, max_nglobal_pivot: usize) -> (TCI2Termination, f64, f64) {
    let amax = (0..N)
        .flat_map(|i| (0..N).map(move |j| entry(i, j).abs()))
        .fold(0.0, f64::max);
    let options = TCI2Options {
        tolerance: TOL,
        max_iter: 20,
        pivot_search: strategy,
        max_nglobal_pivot,
        seed: Some(1),
        ..TCI2Options::default()
    };
    let res = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
        |v: &Vec<usize>| entry(v[0], v[1]),
        None,
        vec![N, N],
        vec![vec![0, 0]],
        options,
    )
    .unwrap();
    let tt = res.tci.to_tensor_train().unwrap();
    let mut true_err = 0.0f64;
    for i in 0..N {
        for j in 0..N {
            let err = (tt.evaluate(&[i, j]).unwrap() - entry(i, j)).abs();
            assert!(err.is_finite(), "non-finite error at ({i}, {j})");
            true_err = true_err.max(err);
        }
    }
    let reported = *res.errors.last().unwrap();
    println!(
        "{strategy:?} max_nglobal_pivot={max_nglobal_pivot}: termination={:?} sweeps={} \
         rank={} reported={reported:.2e} true={:.2e}",
        res.termination,
        res.errors.len(),
        tt.link_dims()[0],
        true_err / amax,
    );
    (res.termination, reported, true_err / amax)
}

#[test]
fn full_true_error_within_reported_bond_error_plus_cleanup_allowance_on_a_smooth_ridge() {
    for max_nglobal_pivot in [0, 5] {
        let (termination, reported, true_rel) = run(PivotSearchStrategy::Full, max_nglobal_pivot);
        // Two sites: the 2-site candidate matrix is the whole tensor, so the
        // reported sweep error is the true maximum error of the 2-site
        // interpolant. The final 1-site cleanup sweep may still drop pivots
        // whose residual is below `tolerance` (relative), which `errors` does
        // not record; that is the only allowance here.
        assert!(
            true_rel <= reported + TOL,
            "reported {reported:.2e} under-estimates the true error {true_rel:.2e}"
        );
        assert!(
            true_rel < TOL,
            "true error {true_rel:.2e} above tolerance {TOL:.0e}"
        );
        assert_eq!(termination, TCI2Termination::Converged);
    }
}

#[test]
fn rook_converged_is_not_a_proof_of_the_tolerance_on_a_smooth_ridge() {
    // The rook walk stops on a pivot 2.5e-9 of the maximum while the
    // interpolant is off by 1.3e-6, and TCI2 still reports `Converged`.
    let (termination, reported, true_rel) = run(PivotSearchStrategy::Rook, 0);
    assert_eq!(termination, TCI2Termination::Converged);
    assert!(
        reported < TOL,
        "reported {reported:.2e} not below {TOL:.0e}"
    );
    assert!(
        true_rel > 100.0 * TOL,
        "true error {true_rel:.2e} is no longer far above the tolerance; \
         the limitation this test pins has changed"
    );
}
