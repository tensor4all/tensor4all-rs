//! Regression for tensor4all-rs#1256: with `PivotSearchStrategy::Rook`, the
//! bond error reported by `crossinterpolate2` must be the true maximum error of
//! the returned interpolant, not the magnitude of the last rook pivot.
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, PivotSearchStrategy, TCI2Options, TCI2Termination};

const N: usize = 200;

fn entry(i: usize, j: usize) -> f64 {
    let (x, y) = (i as f64 / N as f64, (j as f64 + 0.5) / N as f64);
    (-(x - y) * (x - y) / 0.002).exp() + 0.3 * (3.0 * x * y).cos()
}

#[test]
fn rook_reported_error_bounds_the_true_error_on_a_smooth_ridge() {
    let tol = 1e-8;
    let amax = (0..N)
        .flat_map(|i| (0..N).map(move |j| entry(i, j).abs()))
        .fold(0.0, f64::max);
    for max_nglobal_pivot in [0, 5] {
        let options = TCI2Options {
            tolerance: tol,
            max_iter: 20,
            pivot_search: PivotSearchStrategy::Rook,
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
        let true_rel = true_err / amax;
        let reported = *res.errors.last().unwrap();
        println!(
            "max_nglobal_pivot={max_nglobal_pivot}: termination={:?} sweeps={} rank={} \
             reported={reported:.2e} true={true_rel:.2e}",
            res.termination,
            res.errors.len(),
            tt.link_dims()[0],
        );
        // Two sites: the 2-site candidate matrix is the whole tensor, so the
        // reported sweep error is the true maximum error of the 2-site
        // interpolant. The final 1-site cleanup sweep may still drop pivots
        // whose residual is below `tolerance` (relative), which `errors` does
        // not record; that is the only allowance here. Before the fix the
        // reported error was 2.5e-9 and the true error 1.3e-6.
        assert!(
            true_rel <= reported + tol,
            "reported {reported:.2e} under-estimates the true error {true_rel:.2e}"
        );
        assert!(
            true_rel < tol,
            "true error {true_rel:.2e} above tolerance {tol:.0e}"
        );
        assert_eq!(res.termination, TCI2Termination::Converged);
    }
}
