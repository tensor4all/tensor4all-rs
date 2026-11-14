//! A 4-site ridge on which a single rook walk per stop made `Rook` return
//! `Converged` with a true error of 0.39 (tensor4all-rs#1257).
//!
//! `f(x) = exp(-((x1-x2)^2 + (x3-x4)^2) / 0.01) + 0.3 cos(3 x1 x2 x3 x4)` on a
//! 16^4 grid. Columns of the middle bond with `x3 = 0` are exactly rank 2 in the
//! rows, so after two pivots a walk starting there sees a roundoff-only residual
//! and stopped the bond at rank 2 instead of 6. The stop-check walks start in
//! other columns and find the remaining pivots.
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, PivotSearchStrategy, TCI2Options, TCI2Termination};

const N: usize = 16;
const TOL: f64 = 1e-8;

fn f(v: &[usize]) -> f64 {
    let x: Vec<f64> = v.iter().map(|&i| i as f64 / N as f64).collect();
    (-((x[0] - x[1]).powi(2) + (x[2] - x[3]).powi(2)) / 0.01).exp()
        + 0.3 * (3.0 * x[0] * x[1] * x[2] * x[3]).cos()
}

#[test]
fn rook_reaches_full_rank_and_tolerance_on_a_four_site_ridge_without_global_pivots() {
    let mut points = Vec::with_capacity(N.pow(4));
    for a in 0..N {
        for b in 0..N {
            for c in 0..N {
                for d in 0..N {
                    points.push([a, b, c, d]);
                }
            }
        }
    }
    let amax = points.iter().map(|p| f(p).abs()).fold(0.0, f64::max);
    for strategy in [PivotSearchStrategy::Full, PivotSearchStrategy::Rook] {
        let options = TCI2Options {
            tolerance: TOL,
            max_iter: 20,
            pivot_search: strategy,
            max_nglobal_pivot: 0,
            seed: Some(1),
            ..TCI2Options::default()
        };
        let res = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
            |v: &Vec<usize>| f(v),
            None,
            vec![N; 4],
            vec![vec![0; 4]],
            options,
        )
        .unwrap();
        let tt = res.tci.to_tensor_train().unwrap();
        let mut true_err = 0.0f64;
        for p in &points {
            let err = (tt.evaluate(p).unwrap() - f(p)).abs();
            assert!(err.is_finite(), "non-finite error at {p:?}");
            true_err = true_err.max(err);
        }
        let true_rel = true_err / amax;
        assert_eq!(res.termination, TCI2Termination::Converged, "{strategy:?}");
        assert_eq!(tt.link_dims(), vec![16, 6, 16], "{strategy:?}");
        assert!(
            true_rel < TOL,
            "{strategy:?}: true error {true_rel:.2e} above tolerance {TOL:.0e}"
        );
    }
}
