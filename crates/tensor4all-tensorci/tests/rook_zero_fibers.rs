//! Zero-fiber cases compared with TensorCrossInterpolation.jl 0.9.14
//! (`crossinterpolate2` in `src/tensorci2.jl`); expected values are analytic.

use tensor4all_tensorci::{crossinterpolate2, PivotSearchStrategy, TCI2Options, TCI2Termination};

#[test]
fn rook_preserves_known_nonzero_pivots_across_zero_fibers() {
    for n in [2, 3, 8, 16] {
        for diagonal_gap in [false, true] {
            for global_search in [false, true] {
                let f = |p: &Vec<usize>| {
                    if p == &[n - 1, n - 1] {
                        2.0
                    } else if diagonal_gap && p == &[0, 0] {
                        1.0
                    } else {
                        0.0
                    }
                };
                let mut options = TCI2Options {
                    pivot_search: PivotSearchStrategy::Rook,
                    tolerance: 1e-10,
                    max_iter: 20,
                    seed: Some(1234),
                    ..Default::default()
                };
                if !global_search {
                    options.nsearch = 0;
                    options.max_nglobal_pivot = 0;
                }
                let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
                    f,
                    None,
                    vec![n, n],
                    vec![vec![n - 1, n - 1]],
                    options,
                )
                .unwrap();
                assert_eq!(result.termination, TCI2Termination::Converged);
                let (actual, _) = result.tci.to_tensor_train().unwrap().full_tensor().unwrap();
                let mut expected = vec![0.0; n * n];
                expected[n * n - 1] = 2.0;
                if diagonal_gap {
                    expected[0] = 1.0;
                }
                let error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    error < 1e-10,
                    "n={n}, gap={diagonal_gap}, global={global_search}: {error}"
                );
            }
        }
    }
}
