//! Small-scale cases compared with TensorCrossInterpolation.jl 0.9.14
//! (`crossinterpolate2` in `src/tensorci2.jl`); expected values are analytic.

use tensor4all_tensorci::{
    crossinterpolate2, PivotSearchStrategy, TCI2Options, TCI2Termination, TensorCI2,
};

#[test]
fn tci2_reconstructs_small_nonzero_values() {
    for scale in [1.0, 1e-20, 1e-31] {
        for constant in [false, true] {
            for strategy in [PivotSearchStrategy::Full, PivotSearchStrategy::Rook] {
                let f = |p: &Vec<usize>| {
                    scale
                        * if constant {
                            1.0
                        } else {
                            (1 + p[0] + 2 * p[1]) as f64
                        }
                };
                let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
                    f,
                    None,
                    vec![2, 2],
                    vec![vec![1, 1]],
                    TCI2Options {
                        tolerance: 1e-10,
                        max_iter: 6,
                        nsearch: 0,
                        max_nglobal_pivot: 0,
                        pivot_search: strategy,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(result.termination, TCI2Termination::Converged);
                let (actual, _) = result.tci.to_tensor_train().unwrap().full_tensor().unwrap();
                let expected = if constant {
                    vec![1.0; 4]
                } else {
                    vec![1.0, 2.0, 3.0, 4.0]
                };
                let error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a / scale - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    error < 1e-10,
                    "scale={scale}, strategy={strategy:?}, error={error}"
                );
            }
        }
    }
}

#[test]
fn tci_initialization_accepts_tiny_nonzero_but_rejects_zero() {
    for scale in [0.0_f64, 1e-31] {
        let f = |_: &Vec<usize>| scale;
        let tci2 = TensorCI2::from_index_sets(
            vec![2, 2],
            vec![vec![vec![]], vec![vec![0]]],
            vec![vec![vec![0]], vec![vec![]]],
            &f,
        );
        if scale == 0.0 {
            assert!(matches!(
                tci2,
                Err(tensor4all_tensorci::TCIError::InvalidPivot { .. })
            ));
        } else {
            let (actual, _) = tci2
                .unwrap()
                .to_tensor_train()
                .unwrap()
                .full_tensor()
                .unwrap();
            assert!(actual
                .iter()
                .all(|value| (value / scale - 1.0).abs() < 1e-12));
        }
    }
}
