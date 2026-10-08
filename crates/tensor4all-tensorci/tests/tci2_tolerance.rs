//! Regression for #810: normalized/absolute tolerances must reach both LU paths.
//! Compared with TensorCrossInterpolation.jl 0.9.14, updatepivots!/optimize!.
use tensor4all_tensorci::{
    crossinterpolate2, PivotSearchStrategy, TCI2Options, TCI2Termination, TensorCI2,
};

fn options(strategy: PivotSearchStrategy, normalize: bool) -> TCI2Options {
    TCI2Options {
        tolerance: 0.1,
        normalize_error: normalize,
        max_iter: 6,
        nsearch: 0,
        max_nglobal_pivot: 0,
        pivot_search: strategy,
        seed: Some(0), // Global search disabled: no random choice affects this test.
        ..Default::default()
    }
}

fn assert_diagonal(tci: &TensorCI2<f64>, large: f64, small: f64, rank: usize) {
    let tt = tci.to_tensor_train().unwrap();
    let (dense, shape) = tt.full_tensor().unwrap();
    assert_eq!(shape, vec![2, 2]);
    let expected = [large, 0., 0., small]; // Column-major, one dense readout.
    let max_error = dense
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f64, f64::max);
    assert!(
        max_error < 1e-12,
        "max residual={max_error}, dense={dense:?}"
    );
    assert_eq!(tci.link_dims(), vec![rank]);
}

#[test]
fn optimizer_absolute_and_relative_modes_select_different_ranks() {
    for strategy in [PivotSearchStrategy::Full, PivotSearchStrategy::Rook] {
        for batched in [false, true] {
            for (normalize, scale, small, rank) in
                [(false, 1., 0., 1), (false, 20., 0.2, 2), (true, 20., 0., 1)]
            {
                let f = |p: &Vec<usize>| {
                    if p[0] != p[1] {
                        0.
                    } else if p[0] == 0 {
                        scale
                    } else {
                        0.01 * scale
                    }
                };
                let batch = |indices: &[Vec<usize>]| indices.iter().map(&f).collect::<Vec<_>>();
                let result = crossinterpolate2(
                    f,
                    batched.then_some(batch),
                    vec![2, 2],
                    vec![vec![0, 0]],
                    options(strategy, normalize),
                )
                .unwrap();
                assert_diagonal(&result.tci, scale, small, rank);
                assert_eq!(result.termination, TCI2Termination::Converged);
            }
        }
    }
}

#[test]
fn public_two_site_sweep_uses_the_same_absolute_threshold_in_both_directions() {
    for strategy in [PivotSearchStrategy::Full, PivotSearchStrategy::Rook] {
        for forward in [true, false] {
            for batched in [false, true] {
                for normalize in [false, true] {
                    let f = |p: &Vec<usize>| {
                        if p[0] != p[1] {
                            0.
                        } else if p[0] == 0 {
                            20.
                        } else {
                            0.2
                        }
                    };
                    let mut tci = TensorCI2::from_index_sets(
                        vec![2, 2],
                        vec![vec![vec![]], vec![vec![0]]],
                        vec![vec![vec![0]], vec![vec![]]],
                        &f,
                    )
                    .unwrap();
                    let batch = |indices: &[Vec<usize>]| indices.iter().map(&f).collect::<Vec<_>>();
                    tci.sweep2site(
                        &f,
                        &batched.then_some(batch),
                        forward,
                        &options(strategy, normalize),
                    )
                    .unwrap();
                    assert_diagonal(
                        &tci,
                        20.,
                        if normalize { 0. } else { 0.2 },
                        if normalize { 1 } else { 2 },
                    );
                }
            }
        }
    }
}

#[test]
fn normalized_sweep_uses_global_sample_scale_not_each_local_matrix_scale() {
    for strategy in [PivotSearchStrategy::Full, PivotSearchStrategy::Rook] {
        let f = |p: &Vec<usize>| match p.as_slice() {
            [0, 0, 0] => 20.,
            [1, 1, 0] => 5.,
            [0, 0, 1] => 100.,
            _ => 0.,
        };
        let mut tci = TensorCI2::from_index_sets(
            vec![2, 2, 2],
            vec![vec![vec![]], vec![vec![0]], vec![vec![0, 0]]],
            vec![vec![vec![0, 0]], vec![vec![0]], vec![vec![]]],
            &f,
        )
        .unwrap();
        assert_eq!(tci.max_sample_value(), 100.);
        tci.sweep2site(
            &f,
            &None::<fn(&[Vec<usize>]) -> Vec<f64>>,
            true,
            &options(strategy, true),
        )
        .unwrap();
        let (dense, _) = tci.to_tensor_train().unwrap().full_tensor().unwrap();
        let expected = [20., 0., 0., 0., 100., 0., 0., 0.];
        let max_error = dense
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max);
        assert!(
            max_error < 1e-12,
            "max residual={max_error}, dense={dense:?}"
        );
        assert_eq!(tci.link_dims(), vec![1, 1]);
    }
}

#[test]
fn normalized_threshold_overflow_is_rejected_before_sweeping() {
    use std::cell::Cell;
    let calls = Cell::new(0);
    let f = |_: &Vec<usize>| {
        calls.set(calls.get() + 1);
        20.
    };
    let mut tci = TensorCI2::from_index_sets(
        vec![2, 2],
        vec![vec![vec![]], vec![vec![0]]],
        vec![vec![vec![0]], vec![vec![]]],
        &f,
    )
    .unwrap();
    calls.set(0);
    let mut opts = options(PivotSearchStrategy::Full, true);
    opts.tolerance = f64::MAX;
    let error = tci
        .sweep2site(&f, &None::<fn(&[Vec<usize>]) -> Vec<f64>>, true, &opts)
        .unwrap_err();
    assert!(matches!(
        error,
        tensor4all_tensorci::TCIError::InvalidConfiguration { .. }
    ));
    assert_eq!(calls.get(), 0);
}

#[test]
fn absolute_tolerance_also_preserves_complex_directions() {
    use num_complex::Complex64;
    for strategy in [PivotSearchStrategy::Full, PivotSearchStrategy::Rook] {
        let f = |p: &Vec<usize>| {
            Complex64::new(
                0.,
                if p[0] != p[1] {
                    0.
                } else if p[0] == 0 {
                    20.
                } else {
                    0.2
                },
            )
        };
        let result = crossinterpolate2(
            f,
            None::<fn(&[Vec<usize>]) -> Vec<Complex64>>,
            vec![2, 2],
            vec![vec![0, 0]],
            options(strategy, false),
        )
        .unwrap();
        let (dense, _) = result.tci.to_tensor_train().unwrap().full_tensor().unwrap();
        let expected = [20., 0., 0., 0.2];
        let max_error = dense
            .iter()
            .zip(expected)
            .map(|(a, b)| (*a - Complex64::new(0., b)).norm())
            .fold(0_f64, f64::max);
        assert!(max_error < 1e-12, "complex residual={max_error}");
        assert_eq!(result.termination, TCI2Termination::Converged);
    }
}
