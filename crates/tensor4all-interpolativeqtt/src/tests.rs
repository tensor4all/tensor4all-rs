use approx::assert_relative_eq;
use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
use tensor4all_simplett::{Tensor3, Tensor3Ops};

use crate::interpolation::{checked_dim_add, dense_test_points, flatten_index};
use crate::interval::NInterval;

use super::*;

fn quantics_to_indices(quantics: &[usize]) -> Vec<usize> {
    quantics.to_vec()
}

fn grid_1d(r: usize, a: f64, b: f64) -> DiscretizedGrid {
    DiscretizedGrid::builder(&[r])
        .with_lower_bound(&[a])
        .with_upper_bound(&[b])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap()
}

fn grid_nd(r: usize, lower: &[f64], upper: &[f64]) -> DiscretizedGrid {
    DiscretizedGrid::builder(&vec![r; lower.len()])
        .with_lower_bound(lower)
        .with_upper_bound(upper)
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap()
}

fn for_each_grid_index(ndims: usize, points_per_dim: usize, mut f: impl FnMut(&[usize])) {
    let mut index = vec![0usize; ndims];
    loop {
        f(&index);

        let mut advanced = false;
        for value in &mut index {
            if *value + 1 < points_per_dim {
                *value += 1;
                advanced = true;
                break;
            }
            *value = 0;
        }
        if !advanced {
            break;
        }
    }
}

fn assert_tt_matches_grid_1d(
    tt: &SimpleTensorTrain<f64>,
    grid: &DiscretizedGrid,
    f: impl Fn(f64) -> f64,
    tol: f64,
) {
    let n = 1usize << grid.rs()[0];
    for i in 0..n {
        let grid_idx = [i];
        let quantics = grid.grididx_to_quantics(&grid_idx).unwrap();
        let coords = grid.grididx_to_origcoord(&grid_idx).unwrap();
        let value = tt.evaluate(&quantics_to_indices(&quantics)).unwrap();
        let expected = f(coords[0]);
        assert!(
            (value - expected).abs() < tol,
            "grid index {grid_idx:?}: value {value}, expected {expected}, diff {}",
            (value - expected).abs()
        );
    }
}

fn assert_tt_matches_grid_nd(
    tt: &SimpleTensorTrain<f64>,
    grid: &DiscretizedGrid,
    f: impl Fn(&[f64]) -> f64,
    tol: f64,
) {
    let ndims = grid.ndims();
    let n = 1usize << grid.rs()[0];
    for_each_grid_index(ndims, n, |grid_idx| {
        let quantics = grid.grididx_to_quantics(grid_idx).unwrap();
        let coords = grid.grididx_to_origcoord(grid_idx).unwrap();
        let value = tt.evaluate(&quantics_to_indices(&quantics)).unwrap();
        let expected = f(&coords);
        assert!(
            (value - expected).abs() < tol,
            "grid index {grid_idx:?}: value {value}, expected {expected}, diff {}",
            (value - expected).abs()
        );
    });
}

#[test]
fn single_scale_interpolation_1d() {
    let r = 4;
    let a = -2.8;
    let b = std::f64::consts::PI;
    let degree = 20;
    let f = |x: f64| (-x * x).exp();

    let tt =
        interpolate_single_scale(f, a, b, r, degree, &InterpolativeQttOptions::default()).unwrap();
    assert!(tt.rank() <= degree + 1);

    let grid = grid_1d(r, a, b);
    assert_tt_matches_grid_1d(&tt, &grid, f, 1e-10);
}

#[test]
fn single_scale_interpolation_nd_n1() {
    let r = 4;
    let a = -2.8;
    let b = std::f64::consts::PI;
    let degree = 20;
    let f = |x: &[f64]| (-x[0] * x[0]).exp();

    let tt = interpolate_single_scale_nd(
        f,
        &[a],
        &[b],
        r,
        degree,
        &InterpolativeQttOptions::default(),
    )
    .unwrap();
    assert!(tt.rank() <= degree + 1);

    let grid = grid_1d(r, a, b);
    assert_tt_matches_grid_1d(&tt, &grid, |x| f(&[x]), 1e-10);
}

#[test]
fn single_scale_interpolation_n2() {
    let r = 4;
    let degree = 20;
    let lower = [-1.0, -1.0];
    let upper = [2.0, 2.0];
    let f = |x: &[f64]| (-x[0] * x[0] - x[1].powi(3)).exp();

    let tt = interpolate_single_scale_nd(
        f,
        &lower,
        &upper,
        r,
        degree,
        &InterpolativeQttOptions::default().with_tolerance(0.0),
    )
    .unwrap();
    assert!(tt.rank() <= (degree + 1).pow(2));
    assert_eq!(tt.len(), r);

    let grid = grid_nd(r, &lower, &upper);
    assert_tt_matches_grid_nd(&tt, &grid, f, 1e-10);
}

#[test]
fn single_scale_interpolation_n3() {
    // Three sites retain left, interior, and right cores while keeping the full
    // three-dimensional grid check to 8^3 points. The n2 fixture above retains
    // repeated interior cores (four sites).
    let r = 3;
    let degree = 15;
    let lower = [-1.0, -1.0, 0.0];
    let upper = [2.0, 2.0, 1.0];
    let f = |x: &[f64]| (-x[0] * x[0] - x[1].powi(3) - 2.0 * x[2] * x[2]).exp();

    let tt = interpolate_single_scale_nd(
        f,
        &lower,
        &upper,
        r,
        degree,
        &InterpolativeQttOptions::default(),
    )
    .unwrap();
    assert!(tt.rank() <= (degree + 1).pow(3));
    assert_eq!(tt.len(), r);

    let grid = grid_nd(r, &lower, &upper);
    assert_tt_matches_grid_nd(&tt, &grid, f, 1e-7);
}

#[test]
fn multiscale_interpolation_1d() {
    let r = 4;
    let a = -2.0;
    let b = 2.0_f64.sqrt();
    let degree = 25;
    let f = |x: f64| (-x * x).exp() + x.abs();

    let tt = interpolate_multi_scale(
        f,
        a,
        b,
        r,
        degree,
        &[0.0],
        &InterpolativeQttOptions::default(),
    )
    .unwrap();
    assert!(tt.rank() <= degree + 2);

    let grid = grid_1d(r, a, b);
    assert_tt_matches_grid_1d(&tt, &grid, f, 1e-12);
}

#[test]
fn multiscale_interpolation_n2() {
    let r = 4;
    let degree = 10;
    let lower = [0.0, 0.0];
    let upper = [1.0, 1.0];
    let f = |x: &[f64]| (-x[0] * x[0] - 2.0 * x[1] * x[1]).exp();

    let tt = interpolate_multi_scale_nd(
        f,
        &lower,
        &upper,
        r,
        degree,
        &[vec![0.0, 0.0]],
        &InterpolativeQttOptions::default(),
    )
    .unwrap();
    assert!(tt.rank() <= (degree + 2).pow(2));

    let grid = grid_nd(r, &lower, &upper);
    assert_tt_matches_grid_nd(&tt, &grid, f, 1e-10);
}

#[test]
fn ninterval_split_matches_julia_order() {
    let interval = NInterval::new(&[-1.0, -1.0], &[1.0, 1.0]).unwrap();
    let pieces = interval.split().unwrap();

    assert_eq!(
        pieces,
        vec![
            NInterval::new(&[-1.0, -1.0], &[0.0, 0.0]).unwrap(),
            NInterval::new(&[0.0, -1.0], &[1.0, 0.0]).unwrap(),
            NInterval::new(&[-1.0, 0.0], &[0.0, 1.0]).unwrap(),
            NInterval::new(&[0.0, 0.0], &[1.0, 1.0]).unwrap(),
        ]
    );
}

#[test]
fn direct_product_core_tensors_two_tensors() {
    let chi = 3;
    let c1 = Tensor3::from_fn([chi, 2, chi], |[i, s, k]| (100 * i + 10 * s + k) as f64).unwrap();
    let c2 =
        Tensor3::from_fn([chi, 2, chi], |[i, s, k]| (1 + 100 * i + 10 * s + k) as f64).unwrap();
    let fused = direct_product_core_tensors(&[c1.clone(), c2.clone()]).unwrap();

    assert_eq!(fused.left_dim(), chi * chi);
    assert_eq!(fused.site_dim(), 4);
    assert_eq!(fused.right_dim(), chi * chi);
    for i in 0..chi {
        for j in 0..chi {
            for k in 0..2 {
                for l in 0..2 {
                    for m in 0..chi {
                        for n in 0..chi {
                            let left = i + chi * j;
                            let site = k + 2 * l;
                            let right = m + chi * n;
                            let expected = *c1.get3(i, k, m) * *c2.get3(j, l, n);
                            assert_relative_eq!(*fused.get3(left, site, right), expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn direct_product_core_tensors_three_tensors() {
    let chi = 3;
    let c1 = Tensor3::from_fn([chi, 2, chi], |[i, s, k]| (100 * i + 10 * s + k) as f64).unwrap();
    let c2 =
        Tensor3::from_fn([chi, 2, chi], |[i, s, k]| (1 + 100 * i + 10 * s + k) as f64).unwrap();
    let c3 =
        Tensor3::from_fn([chi, 2, chi], |[i, s, k]| (2 + 100 * i + 10 * s + k) as f64).unwrap();
    let fused = direct_product_core_tensors(&[c1.clone(), c2.clone(), c3.clone()]).unwrap();

    assert_eq!(fused.left_dim(), chi * chi * chi);
    assert_eq!(fused.site_dim(), 8);
    assert_eq!(fused.right_dim(), chi * chi * chi);
    for i in 0..chi {
        for j in 0..chi {
            for k in 0..chi {
                for l in 0..2 {
                    for m in 0..2 {
                        for n in 0..2 {
                            for o in 0..chi {
                                for p in 0..chi {
                                    for q in 0..chi {
                                        let left = i + chi * j + chi * chi * k;
                                        let site = l + 2 * m + 4 * n;
                                        let right = o + chi * p + chi * chi * q;
                                        let expected = *c1.get3(i, l, o)
                                            * *c2.get3(j, m, p)
                                            * *c3.get3(k, n, q);
                                        assert_relative_eq!(
                                            *fused.get3(left, site, right),
                                            expected
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn lagrange_polynomials_new_handles_general_grid() {
    let basis = LagrangePolynomials::new(vec![0.0, 0.2, 0.7, 1.0]).unwrap();
    let x = 0.43_f64;

    assert!(!basis.is_empty());
    assert!(basis.barycentric_weights().iter().all(|w| w.is_finite()));
    assert!(basis
        .barycentric_weights()
        .iter()
        .any(|w| w.is_sign_negative()));

    let mut partition_of_unity = 0.0;
    let mut cubic_interpolant = 0.0;
    for alpha in 0..basis.len() {
        let value = basis.evaluate(alpha, x).unwrap();
        partition_of_unity += value;
        cubic_interpolant += value * basis.grid()[alpha].powi(3);
    }

    assert!((partition_of_unity - 1.0).abs() < 1.0e-12);
    assert!((cubic_interpolant - x.powi(3)).abs() < 1.0e-12);
}

#[test]
fn lagrange_polynomials_validation_errors() {
    assert!(LagrangePolynomials::new(vec![0.0]).is_err());
    assert!(LagrangePolynomials::new(vec![0.0, f64::NAN]).is_err());
    assert!(LagrangePolynomials::new(vec![0.0, 0.0]).is_err());

    let basis = LagrangePolynomials::new(vec![0.0, 0.5, 1.0]).unwrap();
    assert!(basis.evaluate(basis.len(), 0.25).is_err());
    assert!(basis.evaluate(0, f64::INFINITY).is_err());
}

#[test]
fn high_degree_chebyshev_basis_stays_finite() {
    let degree = 600;
    let basis = get_chebyshev_grid(degree).unwrap();

    assert_eq!(basis.len(), degree + 1);
    assert!(basis.barycentric_weights().iter().all(|w| w.is_finite()));
    assert!((basis.evaluate(0, basis.grid()[0]).unwrap() - 1.0).abs() < 1.0e-12);
    assert!(basis.evaluate(1, basis.grid()[0]).unwrap().abs() < 1.0e-12);

    let core = interpolation_tensor(&basis).unwrap();
    for left in 0..core.left_dim() {
        for site in 0..core.site_dim() {
            for right in 0..core.right_dim() {
                assert!(core.get3(left, site, right).is_finite());
            }
        }
    }
}

#[test]
fn high_degree_single_scale_compression_does_not_fail_svd() {
    let f = |x: f64| (2.0 * std::f64::consts::PI * x).sin();
    let opts = InterpolativeQttOptions::default();

    let tt = interpolate_single_scale(f, 0.0, 1.0, 3, 600, &opts).unwrap();

    assert_eq!(tt.len(), 3);
    assert!(tt.link_dims().iter().all(|&dim| dim <= 601));
}

#[test]
fn multiscale_interpolation_one_over_x() {
    let r = 12;
    let a = 0.0;
    let b = 1.0;
    let degree = 25;
    let f = |x: f64| if x == 0.0 { 0.0 } else { 1.0 / x };
    let opts = InterpolativeQttOptions::default().with_tolerance(1e-12);

    let tt = interpolate_multi_scale(f, a, b, r, degree, &[0.0], &opts).unwrap();
    assert!(tt.rank() <= degree + 2);

    let grid = grid_1d(r, a, b);
    let n = 1usize << r;
    let mut diff_norm = 0.0;
    let mut ref_norm = 0.0;
    for i in 0..n {
        let grid_idx = [i];
        let quantics = grid.grididx_to_quantics(&grid_idx).unwrap();
        let x = grid.grididx_to_origcoord(&grid_idx).unwrap()[0];
        let value = tt.evaluate(&quantics_to_indices(&quantics)).unwrap();
        let expected = f(x);
        diff_norm += (value - expected).powi(2);
        ref_norm += expected.powi(2);
    }
    assert!((diff_norm.sqrt() / ref_norm.sqrt()) < 1e-11);
}

#[test]
fn estimate_interpolation_error_1d() {
    let basis = get_chebyshev_grid(5).unwrap();
    let err =
        estimate_interpolation_error(|x| (2.0 * std::f64::consts::PI * x).sin(), 0.0, 1.0, &basis)
            .unwrap();
    assert!(err >= 0.0);
    assert!(err < 1.0);
}

#[test]
fn estimate_interpolation_error_n2() {
    let basis = get_chebyshev_grid(5).unwrap();
    let err = estimate_interpolation_error_nd(
        |x| (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).cos(),
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &basis,
    )
    .unwrap();
    assert!(err >= 0.0);
    assert!(err < 1.0);
}

#[test]
fn estimate_interpolation_error_n3() {
    let basis = get_chebyshev_grid(10).unwrap();
    let err = estimate_interpolation_error_nd(
        |x| (-x[0] * x[0] - x[1] * x[1] - x[2] * x[2]).exp(),
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, 1.0],
        &basis,
    )
    .unwrap();
    assert!(err >= 0.0);
    assert!(err < 1.0e-8);
}

#[test]
fn invert_qtt_uncompressed_stage1() {
    let r = 12;
    let degree = 10;
    let a = 0.0;
    let b = 1.0;
    let f = |x: f64| (-x * x).exp();
    let basis = get_chebyshev_grid(degree).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);

    let tt = interpolate_single_scale(f, a, b, r, degree, &opts).unwrap();
    let result = invert_qtt(&tt, &basis, 1).unwrap();

    let k_out = r - 1;
    let s = &result[k_out - 1];
    assert_eq!(s.nrows(), 1usize << k_out);
    assert_eq!(s.ncols(), degree + 1);

    let mut max_err = 0.0_f64;
    for i in 0..(1usize << k_out) {
        for beta in 0..=degree {
            let x_ref = a + (b - a) * (i as f64 + basis.grid()[beta]) / (1usize << k_out) as f64;
            max_err = max_err.max((s[[i, beta]] - f(x_ref)).abs());
        }
    }
    assert!(max_err < 1.0e-6);
}

#[test]
fn invert_qtt_compressed() {
    let r = 12;
    let degree = 10;
    let a = 0.0;
    let b = 1.0;
    let f = |x: f64| (-x * x).exp();
    let basis = get_chebyshev_grid(degree).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(1e-10);

    let tt = interpolate_single_scale(f, a, b, r, degree, &opts).unwrap();
    let result = invert_qtt(&tt, &basis, 1).unwrap();

    let k_out = r - 1;
    let s = &result[k_out - 1];
    let mut max_err = 0.0_f64;
    for i in 0..(1usize << k_out) {
        for beta in 0..=degree {
            let x_ref = a + (b - a) * (i as f64 + basis.grid()[beta]) / (1usize << k_out) as f64;
            max_err = max_err.max((s[[i, beta]] - f(x_ref)).abs());
        }
    }
    assert!(max_err < 1.0e-6);
}

#[test]
fn invert_qtt_round_trip_uncompressed() {
    let r = 8;
    let degree = 10;
    let a = 0.0;
    let b = 1.0;
    let f = |x: f64| (-x * x).exp();
    let basis = get_chebyshev_grid(degree).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);

    let tt = interpolate_single_scale(f, a, b, r, degree, &opts).unwrap();
    let result = invert_qtt(&tt, &basis, 1).unwrap();
    let s = &result[r - 2];
    let grid = grid_1d(r, a, b);

    let mut max_err = 0.0_f64;
    for i in 0..(1usize << (r - 1)) {
        let left_idx = [2 * i];
        let right_idx = [2 * i + 1];
        let left_q = grid.grididx_to_quantics(&left_idx).unwrap();
        let right_q = grid.grididx_to_quantics(&right_idx).unwrap();
        let tt_left = tt.evaluate(&quantics_to_indices(&left_q)).unwrap();
        let tt_right = tt.evaluate(&quantics_to_indices(&right_q)).unwrap();

        let mut v_left = 0.0;
        let mut v_right = 0.0;
        for beta in 0..=degree {
            v_left += basis.evaluate(beta, 0.0).unwrap() * s[[i, beta]];
            v_right += basis.evaluate(beta, 0.5).unwrap() * s[[i, beta]];
        }
        max_err = max_err.max((v_left - tt_left).abs());
        max_err = max_err.max((v_right - tt_right).abs());
    }
    assert!(max_err < 1.0e-11);
}

#[test]
fn invert_qtt_round_trip_compressed() {
    let r = 10;
    let degree = 10;
    let a = 0.0;
    let b = 1.0;
    let f = |x: f64| (-x * x).exp();
    let basis = get_chebyshev_grid(degree).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(1e-8);

    let tt = interpolate_single_scale(f, a, b, r, degree, &opts).unwrap();
    let result = invert_qtt(&tt, &basis, 1).unwrap();
    let s = &result[r - 2];
    let grid = grid_1d(r, a, b);

    let mut max_err = 0.0_f64;
    for i in 0..(1usize << (r - 1)) {
        let left_idx = [2 * i];
        let right_idx = [2 * i + 1];
        let left_q = grid.grididx_to_quantics(&left_idx).unwrap();
        let right_q = grid.grididx_to_quantics(&right_idx).unwrap();
        let tt_left = tt.evaluate(&quantics_to_indices(&left_q)).unwrap();
        let tt_right = tt.evaluate(&quantics_to_indices(&right_q)).unwrap();

        let mut v_left = 0.0;
        let mut v_right = 0.0;
        for beta in 0..=degree {
            v_left += basis.evaluate(beta, 0.0).unwrap() * s[[i, beta]];
            v_right += basis.evaluate(beta, 0.5).unwrap() * s[[i, beta]];
        }
        max_err = max_err.max((v_left - tt_left).abs());
        max_err = max_err.max((v_right - tt_right).abs());
    }
    assert!(max_err < 1.0e-6);
}

#[test]
fn invert_qtt_stage2_coarse_levels() {
    let r = 10;
    let degree = 8;
    let a = -1.0;
    let b = 1.0;
    let f = |x: f64| (std::f64::consts::PI * x).cos();
    let basis = get_chebyshev_grid(degree).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);

    let tt = interpolate_single_scale(f, a, b, r, degree, &opts).unwrap();
    let result = invert_qtt(&tt, &basis, 1).unwrap();

    for (level, s) in result.iter().enumerate() {
        let k = level + 1;
        let mut max_err = 0.0_f64;
        for i in 0..s.nrows() {
            for beta in 0..=degree {
                let x_ref = a + (b - a) * (i as f64 + basis.grid()[beta]) / (1usize << k) as f64;
                max_err = max_err.max((s[[i, beta]] - f(x_ref)).abs());
            }
        }
        assert!(max_err < 1.0e-4, "level {k}: {max_err}");
    }
}

#[test]
fn sparse_and_adaptive_public_api_smoke_tests() {
    let opts = InterpolativeQttOptions::default();

    let sparse = interpolate_single_scale_sparse(|x| x.sin(), 0.0, 1.0, 4, 8, 2, &opts).unwrap();
    assert_eq!(sparse.len(), 4);

    let sparse_nd = interpolate_single_scale_sparse_nd(
        |x| x[0] + x[1],
        &[0.0, 0.0],
        &[1.0, 1.0],
        3,
        6,
        2,
        &opts,
    )
    .unwrap();
    assert_eq!(sparse_nd.site_dims(), vec![4, 4, 4]);

    let adaptive = interpolate_adaptive(|x| x.sin(), 0.0, 1.0, 4, 6, 1e-8, &[], &opts).unwrap();
    assert_eq!(adaptive.len(), 4);

    let adaptive_nd = interpolate_adaptive_nd(
        |x| x[0] + x[1],
        &[0.0, 0.0],
        &[1.0, 1.0],
        3,
        4,
        1e-8,
        &[],
        &opts,
    )
    .unwrap();
    assert_eq!(adaptive_nd.site_dims(), vec![4, 4, 4]);
}

#[test]
fn validation_errors() {
    let opts = InterpolativeQttOptions::default();
    assert!(get_chebyshev_grid(0).is_err());
    assert!(interpolate_single_scale(|x| x, 0.0, 1.0, 1, 4, &opts).is_err());
    assert!(interpolate_single_scale_nd(|x| x[0], &[0.0], &[1.0, 2.0], 3, 4, &opts).is_err());
    assert!(interpolate_single_scale_sparse(|x| x, 0.0, 1.0, 3, 3, 2, &opts).is_err());
}

#[test]
fn interpolation_size_helpers_reject_overflow() {
    assert!(checked_dim_add(usize::MAX, 1, "test").is_err());
    assert!(flatten_index(&[usize::MAX, usize::MAX], 2).is_err());
    assert!(dense_test_points(usize::MAX).is_err());
}

/// Degree 1: the two stored samples determine the whole cell polynomial, so
/// the finest matrix is exact at every node, for compressed and uncompressed
/// trains alike. Small `r` keeps `h` large, so no tolerance can hide an error.
#[test]
fn invert_qtt_degree1_exact_at_every_node() {
    let r = 3;
    let f = |x: f64| 2.0 * x + 1.0;
    let basis = get_chebyshev_grid(1).unwrap();
    for tolerance in [0.0, 1.0e-12] {
        let opts = InterpolativeQttOptions::default().with_tolerance(tolerance);
        let tt = interpolate_single_scale(f, 0.0, 1.0, r, 1, &opts).unwrap();
        let s = &invert_qtt(&tt, &basis, 1).unwrap()[r - 2];
        for i in 0..(1usize << (r - 1)) {
            for beta in 0..=1 {
                let x = (i as f64 + basis.grid()[beta]) / (1usize << (r - 1)) as f64;
                assert!((s[[i, beta]] - f(x)).abs() < 1.0e-12, "tol {tolerance}");
            }
        }
    }
}

/// Degree 2 at small `r`: nodes `c = 0, 1/2` are exact (exact comparison), and
/// the `c = 1` node is the documented two-sample blend `2 p(1/2) - p(0)` of the
/// samples the QTT stores, for compressed and uncompressed trains alike.
#[test]
fn invert_qtt_degree2_finest_level_is_documented_two_sample_blend() {
    let r = 3;
    let degree = 2;
    let f = |x: f64| x * x;
    let basis = get_chebyshev_grid(degree).unwrap();
    let grid = grid_1d(r, 0.0, 1.0);
    for tolerance in [0.0, 1.0e-12] {
        let opts = InterpolativeQttOptions::default().with_tolerance(tolerance);
        let tt = interpolate_single_scale(f, 0.0, 1.0, r, degree, &opts).unwrap();
        let s = &invert_qtt(&tt, &basis, 1).unwrap()[r - 2];
        for i in 0..(1usize << (r - 1)) {
            let sample = |idx: usize| {
                let q = grid.grididx_to_quantics(&[idx]).unwrap();
                tt.evaluate(&quantics_to_indices(&q)).unwrap()
            };
            let (p0, p_half) = (sample(2 * i), sample(2 * i + 1));
            let h = 1.0 / (1usize << (r - 1)) as f64;
            assert!((p0 - f(i as f64 * h)).abs() < 1.0e-12);
            assert!((s[[i, 0]] - p0).abs() < 1.0e-12);
            assert!((s[[i, 1]] - p_half).abs() < 1.0e-12);
            assert!((s[[i, 2]] - (2.0 * p_half - p0)).abs() < 1.0e-12);
        }
    }
}

#[test]
fn invert_qtt_uncompressed_degree2_exact_at_every_level() {
    let basis = get_chebyshev_grid(2).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    for r in [2, 3, 4] {
        let tt = interpolate_single_scale(|x| x * x, 0.0, 1.0, r, 2, &opts).unwrap();
        let values = invert_qtt_uncompressed(&tt, &basis, 1).unwrap();
        for (level, matrix) in values.iter().enumerate() {
            let cells = 1usize << (level + 1);
            let mut max_error = 0.0_f64;
            for row in 0..cells {
                for (col, &c) in basis.grid().iter().enumerate() {
                    let x = (row as f64 + c) / cells as f64;
                    max_error = max_error.max((matrix[[row, col]] - x * x).abs());
                }
            }
            assert!(max_error < 1.0e-12, "r={r}, level={level}: {max_error}");
        }
        assert_eq!(values[r - 2][[(1usize << (r - 1)) - 1, 2]], 1.0);
    }
}

#[test]
fn invert_qtt_uncompressed_recovers_higher_degree_parent_polynomials() {
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    let f = |x: f64| x * x * x - 2.0 * x + 1.0;
    for degree in [3, 6] {
        let basis = get_chebyshev_grid(degree).unwrap();
        for r in [2, 3, 4] {
            let tt = interpolate_single_scale(f, -1.0, 2.0, r, degree, &opts).unwrap();
            let values = invert_qtt_uncompressed(&tt, &basis, 1).unwrap();
            for (level, matrix) in values.iter().enumerate() {
                let mut max_error = 0.0_f64;
                for row in 0..matrix.nrows() {
                    for (col, &c) in basis.grid().iter().enumerate() {
                        let x = -1.0 + 3.0 * (row as f64 + c) / matrix.nrows() as f64;
                        max_error = max_error.max((matrix[[row, col]] - f(x)).abs());
                    }
                }
                assert!(
                    max_error < 1.0e-12,
                    "degree={degree}, r={r}, level={level}: {max_error}"
                );
            }
        }
    }
}

#[test]
fn invert_qtt_uncompressed_accepts_basis_rebuilt_from_identical_nodes() {
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    for degree in [3, 5, 7] {
        let original = get_chebyshev_grid(degree).unwrap();
        let rebuilt = LagrangePolynomials::new(original.grid().to_vec()).unwrap();
        assert_eq!(original.grid(), rebuilt.grid());
        let tt = interpolate_single_scale(|x| x * x, 0.0, 1.0, 3, degree, &opts).unwrap();
        let values = invert_qtt_uncompressed(&tt, &rebuilt, 1).unwrap();
        for matrix in &values {
            for row in 0..matrix.nrows() {
                for (col, &c) in rebuilt.grid().iter().enumerate() {
                    let x = (row as f64 + c) / matrix.nrows() as f64;
                    assert!((matrix[[row, col]] - x * x).abs() < 1.0e-12);
                }
            }
        }
    }
}

#[test]
fn invert_qtt_uncompressed_rejects_compressed_or_mismatched_basis() {
    let basis = get_chebyshev_grid(2).unwrap();
    for opts in [
        InterpolativeQttOptions::default(),
        InterpolativeQttOptions::default()
            .with_tolerance(0.0)
            .with_max_bond_dim(2),
    ] {
        let tt = interpolate_single_scale(|x| x * x, 0.0, 1.0, 3, 2, &opts).unwrap();
        assert!(matches!(
            invert_qtt_uncompressed(&tt, &basis, 1),
            Err(InterpolativeQttError::InvalidArgument { .. })
        ));
    }

    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    let tt = interpolate_single_scale(|x| x * x, 0.0, 1.0, 3, 2, &opts).unwrap();
    for wrong_basis in [
        get_chebyshev_grid(3).unwrap(),
        LagrangePolynomials::new(vec![0.0, 0.25, 1.0]).unwrap(),
        LagrangePolynomials::new(vec![0.0, 0.5, 0.75]).unwrap(),
    ] {
        assert!(matches!(
            invert_qtt_uncompressed(&tt, &wrong_basis, 1),
            Err(InterpolativeQttError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn invert_qtt_uncompressed_rejects_changed_gauge_with_unchanged_bond_dimensions() {
    let basis = get_chebyshev_grid(2).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    let mut tt = interpolate_single_scale(|x| x * x, 0.0, 1.0, 3, 2, &opts).unwrap();
    let before = invert_qtt(&tt, &basis, 1).unwrap();
    for (site, factor) in [(1, 0.5), (2, 2.0)] {
        let core = &mut tt.site_tensors_mut()[site];
        for left in 0..core.left_dim() {
            for digit in 0..core.site_dim() {
                for right in 0..core.right_dim() {
                    core.set3(left, digit, right, *core.get3(left, digit, right) * factor);
                }
            }
        }
    }
    assert_eq!(tt.link_dims(), vec![3, 3]);
    let after = invert_qtt(&tt, &basis, 1).unwrap();
    for (before, after) in before.iter().zip(&after) {
        for row in 0..before.nrows() {
            for col in 0..before.ncols() {
                assert_eq!(before[[row, col]], after[[row, col]]);
            }
        }
    }
    assert!(matches!(
        invert_qtt_uncompressed(&tt, &basis, 1),
        Err(InterpolativeQttError::InvalidArgument { .. })
    ));
}

#[test]
fn invert_qtt_entry_points_reject_invalid_inputs() {
    let basis = get_chebyshev_grid(2).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    let tt = interpolate_single_scale(|x| x * x, 0.0, 1.0, 3, 2, &opts).unwrap();
    for inverse in [invert_qtt, invert_qtt_uncompressed] {
        for q in [0, 2, 3] {
            assert!(matches!(
                inverse(&tt, &basis, q),
                Err(InterpolativeQttError::InvalidArgument { .. })
            ));
        }
        for dims in [&[][..], &[2][..], &[2, 3][..]] {
            let invalid = SimpleTensorTrain::constant(dims, 1.0);
            assert!(matches!(
                inverse(&invalid, &basis, 1),
                Err(InterpolativeQttError::InvalidArgument { .. })
            ));
        }
        for (site, shape) in [(0, [2, 2, 3]), (2, [3, 2, 2]), (1, [4, 2, 3])] {
            let mut invalid = tt.clone();
            invalid.site_tensors_mut()[site] = Tensor3::from_elem(shape, 1.0).unwrap();
            assert!(matches!(
                inverse(&invalid, &basis, 1),
                Err(InterpolativeQttError::InvalidArgument { .. })
            ));
        }
    }
}
