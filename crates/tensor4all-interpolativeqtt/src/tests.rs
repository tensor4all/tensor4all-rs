use approx::assert_relative_eq;
use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, IndexLike};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

use crate::core_array::{core_at, core_from_fn, set_core};

use crate::interpolation::{checked_dim_add, dense_test_points, flatten_index};
use crate::interval::NInterval;

use super::*;

mod complex;
mod heavy_single_scale;
mod refined_batches;
mod split;

#[test]
fn fused_tree_result_orders_levels_msb_first_and_axis_zero_fastest() {
    let (tree, sites) = interpolate_single_scale_nd(
        |x| 1.0 + x[0] + 2.0 * x[1],
        &[0.0, 0.0],
        &[1.0, 1.0],
        3,
        2,
        &InterpolativeQttOptions::default(),
    )
    .unwrap();
    // x = 4/8 and y = 2/8, with axis zero at weight one in each digit.
    let value = tree.evaluate_point(&sites, &[1, 2, 0]).unwrap();
    assert!((value.real() - 2.0).abs() < 1e-12);
}

#[test]
fn inverse_rejects_mismatched_order_and_dtype() {
    let (tree, sites) =
        interpolate_single_scale(|x| 1.0 + x, 0.0, 1.0, 3, 2, &Default::default()).unwrap();
    let basis = get_chebyshev_grid(2).unwrap();
    let mut reversed = sites.clone();
    reversed.reverse();
    assert!(matches!(
        invert_qtt::<f64>(&tree, &reversed, &basis, 1),
        Err(InterpolativeQttError::Tree(_))
    ));
    assert!(matches!(
        invert_qtt::<num_complex::Complex64>(&tree, &sites, &basis, 1),
        Err(InterpolativeQttError::Tree(_))
    ));
}

fn evaluate<T: InterpolativeScalar>(qtt: &InterpolativeQtt, index: &[usize]) -> T {
    evaluate_many(qtt, index)[0]
}

fn evaluate_many<T: InterpolativeScalar>(
    (tree, sites): &InterpolativeQtt,
    indices: &[usize],
) -> Vec<T> {
    let shape = [sites.len(), indices.len() / sites.len()];
    let values = ColMajorArrayRef::new(indices, &shape).unwrap();
    TreeTNCachedEvaluator::new(tree, sites, CachedEvaluatorOptions::default())
        .unwrap()
        .evaluate_batched_typed(values, EvaluationHint::default())
        .unwrap()
}

fn rank(qtt: &InterpolativeQtt) -> usize {
    qtt.0.link_dims().into_iter().max().unwrap_or(1)
}

fn site_dims(qtt: &InterpolativeQtt) -> Vec<usize> {
    qtt.1.iter().map(IndexLike::dim).collect()
}

fn invert<T: InterpolativeScalar>(
    qtt: &InterpolativeQtt,
    basis: &LagrangePolynomials,
) -> Vec<tensor4all_tensorbackend::Matrix<T>> {
    invert_qtt::<T>(&qtt.0, &qtt.1, basis, 1).unwrap()
}

fn finest_level_scale<T: InterpolativeScalar>(
    qtt: &InterpolativeQtt,
    basis: &LagrangePolynomials,
) -> Result<Option<T>> {
    crate::interpolation::finest_level_scale(&qtt.0.chain_cores::<T>(&qtt.1)?, basis)
}

fn rewrite_core(qtt: &mut InterpolativeQtt, level: usize, f: impl FnOnce(&mut ColMajorArray<f64>)) {
    let mut cores = qtt.0.chain_cores(&qtt.1).unwrap();
    f(&mut cores[level]);
    *qtt = TreeTN::from_chain_cores(cores).unwrap();
}

fn scale_finest(qtt: &mut InterpolativeQtt, factor: f64) {
    rewrite_core(qtt, qtt.1.len() - 1, |core| {
        core.data_mut().iter_mut().for_each(|x| *x *= factor);
    });
}

fn grid_1d(r: usize, a: f64, b: f64) -> DiscretizedGrid {
    DiscretizedGrid::builder(&[r])
        .with_lower_bound(&[a])
        .with_upper_bound(&[b])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap()
}

/// The degree-`degree` interpolant of `f` on the parent cell of the finest cell
/// `i`, evaluated at that parent cell's local coordinate `t`: the value the exact
/// finest level has to reproduce (tensor4all-rs#1175).
fn parent_interpolant(
    f: &impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    k_out: usize,
    degree: usize,
    i: usize,
    t: f64,
) -> f64 {
    let basis = get_chebyshev_grid(degree).unwrap();
    let parents = 1usize << (k_out - 1);
    let parent = i / 2;
    (0..=degree)
        .map(|alpha| {
            let x = a + (b - a) * (parent as f64 + basis.grid()[alpha]) / parents as f64;
            basis.evaluate(alpha, t).unwrap() * f(x)
        })
        .sum()
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

fn assert_tt_matches_grid_1d<T: InterpolativeScalar + std::fmt::Debug>(
    tt: &InterpolativeQtt,
    grid: &DiscretizedGrid,
    f: impl Fn(f64) -> T,
    tol: f64,
) {
    assert_tt_matches_grid_nd(tt, grid, |coords| f(coords[0]), tol);
}

fn assert_tt_matches_grid_nd<T: InterpolativeScalar + std::fmt::Debug>(
    tt: &InterpolativeQtt,
    grid: &DiscretizedGrid,
    f: impl Fn(&[f64]) -> T,
    tol: f64,
) {
    let mut indices = Vec::new();
    let mut expected = Vec::new();
    for_each_grid_index(grid.ndims(), 1usize << grid.rs()[0], |idx| {
        indices.extend(grid.grididx_to_quantics(idx).unwrap());
        expected.push(f(&grid.grididx_to_origcoord(idx).unwrap()));
    });
    let actual = evaluate_many::<T>(tt, &indices);
    for (flat, (value, expected)) in actual.into_iter().zip(expected).enumerate() {
        let error = (value - expected).abs_val();
        assert!(
            error < tol,
            "grid offset {flat}: value {value:?}, expected {expected:?}, diff {error}"
        );
    }
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
    assert!(rank(&tt) <= degree + 1);

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
    assert!(rank(&tt) <= degree + 1);

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
    assert!(rank(&tt) <= (degree + 1).pow(2));
    assert_eq!(tt.1.len(), r);

    let grid = grid_nd(r, &lower, &upper);
    assert_tt_matches_grid_nd(&tt, &grid, f, 1e-10);
}

#[test]
fn single_scale_interpolation_n3() {
    // Degree two represents this asymmetric, coupled polynomial exactly.
    assert_single_scale_interpolation_n3(2, |x| {
        1.0 + x[0]
            + 2.0 * x[1] * x[1]
            + 3.0 * x[2]
            + x[0] * x[1]
            + 2.0 * x[1] * x[2]
            + 3.0 * x[0] * x[2]
            + x[0] * x[1] * x[2]
    });
}

fn assert_single_scale_interpolation_n3(degree: usize, f: impl Fn(&[f64]) -> f64) {
    // Three sites retain left, interior, and right cores while keeping the full
    // three-dimensional grid check to 8^3 points. The n2 fixture above retains
    // repeated interior cores (four sites).
    let r = 3;
    let lower = [-1.0, -1.0, 0.0];
    let upper = [2.0, 2.0, 1.0];

    let tt = interpolate_single_scale_nd(
        &f,
        &lower,
        &upper,
        r,
        degree,
        &InterpolativeQttOptions::default(),
    )
    .unwrap();
    assert!(rank(&tt) <= (degree + 1).pow(3));
    assert_eq!(tt.1.len(), r);

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
    assert!(rank(&tt) <= degree + 2);

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
    assert!(rank(&tt) <= (degree + 2).pow(2));

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
fn fused_core_rejects_missing_or_invalid_axes() {
    let wrong_rank = ColMajorArray::new(vec![0.0], vec![1, 1]).unwrap();
    let empty_axis = ColMajorArray::new(vec![], vec![1, 0, 1]).unwrap();
    for input in [vec![], vec![wrong_rank], vec![empty_axis]] {
        assert!(matches!(
            direct_product_core_tensors(&input),
            Err(InterpolativeQttError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn direct_product_core_tensors_two_tensors() {
    let chi = 3;
    let c1 = core_from_fn([chi, 2, chi], |[i, s, k]| (100 * i + 10 * s + k) as f64).unwrap();
    let c2 = core_from_fn([chi, 2, chi], |[i, s, k]| (1 + 100 * i + 10 * s + k) as f64).unwrap();
    let fused = direct_product_core_tensors(&[c1.clone(), c2.clone()]).unwrap();

    assert_eq!(fused.shape()[0], chi * chi);
    assert_eq!(fused.shape()[1], 4);
    assert_eq!(fused.shape()[2], chi * chi);
    for i in 0..chi {
        for j in 0..chi {
            for k in 0..2 {
                for l in 0..2 {
                    for m in 0..chi {
                        for n in 0..chi {
                            let left = i + chi * j;
                            let site = k + 2 * l;
                            let right = m + chi * n;
                            let expected = core_at(&c1, i, k, m) * core_at(&c2, j, l, n);
                            assert_relative_eq!(core_at(&fused, left, site, right), expected);
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
    let c1 = core_from_fn([chi, 2, chi], |[i, s, k]| (100 * i + 10 * s + k) as f64).unwrap();
    let c2 = core_from_fn([chi, 2, chi], |[i, s, k]| (1 + 100 * i + 10 * s + k) as f64).unwrap();
    let c3 = core_from_fn([chi, 2, chi], |[i, s, k]| (2 + 100 * i + 10 * s + k) as f64).unwrap();
    let fused = direct_product_core_tensors(&[c1.clone(), c2.clone(), c3.clone()]).unwrap();

    assert_eq!(fused.shape()[0], chi * chi * chi);
    assert_eq!(fused.shape()[1], 8);
    assert_eq!(fused.shape()[2], chi * chi * chi);
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
                                        let expected = core_at(&c1, i, l, o)
                                            * core_at(&c2, j, m, p)
                                            * core_at(&c3, k, n, q);
                                        assert_relative_eq!(
                                            core_at(&fused, left, site, right),
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
    for left in 0..core.shape()[0] {
        for site in 0..core.shape()[1] {
            for right in 0..core.shape()[2] {
                assert!(core_at(&core, left, site, right).is_finite());
            }
        }
    }
}

#[test]
fn high_degree_single_scale_compression_does_not_fail_svd() {
    let f = |x: f64| (2.0 * std::f64::consts::PI * x).sin();
    let opts = InterpolativeQttOptions::default();

    let tt = interpolate_single_scale(f, 0.0, 1.0, 3, 600, &opts).unwrap();

    assert_eq!(tt.1.len(), 3);
    assert!(tt.0.link_dims().iter().all(|&dim| dim <= 601));
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
    assert!(rank(&tt) <= degree + 2);

    let grid = grid_1d(r, a, b);
    let n = 1usize << r;
    let mut diff_norm = 0.0;
    let mut ref_norm = 0.0;
    for i in 0..n {
        let grid_idx = [i];
        let quantics = grid.grididx_to_quantics(&grid_idx).unwrap();
        let x = grid.grididx_to_origcoord(&grid_idx).unwrap()[0];
        let value = evaluate::<f64>(&tt, &quantics);
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

/// The exact route follows whether a bond was truncated, not whether the
/// function has low rank: a low-rank function carried by a full-rank train still
/// takes it, and the same function compressed onto a shorter bond does not
/// (tensor4all-rs#1175).
#[test]
fn invert_qtt_exact_finest_level_follows_truncation_not_the_function_rank() {
    let r = 8;
    let degree = 10;
    let a = 0.0;
    let b = 1.0;
    // A quadratic: rank three in exact arithmetic, reproduced by degree 10 on any
    // cell, so the parent interpolant is the reference at every node.
    let f = |x: f64| x * x + 1.0;
    let basis = get_chebyshev_grid(degree).unwrap();
    let k_out = r - 1;

    for (tolerance, exact) in [(0.0, true), (1.0e-10, false)] {
        let opts = InterpolativeQttOptions::default().with_tolerance(tolerance);
        let tt = interpolate_single_scale(f, a, b, r, degree, &opts).unwrap();
        assert_eq!(
            finest_level_scale::<f64>(&tt, &basis).unwrap().is_some(),
            exact,
            "tolerance {tolerance} with bonds {:?}",
            tt.0.link_dims()
        );
        let s = &invert::<f64>(&tt, &basis)[k_out - 1];
        let mut max_rel = 0.0_f64;
        for i in 0..(1usize << k_out) {
            for beta in 0..=degree {
                let t = ((i % 2) as f64 + basis.grid()[beta]) / 2.0;
                let reference = parent_interpolant(&f, a, b, k_out, degree, i, t);
                max_rel = max_rel.max((s[[i, beta]] - reference).abs() / reference.abs());
            }
        }
        if exact {
            assert!(max_rel <= 1.0e-14, "uncompressed: {max_rel:e}");
        } else {
            // The blend is what the compressed train keeps, and it is far off.
            assert!(max_rel > 1.0e-9, "compressed blend: {max_rel:e}");
        }
    }
}

/// Scaling the finest core keeps the exact route, including the overall factor.
#[test]
fn invert_qtt_rescaled_train_keeps_the_exact_finest_level() {
    let r = 8;
    let degree = 10;
    let a = 0.0;
    let b = 1.0;
    let f = |x: f64| x * x + 1.0;
    let basis = get_chebyshev_grid(degree).unwrap();
    let k_out = r - 1;
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);

    let mut tt = interpolate_single_scale(f, a, b, r, degree, &opts).unwrap();
    let before = tt.0.link_dims();
    scale_finest(&mut tt, 2.0);
    assert_eq!(tt.0.link_dims(), before, "scaling must not change a bond");
    assert_eq!(finest_level_scale::<f64>(&tt, &basis).unwrap(), Some(2.0));

    let s = &invert::<f64>(&tt, &basis)[k_out - 1];
    let mut max_rel = 0.0_f64;
    for i in 0..(1usize << k_out) {
        for beta in 0..=degree {
            let t = ((i % 2) as f64 + basis.grid()[beta]) / 2.0;
            let reference = 2.0 * parent_interpolant(&f, a, b, k_out, degree, i, t);
            max_rel = max_rel.max((s[[i, beta]] - reference).abs() / reference.abs());
        }
    }
    assert!(max_rel <= 1.0e-14, "rescaled train: {max_rel:e}");
}

/// A finest core that deviates from `L_alpha(sigma / 2)` beyond the proportional
/// bound is rejected: it is multiplied by the parent coefficients, which can be
/// large, so the decode error can exceed the deviation by orders of magnitude. The
/// blend is the safe answer, and the test pins the rejection rather than a tolerance.
#[test]
fn invert_qtt_perturbed_finest_core_falls_back_to_the_blend() {
    let r = 8;
    let degree = 10;
    let a = 0.0;
    let b = 1.0;
    let f = |x: f64| x * x + 1.0;
    let basis = get_chebyshev_grid(degree).unwrap();
    let k_out = r - 1;
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);

    let mut tt = interpolate_single_scale(f, a, b, r, degree, &opts).unwrap();
    assert!(finest_level_scale::<f64>(&tt, &basis).unwrap().is_some());
    rewrite_core(&mut tt, r - 1, |last| {
        let perturbed = core_at(last, 2, 0, 0) + 5.0e-13;
        set_core(last, 2, 0, 0, perturbed);
    });
    assert_eq!(finest_level_scale::<f64>(&tt, &basis).unwrap(), None);

    // The route really changed: the blend is O(h^2) away from the parent interpolant.
    let s = &invert::<f64>(&tt, &basis)[k_out - 1];
    let mut max_rel = 0.0_f64;
    for i in 0..(1usize << k_out) {
        for beta in 0..=degree {
            let t = ((i % 2) as f64 + basis.grid()[beta]) / 2.0;
            let reference = parent_interpolant(&f, a, b, k_out, degree, i, t);
            max_rel = max_rel.max((s[[i, beta]] - reference).abs() / reference.abs());
        }
    }
    assert!(
        max_rel > 1.0e-9,
        "perturbed core kept the exact route: {max_rel:e}"
    );
}

/// A gauge that scales a coefficient column of an earlier core leaves every stored
/// sample intact — the basis is cardinal at the sampled points, so a column whose
/// basis function vanishes there is invisible to `evaluate` — while the polynomial
/// the finest level reconstructs changes. The exact route therefore has to agree
/// with the train's own samples at the cell boundaries, and falls back to the blend
/// when it does not (tensor4all-rs#1175, #1198 review).
#[test]
fn invert_qtt_gauge_scaled_coefficient_column_does_not_reach_the_exact_route() {
    let r = 2;
    let degree = 2;
    let basis = get_chebyshev_grid(degree).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    let mut tt = interpolate_single_scale(|_| 1.0, 0.0, 1.0, r, degree, &opts).unwrap();
    // Column `alpha = 2` of the first core: `L_2` vanishes at both sampled points
    // (`t = 0` and `t = 1/2`), so this is a sample-preserving gauge.
    rewrite_core(&mut tt, 0, |core| {
        for site in 0..core.shape()[1] {
            let scaled = 2.0 * core_at(core, 0, site, 2);
            set_core(core, 0, site, 2, scaled);
        }
    });
    // Every stored sample is unchanged.
    for cell in 0..(1usize << (r - 1)) {
        for sigma in 0..2 {
            let value = evaluate::<f64>(&tt, &[cell, sigma]);
            assert!(
                (value - 1.0).abs() < 1.0e-15,
                "cell {cell} sigma {sigma}: {value}"
            );
        }
    }
    // The reconstructed finest level must still be the constant 1.
    let fine = &invert::<f64>(&tt, &basis)[r - 2];
    for cell in 0..fine.nrows() {
        for beta in 0..=degree {
            assert!(
                (fine[[cell, beta]] - 1.0).abs() < 1.0e-12,
                "cell {cell} node {beta}: {}",
                fine[[cell, beta]]
            );
        }
    }
}

/// The documented limit of the exact route, pinned so nobody mistakes it for a total
/// check: a rewrite whose component vanishes at every point the train stores —
/// `t = 0`, `t = 1/2` and the shared endpoints, for degree >= 3 a basis function such
/// as `L_1` — leaves all samples, all bonds, the finest core and the boundary residual
/// untouched, and the exact route then reports the rewritten coefficients. No
/// comparison inside `invert_qtt` can reject it: the train does not carry the
/// information. See the `# Accuracy of the finest level` contract
/// (tensor4all-rs#1175, #1198 review).
#[test]
fn invert_qtt_sample_preserving_interior_rewrite_is_documented_as_undetectable() {
    let r = 2;
    let degree = 4;
    let basis = get_chebyshev_grid(degree).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    let mut tt = interpolate_single_scale(|_| 1.0, 0.0, 1.0, r, degree, &opts).unwrap();
    // `L_1` vanishes at `0`, `1/2` and `1`, so doubling its coefficient is invisible
    // to every stored sample and to the boundary the check compares.
    rewrite_core(&mut tt, 0, |core| {
        let before = core_at(core, 0, 0, 1);
        set_core(core, 0, 0, 1, 2.0 * before);
    });
    assert_eq!(tt.0.link_dims(), vec![degree + 1]);
    assert!(finest_level_scale::<f64>(&tt, &basis).unwrap().is_some());
    for cell in 0..(1usize << (r - 1)) {
        for sigma in 0..2 {
            let value = evaluate::<f64>(&tt, &[cell, sigma]);
            assert!(
                (value - 1.0).abs() < 1.0e-15,
                "cell {cell} sigma {sigma}: {value}"
            );
        }
    }
    let fine = &invert::<f64>(&tt, &basis)[r - 2];
    // The nodes the train does store are still right; the interior one follows the
    // rewritten coefficient, which is the documented consequence.
    for beta in [0, degree / 2, degree] {
        assert!(
            (fine[[0, beta]] - 1.0).abs() < 1.0e-12,
            "node {beta}: {}",
            fine[[0, beta]]
        );
    }
    assert!(
        (fine[[0, 1]] - 2.0).abs() < 1.0e-12,
        "node 1: {} (the documented limitation has changed)",
        fine[[0, 1]]
    );
}

/// Scaling twice perturbs the finest core by a couple of ulps (the products do
/// not associate), so the canonical comparison has to be proportional rather than
/// bit-exact — otherwise a legitimately rescaled uncompressed train silently falls
/// back to the `O(h^2)` blend (tensor4all-rs#1175, #1198 review).
#[test]
fn invert_qtt_repeatedly_rescaled_train_keeps_the_exact_finest_level() {
    let r = 2;
    let degree = 3;
    let basis = get_chebyshev_grid(degree).unwrap();
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    let f = |x: f64| 1.0 + x * x;
    let mut tt = interpolate_single_scale(f, 0.0, 1.0, r, degree, &opts).unwrap();
    scale_finest(&mut tt, 0.1);
    scale_finest(&mut tt, 0.1);
    assert!(
        finest_level_scale::<f64>(&tt, &basis).unwrap().is_some(),
        "a rescaled uncompressed train must keep the exact route"
    );
    let fine = &invert::<f64>(&tt, &basis)[r - 2];
    // The last cell spans [1/2, 1]; the train was scaled by 0.1 twice.
    let last = fine.nrows() - 1;
    for beta in 0..=degree {
        let x = (last as f64 + basis.grid()[beta]) / fine.nrows() as f64;
        let expected = 0.01 * f(x);
        let actual = fine[[last, beta]];
        assert!(
            (actual - expected).abs() <= 1.0e-12 * expected.abs(),
            "cell {last} node {beta}: {actual} vs {expected}"
        );
    }
}

/// An uncompressed train keeps the canonical finest core, so the finest level is
/// evaluated in the parent basis instead of the two-point blend
/// (tensor4all-rs#1175). The blend is what carried the `O(h^2)` error, so the
/// interpolation error below is the construction's own.
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
    assert!(
        finest_level_scale::<f64>(&tt, &basis).unwrap().is_some(),
        "an uncompressed train has to take the exact finest level"
    );
    let result = invert::<f64>(&tt, &basis);

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

    // A quadratic is reproduced by the degree-10 interpolant on every cell, so the
    // directly evaluated parent interpolant is the reference the exact route has to
    // match; the blend would sit `O(h^2) = 2.4e-7` away from it at the interior
    // nodes.
    let g = |x: f64| x * x + 1.0;
    let exact_tt = interpolate_single_scale(g, a, b, r, degree, &opts).unwrap();
    assert!(finest_level_scale::<f64>(&exact_tt, &basis)
        .unwrap()
        .is_some());
    let exact = &invert::<f64>(&exact_tt, &basis)[k_out - 1];
    let mut max_rel = 0.0_f64;
    for i in 0..(1usize << k_out) {
        for beta in 0..=degree {
            let t = ((i % 2) as f64 + basis.grid()[beta]) / 2.0;
            let reference = parent_interpolant(&g, a, b, k_out, degree, i, t);
            max_rel = max_rel.max((exact[[i, beta]] - reference).abs() / reference.abs());
        }
    }
    assert!(
        max_rel <= 1.0e-14,
        "exact finest level differs from the parent interpolant by {max_rel:e}"
    );
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
    let result = invert::<f64>(&tt, &basis);

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

/// The largest distance between the finest inverted level of `f`'s interpolant
/// on `[0, 1)` and the train's own values at the two samples of each cell.
fn invert_round_trip_error<T: InterpolativeScalar>(
    f: impl Fn(f64) -> T,
    r: usize,
    degree: usize,
    opts: &InterpolativeQttOptions,
) -> f64 {
    let (a, b) = (0.0, 1.0);
    let basis = get_chebyshev_grid(degree).unwrap();

    let tt = interpolate_single_scale(f, a, b, r, degree, opts).unwrap();
    let result = invert::<T>(&tt, &basis);
    let s = &result[r - 2];
    let grid = grid_1d(r, a, b);

    let indices: Vec<_> = (0..1usize << r)
        .flat_map(|i| grid.grididx_to_quantics(&[i]).unwrap())
        .collect();
    let samples = evaluate_many::<T>(&tt, &indices);
    let (pairs, remainder) = samples.as_chunks::<2>();
    assert!(remainder.is_empty());
    let mut max_err = 0.0_f64;
    for (i, &[tt_left, tt_right]) in pairs.iter().enumerate() {
        let mut v_left = T::zero();
        let mut v_right = T::zero();
        for beta in 0..=degree {
            v_left = v_left + T::from_f64(basis.evaluate(beta, 0.0).unwrap()) * s[[i, beta]];
            v_right = v_right + T::from_f64(basis.evaluate(beta, 0.5).unwrap()) * s[[i, beta]];
        }
        max_err = max_err.max((v_left - tt_left).abs_val());
        max_err = max_err.max((v_right - tt_right).abs_val());
    }
    max_err
}

#[test]
fn invert_qtt_round_trip_uncompressed() {
    let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    let max_err = invert_round_trip_error(|x: f64| (-x * x).exp(), 8, 10, &opts);
    assert!(max_err < 1.0e-11);
}

#[test]
fn invert_qtt_round_trip_compressed() {
    let opts = InterpolativeQttOptions::default().with_tolerance(1e-8);
    let max_err = invert_round_trip_error(|x: f64| (-x * x).exp(), 10, 10, &opts);
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
    let result = invert::<f64>(&tt, &basis);

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
    assert_eq!(sparse.1.len(), 4);

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
    assert_eq!(site_dims(&sparse_nd), vec![4, 4, 4]);

    let adaptive = interpolate_adaptive(|x| x.sin(), 0.0, 1.0, 4, 6, 1e-8, &[], &opts).unwrap();
    assert_eq!(adaptive.1.len(), 4);

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
    assert_eq!(site_dims(&adaptive_nd), vec![4, 4, 4]);
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
        let s = &invert::<f64>(&tt, &basis)[r - 2];
        for i in 0..(1usize << (r - 1)) {
            for beta in 0..=1 {
                let x = (i as f64 + basis.grid()[beta]) / (1usize << (r - 1)) as f64;
                assert!((s[[i, beta]] - f(x)).abs() < 1.0e-12, "tol {tolerance}");
            }
        }
    }
}

/// Degree 2 at small `r`: the uncompressed train evaluates the `c = 1` node
/// exactly, while the compressed one keeps the documented two-sample blend
/// `2 p(1/2) - p(0)` of the samples the QTT stores. Nodes `c = 0, 1/2` are exact
/// for both, because the blend reproduces the two stored samples.
#[test]
fn invert_qtt_degree2_exact_when_uncompressed_blend_when_compressed() {
    let r = 3;
    let degree = 2;
    let f = |x: f64| x * x;
    let basis = get_chebyshev_grid(degree).unwrap();
    let grid = grid_1d(r, 0.0, 1.0);
    for (tolerance, exact) in [(0.0, true), (1.0e-12, false)] {
        let opts = InterpolativeQttOptions::default().with_tolerance(tolerance);
        let tt = interpolate_single_scale(f, 0.0, 1.0, r, degree, &opts).unwrap();
        assert_eq!(
            finest_level_scale::<f64>(&tt, &basis).unwrap().is_some(),
            exact
        );
        let s = &invert::<f64>(&tt, &basis)[r - 2];
        for i in 0..(1usize << (r - 1)) {
            let sample = |idx: usize| {
                let q = grid.grididx_to_quantics(&[idx]).unwrap();
                evaluate::<f64>(&tt, &q)
            };
            let (p0, p_half) = (sample(2 * i), sample(2 * i + 1));
            let h = 1.0 / (1usize << (r - 1)) as f64;
            assert!((p0 - f(i as f64 * h)).abs() < 1.0e-12);
            assert!((s[[i, 0]] - p0).abs() < 1.0e-12);
            assert!((s[[i, 1]] - p_half).abs() < 1.0e-12);
            let c_one = if exact {
                f((i + 1) as f64 * h)
            } else {
                2.0 * p_half - p0
            };
            assert!((s[[i, 2]] - c_one).abs() < 1.0e-12, "tolerance {tolerance}");
        }
    }
}
