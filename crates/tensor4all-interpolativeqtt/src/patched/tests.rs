use super::*;
use num_complex::Complex64;
use std::convert::Infallible;

fn batch<T: InterpolativeScalar>(
    points: ColMajorArrayRef<'_, f64>,
    f: impl Fn(&[f64]) -> T,
) -> std::result::Result<Vec<T>, Infallible> {
    Ok(points.data().chunks(points.shape()[0]).map(f).collect())
}

// Materialize each small patch once; compare all entries and exact grid ownership.
fn check_partition<T: InterpolativeScalar>(
    patches: &[InterpolativePatch<T>],
    ndims: usize,
    bits: usize,
    f: impl Fn(&[f64]) -> T,
    tolerance: f64,
) {
    let axis_size = 1usize << bits;
    let mut visits = vec![0; axis_size.pow(ndims as u32)];
    for patch in patches {
        let depth = patch.prefix[0].len();
        assert_eq!(patch.tt.len(), bits - depth);
        assert!(patch.prefix.iter().all(|axis| axis.len() == depth));
        assert!(
            patch.error_estimate <= tolerance,
            "estimate {}",
            patch.error_estimate
        );
        let (dense, _) = patch.tt.full_tensor().unwrap();
        let mut max_error = 0.0_f64;
        for (mut flat, &actual) in dense.iter().enumerate() {
            let mut indices: Vec<usize> = patch
                .prefix
                .iter()
                .map(|axis| axis.iter().fold(0, |n, &b| 2 * n + b))
                .collect();
            for _ in 0..bits - depth {
                let site = flat % (1 << ndims);
                flat >>= ndims;
                for (d, index) in indices.iter_mut().enumerate() {
                    *index = 2 * *index + ((site >> d) & 1);
                }
            }
            let global = indices.iter().rev().fold(0, |n, &i| n * axis_size + i);
            visits[global] += 1;
            let point: Vec<_> = indices
                .iter()
                .map(|&x| x as f64 / axis_size as f64)
                .collect();
            max_error = max_error.max((actual - f(&point)).abs_val());
        }
        assert!(
            max_error <= tolerance,
            "max error {max_error}, prefix {:?}",
            patch.prefix
        );
    }
    assert!(
        visits.iter().all(|&count| count == 1),
        "patches must be disjoint and cover the grid"
    );
    assert!(patches.windows(2).all(|p| p[0].prefix < p[1].prefix));
}

fn smooth<T: InterpolativeScalar>(f: impl Fn(&[f64]) -> T) {
    for dims in [1, 2] {
        let patches = interpolate_patched_nd(
            |p, _| batch(p, &f),
            &vec![0.0; dims],
            &vec![1.0; dims],
            4,
            3,
            &PatchedInterpolativeOptions::default(),
        )
        .unwrap();
        assert_eq!(patches.len(), 1);
        check_partition(&patches, dims, 4, &f, 1e-8);
    }
}

#[test]
fn smooth_real_and_complex_one_patch() {
    smooth(|x| 1.0 + x.iter().sum::<f64>());
    smooth(|x| Complex64::new(1.0 + x.iter().sum::<f64>(), 2.0 - x[0]));
}

fn lorentzian(x: &[f64]) -> f64 {
    1.0 / (1.0 + ((x[0] - 0.31) / 0.025).powi(2))
}

fn localized<T: InterpolativeScalar>(f: impl Fn(&[f64]) -> T) {
    for dims in [1, 2] {
        let options = PatchedInterpolativeOptions {
            tolerance: 1e-4,
            ..Default::default()
        };
        let patches = interpolate_patched_nd(
            |p, _| batch(p, &f),
            &vec![0.0; dims],
            &vec![1.0; dims],
            5,
            5,
            &options,
        )
        .unwrap();
        assert!(patches.len() > 1);
        let depths: Vec<_> = patches.iter().map(|p| p.prefix[0].len()).collect();
        assert!(
            depths.iter().min() < depths.iter().max(),
            "refinement must stay local: {depths:?}"
        );
        check_partition(&patches, dims, 5, &f, options.tolerance);
    }
}

#[test]
fn localized_real_and_complex_cover_and_refine() {
    localized(lorentzian);
    localized(|x| Complex64::new(lorentzian(x), 0.5 * lorentzian(x)));
}

#[test]
fn last_site_is_exact_and_rank_cap_triggers_splitting() {
    let options = PatchedInterpolativeOptions {
        max_bond_dim: Some(1),
        tolerance: 1e-8,
        ..Default::default()
    };
    let f = |x: &[f64]| (3.0 * x[0] + x.get(1).copied().unwrap_or(0.0)).exp();
    for dims in [1, 2] {
        let p = interpolate_patched_nd(
            |p, _| batch(p, f),
            &vec![0.0; dims],
            &vec![1.0; dims],
            3,
            5,
            &options,
        )
        .unwrap();
        assert!(p.len() > 1);
        assert!(p.iter().all(|p| p.tt.rank() == 1));
        check_partition(&p, dims, 3, f, options.tolerance);
    }
    let p = interpolate_patched_nd(|p, _| batch(p, lorentzian), &[0.0], &[1.0], 1, 2, &options)
        .unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].error_estimate, 0.0);
    check_partition(&p, 1, 1, lorentzian, options.tolerance);
}

#[test]
fn limits_and_nonfinite_values_are_errors() {
    let run = |options: &PatchedInterpolativeOptions| {
        interpolate_patched_nd(|p, _| batch(p, lorentzian), &[0.0], &[1.0], 4, 3, options)
    };
    assert!(matches!(
        run(&PatchedInterpolativeOptions {
            max_depth: Some(0),
            ..Default::default()
        }),
        Err(InterpolativeQttError::NotConverged { .. })
    ));
    assert!(matches!(
        run(&PatchedInterpolativeOptions {
            max_patches: 1,
            ..Default::default()
        }),
        Err(InterpolativeQttError::PatchLimit { limit: 1 })
    ));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = interpolate_patched_nd(
            |p, _| batch(p, |_| value),
            &[0.0],
            &[1.0],
            3,
            3,
            &Default::default(),
        );
        assert!(matches!(result, Err(InterpolativeQttError::NonFinite)));
    }
    let result = interpolate_patched_nd(
        |p, _| batch(p, |_| Complex64::new(0.0, f64::NAN)),
        &[0.0],
        &[1.0],
        3,
        3,
        &Default::default(),
    );
    assert!(matches!(result, Err(InterpolativeQttError::NonFinite)));
    let result = interpolate_patched_nd(
        |_, _| Ok::<Vec<f64>, Infallible>(vec![]),
        &[0.0],
        &[1.0],
        3,
        3,
        &Default::default(),
    );
    assert!(matches!(
        result,
        Err(InterpolativeQttError::BatchLength { actual: 0, .. })
    ));
}

#[derive(Debug, thiserror::Error)]
#[error("target callback failed")]
struct TargetError;

#[test]
fn evaluator_error_preserves_source() {
    let result = interpolate_patched_nd(
        |_, _| Err::<Vec<f64>, _>(TargetError),
        &[0.0],
        &[1.0],
        3,
        3,
        &Default::default(),
    )
    .unwrap_err();
    match result {
        InterpolativeQttError::Evaluation { source } => {
            assert!(source.downcast_ref::<TargetError>().is_some())
        }
        e => panic!("{e}"),
    }
}

#[test]
fn validate_before_calling_evaluator() {
    let invalid =
        |lower: &[f64], upper: &[f64], bits, degree, options: &PatchedInterpolativeOptions| {
            let result = interpolate_patched_nd(
                |_, _| -> std::result::Result<Vec<f64>, Infallible> {
                    panic!("invalid inputs reached evaluator")
                },
                lower,
                upper,
                bits,
                degree,
                options,
            );
            assert!(
                matches!(result, Err(InterpolativeQttError::InvalidArgument { .. })),
                "{result:?}"
            );
        };
    for (lower, upper) in [
        (vec![], vec![]),
        (vec![0.0], vec![]),
        (vec![1.0], vec![0.0]),
        (vec![0.0], vec![0.0]),
        (vec![f64::NAN], vec![1.0]),
        (vec![0.0], vec![f64::INFINITY]),
        (vec![-f64::MAX], vec![f64::MAX]),
        (vec![0.0; 64], vec![1.0; 64]),
    ] {
        invalid(&lower, &upper, 3, 3, &Default::default());
    }
    for bits in [0, 53, usize::MAX] {
        invalid(&[0.0], &[1.0], bits, 3, &Default::default());
    }
    for degree in [0, usize::MAX, usize::MAX / 2] {
        invalid(&[0.0], &[1.0], 3, degree, &Default::default());
    }
    let mut options = PatchedInterpolativeOptions::default();
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        options.tolerance = tolerance;
        invalid(&[0.0], &[1.0], 3, 3, &options);
    }
    options = Default::default();
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        options.compression_tolerance = tolerance;
        invalid(&[0.0], &[1.0], 3, 3, &options);
    }
    for options in [
        PatchedInterpolativeOptions {
            max_bond_dim: Some(0),
            ..Default::default()
        },
        PatchedInterpolativeOptions {
            max_depth: Some(3),
            ..Default::default()
        },
        PatchedInterpolativeOptions {
            max_patches: 0,
            ..Default::default()
        },
        PatchedInterpolativeOptions {
            max_stencil_elements: 0,
            ..Default::default()
        },
        PatchedInterpolativeOptions {
            max_stencil_elements: 1,
            ..Default::default()
        },
    ] {
        invalid(&[0.0], &[1.0], 3, 3, &options);
    }
}

#[test]
fn shared_multiscale_values_are_unchanged() {
    let options = PatchedInterpolativeOptions {
        tolerance: 1e-6,
        ..Default::default()
    };
    let f = |p: &[f64]| (p[0] + 2.0 * p[1]).exp();
    let patches =
        interpolate_patched_nd(|p, _| batch(p, f), &[0.0; 2], &[1.0; 2], 4, 7, &options).unwrap();
    assert_eq!(patches.len(), 1);
    let expected = crate::interpolate_multi_scale_nd(
        f,
        &[0.0; 2],
        &[1.0; 2],
        4,
        7,
        &[],
        &crate::InterpolativeQttOptions::default(),
    )
    .unwrap();
    assert_eq!(patches[0].tt.site_tensors(), expected.site_tensors());
}

#[test]
fn compression_bound_prevents_false_acceptance() {
    let f = |p: &[f64]| 1.0 + p[0];
    let result = interpolate_patched_nd(
        |p, _| batch(p, f),
        &[0.0],
        &[1.0],
        4,
        3,
        &PatchedInterpolativeOptions {
            tolerance: 1e-6,
            compression_tolerance: 0.9,
            max_depth: Some(0),
            ..Default::default()
        },
    );
    assert!(matches!(
        result,
        Err(InterpolativeQttError::NotConverged { .. })
    ));
}

#[cfg(feature = "parallel")]
mod parallel;

#[test]
fn nonrepresentable_grid_is_rejected_before_sampling() {
    let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);
    for bits in [1, 3] {
        let result = interpolate_patched_nd(
            |_, _| -> std::result::Result<Vec<f64>, Infallible> {
                panic!("invalid grid reached evaluator")
            },
            &[1.0],
            &[adjacent],
            bits,
            3,
            &Default::default(),
        );
        assert!(matches!(
            result,
            Err(InterpolativeQttError::InvalidArgument { .. })
        ));
    }
    let domain = NInterval::new(&[-0.1], &[0.2]).unwrap();
    let children = split_box(&domain).unwrap();
    assert_eq!(children[0].start(), domain.start());
    assert_eq!(children[0].stop(), children[1].start());
    assert_eq!(children[1].stop(), domain.stop());
}

#[test]
fn long_grid_stays_in_tensor_train_storage() {
    // Exercise the untruncated path: the extensive Frobenius bound is
    // intentionally conservative on very long grids. No dense tensor fits here.
    let patches = interpolate_patched_nd(
        |p, _| batch(p, |_| 3.0_f64),
        &[0.0],
        &[1.0],
        48,
        3,
        &PatchedInterpolativeOptions {
            compression_tolerance: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].tt.len(), 48);
    assert!(patches[0].tt.rank() <= 4);
    assert!((patches[0].tt.evaluate(&[1; 48]).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn physical_bounds_and_overflowing_residuals() {
    let lower = [-0.1, 1.2];
    let upper = [0.2, 2.3];
    let f = |p: &[f64]| Complex64::new(1.0 + p[0], 2.0 * p[1]);
    let patches = interpolate_patched_nd(
        |p, _| batch(p, f),
        &lower,
        &upper,
        4,
        3,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(patches.len(), 1);
    check_partition(
        &patches,
        2,
        4,
        |p| {
            f(&[
                lower[0] + (upper[0] - lower[0]) * p[0],
                lower[1] + (upper[1] - lower[1]) * p[1],
            ])
        },
        1e-8,
    );
    let result = interpolate_patched_nd(
        |p, _| batch(p, |_| f64::MAX),
        &[0.0],
        &[1.0],
        3,
        3,
        &Default::default(),
    );
    assert!(matches!(result, Err(InterpolativeQttError::NonFinite)));
    assert!(checked_product(usize::MAX, 2).is_err());
    assert!(checked_power(usize::MAX / 2, 3).is_err());
    assert!(
        split_box(&NInterval::new(&[1.0], &[f64::from_bits(1.0_f64.to_bits() + 1)]).unwrap())
            .is_err()
    );
}

#[test]
fn broken_engine_error_is_preserved() {
    let mut engine = CpuBackend::with_threads(1).unwrap().into_engine();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = engine.with_exec(|_| panic!("break test engine"));
    }));
    let result = sample(&|p, _| batch(p, |_| 1.0), &mut engine, &[vec![0.0]]);
    assert!(matches!(
        result,
        Err(InterpolativeQttError::Execution { .. })
    ));
}
