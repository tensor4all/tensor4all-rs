use super::*;
use num_complex::Complex64;
use std::convert::Infallible;
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};

// Run the serial entry on a freshly configured one-thread context.
fn serial<T, F, E>(
    evaluate: F,
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    degree: usize,
    options: &PatchedInterpolativeOptions,
) -> Result<Vec<InterpolativePatch<T>>>
where
    T: InterpolativeScalar,
    F: Fn(ColMajorArrayRef<'_, f64>, Exec<'_>) -> std::result::Result<Vec<T>, E>,
    E: Into<Box<dyn Error + Send + Sync>>,
{
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
    interpolate_patched_nd(&context, evaluate, lower, upper, num_bits, degree, options)
}

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
        let patches = serial(
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
        let patches = serial(
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
        let p = serial(
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
    let p = serial(|p, _| batch(p, lorentzian), &[0.0], &[1.0], 1, 2, &options).unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].error_estimate, 0.0);
    check_partition(&p, 1, 1, lorentzian, options.tolerance);
}

#[test]
fn limits_and_nonfinite_values_are_errors() {
    let run = |options: &PatchedInterpolativeOptions| {
        serial(|p, _| batch(p, lorentzian), &[0.0], &[1.0], 4, 3, options)
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
        let result = serial(
            |p, _| batch(p, |_| value),
            &[0.0],
            &[1.0],
            3,
            3,
            &Default::default(),
        );
        assert!(matches!(result, Err(InterpolativeQttError::NonFinite)));
    }
    let result = serial(
        |p, _| batch(p, |_| Complex64::new(0.0, f64::NAN)),
        &[0.0],
        &[1.0],
        3,
        3,
        &Default::default(),
    );
    assert!(matches!(result, Err(InterpolativeQttError::NonFinite)));
    let result = serial(
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
    let result = serial(
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
            let result = serial(
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
    let patches = serial(|p, _| batch(p, f), &[0.0; 2], &[1.0; 2], 4, 7, &options).unwrap();
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
fn compression_change_counts_against_the_tolerance() {
    // A relative cutoff of 0.9 truncates the rank-2 train of 1 + x on [0, 1]
    // to rank one, which changes values by far more than the tolerance. The
    // change is measured on the returned train, so the root box is rejected.
    let f = |p: &[f64]| 1.0 + p[0];
    let bits = 4;
    let lossy = PatchedInterpolativeOptions {
        tolerance: 1e-6,
        compression_tolerance: 0.9,
        ..Default::default()
    };
    let rejected = serial(
        |p, _| batch(p, f),
        &[0.0],
        &[1.0],
        bits,
        3,
        &PatchedInterpolativeOptions {
            max_depth: Some(0),
            ..lossy
        },
    );
    assert!(matches!(
        rejected,
        Err(InterpolativeQttError::NotConverged { error_estimate, rank: 1, .. })
            if error_estimate > 1e-6
    ));
    // With splitting allowed, and with a rank cap that the truncated rank-one
    // train would satisfy, every accepted patch still meets the tolerance on
    // the whole grid.
    for max_bond_dim in [None, Some(1)] {
        let options = PatchedInterpolativeOptions {
            max_bond_dim,
            ..lossy
        };
        let patches = serial(|p, _| batch(p, f), &[0.0], &[1.0], bits, 3, &options).unwrap();
        assert!(patches.len() > 1);
        check_partition(&patches, 1, bits, f, options.tolerance);
    }
}

#[test]
fn uncompressed_train_matches_the_child_interpolants_it_is_checked_against() {
    // compression_change compares the compressed train with the immediate
    // children's Chebyshev interpolants. Against the uncompressed train the
    // difference must be roundoff, for both value types, 1D and 2D, and for
    // trains short enough that every grid point is checked.
    fn check<T: InterpolativeScalar>(f: impl Fn(&[f64]) -> T) {
        for (dims, sites, degree) in [(1, 2, 3), (1, 6, 4), (2, 5, 3), (2, 3, 2)] {
            let options = PatchedInterpolativeOptions::default();
            let lower = vec![-0.3; dims];
            let upper = vec![0.9; dims];
            let plan = Plan::<T>::new(&lower, &upper, sites, degree, &options).unwrap();
            let mut children = Vec::new();
            let cores = build_refined_cores(
                &mut |batch: SampleBatch<'_>| {
                    let values: Vec<T> = batch.points().iter().map(|p| f(p)).collect();
                    if let SampleBatch::Nodes { interval, .. } = batch {
                        children.push((interval.clone(), values.clone()));
                    }
                    Ok(values)
                },
                &plan.domain,
                sites,
                &plan.basis,
                |_, _| true,
            )
            .unwrap();
            let train = SimpleTensorTrain::new(cores).unwrap();
            let change = compression_change(&plan, &plan.domain, &children, &train).unwrap();
            assert!(change < 1e-13, "D={dims}, L={sites}, p={degree}: {change}");
        }
    }
    check(|p: &[f64]| (p[0] - 0.5 * p.get(1).copied().unwrap_or(0.0)).exp());
    check(|p: &[f64]| Complex64::new(p[0].sin(), p.iter().sum::<f64>().cos()));
}

#[test]
fn compression_check_indices_cover_short_axes_and_both_halves_of_long_ones() {
    for degree in 1..=8 {
        let tests = dense_test_points(degree + 1).unwrap();
        for sites in 1..=12 {
            let size = 1u64 << sites;
            let half = size / 2;
            let indices = compression_check_indices(&tests, sites);
            assert!(
                indices.windows(2).all(|w| w[0] < w[1]),
                "p={degree}, L={sites}"
            );
            assert!(indices.iter().all(|&k| k < size));
            if size <= 2 * tests.len() as u64 {
                // Short axes: every grid point, including those the rounded
                // Chebyshev coordinates would skip (p=3, L=3 misses 2 and 5).
                assert_eq!(
                    indices,
                    (0..size).collect::<Vec<_>>(),
                    "p={degree}, L={sites}"
                );
            } else {
                // Long axes: both ends of both halves, so the patch midpoint
                // where the child interpolants meet is checked, plus interior
                // points in each half.
                for k in [0, half - 1, half, size - 1] {
                    assert!(indices.contains(&k), "p={degree}, L={sites}: {k}");
                }
                for range in [1..half - 1, half + 1..size - 1] {
                    assert!(indices.iter().any(|k| range.contains(k)));
                }
                assert!(indices.len() <= 2 * tests.len());
            }
        }
    }
}

#[test]
fn reentering_the_context_is_an_execution_error() {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
    let result = context
        .with_session(|_| {
            interpolate_patched_nd(
                &context,
                |p, _| batch(p, |_| 1.0),
                &[0.0],
                &[1.0],
                3,
                3,
                &Default::default(),
            )
        })
        .unwrap();
    assert!(matches!(
        result,
        Err(InterpolativeQttError::Execution { .. })
    ));
}

/// Evaluate the patch that owns grid point `indices` (one integer per axis).
fn evaluate_on_grid<T: InterpolativeScalar>(
    patches: &[InterpolativePatch<T>],
    bits: usize,
    indices: &[u64],
) -> T {
    let bit = |axis: usize, level: usize| ((indices[axis] >> (bits - 1 - level)) & 1) as usize;
    let patch = patches
        .iter()
        .find(|p| {
            p.prefix
                .iter()
                .enumerate()
                .all(|(axis, prefix)| prefix.iter().enumerate().all(|(l, &b)| b == bit(axis, l)))
        })
        .unwrap();
    let depth = patch.prefix[0].len();
    let sites: Vec<usize> = (depth..bits)
        .map(|level| {
            (0..indices.len())
                .map(|axis| bit(axis, level) << axis)
                .sum()
        })
        .collect();
    patch.tt.evaluate(&sites).unwrap()
}

#[test]
fn default_options_do_not_over_refine_deep_2d_grids() {
    // Acceptance compares only per-value quantities with the per-value
    // tolerance, so default options keep a deep smooth 2D grid in one patch and
    // its estimate below the tolerance.
    let smooth = |p: &[f64]| (p[0] + 0.5 * p[1]).exp();
    let constant = |_: &[f64]| 1.0;
    let options = PatchedInterpolativeOptions::default();
    for (f, bits) in [
        (&constant as &dyn Fn(&[f64]) -> f64, 30),
        (&smooth, 30),
        (&smooth, 40),
    ] {
        let patches = serial(|p, _| batch(p, f), &[0.0; 2], &[1.0; 2], bits, 8, &options).unwrap();
        assert!(patches.len() <= 4, "{bits} bits: {} patches", patches.len());
        assert!(patches
            .iter()
            .all(|p| p.error_estimate <= options.tolerance));
        let size = 1u64 << bits;
        let mut max_error = 0.0_f64;
        for (i, j) in [
            (0, 0),
            (1, size - 1),
            (size / 3, 2 * size / 3),
            (size - 1, size - 1),
        ] {
            let point = [i as f64 / size as f64, j as f64 / size as f64];
            max_error =
                max_error.max((evaluate_on_grid(&patches, bits, &[i, j]) - f(&point)).abs());
        }
        assert!(
            max_error <= options.tolerance,
            "{bits} bits: error {max_error}"
        );
    }
}

#[cfg(feature = "parallel")]
mod parallel;

#[test]
fn nonrepresentable_grid_is_rejected_before_sampling() {
    let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);
    for bits in [1, 3] {
        let result = serial(
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
    let patches = serial(
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
    let patches = serial(
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
    // Keep the Chebyshev node values small enough that the stencil GEMM cannot
    // overflow, and put -MAX only at the residual test points: the residual is
    // then ~(1 + 2^-10) * MAX in every summation order, so the overflow does not
    // depend on how the BLAS provider orders or fuses the products.
    let degree = 3;
    let test_batch = 2 * (degree + 1);
    let result = serial(
        |p, _| {
            let target = if p.shape()[1] == test_batch {
                -f64::MAX
            } else {
                f64::MAX / 1024.0
            };
            batch(p, |_| target)
        },
        &[0.0],
        &[1.0],
        3,
        degree,
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
