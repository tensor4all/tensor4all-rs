use super::*;

use hataori_tenferro::TenferroLanes;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn lanes(count: usize, threads: usize) -> TenferroLanes {
    TenferroLanes::new(
        (0..count)
            .map(|_| CpuBackend::with_threads(threads).unwrap().into_engine())
            .collect(),
    )
    .unwrap()
}

/// Bitwise equality, including a zero's sign and both parts of a complex.
trait SameBits {
    fn same_bits(&self, other: &Self) -> bool;
}

impl SameBits for f64 {
    fn same_bits(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl SameBits for Complex64 {
    fn same_bits(&self, other: &Self) -> bool {
        self.re.to_bits() == other.re.to_bits() && self.im.to_bits() == other.im.to_bits()
    }
}

fn compare_bits<T: InterpolativeScalar + SameBits + std::fmt::Debug>(
    serial: &ChebyshevPatches<T>,
    parallel: &ChebyshevPatches<T>,
) {
    assert_eq!(serial.patches().len(), parallel.patches().len());
    assert_eq!(serial.degree(), parallel.degree());
    assert_eq!(serial.num_evaluations(), parallel.num_evaluations());
    for (a, b) in serial.patches().iter().zip(parallel.patches()) {
        assert_eq!(a.prefix, b.prefix);
        assert_eq!(a.lower, b.lower);
        assert_eq!(a.upper, b.upper);
        assert_eq!(a.error_estimate.to_bits(), b.error_estimate.to_bits());
        assert_eq!(a.values.len(), b.values.len());
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!(x.same_bits(y), "{x:?} vs {y:?}");
        }
    }
}

fn compare<T: InterpolativeScalar + SameBits + Sync + std::fmt::Debug>(
    f: impl Fn(&[f64]) -> T + Sync,
    options: &ChebyshevPatchOptions,
) {
    for dims in [1, 2] {
        let lower = vec![0.0; dims];
        let upper = vec![1.0; dims];
        let degree = 4;
        let serial = build(&f, &lower, &upper, degree, options).unwrap();
        assert!(serial.patches().len() > 1, "the run must split");
        for (count, threads) in [(2, 1), (2, 2)] {
            let mut executor = lanes(count, threads);
            let parallel = interpolate_chebyshev_patches_on_lanes(
                &mut executor,
                |points, exec| {
                    assert_eq!(exec.budget().get(), threads);
                    values(points, &f)
                },
                &lower,
                &upper,
                degree,
                options,
            )
            .unwrap();
            compare_bits(&serial, &parallel);
        }
    }
}

#[test]
fn lanes_match_serial_bitwise() {
    let plain = |point: &[f64]| 1.0 / ((point[0] - 0.3).powi(2) + 1e-2);
    let complex = |point: &[f64]| {
        let radius = (point[0] - 0.3).powi(2) + (point[point.len() - 1] - 0.5).powi(2);
        Complex64::new(1.0 / (radius + 1e-2), 0.5 / (1.0 + radius))
    };
    for extra in [
        None,
        Some(ExtraValidation::Fixed { points_per_axis: 2 }),
        Some(ExtraValidation::Greedy {
            starts: 2,
            max_steps: 3,
        }),
    ] {
        let options = ChebyshevPatchOptions {
            tolerance: 1e-4,
            extra_validation: extra,
            ..Default::default()
        };
        compare(plain, &options);
        compare(complex, &options);
    }
}

#[test]
fn canonical_first_error_wins() {
    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    struct Failure(&'static str);

    // Degree 2, tolerance 0: the root's node grid holds a constant 1 that its
    // validation grid contradicts, so it splits into [0, 0.5] and [0.5, 1].
    // Both children fail, so the left one must be reported whichever lane
    // reaches a given child first.
    let evaluate = |points: ColMajorArrayRef<'_, f64>, _exec: Exec<'_>| {
        let count = points.shape()[1];
        let data = points.data();
        // A child's validation grid spans one half of the domain; the root's
        // batches span both halves.
        if count == 5 && data.iter().all(|&x| x <= 0.5) {
            return Err(Failure("left box"));
        }
        if count == 5 && data.iter().all(|&x| x >= 0.5) {
            return Err(Failure("right box"));
        }
        let value = if count == 3 { 1.0 } else { 0.0 };
        Ok(vec![value; count])
    };
    let options = ChebyshevPatchOptions {
        tolerance: 0.0,
        ..Default::default()
    };
    let serial = interpolate_chebyshev_patches(&context(), evaluate, &[0.0], &[1.0], 2, &options)
        .unwrap_err();
    match serial {
        InterpolativeQttError::Evaluation { source } => {
            assert_eq!(source.to_string(), "left box");
        }
        error => panic!("{error}"),
    }

    let mut executor = lanes(3, 1);
    let parallel = interpolate_chebyshev_patches_on_lanes(
        &mut executor,
        evaluate,
        &[0.0],
        &[1.0],
        2,
        &options,
    )
    .unwrap_err();
    match parallel {
        InterpolativeQttError::Evaluation { source } => {
            assert_eq!(source.to_string(), "left box");
        }
        error => panic!("{error}"),
    }
}

#[test]
fn ranges_and_external_evaluators_agree() {
    let lower = [0.0];
    let upper = [1.0];
    let options = ChebyshevPatchOptions {
        tolerance: 1e-4,
        ..Default::default()
    };
    let f = |point: &[f64]| 1.0 / ((point[0] - 0.3).powi(2) + 1e-2);
    let serial = build(f, &lower, &upper, 4, &options).unwrap();
    let mut executor = lanes(2, 2);
    assert!(serial.patches().len() > 1, "the run must split");

    let widths = AtomicUsize::new(0);
    let ranges = interpolate_chebyshev_patches_on_lanes(
        &mut executor,
        |points, exec| {
            let outputs: Vec<std::sync::OnceLock<f64>> = (0..points.shape()[1])
                .map(|_| std::sync::OnceLock::new())
                .collect();
            widths.fetch_max(exec.budget().get(), Ordering::Relaxed);
            exec.for_ranges(outputs.len(), 1.try_into().unwrap(), |range| {
                for index in range {
                    outputs[index]
                        .set(f(&points.data()[index..index + 1]))
                        .unwrap();
                }
            });
            Ok::<_, std::convert::Infallible>(
                outputs
                    .into_iter()
                    .map(|slot| slot.into_inner().unwrap())
                    .collect(),
            )
        },
        &lower,
        &upper,
        4,
        &options,
    )
    .unwrap();
    let external = interpolate_chebyshev_patches_on_lanes(
        &mut executor,
        |points, exec| {
            exec.external(|width| {
                widths.fetch_max(width.get(), Ordering::Relaxed);
                Ok::<_, std::convert::Infallible>(
                    points.data().par_iter().map(|&x| f(&[x])).collect(),
                )
            })
        },
        &lower,
        &upper,
        4,
        &options,
    )
    .unwrap();
    assert_eq!(widths.load(Ordering::Relaxed), 2);
    compare_bits(&serial, &ranges);
    compare_bits(&serial, &external);
}
