#[cfg(feature = "parallel")]
mod parallel;

use super::*;

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::sync::Mutex;

use num_complex::Complex64;
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};

fn context() -> CpuExecutionContext {
    CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap())
}

fn values<T: InterpolativeScalar>(
    points: ColMajorArrayRef<'_, f64>,
    f: &impl Fn(&[f64]) -> T,
) -> std::result::Result<Vec<T>, Infallible> {
    Ok(points
        .data()
        .chunks_exact(points.shape()[0])
        .map(f)
        .collect())
}

fn build<T: InterpolativeScalar>(
    f: impl Fn(&[f64]) -> T,
    lower: &[f64],
    upper: &[f64],
    degree: usize,
    options: &ChebyshevPatchOptions,
) -> Result<ChebyshevPatches<T>> {
    interpolate_chebyshev_patches(
        &context(),
        |points, _exec| values(points, &f),
        lower,
        upper,
        degree,
        options,
    )
}

fn options(tolerance: f64) -> ChebyshevPatchOptions {
    ChebyshevPatchOptions {
        tolerance,
        ..Default::default()
    }
}

fn lorentzian(center: f64, width: f64) -> impl Fn(&[f64]) -> f64 {
    move |point: &[f64]| 1.0 / ((point[0] - center).powi(2) + width * width)
}

fn feature(point: &[f64]) -> f64 {
    (-((point[0] - 0.34) / 0.02).powi(2)).exp()
}

/// Number of boxes the driver processed: every leaf plus all its ancestors.
fn boxes_processed<T>(patches: &ChebyshevPatches<T>) -> usize {
    let mut boxes: BTreeSet<Vec<Vec<usize>>> = BTreeSet::new();
    for patch in patches.patches() {
        let depth = patch.prefix[0].len();
        for level in 0..=depth {
            boxes.insert(
                patch
                    .prefix
                    .iter()
                    .map(|axis| axis[..level].to_vec())
                    .collect(),
            );
        }
    }
    boxes.len()
}

fn depth(patch: &ChebyshevPatch<impl Sized>) -> usize {
    patch.prefix[0].len()
}

fn contains<T>(patch: &ChebyshevPatch<T>, point: &[f64]) -> bool {
    point
        .iter()
        .zip(patch.lower.iter().zip(&patch.upper))
        .all(|(&x, (&lower, &upper))| x >= lower && x <= upper)
}

fn patch_containing<'a, T>(
    patches: &'a ChebyshevPatches<T>,
    point: &[f64],
) -> &'a ChebyshevPatch<T> {
    patches
        .patches()
        .iter()
        .filter(|patch| contains(patch, point))
        .max_by_key(|patch| depth(patch))
        .expect("the domain is covered")
}

fn node_axis(a: f64, b: f64, degree: usize) -> Vec<f64> {
    axis_nodes(a, b, get_chebyshev_grid(degree).unwrap().grid())
}

fn sampled_nodes<T: InterpolativeScalar>(
    f: &impl Fn(&[f64]) -> T,
    patch: &ChebyshevPatch<T>,
    degree: usize,
) -> Vec<T> {
    let axes: Vec<Vec<f64>> = (0..patch.lower.len())
        .map(|axis| node_axis(patch.lower[axis], patch.upper[axis], degree))
        .collect();
    let points = tensor_points(&axes);
    points.chunks_exact(patch.lower.len()).map(f).collect()
}

fn assert_same_bits(actual: &[f64], expected: &[f64]) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "entry {index}: {actual} vs {expected}"
        );
    }
}

// Deterministic interior points, so the sampled-error test does not depend on
// the test harness's randomness.
fn random_points(lower: &[f64], upper: &[f64], count: usize) -> Vec<f64> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mut points = Vec::with_capacity(count * lower.len());
    for _ in 0..count {
        for axis in 0..lower.len() {
            points.push(lower[axis] + (upper[axis] - lower[axis]) * next());
        }
    }
    points
}

#[test]
fn polynomial_is_one_patch() {
    // A polynomial below the local order is reproduced exactly.
    let one = build(
        |point| 1.0 - 2.0 * point[0] + 3.0 * point[0].powi(2) - point[0].powi(3),
        &[-1.0],
        &[2.0],
        4,
        &options(1e-12),
    )
    .unwrap();
    assert_eq!(one.patches().len(), 1);
    assert_eq!(one.patches()[0].prefix, vec![Vec::<usize>::new()]);
    assert!(one.patches()[0].error_estimate < 1e-12);

    let two = build(
        |point| 1.0 + 2.0 * point[0] - 3.0 * point[1] + point[0] * point[1],
        &[-1.0, 0.0],
        &[3.0, 2.0],
        3,
        &options(1e-12),
    )
    .unwrap();
    assert_eq!(two.patches().len(), 1);
    assert!(two.patches()[0].error_estimate < 1e-12);
    assert_eq!(two.num_evaluations(), 16 + 49);

    let complex = build(
        |point| Complex64::new(1.0 + point[0], -1.0 - point[1] * point[1]),
        &[0.0, 0.0],
        &[1.0, 1.0],
        4,
        &options(1e-12),
    )
    .unwrap();
    assert_eq!(complex.patches().len(), 1);
    assert_eq!(complex.patches()[0].values.len(), 25);
    assert_eq!(complex.patches()[0].values[0], Complex64::new(1.0, -1.0));
}

#[test]
fn smooth_function_is_few_patches() {
    let one: Result<ChebyshevPatches<f64>> = build(
        |point| (-point[0] * point[0]).exp(),
        &[-1.0],
        &[1.0],
        8,
        &options(1e-10),
    );
    let one = one.unwrap();
    assert!(one.patches().len() <= 8, "{} patches", one.patches().len());

    let two = build(
        |point| (-point[0] * point[0] - point[1] * point[1]).exp(),
        &[-1.0, -1.0],
        &[1.0, 1.0],
        12,
        &options(1e-10),
    )
    .unwrap();
    assert!(two.patches().len() <= 8, "{} patches", two.patches().len());
}

#[test]
fn localized_feature_refines_locally() {
    let one = build(lorentzian(0.3, 1e-3), &[0.0], &[1.0], 8, &options(1e-6)).unwrap();
    assert!(one.patches().len() > 2);
    let feature = patch_containing(&one, &[0.3]);
    let far = patch_containing(&one, &[0.95]);
    assert!(
        depth(feature) > depth(far),
        "feature depth {} vs far depth {}",
        depth(feature),
        depth(far)
    );

    // A complex ridge: the magnitude peaks at the upper corner.
    let complex = build(
        |point| {
            let radius = (point[0] - 0.3).powi(2) + (point[1] - 0.6).powi(2);
            Complex64::new(1.0 / (radius + 1e-2), 0.5 / (1.0 + radius))
        },
        &[0.0, 0.0],
        &[1.0, 1.0],
        6,
        &options(1e-3),
    )
    .unwrap();
    let peak = patch_containing(&complex, &[0.3, 0.6]);
    let quiet = patch_containing(&complex, &[0.95, 0.05]);
    assert!(
        depth(peak) > depth(quiet),
        "peak depth {} vs quiet depth {}",
        depth(peak),
        depth(quiet)
    );
}

#[test]
fn pointwise_error_within_tolerance() {
    let tolerance = 1e-6;
    let one = build(
        lorentzian(0.3, 1e-3),
        &[0.0],
        &[1.0],
        8,
        &options(tolerance),
    )
    .unwrap();
    let two = build(
        |point| (-point[0] * point[0] - point[1] * point[1]).exp(),
        &[-1.0, -1.0],
        &[1.0, 1.0],
        6,
        &options(tolerance),
    )
    .unwrap();
    for (patches, lower, upper, f) in [
        (&one, vec![0.0], vec![1.0], 0),
        (&two, vec![-1.0, -1.0], vec![1.0, 1.0], 1),
    ] {
        let ndims = lower.len();
        let points = random_points(&lower, &upper, 200);
        let evaluated = patches
            .evaluate(ColMajorArrayRef::new(&points, &[ndims, 200]).unwrap())
            .unwrap();
        // The estimate is sampled, not a certificate, so allow a safety factor.
        let safety = 10.0 * tolerance;
        for (index, value) in evaluated.iter().enumerate() {
            let point = &points[index * ndims..(index + 1) * ndims];
            let expected = if f == 0 {
                lorentzian(0.3, 1e-3)(point)
            } else {
                (-point[0] * point[0] - point[1] * point[1]).exp()
            };
            assert!(
                (value - expected).abs() <= safety,
                "point {point:?}: {value} vs {expected}"
            );
        }
    }
}

#[test]
fn patches_partition_the_domain() {
    let lower = [-1.0, -1.0];
    let upper = [3.0, 2.0];
    let patches = build(
        |point| 1.0 / ((point[0] - 0.3).powi(2) + (point[1] - 0.6).powi(2) + 1e-2),
        &lower,
        &upper,
        6,
        &options(1e-3),
    )
    .unwrap();
    let mut prefixes: Vec<&Vec<Vec<usize>>> = patches
        .patches()
        .iter()
        .map(|patch| &patch.prefix)
        .collect();
    prefixes.sort();
    for (index, prefix) in prefixes.iter().enumerate() {
        for other in &prefixes[index + 1..] {
            let shared = prefix[0].len().min(other[0].len());
            let nested = prefix
                .iter()
                .zip(other.iter())
                .all(|(axis, other)| axis[..shared] == other[..shared]);
            assert!(!nested, "one accepted prefix is an ancestor of another");
        }
    }
    for patch in patches.patches() {
        for axis in 0..lower.len() {
            assert!(patch.lower[axis] >= lower[axis] - 1e-12);
            assert!(patch.upper[axis] <= upper[axis] + 1e-12);
        }
    }
    let volume = |lower: &[f64], upper: &[f64]| -> f64 {
        (0..lower.len())
            .map(|axis| upper[axis] - lower[axis])
            .product()
    };
    let total: f64 = patches
        .patches()
        .iter()
        .map(|patch| volume(&patch.lower, &patch.upper))
        .sum();
    assert!((total - volume(&lower, &upper)).abs() < 1e-12);
}

#[derive(Default)]
struct Recorded {
    batches: Vec<usize>,
    coordinates: Vec<Vec<f64>>,
}

#[test]
/// The boxes are disjoint by construction, so counting the sampled points is
/// equivalent to checking that no grid point is sampled twice; the exact-count
/// identity below is that check.
fn no_value_is_evaluated_twice() {
    let degree = 4;
    let options = options(1e-6);
    let record = Mutex::new(Recorded::default());
    let f = lorentzian(0.3, 3e-3);
    let patches = interpolate_chebyshev_patches(
        &context(),
        |points, _exec| {
            let ndims = points.shape()[0];
            let mut recorder = record.lock().unwrap();
            recorder.batches.push(points.shape()[1]);
            for point in points.data().chunks_exact(ndims) {
                recorder.coordinates.push(point.to_vec());
            }
            values(points, &f)
        },
        &[0.0],
        &[1.0],
        degree,
        &options,
    )
    .unwrap();
    let recorded = record.into_inner().unwrap();
    let node_dim = degree + 1;
    let test_dim = 2 * degree + 1;
    // Only the root samples its node grid; every other batch is a validation grid.
    assert_eq!(
        recorded
            .batches
            .iter()
            .filter(|&&length| length == node_dim)
            .count(),
        1
    );
    assert!(recorded
        .batches
        .iter()
        .all(|&length| length == node_dim || length == test_dim));
    let boxes = boxes_processed(&patches);
    assert_eq!(recorded.batches.len(), boxes + 1);
    assert_eq!(
        recorded.coordinates.len(),
        node_dim + boxes * test_dim,
        "node samples are inherited, not repeated"
    );
    assert_eq!(patches.num_evaluations(), recorded.coordinates.len());
}

#[test]
fn inherited_node_values_are_bitwise_correct() {
    let degree = 6;
    let f = |point: &[f64]| 1.0 / ((point[0] - 0.3).powi(2) + 1e-2);
    let patches = build(f, &[0.0], &[1.0], degree, &options(1e-4)).unwrap();
    assert!(patches.patches().iter().any(|patch| depth(patch) > 0));
    for patch in patches.patches() {
        let fresh = sampled_nodes(&f, patch, degree);
        assert_same_bits(&patch.values, &fresh);
    }

    let g = |point: &[f64]| {
        let radius = (point[0] - 0.3).powi(2) + (point[1] - 0.6).powi(2);
        Complex64::new(1.0 / (radius + 1e-2), 0.5 / (1.0 + radius))
    };
    let complex = build(g, &[0.0, 0.0], &[1.0, 1.0], degree, &options(1e-4)).unwrap();
    assert!(complex.patches().iter().any(|patch| depth(patch) > 0));
    for patch in complex.patches() {
        let fresh = sampled_nodes(&g, patch, degree);
        for (&actual, &expected) in patch.values.iter().zip(&fresh) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }
}

#[test]
fn evaluate_reproduces_node_values() {
    let degree = 3;
    let f = |point: &[f64]| 1.0 / ((point[0] - 0.3).powi(2) + 3e-3);
    let patches = build(f, &[0.0], &[1.0], degree, &options(1e-5)).unwrap();
    assert!(patches.patches().len() > 1);

    for patch in patches.patches() {
        let coordinates = node_axis(patch.lower[0], patch.upper[0], degree);
        for (index, &coordinate) in coordinates.iter().enumerate() {
            let evaluated = patches
                .evaluate(ColMajorArrayRef::new(&[coordinate], &[1, 1]).unwrap())
                .unwrap();
            assert_same_bits(&evaluated, &[patch.values[index]]);
        }
    }

    // A shared face: the upper patch owns the point, and both patches store the
    // same inherited value there.
    let by_prefix: BTreeMap<&Vec<Vec<usize>>, &ChebyshevPatch<f64>> = patches
        .patches()
        .iter()
        .map(|patch| (&patch.prefix, patch))
        .collect();
    let mut shared = 0;
    for patch in patches.patches() {
        for axis in 0..patch.prefix.len() {
            let mut flipped = patch.prefix.clone();
            if flipped[axis].last() != Some(&1) {
                continue;
            }
            *flipped[axis].last_mut().unwrap() = 0;
            let Some(left) = by_prefix.get(&flipped) else {
                continue;
            };
            shared += 1;
            // Both inherit the same coordinate from their common parent.
            assert_eq!(
                left.values[left.values.len() - 1].to_bits(),
                patch.values[0].to_bits()
            );
            let evaluated = patches
                .evaluate(ColMajorArrayRef::new(&[patch.lower[0]], &[1, 1]).unwrap())
                .unwrap();
            assert_same_bits(&evaluated, &[patch.values[0]]);
        }
    }
    assert!(shared > 0, "the run has no shared face");
}

#[test]
fn evaluate_at_a_coarse_fine_interface() {
    // A localized feature refines one region only, so faces exist where the two
    // neighbouring patches have different depths. The upper (higher-prefix)
    // patch owns such a point, and it reproduces the values it stores there.
    let degree = 3;
    let f = |point: &[f64]| {
        let radius = (point[0] - 0.3).powi(2) + (point[1] - 0.5).powi(2);
        1.0 / (radius + 1e-2)
    };
    let patches = build(f, &[0.0, 0.0], &[1.0, 1.0], degree, &options(1e-3)).unwrap();
    let depths: BTreeSet<usize> = patches.patches().iter().map(depth).collect();
    assert!(
        depths.len() > 1,
        "the run has no coarse/fine face: {depths:?}"
    );

    let mut checked = 0;
    for patch in patches.patches() {
        if patch.prefix[0].is_empty() {
            continue;
        }
        let face = patch.lower[0];
        // The patch's own upper y face belongs to the next patch up, so only its
        // lower and interior y nodes are points it owns.
        let coordinates: Vec<f64> = node_axis(patch.lower[1], patch.upper[1], degree)
            .into_iter()
            .take(degree)
            .collect();
        let points: Vec<f64> = coordinates.iter().flat_map(|&y| [face, y]).collect();
        let evaluated = patches
            .evaluate(ColMajorArrayRef::new(&points, &[2, coordinates.len()]).unwrap())
            .unwrap();
        // Axis zero is fastest, so the node at (x index 0, y index `index`) is
        // the `(degree + 1) * index`-th stored value.
        for (index, value) in evaluated.into_iter().enumerate() {
            assert_same_bits(&[value], &[patch.values[(degree + 1) * index]]);
            checked += 1;
        }
    }
    assert!(checked > 0, "no patch has an internal lower face");
}

#[test]
fn evaluate_rejects_invalid_points() {
    let patches = build(|point| 1.0 + point[0], &[0.0], &[1.0], 3, &options(1e-12)).unwrap();
    let outside = patches
        .evaluate(ColMajorArrayRef::new(&[1.5], &[1, 1]).unwrap())
        .unwrap_err();
    assert!(matches!(
        outside,
        InterpolativeQttError::InvalidArgument { .. }
    ));
    let non_finite = patches
        .evaluate(ColMajorArrayRef::new(&[f64::NAN], &[1, 1]).unwrap())
        .unwrap_err();
    assert!(matches!(
        non_finite,
        InterpolativeQttError::InvalidArgument { .. }
    ));
    // A rank-one or rank-three array with a consistent element count must be
    // rejected as a shape, not read as if it were `[D, n]`.
    for shape in [&[2, 1][..], &[2][..], &[1, 1, 2][..]] {
        let wrong_shape = patches
            .evaluate(ColMajorArrayRef::new(&[0.5, 0.5], shape).unwrap())
            .unwrap_err();
        assert!(
            matches!(wrong_shape, InterpolativeQttError::InvalidArgument { .. }),
            "{shape:?}: {wrong_shape}"
        );
    }
    // The domain's upper corner is inside the closed domain.
    let upper_corner = patches
        .evaluate(ColMajorArrayRef::new(&[1.0], &[1, 1]).unwrap())
        .unwrap();
    assert!((upper_corner[0] - 2.0).abs() < 1e-12);
}

#[test]
fn rounded_midpoint_reports_the_interpolation_error() {
    // `[1, 1 + 3u]` has no representable midpoint: it rounds to `1 + 2u`, which
    // is at local coordinate 2/3, not 1/2. The stencil must use the coordinate
    // that was actually sampled, otherwise it reports its own rounding as
    // interpolation error and rejects an affine function.
    let u = f64::EPSILON;
    let (lower, upper) = (1.0, 1.0 + 3.0 * u);
    let affine = |point: &[f64]| (point[0] - 1.0) / u;
    let patches = build(
        affine,
        &[lower],
        &[upper],
        1,
        &ChebyshevPatchOptions {
            tolerance: 1e-12,
            max_depth: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(patches.patches().len(), 1);
    assert!(patches.patches()[0].error_estimate < 1e-12);

    let midpoint = lower + (upper - lower) / 2.0;
    assert_eq!(midpoint, 1.0 + 2.0 * u, "the midpoint really is rounded");
    let evaluated = patches
        .evaluate(ColMajorArrayRef::new(&[midpoint], &[1, 1]).unwrap())
        .unwrap();
    assert!((evaluated[0] - 2.0).abs() < 1e-12, "{evaluated:?}");
}

#[test]
fn evaluator_error_is_preserved() {
    #[derive(Debug, thiserror::Error)]
    #[error("refused at {0}")]
    struct Refused(usize);

    let error = interpolate_chebyshev_patches(
        &context(),
        |points: ColMajorArrayRef<'_, f64>, _exec| {
            if points.shape()[1] > 5 {
                return Err(Refused(points.shape()[1]));
            }
            Ok::<_, Refused>(points.data().iter().map(|&x| 1.0 + x).collect())
        },
        &[0.0],
        &[1.0],
        5,
        &options(1e-8),
    )
    .unwrap_err();
    match error {
        InterpolativeQttError::Evaluation { source } => {
            assert_eq!(source.downcast_ref::<Refused>().unwrap().0, 6)
        }
        other => panic!("{other}"),
    }
}

#[test]
fn wrong_batch_length_is_reported() {
    let error = interpolate_chebyshev_patches(
        &context(),
        |points: ColMajorArrayRef<'_, f64>, _exec| {
            Ok::<_, Infallible>(points.data()[..points.shape()[1] - 1].to_vec())
        },
        &[0.0],
        &[1.0],
        3,
        &options(1e-8),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        InterpolativeQttError::BatchLength {
            expected: 4,
            actual: 3
        }
    ));
}

#[test]
fn non_finite_value_is_reported() {
    let error = interpolate_chebyshev_patches(
        &context(),
        |_points: ColMajorArrayRef<'_, f64>, _exec| Ok::<_, Infallible>(vec![f64::NAN; 4]),
        &[0.0],
        &[1.0],
        3,
        &options(1e-8),
    )
    .unwrap_err();
    assert!(matches!(error, InterpolativeQttError::NonFinite));
}

#[test]
fn broken_engine_is_reported() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = backend
            .engine_mut()
            .with_exec(|_exec| panic!("break the engine"));
    }));
    let context = CpuExecutionContext::from_backend(backend);
    let error = interpolate_chebyshev_patches(
        &context,
        |points: ColMajorArrayRef<'_, f64>, _exec| {
            Ok::<_, Infallible>(points.data().iter().map(|&x| 1.0 + x).collect())
        },
        &[0.0],
        &[1.0],
        3,
        &options(1e-8),
    )
    .unwrap_err();
    assert!(matches!(error, InterpolativeQttError::Execution { .. }));
}

#[test]
fn non_convergence_reports_the_prefix() {
    let error = build(
        lorentzian(0.3, 1e-3),
        &[0.0],
        &[1.0],
        4,
        &ChebyshevPatchOptions {
            tolerance: 1e-12,
            max_depth: 0,
            ..Default::default()
        },
    )
    .unwrap_err();
    match error {
        InterpolativeQttError::ChebyshevNotConverged {
            prefix,
            error_estimate,
        } => {
            assert_eq!(prefix, vec![Vec::<usize>::new()]);
            assert!(error_estimate > 1e-12);
        }
        other => panic!("{other}"),
    }
}

#[test]
fn patch_limit_is_reported() {
    let error = build(
        lorentzian(0.3, 1e-3),
        &[0.0],
        &[1.0],
        4,
        &ChebyshevPatchOptions {
            tolerance: 1e-12,
            max_patches: 1,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        InterpolativeQttError::PatchLimit { limit: 1 }
    ));
}

#[test]
fn invalid_arguments_are_rejected() {
    let cases: Vec<(&[f64], &[f64], usize, ChebyshevPatchOptions)> = vec![
        (&[], &[], 3, options(1e-8)),
        (&[0.0], &[1.0, 2.0], 3, options(1e-8)),
        (&[1.0], &[0.0], 3, options(1e-8)),
        (&[0.0], &[f64::INFINITY], 3, options(1e-8)),
        (&[0.0], &[1.0], 0, options(1e-8)),
        (&[0.0], &[1.0], 3, options(-1.0)),
        (&[0.0], &[1.0], 3, options(f64::NAN)),
        (
            &[0.0],
            &[1.0],
            3,
            ChebyshevPatchOptions {
                max_patches: 0,
                ..Default::default()
            },
        ),
        (&[0.0], &[1.0], usize::MAX, options(1e-8)),
        (
            &[0.0],
            &[1.0],
            3,
            ChebyshevPatchOptions {
                extra_validation: Some(ExtraValidation::Fixed { points_per_axis: 0 }),
                ..Default::default()
            },
        ),
        (
            &[0.0],
            &[1.0],
            3,
            ChebyshevPatchOptions {
                extra_validation: Some(ExtraValidation::Greedy {
                    starts: 0,
                    max_steps: 1,
                }),
                ..Default::default()
            },
        ),
        (
            &[0.0],
            &[1.0],
            3,
            ChebyshevPatchOptions {
                extra_validation: Some(ExtraValidation::Greedy {
                    starts: 1,
                    max_steps: 0,
                }),
                ..Default::default()
            },
        ),
    ];
    for (lower, upper, degree, settings) in cases {
        let error = build(|point| 1.0 + point[0], lower, upper, degree, &settings).unwrap_err();
        assert!(
            matches!(error, InterpolativeQttError::InvalidArgument { .. }),
            "{error}"
        );
    }
}

#[test]
fn extra_validation_is_off_by_default() {
    // `feature` peaks between the inherited grid points, so the grid check
    // accepts the root box.
    let patches = build(feature, &[0.0], &[1.0], 4, &options(1e-6)).unwrap();
    assert_eq!(patches.patches().len(), 1);
    assert!(patches.num_evaluations() < 5 + 9 + 1);
}

#[test]
fn extra_validation_rejects_a_box_the_grid_accepts() {
    // A spike at 0.3 with width 0.008 is invisible to the degree-4 node and
    // validation grids, and 0.3 is one of the five `Fixed` points. With
    // `max_depth = 0` the run can only fail if the extra check, not the grid,
    // rejected the box.
    let spike = |point: &[f64]| (-((point[0] - 0.3) / 0.008).powi(2)).exp();
    let grid_only = ChebyshevPatchOptions {
        tolerance: 1e-6,
        max_depth: 0,
        ..Default::default()
    };
    let accepted = build(spike, &[0.0], &[1.0], 4, &grid_only).unwrap();
    assert_eq!(
        accepted.patches().len(),
        1,
        "the node grid misses the spike"
    );

    let rejected = build(
        spike,
        &[0.0],
        &[1.0],
        4,
        &ChebyshevPatchOptions {
            extra_validation: Some(ExtraValidation::Fixed { points_per_axis: 5 }),
            ..grid_only
        },
    );
    assert!(matches!(
        rejected.unwrap_err(),
        InterpolativeQttError::ChebyshevNotConverged { .. }
    ));
}

#[test]
fn extra_validation_splits_a_missed_peak() {
    for extra in [
        ExtraValidation::Fixed { points_per_axis: 4 },
        ExtraValidation::Greedy {
            starts: 1,
            max_steps: 4,
        },
    ] {
        let patches = build(
            feature,
            &[0.0],
            &[1.0],
            4,
            &ChebyshevPatchOptions {
                tolerance: 1e-6,
                extra_validation: Some(extra),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            patches.patches().len() > 1,
            "{extra:?} did not split the box containing the peak"
        );
        // Every accepted patch still satisfies the sampled criterion, including
        // its extra points.
        assert!(patches
            .patches()
            .iter()
            .all(|patch| patch.error_estimate <= 1e-6));
        let feature_patch = patch_containing(&patches, &[0.34]);
        let evaluated = patches
            .evaluate(ColMajorArrayRef::new(&[0.34], &[1, 1]).unwrap())
            .unwrap();
        assert!((evaluated[0] - 1.0).abs() <= 1e-3);
        assert!(depth(feature_patch) > 0);
    }
}

#[test]
fn extra_validation_runs_once_and_is_not_inherited() {
    // `max_depth = 0` stops the run at the root, so the recorded batches are
    // exactly the root's node grid, its validation grid and one extra batch.
    let record = Mutex::new(Recorded::default());
    let error = interpolate_chebyshev_patches(
        &context(),
        |points: ColMajorArrayRef<'_, f64>, _exec| {
            record.lock().unwrap().batches.push(points.shape()[1]);
            values(points, &feature)
        },
        &[0.0],
        &[1.0],
        4,
        &ChebyshevPatchOptions {
            tolerance: 1e-6,
            max_depth: 0,
            extra_validation: Some(ExtraValidation::Fixed { points_per_axis: 4 }),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        InterpolativeQttError::ChebyshevNotConverged { .. }
    ));
    // Root nodes, root validation grid, then exactly one extra batch: the extra
    // points are sampled once and never stored for the children.
    assert_eq!(record.into_inner().unwrap().batches, vec![5, 9, 4]);

    // A run that does split still samples only the grid for its children, and
    // their inherited values equal a fresh sample of their own nodes.
    let patches = build(
        feature,
        &[0.0],
        &[1.0],
        4,
        &ChebyshevPatchOptions {
            tolerance: 1e-6,
            extra_validation: Some(ExtraValidation::Fixed { points_per_axis: 4 }),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(patches.patches().iter().any(|patch| depth(patch) > 0));
    for patch in patches.patches() {
        assert_same_bits(&patch.values, &sampled_nodes(&feature, patch, 4));
    }
}

#[test]
fn greedy_stays_within_its_budget() {
    let starts = 2;
    let max_steps = 5;
    let record = Mutex::new(Recorded::default());
    let patches = interpolate_chebyshev_patches(
        &context(),
        |points: ColMajorArrayRef<'_, f64>, _exec| {
            record.lock().unwrap().batches.push(points.shape()[1]);
            values(points, &feature)
        },
        &[0.0],
        &[1.0],
        4,
        &ChebyshevPatchOptions {
            tolerance: 1e-6,
            extra_validation: Some(ExtraValidation::Greedy { starts, max_steps }),
            ..Default::default()
        },
    )
    .unwrap();
    let recorded = record.into_inner().unwrap();
    let grid: usize = recorded
        .batches
        .iter()
        .filter(|&&length| length == 5 || length == 9)
        .sum();
    assert_eq!(
        patches.num_evaluations(),
        recorded.batches.iter().sum::<usize>()
    );
    let extra = patches.num_evaluations() - grid;
    let paying = boxes_processed(&patches);
    assert!(
        extra <= paying * starts * max_steps,
        "extra {extra} for {paying} paying boxes"
    );
    assert!(extra > 0, "the greedy search spent nothing");
}
