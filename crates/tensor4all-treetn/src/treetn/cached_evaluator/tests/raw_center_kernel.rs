//! The raw center fold against its original recursion, bit for bit.
//!
//! `contract_raw_center` serves every point evaluated around an internal center. Its value must not depend on how
//! the loops are arranged: each level's sum adds, in ascending bond order, the inner sum times its component value,
//! started from zero, and the innermost sum adds the core entries times its component values the same way.
//! Every non-NaN result is compared bit for bit (signed zeros and infinities included). A NaN result is NaN in
//! both, but its sign and payload follow the operand order inside each addition and the target's floating-point
//! unit (they differ between x86_64 and aarch64), so they are not part of the contract.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::*;

/// The fold as it was before the next-to-innermost level computed several inner sums per pass.
fn reference_fold<T>(
    core: &[T],
    strides: &[usize],
    components: &[RawCenterComponent<T>],
    point: usize,
    level: usize,
    offset: usize,
) -> T
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let component = &components[level];
    let assignment = component.point_to_assignment[point];
    let innermost = level + 1 == components.len();
    let mut sum = T::default();
    for bond in 0..component.dim {
        let value = component.values[assignment * component.dim + bond];
        let offset = offset + bond * strides[component.axis];
        let inner = if innermost {
            core[offset]
        } else {
            reference_fold(core, strides, components, point, level + 1, offset)
        };
        sum += inner * value;
    }
    sum
}

/// Bond dimensions below, at and above the lane count, with remainders, for two to four components; the physical
/// axis first or last; components in descending (as the evaluator builds them) and ascending axis order.
fn assert_matches_reference<T>(mut random: impl FnMut(&mut ChaCha8Rng) -> T, seed: u64)
where
    T: Copy + Default + PartialEq + Debug + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let bond_sets: [&[usize]; 7] = [
        &[1, 1],
        &[3, 5],
        &[4, 9],
        &[7, 4, 6],
        &[9, 13, 5],
        &[2, 5, 3, 8],
        &[6, 1, 9, 4],
    ];
    // 1, 4, 6, 9 and 13 points: zero to three full four-point blocks and every remainder.
    for (bonds, point_count) in bond_sets
        .iter()
        .flat_map(|&bonds| [1, 4, 6, 9, 13].map(|points| (bonds, points)))
    {
        for physical_last in [false, true] {
            for descending in [true, false] {
                let physical_dim = 3;
                let physical_axis = if physical_last { bonds.len() } else { 0 };
                let bond_axes: Vec<usize> =
                    (0..=bonds.len()).filter(|&a| a != physical_axis).collect();
                let mut dims = vec![0; bonds.len() + 1];
                dims[physical_axis] = physical_dim;
                for (&axis, &dim) in bond_axes.iter().zip(bonds) {
                    dims[axis] = dim;
                }
                let strides: Vec<usize> = dims
                    .iter()
                    .scan(1, |stride, &dim| {
                        let current = *stride;
                        *stride *= dim;
                        Some(current)
                    })
                    .collect();
                let core: Vec<T> = (0..dims.iter().product::<usize>())
                    .map(|_| random(&mut rng))
                    .collect();
                let physical_values: Vec<usize> = (0..point_count)
                    .map(|_| rng.random_range(0..physical_dim))
                    .collect();
                let mut components: Vec<RawCenterComponent<T>> = bond_axes
                    .iter()
                    .zip(bonds)
                    .map(|(&axis, &dim)| {
                        // Up to one assignment per point, so no block can reuse another's values unnoticed.
                        let assignments = point_count;
                        RawCenterComponent {
                            axis,
                            dim,
                            point_to_assignment: (0..point_count)
                                .map(|_| rng.random_range(0..assignments))
                                .collect(),
                            values: (0..assignments * dim).map(|_| random(&mut rng)).collect(),
                        }
                    })
                    .collect();
                if descending {
                    components.reverse();
                }
                let expected: Vec<T> = physical_values
                    .iter()
                    .enumerate()
                    .map(|(point, &physical)| {
                        reference_fold(
                            &core,
                            &strides,
                            &components,
                            point,
                            0,
                            physical * strides[physical_axis],
                        )
                    })
                    .collect();
                reset_raw_center_core_visits_for_test();
                let actual =
                    contract_raw_center(&core, &dims, physical_axis, &physical_values, &components)
                        .unwrap();
                // Debug formatting is the shortest exact representation, so it tells -0.0 from +0.0.
                assert_eq!(
                    format!("{actual:?}"),
                    format!("{expected:?}"),
                    "bonds {bonds:?}, {point_count} points, physical axis {physical_axis}, descending {descending}"
                );
                assert_eq!(
                    raw_center_core_visits_for_test(),
                    point_count * bonds.iter().product::<usize>(),
                    "every point visits each core entry of its physical slice once"
                );
            }
        }
    }
}

#[test]
fn raw_center_fold_is_bitwise_the_original_recursion_for_f64() {
    assert_matches_reference(
        |rng| rng.random_range(-1.0..1.0_f64) * 10f64.powi(rng.random_range(-3..4)),
        11,
    );
}

#[test]
fn raw_center_fold_is_bitwise_the_original_recursion_for_f32() {
    assert_matches_reference(|rng| rng.random_range(-1.0..1.0_f32), 12);
}

#[test]
fn raw_center_fold_is_bitwise_the_original_recursion_for_complex64() {
    assert_matches_reference(
        |rng| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)),
        13,
    );
}

/// The fold and the reference on five points (one four-point block and one point alone) of a 5 × 5 centre (one
/// full lane pass and a remainder); the core and each point's component values cycle through `specials`.
fn fold_and_reference(specials: &[f64]) -> Vec<(f64, f64)> {
    let points = 5;
    let dims = [1, 5, 5];
    let core: Vec<f64> = (0..25)
        .map(|k| specials[(k * 7) % specials.len()])
        .collect();
    let components: Vec<RawCenterComponent<f64>> = [2, 1]
        .into_iter()
        .map(|axis| RawCenterComponent {
            axis,
            dim: 5,
            point_to_assignment: (0..points).collect(),
            values: (0..points * 5)
                .map(|k| specials[(k * 3 + axis) % specials.len()])
                .collect(),
        })
        .collect();
    let strides = [1, 1, 5];
    let actual = contract_raw_center(&core, &dims, 0, &[0; 5], &components).unwrap();
    assert_eq!(actual.len(), points);
    (0..points)
        .map(|point| {
            (
                actual[point],
                reference_fold(&core, &strides, &components, point, 0, 0),
            )
        })
        .collect()
}

/// #1181 review: eight points in two four-point blocks, each with its own value. A block that took its points'
/// assignments from the first block would give [1, 2, 3, 4, 1, 2, 3, 4].
#[test]
fn raw_center_fold_numbers_the_points_of_every_block() {
    let components = vec![
        RawCenterComponent {
            axis: 2,
            dim: 1,
            point_to_assignment: (0..8).collect(),
            values: (1..=8).map(f64::from).collect(),
        },
        RawCenterComponent {
            axis: 1,
            dim: 1,
            point_to_assignment: vec![0; 8],
            values: vec![1.0],
        },
    ];
    let actual = contract_raw_center(&[1.0], &[1, 1, 1], 0, &[0; 8], &components).unwrap();
    assert_eq!(actual, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

/// Special values through the four-point path, against literals computed independently (same IEEE operations and
/// order): every non-NaN result bit for bit, sign included. Sums start from +0.0, so a zero result is +0.0 even when
/// every term is ±0.0; a NaN result is NaN in both, its sign and payload are not part of the contract.
#[test]
fn raw_center_fold_keeps_non_finite_values_and_signed_zero() {
    let inf = f64::INFINITY;
    let cases: [(&[f64], [f64; 5]); 8] = [
        (
            &[-0.0, 0.0, 1.0, -0.0, 2.0, -1.0],
            [-10.0, -22.0, -10.0, -22.0, -10.0],
        ),
        (
            &[-0.0, -1.0, -0.0, -2.0],
            [-18.0, -18.0, -24.0, -21.0, -18.0],
        ),
        (&[-0.0, 0.0, -0.0], [0.0; 5]),
        (&[1e300, 2.0, 1e300, 1.0, 3.0], [inf; 5]),
        (&[inf, 1.0, 2.0, 0.5], [inf; 5]),
        (&[1e300, -1.0], [-inf; 5]),
        (&[-inf, 1.0, 2.0], [-inf; 5]),
        (
            &[
                inf,
                -0.0,
                0.0,
                f64::NAN,
                1.0,
                -0.0,
                -inf,
                2.0,
                -1e300,
                1e300,
            ],
            [f64::NAN; 5],
        ),
    ];
    for (specials, literal) in cases {
        for (point, ((actual, reference), literal)) in fold_and_reference(specials)
            .into_iter()
            .zip(literal)
            .enumerate()
        {
            for (what, value) in [("fold", actual), ("reference", reference)] {
                if literal.is_nan() {
                    assert!(value.is_nan(), "{specials:?} point {point}: {what} {value}");
                } else {
                    assert_eq!(
                        value.to_bits(),
                        literal.to_bits(),
                        "{specials:?} point {point}: {what} {value}, expected {literal}"
                    );
                }
            }
        }
    }
}
