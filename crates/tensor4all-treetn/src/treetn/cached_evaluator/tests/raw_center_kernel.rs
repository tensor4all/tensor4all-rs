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
    for bonds in bond_sets {
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
                let point_count = 6;
                let physical_values: Vec<usize> = (0..point_count)
                    .map(|_| rng.random_range(0..physical_dim))
                    .collect();
                let mut components: Vec<RawCenterComponent<T>> = bond_axes
                    .iter()
                    .zip(bonds)
                    .map(|(&axis, &dim)| {
                        let assignments = 2;
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
                assert_eq!(
                    actual, expected,
                    "bonds {bonds:?}, physical axis {physical_axis}, descending {descending}"
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

/// The fold and the reference on one point of a 5 × 5 centre (one full lane pass and a remainder) whose core and
/// component entries cycle through `specials`.
fn fold_and_reference(specials: &[f64]) -> (f64, f64) {
    let dims = [1, 5, 5];
    let core: Vec<f64> = (0..25)
        .map(|k| specials[(k * 7) % specials.len()])
        .collect();
    let components: Vec<RawCenterComponent<f64>> = [2, 1]
        .into_iter()
        .map(|axis| RawCenterComponent {
            axis,
            dim: 5,
            point_to_assignment: vec![0],
            values: (0..5)
                .map(|k| specials[(k * 3 + axis) % specials.len()])
                .collect(),
        })
        .collect();
    let strides = [1, 1, 5];
    let expected = reference_fold(&core, &strides, &components, 0, 0, 0);
    let actual = contract_raw_center(&core, &dims, 0, &[0], &components).unwrap();
    assert_eq!(actual.len(), 1);
    (actual[0], expected)
}

#[test]
fn raw_center_fold_keeps_non_finite_values_and_signed_zero() {
    // NaN anywhere: the result is NaN in both (sign and payload are not specified).
    let (actual, expected) = fold_and_reference(&[
        f64::INFINITY,
        -0.0,
        0.0,
        f64::NAN,
        1.0,
        -0.0,
        f64::NEG_INFINITY,
        2.0,
        -1e300,
        1e300,
    ]);
    assert!(actual.is_nan() && expected.is_nan(), "{actual} {expected}");

    // Without NaN, bit for bit: signed zeros, products that overflow to ±inf, and +inf with positive weights.
    for specials in [
        &[-0.0, 0.0, 1.0, -0.0, 2.0, -1.0][..],
        &[-0.0, -1.0, -0.0, -2.0][..],
        &[1e300, 2.0, 1e300, 1.0, 3.0][..],
        &[f64::INFINITY, 1.0, 2.0, 0.5][..],
    ] {
        let (actual, expected) = fold_and_reference(specials);
        assert!(
            !expected.is_nan(),
            "{specials:?} gives NaN; use a NaN-free case"
        );
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{specials:?}: {actual} {expected}"
        );
    }
}
