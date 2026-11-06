//! The scalar chain kernel against its original loop, bit for bit.
//!
//! `scalar_chain_message_contraction` serves the calls too small for the grouped matrix product, among them every
//! one-point readout. Its result must not depend on how the loops are arranged: each output value is the sum over
//! the child values in ascending order, started from zero.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::*;

/// The kernel as it was before the loops were rearranged: one running sum per output value, the offset rebuilt
/// from the three axis values at every step.
fn reference_scalar_chain_message_contraction<T>(
    spec: ChainContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child_columns: &[T],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let ChainContractionSpec {
        strides,
        physical_axis,
        parent_axis,
        child_axis,
        parent_dim,
        child_dim,
    } = spec;
    let mut output = vec![T::default(); physical_values.len() * parent_dim];
    for (point, &physical_value) in physical_values.iter().enumerate() {
        for parent_value in 0..parent_dim {
            let mut sum = T::default();
            for child_value in 0..child_dim {
                let mut axis_values = [0usize; 3];
                axis_values[physical_axis] = physical_value;
                axis_values[parent_axis] = parent_value;
                axis_values[child_axis] = child_value;
                let flat = axis_values[0] * strides[0]
                    + axis_values[1] * strides[1]
                    + axis_values[2] * strides[2];
                sum += *raw
                    .get(flat)
                    .ok_or_else(|| anyhow::anyhow!("chain tensor offset is out of bounds"))?
                    * child_columns[point * child_dim + child_value];
            }
            output[point * parent_dim + parent_value] = sum;
        }
    }
    Ok(output)
}

const AXIS_ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Every order of the three axes, small and awkward sizes, 0 to 5 points: bitwise equal to the original loop.
fn assert_matches_reference<T>(mut random: impl FnMut(&mut ChaCha8Rng) -> T, seed: u64)
where
    T: Copy + Default + PartialEq + Debug + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for [physical_axis, parent_axis, child_axis] in AXIS_ORDERS {
        for (physical_dim, parent_dim, child_dim) in [
            (2, 1, 1),
            (2, 3, 2),
            (2, 5, 7),
            (3, 4, 4),
            (2, 9, 3),
            (1, 6, 6),
        ] {
            let mut dims = [0usize; 3];
            dims[physical_axis] = physical_dim;
            dims[parent_axis] = parent_dim;
            dims[child_axis] = child_dim;
            let spec = ChainContractionSpec {
                strides: [1, dims[0], dims[0] * dims[1]],
                physical_axis,
                parent_axis,
                child_axis,
                parent_dim,
                child_dim,
            };
            let raw: Vec<T> = (0..dims.iter().product::<usize>())
                .map(|_| random(&mut rng))
                .collect();
            for point_count in 0..=5 {
                let physical_values: Vec<usize> = (0..point_count)
                    .map(|_| rng.random_range(0..physical_dim))
                    .collect();
                let child_columns: Vec<T> = (0..point_count * child_dim)
                    .map(|_| random(&mut rng))
                    .collect();
                let expected = reference_scalar_chain_message_contraction(
                    spec,
                    &raw,
                    &physical_values,
                    &child_columns,
                )
                .unwrap();
                let actual =
                    scalar_chain_message_contraction(spec, &raw, &physical_values, &child_columns)
                        .unwrap();
                assert_eq!(
                    actual, expected,
                    "axes ({physical_axis}, {parent_axis}, {child_axis}), dims ({physical_dim}, {parent_dim}, \
                     {child_dim}), {point_count} points"
                );
            }
        }
    }
}

#[test]
fn scalar_chain_kernel_is_bitwise_the_original_loop_for_f64() {
    assert_matches_reference(
        |rng| rng.random_range(-1.0..1.0_f64) * 10f64.powi(rng.random_range(-3..4)),
        1,
    );
}

#[test]
fn scalar_chain_kernel_is_bitwise_the_original_loop_for_f32() {
    assert_matches_reference(|rng| rng.random_range(-1.0..1.0_f32), 2);
}

#[test]
fn scalar_chain_kernel_is_bitwise_the_original_loop_for_complex64() {
    assert_matches_reference(
        |rng| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)),
        3,
    );
}

#[test]
fn scalar_chain_kernel_keeps_non_finite_values_and_signed_zero() {
    let spec = ChainContractionSpec {
        strides: [1, 2, 4],
        physical_axis: 0,
        parent_axis: 1,
        child_axis: 2,
        parent_dim: 2,
        child_dim: 2,
    };
    let raw = [f64::INFINITY, -0.0, 0.0, f64::NAN, 1.0, -0.0, -0.0, 2.0];
    let physical_values = [0usize, 1];
    let child_columns = [0.0, 1.0, -0.0, -0.0];
    let expected =
        reference_scalar_chain_message_contraction(spec, &raw, &physical_values, &child_columns)
            .unwrap();
    let actual =
        scalar_chain_message_contraction(spec, &raw, &physical_values, &child_columns).unwrap();
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&actual), bits(&expected));
}

#[test]
fn scalar_chain_kernel_rejects_a_tensor_that_is_too_short_for_the_physical_value() {
    let spec = ChainContractionSpec {
        strides: [1, 2, 4],
        physical_axis: 0,
        parent_axis: 1,
        child_axis: 2,
        parent_dim: 2,
        child_dim: 2,
    };
    let raw = [1.0_f64; 8];
    let error = scalar_chain_message_contraction(spec, &raw, &[2], &[1.0, 1.0]).unwrap_err();
    assert!(error.to_string().contains("out of bounds"), "{error}");
}
