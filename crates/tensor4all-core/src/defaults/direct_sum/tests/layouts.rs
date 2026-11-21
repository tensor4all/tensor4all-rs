//! Binary direct sum on operands whose axes are in different orders, against
//! an independent element-by-element reference.
//!
//! The kernel copies each operand from its own layout. Depending on how the
//! operand's axes line up with the result, it moves the whole block at once,
//! merges a contiguous prefix of axes into one run, or walks the first axis
//! with the operand's stride; these cases pin all three.

use super::super::*;
use crate::index_like::IndexLike;
use num_complex::{Complex32, Complex64};

trait TestScalar: TensorElement + Zero + Copy + PartialEq + std::fmt::Debug {
    fn make(k: usize) -> Self;
}

impl TestScalar for f64 {
    fn make(k: usize) -> Self {
        1.0 + k as f64
    }
}

impl TestScalar for f32 {
    fn make(k: usize) -> Self {
        1.0 + k as f32
    }
}

impl TestScalar for Complex32 {
    fn make(k: usize) -> Self {
        Complex32::new(1.0 + k as f32, -(k as f32) - 0.5)
    }
}

impl TestScalar for Complex64 {
    fn make(k: usize) -> Self {
        Complex64::new(1.0 + k as f64, -(k as f64) - 0.5)
    }
}

fn filled<T: TestScalar>(indices: Vec<DynIndex>, seed: usize) -> IdxTensor {
    let len: usize = indices.iter().map(|index| index.dim()).product();
    IdxTensor::from_dense(indices, (0..len).map(|k| T::make(seed + k)).collect()).unwrap()
}

/// Walks every element of both operands in their own column-major layouts and
/// writes it to its block position in the result.
fn reference<T: TestScalar>(
    a: &IdxTensor,
    b: &IdxTensor,
    pairs: &[(DynIndex, DynIndex)],
    result: &IdxTensor,
) -> Vec<T> {
    let result_dims = result.dims();
    let n_common = result_dims.len() - pairs.len();
    let mut expected = vec![T::zero(); result_dims.iter().product()];
    for (side, operand) in [a, b].into_iter().enumerate() {
        let dims = operand.dims();
        for (linear, value) in operand.to_vec::<T>().unwrap().into_iter().enumerate() {
            let mut rest = linear;
            let mut coordinates = vec![0usize; result_dims.len()];
            for (axis, index) in operand.indices().iter().enumerate() {
                let coordinate = rest % dims[axis];
                rest /= dims[axis];
                let paired = pairs
                    .iter()
                    .position(|(left, right)| index == if side == 0 { left } else { right });
                match paired {
                    Some(p) => {
                        let offset = if side == 0 { 0 } else { pairs[p].0.dim() };
                        coordinates[n_common + p] = offset + coordinate;
                    }
                    None => {
                        let position = result.indices().iter().position(|own| own == index);
                        coordinates[position.unwrap()] = coordinate;
                    }
                }
            }
            let (mut offset, mut stride) = (0, 1);
            for (axis, &coordinate) in coordinates.iter().enumerate() {
                offset += coordinate * stride;
                stride *= result_dims[axis];
            }
            expected[offset] = value;
        }
    }
    expected
}

fn assert_matches_reference<T: TestScalar>(
    a: &IdxTensor,
    b: &IdxTensor,
    pairs: &[(DynIndex, DynIndex)],
) {
    let (result, _) = direct_sum(a, b, pairs).unwrap();
    assert_eq!(
        result.to_vec::<T>().unwrap(),
        reference::<T>(a, b, pairs, &result)
    );
}

/// Two common and two paired axes, unequal bond dimensions, and each operand's
/// axes in a different order: `a` merges a prefix of two axes, `b` is walked
/// with a non-unit stride on the first result axis.
fn check_permuted_operands<T: TestScalar>() {
    let (s, t) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    let (l0, r0) = (DynIndex::new_dyn(1), DynIndex::new_dyn(2));
    let (l1, r1) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    let a = filled::<T>(vec![s.clone(), t.clone(), r0.clone(), l0.clone()], 0);
    let b = filled::<T>(vec![r1.clone(), t.clone(), s.clone(), l1.clone()], 100);

    assert_matches_reference::<T>(&a, &b, &[(l0, l1), (r0, r1)]);
}

#[test]
fn direct_sum_of_permuted_operands_matches_reference_f64() {
    check_permuted_operands::<f64>();
}

#[test]
fn direct_sum_of_permuted_operands_matches_reference_c64() {
    check_permuted_operands::<Complex64>();
}

#[test]
fn direct_sum_of_permuted_operands_matches_reference_f32() {
    check_permuted_operands::<f32>();
}

#[test]
fn direct_sum_of_permuted_operands_matches_reference_c32() {
    check_permuted_operands::<Complex32>();
}

/// A chain core `(left, site, right)`: the site is the common axis, so in
/// result order `(site, left, right)` neither operand is contiguous along the
/// first axis.
#[test]
fn direct_sum_of_chain_cores_matches_reference() {
    let site = DynIndex::new_dyn(2);
    let (l0, r0, l1, r1) = (
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(4),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(5),
    );
    let a = filled::<f64>(vec![l0.clone(), site.clone(), r0.clone()], 0);
    let b = filled::<f64>(vec![l1.clone(), site, r1.clone()], 1000);

    assert_matches_reference::<f64>(&a, &b, &[(l0, l1), (r0, r1)]);
}

/// Both operands already in result order. Along the second paired axis each
/// operand is contiguous in its own buffer but not in the result, whose extent
/// there is the summed dimension, so the copy must not merge that axis.
#[test]
fn direct_sum_keeps_result_strides_for_operands_in_result_order() {
    let (l0, r0) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let (l1, r1) = (DynIndex::new_dyn(1), DynIndex::new_dyn(3));
    let a = filled::<f64>(vec![l0.clone(), r0.clone()], 0);
    let b = filled::<f64>(vec![l1.clone(), r1.clone()], 10);

    assert_matches_reference::<f64>(&a, &b, &[(l0, l1), (r0, r1)]);
}

/// Operands of the same kind but different precision promote to the wider
/// type of that kind, and every copied value is exact there.
#[test]
fn direct_sum_promotes_mixed_precision() {
    let (l0, l1) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let pairs = [(l0.clone(), l1.clone())];

    let f64_tensor = IdxTensor::from_dense(vec![l0.clone()], vec![1.0_f64, 2.0]).unwrap();
    let f32_tensor = IdxTensor::from_dense(vec![l1.clone()], vec![3.0_f32, 4.0]).unwrap();
    let (real, _) = direct_sum(&f64_tensor, &f32_tensor, &pairs).unwrap();
    assert!(real.is_f64());
    assert_eq!(real.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let c64_tensor = IdxTensor::from_dense(
        vec![l0.clone()],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)],
    )
    .unwrap();
    let c32_tensor = IdxTensor::from_dense(
        vec![l1.clone()],
        vec![Complex32::new(3.0, -1.0), Complex32::new(4.0, -2.0)],
    )
    .unwrap();
    let (complex, _) = direct_sum(&c64_tensor, &c32_tensor, &pairs).unwrap();
    assert!(complex.is_c64());
    assert_eq!(
        complex.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(4.0, -2.0),
        ]
    );
}

/// The real and complex kind cross the real precision boundary the way the
/// eager arithmetic does: `f64 + Complex32 -> Complex64`, and `f32 + Complex64`
/// keeps the complex value.
#[test]
fn direct_sum_promotes_across_the_real_complex_boundary() {
    let (l0, l1) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let pairs = [(l0.clone(), l1.clone())];

    let f64_tensor = IdxTensor::from_dense(vec![l0.clone()], vec![1.5_f64, 2.5]).unwrap();
    let c32_tensor = IdxTensor::from_dense(
        vec![l1.clone()],
        vec![Complex32::new(3.5, 0.25), Complex32::new(4.5, -0.25)],
    )
    .unwrap();
    let (mixed, _) = direct_sum(&f64_tensor, &c32_tensor, &pairs).unwrap();
    assert!(mixed.is_c64());
    assert_eq!(
        mixed.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(1.5, 0.0),
            Complex64::new(2.5, 0.0),
            Complex64::new(3.5, 0.25),
            Complex64::new(4.5, -0.25),
        ]
    );

    let f32_tensor = IdxTensor::from_dense(vec![l0], vec![5.0_f32, 6.0]).unwrap();
    let c64_tensor = IdxTensor::from_dense(
        vec![l1],
        vec![Complex64::new(7.0, 0.5), Complex64::new(8.0, -0.5)],
    )
    .unwrap();
    let (wide, _) = direct_sum(&f32_tensor, &c64_tensor, &pairs).unwrap();
    assert!(wide.is_c64());
    assert_eq!(
        wide.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.5),
            Complex64::new(8.0, -0.5),
        ]
    );
}
