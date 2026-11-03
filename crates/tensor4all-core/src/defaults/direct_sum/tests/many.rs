//! N-ary direct sum against an independent element-by-element reference.

use super::super::*;
use crate::index_like::IndexLike;
use num_complex::Complex64;

/// Deterministic, distinct, nonzero test values for every supported dtype.
trait TestScalar: TensorElement + Zero + Copy + PartialEq + std::fmt::Debug {
    fn make(k: usize) -> Self;
}

impl TestScalar for f64 {
    fn make(k: usize) -> Self {
        1.0 + k as f64
    }
}

impl TestScalar for Complex64 {
    fn make(k: usize) -> Self {
        Complex64::new(1.0 + k as f64, -(k as f64) - 0.5)
    }
}

fn filled<T: TestScalar>(indices: Vec<DynIndex>, seed: usize) -> IdxTensor {
    let len: usize = indices.iter().map(|index| index.dim()).product();
    let values = (0..len).map(|k| T::make(seed + k)).collect();
    IdxTensor::from_dense(indices, values).unwrap()
}

/// Reference result: walks every element of every operand in its own
/// column-major layout and writes it to its block position.
fn reference<T: TestScalar>(
    operands: &[&IdxTensor],
    groups: &[Vec<DynIndex>],
    result: &IdxTensor,
) -> Vec<T> {
    let result_dims = result.dims();
    let n_common = result_dims.len() - groups.len();
    let mut expected = vec![T::zero(); result_dims.iter().product()];
    for (i, operand) in operands.iter().enumerate() {
        let dims = operand.dims();
        let values = operand.to_vec::<T>().unwrap();
        for (linear, &value) in values.iter().enumerate() {
            let mut rest = linear;
            let mut coordinates = vec![0usize; result_dims.len()];
            for (axis, index) in operand.indices().iter().enumerate() {
                let coordinate = rest % dims[axis];
                rest /= dims[axis];
                let result_axis = match groups.iter().position(|group| &group[i] == index) {
                    Some(g) => {
                        let offset: usize = groups[g][..i].iter().map(|index| index.dim()).sum();
                        coordinates[n_common + g] = offset + coordinate;
                        continue;
                    }
                    None => result
                        .indices()
                        .iter()
                        .position(|own| own == index)
                        .unwrap(),
                };
                coordinates[result_axis] = coordinate;
            }
            let mut offset = 0;
            let mut stride = 1;
            for (axis, &coordinate) in coordinates.iter().enumerate() {
                offset += coordinate * stride;
                stride *= result_dims[axis];
            }
            expected[offset] = value;
        }
    }
    expected
}

/// Three operands, two common indices and two groups, with each operand's
/// axes in a different order and unequal bond dimensions per operand.
fn check_three_operands_match_reference<T: TestScalar>() {
    let s = DynIndex::new_dyn(2);
    let t = DynIndex::new_dyn(3);
    let l: Vec<DynIndex> = [1, 2, 3].iter().map(|&d| DynIndex::new_dyn(d)).collect();
    let r: Vec<DynIndex> = [2, 1, 2].iter().map(|&d| DynIndex::new_dyn(d)).collect();

    let a = filled::<T>(vec![s.clone(), l[0].clone(), t.clone(), r[0].clone()], 0);
    let b = filled::<T>(vec![r[1].clone(), t.clone(), s.clone(), l[1].clone()], 100);
    let c = filled::<T>(vec![l[2].clone(), r[2].clone(), s.clone(), t.clone()], 200);
    let operands = [&a, &b, &c];
    let groups = vec![l.clone(), r.clone()];

    let (result, new_indices) = direct_sum_many(&operands, &groups).unwrap();

    assert_eq!(new_indices.len(), 2);
    assert_eq!(new_indices[0].dim(), 6);
    assert_eq!(new_indices[1].dim(), 5);
    assert_eq!(
        result.indices(),
        &[s, t, new_indices[0].clone(), new_indices[1].clone()]
    );
    let expected = reference::<T>(&operands, &groups, &result);
    assert_eq!(result.to_vec::<T>().unwrap(), expected);
}

#[test]
fn direct_sum_many_matches_reference_f64() {
    check_three_operands_match_reference::<f64>();
}

#[test]
fn direct_sum_many_matches_reference_c64() {
    check_three_operands_match_reference::<Complex64>();
}

#[test]
fn direct_sum_many_without_common_indices_is_block_diagonal() {
    let i: Vec<DynIndex> = [2, 1].iter().map(|&d| DynIndex::new_dyn(d)).collect();
    let j: Vec<DynIndex> = [1, 2].iter().map(|&d| DynIndex::new_dyn(d)).collect();
    let a = IdxTensor::from_dense(vec![i[0].clone(), j[0].clone()], vec![1.0, 2.0]).unwrap();
    let b = IdxTensor::from_dense(vec![j[1].clone(), i[1].clone()], vec![3.0, 4.0]).unwrap();

    let (result, _) = direct_sum_many(&[&a, &b], &[i, j]).unwrap();

    // 3 x 3 column-major: a fills rows 0..2 of column 0, b row 2 of columns 1..3.
    assert_eq!(
        result.to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0]
    );
}

#[test]
fn direct_sum_many_of_one_operand_renames_the_grouped_index() {
    let s = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let a = filled::<f64>(vec![bond.clone(), s.clone()], 0);

    let (result, new_indices) = direct_sum_many(&[&a], &[vec![bond.clone()]]).unwrap();

    assert_eq!(result.indices(), &[s.clone(), new_indices[0].clone()]);
    let renamed = a.replaceind(&bond, &new_indices[0]).unwrap();
    assert!(result.distance(&renamed).unwrap() < 1e-14);
}

#[test]
fn direct_sum_many_rejects_empty_operands() {
    let err = direct_sum_many(&[], &[vec![]]).unwrap_err();
    assert!(
        format!("{err:#}").contains("at least one operand"),
        "{err:#}"
    );
}

#[test]
fn direct_sum_many_rejects_group_without_one_index_per_operand() {
    let s = DynIndex::new_dyn(2);
    let (l0, l1) = (DynIndex::new_dyn(1), DynIndex::new_dyn(1));
    let a = filled::<f64>(vec![s.clone(), l0.clone()], 0);
    let b = filled::<f64>(vec![s, l1], 10);

    let err = direct_sum_many(&[&a, &b], &[vec![l0]]).unwrap_err();
    assert!(
        format!("{err:#}").contains("has 1 entries, expected one per operand (2)"),
        "{err:#}"
    );
}

#[test]
fn direct_sum_many_rejects_index_in_two_groups() {
    let (l0, l1, r1) = (
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    );
    let a = filled::<f64>(vec![l0.clone()], 0);
    let b = filled::<f64>(vec![l1, r1.clone()], 10);

    let err =
        direct_sum_many(&[&a, &b], &[vec![l0.clone(), r1.clone()], vec![l0, r1]]).unwrap_err();
    assert!(
        format!("{err:#}").contains("appears in two groups"),
        "{err:#}"
    );
}

#[test]
fn direct_sum_many_rejects_extra_common_index_on_a_later_operand() {
    let s = DynIndex::new_dyn(2);
    let extra = DynIndex::new_dyn(2);
    let l: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(1)).collect();
    let a = filled::<f64>(vec![s.clone(), l[0].clone()], 0);
    let b = filled::<f64>(vec![s.clone(), l[1].clone()], 10);
    let c = filled::<f64>(vec![s, extra, l[2].clone()], 20);

    match direct_sum_many(&[&a, &b, &c], &[l]) {
        Err(IdxTensorError::ShapeMismatch {
            operation,
            expected,
            ..
        }) => {
            assert_eq!(operation, "direct_sum");
            assert!(expected.contains("third operand"), "{expected}");
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn direct_sum_many_rejects_mixed_dtypes_in_a_later_operand() {
    let s = DynIndex::new_dyn(2);
    let l: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(1)).collect();
    let a = filled::<f64>(vec![s.clone(), l[0].clone()], 0);
    let b = filled::<f64>(vec![s.clone(), l[1].clone()], 10);
    let c = IdxTensor::from_dense(vec![s, l[2].clone()], vec![1.0_f32, 2.0]).unwrap();

    let err = direct_sum_many(&[&a, &b, &c], &[l]).unwrap_err();
    assert!(
        format!("{err:#}").contains("same dense scalar type"),
        "{err:#}"
    );
}

#[test]
fn direct_sum_many_rejects_a_tracked_operand() {
    let s = DynIndex::new_dyn(2);
    let (l0, l1) = (DynIndex::new_dyn(1), DynIndex::new_dyn(1));
    let a = filled::<f64>(vec![s.clone(), l0.clone()], 0);
    let b = filled::<f64>(vec![s, l1.clone()], 10)
        .enable_grad()
        .unwrap();
    assert!(b.tracks_grad());

    let err = direct_sum_many(&[&a, &b], &[vec![l0, l1]]).unwrap_err();
    assert!(format!("{err:#}").contains("tracked tensors"), "{err:#}");
}

/// An operand already laid out in result order is contiguous along its second
/// group axis in its own buffer but not in the result, whose extent there is
/// the summed dimension: the copy must not merge that axis into one run.
#[test]
fn direct_sum_many_keeps_result_strides_for_an_operand_in_result_order() {
    let l: Vec<DynIndex> = [2, 1].iter().map(|&d| DynIndex::new_dyn(d)).collect();
    let r: Vec<DynIndex> = [2, 1].iter().map(|&d| DynIndex::new_dyn(d)).collect();
    let a = filled::<f64>(vec![l[0].clone(), r[0].clone()], 0);
    let b = filled::<f64>(vec![l[1].clone(), r[1].clone()], 10);
    let operands = [&a, &b];
    let groups = vec![l, r];

    let (result, _) = direct_sum_many(&operands, &groups).unwrap();

    let expected = reference::<f64>(&operands, &groups, &result);
    assert_eq!(result.to_vec::<f64>().unwrap(), expected);
}

/// `direct_sum_many` equals repeated binary `direct_sum`, `((a (+) b) (+) c)`,
/// on operands of different shapes: different bond dimensions per operand,
/// different axis orders, and a group whose dimensions differ in every
/// operand.
#[test]
fn direct_sum_many_equals_repeated_binary_direct_sum() {
    let (s, t) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    let l: Vec<DynIndex> = [1, 3, 2].iter().map(|&d| DynIndex::new_dyn(d)).collect();
    let r: Vec<DynIndex> = [2, 1, 4].iter().map(|&d| DynIndex::new_dyn(d)).collect();
    let a = filled::<Complex64>(vec![s.clone(), l[0].clone(), t.clone(), r[0].clone()], 0);
    let b = filled::<Complex64>(vec![r[1].clone(), t.clone(), s.clone(), l[1].clone()], 100);
    let c = filled::<Complex64>(vec![l[2].clone(), r[2].clone(), t.clone(), s.clone()], 200);

    let (many, many_new) = direct_sum_many(&[&a, &b, &c], &[l.clone(), r.clone()]).unwrap();
    let (ab, ab_new) = direct_sum(
        &a,
        &b,
        &[(l[0].clone(), l[1].clone()), (r[0].clone(), r[1].clone())],
    )
    .unwrap();
    let (abc, abc_new) = direct_sum(
        &ab,
        &c,
        &[
            (ab_new[0].clone(), l[2].clone()),
            (ab_new[1].clone(), r[2].clone()),
        ],
    )
    .unwrap();

    assert_eq!(many.dims(), abc.dims());
    assert_eq!(&many.indices()[..2], &abc.indices()[..2]);
    assert_eq!(
        many_new.iter().map(|index| index.dim()).collect::<Vec<_>>(),
        abc_new.iter().map(|index| index.dim()).collect::<Vec<_>>()
    );
    assert_eq!(
        many.to_vec::<Complex64>().unwrap(),
        abc.to_vec::<Complex64>().unwrap()
    );
}
