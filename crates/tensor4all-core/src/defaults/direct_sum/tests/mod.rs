use super::*;
use crate::tagset::TagSetLike;
use num_complex::Complex64;

#[test]
fn checked_direct_sum_shapes_reject_overflow() {
    assert!(checked_product(&[usize::MAX, 2], "test").is_err());
    assert!(checked_strides(&[2, usize::MAX, 2], "test").is_err());
    assert!(multi_to_linear(&[usize::MAX], &[2]).is_err());
}

#[test]
fn test_direct_sum_simple() {
    // Create two tensors with one common index
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);

    // A[i, j] - 2x3 tensor
    let a = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![
            1.0, 2.0, 3.0, // i=0
            4.0, 5.0, 6.0, // i=1
        ],
    )
    .unwrap();

    // B[i, k] - 2x4 tensor
    let b = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![
            10.0, 20.0, 30.0, 40.0, // i=0
            50.0, 60.0, 70.0, 80.0, // i=1
        ],
    )
    .unwrap();

    // Direct sum along (j, k)
    let (result, new_indices) = direct_sum(&a, &b, &[(j.clone(), k.clone())]).unwrap();

    // Result should be 2x7 (common i, merged j+k)
    assert_eq!(result.dims().len(), 2);
    let result_dims = result.dims();
    assert_eq!(result_dims[0], 2); // i dimension
    assert_eq!(result_dims[1], 7); // j+k dimension (3+4)
    assert_eq!(new_indices.len(), 1);
    assert_eq!(new_indices[0].dim(), 7);

    // Check column-major logical values.
    let data = result.to_vec::<f64>().unwrap();
    assert_eq!(
        data,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,]
    );
}

#[test]
fn test_direct_sum_multiple_pairs() {
    // Create tensors with two indices to be paired
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(3);
    let l = DynIndex::new_dyn(3);

    // A[i, j] - 2x2 tensor
    let a = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    // B[k, l] - 3x3 tensor (no common indices)
    let b = IdxTensor::from_dense(
        vec![k.clone(), l.clone()],
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    )
    .unwrap();

    // Direct sum along (i, k) and (j, l)
    let (result, new_indices) =
        direct_sum(&a, &b, &[(i.clone(), k.clone()), (j.clone(), l.clone())]).unwrap();

    // Result should be 5x5 (2+3, 2+3)
    assert_eq!(result.dims().len(), 2);
    let result_dims = result.dims();
    assert_eq!(result_dims[0], 5);
    assert_eq!(result_dims[1], 5);
    assert_eq!(new_indices.len(), 2);
}

#[test]
fn test_direct_sum_preserves_same_id_prime_pair_common_indices() {
    let i = DynIndex::new_dyn(2);
    let i_prime = i.prime();
    let j_a = DynIndex::new_dyn(2);
    let j_b = DynIndex::new_dyn(3);

    let a = IdxTensor::from_dense(
        vec![i.clone(), i_prime.clone(), j_a.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let b = IdxTensor::from_dense(
        vec![i.clone(), i_prime.clone(), j_b.clone()],
        vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ],
    )
    .unwrap();

    let (result, new_indices) = direct_sum(&a, &b, &[(j_a, j_b)]).unwrap();

    assert_eq!(result.indices()[0], i);
    assert_eq!(result.indices()[1], i_prime);
    assert_eq!(result.indices()[2], new_indices[0]);
    assert_eq!(result.dims(), vec![2, 2, 5]);

    let data = result.to_vec::<f64>().unwrap();
    assert_eq!(&data[0..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        &data[8..],
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    );
}

#[test]
fn test_direct_sum_c64() {
    // Test the complex64 code path
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(3);

    // A[i, j] - 2x2 complex tensor
    let a = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 1.5),
            Complex64::new(4.0, 2.0),
        ],
    )
    .unwrap();

    // B[i, k] - 2x3 complex tensor
    let b = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![
            Complex64::new(10.0, 0.1),
            Complex64::new(20.0, 0.2),
            Complex64::new(30.0, 0.3),
            Complex64::new(40.0, 0.4),
            Complex64::new(50.0, 0.5),
            Complex64::new(60.0, 0.6),
        ],
    )
    .unwrap();

    // Direct sum along (j, k)
    let (result, new_indices) = direct_sum(&a, &b, &[(j.clone(), k.clone())]).unwrap();

    // Result should be 2x5 (common i, merged j+k = 2+3)
    let result_dims = result.dims();
    assert_eq!(result_dims[0], 2);
    assert_eq!(result_dims[1], 5);
    assert_eq!(new_indices.len(), 1);
    assert_eq!(new_indices[0].dim(), 5);

    // Verify the data is complex
    let data = result.to_vec::<Complex64>().unwrap();
    assert_eq!(data.len(), 10);
    // First block comes from A (j indices 0..2)
    assert_eq!(data[0], Complex64::new(1.0, 0.5));
    assert_eq!(data[1], Complex64::new(2.0, 1.0));
    // Second block comes from B (k indices, offset by dim_a=2)
    assert_eq!(data[4], Complex64::new(10.0, 0.1));
}

#[test]
fn test_direct_sum_mixed_types_error() {
    // One f64 and one complex tensor should error
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(3);

    let a = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64; 6]).unwrap();
    let b = IdxTensor::from_dense(
        vec![i.clone(), k.clone()],
        vec![Complex64::new(1.0, 0.0); 6],
    )
    .unwrap();

    let result = direct_sum(&a, &b, &[(j.clone(), k.clone())]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("same dense scalar type"));
}

#[test]
fn test_direct_sum_empty_pairs_error() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);

    let a = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64; 6]).unwrap();
    let b = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![2.0_f64; 6]).unwrap();

    let result = direct_sum(&a, &b, &[]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("at least one index pair"));
}

#[test]
fn test_direct_sum_index_not_in_first_tensor_error() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(3);
    let phantom = DynIndex::new_dyn(3); // not in either tensor

    let a = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64; 6]).unwrap();
    let b = IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![2.0_f64; 6]).unwrap();

    // phantom is not in tensor a
    let result = direct_sum(&a, &b, &[(phantom.clone(), k.clone())]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("not found in first tensor"));
}

#[test]
fn test_direct_sum_index_not_in_second_tensor_error() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(3);
    let phantom = DynIndex::new_dyn(3); // not in either tensor

    let a = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64; 6]).unwrap();
    let b = IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![2.0_f64; 6]).unwrap();

    // phantom is not in tensor b
    let result = direct_sum(&a, &b, &[(j.clone(), phantom.clone())]);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("not found in second tensor"));
}

#[test]
fn test_direct_sum_common_index_dimension_mismatch_error() {
    // Index identity ignores the dimension, so the same id and tags with a
    // different dimension is a common index whose extents disagree.
    let i = DynIndex::new_dyn(2);
    let i_wide = DynIndex::new_with_tags(*i.id(), 3, i.tags().clone());
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);
    let a = IdxTensor::from_dense(vec![i, j.clone()], vec![1.0_f64; 6]).unwrap();
    let b = IdxTensor::from_dense(vec![i_wide, k.clone()], vec![2.0_f64; 12]).unwrap();

    let err = direct_sum(&a, &b, &[(j, k)]).unwrap_err();
    assert!(
        format!("{err:#}").contains("Dimension mismatch for common index: 2 vs 3"),
        "{err:#}"
    );
}

#[test]
fn test_direct_sum_rejects_distinct_unpaired_indices() {
    // Formerly accepted: i_a and i_b are different indices, so each is
    // unpaired on one side only and would be read at coordinate 0.
    let i_a = DynIndex::new_dyn(2);
    let i_b = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);
    let a = IdxTensor::from_dense(vec![i_a, j.clone()], vec![1.0_f64; 6]).unwrap();
    let b = IdxTensor::from_dense(vec![i_b, k.clone()], vec![2.0_f64; 8]).unwrap();
    assert_unpaired_index_rejected(&a, &b, &[(j, k)]);
}

/// Asserts that `direct_sum` rejects the operands with the typed index-space
/// mismatch (tensor4all/tensor4all-rs#1018).
fn assert_unpaired_index_rejected(a: &IdxTensor, b: &IdxTensor, pairs: &[(DynIndex, DynIndex)]) {
    match direct_sum(a, b, pairs) {
        Err(IdxTensorError::ShapeMismatch { operation, .. }) => {
            assert_eq!(operation, "direct_sum");
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_direct_sum_rejects_right_only_unpaired_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let a = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let b = IdxTensor::from_dense(vec![i.clone(), j], vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_unpaired_index_rejected(&a, &b, &[(i.clone(), i)]);
}

#[test]
fn test_direct_sum_rejects_left_only_unpaired_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let a = IdxTensor::from_dense(vec![j, i.clone()], vec![1.0; 6]).unwrap();
    let b = IdxTensor::from_dense(vec![i.clone()], vec![3.0, 4.0]).unwrap();
    assert_unpaired_index_rejected(&a, &b, &[(i.clone(), i)]);
}

#[test]
fn test_direct_sum_rejects_unpaired_indices_differing_in_identity() {
    // Same id, but a different prime level or tag set is a different index,
    // so neither operand's unpaired index has a counterpart.
    let i = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let a = IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0; 4]).unwrap();
    let primed = IdxTensor::from_dense(vec![i.prime(), k.clone()], vec![1.0; 4]).unwrap();
    assert_unpaired_index_rejected(&a, &primed, &[(k.clone(), k.clone())]);

    let mut tags = i.tags().clone();
    tags.add_tag("other").unwrap();
    let tagged_i = DynIndex::new_with_tags(*i.id(), 2, tags);
    let tagged = IdxTensor::from_dense(vec![tagged_i, k.clone()], vec![1.0; 4]).unwrap();
    assert_unpaired_index_rejected(&a, &tagged, &[(k.clone(), k)]);
}

#[test]
fn test_direct_sum_rejects_index_paired_on_one_side_only() {
    // `i` is common to both operands but paired only as a's side of (i, j):
    // b's copy of `i` is then unpaired with no unpaired counterpart in a.
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let a = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let b = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0; 4]).unwrap();
    assert_unpaired_index_rejected(&a, &b, &[(i, j)]);
}
