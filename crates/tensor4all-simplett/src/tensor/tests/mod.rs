use super::*;

use crate::einsum_helper::{tensor_to_col_major_vec, typed_tensor_from_col_major_slice};

#[test]
fn test_tensor2_from_elem() {
    let t: Tensor2<f64> = Tensor2::from_elem([3, 4], 0.0).unwrap();
    assert_eq!(t.len(), 12);
    assert_eq!(t.dim(0), 3);
    assert_eq!(t.dim(1), 4);
    assert_eq!(t[[0, 0]], 0.0);
}

#[test]
fn fallible_constructors_reject_invalid_capacity_before_evaluating_elements() {
    let mut called = false;
    let overflow_fn = Tensor2::<f64>::from_fn([usize::MAX, 2], |_| {
        called = true;
        0.0
    });
    let overflow_elem = Tensor2::from_elem([usize::MAX, 2], 0.0_f64);
    let capacity_fn = Tensor2::<f64>::from_fn([usize::MAX, 1], |_| {
        called = true;
        0.0
    });
    let capacity_elem = Tensor2::from_elem([usize::MAX, 1], 0.0_f64);

    assert!(overflow_fn.is_err());
    assert!(overflow_elem.is_err());
    assert!(capacity_fn.is_err());
    assert!(capacity_elem.is_err());
    assert!(!called);
}

#[test]
fn test_tensor2_indexing() {
    let mut t: Tensor2<f64> = Tensor2::from_elem([2, 3], 0.0).unwrap();
    t[[0, 0]] = 1.0;
    t[[0, 1]] = 2.0;
    t[[0, 2]] = 3.0;
    t[[1, 0]] = 4.0;
    t[[1, 1]] = 5.0;
    t[[1, 2]] = 6.0;
    assert_eq!(t[[0, 0]], 1.0);
    assert_eq!(t[[1, 2]], 6.0);
    assert_eq!(t.to_col_major_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_tensor3_from_fn() {
    let t: Tensor3<f64> =
        Tensor3::from_fn([2, 3, 4], |[i, j, k]| (i * 100 + j * 10 + k) as f64).unwrap();
    assert_eq!(t[[0, 0, 0]], 0.0);
    assert_eq!(t[[1, 2, 3]], 123.0);
    assert_eq!(t[[0, 1, 2]], 12.0);
    assert_eq!(t.dim(0), 2);
    assert_eq!(t.dim(1), 3);
    assert_eq!(t.dim(2), 4);
    assert_eq!(t.len(), 24);
}

#[test]
fn test_tensor4_from_fn() {
    let t: Tensor4<f64> = Tensor4::from_fn([2, 3, 4, 5], |[i, j, k, l]| {
        (i * 1000 + j * 100 + k * 10 + l) as f64
    })
    .unwrap();
    assert_eq!(t[[1, 2, 3, 4]], 1234.0);
    assert_eq!(t.dim(0), 2);
    assert_eq!(t.dim(1), 3);
    assert_eq!(t.dim(2), 4);
    assert_eq!(t.dim(3), 5);
    assert_eq!(t.len(), 120);
}

#[test]
fn test_iter() {
    let t: Tensor2<f64> = Tensor2::from_fn([2, 3], |[i, j]| (i * 3 + j) as f64).unwrap();
    let collected: Vec<f64> = t.iter().copied().collect();
    assert_eq!(collected, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn test_dims() {
    let t: Tensor3<f64> = Tensor3::from_elem([2, 3, 4], 1.0).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4]);
}

#[test]
fn test_from_tenferro_roundtrip() {
    let inner =
        typed_tensor_from_col_major_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]).unwrap();
    let tensor = Tensor2::from_tenferro(inner).unwrap();

    assert_eq!(tensor.dims(), &[2, 3]);
    assert_eq!(
        tensor.to_col_major_vec(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    let inner_again = tensor.clone().into_inner();
    assert_eq!(
        tensor_to_col_major_vec(&inner_again).unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn test_from_tenferro_rank_mismatch_errors() {
    let inner = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let err = Tensor3::from_tenferro(inner).unwrap_err();

    assert!(err.to_string().contains("expected rank 3"));
    assert!(err.to_string().contains("got 2"));
}

#[test]
fn test_try_from_tenferro_rank_mismatch_errors() {
    let inner = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let err = Tensor3::try_from_tenferro(inner).unwrap_err();

    assert!(err.to_string().contains("expected rank 3"));
    assert!(err.to_string().contains("got 2"));
}

#[test]
fn test_clone_allows_index_mut() {
    let original: Tensor2<f64> = Tensor2::from_fn([2, 2], |[i, j]| (i * 2 + j) as f64).unwrap();
    let mut cloned = original.clone();
    cloned[[1, 0]] = 99.0;

    assert_eq!(original[[1, 0]], 2.0);
    assert_eq!(cloned[[1, 0]], 99.0);
}

#[test]
fn test_partial_eq_compares_dims_then_values() {
    let column_major: Tensor2<f64> = Tensor2::from_fn([2, 3], |[i, j]| (i + 2 * j) as f64).unwrap();
    let same: Tensor2<f64> = Tensor2::from_fn([2, 3], |[i, j]| (i + 2 * j) as f64).unwrap();
    let other_values: Tensor2<f64> = Tensor2::from_fn([2, 3], |[i, j]| (i * 3 + j) as f64).unwrap();
    // Same element count and identical column-major flat values, different
    // shape: equality must compare dims, not only values.
    let other_shape: Tensor2<f64> = Tensor2::from_fn([3, 2], |[i, j]| (i + 3 * j) as f64).unwrap();

    assert_eq!(column_major, same);
    assert_ne!(column_major, other_values);
    assert_ne!(column_major, other_shape);
}

#[test]
fn test_is_empty_tracks_zero_element_shapes() {
    let empty: Tensor2<f64> = Tensor2::from_elem([0, 3], 1.0).unwrap();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
    assert_eq!(empty.to_col_major_vec(), Vec::<f64>::new());
    assert_eq!(empty.iter().count(), 0);

    let single: Tensor2<f64> = Tensor2::from_elem([1, 1], 7.0).unwrap();
    assert_eq!(single.len(), 1);
    assert!(!single.is_empty());
}

#[test]
fn test_iter_mut_writes_every_element_once_in_column_major_order() {
    let mut tensor: Tensor2<f64> = Tensor2::from_elem([2, 3], 0.0).unwrap();

    {
        let iter = tensor.iter_mut();
        assert_eq!(iter.size_hint(), (6, Some(6)));
        assert_eq!(iter.len(), 6);
        for (linear, value) in iter.enumerate() {
            assert_eq!(*value, 0.0);
            *value = (linear + 1) as f64;
        }
    }

    assert_eq!(
        tensor.to_col_major_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    // [i, j] = 1 + i + 2 * j in column-major order.
    assert_eq!(tensor[[0, 0]], 1.0);
    assert_eq!(tensor[[1, 2]], 6.0);

    let mut empty: Tensor3<f64> = Tensor3::from_elem([0, 2, 2], 0.0).unwrap();
    let mut iter = empty.iter_mut();
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert!(iter.next().is_none());
}
