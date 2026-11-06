use super::*;

// -- 1D creation + get --------------------------------------------------

#[test]
fn test_1d_creation_and_get() {
    let arr = ColMajorArray::new(vec![10, 20, 30], vec![3]).unwrap();
    assert_eq!(arr.ndim(), 1);
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.len(), 3);
    assert!(!arr.is_empty());

    assert_eq!(arr.get(&[0]), Some(&10));
    assert_eq!(arr.get(&[1]), Some(&20));
    assert_eq!(arr.get(&[2]), Some(&30));
}

// -- 2D creation + get --------------------------------------------------

#[test]
fn test_2d_creation_and_get() {
    // 2x3 matrix in column-major:
    // Column 0: [1, 2], Column 1: [3, 4], Column 2: [5, 6]
    // Flat: [1, 2, 3, 4, 5, 6]
    let arr = ColMajorArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    assert_eq!(arr.ndim(), 2);
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr.len(), 6);

    // (row, col)
    assert_eq!(arr.get(&[0, 0]), Some(&1));
    assert_eq!(arr.get(&[1, 0]), Some(&2));
    assert_eq!(arr.get(&[0, 1]), Some(&3));
    assert_eq!(arr.get(&[1, 1]), Some(&4));
    assert_eq!(arr.get(&[0, 2]), Some(&5));
    assert_eq!(arr.get(&[1, 2]), Some(&6));
}

// -- 3D creation + get --------------------------------------------------

#[test]
fn test_3d_creation_and_get() {
    // Shape [2, 3, 2]: total 12 elements
    let data: Vec<i32> = (0..12).collect();
    let arr = ColMajorArray::new(data.clone(), vec![2, 3, 2]).unwrap();
    assert_eq!(arr.ndim(), 3);
    assert_eq!(arr.len(), 12);

    // Verify column-major offset: i0 + 2*(i1 + 3*i2)
    for i2 in 0..2 {
        for i1 in 0..3 {
            for i0 in 0..2 {
                let expected_offset = i0 + 2 * (i1 + 3 * i2);
                assert_eq!(
                    arr.get(&[i0, i1, i2]),
                    Some(&(expected_offset as i32)),
                    "Mismatch at [{i0}, {i1}, {i2}]"
                );
            }
        }
    }
}

// -- Column-major order verification (2D) -------------------------------

#[test]
fn test_column_major_order_2d() {
    let nrows = 3;
    let ncols = 4;
    let data: Vec<i32> = (0..(nrows * ncols) as i32).collect();
    let arr = ColMajorArray::new(data.clone(), vec![nrows, ncols]).unwrap();

    // In column-major, data[i + nrows * j] == arr[(i, j)]
    for j in 0..ncols {
        for i in 0..nrows {
            assert_eq!(arr.get(&[i, j]), Some(&data[i + nrows * j]));
        }
    }
}

// -- get_mut ------------------------------------------------------------

#[test]
fn test_get_mut() {
    let mut arr = ColMajorArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    if let Some(v) = arr.get_mut(&[1, 0]) {
        *v = 42;
    }
    assert_eq!(arr.get(&[1, 0]), Some(&42));
    // Other elements unchanged
    assert_eq!(arr.get(&[0, 0]), Some(&1));
    assert_eq!(arr.get(&[0, 1]), Some(&3));
    assert_eq!(arr.get(&[1, 1]), Some(&4));
}

// -- push_column --------------------------------------------------------

#[test]
fn test_push_column() {
    let mut arr = ColMajorArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    assert_eq!(arr.ncols(), Some(2));

    arr.push_column(&[5, 6]).unwrap();
    assert_eq!(arr.ncols(), Some(3));
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr.len(), 6);
    assert_eq!(arr.get(&[0, 2]), Some(&5));
    assert_eq!(arr.get(&[1, 2]), Some(&6));
}

#[test]
fn test_push_column_wrong_length() {
    let mut arr = ColMajorArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let err = arr.push_column(&[5, 6, 7]).unwrap_err();
    assert_eq!(
        err,
        ColMajorArrayError::ColumnLengthMismatch {
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn test_push_column_not_2d() {
    let mut arr = ColMajorArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let err = arr.push_column(&[4]).unwrap_err();
    assert_eq!(err, ColMajorArrayError::Not2D { ndim: 1 });
}

// -- column() slice access ----------------------------------------------

#[test]
fn test_column_access() {
    let arr = ColMajorArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    assert_eq!(arr.column(0), Some([1, 2].as_slice()));
    assert_eq!(arr.column(1), Some([3, 4].as_slice()));
    assert_eq!(arr.column(2), Some([5, 6].as_slice()));
    assert_eq!(arr.column(3), None); // out of bounds
}

// -- zeros, filled ------------------------------------------------------

#[test]
fn test_zeros() {
    let arr: ColMajorArray<f64> = ColMajorArray::zeros(vec![3, 2]).unwrap();
    assert_eq!(arr.len(), 6);
    assert!(arr.data().iter().all(|&v| v == 0.0));
}

#[test]
fn test_filled() {
    let arr = ColMajorArray::filled(vec![2, 3], 7i32).unwrap();
    assert_eq!(arr.len(), 6);
    assert!(arr.data().iter().all(|&v| v == 7));
}

// -- Shape mismatch error -----------------------------------------------

#[test]
fn test_shape_mismatch() {
    let result = ColMajorArray::new(vec![1, 2, 3], vec![2, 2]);
    assert_eq!(
        result.unwrap_err(),
        ColMajorArrayError::ShapeMismatch {
            shape: vec![2, 2],
            expected: 4,
            actual: 3,
        }
    );
}

// -- Out-of-bounds -> None ----------------------------------------------

#[test]
fn test_out_of_bounds() {
    let arr = ColMajorArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    // Index out of range
    assert_eq!(arr.get(&[2, 0]), None);
    assert_eq!(arr.get(&[0, 2]), None);
    // Wrong number of indices
    assert_eq!(arr.get(&[0]), None);
    assert_eq!(arr.get(&[0, 0, 0]), None);
}

// -- Ref and Mut views --------------------------------------------------

#[test]
fn test_as_ref() {
    let arr = ColMajorArray::new(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    let view = arr.as_ref();
    assert_eq!(view.ndim(), 2);
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.get(&[1, 1]), Some(&40));
    assert_eq!(view.data(), arr.data());
}

#[test]
fn test_as_mut() {
    let mut arr = ColMajorArray::new(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    {
        let mut view = arr.as_mut();
        if let Some(v) = view.get_mut(&[0, 1]) {
            *v = 99;
        }
    }
    assert_eq!(arr.get(&[0, 1]), Some(&99));
}

// -- into_data ----------------------------------------------------------

#[test]
fn test_into_data() {
    let arr = ColMajorArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let data = arr.into_data();
    assert_eq!(data, vec![1, 2, 3]);
}

// -- Empty arrays -------------------------------------------------------

#[test]
fn test_empty_array() {
    let arr: ColMajorArray<i32> = ColMajorArray::new(vec![], vec![0]).unwrap();
    assert!(arr.is_empty());
    assert_eq!(arr.len(), 0);
    assert_eq!(arr.ndim(), 1);
    assert_eq!(arr.nrows(), None);
    assert_eq!(arr.ncols(), None);
}

#[test]
fn test_empty_2d_array() {
    let arr: ColMajorArray<i32> = ColMajorArray::new(vec![], vec![3, 0]).unwrap();
    assert!(arr.is_empty());
    assert_eq!(arr.len(), 0);
    assert_eq!(arr.nrows(), Some(3));
    assert_eq!(arr.ncols(), Some(0));
}

// -- ColMajorArrayRef construction --------------------------------------

#[test]
fn test_ref_new() {
    let data = [1, 2, 3, 4, 5, 6];
    let shape = [2, 3];
    let view = ColMajorArrayRef::new(&data, &shape).unwrap();
    assert_eq!(view.ndim(), 2);
    assert_eq!(view.len(), 6);
    assert_eq!(view.get(&[1, 2]), Some(&6));
}

// -- ColMajorArrayMut construction --------------------------------------

#[test]
fn test_mut_new() {
    let mut data = [1, 2, 3, 4, 5, 6];
    let shape = [2, 3];
    let mut view = ColMajorArrayMut::new(&mut data, &shape).unwrap();
    assert_eq!(view.ndim(), 2);
    assert_eq!(view.len(), 6);
    *view.get_mut(&[0, 0]).unwrap() = 100;
    assert_eq!(view.get(&[0, 0]), Some(&100));
}

// -- Overflow detection ---------------------------------------------------

#[test]
fn test_new_rejects_overflow_shape() {
    let result = ColMajorArray::<u8>::new(vec![], vec![usize::MAX, 2]);
    assert!(
        matches!(result, Err(ColMajorArrayError::ShapeOverflow { .. })),
        "expected ShapeOverflow, got {:?}",
        result
    );
}

#[test]
fn test_filled_rejects_overflow_shape() {
    let result = ColMajorArray::filled(vec![usize::MAX, 2], 0u8);
    assert!(
        matches!(result, Err(ColMajorArrayError::ShapeOverflow { .. })),
        "expected ShapeOverflow, got {:?}",
        result
    );
}

#[test]
fn test_zeros_rejects_overflow_shape() {
    let result = ColMajorArray::<u8>::zeros(vec![usize::MAX, 2]);
    assert!(
        matches!(result, Err(ColMajorArrayError::ShapeOverflow { .. })),
        "expected ShapeOverflow, got {:?}",
        result
    );
}
