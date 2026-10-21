//! Runtime-free `select_indices` and `stack_along_new_index` on compact storage.
//!
//! Both operations are host data movements on compact storage, so their results
//! must stay compact host storage instead of entering the eager runtime for
//! every compact scalar type.

use std::sync::Arc;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_tensorbackend::{Storage, StorageScalar};

/// Compact (host-storage) tensor over `[i = 2, j = 3]` from column-major data.
fn compact_pair<T: StorageScalar>(i: DynIndex, j: DynIndex, data: Vec<T>) -> IdxTensor {
    let storage =
        Storage::from_dense_col_major(data, &[i.dim(), j.dim()]).expect("compact storage");
    IdxTensor::from_storage(vec![i, j], Arc::new(storage)).expect("compact tensor")
}

fn exercise_select_and_stack<T: StorageScalar + std::fmt::Debug>(
    left: T,
    middle: T,
    right: T,
    lower: T,
    upper: T,
    other: T,
) {
    // Column-major [i, j]: j varies fastest.
    let data = vec![left, middle, right, lower, upper, other];
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = compact_pair(i.clone(), j.clone(), data.clone());

    // Fix j at 1: keep i, values are the second column.
    let along_j = tensor
        .select_indices(std::slice::from_ref(&j), &[1])
        .expect("select along j");
    // Fix i at 1: keep j, values are the second row.
    let along_i = tensor
        .select_indices(std::slice::from_ref(&i), &[1])
        .expect("select along i");

    let batch = DynIndex::new_dyn(2);
    let stacked_last = IdxTensor::stack_along_new_index(&[&tensor, &tensor], batch.clone(), -1)
        .expect("stack trailing batch axis");
    let stacked_first = IdxTensor::stack_along_new_index(&[&tensor, &tensor], batch, 0)
        .expect("stack leading axis");

    // Host-storage inputs keep producing host-storage results: none of these
    // movements enters the eager runtime.
    for result in [&along_j, &along_i, &stacked_last, &stacked_first] {
        assert!(
            result.storage().is_ok(),
            "compact select/stack must publish compact storage"
        );
    }

    assert_eq!(along_j.dims(), vec![2]);
    assert_eq!(along_j.to_vec::<T>().unwrap(), vec![data[2], data[3]]);

    assert_eq!(along_i.dims(), vec![3]);
    assert_eq!(
        along_i.to_vec::<T>().unwrap(),
        vec![data[1], data[3], data[5]]
    );

    // Trailing batch axis stacks whole column-major payloads one after another.
    assert_eq!(stacked_last.indices().len(), 3);
    assert_eq!(stacked_last.indices()[2].dim(), 2);
    let mut concatenated = data.clone();
    concatenated.extend_from_slice(&data);
    assert_eq!(stacked_last.to_vec::<T>().unwrap(), concatenated);

    // Leading batch axis interleaves the two payloads point by point.
    assert_eq!(stacked_first.indices()[0].dim(), 2);
    let mut interleaved = Vec::with_capacity(2 * data.len());
    for value in &data {
        interleaved.push(*value);
        interleaved.push(*value);
    }
    assert_eq!(stacked_first.to_vec::<T>().unwrap(), interleaved);
}

#[test]
fn compact_select_and_stack_are_runtime_free_for_f64() {
    exercise_select_and_stack(1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0);
}

#[test]
fn compact_select_and_stack_are_runtime_free_for_f32() {
    exercise_select_and_stack(1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0);
}

#[test]
fn compact_select_and_stack_are_runtime_free_for_c64() {
    use num_complex::Complex64;
    exercise_select_and_stack(
        Complex64::new(1.0, -0.5),
        Complex64::new(2.0, 0.5),
        Complex64::new(-3.0, 0.0),
        Complex64::new(0.5, 2.0),
        Complex64::new(1.5, -0.5),
        Complex64::new(4.0, 1.0),
    );
}

#[test]
fn compact_select_and_stack_are_runtime_free_for_c32() {
    use num_complex::Complex32;
    exercise_select_and_stack(
        Complex32::new(1.0, -0.5),
        Complex32::new(2.0, 0.5),
        Complex32::new(-3.0, 0.0),
        Complex32::new(0.5, 2.0),
        Complex32::new(1.5, -0.5),
        Complex32::new(4.0, 1.0),
    );
}
