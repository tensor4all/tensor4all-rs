//! Runtime-free `select_indices`, `stack_along_new_index`, and `scale_in` adoption.
//!
//! These operations move host data on compact storage, so their results must stay
//! compact host storage instead of entering the eager runtime for every compact
//! scalar type. The tests hold the process-global engine while the operations run
//! on another thread: a route that entered the implicit engine would block there
//! instead of computing, and value equality alone would not tell the two routes
//! apart.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, IndexLike};
use tensor4all_tensorbackend::{
    with_default_backend, CpuBackend, CpuExecutionContext, Storage, StorageScalar,
};

/// Run `run` on another thread while this thread holds the process-global engine.
///
/// The worker thread lets a wrong route fail the test instead of hanging the
/// suite: the panic releases the borrow.
fn while_the_default_engine_is_held<R: Send + 'static>(
    run: impl FnOnce() -> R + Send + 'static,
) -> R {
    with_default_backend(|_global| {
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = run();
            let _ = done_tx.send(());
            result
        });
        match done_rx.recv_timeout(Duration::from_secs(60)) {
            Ok(()) => worker.join().expect("worker thread"),
            Err(_) => panic!(
                "the operation did not finish while the process-global engine was borrowed: \
                 it entered the implicit engine route"
            ),
        }
    })
}

/// Compact (host-storage) tensor over `[i = 2, j = 3]` from column-major data.
fn compact_pair<T: StorageScalar>(i: DynIndex, j: DynIndex, data: Vec<T>) -> IdxTensor {
    let storage =
        Storage::from_dense_col_major(data, &[i.dim(), j.dim()]).expect("compact storage");
    IdxTensor::from_storage(vec![i, j], Arc::new(storage)).expect("compact tensor")
}

fn select_and_stack<T: StorageScalar + std::fmt::Debug + Send + 'static>(
    data: Vec<T>,
    i: DynIndex,
    j: DynIndex,
) -> [IdxTensor; 4] {
    let tensor = compact_pair(i.clone(), j.clone(), data);
    while_the_default_engine_is_held(move || {
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
        [along_j, along_i, stacked_last, stacked_first]
    })
}

fn exercise_select_and_stack<T: StorageScalar + std::fmt::Debug + Send + 'static>(
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
    // The movements run while the process-global engine is held elsewhere, so
    // reaching it anywhere below would block instead of returning.
    let [along_j, along_i, stacked_last, stacked_first] =
        select_and_stack(data.clone(), i.clone(), j.clone());

    // Numeric results are checked after the borrow ends.
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

    // Host-storage inputs keep producing host compact storage.
    for result in [&along_j, &along_i, &stacked_last, &stacked_first] {
        assert!(result.storage().is_ok());
    }
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

#[test]
fn scale_in_adopts_runtime_free_storage_without_the_default_engine() {
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).expect("CPU backend"),
    )));
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    // Compact host storage with no owning eager runtime: adoption reads its
    // payload directly instead of materializing through the implicit engine.
    let tensor = IdxTensor::from_dense_compact(vec![i, j], vec![1.0_f64, 2.0, 3.0, 4.0])
        .expect("runtime-free tensor");

    let scaled = while_the_default_engine_is_held(move || {
        tensor
            .scale_in(3.0, &context)
            .expect("runtime-free adoption must not reach the process-global engine")
    });
    assert_eq!(scaled.to_vec::<f64>().unwrap(), vec![3.0, 6.0, 9.0, 12.0]);
}
