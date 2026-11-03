//! `mat_mul` must dispatch borrowed operands without staging an owned copy.
//!
//! Before #1096 `BlasMul::blas_mat_mul` rebuilt both operands with
//! `to_typed_tensor()`, which clones each buffer: for the shapes tracked by the
//! issue the right operand alone was a 636 KB copy. Both matrices now enter the
//! engine as borrowed `TensorRead::View`s, so a multiplication allocates its
//! `m x n` output and engine-internal scratch, but no buffer the size of either
//! operand.
//!
//! The check is allocator-level and therefore deterministic: it times nothing,
//! so a loaded host cannot change the verdict.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use num_complex::Complex64;
use tensor4all_tensorbackend::{mat_mul, BlasMul, Matrix};

struct CountingAllocator;

thread_local! {
    /// (total bytes, largest single allocation) on this thread.
    static ALLOCATIONS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn record(bytes: usize) {
    let _ = ALLOCATIONS.try_with(|cell| {
        let (total, largest) = cell.get();
        cell.set((total + bytes, largest.max(bytes)));
    });
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Largest single allocation seen during `f`, and the total allocated bytes.
fn allocations_during<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    ALLOCATIONS.with(|cell| cell.set((0, 0)));
    let out = f();
    let (total, largest) = ALLOCATIONS.with(Cell::get);
    (out, total, largest)
}

/// Multiply `m x k` by `k x n` and assert no allocation reached either operand.
fn assert_operands_are_read_in_place<T>(m: usize, k: usize, n: usize)
where
    T: BlasMul + Copy + std::fmt::Debug + Default,
{
    let a = Matrix::<T>::from_col_major_vec(m, k, vec![T::default(); m * k]);
    let b = Matrix::<T>::from_col_major_vec(k, n, vec![T::default(); k * n]);
    let operand_bytes =
        size_of_val(a.as_col_major_slice()).max(size_of_val(b.as_col_major_slice()));
    assert_eq!(operand_bytes, m.max(n) * k * size_of::<T>());

    // Warm up on the same shape so engine buffers, plans and pools are already
    // allocated; only the per-call work is counted afterwards.
    let warm = mat_mul(&a, &b).unwrap();
    assert_eq!((warm.nrows(), warm.ncols()), (m, n));

    let (c, total, largest) = allocations_during(|| mat_mul(&a, &b).unwrap());
    assert_eq!((c.nrows(), c.ncols()), (m, n));

    assert!(
        largest < operand_bytes,
        "{m}x{k} * {k}x{n}: mat_mul allocated {largest} bytes in one allocation; the \
         {operand_bytes}-byte operand must be read in place"
    );
    assert!(
        total < operand_bytes,
        "{m}x{k} * {k}x{n}: mat_mul allocated {total} bytes in total; the {operand_bytes}-byte \
         operand must be read in place"
    );
}

const K: usize = 282;
const N: usize = 141;

#[test]
fn mat_mul_f64_reads_the_right_operand_in_place() {
    assert_operands_are_read_in_place::<f64>(1, K, N);
}

#[test]
fn mat_mul_c64_reads_the_right_operand_in_place() {
    assert_operands_are_read_in_place::<Complex64>(1, K, N);
}

#[test]
fn mat_mul_c64_reads_a_large_left_operand_in_place() {
    assert_operands_are_read_in_place::<Complex64>(N, K, 1);
}
