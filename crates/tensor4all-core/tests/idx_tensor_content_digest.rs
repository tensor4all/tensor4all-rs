//! `IdxTensor::content_digest` fingerprints a payload without materializing it (#1086).
//!
//! Patch ordering in `tensor4all-partitionedtreetn` needs a content tiebreak that
//! a comparator can afford: `to_storage` builds a compact snapshot, and
//! `with_dense_slice`/`with_dense_host_slice` densify a diagonal or structured
//! payload. The digest must instead read the payload in place, and it must report
//! `None` rather than copy when the payload cannot be read that way.
//!
//! The allocation checks are allocator-level and therefore deterministic: they
//! time nothing, so a loaded host cannot change the verdict.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use tensor4all_core::{outer_product, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{Storage, StructuredStorage};

struct CountingAllocator;

thread_local! {
    /// (total bytes, largest single allocation) on this thread.
    static ALLOCATIONS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

// SAFETY: every method forwards the caller's allocation to `System` unchanged,
// so ownership and layout stay with the system allocator exactly as if this
// wrapper were absent; `record` only bumps a thread-local counter and never
// touches the pointer.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: `layout` is the layout the caller passed to the allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` are the pair the caller received from `alloc`
        // through this wrapper, which forwarded them to `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        // SAFETY: `ptr`/`layout` are the caller's live allocation and
        // `new_size` its requested size, both forwarded unchanged.
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

fn dense_storage_payload_bytes(dims: &[usize]) -> usize {
    dims.iter().product::<usize>() * size_of::<f64>()
}

/// A dense `Materialized` payload is fingerprinted in place.
#[test]
fn content_digest_borrows_a_dense_materialized_payload() {
    let dims = [64, 64];
    let storage = Arc::new(
        Storage::from_dense_col_major((0..4096).map(|i| i as f64).collect(), &dims).unwrap(),
    );
    let tensor = IdxTensor::from_storage(
        dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect(),
        storage,
    )
    .unwrap();

    // Warm up so lazily initialized process state is not counted.
    assert!(tensor.content_digest().is_some());

    let ((first, second), total, largest) =
        allocations_during(|| (tensor.content_digest(), tensor.content_digest()));
    assert_eq!(first, second, "the digest must be stable across calls");
    assert_eq!(
        (total, largest),
        (0, 0),
        "a {} byte payload must be hashed in place",
        dense_storage_payload_bytes(&dims)
    );
}

/// A diagonal payload is hashed as the diagonal, never densified: a densified
/// `512 x 512` payload would allocate 2 MiB and the digest must not.
#[test]
fn content_digest_does_not_densify_a_diagonal_payload() {
    let dim = 512;
    let storage = Arc::new(
        Storage::from_diag_col_major((0..dim).map(|i| i as f64 + 0.5).collect(), 2).unwrap(),
    );
    let tensor = IdxTensor::from_storage(
        vec![DynIndex::new_dyn(dim), DynIndex::new_dyn(dim)],
        storage,
    )
    .unwrap();
    assert!(tensor.is_diag());
    assert!(tensor.content_digest().is_some());

    let (digest, total, largest) = allocations_during(|| tensor.content_digest());
    assert!(digest.is_some());
    assert_eq!(
        (total, largest),
        (0, 0),
        "densifying the {}x{} diagonal payload would allocate {} bytes",
        dim,
        dim,
        dim * dim * size_of::<f64>()
    );
}

/// An eager host payload is fingerprinted without a payload-sized copy.
///
/// Reading an eager payload above rank 8 copies only the axis layout, so the
/// allowed allocation is bounded by the rank and not by the payload.
#[test]
fn content_digest_borrows_an_eager_payload() {
    for (rank, dim, payload_values) in [(2usize, 64usize, 4096usize), (12, 2, 4096)] {
        let dims = vec![dim; rank];
        let values: Vec<f64> = (0..payload_values).map(|i| i as f64 * 0.5).collect();
        let tensor = IdxTensor::from_dense(
            dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect(),
            values,
        )
        .unwrap();
        let payload_bytes = payload_values * size_of::<f64>();

        assert!(tensor.content_digest().is_some());
        let (digest, total, largest) = allocations_during(|| tensor.content_digest());
        assert!(digest.is_some());
        // Shape and stride copies for the borrowed read, not payload data.
        let layout_bound = 128 * rank;
        assert!(
            total <= layout_bound,
            "rank {rank}: the digest allocated {total} bytes, more than the \
             {layout_bound}-byte axis layout; the {payload_bytes}-byte payload must be \
             read in place"
        );
        assert!(
            largest < payload_bytes,
            "rank {rank}: the digest allocated {largest} bytes in one allocation; the \
             {payload_bytes}-byte payload must be read in place"
        );
    }
}

/// A payload that is only readable by copying reports `None` instead.
#[test]
fn content_digest_is_none_for_a_non_contiguous_payload() {
    // Column-major strides for `[2, 3]` are `[1, 2]`; `[1, 4]` needs 10 values.
    let strided = StructuredStorage::new(
        (0..10).map(|i| i as f64).collect(),
        vec![2, 3],
        vec![1, 4],
        vec![0, 1],
    )
    .unwrap();
    let tensor = IdxTensor::from_storage(
        vec![DynIndex::new_dyn(2), DynIndex::new_dyn(3)],
        Arc::new(strided.into_storage()),
    )
    .unwrap();

    let (digest, total, largest) = allocations_during(|| tensor.content_digest());
    assert_eq!(digest, None);
    assert_eq!(
        (total, largest),
        (0, 0),
        "reporting `None` must not copy or densify the payload either"
    );
}

/// Equal payloads share a digest, and different content does not: the digest is
/// over bit patterns, so `0.0` and `-0.0` differ and `f32`/`f64` differ.
#[test]
fn content_digest_distinguishes_content_and_scalar_kind() {
    let index = DynIndex::new_dyn(3);
    let a = IdxTensor::from_dense(vec![index.clone()], vec![1.0_f64, 0.0, -2.0]).unwrap();
    let b = IdxTensor::from_dense(vec![index.clone()], vec![1.0_f64, 0.0, -2.0]).unwrap();
    let other = IdxTensor::from_dense(vec![index.clone()], vec![1.0_f64, -0.0, -2.0]).unwrap();
    let narrow = IdxTensor::from_dense(vec![index.clone()], vec![1.0_f32, 0.0, -2.0]).unwrap();
    let diag = IdxTensor::from_storage(
        vec![index.clone(), DynIndex::new_dyn(3)],
        Arc::new(Storage::from_diag_col_major(vec![1.0_f64, 0.0, -2.0], 2).unwrap()),
    )
    .unwrap();

    assert_eq!(a.content_digest(), b.content_digest());
    assert_ne!(a.content_digest(), other.content_digest());

    assert_ne!(a.content_digest(), narrow.content_digest());
    assert_ne!(a.content_digest(), diag.content_digest());
    assert!(a.content_digest().is_some());
}

/// Paired sign flips must not cancel in the word fold.
///
/// An FNV-1a step (`state = (state ^ word) * prime`) is linear over GF(2), so
/// `[1.0, 2.0]` and `[-1.0, -2.0]` fold to the same state and a very common pair
/// of payloads would collide. This test pins the fix, not just the mixer.
#[test]
fn content_digest_does_not_cancel_paired_sign_flips() {
    let index = DynIndex::new_dyn(2);
    let positive = IdxTensor::from_dense(vec![index.clone()], vec![1.0_f64, 2.0]).unwrap();
    let negative = IdxTensor::from_dense(vec![index], vec![-1.0_f64, -2.0]).unwrap();
    assert_ne!(positive.content_digest(), negative.content_digest());
}

fn zero_leaf(index: &DynIndex) -> IdxTensor {
    IdxTensor::from_dense(vec![index.clone()], vec![0.0_f64, 0.0])
        .unwrap()
        .enable_grad()
        .unwrap()
}

/// At `x = 0`, `x` and `2x` have bitwise equal primals and different
/// derivatives, and the digest cannot separate them: nothing readable
/// distinguishes the two payloads before a gradient is accumulated. The graph is
/// not part of a payload's readable content, so this is the documented tie
/// rather than a digest failure.
#[test]
fn content_digest_ties_on_equal_primals_with_different_ad_graphs() {
    let index = DynIndex::new_dyn(2);
    let leaf = zero_leaf(&index);
    let scaled = outer_product(&leaf, &IdxTensor::scalar(2.0_f64).unwrap()).unwrap();

    assert_eq!(
        leaf.with_dense_slice::<f64, _>(<[f64]>::to_vec).unwrap(),
        scaled.with_dense_slice::<f64, _>(<[f64]>::to_vec).unwrap(),
        "the primals must be bitwise equal for this test to be about the graph"
    );
    assert_eq!(leaf.content_digest(), scaled.content_digest());
}

/// A tracked payload's accumulated gradient is part of what a consumer can read
/// off it, so two payloads with equal primals and different gradients must not
/// share a digest. This separates accumulation, not a forward tangent: this
/// crate has no forward-mode tangent to read.
#[test]
fn content_digest_separates_equal_primals_with_different_accumulated_gradients() {
    let index = DynIndex::new_dyn(2);
    let primal_leaf = zero_leaf(&index);
    let scaled_leaf = zero_leaf(&index);

    // Before any gradient is accumulated the two payloads are indistinguishable,
    // which is the residual tie the ordering key documents.
    assert_eq!(primal_leaf.content_digest(), scaled_leaf.content_digest());

    let two = IdxTensor::scalar(2.0_f64).unwrap();
    let scaled = outer_product(&scaled_leaf, &two).unwrap();
    scaled.sum().unwrap().backward().unwrap();
    primal_leaf.sum().unwrap().backward().unwrap();

    // Same primal, different accumulated gradient.
    let primal = |tensor: &IdxTensor| tensor.with_dense_slice::<f64, _>(<[f64]>::to_vec).unwrap();
    assert_eq!(primal(&primal_leaf), vec![0.0, 0.0]);
    assert_eq!(primal(&scaled_leaf), vec![0.0, 0.0]);
    assert_eq!(
        primal_leaf
            .grad()
            .unwrap()
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        scaled_leaf
            .grad()
            .unwrap()
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![2.0, 2.0]
    );
    assert_ne!(
        primal_leaf.content_digest(),
        scaled_leaf.content_digest(),
        "equal primals with different accumulated gradients must not share a digest"
    );
}
