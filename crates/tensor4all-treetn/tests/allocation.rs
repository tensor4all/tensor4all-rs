//! Allocation budgets of the cached evaluator's public batch paths.
//!
//! These measurements install a process-wide counting allocator, so they live
//! in their own `[[test]]` target: the crate's unit tests and the shared
//! `integration` target keep the default system allocator.
//!
//! [AI Supplied] #709 needs an objective, causal resource measurement rather
//! than a hand-placed counter at the sites a change happens to touch: the
//! counter below observes every heap block the process requests on the
//! measuring thread, so a removed short-lived vector or a removed rank-zero
//! tensor shows up whether or not the test knows where it came from. Counting
//! is thread-local and const-initialized, so it never allocates and never
//! observes another test's concurrent work.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static REQUESTED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

impl CountingAllocator {
    fn record(bytes: usize) {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get().wrapping_add(1)));
        let _ = REQUESTED_BYTES.try_with(|count| count.set(count.get().saturating_add(bytes)));
    }
}

// SAFETY: every method forwards to `System` with the same arguments; the only
// added work is a thread-local counter increment that cannot allocate or
// re-enter the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record(layout.size());
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::record(new_size);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static COUNTING_ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations() -> u64 {
    ALLOCATIONS.try_with(Cell::get).unwrap_or_default()
}

/// Runs `body` and returns its result with the number of heap blocks the
/// current thread requested while it ran.
fn measure<R>(body: impl FnOnce() -> R) -> (R, u64) {
    let before = allocations();
    let result = body();
    let after = allocations();
    (result, after.saturating_sub(before))
}

/// Counts requested payload bytes, including reallocations, on this thread.
fn measure_requested_bytes<R>(body: impl FnOnce() -> R) -> (R, usize) {
    let before = REQUESTED_BYTES.with(Cell::get);
    let result = body();
    let after = REQUESTED_BYTES.with(Cell::get);
    (result, after.saturating_sub(before))
}

/// Builds a chain of `n_sites` rank-3 cores with the requested bond
/// dimension, matching the Guard's 16-site floating-zone fixture shape.
fn chain_fixture(n_sites: usize, bond_dim: usize) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let physical = (0..n_sites)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let bonds = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let mut indices = Vec::with_capacity(3);
        if site > 0 {
            indices.push(bonds[site - 1].clone());
        }
        indices.push(physical[site].clone());
        if site + 1 < n_sites {
            indices.push(bonds[site].clone());
        }
        let len: usize = indices.iter().map(IndexLike::dim).product();
        tensors.push(
            IdxTensor::from_dense(
                indices,
                (0..len)
                    .map(|flat| 1.0 + (flat % 7) as f64)
                    .collect::<Vec<f64>>(),
            )
            .expect("fixture tensor is dense"),
        );
    }
    (
        TreeTN::from_tensors(tensors, (0..n_sites).collect()).expect("fixture chain is valid"),
        physical,
    )
}

/// Builds a TreeTN from an edge list and one physical index of dimension two
/// per node, with every bond dimension taken from `bond_dims` in edge order.
fn tree_from_edges(
    edges: &[(usize, usize)],
    bond_dims: &[usize],
) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    assert_eq!(edges.len(), bond_dims.len());
    let n_nodes = edges
        .iter()
        .flat_map(|&(left, right)| [left, right])
        .max()
        .map_or(1, |node| node + 1);
    let physical = (0..n_nodes)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let bonds = bond_dims
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(n_nodes);
    for (node, physical_index) in physical.iter().enumerate() {
        let mut tensor_indices = vec![physical_index.clone()];
        for (edge, &(left, right)) in edges.iter().enumerate() {
            if node == left || node == right {
                tensor_indices.push(bonds[edge].clone());
            }
        }
        let len: usize = tensor_indices.iter().map(IndexLike::dim).product();
        tensors.push(
            IdxTensor::from_dense(
                tensor_indices,
                (0..len)
                    .map(|flat| 0.125 + ((flat + node) % 7) as f64 * 0.25)
                    .collect::<Vec<f64>>(),
            )
            .expect("fixture tensor is dense"),
        );
    }
    (
        TreeTN::from_tensors(tensors, (0..n_nodes).collect()).expect("fixture tree is valid"),
        physical,
    )
}

/// [AI Supplied] #709 resource gate. Two claims are measured with the
/// counting allocator rather than asserted from a diff:
///
/// 1. the typed batch route allocates at least one fewer heap block per
///    result than the `AnyScalar` wrapper, because it constructs no rank-zero
///    tensor per result;
/// 2. a warm 16-site call stays far below the audited baseline of at least
///    3,072 short-lived assignment vectors per 16-site/64-point call recorded
///    in `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`.
///
/// Both evaluators are fully warm, so the measured difference is the result
/// boundary and the per-call assignment work, not cache misses.
#[test]
fn typed_batch_and_warm_assignment_allocations_stay_below_the_audited_baseline() {
    const N_SITES: usize = 16;
    const N_POINTS: usize = 64;
    const AUDITED_ASSIGNMENT_VECTORS: u64 = 3_072;

    let (tree, indices) = chain_fixture(N_SITES, 16);
    let varying = N_SITES / 2;
    let mut values = vec![0usize; N_SITES * N_POINTS];
    for point in 0..N_POINTS {
        values[varying + N_SITES * point] = point % 2;
    }
    let points = ColMajorArrayRef::new(&values, &[N_SITES, N_POINTS]).expect("shape matches");
    let mut two_point_values = vec![0usize; N_SITES * 2];
    two_point_values[varying + N_SITES] = 1;
    let two_points =
        ColMajorArrayRef::new(&two_point_values, &[N_SITES, 2]).expect("shape matches");

    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(varying),
            ..Default::default()
        },
    )
    .expect("evaluator accepts the fixture");
    let hint = EvaluationHint::around(varying);

    // Warm every message the fixture needs before measuring.
    evaluator
        .evaluate_batched_with_hint(points, hint.clone())
        .expect("warm-up call succeeds");
    evaluator
        .evaluate_batched_with_hint(two_points, hint.clone())
        .expect("warm-up two-point call succeeds");

    let (wrapped, wrapper_allocations) = measure(|| {
        evaluator
            .evaluate_batched_with_hint(points, hint.clone())
            .expect("warm wrapper call succeeds")
    });
    let (typed, typed_allocations) = measure(|| {
        evaluator
            .evaluate_batched_typed::<f64>(points, hint.clone())
            .expect("warm typed call succeeds")
    });
    let (_, two_point_allocations) = measure(|| {
        evaluator
            .evaluate_batched_typed::<f64>(two_points, hint.clone())
            .expect("warm two-point typed call succeeds")
    });

    eprintln!(
        "#709 warm allocation counts: sites={N_SITES} points={N_POINTS} wrapper={wrapper_allocations} typed={typed_allocations} two_point_typed={two_point_allocations}"
    );

    assert_eq!(typed.len(), wrapped.len());
    for (typed, wrapped) in typed.iter().zip(&wrapped) {
        assert_eq!(*typed, wrapped.real());
        assert_eq!(wrapped.imag(), 0.0);
    }

    assert!(
        typed_allocations + N_POINTS as u64 <= wrapper_allocations,
        "the typed route must save at least one allocation per result: typed={typed_allocations} wrapper={wrapper_allocations}"
    );
    assert!(
        typed_allocations < AUDITED_ASSIGNMENT_VECTORS,
        "a warm {N_SITES}-site {N_POINTS}-point typed call must stay far below the audited {AUDITED_ASSIGNMENT_VECTORS} short-lived assignment vectors, got {typed_allocations}"
    );
    assert!(
        two_point_allocations < 256,
        "a warm two-point Guard-shaped call must stay small, got {two_point_allocations}"
    );
}

/// [AI Supplied] #671: already retained columns may take arithmetic
/// proportional to the bond, but must not be copied into transient heaps.
/// The allocator observes requested bytes on the public batch path, so the
/// number is independent of any internal counter.
#[test]
fn warm_edge_cut_requested_bytes_are_independent_of_bond_dimension() {
    fn requested_bytes(bond: usize) -> usize {
        let (tree, indices) = tree_from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], &[bond; 4]);
        let values = (0..64)
            .flat_map(|point| (0..5).map(move |site| (point >> site) & 1))
            .collect::<Vec<_>>();
        let points = ColMajorArrayRef::new(&values, &[5, 64]).expect("shape matches");
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default())
                .expect("evaluator accepts the fixture");
        let cold = evaluator
            .evaluate_batched_typed::<f64>(points, EvaluationHint::around(2))
            .expect("cold typed call succeeds");
        let (warm, bytes) = measure_requested_bytes(|| {
            evaluator
                .evaluate_batched_typed::<f64>(points, EvaluationHint::around(2))
                .expect("warm typed call succeeds")
        });
        assert_eq!(warm, cold);
        bytes
    }
    assert_eq!(
        requested_bytes(2),
        requested_bytes(32),
        "warm queries must borrow packed columns instead of copying bond-sized payloads"
    );
}
