//! CUDA parity of the sketched Hadamard product (issue #1158 step 1).
//!
//! The sketch keeps its blocks in host buffers and only dispatches the
//! contractions through the caller's context, so a CUDA run must reproduce the
//! true pointwise product and the CPU run. These tests need visible CUDA
//! ordinal 0, so they stay `#[ignore]`d like the other hardware tests.

#![cfg(feature = "tenferro-cuda")]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{
    CudaExecutionContext, DynIndex, ExecutionContext, IdxTensor, IdxTensorError, Scalar,
};
use tensor4all_tensorbackend::{
    CpuBackend, CpuExecutionContext, CpuExecutionContextError, ExecutionContextEntryError,
};
use tensor4all_treersi::{hadamard_many_with_rng_in, TreeRsiError, TreeRsiOptions, TreeRsiScalar};
use tensor4all_treetn::TreeTN;

use crate::support;
use support::random_tree;

type Tree = TreeTN<IdxTensor, usize>;

const EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)];
const NUM_SITES: usize = 7;
const BOND_DIM: usize = 2;
const SEED: u64 = 100;
/// Probe width below the complement assignment count of the first nodes (32 and
/// 16), so at least one edge is sketched; an all-exact run would never reach the
/// diag-probe contraction.
const SKETCH_DIM: usize = 9;

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )))
}

fn cuda_context() -> (Arc<CudaExecutionContext>, ExecutionContext) {
    let cuda = Arc::new(CudaExecutionContext::new().expect("CUDA ordinal 0 must be available"));
    (Arc::clone(&cuda), ExecutionContext::Cuda(cuda))
}

/// Rebuild a host tree inside `context`, preserving index identities so the
/// copy stays connected.
fn scoped_copy<T: TreeRsiScalar>(tree: &Tree, context: &ExecutionContext) -> Tree {
    let names = tree.node_names();
    let tensors = names
        .iter()
        .map(|name| {
            let tensor = tree.tensor(tree.node_index(name).unwrap()).unwrap();
            let indices = tensor.indices().to_vec();
            let values = tensor.to_vec::<T>().unwrap();
            IdxTensor::from_dense_in(context, indices, values).unwrap()
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, names).unwrap()
}

/// Whole-dense values in site order.
fn site_ordered<T: TreeRsiScalar>(tree: &Tree, sites: &[DynIndex]) -> Vec<T> {
    tree.to_dense()
        .unwrap()
        .permute_indices(sites)
        .unwrap()
        .to_vec::<T>()
        .unwrap()
}

/// Largest absolute value, refusing a non-finite entry (`f64::max` drops NaN).
fn max_abs<T: Scalar>(values: &[T]) -> f64 {
    let mut worst = 0.0_f64;
    for (column, value) in values.iter().enumerate() {
        let magnitude = value.abs_val();
        assert!(
            magnitude.is_finite(),
            "non-finite value at column {column}: magnitude {magnitude}"
        );
        worst = worst.max(magnitude);
    }
    worst
}

/// Largest absolute componentwise difference.
///
/// `f64::max` drops a NaN operand and `zip` stops at the shorter side, so a
/// non-finite or missing entry would otherwise read as a zero residual.
fn max_diff<T: Scalar>(left: &[T], right: &[T]) -> f64 {
    assert_eq!(
        left.len(),
        right.len(),
        "compared vectors must have equal length: {} and {}",
        left.len(),
        right.len()
    );
    let mut worst = 0.0_f64;
    for (column, (a, b)) in left.iter().zip(right).enumerate() {
        let error = (*a - *b).abs_val();
        assert!(
            error.is_finite(),
            "non-finite error at column {column}: magnitudes {} and {}",
            a.abs_val(),
            b.abs_val()
        );
        worst = worst.max(error);
    }
    worst
}

fn options<T: TreeRsiScalar>() -> TreeRsiOptions<usize> {
    TreeRsiOptions {
        max_bond_dim: Some(16),
        sketch_dim: Some(SKETCH_DIM),
        root: Some(NUM_SITES - 1),
        rel_tol: T::epsilon() * 10.0,
        ..Default::default()
    }
}

fn assert_cuda_matches_cpu<T: TreeRsiScalar>(
    sample: impl Fn(f64, f64) -> T + Copy,
    tolerance: f64,
) {
    let (cuda, cuda_context_value) = cuda_context();
    let cpu = cpu_context();
    let sites = (0..NUM_SITES)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let edges = EDGES.to_vec();
    let inputs = (0..2)
        .map(|a| random_tree(&sites, &edges, BOND_DIM, SEED + a, sample))
        .collect::<Vec<_>>();

    let cpu_inputs = inputs
        .iter()
        .map(|input| scoped_copy::<T>(input, &cpu))
        .collect::<Vec<_>>();
    let cuda_inputs = inputs
        .iter()
        .map(|input| scoped_copy::<T>(input, &cuda_context_value))
        .collect::<Vec<_>>();

    let mut cpu_rng = ChaCha8Rng::seed_from_u64(SEED);
    let reference =
        hadamard_many_with_rng_in::<T, _, _>(&cpu_inputs, &options::<T>(), &mut cpu_rng, &cpu)
            .expect("CPU reference must succeed");

    let mut cuda_rng = ChaCha8Rng::seed_from_u64(SEED);
    let result = hadamard_many_with_rng_in::<T, _, _>(
        &cuda_inputs,
        &options::<T>(),
        &mut cuda_rng,
        &cuda_context_value,
    )
    .expect("CUDA run must succeed");

    // At least one sketched edge, or the diag-probe contraction is never reached.
    assert!(
        result
            .diagnostics
            .edges
            .iter()
            .any(|edge| !edge.exact_columns),
        "fixture must include a sketched edge"
    );

    // Residency on the run's own scalar type: a host-readable core would mean
    // the result silently left the caller's context.
    for node in result.tree.node_indices() {
        let tensor = result.tree.tensor(node).unwrap();
        tensor
            .validate_context(&cuda_context_value)
            .expect("output core must belong to the CUDA context");
        assert!(
            tensor.to_vec::<T>().is_err(),
            "output core is host-readable; residency lost"
        );
    }

    let got = site_ordered::<T>(
        &result.tree.download(&cuda).expect("download must succeed"),
        &sites,
    );
    let want = site_ordered::<T>(&reference.tree, &sites);
    // Independent reference: the pointwise product of the host inputs.
    let mut expected = vec![T::one(); 1 << NUM_SITES];
    for input in &inputs {
        let values = input
            .to_dense()
            .unwrap()
            .permute_indices(&sites)
            .unwrap()
            .to_vec::<T>()
            .unwrap();
        for (value, factor) in expected.iter_mut().zip(values) {
            *value = *value * factor;
        }
    }

    assert!(
        max_abs(&expected) > 0.0,
        "product must not be identically zero"
    );
    // Absolute maximum residual, the criterion and numbers of
    // `tensor4all-treetn/tests/cuda_src_contraction.rs`.
    let cpu_error = max_diff(&want, &expected);
    assert!(
        cpu_error < tolerance,
        "CPU product residual {cpu_error} exceeds {tolerance}"
    );
    let oracle_error = max_diff(&got, &expected);
    assert!(
        oracle_error < tolerance,
        "CUDA product residual {oracle_error} exceeds {tolerance}"
    );
    let parity = max_diff(&got, &want);
    assert!(
        parity < tolerance,
        "CUDA/CPU parity residual {parity} exceeds {tolerance}"
    );
}

/// A CUDA run reads its input cores back through the process-global CPU eager
/// context, so it cannot nest a session of that exact context: the readback is
/// rejected (`recursive CPU execution-context session entry on one thread`)
/// instead of silently reusing the caller's session. A session of any other CPU
/// context is unrelated and does nest — measured, that variant of this test
/// succeeds. This needs a CUDA context, so it cannot run on a GPU-less host.
#[test]
#[ignore = "requires visible CUDA ordinal 0"]
fn cuda_product_inside_a_cpu_session_is_rejected() {
    let (_, cuda_context_value) = cuda_context();
    // The readback rebuilds each core in the process-global CPU context, so the
    // nesting under test is a session of that exact context.
    let cpu = tensor4all_tensorbackend::default_cpu_execution_context();
    let sites = (0..NUM_SITES)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let edges = EDGES.to_vec();
    let inputs = (0..2)
        .map(|a| random_tree(&sites, &edges, BOND_DIM, SEED + a, |x, _| x))
        .collect::<Vec<_>>();
    let cuda_inputs = inputs
        .iter()
        .map(|input| scoped_copy::<f64>(input, &cuda_context_value))
        .collect::<Vec<_>>();

    let nested = cpu.with_concrete_session(|_session| {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        hadamard_many_with_rng_in::<f64, _, _>(
            &cuda_inputs,
            &options::<f64>(),
            &mut rng,
            &cuda_context_value,
        )
    });
    // The outer entry itself succeeds; the product call is what fails.
    let error = nested
        .expect("entering one CPU session succeeds")
        .expect_err("a CUDA run cannot nest inside a CPU session");
    // The dtype matches, so the rejection has to stay a host-core failure, and
    // the chain itself has to identify the re-entry rather than its message.
    let TreeRsiError::Numerical { source } = error else {
        panic!("a rejected session entry must not be reported as a dtype mismatch: {error}");
    };
    assert!(
        matches!(
            source.downcast_ref::<IdxTensorError>(),
            Some(IdxTensorError::EngineEntry {
                source: ExecutionContextEntryError::Cpu(CpuExecutionContextError::Reentered)
            })
        ),
        "the host core must be rejected by the CPU session re-entry: {source}"
    );
}

#[test]
#[ignore = "requires visible CUDA ordinal 0"]
fn cuda_hadamard_product_matches_cpu_f64() {
    assert_cuda_matches_cpu(|x, _| x, 1e-9);
}

#[test]
#[ignore = "requires visible CUDA ordinal 0"]
fn cuda_hadamard_product_matches_cpu_f32() {
    // The f64 case's criterion, bounded by this scalar's own precision: f32
    // epsilon is 1.19e-7 and the measured residual is 5.59e-9.
    assert_cuda_matches_cpu(|x, _| x as f32, 1e-7);
}

#[test]
#[ignore = "requires visible CUDA ordinal 0"]
fn cuda_hadamard_product_matches_cpu_c64() {
    assert_cuda_matches_cpu(Complex64::new, 1e-8);
}

#[test]
#[ignore = "requires visible CUDA ordinal 0"]
fn cuda_hadamard_product_matches_cpu_c32() {
    // A Complex32 component carries f32 precision and the measured residual is
    // 5.99e-7, so the bound sits an order of magnitude above it.
    assert_cuda_matches_cpu(|x, y| Complex32::new(x as f32, y as f32), 1e-5);
}
