//! Cached evaluator results on an explicit session match the global route.
//!
//! The session route must use the caller's session for every numerical leaf,
//! including the grouped GEMM kernels, and publish compact host tensors. These
//! tests compare the two routes for all four floating dtypes and check that a
//! session evaluation completes while another thread holds the process-global
//! backend.

use std::sync::mpsc;
use std::time::Duration;

use num_complex::{Complex32, Complex64};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike, TensorElement};
use tensor4all_tensorbackend::{
    explicit::Session, with_default_backend, CpuBackend, CpuExecutionContext,
};
use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};

/// One three-node chain with bond dimension 16 and site dimensions `[2, 8, 2]`.
///
/// Node 1's raw message contraction has `parent_dim * child_dim * points =
/// 16 * 16 * 32 = 8192`, above `CHAIN_BLAS_WORK_THRESHOLD` (4096), so the
/// grouped GEMM kernel is exercised rather than the scalar fallback.
fn chain_tree<T: TensorElement>(
    value: impl Fn(usize) -> T + Copy,
) -> (Vec<DynIndex>, TreeTN<IdxTensor, usize>) {
    let sites: Vec<DynIndex> = [2usize, 8, 2]
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect();
    let bond0 = DynIndex::new_dyn(16);
    let bond1 = DynIndex::new_dyn(16);

    let mut offset = 0usize;
    let mut take = |count: usize| {
        let values = (0..count)
            .map(|local| value(offset + local))
            .collect::<Vec<_>>();
        offset += count;
        values
    };

    let node0 = IdxTensor::from_dense(
        vec![sites[0].clone(), bond0.clone()],
        take(sites[0].dim() * bond0.dim()),
    )
    .expect("node 0");
    let node1 = IdxTensor::from_dense(
        vec![bond0.clone(), sites[1].clone(), bond1.clone()],
        take(bond0.dim() * sites[1].dim() * bond1.dim()),
    )
    .expect("node 1");
    let node2 = IdxTensor::from_dense(
        vec![bond1, sites[2].clone()],
        take(bond1_dim() * sites[2].dim()),
    )
    .expect("node 2");
    let tree =
        TreeTN::from_tensors(vec![node0, node1, node2], vec![0usize, 1, 2]).expect("chain tree");
    (sites, tree)
}

fn bond1_dim() -> usize {
    16
}

fn all_points(sites: &[DynIndex]) -> (Vec<usize>, Vec<usize>) {
    let dims: Vec<usize> = sites.iter().map(IndexLike::dim).collect();
    let count: usize = dims.iter().product();
    let mut data = Vec::with_capacity(count * sites.len());
    for point in 0..count {
        let mut remaining = point;
        for &dim in &dims {
            data.push(remaining % dim);
            remaining /= dim;
        }
    }
    (data, vec![sites.len(), count])
}

trait RouteDiff: Sized {
    fn max_abs_diff(left: &[Self], right: &[Self]) -> f64;
}

impl RouteDiff for f64 {
    fn max_abs_diff(left: &[Self], right: &[Self]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }
}

impl RouteDiff for f32 {
    fn max_abs_diff(left: &[Self], right: &[Self]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max)
    }
}

impl RouteDiff for Complex64 {
    fn max_abs_diff(left: &[Self], right: &[Self]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max)
    }
}

impl RouteDiff for Complex32 {
    fn max_abs_diff(left: &[Self], right: &[Self]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(a, b)| f64::from((a - b).norm()))
            .fold(0.0, f64::max)
    }
}

fn compare_routes<T: TensorElement + RouteDiff + std::fmt::Debug>(
    value: impl Fn(usize) -> T + Copy,
    tolerance: f64,
) {
    let (sites, tree) = chain_tree(value);
    let (flat, shape) = all_points(&sites);
    let points = ColMajorArrayRef::new(&flat, &shape).expect("points");

    let options = CachedEvaluatorOptions {
        center: Some(0usize),
        ..Default::default()
    };
    let global = {
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &sites, options.clone()).expect("global evaluator");
        evaluator
            .evaluate_batched_typed::<T>(points, Default::default())
            .expect("global route")
    };

    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).expect("1T CPU"));
    let session = {
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &sites, options).expect("session evaluator");
        context
            .with_concrete_session(|session| {
                evaluator.evaluate_batched_typed_on_session::<T>(
                    session,
                    points,
                    Default::default(),
                )
            })
            .expect("session entry")
            .expect("session route")
    };

    let error = T::max_abs_diff(&global, &session);
    assert!(
        error <= tolerance,
        "session route differs from the global route by {error} (tolerance {tolerance})"
    );
}

fn real_value(index: usize) -> f64 {
    ((index * 7 + 3) % 11) as f64 / 5.0 - 1.0
}

fn complex_value(index: usize) -> Complex64 {
    Complex64::new(real_value(index), ((index * 3 + 1) % 7) as f64 / 7.0)
}

#[test]
fn cached_evaluator_session_route_matches_global_for_f64_across_the_grouped_kernel() {
    compare_routes::<f64>(real_value, 1e-10);
}

#[test]
fn cached_evaluator_session_route_matches_global_for_c64() {
    compare_routes::<Complex64>(complex_value, 1e-10);
}

#[test]
fn cached_evaluator_session_route_matches_global_for_f32() {
    compare_routes::<f32>(|index| real_value(index) as f32, 1e-5);
}

#[test]
fn cached_evaluator_session_route_matches_global_for_c32() {
    compare_routes::<Complex32>(
        |index| {
            let value = complex_value(index);
            Complex32::new(value.re as f32, value.im as f32)
        },
        1e-5,
    );
}

#[test]
fn session_evaluation_completes_while_the_global_backend_is_held() {
    let (sites, tree) = chain_tree(real_value);
    let (flat, shape) = all_points(&sites);
    let points = ColMajorArrayRef::new(&flat, &shape).expect("points");
    let options = CachedEvaluatorOptions {
        center: Some(0usize),
        ..Default::default()
    };

    // The one legitimate global use: the reference before the global backend is held.
    let expected = {
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &sites, options.clone()).expect("global evaluator");
        evaluator
            .evaluate_batched_typed::<f64>(points, Default::default())
            .expect("global route")
    };

    let (values, _) = with_default_backend(|_global| {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<Result<Vec<f64>, String>>();

        let worker = std::thread::spawn(move || {
            let outcome = (|| -> Result<Vec<f64>, String> {
                let mut backend = CpuBackend::with_threads(1).map_err(|error| error.to_string())?;
                ready_tx.send(()).expect("signal session ready");
                go_rx.recv().expect("start the session work");
                let mut session = Session::new(backend.engine_mut());
                let values = {
                    let points = ColMajorArrayRef::new(&flat, &shape).expect("points");
                    let mut evaluator = TreeTNCachedEvaluator::new(&tree, &sites, options.clone())
                        .expect("evaluator");
                    evaluator.evaluate_batched_typed_on_session::<f64>(
                        &mut session,
                        points,
                        Default::default(),
                    )
                }
                .map_err(|error| error.to_string())?;
                Ok(values)
            })();
            let _ = done_tx.send(outcome);
        });

        ready_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the session worker is ready");
        go_tx.send(()).expect("start the session worker");
        let result = match done_rx.recv_timeout(Duration::from_secs(120)) {
            Ok(result) => result,
            Err(_) => panic!(
                "session evaluation did not finish while the process-global backend was held: a \
                 numerical path on this route still consulted the global session"
            ),
        };
        worker.join().expect("session worker thread");
        (result.expect("session route result"), ())
    });

    let error = f64::max_abs_diff(&expected, &values);
    assert!(
        error <= 1e-10,
        "held-backend session result differs by {error}"
    );
}
