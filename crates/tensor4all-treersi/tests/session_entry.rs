//! The session entry (`hadamard_many_with_rng_on_session`, #1236).
//!
//! Value parity alone would also hold for a route that entered the process-global
//! context, so the dispatch test holds the process-global backend while the session
//! entry runs on a worker thread: a numerical step that consulted the global session
//! (the default CPU context a CUDA run falls back to for its rank decision, or the
//! global eager readback) would block there, and the timeout turns that into a failure.

use std::sync::Arc;
use std::time::Duration;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{permute_on_session, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{
    with_default_backend, CpuBackend, CpuExecutionContext, ExecutionContext, StorageKind,
    StorageScalar,
};
use tensor4all_treersi::{
    hadamard_many_with_rng_in, hadamard_many_with_rng_on_session, TreeRsiError, TreeRsiOptions,
    TreeRsiResult, TreeRsiScalar,
};
use tensor4all_treetn::TreeTN;

use crate::support::{random_cores, tree_from_cores};

type Tree = TreeTN<IdxTensor, usize>;
type Cores<T> = Vec<(Vec<DynIndex>, Vec<T>)>;

/// The scalars these tests compare exactly and read from compact storage.
trait Exact: TreeRsiScalar + StorageScalar + PartialEq + std::fmt::Debug {}
impl<T: TreeRsiScalar + StorageScalar + PartialEq + std::fmt::Debug> Exact for T {}

const CHAIN: &[(usize, usize)] = &[(0, 1), (1, 2), (2, 3), (3, 4)];
const BINARY: &[(usize, usize)] = &[(0, 1), (0, 2), (1, 3), (1, 4)];
const STAR: &[(usize, usize)] = &[(0, 1), (0, 2), (0, 3), (0, 4)];

/// A private context with a fixed backend configuration: the context entry and
/// the session entry are compared on the same configuration.
fn context() -> CpuExecutionContext {
    CpuExecutionContext::from_backend(CpuBackend::with_threads(1).expect("1T CPU backend"))
}

fn options<T: TreeRsiScalar>(k: usize, root: usize) -> TreeRsiOptions<usize> {
    TreeRsiOptions {
        max_bond_dim: Some(6),
        sketch_dim: Some(k),
        root: Some(root),
        rel_tol: T::epsilon() * 10.0,
        ..Default::default()
    }
}

/// Three operands of one topology, as raw cores sharing their indices.
fn operands<T: TreeRsiScalar>(
    sites: &[DynIndex],
    edges: &[(usize, usize)],
    sample: impl Fn(f64, f64) -> T + Copy,
) -> Vec<Cores<T>> {
    (0..3)
        .map(|a| random_cores(sites, edges, 2, 300 + a, sample))
        .collect()
}

fn compact_trees<T: TreeRsiScalar>(operands: &[Cores<T>]) -> Vec<Tree> {
    operands
        .iter()
        .map(|cores| {
            tree_from_cores(cores.clone(), |indices, data| {
                IdxTensor::from_dense_compact(indices, data).expect("compact core")
            })
        })
        .collect()
}

fn context_trees<T: TreeRsiScalar>(operands: &[Cores<T>], context: &ExecutionContext) -> Vec<Tree> {
    operands
        .iter()
        .map(|cores| {
            tree_from_cores(cores.clone(), |indices, data| {
                IdxTensor::from_dense_in(context, indices, data).expect("context core")
            })
        })
        .collect()
}

/// The result's dense values over `sites`, read without entering any runtime:
/// every output core of the session entry is compact host storage.
fn compact_dense<T: Exact>(result: &TreeRsiResult<usize>, sites: &[DynIndex]) -> Vec<T> {
    for node in 0..sites.len() {
        let core = result
            .tree
            .tensor(result.tree.node_index(&node).expect("node"))
            .expect("core");
        assert_eq!(core.storage_kind(), StorageKind::Dense);
    }
    result
        .tree
        .to_dense()
        .expect("dense result")
        .permute_indices(sites)
        .expect("site order")
        .dense_host_values::<T>()
        .expect("compact values")
}

fn context_dense<T: TreeRsiScalar>(result: &TreeRsiResult<usize>, sites: &[DynIndex]) -> Vec<T> {
    result
        .tree
        .to_dense()
        .expect("dense result")
        .permute_indices(sites)
        .expect("site order")
        .to_vec::<T>()
        .expect("values")
}

/// Same diagnostics, same values to the last bit, and the same RNG position.
fn assert_same_result<T: Exact>(
    on_session: &TreeRsiResult<usize>,
    in_context: &TreeRsiResult<usize>,
    sites: &[DynIndex],
    label: &str,
) {
    let (a, b) = (&on_session.diagnostics, &in_context.diagnostics);
    assert_eq!(a.root, b.root, "{label}");
    assert_eq!(a.sketch_dim, b.sketch_dim, "{label}");
    assert_eq!(a.edges.len(), b.edges.len(), "{label}");
    for (x, y) in a.edges.iter().zip(&b.edges) {
        assert_eq!(
            (
                x.child,
                x.parent,
                x.rank,
                x.rows,
                x.columns,
                x.exact_columns,
                x.rank_limit_reached
            ),
            (
                y.child,
                y.parent,
                y.rank,
                y.rows,
                y.columns,
                y.exact_columns,
                y.rank_limit_reached
            ),
            "{label}"
        );
        assert_eq!(
            x.relative_pivot.to_bits(),
            y.relative_pivot.to_bits(),
            "{label}"
        );
    }
    let session_values = compact_dense::<T>(on_session, sites);
    let context_values = context_dense::<T>(in_context, sites);
    assert_eq!(session_values, context_values, "{label}");
}

fn session_entry_matches_the_context_entry<T: Exact>(sample: impl Fn(f64, f64) -> T + Copy) {
    for (name, edges, root) in [
        ("chain", CHAIN, 4),
        ("binary", BINARY, 0),
        ("star", STAR, 2),
    ] {
        let sites = (0..5).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
        let operands = operands(&sites, edges, sample);
        // k = 3 leaves sketched edges, k = 32 makes every edge exact.
        for k in [3, 32] {
            let label = format!("{name}, k={k}");
            let options = options::<T>(k, root);
            let cpu = ExecutionContext::Cpu(Arc::new(context()));
            let mut context_rng = ChaCha8Rng::seed_from_u64(23);
            let in_context = hadamard_many_with_rng_in::<T, _, _>(
                &context_trees(&operands, &cpu),
                &options,
                &mut context_rng,
                &cpu,
            )
            .expect("context entry");
            let mut session_rng = ChaCha8Rng::seed_from_u64(23);
            let inputs = compact_trees(&operands);
            let on_session = context()
                .with_concrete_session(|session| {
                    hadamard_many_with_rng_on_session::<T, _, _>(
                        &inputs,
                        &options,
                        &mut session_rng,
                        session,
                    )
                })
                .expect("session entry")
                .expect("run on session");
            if k == 3 {
                assert!(
                    on_session
                        .diagnostics
                        .edges
                        .iter()
                        .any(|e| !e.exact_columns),
                    "{label}: the fixture must exercise a sketched edge"
                );
            }
            assert_same_result::<T>(&on_session, &in_context, &sites, &label);
            assert_eq!(
                session_rng.random::<u64>(),
                context_rng.random::<u64>(),
                "{label}: both entries must consume the same draws"
            );
        }
    }
}

#[test]
fn session_entry_matches_the_context_entry_f64() {
    session_entry_matches_the_context_entry::<f64>(|x, _| x);
}

#[test]
fn session_entry_matches_the_context_entry_c64() {
    session_entry_matches_the_context_entry::<Complex64>(Complex64::new);
}

#[test]
fn session_entry_runs_while_the_process_global_backend_is_held() {
    let sites = (0..5).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let operands = operands::<f64>(&sites, BINARY, |x, _| x);
    let options = options::<f64>(3, 0);
    let cpu = ExecutionContext::Cpu(Arc::new(context()));
    let mut rng = ChaCha8Rng::seed_from_u64(29);
    let reference = hadamard_many_with_rng_in::<f64, _, _>(
        &context_trees(&operands, &cpu),
        &options,
        &mut rng,
        &cpu,
    )
    .expect("reference");
    assert!(
        reference
            .diagnostics
            .edges
            .iter()
            .any(|e| e.rank > 1 && !e.exact_columns),
        "the fixture must take nontrivial rank decisions on sketched edges"
    );
    // Warm the process-global eager runtime first: cold initialization takes the global
    // context's backend lock, which this test holds. The warm-up is a plain dense
    // readback through the global route, not the code under test, so a broken session
    // route cannot fail here instead of under the held lock.
    let warm = compact_trees(&operands)[0]
        .to_dense()
        .expect("warm-up")
        .to_vec::<f64>()
        .expect("warm-up");
    assert_eq!(warm.len(), 32);

    let inputs = compact_trees(&operands);
    let result = with_default_backend(|_global| {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = context().with_concrete_session(|session| {
                let mut rng = ChaCha8Rng::seed_from_u64(29);
                hadamard_many_with_rng_on_session::<f64, _, _>(&inputs, &options, &mut rng, session)
            });
            let _ = done_tx.send(());
            result
        });
        match done_rx.recv_timeout(Duration::from_secs(120)) {
            Ok(()) => worker.join().expect("worker thread"),
            Err(_) => panic!(
                "the session entry did not finish while the process-global backend was held: a \
                 numerical step on this route still consulted the global session (for example \
                 the default CPU context of the CUDA rank-decision fallback)"
            ),
        }
    })
    .expect("session entry")
    .expect("run on session");
    assert_same_result::<f64>(&result, &reference, &sites, "held global");
}

#[test]
fn inputs_published_on_another_session_are_accepted_and_the_result_outlives_its_session() {
    let sites = (0..5).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let operands = operands::<f64>(&sites, CHAIN, |x, _| x);
    let options = options::<f64>(3, 4);

    // Each core is published by session A as compact host storage, with its axes
    // reversed so the publication is real session work rather than a metadata copy.
    let publisher = context();
    let inputs = publisher
        .with_concrete_session(|session| {
            compact_trees(&operands)
                .into_iter()
                .map(|tree| {
                    let tensors = (0..sites.len())
                        .map(|node| {
                            let core = tree.tensor(tree.node_index(&node).unwrap()).unwrap();
                            let reversed = (0..core.indices().len()).rev().collect::<Vec<_>>();
                            permute_on_session(session, core, &reversed).unwrap()
                        })
                        .collect();
                    TreeTN::from_tensors(tensors, (0..sites.len()).collect()).unwrap()
                })
                .collect::<Vec<_>>()
        })
        .expect("publishing session");
    drop(publisher);
    for tree in &inputs {
        let core = tree.tensor(tree.node_index(&0).unwrap()).unwrap();
        assert_eq!(core.storage_kind(), StorageKind::Dense);
    }

    // Session B runs the product and is released, and its context dropped, before
    // the result is read.
    let result = {
        let runner = context();
        let result = runner
            .with_concrete_session(|session| {
                let mut rng = ChaCha8Rng::seed_from_u64(31);
                hadamard_many_with_rng_on_session::<f64, _, _>(&inputs, &options, &mut rng, session)
            })
            .expect("session entry")
            .expect("run on session");
        drop(runner);
        result
    };

    let cpu = ExecutionContext::Cpu(Arc::new(context()));
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let reference = hadamard_many_with_rng_in::<f64, _, _>(
        &context_trees(&operands, &cpu),
        &options,
        &mut rng,
        &cpu,
    )
    .expect("reference");
    assert_same_result::<f64>(&result, &reference, &sites, "published inputs");

    // The product itself, checked against the dense oracle: the comparison above
    // would also pass for two equally wrong entries.
    let mut expected = vec![1.0_f64; 32];
    for cores in &operands {
        let tree = tree_from_cores(cores.clone(), |indices, data| {
            IdxTensor::from_dense_compact(indices, data).unwrap()
        });
        let dense = tree
            .to_dense()
            .unwrap()
            .permute_indices(&sites)
            .unwrap()
            .dense_host_values::<f64>()
            .unwrap();
        for (x, y) in expected.iter_mut().zip(dense) {
            *x *= y;
        }
    }
    let actual = compact_dense::<f64>(&result, &sites);
    let oracle = IdxTensor::from_dense(sites.clone(), expected).unwrap();
    let error = IdxTensor::from_dense(sites.clone(), actual)
        .unwrap()
        .sub(&oracle)
        .unwrap()
        .maxabs()
        .unwrap();
    let scale = oracle.maxabs().unwrap();
    assert!(error / scale < 1e-10, "rel max error {}", error / scale);
}

#[test]
fn session_entry_reports_another_scalar_type_as_a_scalar_kind() {
    let sites = (0..5).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let inputs = compact_trees(&operands::<f64>(&sites, CHAIN, |x, _| x));
    let mut rng = ChaCha8Rng::seed_from_u64(37);
    let mut expected = rng.clone();
    let error = context()
        .with_concrete_session(|session| {
            hadamard_many_with_rng_on_session::<f32, _, _>(
                &inputs,
                &options::<f32>(3, 0),
                &mut rng,
                session,
            )
        })
        .expect("session entry")
        .expect_err("f64 cores are not f32");
    assert!(matches!(error, TreeRsiError::ScalarKind { .. }), "{error}");
    assert_eq!(rng.random::<u64>(), expected.random::<u64>());
}
