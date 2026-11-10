//! Explicit-session tree factorization, canonicalization and norm.
//!
//! The concrete-frontend routes must run entirely on the supplied session: a
//! thread that holds one must not reach the process-global backend, and the work
//! must complete while another thread owns that global backend.

use std::sync::mpsc;
use std::time::Duration;

use tensor4all_core::{DynIndex, FactorizeOptions, IdxTensor, IndexLike};
use tensor4all_tensorbackend::{explicit::Session, with_default_backend, CpuBackend};
use tensor4all_treetn::{
    factorize_tensor_to_treetn_with_on_session, CanonicalizationOptions, TreeTopology,
};

fn chain_topology() -> (Vec<DynIndex>, TreeTopology<usize, DynIndex>) {
    let sites: Vec<DynIndex> = [2, 3, 2]
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect();
    let topology = TreeTopology::new(
        [
            (0usize, vec![sites[0].clone()]),
            (1usize, vec![sites[1].clone()]),
            (2usize, vec![sites[2].clone()]),
        ]
        .into(),
        vec![(0, 1), (1, 2)],
    );
    (sites, topology)
}

/// A dense tensor over the chain's sites in `[site0, site1, site2]` order.
fn dense_chain(sites: &[DynIndex]) -> IdxTensor {
    let dims: Vec<usize> = sites.iter().map(IndexLike::dim).collect();
    let count: usize = dims.iter().product();
    let values: Vec<f64> = (0..count)
        .map(|flat| ((flat * 7 + 3) % 11) as f64 / 5.0 - 1.0)
        .collect();
    IdxTensor::from_dense(sites.to_vec(), values).expect("dense chain tensor")
}

#[test]
fn factorization_canonicalization_and_norm_complete_while_the_global_backend_is_held() {
    let (sites, topology) = chain_topology();
    let dense = dense_chain(&sites);
    // Warm the process-global eager runtime and compute the reference before the
    // global backend is held: those are the one legitimate global use.
    let expected_norm = dense.norm().expect("reference norm");

    // Holding the process-global backend makes any route that consults it block
    // here; the worker runs the explicit-session route on its own context.
    let (norm, centers, nodes) = with_default_backend(|_global| {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<Result<(f64, usize, usize), String>>();

        let worker = std::thread::spawn(move || {
            let outcome = (|| -> Result<(f64, usize, usize), String> {
                let mut backend = CpuBackend::with_threads(1).map_err(|error| error.to_string())?;
                // The holder is ready as soon as its own session exists; the
                // process-global backend is already held by the caller.
                ready_tx.send(()).expect("signal session ready");
                go_rx.recv().expect("start the session work");
                let mut session = Session::new(backend.engine_mut());
                let (norm, centers, nodes) = (|| {
                    let mut tree = factorize_tensor_to_treetn_with_on_session(
                        &dense,
                        &topology,
                        FactorizeOptions::qr(),
                        &1usize,
                        &mut session,
                    )?;
                    tree.canonicalize_mut_on_session(
                        [1usize],
                        CanonicalizationOptions::default(),
                        &mut session,
                    )?;
                    let norm = tree.norm_on_session(&mut session)?;
                    let centers = tree.canonical_region().len();
                    Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                        norm,
                        centers,
                        tree.node_count(),
                    ))
                })()
                .map_err(|error| error.to_string())?;
                Ok((norm, centers, nodes))
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
                "session factorization did not finish while the process-global backend was held: a \
                 numerical path on this route still consulted the global session"
            ),
        };
        worker.join().expect("session worker thread");
        result.expect("session route result")
    });

    assert_eq!(nodes, 3, "all topology nodes are represented");
    assert_eq!(centers, 1, "canonicalization left a single center");
    assert!(
        (norm - expected_norm).abs() <= 1e-10 * expected_norm.max(1.0),
        "session norm {norm} does not match the dense reference {expected_norm}"
    );
}

/// CI canonicalization runs on the session too: on a branched tree it gives
/// the compatibility route's tensors bit for bit, and it completes while the
/// process-global backend is borrowed on this thread (any global entry would
/// be refused as a re-entry).
#[test]
fn ci_canonicalization_on_a_session_matches_the_compatibility_route() {
    use tensor4all_treetn::{CanonicalForm, TreeTN};

    // A star 0 - {1, 2, 3} with bond dimension 2 and one site per node.
    let sites: Vec<DynIndex> = (0..4).map(|_| DynIndex::new_dyn(3)).collect();
    let bonds: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let fill = |legs: Vec<DynIndex>, salt: usize| {
        let count: usize = legs.iter().map(IndexLike::dim).product();
        let values: Vec<f64> = (0..count)
            .map(|flat| (((flat + salt) * 7 + 3) % 13) as f64 / 6.0 - 1.0)
            .collect();
        IdxTensor::from_dense(legs, values).expect("tensor")
    };
    let mut tensors = vec![fill(
        vec![
            sites[0].clone(),
            bonds[0].clone(),
            bonds[1].clone(),
            bonds[2].clone(),
        ],
        0,
    )];
    for leaf in 1..4 {
        tensors.push(fill(
            vec![bonds[leaf - 1].clone(), sites[leaf].clone()],
            leaf,
        ));
    }
    let tree = TreeTN::<IdxTensor, usize>::from_tensors(tensors, vec![0, 1, 2, 3]).expect("tree");
    let options = CanonicalizationOptions::default().with_form(CanonicalForm::CI);

    let mut global = tree.clone();
    global
        .canonicalize_mut([0usize], options.clone())
        .expect("compatibility CI canonicalization");

    let mut backend = CpuBackend::with_threads(1).expect("backend");
    let session_tree = with_default_backend(|_global| {
        let mut session = Session::new(backend.engine_mut());
        let mut session_tree = tree.clone();
        session_tree
            .canonicalize_mut_on_session([0usize], options.clone(), &mut session)
            .expect("the CI canonicalization must not reach the borrowed global backend");
        session_tree
    });

    assert_eq!(session_tree.canonical_form(), Some(CanonicalForm::CI));
    for node in 0..4usize {
        let tensor = |tree: &TreeTN<IdxTensor, usize>| {
            let index = tree.node_index(&node).expect("node");
            let tensor = tree.tensor(index).expect("tensor");
            (
                tensor.dims(),
                tensor.dense_host_values::<f64>().expect("host values"),
            )
        };
        assert_eq!(tensor(&session_tree), tensor(&global), "node {node}");
    }
}
