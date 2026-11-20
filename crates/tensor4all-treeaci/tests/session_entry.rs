//! The session-scoped entry `tree_elementwise_batched_with_rng_on_session`
//! (issue #1235).
//!
//! It runs the whole algorithm on a caller's explicit session: the result is
//! the plain route's bit for bit, the run completes while the process-global
//! default engine is borrowed on the same thread (any global entry would be
//! refused as a re-entry), it accepts inputs published as compact host
//! storage, and its result outlives the session.

use crate::g0;

use num_complex::Complex64;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_tensorbackend::{explicit::Session, with_default_backend, CpuBackend};
use tensor4all_treeaci::{
    tree_elementwise_batched_with_rng, tree_elementwise_batched_with_rng_on_session,
    TreeAciOptions, TreeAciResult, TreeElementwiseBatch,
};
use tensor4all_treetn::TreeTN;

/// The product of all inputs at every point.
fn product(
    batch: TreeElementwiseBatch<'_, Complex64>,
    out: &mut [Complex64],
) -> Result<(), tensor4all_treeaci::TreeAciError> {
    let n_inputs = batch.n_inputs();
    for (value, inputs) in out
        .iter_mut()
        .zip(batch.as_col_major_slice().chunks_exact(n_inputs))
    {
        *value = inputs.iter().product();
    }
    Ok(())
}

fn options() -> TreeAciOptions<usize> {
    TreeAciOptions {
        rng_seed: 7,
        ..Default::default()
    }
}

fn plain(inputs: &[TreeTN<IdxTensor, usize>]) -> TreeAciResult<usize> {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    tree_elementwise_batched_with_rng(product, inputs, &options(), &mut rng).unwrap()
}

/// Run the session entry while the process-global default engine is borrowed.
fn on_session(inputs: &[TreeTN<IdxTensor, usize>]) -> TreeAciResult<usize> {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    with_default_backend(|_global| {
        let mut session = Session::new(backend.engine_mut());
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        tree_elementwise_batched_with_rng_on_session(
            product,
            inputs,
            &options(),
            &mut rng,
            &mut session,
        )
        .expect("the session entry must not reach the borrowed global engine")
    })
}

fn dense(result: &TreeAciResult<usize>) -> Vec<Complex64> {
    result
        .tree
        .to_dense()
        .unwrap()
        .to_vec::<Complex64>()
        .unwrap()
}

/// The same networks with every tensor rebuilt as compact host storage.
fn compact(inputs: &[TreeTN<IdxTensor, usize>]) -> Vec<TreeTN<IdxTensor, usize>> {
    inputs
        .iter()
        .map(|input| {
            let mut names = input.node_names();
            names.sort();
            let tensors = names
                .iter()
                .map(|node| {
                    let tensor = input.tensor(input.node_index(node).unwrap()).unwrap();
                    IdxTensor::from_dense_compact(
                        tensor.indices().to_vec(),
                        tensor.to_vec::<Complex64>().unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            TreeTN::from_tensors(tensors, names).unwrap()
        })
        .collect()
}

#[test]
fn the_session_entry_matches_the_plain_route_without_entering_the_global_runtime() {
    // CTTN: a branched tree whose junction node has degree three.
    let (_sites, inputs) = g0::fixture(3, "cttn").unwrap();
    let reference = plain(&inputs);
    let result = on_session(&inputs);
    assert_eq!(result.max_ranks, reference.max_ranks);
    assert_eq!(result.termination, reference.termination);
    assert_eq!(dense(&result), dense(&reference));
}

#[test]
fn compact_inputs_are_accepted_and_the_result_outlives_its_session() {
    let (_sites, inputs) = g0::fixture(3, "cttn").unwrap();
    let reference = dense(&plain(&inputs));
    let result = on_session(&compact(&inputs));
    // The backend and session of `on_session` are gone here.
    assert_eq!(dense(&result), reference);
}

#[test]
fn a_single_node_runs_on_the_session() {
    let site = DynIndex::new_dyn(3);
    let input = |values: Vec<Complex64>| {
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
            vec![0usize],
        )
        .unwrap()
    };
    let values = |scale: f64| {
        (0..3)
            .map(|k| Complex64::new(scale * (k + 1) as f64, -(k as f64)))
            .collect::<Vec<_>>()
    };
    let inputs = [input(values(1.0)), input(values(0.5))];
    let result = on_session(&inputs);
    let expected: Vec<Complex64> = values(1.0)
        .iter()
        .zip(values(0.5))
        .map(|(a, b)| a * b)
        .collect();
    assert_eq!(dense(&result), expected);
}

#[test]
fn the_guard_and_the_final_canonicalization_ran_on_the_session() {
    let (_sites, inputs) = g0::fixture(3, "cttn").unwrap();
    let result = on_session(&inputs);
    // The global guard ran (it is on by default) and the output was put into
    // CI form on the session.
    assert!(options().enable_global_guard);
    assert!(!result.global_pivots_found.is_empty());
    assert_eq!(
        result.tree.canonical_form(),
        Some(tensor4all_treetn::CanonicalForm::CI)
    );
}
