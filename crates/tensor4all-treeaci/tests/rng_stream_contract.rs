//! The caller-owned RNG contract of the TreeACI entry point (issue #796).
//!
//! Initialization and every global guard search must draw from the *supplied*
//! stream: a hidden generator leaves the count at zero, and a run with the
//! guard disabled must consume strictly less.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treeaci::{tree_elementwise_batched_with_rng, TreeAciOptions, TreeElementwiseBatch};
use tensor4all_treetn::TreeTN;

/// Sum of the batch's inputs, so the operator is exact and deterministic.
fn sum_operator(
    batch: TreeElementwiseBatch<'_, f64>,
    output: &mut [f64],
) -> tensor4all_treeaci::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for input in 0..batch.n_inputs() {
            sum += batch.get(input, point)?;
        }
        *value = sum;
    }
    Ok(())
}

/// A two-node tree tensor network over two shared site indices.
fn two_node_tree(sites: &[DynIndex; 2], scale: f64) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![sites[0].clone(), bond.clone()],
        (0..sites[0].dim * 2)
            .map(|offset| scale * (offset + 1) as f64)
            .collect(),
    )
    .unwrap();
    let right = IdxTensor::from_dense(
        vec![bond, sites[1].clone()],
        (0..2 * sites[1].dim)
            .map(|offset| scale * (offset + 2) as f64)
            .collect(),
    )
    .unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

/// Initialization draws from the supplied stream, and an erased-stream call
/// leaves the same stream in the same place.
#[test]
fn initialization_consumes_the_supplied_stream_and_accepts_an_erased_one() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![two_node_tree(&sites, 1.0), two_node_tree(&sites, 2.0)];
    let options = TreeAciOptions {
        max_sweeps: 2,
        rng_seed: 5,
        ..TreeAciOptions::default()
    };
    let run = |rng: &mut dyn RngCore| {
        let _ = tree_elementwise_batched_with_rng(sum_operator, &inputs, &options, rng).unwrap();
        (0..16).map(|_| rng.next_u64()).collect::<Vec<_>>()
    };

    let mut plain = ChaCha8Rng::seed_from_u64(5);
    let mut reference = ChaCha8Rng::seed_from_u64(5);
    let plain_tail = run(&mut plain);
    assert_ne!(
        plain_tail,
        (0..16).map(|_| reference.next_u64()).collect::<Vec<_>>(),
        "initialization must draw from the supplied stream"
    );

    // The erased-stream call consumes the same stream in the same way.
    let mut erased_inner = ChaCha8Rng::seed_from_u64(5);
    let erased_tail = run(&mut erased_inner);
    assert_eq!(plain_tail, erased_tail);
}
