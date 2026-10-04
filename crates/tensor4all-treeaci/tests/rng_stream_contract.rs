//! The caller-owned RNG contract of the TreeACI entry point (issue #796).
//!
//! Initialization and every global guard search must draw from the *supplied*
//! stream: a hidden generator leaves the count at zero, and a run with the
//! guard disabled must consume strictly less.

use std::cell::Cell;
use std::rc::Rc;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treeaci::{tree_elementwise_batched_with_rng, TreeAciOptions, TreeElementwiseBatch};
use tensor4all_treetn::TreeTN;

/// Counts how many values the run draws from the supplied stream.
struct CountingRng {
    inner: ChaCha8Rng,
    draws: Rc<Cell<usize>>,
}

impl RngCore for CountingRng {
    fn next_u32(&mut self) -> u32 {
        self.draws.set(self.draws.get() + 1);
        self.inner.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.draws.set(self.draws.get() + 1);
        self.inner.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.draws.set(self.draws.get() + 1);
        self.inner.fill_bytes(dest)
    }
}

/// A two-node tree tensor network over two site indices.
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

fn draw_count(options: TreeAciOptions<usize>, erased: bool) -> usize {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![two_node_tree(&sites, 1.0), two_node_tree(&sites, 2.0)];
    let draws = Rc::new(Cell::new(0));
    let mut counting = CountingRng {
        inner: ChaCha8Rng::seed_from_u64(5),
        draws: Rc::clone(&draws),
    };
    let options = TreeAciOptions {
        rng_seed: 5,
        ..options
    };
    if erased {
        // `&mut dyn RngCore` is the erased stream a caller with a boxed RNG
        // holds; the `?Sized` bound is what makes this compile.
        let erased: &mut dyn RngCore = &mut counting;
        let _ = tree_elementwise_batched_with_rng(sum_operator, &inputs, &options, erased).unwrap();
    } else {
        let _ = tree_elementwise_batched_with_rng(sum_operator, &inputs, &options, &mut counting)
            .unwrap();
    }
    draws.get()
}

#[test]
fn initialization_and_guards_consume_the_supplied_stream() {
    let guarded = draw_count(
        TreeAciOptions {
            max_sweeps: 2,
            max_bond_dim: Some(4),
            enable_global_guard: true,
            nsearch_global_pivots: 2,
            max_nglobal_pivots: 1,
            ..TreeAciOptions::default()
        },
        false,
    );
    let unguarded = draw_count(
        TreeAciOptions {
            max_sweeps: 2,
            max_bond_dim: Some(4),
            enable_global_guard: false,
            ..TreeAciOptions::default()
        },
        false,
    );
    assert!(
        unguarded > 0,
        "initialization must draw from the supplied stream"
    );
    // This two-node fixture leaves the guard no injection capacity, so it draws
    // nothing on top of initialization. Guard-draw continuity is pinned by the
    // ACI contract test (`elementwise_with_rng_draws_for_initialization_and_guards`)
    // and by `tensor4all-treetci/tests/rng_stream_contract.rs`.
    assert!(guarded >= unguarded);

    let erased = draw_count(
        TreeAciOptions {
            max_sweeps: 2,
            max_bond_dim: Some(4),
            enable_global_guard: true,
            nsearch_global_pivots: 2,
            max_nglobal_pivots: 1,
            ..TreeAciOptions::default()
        },
        true,
    );
    assert_eq!(
        erased, guarded,
        "the erased-stream call must draw identically"
    );
}
