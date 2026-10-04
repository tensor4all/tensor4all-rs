//! The caller-owned RNG contract of the treetci entry points (issue #796).
//!
//! A run whose randomized global search is disabled must not touch the supplied
//! stream at all, and a run that does search must consume it directly.

use std::cell::Cell;
use std::rc::Rc;

use anyhow::Result;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_treetci::{
    optimize_with_proposer_with_rng, GlobalIndexBatch, SimpleProposer, TreeTCI2, TreeTciEdge,
    TreeTciGraph, TreeTciOptions,
};

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

fn two_site_state() -> TreeTCI2<f64> {
    let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
    let mut state = TreeTCI2::<f64>::new(vec![2, 2], graph).unwrap();
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    state.max_sample_value = 1.0;
    state
}

fn evaluate(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
    let mut values = Vec::with_capacity(batch.n_points());
    for point in 0..batch.n_points() {
        let i = batch.get(0, point).unwrap();
        let j = batch.get(1, point).unwrap();
        values.push(if i == j { 1.0 } else { 0.0 });
    }
    Ok(values)
}

#[test]
fn a_run_without_a_global_search_does_not_touch_the_supplied_stream() {
    let draws = Rc::new(Cell::new(0));
    let mut rng = CountingRng {
        inner: ChaCha8Rng::seed_from_u64(3),
        draws: Rc::clone(&draws),
    };
    let options = TreeTciOptions {
        tolerance: 1e-10,
        max_iter: 2,
        enable_global_pivots: false,
        ..TreeTciOptions::default()
    };
    let (ranks, errors) = optimize_with_proposer_with_rng(
        &mut two_site_state(),
        evaluate,
        &options,
        &SimpleProposer::default(),
        &mut rng,
    )
    .unwrap();
    assert_eq!(ranks.len(), errors.len());
    assert_eq!(
        draws.get(),
        0,
        "a run with the global search disabled must not consume the caller's stream"
    );
}

#[test]
fn a_run_with_a_global_search_consumes_the_supplied_stream() {
    let draws = Rc::new(Cell::new(0));
    let mut rng = CountingRng {
        inner: ChaCha8Rng::seed_from_u64(3),
        draws: Rc::clone(&draws),
    };
    let options = TreeTciOptions {
        tolerance: 1e-10,
        max_iter: 3,
        enable_global_pivots: true,
        nsearch: 2,
        max_nglobal_pivot: 2,
        ..TreeTciOptions::default()
    };
    optimize_with_proposer_with_rng(
        &mut two_site_state(),
        evaluate,
        &options,
        &SimpleProposer::default(),
        &mut rng,
    )
    .unwrap();
    assert!(
        draws.get() > 0,
        "an enabled global search must draw its starts from the supplied stream"
    );
}

#[test]
fn the_low_level_entry_point_accepts_an_erased_stream() {
    // `&mut dyn RngCore` exercises the `?Sized` bound of the public entry point.
    let mut inner = ChaCha8Rng::seed_from_u64(3);
    let rng: &mut dyn RngCore = &mut inner;
    let options = TreeTciOptions {
        tolerance: 1e-10,
        max_iter: 2,
        enable_global_pivots: true,
        nsearch: 1,
        max_nglobal_pivot: 1,
        ..TreeTciOptions::default()
    };
    let (ranks, errors) = optimize_with_proposer_with_rng(
        &mut two_site_state(),
        evaluate,
        &options,
        &SimpleProposer::default(),
        rng,
    )
    .unwrap();
    assert_eq!(ranks.len(), errors.len());
}
