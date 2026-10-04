//! The caller-owned RNG contract of the tensorci entry points (issue #796).
//!
//! These tests pin that the supplied stream *is* the stream the finder consumes
//! — not merely that some randomness happened — and that consecutive calls
//! continue it. A finder that derived a seed for a hidden RNG, or that reset
//! the stream, fails them.

use rand::{Rng as _, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cell::Cell;
use std::rc::Rc;
use tensor4all_simplett::SimpleTensorTrain;
use tensor4all_tensorci::{
    crossinterpolate2, crossinterpolate2_with_rng, DefaultGlobalPivotFinder, GlobalPivotFinder,
    GlobalPivotSearchInput, TCI2Options,
};

/// A zero tensor train on `[4, 4]` and `f(i, j) = i`.
///
/// The coordinate walk maximises the first coordinate (`i = 3`) and never
/// improves the second one, because the residual does not depend on `j`. The
/// returned pivot therefore exposes the stream's second drawn coordinate, which
/// is what makes the stream observable from the public API.
fn input() -> GlobalPivotSearchInput<f64> {
    GlobalPivotSearchInput {
        local_dims: vec![4, 4],
        current_tt: SimpleTensorTrain::<f64>::constant(&[4, 4], 0.0),
        max_sample_value: 4.0,
        i_set: vec![vec![vec![]], vec![vec![0]]],
        j_set: vec![vec![vec![0]], vec![vec![]]],
    }
}

fn f(index: &Vec<usize>) -> f64 {
    index[0] as f64
}

/// A nonzero-valued target for the interpolation-level comparison.
fn g(index: &Vec<usize>) -> f64 {
    (index[0] + index[1] + 1) as f64
}

/// Counts how many values the finder draws from the supplied stream.
///
/// A finder that derived a seed for a hidden generator, or that created its own
/// generator, leaves the count at zero.
struct CountingRng {
    inner: ChaCha8Rng,
    draws: Rc<Cell<usize>>,
}

impl rand::RngCore for CountingRng {
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

#[test]
fn the_finder_consumes_the_supplied_stream_and_continues_it() {
    let finder = DefaultGlobalPivotFinder::new(1, 1, 10.0);
    let draws = Rc::new(Cell::new(0));
    let mut stream = CountingRng {
        inner: ChaCha8Rng::seed_from_u64(11),
        draws: Rc::clone(&draws),
    };

    // One search draws one starting point: one coordinate per site.
    finder
        .find_global_pivots(&input(), &f, 0.1, &mut stream)
        .unwrap();
    assert_eq!(
        draws.get(),
        2,
        "one search with nsearch = 1 must draw exactly one start from the supplied stream"
    );

    // A second search continues the same stream instead of restarting it.
    finder
        .find_global_pivots(&input(), &f, 0.1, &mut stream)
        .unwrap();
    assert_eq!(draws.get(), 4, "consecutive calls must continue the stream");
}

#[test]
fn the_seeded_entry_point_is_the_low_level_entry_point_with_a_derived_stream() {
    let options = || TCI2Options {
        max_iter: 2,
        seed: Some(7),
        ..TCI2Options::default()
    };
    let batch: Option<fn(&[Vec<usize>]) -> Vec<f64>> = None;

    let seeded =
        crossinterpolate2::<f64, _, _>(g, batch, vec![4, 4], vec![vec![0, 0]], options()).unwrap();

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let streamed = crossinterpolate2_with_rng::<f64, _, _, _>(
        g,
        batch,
        vec![4, 4],
        vec![vec![0, 0]],
        options(),
        &mut rng,
    )
    .unwrap();

    assert_eq!(seeded.ranks, streamed.ranks);
    assert_eq!(seeded.errors, streamed.errors);
}

#[test]
fn the_finder_leaves_the_supplied_stream_at_the_expected_position() {
    let finder = DefaultGlobalPivotFinder::new(1, 1, 10.0);
    let mut stream = ChaCha8Rng::seed_from_u64(11);
    let mut reference = ChaCha8Rng::seed_from_u64(11);

    finder
        .find_global_pivots(&input(), &f, 0.1, &mut stream)
        .unwrap();
    // One search draws exactly the starting point: one coordinate per site.
    let _ = (reference.random_range(0..4), reference.random_range(0..4));

    assert_eq!(
        stream.next_u64(),
        reference.next_u64(),
        "the finder must consume the supplied stream at the draw position the rule describes \
         (no extra draws, no hidden generator)"
    );
}

#[test]
fn the_low_level_entry_point_accepts_an_erased_stream() {
    // `&mut dyn RngCore` is the erased stream a caller with a boxed RNG holds;
    // the `?Sized` bound is what makes this compile.
    let mut inner = ChaCha8Rng::seed_from_u64(5);
    let rng: &mut dyn RngCore = &mut inner;
    let finder = DefaultGlobalPivotFinder::new(1, 1, 10.0);
    let pivots = finder.find_global_pivots(&input(), &f, 0.1, rng).unwrap();
    assert_eq!(pivots.len(), 1);

    let mut inner = ChaCha8Rng::seed_from_u64(5);
    let rng: &mut dyn RngCore = &mut inner;
    let batch: Option<fn(&[Vec<usize>]) -> Vec<f64>> = None;
    let result = crossinterpolate2_with_rng::<f64, _, _, _>(
        g,
        batch,
        vec![4, 4],
        vec![vec![0, 0]],
        TCI2Options {
            max_iter: 2,
            ..TCI2Options::default()
        },
        rng,
    )
    .unwrap();
    assert!(!result.ranks.is_empty());
}
