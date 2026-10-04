//! The caller-owned RNG contract of the tensorci entry points (issue #796).
//!
//! The tests compare the *stream position* before and after a call with a
//! reference `ChaCha8Rng` advanced by the documented number of draws. A finder
//! that derived a seed for a hidden generator, or that created its own stream,
//! leaves the caller's position untouched and fails.

use rand::{Rng as _, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_simplett::SimpleTensorTrain;
use tensor4all_tensorci::{
    crossinterpolate2, crossinterpolate2_with_rng, DefaultGlobalPivotFinder, GlobalPivotFinder,
    GlobalPivotSearchInput, TCI2Options,
};

/// The batched-callback type the interpolation entry points accept.
type EmptyBatch = Option<fn(&[Vec<usize>]) -> Vec<f64>>;

/// A zero tensor train on `[4, 4]` and `f(i, j) = i`.
#[allow(clippy::ptr_arg)] // the finder's callback type is `Fn(&MultiIndex) -> T`
fn f(index: &Vec<usize>) -> f64 {
    index[0] as f64
}

/// A nonzero-valued target for the interpolation-level comparison.
#[allow(clippy::ptr_arg)] // the interpolation callback type is `Fn(&MultiIndex) -> T`
fn g(index: &Vec<usize>) -> f64 {
    (index[0] + index[1] + 1) as f64
}

fn input() -> GlobalPivotSearchInput<f64> {
    GlobalPivotSearchInput {
        local_dims: vec![4, 4],
        current_tt: SimpleTensorTrain::<f64>::constant(&[4, 4], 0.0),
        max_sample_value: 4.0,
        i_set: vec![vec![vec![]], vec![vec![0]]],
        j_set: vec![vec![vec![0]], vec![vec![]]],
    }
}

/// One search with `nsearch = 1` on two sites draws one starting point: one
/// coordinate per site, and nothing else.
#[test]
fn the_finder_consumes_exactly_one_starting_point_from_the_supplied_stream() {
    let finder = DefaultGlobalPivotFinder::new(1, 1, 10.0);
    let mut stream = ChaCha8Rng::seed_from_u64(11);
    let mut reference = ChaCha8Rng::seed_from_u64(11);

    let pivots = finder
        .find_global_pivots(&input(), &f, 0.1, &mut stream)
        .unwrap();
    assert_eq!(pivots.len(), 1);
    let _ = (reference.random_range(0..4), reference.random_range(0..4));

    assert_eq!(
        stream.next_u64(),
        reference.next_u64(),
        "the finder must leave the supplied stream where the starting point's draws leave it"
    );
}

/// A second search continues the same stream instead of restarting it.
#[test]
fn consecutive_searches_continue_the_supplied_stream() {
    let finder = DefaultGlobalPivotFinder::new(1, 1, 10.0);
    let mut stream = ChaCha8Rng::seed_from_u64(11);
    let mut reference = ChaCha8Rng::seed_from_u64(11);

    for call in 0..3 {
        finder
            .find_global_pivots(&input(), &f, 0.1, &mut stream)
            .unwrap();
        let _ = (reference.random_range(0..4), reference.random_range(0..4));
        assert_eq!(
            stream.next_u64(),
            reference.next_u64(),
            "call {call} must continue the stream rather than restart it"
        );
    }
}

/// The documented delegation: the seeded entry point is the low-level one with
/// a derived `ChaCha8Rng`.
#[test]
fn the_seeded_entry_point_is_the_low_level_entry_point_with_a_derived_stream() {
    let options = || TCI2Options {
        max_iter: 2,
        seed: Some(7),
        ..TCI2Options::default()
    };
    let batch: EmptyBatch = None;

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

/// `&mut dyn rand::RngCore` is the erased stream a caller with a boxed RNG
/// holds; the `?Sized` bound is what makes this compile.
#[test]
fn the_low_level_entry_points_accept_an_erased_stream() {
    let mut inner = ChaCha8Rng::seed_from_u64(5);
    let erased: &mut dyn rand::RngCore = &mut inner;
    let finder = DefaultGlobalPivotFinder::new(1, 1, 10.0);
    assert_eq!(
        finder
            .find_global_pivots(&input(), &f, 0.1, erased)
            .unwrap()
            .len(),
        1
    );

    let mut inner = ChaCha8Rng::seed_from_u64(5);
    let erased: &mut dyn rand::RngCore = &mut inner;
    let batch: EmptyBatch = None;
    let result = crossinterpolate2_with_rng::<f64, _, _, _>(
        g,
        batch,
        vec![4, 4],
        vec![vec![0, 0]],
        TCI2Options {
            max_iter: 2,
            ..TCI2Options::default()
        },
        erased,
    )
    .unwrap();
    assert!(!result.ranks.is_empty());
}
