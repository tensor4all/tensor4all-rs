//! A run through a unit junction returns a tree in the pivot basis (#1435).
//!
//! The unit-vertex expansion lets a commit select a row candidate outside the
//! neighbouring bond's basis, which its factor then restricts to that basis
//! (`CandidateBasisProjection`). This checks the returned tree of the case
//! the expansion fixes with [`crate::test_support::pivot_basis_deviation`].

use rand::SeedableRng;

use super::super::run_sweeps_and_cleanup;
use crate::{
    exec::Exec,
    state::TreeAciState,
    test_support::{pivot_basis_deviation, unit_junction},
    unit_vertex::expansion_debug_log,
    TreeAciOptions, TreeAciTermination,
};

fn identity(batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, point)?;
    }
    Ok(())
}

/// Seed 1, tolerance 1e-10: 10 expansions, the true centre ranks, the exact
/// tree, and max|M - I| = 0 on the returned tree.
#[test]
fn unit_junction_run_returns_a_tree_in_the_pivot_basis() {
    let input = unit_junction::input().unwrap();
    let inputs = vec![input.clone()];
    let options = TreeAciOptions {
        tolerance: 1e-10,
        rng_seed: 1,
        ..TreeAciOptions::default()
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(options.rng_seed);
    let mut exec = Exec::Global;
    let mut state =
        TreeAciState::initialize_with_rng(&inputs, &options, &mut rng, &mut exec).unwrap();
    expansion_debug_log::take();
    let (history, _) =
        run_sweeps_and_cleanup(&mut state, &options, &mut identity, &mut rng, &mut exec).unwrap();

    assert_eq!(history.termination, TreeAciTermination::Converged);
    assert_eq!(expansion_debug_log::take().len(), 10);
    assert_eq!(
        unit_junction::centre_ranks(&state.output).unwrap(),
        [12, 1, 12]
    );
    let error = unit_junction::dense_relative_error(&state.output, &input).unwrap();
    assert!(error <= 1e-12, "dense relative error {error:e}");
    let deviation = pivot_basis_deviation(&state);
    assert!(deviation <= 1e-12, "returned max|M - I| = {deviation:e}");
}
