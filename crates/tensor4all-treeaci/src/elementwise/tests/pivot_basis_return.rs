//! The tree a run returns keeps the pivot basis (#1450).
//!
//! Every bond of a returned tree has one basis shared by both adjacent cores,
//! with bond axis `k` the `k`-th retained candidate. The check
//! ([`crate::test_support::pivot_basis_deviation`]) evaluates every retained
//! candidate, so it is a test, not a library assertion.

#[path = "../../../tests/support/g0.rs"]
mod g0;

use rand::SeedableRng;

use super::super::run_sweeps_and_cleanup;
use crate::{
    exec::Exec, state::TreeAciState, test_support::pivot_basis_deviation, tree_elementwise_batched,
    TreeAciTermination,
};

/// G0 r = 5 on the comb tree, seed 42, tolerance 1e-3, one pass: the Guard
/// injects after pass 1, so the run ends with the return cleanup pass. Before
/// #1450 that pass re-committed only the spine, and the returned tree had
/// max|M - I| = 0.76.
#[test]
fn one_pass_cleanup_returns_a_tree_in_the_pivot_basis() -> g0::TestResult<()> {
    let (_, inputs) = g0::fixture(5, "cttn")?;
    let mut options = g0::options();
    options.rng_seed = 42;
    options.tolerance = 1e-3;
    options.min_sweeps = 1;
    options.max_sweeps = 1;

    // The same run as `tree_elementwise_batched`, keeping the state.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(options.rng_seed);
    let mut exec = Exec::Global;
    let mut state = TreeAciState::initialize_with_rng(&inputs, &options, &mut rng, &mut exec)?;
    let mut operator = g0::operator;
    let (history, evaluated_points) =
        run_sweeps_and_cleanup(&mut state, &options, &mut operator, &mut rng, &mut exec)?;
    let public = tree_elementwise_batched(g0::operator, &inputs, &options)?;
    assert_eq!(history.termination, public.termination);
    assert_eq!(history.max_ranks, public.max_ranks);
    assert_eq!(evaluated_points, public.diagnostics.evaluated_points);
    assert_eq!(state.output.link_dims(), public.tree.link_dims());

    // The Guard found pivots after the only pass, so the cleanup ran.
    assert_eq!(history.termination, TreeAciTermination::MaxSweeps);
    assert!(history.needs_cleanup);
    let deviation = pivot_basis_deviation(&state);
    assert!(
        deviation <= 1e-12,
        "returned max|M - I| = {deviation:e} (termination {:?})",
        history.termination
    );
    Ok(())
}
