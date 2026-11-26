//! Two bonds through a vertex of local dimension 1 must not cap each other
//! (#1435, mechanism 1).
//!
//! The fixture (`support/unit_junction.rs`) joins three branches at a centre
//! whose only site has dimension 1. Each commit of a bond into the centre read
//! as candidates only the product of the other two bonds' candidates, so the x
//! and z bonds could not grow past each other: every update recorded a
//! residual of ~1e-16 while the output stayed wrong. The true centre ranks are
//! (12, 1, 12).
//!
//! The target is 0 off the plane `y = 16`. A run whose first Guard searches
//! draw no start near the plane also stopped with the zero output (#1435,
//! mechanism 2), until the third Guard stage (#1466) enumerated the branches
//! around the searches' starts.

use tensor4all_treeaci::{tree_elementwise, TreeAciOptions, TreeAciTermination};

use crate::unit_junction;

/// Runs the identity on the fixture with tolerance 1e-10 and `rng_seed`.
fn run(rng_seed: u64) -> (tensor4all_treeaci::TreeAciResult<usize>, f64) {
    let input = unit_junction::input().unwrap();
    let options = TreeAciOptions {
        tolerance: 1e-10,
        rng_seed,
        ..TreeAciOptions::default()
    };
    let result =
        tree_elementwise::<f64, _, _>(|v| v[0], std::slice::from_ref(&input), &options).unwrap();
    let error = unit_junction::dense_relative_error(&result.tree, &input).unwrap();
    (result, error)
}

/// Asserts that `rng_seed` stops `Converged` after `passes` passes at the true
/// centre ranks with the exact tree.
fn assert_exact(rng_seed: u64, passes: usize) {
    let (result, error) = run(rng_seed);
    let observed = (
        result.termination,
        result.max_ranks.len(),
        unit_junction::centre_ranks(&result.tree).unwrap(),
        error <= 1e-12,
    );
    assert_eq!(
        observed,
        (TreeAciTermination::Converged, passes, [12, 1, 12], true),
        "(termination, passes, centre ranks, exact): dense relative error {error:e}, \
         recorded errors {:?}",
        result.max_errors
    );
}

/// Before the expansion: `Converged` after 8 passes at centre ranks (8, 1, 8)
/// with a dense relative error of 6.1e-6.
#[test]
fn unit_junction_seed_1_reaches_the_true_centre_ranks() {
    assert_exact(1, 6);
}

/// Before the expansion: `Converged` after 7 passes at centre ranks (2, 1, 2)
/// with a dense relative error of 0.33. With the expansion and before the
/// third Guard stage (#1466): 8 passes, the same exact tree.
#[test]
fn unit_junction_seed_4_reaches_the_true_centre_ranks() {
    assert_exact(4, 6);
}

/// The zero-output case (#1435 mechanism 2, #1466). Before the third Guard
/// stage this seed stopped `Converged` after 2 passes with the zero output
/// (centre ranks (1, 1, 1), dense relative error 1): every start of the first
/// two Guard searches lies two or more `y` bits off the plane `y = 16`, where
/// every residual of the zero output is exactly 0 and no walk moves.
#[test]
fn unit_junction_seed_2_leaves_the_zero_output() {
    assert_exact(2, 6);
}
