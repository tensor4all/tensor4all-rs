//! `TreeRsiInterpolator`, tree RSI behind the element-wise engine contract
//! of `tensor4all_treetn::interpolation`, for products.
//!
//! The engine runs the session entry with the problem's cap and seed: on a
//! binary tree whose node 1 has degree three its network equals that entry's
//! bit for bit, while the process-global engine is borrowed, and matches the
//! dense product. The remaining tests cover the cap, a problem without a
//! cap, and each class of failure.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArray, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{explicit::Session, with_default_backend, CpuBackend};
use tensor4all_treersi::{
    hadamard_many_with_rng_on_session, TreeRsiError, TreeRsiInterpolator, TreeRsiOptions,
    TreeRsiProbes,
};
use tensor4all_treetn::interpolation::{
    ElementwiseInterpolator, EvalContext, InterpolationError, InterpolationOutcome,
    InterpolationProblem, InterpolationTermination, Parallelism, Product,
};
use tensor4all_treetn::TreeTN;

use crate::support::{random_cores, tree_from_cores};

type Tree = TreeTN<IdxTensor, usize>;

/// Node 1 has degree three.
const BINARY: &[(usize, usize)] = &[(0, 1), (0, 2), (1, 3), (1, 4)];

/// Two bond-dimension-2 operands on `BINARY` with ternary sites, as compact
/// host storage.
fn operands(seed: u64) -> (Vec<DynIndex>, Vec<Tree>) {
    let sites: Vec<DynIndex> = (0..5).map(|_| DynIndex::new_dyn(3)).collect();
    let trees = (0..2)
        .map(|a| {
            tree_from_cores(
                random_cores(&sites, BINARY, 2, seed + a, |re, _| re),
                |indices, data| IdxTensor::from_dense_compact(indices, data).unwrap(),
            )
        })
        .collect();
    (sites, trees)
}

/// The problem over the sites and topology of `input`.
fn problem_of(input: &Tree, cap: Option<usize>, seed: u64) -> InterpolationProblem<usize> {
    let node_sites: BTreeMap<usize, Vec<DynIndex>> = input
        .node_names()
        .into_iter()
        .map(|node| {
            let sites = input.node_site_indices(&node).unwrap();
            (node, sites)
        })
        .collect();
    let n_sites = node_sites.values().map(Vec::len).sum();
    InterpolationProblem::new(
        input.site_index_network().topology().clone(),
        node_sites,
        ColMajorArray::new(vec![0; n_sites], vec![n_sites, 1]).unwrap(),
        1e-12,
        cap.map(|cap| NonZeroUsize::new(cap).unwrap()),
        seed,
    )
    .unwrap()
}

fn run(
    engine: &TreeRsiInterpolator,
    problem: &InterpolationProblem<usize>,
    inputs: &[Tree],
) -> Result<InterpolationOutcome<usize>, InterpolationError> {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
    ElementwiseInterpolator::<f64, _>::interpolate_elementwise(
        engine, problem, inputs, &Product, &mut ctx,
    )
}

fn dense(tree: &Tree, sites: &[DynIndex]) -> Vec<f64> {
    tree.to_dense()
        .unwrap()
        .permute_indices(sites)
        .unwrap()
        .dense_host_values::<f64>()
        .unwrap()
}

#[test]
fn the_engine_runs_the_session_entry_with_the_problems_settings() {
    let (sites, inputs) = operands(40);
    let problem = problem_of(&inputs[0], Some(6), 11);
    // The engine's own cap and seed are replaced by the problem's.
    let options = TreeRsiOptions::<usize> {
        rel_tol: 1e-13,
        ..Default::default()
    };
    let engine = TreeRsiInterpolator::new(TreeRsiOptions {
        max_bond_dim: Some(1),
        seed: 99,
        ..options.clone()
    })
    .unwrap();

    let mut backend = CpuBackend::with_threads(1).unwrap();
    let outcome = with_default_backend(|_global| {
        let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
        ElementwiseInterpolator::<f64, _>::interpolate_elementwise(
            &engine, &problem, &inputs, &Product, &mut ctx,
        )
        .expect("the engine must not reach the borrowed global engine")
    });

    let direct = {
        let mut backend = CpuBackend::with_threads(1).unwrap();
        let mut session = Session::new(backend.engine_mut());
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        hadamard_many_with_rng_on_session::<f64, _, _>(
            &inputs,
            &TreeRsiOptions {
                max_bond_dim: Some(6),
                seed: 11,
                ..options
            },
            &mut rng,
            &mut session,
        )
        .unwrap()
    };
    assert_eq!(dense(&outcome.network, &sites), dense(&direct.tree, &sites));
    // The product of two bond-2 operands has rank at most 4, below the cap.
    assert!(outcome.network.link_dims().iter().all(|&rank| rank <= 4));
    assert_eq!(
        outcome.termination,
        InterpolationTermination::ErrorNotMeasured
    );
    assert_eq!(outcome.error_estimate, f64::INFINITY);
    assert_eq!(outcome.max_sample_magnitude, 0.0);
    assert!(outcome.pivots.is_none());

    // Below the cap the product is exact to rounding.
    let (x, y) = (dense(&inputs[0], &sites), dense(&inputs[1], &sites));
    let expected: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a * b).collect();
    let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let error = dense(&outcome.network, &sites)
        .iter()
        .zip(&expected)
        .fold(0.0_f64, |m, (a, e)| m.max((a - e).abs()));
    assert!(error <= 1e-10 * scale, "max error {error:e}");
}

#[test]
fn a_rank_at_the_cap_is_reported_as_bond_cap_reached() {
    let (_sites, inputs) = operands(50);
    let outcome = run(
        &TreeRsiInterpolator::default(),
        &problem_of(&inputs[0], Some(2), 0),
        &inputs,
    )
    .unwrap();
    assert_eq!(
        outcome.termination,
        InterpolationTermination::BondCapReached
    );
    assert!(outcome.network.link_dims().iter().all(|&rank| rank <= 2));
}

#[test]
fn a_single_node_is_never_at_the_cap() {
    let site = DynIndex::new_dyn(3);
    let input = |values: Vec<f64>| {
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense_compact(vec![site.clone()], values).unwrap()],
            vec![0usize],
        )
        .unwrap()
    };
    let inputs = [input(vec![1.0, 2.0, 3.0]), input(vec![4.0, -1.0, 0.5])];
    let outcome = run(
        &TreeRsiInterpolator::default(),
        &problem_of(&inputs[0], Some(1), 0),
        &inputs,
    )
    .unwrap();
    assert_eq!(
        outcome.termination,
        InterpolationTermination::ErrorNotMeasured
    );
    assert_eq!(dense(&outcome.network, &[site]), vec![4.0, -2.0, 1.5]);
}

#[test]
fn a_problem_without_a_cap_needs_the_engines_sketch_width() {
    let (_sites, inputs) = operands(60);
    let uncapped = problem_of(&inputs[0], None, 0);
    let error = run(&TreeRsiInterpolator::default(), &uncapped, &inputs).unwrap_err();
    assert!(
        matches!(error, InterpolationError::InvalidProblem { .. }),
        "{error:?}"
    );
    let sketched = TreeRsiInterpolator::new(TreeRsiOptions::<usize> {
        sketch_dim: Some(8),
        ..Default::default()
    })
    .unwrap();
    let outcome = run(&sketched, &uncapped, &inputs).unwrap();
    assert_eq!(
        outcome.termination,
        InterpolationTermination::ErrorNotMeasured
    );
}

#[test]
fn invalid_problems_are_rejected() {
    let (_sites, inputs) = operands(70);
    let (_other_sites, other) = operands(80);
    let engine = TreeRsiInterpolator::default();
    let problem = problem_of(&inputs[0], Some(6), 0);
    let invalid = |result: Result<InterpolationOutcome<usize>, InterpolationError>| {
        matches!(result, Err(InterpolationError::InvalidProblem { .. }))
    };
    // No inputs.
    assert!(invalid(run(&engine, &problem, &[])));
    // Inputs over other sites than the problem's.
    let foreign = problem_of(&other[0], Some(6), 0);
    assert!(invalid(run(&engine, &foreign, &inputs)));
    // A second input that disagrees with the first.
    let mismatched = [inputs[0].clone(), other[0].clone()];
    assert!(invalid(run(&engine, &problem, &mismatched)));
}

#[test]
fn options_that_name_nodes_or_fail_validation_are_rejected() {
    let probes = TreeRsiOptions::<usize> {
        probes: Some(TreeRsiProbes::new(Vec::new())),
        ..Default::default()
    };
    assert!(matches!(
        TreeRsiInterpolator::new(probes),
        Err(TreeRsiError::InvalidOption {
            option: "probes",
            ..
        })
    ));
    let negative = TreeRsiOptions::<usize> {
        rel_tol: -1.0,
        ..Default::default()
    };
    assert!(matches!(
        TreeRsiInterpolator::new(negative),
        Err(TreeRsiError::InvalidOption {
            option: "rel_tol",
            ..
        })
    ));
    // The cap comes from each problem, so an engine without one is valid.
    assert!(TreeRsiInterpolator::new(TreeRsiOptions::<usize>::default()).is_ok());
}

/// The positive counterpart of the `compile_fail` example on
/// `TreeRsiInterpolator`: the same generic call compiles with `Product`, so
/// that example fails only because its operator is not a product.
#[test]
fn the_product_operator_satisfies_the_engine_bound() {
    fn engine<Op>(_: &impl ElementwiseInterpolator<f64, Op>, _: &Op) {}
    engine(&TreeRsiInterpolator::default(), &Product);
}
