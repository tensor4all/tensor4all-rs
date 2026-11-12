//! `TreeAciInterpolator`, tree ACI behind the element-wise engine contract
//! of `tensor4all_treetn::interpolation`.
//!
//! The engine runs the session entry with the problem's tolerance divided by
//! its tolerance factor, cap and seed: on the low-temperature G0 fixture (a CTTN whose junction has degree
//! three) its network equals that entry's bit for bit and meets the exact
//! function, while the process-global engine is borrowed. The remaining
//! tests cover the mapping of termination, error estimate and sample scale,
//! and each class of failure.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use crate::g0;

use num_complex::Complex64;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{explicit::Session, with_default_backend, CpuBackend};
use tensor4all_treeaci::{
    tree_elementwise_batched_with_rng_on_session, TreeAciInterpolator, TreeAciOptions,
};
use tensor4all_treetn::interpolation::{
    ElementwiseInterpolator, ElementwiseOperator, EvalContext, InterpolationError,
    InterpolationOutcome, InterpolationProblem, InterpolationTermination, Parallelism, Product,
};
use tensor4all_treetn::TreeTN;

/// The problem over the sites and topology of `input`.
fn problem_of(
    input: &TreeTN<IdxTensor, usize>,
    tolerance: f64,
    cap: Option<usize>,
    seed: u64,
) -> InterpolationProblem<usize> {
    let topology = input.site_index_network().topology().clone();
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
        topology,
        node_sites,
        ColMajorArray::new(vec![0; n_sites], vec![n_sites, 1]).unwrap(),
        tolerance,
        cap.map(|cap| NonZeroUsize::new(cap).unwrap()),
        seed,
    )
    .unwrap()
}

/// Run `engine` on a fresh single-thread session.
fn run<T, Op>(
    engine: &TreeAciInterpolator,
    problem: &InterpolationProblem<usize>,
    inputs: &[TreeTN<IdxTensor, usize>],
    operator: &Op,
) -> Result<InterpolationOutcome<usize>, InterpolationError>
where
    T: tensor4all_treeaci::TreeAciScalar,
    Op: ElementwiseOperator<T>,
{
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
    engine.interpolate_elementwise(problem, inputs, operator, &mut ctx)
}

/// The G0 operator `1 / (1/2 + 2 a + 2 b + i c - d)` on a column-major
/// `[4, n_points]` batch.
fn g0_map(inputs: ColMajorArrayRef<'_, Complex64>, output: &mut [Complex64]) -> anyhow::Result<()> {
    for (column, value) in inputs.data().chunks(4).zip(output.iter_mut()) {
        *value = 1.0
            / (Complex64::new(0.5, 0.0)
                + 2.0 * column[0]
                + 2.0 * column[1]
                + Complex64::i() * column[2]
                - column[3]);
    }
    Ok(())
}

/// The exact G0 values on the fixture's sites, in site order.
fn g0_exact(sites: &[DynIndex], r: usize) -> IdxTensor {
    let values: Vec<Complex64> = (0..1usize << (3 * r))
        .map(|flat| {
            let mut point = [0; 3];
            for bit in 0..r {
                for (var, x) in point.iter_mut().enumerate() {
                    *x = 2 * *x + ((flat >> (3 * bit + var)) & 1);
                }
            }
            g0::exact(r, point[0], point[1], point[2])
        })
        .collect();
    IdxTensor::from_dense(sites.to_vec(), values).unwrap()
}

#[test]
fn the_engine_runs_the_session_entry_with_the_problems_settings() {
    let r = 3;
    let (sites, inputs) = g0::fixture(r, "cttn").unwrap();
    let options = g0::options();
    let tolerance = options.tolerance;
    let problem = problem_of(&inputs[0], tolerance, Some(64), 11);
    // The engine's own tolerance, cap, scaling and seed are all replaced.
    let engine = TreeAciInterpolator::new(TreeAciOptions {
        tolerance: 0.5,
        scale_tolerance: true,
        max_bond_dim: Some(1),
        rng_seed: 99,
        ..options.clone()
    })
    .unwrap();

    let mut backend = CpuBackend::with_threads(1).unwrap();
    let outcome = with_default_backend(|_global| {
        let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
        engine
            .interpolate_elementwise(&problem, &inputs, &g0_map, &mut ctx)
            .expect("the engine must not reach the borrowed global engine")
    });

    // The session entry with the problem's settings, called directly at the
    // default tolerance factor of 3.
    assert_eq!(engine.tolerance_factor(), 3.0);
    let direct = {
        let mut backend = CpuBackend::with_threads(1).unwrap();
        let mut session = Session::new(backend.engine_mut());
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        tree_elementwise_batched_with_rng_on_session(
            g0::operator,
            &inputs,
            &TreeAciOptions {
                tolerance: tolerance / 3.0,
                scale_tolerance: false,
                max_bond_dim: Some(64),
                rng_seed: 11,
                ..options.clone()
            },
            &mut rng,
            &mut session,
        )
        .unwrap()
    };
    let dense = |tree: &TreeTN<IdxTensor, usize>| {
        tree.to_dense()
            .unwrap()
            .permute_indices(&sites)
            .unwrap()
            .to_vec::<Complex64>()
            .unwrap()
    };
    assert_eq!(dense(&outcome.network), dense(&direct.tree));
    assert_eq!(outcome.termination, InterpolationTermination::Converged);
    assert_eq!(outcome.error_estimate, *direct.max_errors.last().unwrap());
    assert!(outcome.error_estimate <= tolerance);
    assert_eq!(outcome.max_sample_magnitude, direct.max_sample_magnitude);
    assert!(outcome.pivots.is_none());

    // The network meets the exact function within the guard's margin.
    let exact = g0_exact(&sites, r);
    let error = outcome.network.to_dense().unwrap().sub(&exact).unwrap();
    let error = error.maxabs().unwrap();
    assert!(
        error <= tolerance * options.global_tolerance_margin,
        "max absolute residual {error:e}"
    );
    // Samples are values of f, so their scale is at most the largest |f|.
    assert!(outcome.max_sample_magnitude <= exact.maxabs().unwrap() * (1.0 + 1e-12));
}

/// With a tolerance factor of one the engine runs the session entry at the
/// problem's tolerance itself, bit for bit.
#[test]
fn a_tolerance_factor_of_one_runs_the_session_entry_at_the_problems_tolerance() {
    let r = 3;
    let (sites, inputs) = g0::fixture(r, "cttn").unwrap();
    let options = g0::options();
    let tolerance = options.tolerance;
    let problem = problem_of(&inputs[0], tolerance, Some(64), 5);
    let engine = TreeAciInterpolator::new(options.clone())
        .unwrap()
        .with_tolerance_factor(1.0)
        .unwrap();
    let outcome = run(&engine, &problem, &inputs, &g0_map).unwrap();

    let direct = {
        let mut backend = CpuBackend::with_threads(1).unwrap();
        let mut session = Session::new(backend.engine_mut());
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        tree_elementwise_batched_with_rng_on_session(
            g0::operator,
            &inputs,
            &TreeAciOptions {
                tolerance,
                scale_tolerance: false,
                max_bond_dim: Some(64),
                rng_seed: 5,
                ..options.clone()
            },
            &mut rng,
            &mut session,
        )
        .unwrap()
    };
    let dense = |tree: &TreeTN<IdxTensor, usize>| {
        tree.to_dense()
            .unwrap()
            .permute_indices(&sites)
            .unwrap()
            .to_vec::<Complex64>()
            .unwrap()
    };
    assert_eq!(dense(&outcome.network), dense(&direct.tree));
    assert_eq!(outcome.termination, InterpolationTermination::Converged);
    assert_eq!(outcome.error_estimate, *direct.max_errors.last().unwrap());
    assert!(outcome.error_estimate <= tolerance);
}

/// A single node is evaluated exactly: its sample scale is the largest |f|
/// on the grid and its error estimate is zero.
#[test]
fn a_single_node_is_exact_and_reports_its_largest_value() {
    let site = DynIndex::new_dyn(4);
    let input = |values: Vec<f64>| {
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
            vec![0usize],
        )
        .unwrap()
    };
    let inputs = [
        input(vec![1.0, -2.0, 3.0, 0.5]),
        input(vec![2.0, 4.0, -1.0, 6.0]),
    ];
    let problem = problem_of(&inputs[0], 1e-12, Some(4), 0);
    let outcome =
        run::<f64, _>(&TreeAciInterpolator::default(), &problem, &inputs, &Product).unwrap();
    assert_eq!(outcome.termination, InterpolationTermination::Converged);
    assert_eq!(outcome.error_estimate, 0.0);
    assert_eq!(outcome.max_sample_magnitude, 8.0);
    assert_eq!(
        outcome.network.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0, -8.0, -3.0, 3.0]
    );
}

/// Two sites on a path, `x(a, b) = sum_k u_k(a) v_k(b)` of rank two.
fn rank_two_inputs() -> Vec<TreeTN<IdxTensor, usize>> {
    let (a, b) = (DynIndex::new_dyn(3), DynIndex::new_dyn(3));
    let input = |left: Vec<f64>, right: Vec<f64>| {
        let bond = DynIndex::new_dyn(2);
        TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(vec![a.clone(), bond.clone()], left).unwrap(),
                IdxTensor::from_dense(vec![bond, b.clone()], right).unwrap(),
            ],
            vec![0usize, 1],
        )
        .unwrap()
    };
    vec![
        input(
            vec![1.0, 2.0, 0.5, -1.0, 0.25, 3.0],
            vec![1.0, -0.5, 2.0, 0.75, 1.5, -2.0],
        ),
        input(
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ),
    ]
}

#[test]
fn a_rank_at_the_cap_is_reported_as_bond_cap_reached() {
    // The second input is all ones, so the product is the rank-two first one.
    let inputs = rank_two_inputs();
    let uncapped = run::<f64, _>(
        &TreeAciInterpolator::default(),
        &problem_of(&inputs[0], 1e-12, None, 0),
        &inputs,
        &Product,
    )
    .unwrap();
    assert_eq!(uncapped.termination, InterpolationTermination::Converged);
    assert_eq!(uncapped.network.link_dims(), vec![2]);

    // Converged at a rank equal to the cap is not `Converged`.
    let at_cap = run::<f64, _>(
        &TreeAciInterpolator::default(),
        &problem_of(&inputs[0], 1e-12, Some(2), 0),
        &inputs,
        &Product,
    )
    .unwrap();
    assert_eq!(at_cap.termination, InterpolationTermination::BondCapReached);
    assert_eq!(at_cap.network.link_dims(), vec![2]);

    // Below the rank, tree ACI is rank limited and the network obeys the cap.
    let below = run::<f64, _>(
        &TreeAciInterpolator::default(),
        &problem_of(&inputs[0], 1e-12, Some(1), 0),
        &inputs,
        &Product,
    )
    .unwrap();
    assert_eq!(below.termination, InterpolationTermination::BondCapReached);
    assert_eq!(below.network.link_dims(), vec![1]);
}

#[test]
fn the_sweep_limit_is_reported_as_an_iteration_limit() {
    let (_sites, inputs) = g0::fixture(3, "cttn").unwrap();
    let engine = TreeAciInterpolator::new(TreeAciOptions::<usize> {
        max_sweeps: 1,
        min_sweeps: 1,
        ..g0::options()
    })
    .unwrap();
    let problem = problem_of(&inputs[0], g0::options().tolerance, None, 0);
    let outcome = run(&engine, &problem, &inputs, &g0_map).unwrap();
    assert_eq!(
        outcome.termination,
        InterpolationTermination::IterationLimit
    );
}

#[derive(Debug, thiserror::Error)]
#[error("out of domain at the requested point")]
struct OutOfDomain;

#[test]
fn an_operator_failure_is_an_evaluator_error_with_its_source() {
    let inputs = rank_two_inputs();
    let problem = problem_of(&inputs[0], 1e-12, None, 0);
    let failing = |_: ColMajorArrayRef<'_, f64>, _: &mut [f64]| -> anyhow::Result<()> {
        Err(OutOfDomain.into())
    };
    let error = run(&TreeAciInterpolator::default(), &problem, &inputs, &failing).unwrap_err();
    let InterpolationError::Evaluator { source } = error else {
        panic!("expected an evaluator error, got {error:?}");
    };
    assert!(source.downcast_ref::<OutOfDomain>().is_some());

    let infinite = |_: ColMajorArrayRef<'_, f64>, output: &mut [f64]| -> anyhow::Result<()> {
        output.fill(f64::INFINITY);
        Ok(())
    };
    let error = run(
        &TreeAciInterpolator::default(),
        &problem,
        &inputs,
        &infinite,
    )
    .unwrap_err();
    assert!(
        matches!(error, InterpolationError::Evaluator { .. }),
        "{error:?}"
    );
}

#[test]
fn invalid_problems_are_rejected() {
    let inputs = rank_two_inputs();
    let engine = TreeAciInterpolator::default();
    let invalid = |result: Result<InterpolationOutcome<usize>, InterpolationError>| {
        matches!(result, Err(InterpolationError::InvalidProblem { .. }))
    };
    // No inputs.
    let problem = problem_of(&inputs[0], 1e-12, None, 0);
    assert!(invalid(run::<f64, _>(&engine, &problem, &[], &Product)));
    // A zero tolerance: tree ACI's criterion needs a positive one.
    let exact = problem_of(&inputs[0], 0.0, None, 0);
    assert!(invalid(run::<f64, _>(&engine, &exact, &inputs, &Product)));
    // Inputs over other sites than the problem's.
    let (_sites, other) = g0::fixture(2, "cttn").unwrap();
    let foreign = problem_of(&other[0], 1e-12, None, 0);
    assert!(invalid(run::<f64, _>(&engine, &foreign, &inputs, &Product)));
    // A second input that disagrees with the first.
    let mismatched = [inputs[0].clone(), other[0].clone()];
    assert!(invalid(run::<f64, _>(
        &engine,
        &problem,
        &mismatched,
        &Product
    )));
}

#[test]
fn options_that_name_nodes_or_fail_validation_are_rejected() {
    use tensor4all_treeaci::TreeAciError;

    let guessed = TreeAciOptions {
        initial_guess: Some(rank_two_inputs().remove(0)),
        ..Default::default()
    };
    assert!(matches!(
        TreeAciInterpolator::new(guessed),
        Err(TreeAciError::InvalidOption {
            option: "initial_guess",
            ..
        })
    ));
    let no_sweeps = TreeAciOptions::<usize> {
        min_sweeps: 0,
        ..Default::default()
    };
    assert!(matches!(
        TreeAciInterpolator::new(no_sweeps),
        Err(TreeAciError::InvalidOption { .. })
    ));
}
