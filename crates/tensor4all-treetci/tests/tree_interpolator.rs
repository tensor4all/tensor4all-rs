//! `TreeTciInterpolator` through the engine-independent contract of
//! `tensor4all_treetn::interpolation`, and a test-only mock engine through the
//! same generic helper.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;

use num_complex::Complex64;
use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, Index, IndexLike, TagSet,
    TensorElement,
};
use tensor4all_treetci::{TreeTciInterpolator, TreeTciOptions};
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationOutcome, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};
use tensor4all_treetn::{factorize_tensor_to_treetn, NodeNameNetwork, TreeTopology};

type Name = String;

fn name(node: &str) -> Name {
    node.to_string()
}

fn topology(nodes: &[&str], edges: &[(&str, &str)]) -> NodeNameNetwork<Name> {
    let mut network = NodeNameNetwork::new();
    for node in nodes {
        network.add_node(name(node)).unwrap();
    }
    for (a, b) in edges {
        network.add_edge(&name(a), &name(b)).unwrap();
    }
    network
}

fn topology_edges(topology: &NodeNameNetwork<Name>) -> Vec<(Name, Name)> {
    let graph = topology.graph();
    graph
        .edge_indices()
        .map(|edge| {
            let (a, b) = graph.edge_endpoints(edge).unwrap();
            (
                topology.node_name(a).unwrap().clone(),
                topology.node_name(b).unwrap().clone(),
            )
        })
        .collect()
}

/// Every point of the site domain, column-major (first site fastest), as a
/// `[n_sites, n_points]` buffer.
fn full_domain(dims: &[usize]) -> Vec<usize> {
    let n_points: usize = dims.iter().product();
    let mut points = Vec::with_capacity(n_points * dims.len());
    for mut linear in 0..n_points {
        for &dim in dims {
            points.push(linear % dim);
            linear /= dim;
        }
    }
    points
}

fn site_dims(problem: &InterpolationProblem<Name>) -> Vec<usize> {
    problem.site_order().iter().map(|site| site.dim()).collect()
}

/// Batch evaluator for a point function; rejects malformed batches so an
/// engine that sends one fails the test through `Evaluator`.
fn evaluator<'f, T>(
    f: &'f dyn Fn(&[usize]) -> T,
    dims: Vec<usize>,
) -> impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> + 'f {
    move |batch| {
        anyhow::ensure!(
            batch.ndim() == 2 && batch.shape()[0] == dims.len(),
            "batch shape {:?} does not match {} sites",
            batch.shape(),
            dims.len()
        );
        batch
            .data()
            .chunks(dims.len())
            .map(|point| {
                anyhow::ensure!(
                    point.iter().zip(&dims).all(|(&value, &dim)| value < dim),
                    "point {point:?} out of range for {dims:?}"
                );
                Ok(f(point))
            })
            .collect()
    }
}

fn pivots_from(points: &[Vec<usize>]) -> ColMajorArray<usize> {
    let n_sites = points[0].len();
    ColMajorArray::new(points.concat(), vec![n_sites, points.len()]).unwrap()
}

/// Checks shared by every engine and every run, whatever the verdict.
struct Checked {
    outcome: InterpolationOutcome<Name>,
    /// Max-norm residual against the dense reference.
    residual: f64,
    /// Largest magnitude of the function on the domain.
    max_abs: f64,
}

/// Run `engine` on `problem` and check the contract: node names, topology,
/// and site identities equal the problem's; returned pivots are distinct,
/// in-range full-domain points; the reported maximum sample is attained by the
/// function. The whole result is materialized once and compared with the dense
/// reference; the residual is returned for the caller to judge.
fn run_and_check<T, E>(
    engine: &E,
    problem: &InterpolationProblem<Name>,
    f: &dyn Fn(&[usize]) -> T,
) -> Checked
where
    T: TensorElement + CommonScalar,
    E: TreeInterpolator<T>,
{
    let dims = site_dims(problem);
    let outcome = engine
        .interpolate(problem, evaluator(f, dims.clone()))
        .unwrap();
    let network = &outcome.network;

    let mut names = network.node_names();
    names.sort();
    assert_eq!(
        names,
        problem.node_sites().keys().cloned().collect::<Vec<_>>()
    );
    assert_eq!(network.edge_count(), problem.topology().edge_count());
    for (a, b) in topology_edges(problem.topology()) {
        assert!(
            network.edge_between(&a, &b).is_some(),
            "missing edge {a}-{b}"
        );
    }
    for (node, sites) in problem.node_sites() {
        let expected: HashSet<DynIndex> = sites.iter().cloned().collect();
        let actual = network.site_space(node).cloned().unwrap_or_default();
        assert_eq!(actual, expected, "site space of node {node}");
    }

    let domain = full_domain(&dims);
    let values: Vec<T> = domain.chunks(dims.len()).map(f).collect();
    let max_abs = values
        .iter()
        .map(|value| CommonScalar::abs_val(*value))
        .fold(0.0_f64, f64::max);
    let reference = IdxTensor::from_dense(problem.site_order().to_vec(), values).unwrap();
    let dense = network.to_dense().unwrap();
    let residual = dense.sub(&reference).unwrap().maxabs().unwrap();

    assert!(outcome.max_sample_magnitude > 0.0);
    assert!(
        outcome.max_sample_magnitude <= max_abs * (1.0 + 1e-14),
        "max sample {} exceeds max |f| {max_abs}",
        outcome.max_sample_magnitude
    );
    assert!(outcome.error_estimate >= 0.0);

    if let Some(pivots) = &outcome.pivots {
        assert_eq!(pivots.nrows(), Some(dims.len()));
        let n_pivots = pivots.ncols().unwrap();
        assert!(n_pivots >= 1);
        let mut seen = HashSet::new();
        for column in 0..n_pivots {
            let point = pivots.column(column).unwrap();
            assert!(
                point.iter().zip(&dims).all(|(&value, &dim)| value < dim),
                "pivot {point:?} out of range for {dims:?}"
            );
            assert!(seen.insert(point.to_vec()), "duplicate pivot {point:?}");
        }
    }

    Checked {
        outcome,
        residual,
        max_abs,
    }
}

/// Chain n0 - n1 - n2 - n3 (max degree 2), one site per node.
fn chain_problem(tolerance: f64, cap: Option<usize>, seed: u64) -> InterpolationProblem<Name> {
    let dims = [2, 3, 2, 2];
    let names = ["n0", "n1", "n2", "n3"];
    let node_sites = names
        .iter()
        .zip(dims)
        .map(|(node, dim)| (name(node), vec![DynIndex::new_dyn(dim)]))
        .collect();
    InterpolationProblem::new(
        topology(&names, &[("n0", "n1"), ("n1", "n2"), ("n2", "n3")]),
        node_sites,
        pivots_from(&[vec![0, 0, 0, 0]]),
        tolerance,
        cap.and_then(NonZeroUsize::new),
        seed,
    )
    .unwrap()
}

fn chain_function(p: &[usize]) -> f64 {
    1.0 / (1.0 + 0.3 * p[0] as f64 + 0.5 * p[1] as f64 + 0.7 * p[2] as f64 + 0.2 * p[3] as f64)
}

/// Branched tree, max degree 3 at "c":
///
/// ```text
///   a - c - d - z
///       |
///       b - e
/// ```
///
/// "a" carries two sites (a fused vertex), "b" (internal) and "z" (leaf)
/// carry none. Site order: a0, a1, c0, d0, e0.
fn branched_problem(
    pivots: Option<ColMajorArray<usize>>,
    tolerance: f64,
    seed: u64,
) -> InterpolationProblem<Name> {
    let node_sites = BTreeMap::from([
        (name("a"), vec![DynIndex::new_dyn(2), DynIndex::new_dyn(3)]),
        (name("b"), vec![]),
        (name("c"), vec![DynIndex::new_dyn(2)]),
        (name("d"), vec![DynIndex::new_dyn(2)]),
        (name("e"), vec![DynIndex::new_dyn(3)]),
        (name("z"), vec![]),
    ]);
    InterpolationProblem::new(
        topology(
            &["a", "b", "c", "d", "e", "z"],
            &[("a", "c"), ("c", "b"), ("c", "d"), ("b", "e"), ("d", "z")],
        ),
        node_sites,
        pivots.unwrap_or_else(|| pivots_from(&[vec![0, 0, 0, 0, 0]])),
        tolerance,
        None,
        seed,
    )
    .unwrap()
}

/// Rank at most three across every cut.
fn branched_function(p: &[usize]) -> f64 {
    let phase = 0.5 * p[0] as f64
        + 0.2 * p[1] as f64
        + 0.3 * p[2] as f64
        + 0.7 * p[3] as f64
        + 0.4 * p[4] as f64;
    phase.cos() + 0.5
}

/// Star with center "c" of degree 3 (leaves "l0", "l1", "l2"), one site per
/// node. Site order: c, l0, l1, l2.
fn star_problem(cap: Option<NonZeroUsize>, seed: u64) -> InterpolationProblem<Name> {
    let node_sites: BTreeMap<Name, Vec<DynIndex>> = [("l0", 2), ("c", 4), ("l1", 4), ("l2", 4)]
        .iter()
        .map(|&(node, dim)| (name(node), vec![DynIndex::new_dyn(dim)]))
        .collect();
    InterpolationProblem::new(
        topology(
            &["l0", "c", "l1", "l2"],
            &[("l0", "c"), ("c", "l1"), ("c", "l2")],
        ),
        node_sites,
        pivots_from(&[vec![0, 0, 0, 0]]),
        1e-10,
        cap,
        seed,
    )
    .unwrap()
}

/// Rank three across every cut of the star.
fn star_function(p: &[usize]) -> f64 {
    (0.5 * p[0] as f64 + 0.3 * p[1] as f64 + 0.7 * p[2] as f64 + 0.9 * p[3] as f64).cos() + 0.1
}

/// Single node "only" with three fused sites.
fn single_node_problem(pivot: Vec<usize>) -> InterpolationProblem<Name> {
    let node_sites = BTreeMap::from([(
        name("only"),
        vec![
            DynIndex::new_dyn(2),
            DynIndex::new_dyn(3),
            DynIndex::new_dyn(2),
        ],
    )]);
    InterpolationProblem::new(
        topology(&["only"], &[]),
        node_sites,
        pivots_from(&[pivot]),
        1e-12,
        None,
        0,
    )
    .unwrap()
}

fn single_node_function(p: &[usize]) -> f64 {
    (p[0] + 2 * p[1]) as f64 - 3.0 * p[2] as f64
}

fn assert_accurate(checked: &Checked, relative: f64) {
    assert!(
        checked.residual <= relative * checked.max_abs,
        "residual {} exceeds {relative} * max |f| {}",
        checked.residual,
        checked.max_abs
    );
}

// ---------------------------------------------------------------------------
// A test-only mock engine: dense evaluation plus exact factorization. It uses
// only public treetn and core APIs, showing that a second engine needs no
// engine-specific code in the contract or the helper.
// ---------------------------------------------------------------------------

struct DenseMockEngine;

impl<T> TreeInterpolator<T> for DenseMockEngine
where
    T: TensorElement + CommonScalar,
{
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
    {
        let engine = |source: anyhow::Error| InterpolationError::Engine { source };
        let evaluator = |source: anyhow::Error| InterpolationError::Evaluator { source };
        let dims: Vec<usize> = problem.site_order().iter().map(|site| site.dim()).collect();

        let initial = evaluate(problem.initial_pivots().as_ref()).map_err(evaluator)?;
        if initial
            .iter()
            .all(|value| CommonScalar::abs_val(*value) == 0.0)
        {
            return Err(InterpolationError::AllSamplesZero);
        }

        let domain = full_domain(&dims);
        let shape = [dims.len(), domain.len() / dims.len()];
        let batch = ColMajorArrayRef::new(&domain, &shape).map_err(|e| engine(e.into()))?;
        let values = evaluate(batch).map_err(evaluator)?;
        let max_sample_magnitude = values
            .iter()
            .map(|value| CommonScalar::abs_val(*value))
            .fold(0.0_f64, f64::max);
        let dense = IdxTensor::from_dense(problem.site_order().to_vec(), values)
            .map_err(|e| engine(e.into()))?;

        let topology = problem.topology();
        let graph = topology.graph();
        let edges = graph
            .edge_indices()
            .filter_map(|edge| graph.edge_endpoints(edge))
            .filter_map(|(a, b)| {
                Some((
                    topology.node_name(a)?.clone(),
                    topology.node_name(b)?.clone(),
                ))
            })
            .collect();
        let nodes: HashMap<V, Vec<DynIndex>> = problem
            .node_sites()
            .iter()
            .map(|(node, sites)| (node.clone(), sites.clone()))
            .collect();
        let root = problem
            .node_sites()
            .keys()
            .next()
            .ok_or_else(|| engine(anyhow::anyhow!("empty problem")))?;
        let network = factorize_tensor_to_treetn(&dense, &TreeTopology::new(nodes, edges), root)
            .map_err(|e| engine(e.into()))?;
        Ok(InterpolationOutcome {
            network,
            termination: InterpolationTermination::Converged,
            error_estimate: 0.0,
            max_sample_magnitude,
            pivots: None,
        })
    }
}

#[test]
fn mock_engine_runs_through_the_same_helper() {
    let chain = run_and_check(
        &DenseMockEngine,
        &chain_problem(1e-12, None, 0),
        &chain_function,
    );
    assert_eq!(
        chain.outcome.termination,
        InterpolationTermination::Converged
    );
    assert_accurate(&chain, 1e-12);

    // The mock factorizes densely and needs a site on every node, so it runs
    // on the star (max degree 3) rather than on the tree with site-free nodes.
    let star = run_and_check(&DenseMockEngine, &star_problem(None, 0), &star_function);
    assert_accurate(&star, 1e-12);

    let single = run_and_check(
        &DenseMockEngine,
        &single_node_problem(vec![1, 1, 0]),
        &single_node_function,
    );
    assert_accurate(&single, 1e-12);
}

// ---------------------------------------------------------------------------
// TreeTciInterpolator
// ---------------------------------------------------------------------------

#[test]
fn chain_converges_to_dense_reference() {
    let checked = run_and_check(
        &TreeTciInterpolator::default(),
        &chain_problem(1e-12, None, 3),
        &chain_function,
    );
    assert_eq!(
        checked.outcome.termination,
        InterpolationTermination::Converged
    );
    assert!(checked.outcome.error_estimate < 1e-12);
    assert_accurate(&checked, 1e-10);
}

#[test]
fn branched_tree_with_fused_and_site_free_nodes_converges() {
    let checked = run_and_check(
        &TreeTciInterpolator::default(),
        &branched_problem(None, 1e-12, 5),
        &branched_function,
    );
    assert_eq!(
        checked.outcome.termination,
        InterpolationTermination::Converged
    );
    assert_accurate(&checked, 1e-10);
}

#[test]
fn branched_tree_converges_for_complex_scalars() {
    let f = |p: &[usize]| -> Complex64 {
        let phase = 0.5 * p[0] as f64
            + 0.2 * p[1] as f64
            + 0.3 * p[2] as f64
            + 0.7 * p[3] as f64
            + 0.4 * p[4] as f64;
        Complex64::from_polar(1.0, phase) + Complex64::new(0.5, -0.25)
    };
    let checked = run_and_check(
        &TreeTciInterpolator::default(),
        &branched_problem(None, 1e-12, 2),
        &f,
    );
    assert_eq!(
        checked.outcome.termination,
        InterpolationTermination::Converged
    );
    assert_accurate(&checked, 1e-10);
}

#[test]
fn single_node_is_evaluated_exactly() {
    let problem = single_node_problem(vec![1, 2, 0]);
    let checked = run_and_check(
        &TreeTciInterpolator::default(),
        &problem,
        &single_node_function,
    );
    assert_eq!(
        checked.outcome.termination,
        InterpolationTermination::Converged
    );
    assert_eq!(checked.outcome.error_estimate, 0.0);
    assert!(checked.outcome.pivots.is_none());
    // Largest magnitude over the full domain: f(1, 2, 0) = 5.
    assert_eq!(checked.outcome.max_sample_magnitude, 5.0);
    assert_eq!(checked.residual, 0.0);
}

#[test]
fn node_names_and_same_id_site_identities_are_preserved() {
    // Chain p - q - r (max degree 2). "p" fuses a site and its primed copy;
    // "q" carries the same ID with a tag; the three are distinct indices.
    let base = DynIndex::new_dyn(2);
    let primed = base.prime();
    let tagged = Index::new_with_tags(base.id, 2, TagSet::from_str("Site").unwrap());
    let other = DynIndex::new_dyn(3);
    let node_sites = BTreeMap::from([
        (name("p"), vec![base.clone(), primed.clone()]),
        (name("q"), vec![tagged.clone()]),
        (name("r"), vec![other.clone()]),
    ]);
    let problem = InterpolationProblem::new(
        topology(&["p", "q", "r"], &[("p", "q"), ("q", "r")]),
        node_sites,
        pivots_from(&[vec![0, 0, 0, 0]]),
        1e-12,
        None,
        9,
    )
    .unwrap();
    assert_eq!(problem.site_order(), &[base, primed, tagged, other][..]);

    // Not symmetric in the same-ID sites, so a swapped leg would show up.
    let f = |p: &[usize]| {
        1.0 + p[0] as f64 + 2.0 * p[1] as f64 * (1.0 + p[2] as f64) + 0.1 * (p[3] * p[3]) as f64
    };
    let checked = run_and_check(&TreeTciInterpolator::default(), &problem, &f);
    assert_eq!(
        checked.outcome.termination,
        InterpolationTermination::Converged
    );
    assert_accurate(&checked, 1e-10);
}

#[test]
fn criterion_met_at_the_cap_is_bond_cap_reached() {
    // Chain of three dimension-2 nodes (max degree 2). 1 + p0 + p1 + p2 has
    // rank 2 across both cuts, which is also the largest rank a cut of this
    // domain can have, so with a cap of 2 TreeTCI reaches the cap with a zero
    // bond error: the error criterion holds at a rank equal to the cap. The
    // saturation stop, which the loop checks before convergence, ends the run
    // (`MaxBondDimension`), mapped to `BondCapReached`. Because of that
    // precedence TreeTCI never reports `Converged` at the cap; the mapping's
    // guard for that case is unit-tested in `src/interpolator/tests.rs`.
    let node_sites: BTreeMap<Name, Vec<DynIndex>> = ["n0", "n1", "n2"]
        .iter()
        .map(|node| (name(node), vec![DynIndex::new_dyn(2)]))
        .collect();
    let problem = InterpolationProblem::new(
        topology(&["n0", "n1", "n2"], &[("n0", "n1"), ("n1", "n2")]),
        node_sites,
        pivots_from(&[vec![0, 0, 0]]),
        1e-12,
        NonZeroUsize::new(2),
        0,
    )
    .unwrap();
    let f = |p: &[usize]| (1 + p[0] + p[1] + p[2]) as f64;
    let checked = run_and_check(&TreeTciInterpolator::default(), &problem, &f);
    assert_eq!(
        checked.outcome.termination,
        InterpolationTermination::BondCapReached
    );
    assert!(checked.outcome.error_estimate < problem.absolute_tolerance());
    assert_eq!(checked.outcome.network.link_dims(), vec![2, 2]);
    // The cap is sufficient here, so the result is also accurate.
    assert_accurate(&checked, 1e-10);
}

#[test]
fn capped_branched_tree_stops_cleanly_at_the_cap() {
    // Star with center "c" of degree 3; the function has rank 3 across every
    // cut, so a cap of 2 is insufficient. Only a clean stop is checked: an
    // inaccurate result is expected at an insufficient cap.
    for seed in 0..4 {
        let problem = star_problem(NonZeroUsize::new(2), seed);
        let f = star_function;
        let checked = run_and_check(&TreeTciInterpolator::default(), &problem, &f);
        assert_eq!(
            checked.outcome.termination,
            InterpolationTermination::BondCapReached,
            "seed {seed}"
        );
        assert!(checked
            .outcome
            .network
            .link_dims()
            .iter()
            .all(|&dim| dim <= 2));
    }
}

#[test]
fn iteration_limit_is_reported() {
    // A zero tolerance is never met, and without a cap the loop can only end
    // at the iteration limit.
    let engine = TreeTciInterpolator::new(TreeTciOptions {
        max_iter: 3,
        ..Default::default()
    })
    .unwrap();
    let checked = run_and_check(&engine, &chain_problem(0.0, None, 1), &chain_function);
    assert_eq!(
        checked.outcome.termination,
        InterpolationTermination::IterationLimit
    );
}

#[test]
fn engine_requires_three_sweeps() {
    for max_iter in [1, 2] {
        let error = TreeTciInterpolator::new(TreeTciOptions {
            max_iter,
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("max_iter must be at least 3"));
    }
    assert!(TreeTciInterpolator::new(TreeTciOptions {
        max_iter: 3,
        ..Default::default()
    })
    .is_ok());
}

#[test]
fn returned_pivots_seed_a_second_run() {
    let engine = TreeTciInterpolator::default();
    let first = run_and_check(
        &engine,
        &branched_problem(None, 1e-12, 4),
        &branched_function,
    );
    let pivots = first.outcome.pivots.clone().unwrap();
    assert!(pivots.ncols().unwrap() > 1);

    let second = run_and_check(
        &engine,
        &branched_problem(Some(pivots), 1e-12, 4),
        &branched_function,
    );
    assert_eq!(
        second.outcome.termination,
        InterpolationTermination::Converged
    );
    assert_accurate(&second, 1e-10);
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// Pseudo-random values on a ~3% support of a dimension-4 domain (the
/// function of issue #692), for which the global pivot search keeps finding
/// missed support, so its random starting points matter.
fn sparse_value(point: &[usize]) -> f64 {
    let key = point.iter().fold(0u64, |key, &v| key * 4 + v as u64);
    let h = mix64(key);
    if h % 32 < 1 {
        ((h >> 8) as f64 / u64::MAX as f64) - 0.5
    } else {
        0.0
    }
}

/// Chain of five dimension-4 nodes (max degree 2), seeded with eight
/// support points and capped at 4.
fn sparse_chain_problem(seed: u64) -> InterpolationProblem<Name> {
    const N_SITES: usize = 5;
    let mut pivots = Vec::new();
    let mut key = 1u64;
    while pivots.len() < 8 {
        key = mix64(key).max(1);
        let point: Vec<usize> = (0..N_SITES)
            .map(|site| ((key >> (2 * site)) % 4) as usize)
            .collect();
        if sparse_value(&point) != 0.0 {
            pivots.push(point);
        }
    }
    let names = ["s0", "s1", "s2", "s3", "s4"];
    let node_sites = names
        .iter()
        .map(|node| (name(node), vec![DynIndex::new_dyn(4)]))
        .collect();
    InterpolationProblem::new(
        topology(
            &names,
            &[("s0", "s1"), ("s1", "s2"), ("s2", "s3"), ("s3", "s4")],
        ),
        node_sites,
        pivots_from(&pivots),
        1e-12,
        NonZeroUsize::new(4),
        seed,
    )
    .unwrap()
}

#[test]
fn problem_seed_overrides_the_engine_seed() {
    let run = |problem: &InterpolationProblem<Name>, engine_seed| {
        let engine = TreeTciInterpolator::new(TreeTciOptions {
            seed: Some(engine_seed),
            max_iter: 4,
            nsearch: 5,
            max_nglobal_pivot: 5,
            ..Default::default()
        })
        .unwrap();
        engine
            .interpolate(problem, evaluator(&sparse_value, site_dims(problem)))
            .unwrap()
    };
    let dense_difference = |a: &InterpolationOutcome<Name>, b: &InterpolationOutcome<Name>| {
        a.network
            .to_dense()
            .unwrap()
            .sub(&b.network.to_dense().unwrap())
            .unwrap()
            .maxabs()
            .unwrap()
    };

    // Different engine seeds, same problem seed: identical runs.
    let problem = sparse_chain_problem(7);
    let (first, second) = (run(&problem, 1), run(&problem, 2));
    assert_eq!(first.termination, second.termination);
    assert_eq!(first.pivots, second.pivots);
    assert_eq!(first.error_estimate, second.error_estimate);
    assert_eq!(dense_difference(&first, &second), 0.0);

    // The seed is live: some other problem seed changes the run.
    assert!(
        (8..12).any(|seed| run(&sparse_chain_problem(seed), 1).pivots != first.pivots),
        "no problem seed changed the pivots"
    );
}

fn expect_evaluator_error<T: Debug>(result: Result<T, InterpolationError>, needle: &str) {
    match result {
        Err(InterpolationError::Evaluator { source }) => assert!(
            format!("{source:#}").contains(needle),
            "evaluator error {source:#} does not mention {needle:?}"
        ),
        other => panic!("expected Evaluator mentioning {needle:?}, got {other:?}"),
    }
}

/// Evaluator that behaves until its `fail_on`-th call, then either fails or
/// returns one value too many.
fn faulty_evaluator(
    calls: &Cell<usize>,
    fail_on: usize,
    wrong_length: bool,
    n_sites: usize,
) -> impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>> + '_ {
    move |batch| {
        calls.set(calls.get() + 1);
        let mut values: Vec<f64> = batch.data().chunks(n_sites).map(chain_like).collect();
        if calls.get() == fail_on {
            if wrong_length {
                values.push(0.0);
            } else {
                anyhow::bail!("planned failure on call {fail_on}");
            }
        }
        Ok(values)
    }
}

fn chain_like(p: &[usize]) -> f64 {
    1.0 + p
        .iter()
        .enumerate()
        .map(|(i, &v)| (i + 1) as f64 * v as f64)
        .sum::<f64>()
        .sin()
}

#[test]
fn evaluator_failures_are_reported_as_evaluator_errors() {
    let engine = TreeTciInterpolator::default();
    let chain = chain_problem(1e-12, None, 0);
    let branched = branched_problem(None, 1e-12, 0);
    let single = single_node_problem(vec![1, 1, 1]);
    // Call 1 evaluates the initial pivots; later calls come from inside
    // TreeTCI (sweeps, global pivot search, materialization) or, for a
    // single node, from the exact full evaluation.
    for (problem, fail_on) in [
        (&chain, 1),
        (&chain, 2),
        (&chain, 5),
        (&branched, 3),
        (&single, 2),
    ] {
        let n_sites = problem.site_order().len();
        let calls = Cell::new(0);
        expect_evaluator_error(
            engine.interpolate(problem, faulty_evaluator(&calls, fail_on, false, n_sites)),
            &format!("planned failure on call {fail_on}"),
        );
        assert_eq!(calls.get(), fail_on);

        let calls = Cell::new(0);
        expect_evaluator_error(
            engine.interpolate(problem, faulty_evaluator(&calls, fail_on, true, n_sites)),
            "values for",
        );
        assert_eq!(calls.get(), fail_on);
    }
}

#[test]
fn all_zero_initial_pivots_return_all_samples_zero() {
    let engine = TreeTciInterpolator::default();
    // f vanishes only where the first site is zero.
    let f = |p: &[usize]| p[0] as f64;

    let single = single_node_problem(vec![0, 2, 1]);
    let result = engine.interpolate(&single, evaluator(&f, site_dims(&single)));
    assert!(matches!(result, Err(InterpolationError::AllSamplesZero)));

    let chain = chain_problem(1e-12, None, 0);
    let result = engine.interpolate(&chain, evaluator(&f, site_dims(&chain)));
    assert!(matches!(result, Err(InterpolationError::AllSamplesZero)));
}

#[test]
fn oversized_nodes_are_invalid_problems() {
    let engine = TreeTciInterpolator::default();
    let calls = Cell::new(0);
    let counting = |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
        calls.set(calls.get() + 1);
        Ok(vec![1.0; batch.shape()[1]])
    };

    // The fused dimension of node "big" overflows usize: rejected before any
    // evaluation.
    let huge = 1usize << (usize::BITS / 2 + 1);
    let problem = InterpolationProblem::new(
        topology(&["big", "small"], &[("big", "small")]),
        BTreeMap::from([
            (
                name("big"),
                vec![DynIndex::new_dyn(huge), DynIndex::new_dyn(huge)],
            ),
            (name("small"), vec![DynIndex::new_dyn(2)]),
        ]),
        pivots_from(&[vec![0, 0, 0]]),
        1e-12,
        None,
        0,
    )
    .unwrap();
    let result = engine.interpolate(&problem, counting);
    assert!(matches!(
        result,
        Err(InterpolationError::InvalidProblem { .. })
    ));
    assert_eq!(calls.get(), 0);

    // A single node whose index set fits in usize but whose point buffer
    // (points times sites) does not.
    let problem = InterpolationProblem::new(
        topology(&["only"], &[]),
        BTreeMap::from([(
            name("only"),
            vec![
                DynIndex::new_dyn(1usize << (usize::BITS - 2)),
                DynIndex::new_dyn(2),
            ],
        )]),
        pivots_from(&[vec![0, 1]]),
        1e-12,
        None,
        0,
    )
    .unwrap();
    let result = engine.interpolate(&problem, counting);
    match result {
        Err(InterpolationError::InvalidProblem { message }) => {
            assert!(message.contains("overflows usize"), "{message}")
        }
        other => panic!("expected InvalidProblem, got {other:?}"),
    }
}

#[test]
fn evaluator_failure_during_materialization_is_an_evaluator_error() {
    let engine = TreeTciInterpolator::default();
    let problem = chain_problem(1e-12, None, 0);
    let n_sites = problem.site_order().len();

    // Count the calls of a successful run; the run is deterministic.
    let calls = Cell::new(0);
    engine
        .interpolate(
            &problem,
            faulty_evaluator(&calls, usize::MAX, false, n_sites),
        )
        .unwrap();
    let total = calls.get();

    // Materialization evaluates every node's tensor after the optimization
    // loop and nothing evaluates after it, so the last call of the run is a
    // materialization call.
    let calls = Cell::new(0);
    expect_evaluator_error(
        engine.interpolate(&problem, faulty_evaluator(&calls, total, false, n_sites)),
        &format!("planned failure on call {total}"),
    );
    assert_eq!(calls.get(), total);
}

#[test]
fn non_finite_initial_samples_are_evaluator_errors() {
    let engine = TreeTciInterpolator::default();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // Non-finite at the first initial pivot only, finite elsewhere.
        let f = move |p: &[usize]| if p.iter().all(|&v| v == 0) { bad } else { 1.0 };

        let chain = chain_problem(1e-12, None, 0);
        expect_evaluator_error(
            engine.interpolate(&chain, evaluator(&f, site_dims(&chain))),
            "non-finite value or magnitude at batch point 0",
        );

        // A finite nonzero sample beside the non-finite one does not hide it.
        let two_pivots = InterpolationProblem::new(
            chain.topology().clone(),
            chain.node_sites().clone(),
            pivots_from(&[vec![1, 0, 0, 0], vec![0, 0, 0, 0]]),
            1e-12,
            None,
            0,
        )
        .unwrap();
        expect_evaluator_error(
            engine.interpolate(&two_pivots, evaluator(&f, site_dims(&two_pivots))),
            "non-finite value or magnitude at batch point 1",
        );

        let single = single_node_problem(vec![0, 0, 0]);
        expect_evaluator_error(
            engine.interpolate(&single, evaluator(&f, site_dims(&single))),
            "non-finite value or magnitude at batch point 0",
        );
    }
}

#[test]
fn single_node_point_lists_reject_byte_capacity_overflow() {
    let dim = (isize::MAX as usize / std::mem::size_of::<usize>()) + 1;
    let problem = InterpolationProblem::new(
        topology(&["only"], &[]),
        BTreeMap::from([(name("only"), vec![DynIndex::new_dyn(dim)])]),
        pivots_from(&[vec![0]]),
        1e-12,
        None,
        0,
    )
    .unwrap();
    let calls = Cell::new(0);
    let result = TreeTciInterpolator::default().interpolate(
        &problem,
        |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
            calls.set(calls.get() + 1);
            Ok(vec![1.0; batch.shape()[1]])
        },
    );
    match result {
        Err(InterpolationError::InvalidProblem { message }) => {
            assert!(message.contains("point list"), "{message}");
        }
        other => panic!("expected InvalidProblem, got {other:?}"),
    }
    // Only the one initial pivot may be evaluated; enumeration never runs.
    assert!(calls.get() <= 1);
}

#[test]
fn non_finite_samples_after_initial_pivots_are_evaluator_errors() {
    let engine = TreeTciInterpolator::default();
    let problem = InterpolationProblem::new(
        topology(&["only"], &[]),
        BTreeMap::from([(name("only"), vec![DynIndex::new_dyn(2)])]),
        pivots_from(&[vec![0]]),
        1e-12,
        None,
        0,
    )
    .unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        expect_evaluator_error(
            engine.interpolate(
                &problem,
                evaluator(&|p| if p[0] == 0 { 1.0 } else { bad }, vec![2]),
            ),
            "non-finite",
        );
    }
}

#[test]
fn non_finite_sweep_and_materialization_samples_are_evaluator_errors() {
    let engine = TreeTciInterpolator::default();
    let problem = chain_problem(1e-12, None, 0);
    fn make_evaluator(
        calls: &Cell<usize>,
        fail_on: usize,
        bad: f64,
        n_sites: usize,
    ) -> impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>> + '_ {
        move |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
            calls.set(calls.get() + 1);
            Ok(batch
                .data()
                .chunks(n_sites)
                .map(|point| {
                    if calls.get() == fail_on {
                        bad
                    } else {
                        1.0 + point.iter().sum::<usize>() as f64
                    }
                })
                .collect())
        }
    }
    let calls = Cell::new(0);
    engine
        .interpolate(
            &problem,
            make_evaluator(&calls, usize::MAX, f64::NAN, problem.site_order().len()),
        )
        .unwrap();
    let total = calls.get();
    assert!(total > 2);
    // Call one samples initial pivots; call two enters the sweep, and the
    // final call materializes the last node after optimization.
    for fail_on in [2, total] {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let calls = Cell::new(0);
            expect_evaluator_error(
                engine.interpolate(
                    &problem,
                    make_evaluator(&calls, fail_on, bad, problem.site_order().len()),
                ),
                "non-finite",
            );
            assert_eq!(calls.get(), fail_on);
        }
    }
}
