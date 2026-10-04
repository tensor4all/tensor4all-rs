//! Shared helpers of the adaptive interpolation integration tests.
#![allow(dead_code)]

mod dense_engine;
mod fiber_engine;
mod scripted_engine;

#[allow(unused_imports)]
pub(crate) use dense_engine::*;
#[allow(unused_imports)]
pub(crate) use fiber_engine::*;
#[allow(unused_imports)]
pub(crate) use scripted_engine::*;

use std::collections::{BTreeMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use tensor4all_core::{
    ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, IndexLike, TensorElement,
};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate, GlobalL2Error, NormReport, PatchStatus, PatchedInterpolationError,
    PatchedInterpolationOptions, PatchedInterpolationReport, PatchedInterpolationResult,
    GLOBAL_ROUNDING_MARGIN,
};
use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, L2Reference, Projector};
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationProblem, TreeInterpolator,
};
use tensor4all_treetn::NodeNameNetwork;

/// Accuracy bound of accepted patches against a dense reference, in units of
/// `rtol * max_reference` (under `ErrorNorm::SampledMax`). The acceptance criterion is the engine's sampled
/// error estimate, not a certified bound, so the tests allow this margin.
pub(crate) const ACCURACY_FACTOR: f64 = 10.0;

pub(crate) type Name = String;

// ---------------------------------------------------------------------------
// Options and report accessors
// ---------------------------------------------------------------------------

/// Options with the M2 criterion (`ErrorNorm::sampled_max()`).
pub(crate) fn sampled_max(max_bond_dim: usize) -> PatchedInterpolationOptions {
    PatchedInterpolationOptions::new(max_bond_dim).with_error_norm(ErrorNorm::sampled_max())
}

/// A relative tolerance without an absolute floor.
pub(crate) fn tol(rtol: f64) -> ErrorTolerance {
    ErrorTolerance { rtol, atol: 0.0 }
}

/// L2 options with a given reference norm and relative tolerance.
pub(crate) fn l2_given(cap: usize, norm: f64, rtol: f64) -> PatchedInterpolationOptions {
    PatchedInterpolationOptions::new(cap)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
        .with_tolerance(tol(rtol))
}

/// The `tau` of an L2 report.
pub(crate) fn tau(report: &PatchedInterpolationReport) -> f64 {
    match report.norm {
        NormReport::L2 { tau, .. } => tau,
        ref other => panic!("expected an L2 report, got {other:?}"),
    }
}

/// The max-norm reference of a `SampledMax` run.
pub(crate) fn max_reference(report: &PatchedInterpolationReport) -> f64 {
    match report.norm {
        NormReport::SampledMax { max_reference, .. } => max_reference,
        ref other => panic!("expected a SampledMax report, got {other:?}"),
    }
}

/// The projectors of the zero patches, in report order.
pub(crate) fn zero_projectors(report: &PatchedInterpolationReport) -> Vec<Projector> {
    report
        .zero_patches
        .iter()
        .map(|record| record.projector.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Problems
// ---------------------------------------------------------------------------

/// A tree topology with the sites of every node and the derived site order.
pub(crate) struct Problem {
    pub(crate) topology: NodeNameNetwork<Name>,
    pub(crate) node_sites: BTreeMap<Name, Vec<DynIndex>>,
    pub(crate) sites: Vec<DynIndex>,
}

impl Problem {
    /// Nodes with their site dimensions, and edges.
    pub(crate) fn new(nodes: &[(&str, &[usize])], edges: &[(&str, &str)]) -> Self {
        let node_sites = nodes
            .iter()
            .map(|(node, dims)| {
                let sites = dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
                (node.to_string(), sites)
            })
            .collect();
        Self::with_sites(node_sites, edges)
    }

    pub(crate) fn with_sites(
        node_sites: BTreeMap<Name, Vec<DynIndex>>,
        edges: &[(&str, &str)],
    ) -> Self {
        let nodes: Vec<&str> = node_sites.keys().map(String::as_str).collect();
        let topology = topology(&nodes, edges);
        let sites = InterpolationProblem::derive_site_order(&node_sites);
        Self {
            topology,
            node_sites,
            sites,
        }
    }

    pub(crate) fn dims(&self) -> Vec<usize> {
        self.sites.iter().map(IndexLike::dim).collect()
    }

    pub(crate) fn site(&self, node: &str, k: usize) -> DynIndex {
        self.node_sites[node][k].clone()
    }

    pub(crate) fn position(&self, site: &DynIndex) -> usize {
        self.sites.iter().position(|s| s == site).unwrap()
    }

    pub(crate) fn pivots(&self, points: &[Vec<usize>]) -> ColMajorArray<usize> {
        ColMajorArray::new(points.concat(), vec![self.sites.len(), points.len()]).unwrap()
    }
}

/// Every point of a domain, column-major (first site fastest).
pub(crate) fn full_domain(dims: &[usize]) -> Vec<Vec<usize>> {
    let n_points: usize = dims.iter().product();
    (0..n_points)
        .map(|mut linear| {
            dims.iter()
                .map(|&dim| {
                    let value = linear % dim;
                    linear /= dim;
                    value
                })
                .collect()
        })
        .collect()
}

/// Branched tree with a junction of degree three:
///
/// ```text
///   a - j - b - c - e
///       |
///       d
/// ```
///
/// `j` (junction, site j0: 2), `a` (leaf, a0: 3), `b` (internal, b0: 2),
/// `c` (internal, c0: 2), `d` (leaf, two sites d0, d1: 2), `e` (leaf, no
/// site). Site order: a0, b0, c0, d0, d1, j0.
pub(crate) fn branched() -> Problem {
    Problem::new(
        &[
            ("a", &[3]),
            ("b", &[2]),
            ("c", &[2]),
            ("d", &[2, 2]),
            ("e", &[]),
            ("j", &[2]),
        ],
        &[("j", "a"), ("j", "b"), ("j", "d"), ("b", "c"), ("c", "e")],
    )
}

/// A topology with the given node names and edges; it need not be a tree.
pub(crate) fn topology(nodes: &[&str], edges: &[(&str, &str)]) -> NodeNameNetwork<Name> {
    let mut topology = NodeNameNetwork::new();
    for node in nodes {
        topology.add_node(node.to_string()).unwrap();
    }
    for (left, right) in edges {
        topology
            .add_edge(&left.to_string(), &right.to_string())
            .unwrap();
    }
    topology
}

/// A chain of `n` nodes named `{prefix}{k}` (zero-padded to sort in chain
/// order), each with one site of dimension `dim`.
pub(crate) fn chain(prefix: &str, n: usize, dim: usize) -> Problem {
    let width = (n.max(2) - 1).to_string().len();
    let names: Vec<String> = (0..n).map(|k| format!("{prefix}{k:0width$}")).collect();
    let dims = [dim];
    let nodes: Vec<(&str, &[usize])> = names
        .iter()
        .map(|name| (name.as_str(), &dims[..]))
        .collect();
    let edges: Vec<(&str, &str)> = names
        .windows(2)
        .map(|pair| (pair[0].as_str(), pair[1].as_str()))
        .collect();
    Problem::new(&nodes, &edges)
}

/// Chain n0 - n1 - n2 with binary sites.
pub(crate) fn chain3() -> Problem {
    chain("n", 3, 2)
}

pub(crate) fn single_node(dims: &[usize]) -> Problem {
    Problem::new(&[("only", dims)], &[])
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// A product function whose factors depend on `variant`.
pub(crate) fn product(point: &[usize], variant: usize) -> f64 {
    point
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            1.0 + 0.1 * ((variant + 1) * (i + 2)) as f64 * x as f64 + 0.05 * (x * x) as f64
        })
        .product()
}

/// `f = g_{x_s}(x)`: a different product function for every value of the
/// site at `position`. Rank two or more across every edge; rank one once the
/// site is fixed.
pub(crate) fn switch_on(position: usize) -> impl Fn(&[usize]) -> f64 + Sync {
    move |point| product(point, point[position])
}

/// Quantics coordinate in [0, 1) of `bits` (most significant first).
pub(crate) fn quantics(bits: &[usize]) -> f64 {
    bits.iter()
        .enumerate()
        .map(|(k, &bit)| bit as f64 * 0.5_f64.powi(k as i32 + 1))
        .sum()
}

pub(crate) fn gaussian(x: f64, center: f64, width: f64) -> f64 {
    (-((x - center) / width).powi(2)).exp()
}

/// Largest magnitude of `f` over the whole (small) domain.
pub(crate) fn max_abs(problem: &Problem, f: &dyn Fn(&[usize]) -> f64) -> f64 {
    full_domain(&problem.dims())
        .iter()
        .map(|p| f(p).abs())
        .fold(0.0, f64::max)
}

/// Two quantics variables on branches of a site-free junction `r` of degree
/// three; the third branch `z` is a binary flag.
pub(crate) fn quantics_tree() -> Problem {
    Problem::new(
        &[
            ("r", &[]),
            ("x0", &[2]),
            ("x1", &[2]),
            ("x2", &[2]),
            ("y0", &[2]),
            ("y1", &[2]),
            ("y2", &[2]),
            ("z", &[2]),
        ],
        &[
            ("r", "x0"),
            ("x0", "x1"),
            ("x1", "x2"),
            ("r", "y0"),
            ("y0", "y1"),
            ("y1", "y2"),
            ("r", "z"),
        ],
    )
}

/// The M2 `quantics_tree` extended by a leaf `w` (attached to `z`) with two
/// sites of dimensions 2 and 3: `2^7 * 6 = 768` points. The junction `r` has
/// degree three and carries no site, and `w` carries two sites, so the
/// cached evaluator takes its generic path. Site order w0, w1, x0, x1, x2,
/// y0, y1, y2, z.
pub(crate) fn extended_quantics_tree() -> Problem {
    Problem::new(
        &[
            ("r", &[]),
            ("w", &[2, 3]),
            ("x0", &[2]),
            ("x1", &[2]),
            ("x2", &[2]),
            ("y0", &[2]),
            ("y1", &[2]),
            ("y2", &[2]),
            ("z", &[2]),
        ],
        &[
            ("r", "x0"),
            ("x0", "x1"),
            ("x1", "x2"),
            ("r", "y0"),
            ("y0", "y1"),
            ("y1", "y2"),
            ("r", "z"),
            ("z", "w"),
        ],
    )
}

/// One binary site per node around a junction `c` of degree three, so the
/// cached evaluator takes its raw kernels. Site order a0, a1, b0, b1, c, d0,
/// d1 (128 points).
pub(crate) fn raw_kernel_tree() -> Problem {
    Problem::new(
        &[
            ("a0", &[2]),
            ("a1", &[2]),
            ("b0", &[2]),
            ("b1", &[2]),
            ("c", &[2]),
            ("d0", &[2]),
            ("d1", &[2]),
        ],
        &[
            ("c", "a0"),
            ("a0", "a1"),
            ("c", "b0"),
            ("b0", "b1"),
            ("c", "d0"),
            ("d0", "d1"),
        ],
    )
}

/// Largest node degree of a problem's topology.
pub(crate) fn max_degree(problem: &Problem) -> usize {
    let graph = problem.topology.graph();
    graph
        .node_indices()
        .map(|node| graph.neighbors(node).count())
        .max()
        .unwrap_or(0)
}

/// Split order of [`extended_quantics_tree`]: most significant bits first,
/// then the flag and the two sites of the leaf `w`.
pub(crate) fn extended_order(problem: &Problem) -> Vec<DynIndex> {
    let mut order: Vec<DynIndex> = ["x0", "y0", "x1", "y1", "x2", "y2", "z"]
        .iter()
        .map(|node| problem.site(node, 0))
        .collect();
    order.push(problem.site("w", 0));
    order.push(problem.site("w", 1));
    order
}

/// A localized peak on [`extended_quantics_tree`]: a narrow Gaussian in
/// (x, y) whose height depends on the flag `z` and the leaf `w`.
pub(crate) fn extended_peak(p: &[usize]) -> f64 {
    let (x, y) = (quantics(&p[2..5]), quantics(&p[5..8]));
    (1.0 + 0.25 * p[0] as f64 + 0.125 * p[1] as f64)
        * (1.0 + 0.5 * p[8] as f64)
        * gaussian(x, 0.3, 0.12)
        * gaussian(y, 0.6, 0.12)
        + 0.05 * x * y
}

/// Every value of `f` on the domain of `problem` as a dense tensor, and its
/// L2 norm.
pub(crate) fn dense_reference<T>(problem: &Problem, f: &dyn Fn(&[usize]) -> T) -> (IdxTensor, f64)
where
    T: CommonScalar + TensorElement,
{
    let values: Vec<T> = full_domain(&problem.dims()).iter().map(|p| f(p)).collect();
    let reference = IdxTensor::from_dense(problem.sites.clone(), values).unwrap();
    let norm = reference.norm().unwrap();
    (reference, norm)
}

/// `||partition - f||` over the whole domain, materialized once.
pub(crate) fn dense_l2_residual(
    result: &PatchedInterpolationResult<Name>,
    reference: &IdxTensor,
) -> f64 {
    if result.partition.is_empty() {
        return reference.norm().unwrap();
    }
    let dense = result
        .partition
        .to_treetn()
        .unwrap()
        .contract_to_tensor()
        .unwrap();
    dense.sub(reference).unwrap().norm().unwrap()
}

/// A peak at (0.3, 0.6) whose height depends on the flag; site order
/// x0, x1, x2, y0, y1, y2, z.
pub(crate) fn tree_peak(p: &[usize]) -> f64 {
    let (x, y) = (quantics(&p[0..3]), quantics(&p[3..6]));
    (1.0 + 0.5 * p[6] as f64) * gaussian(x, 0.3, 0.12) * gaussian(y, 0.6, 0.12)
        + 0.2 * (x * y + p[6] as f64 * x)
}

// ---------------------------------------------------------------------------
// Evaluation bookkeeping and the generic run helper
// ---------------------------------------------------------------------------

/// An evaluator that must never be called.
pub(crate) fn never(batch: ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>> {
    panic!("the evaluator was called with shape {:?}", batch.shape())
}

/// Records every evaluated point and fails on a repeated one.
#[derive(Default)]
pub(crate) struct Recorder {
    pub(crate) seen: Mutex<HashSet<Vec<usize>>>,
    pub(crate) calls: AtomicUsize,
}

impl Recorder {
    pub(crate) fn points(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

pub(crate) fn recording_evaluator<'a, T>(
    f: &'a (dyn Fn(&[usize]) -> T + Sync),
    n_sites: usize,
    recorder: &'a Recorder,
) -> impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> + Send + Sync + 'a {
    move |batch| {
        assert_eq!(batch.shape()[0], n_sites, "batch rows");
        recorder.calls.fetch_add(1, Ordering::SeqCst);
        let mut seen = recorder.seen.lock().unwrap();
        Ok(batch
            .data()
            .chunks(n_sites)
            .map(|point| {
                assert!(
                    seen.insert(point.to_vec()),
                    "point {point:?} evaluated twice"
                );
                f(point)
            })
            .collect())
    }
}

/// The path of a projector in split order: (position, coordinate) pairs.
pub(crate) fn path_of(
    projector: &Projector,
    split_order: &[DynIndex],
    problem: &Problem,
) -> Vec<(usize, usize)> {
    split_order
        .iter()
        .filter_map(|site| {
            projector
                .get(site)
                .map(|value| (problem.position(site), value))
        })
        .collect()
}

/// Check the partition and report against the problem: accepted and zero
/// projectors are disjoint and cover the domain, both lists are in canonical
/// path order, and the partition holds exactly the accepted projectors.
pub(crate) fn check_invariants(
    result: &PatchedInterpolationResult<Name>,
    problem: &Problem,
    options: &PatchedInterpolationOptions,
) {
    let split_order = if options.patch_order.is_empty() {
        problem.sites.clone()
    } else {
        options.patch_order.clone()
    };
    let report = &result.report;
    let accepted: Vec<&Projector> = report.accepted.iter().map(|r| &r.projector).collect();
    let zeros = zero_projectors(report);
    for list in [accepted.clone(), zeros.iter().collect()] {
        let paths: Vec<_> = list
            .iter()
            .map(|projector| path_of(projector, &split_order, problem))
            .collect();
        assert!(
            paths.windows(2).all(|w| w[0] < w[1]),
            "not canonical: {paths:?}"
        );
    }
    let all: Vec<&Projector> = accepted.iter().copied().chain(zeros.iter()).collect();
    for (i, left) in all.iter().enumerate() {
        for right in &all[i + 1..] {
            assert!(
                !left.is_compatible_with(right),
                "{left:?} overlaps {right:?}"
            );
        }
    }
    // Volumes as f64: exact for these power-of-two and small domains, and
    // free of overflow for the 129-site domain.
    let volume = |sites: &mut dyn Iterator<Item = &DynIndex>| {
        sites.map(|site| site.dim() as f64).product::<f64>()
    };
    let covered: f64 = all
        .iter()
        .map(|projector| {
            volume(
                &mut problem
                    .sites
                    .iter()
                    .filter(|site| !projector.is_projected_at(site)),
            )
        })
        .sum();
    assert_eq!(covered, volume(&mut problem.sites.iter()));
    assert_eq!(result.partition.len(), accepted.len());
    for projector in accepted {
        assert!(result.partition.contains(projector));
    }
    // Under L2 the status agrees with the deciding measurement.
    if let NormReport::L2 { tau, .. } = report.norm {
        for record in &report.accepted {
            let rms = record
                .acceptance
                .as_ref()
                .expect("an L2 record has a measurement")
                .rms;
            match record.status {
                PatchStatus::WithinTolerance => assert!(rms <= tau, "{record:?} exceeds {tau}"),
                PatchStatus::ToleranceNotMet => assert!(rms > tau, "{record:?} meets {tau}"),
                status => panic!("unexpected status {status:?}"),
            }
        }
    }
    assert_eq!(
        report.tolerance_met(),
        report
            .accepted
            .iter()
            .all(|record| record.status == PatchStatus::WithinTolerance)
    );
}

/// Run the driver with the given batch evaluator and full-domain pivots.
pub(crate) fn run_with<T, E, F>(
    engine: &E,
    problem: &Problem,
    evaluate: F,
    pivots: &[Vec<usize>],
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<Name>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    E: TreeInterpolator<T> + Sync,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> + Send + Sync,
{
    let initial = if pivots.is_empty() {
        ColMajorArray::new(vec![], vec![problem.sites.len(), 0]).unwrap()
    } else {
        problem.pivots(pivots)
    };
    patched_interpolate(
        engine,
        problem.topology.clone(),
        problem.node_sites.clone(),
        initial,
        evaluate,
        options,
    )
}

/// Run the driver with a recording evaluator. On success, check the
/// invariants and that the reported evaluation count matches the evaluator.
pub(crate) fn run<T, E>(
    engine: &E,
    problem: &Problem,
    f: &(dyn Fn(&[usize]) -> T + Sync),
    pivots: &[Vec<usize>],
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<Name>, PatchedInterpolationError>
where
    T: CommonScalar + TensorElement,
    E: TreeInterpolator<T> + Sync,
{
    let recorder = Recorder::default();
    let evaluate = recording_evaluator(f, problem.sites.len(), &recorder);
    let result = run_with(engine, problem, evaluate, pivots, options);
    if let Ok(result) = &result {
        assert_eq!(result.report.function_evaluations, recorder.points());
        check_invariants(result, problem, options);
    }
    result
}

/// Materialize the partition once and return `maxabs(partition - f)` and
/// `max |f|` over the whole domain.
pub(crate) fn dense_residual<T>(
    result: &PatchedInterpolationResult<Name>,
    problem: &Problem,
    f: &dyn Fn(&[usize]) -> T,
) -> (f64, f64)
where
    T: CommonScalar + TensorElement,
{
    let values: Vec<T> = full_domain(&problem.dims()).iter().map(|p| f(p)).collect();
    let reference = IdxTensor::from_dense(problem.sites.clone(), values).unwrap();
    let max_abs = reference.maxabs().unwrap();
    if result.partition.is_empty() {
        return (max_abs, max_abs);
    }
    let dense = result
        .partition
        .to_treetn()
        .unwrap()
        .contract_to_tensor()
        .unwrap();
    (dense.sub(&reference).unwrap().maxabs().unwrap(), max_abs)
}

pub(crate) fn assert_accurate<T>(
    result: &PatchedInterpolationResult<Name>,
    problem: &Problem,
    f: &dyn Fn(&[usize]) -> T,
    rtol: f64,
) where
    T: CommonScalar + TensorElement,
{
    let (residual, _) = dense_residual(result, problem, f);
    let bound = ACCURACY_FACTOR * rtol * max_reference(&result.report);
    assert!(residual <= bound, "residual {residual} exceeds {bound}");
}

pub(crate) fn expect_invalid<T: Debug>(
    result: Result<T, PatchedInterpolationError>,
    needle: &str,
) -> String {
    match result {
        Err(PatchedInterpolationError::InvalidInput { message }) => {
            assert!(message.contains(needle), "{message:?} lacks {needle:?}");
            message
        }
        other => panic!("expected InvalidInput mentioning {needle:?}, got {other:?}"),
    }
}

pub(crate) fn expect_interpolation<T: Debug>(
    result: Result<T, PatchedInterpolationError>,
) -> (Projector, InterpolationError) {
    match result {
        Err(PatchedInterpolationError::Interpolation { projector, source }) => (projector, source),
        other => panic!("expected an Interpolation error, got {other:?}"),
    }
}

/// One leg of a stored node tensor, described without run-specific index IDs:
/// a site by its position in the site order, a bond by the node it links to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Leg {
    Site(usize),
    Bond { to: Name, dim: usize },
}

/// The stored node tensor of one patch: its legs in positional order and its
/// raw column-major `f64` data as bit patterns.
pub(crate) type NodeFingerprint = (Name, Vec<Leg>, Vec<u64>);

/// One stored patch: its projector as sorted (site position, coordinate)
/// pairs and its node fingerprints in node-name order.
pub(crate) type PatchFingerprint = (Vec<(usize, usize)>, Vec<NodeFingerprint>);

/// A bitwise, positional description of every stored patch, in report order:
/// the projector as sorted (site position, coordinate) pairs and, for every
/// node in name order, its legs and raw data. Two runs with equal
/// fingerprints store the same tensors with the same axis order.
pub(crate) fn fingerprint(
    result: &PatchedInterpolationResult<Name>,
    problem: &Problem,
) -> Vec<PatchFingerprint> {
    result
        .report
        .accepted
        .iter()
        .map(|record| {
            let mut entries: Vec<(usize, usize)> = record
                .projector
                .iter()
                .map(|(site, &value)| (problem.position(site), value))
                .collect();
            entries.sort_unstable();
            let data = result.partition.get(&record.projector).unwrap().data();
            let mut names = data.node_names();
            names.sort();
            let tensor_of = |name: &Name| data.tensor(data.node_index(name).unwrap()).unwrap();
            let nodes = names
                .iter()
                .map(|name| {
                    let tensor = tensor_of(name);
                    assert!(tensor.is_f64(), "node {name} is not f64");
                    let legs = tensor
                        .indices()
                        .iter()
                        .map(
                            |index| match problem.sites.iter().position(|s| s == index) {
                                Some(position) => Leg::Site(position),
                                None => {
                                    let to = names
                                        .iter()
                                        .find(|other| {
                                            *other != name
                                                && tensor_of(other).indices().contains(index)
                                        })
                                        .unwrap()
                                        .clone();
                                    Leg::Bond {
                                        to,
                                        dim: index.dim(),
                                    }
                                }
                            },
                        )
                        .collect();
                    let bits = tensor
                        .to_vec::<f64>()
                        .unwrap()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect();
                    (name.clone(), legs, bits)
                })
                .collect();
            (entries, nodes)
        })
        .collect()
}

/// Whether two values agree within `GLOBAL_ROUNDING_MARGIN`.
fn within_margin(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= GLOBAL_ROUNDING_MARGIN * a.abs().max(b.abs()),
        (None, None) => true,
        _ => false,
    }
}

/// The norm reports of two runs agree: bitwise except the fields exempt from
/// the determinism claim, which agree within `GLOBAL_ROUNDING_MARGIN`;
/// `rounding_limited` is compared only away from its threshold.
pub(crate) fn assert_same_norm(a: &NormReport, b: &NormReport) {
    match (a, b) {
        (
            NormReport::L2 {
                reference_rms: ra,
                source: sa,
                tau: ta,
                error: ea,
                ..
            },
            NormReport::L2 {
                reference_rms: rb,
                source: sb,
                tau: tb,
                error: eb,
                ..
            },
        ) => {
            assert_eq!(ra.map(f64::to_bits), rb.map(f64::to_bits));
            assert_eq!(sa, sb);
            assert_eq!(ta.to_bits(), tb.to_bits());
            assert_eq!(ea.domain_points.to_bits(), eb.domain_points.to_bits());
            assert_eq!(
                ea.certified_fraction.to_bits(),
                eb.certified_fraction.to_bits()
            );
            assert!(within_margin(ea.approximation_rms, eb.approximation_rms));
            match (&ea.global, &eb.global) {
                (
                    GlobalL2Error::Certified {
                        rms_error: xa,
                        rounding_allowance_rms: aa,
                        rounding_limited: la,
                        relative_error_bound: ba,
                        ..
                    },
                    GlobalL2Error::Certified {
                        rms_error: xb,
                        rounding_allowance_rms: ab,
                        rounding_limited: lb,
                        relative_error_bound: bb,
                        ..
                    },
                ) => {
                    assert_eq!(xa.to_bits(), xb.to_bits());
                    assert!(within_margin(*aa, *ab));
                    assert!(within_margin(*ba, *bb));
                    let near_threshold = aa.is_some_and(|aa| within_margin(Some(aa), Some(*ta)));
                    if !near_threshold {
                        assert_eq!(la, lb);
                    }
                }
                (
                    GlobalL2Error::Audited {
                        rms_error_estimate: xa,
                        mean_square_rel_std_error: sa,
                        relative_bound_estimate: ba,
                        ..
                    },
                    GlobalL2Error::Audited {
                        rms_error_estimate: xb,
                        mean_square_rel_std_error: sb,
                        relative_bound_estimate: bb,
                        ..
                    },
                ) => {
                    assert_eq!(xa.to_bits(), xb.to_bits());
                    assert_eq!(sa.to_bits(), sb.to_bits());
                    assert!(within_margin(*ba, *bb));
                }
                (x, y) => assert_eq!(x, y),
            }
        }
        (x, y) => assert_eq!(x, y),
    }
}

/// Two runs are identical: the same records in the same order, the same
/// counts, and bitwise-identical stored patches (same legs in the same
/// positional order, same raw column-major data). The fields exempt from the
/// determinism claim are compared as in [`assert_same_norm`].
pub(crate) fn assert_same_run(
    problem: &Problem,
    first: &PatchedInterpolationResult<Name>,
    second: &PatchedInterpolationResult<Name>,
) {
    let (a, b) = (&first.report, &second.report);
    assert_eq!(a.tolerance, b.tolerance);
    assert_same_norm(&a.norm, &b.norm);
    assert_eq!(a.splits, b.splits);
    assert_eq!(a.function_evaluations, b.function_evaluations);
    assert_eq!(a.cache_hits, b.cache_hits);
    assert_eq!(a.measurement_evaluations, b.measurement_evaluations);
    assert_eq!(a.audit_evaluations, b.audit_evaluations);
    assert_eq!(a.verification_failures, b.verification_failures);
    assert_eq!(a.engine_retries, b.engine_retries);
    assert_eq!(a.zero_patches, b.zero_patches);
    assert_eq!(a.accepted.len(), b.accepted.len());
    for (x, y) in a.accepted.iter().zip(&b.accepted) {
        assert_eq!(x.projector, y.projector);
        assert_eq!(x.termination, y.termination);
        assert_eq!(
            x.engine_error_estimate.to_bits(),
            y.engine_error_estimate.to_bits()
        );
        assert_eq!(
            x.max_sample_magnitude.to_bits(),
            y.max_sample_magnitude.to_bits()
        );
        assert_eq!(x.max_bond_dim, y.max_bond_dim);
        assert_eq!(x.retries_used, y.retries_used);
        assert_eq!(x.acceptance, y.acceptance);
        assert_eq!(x.audit, y.audit);
        assert_eq!(x.status, y.status);
    }
    assert_eq!(first.partition.len(), second.partition.len());
    assert_eq!(fingerprint(first, problem), fingerprint(second, problem));
}

/// What two runs on separately built problems (fresh site IDs) can be
/// compared by: the ID-free fingerprint of the stored patches, the zero-patch
/// projectors as sorted (site position, coordinate) pairs, and the report.
pub(crate) struct RunDigest {
    pub(crate) patches: Vec<PatchFingerprint>,
    pub(crate) zero_patches: Vec<Vec<(usize, usize)>>,
    pub(crate) report: PatchedInterpolationReport,
}

/// The digest of one run.
pub(crate) fn run_digest(result: PatchedInterpolationResult<Name>, problem: &Problem) -> RunDigest {
    let patches = fingerprint(&result, problem);
    let zero_patches = result
        .report
        .zero_patches
        .iter()
        .map(|record| {
            let mut key: Vec<(usize, usize)> = record
                .projector
                .iter()
                .map(|(site, &value)| (problem.position(site), value))
                .collect();
            key.sort_unstable();
            key
        })
        .collect();
    RunDigest {
        patches,
        zero_patches,
        report: result.report,
    }
}

/// Two runs on separately built problems agree: every count and record
/// bitwise, the projectors (accepted and zero) by site position, the stored
/// patches by fingerprint, and the fields exempt from the determinism claim
/// as in [`assert_same_norm`].
pub(crate) fn assert_same_runs_across_problems(first: &RunDigest, second: &RunDigest) {
    let (a, b) = (&first.report, &second.report);
    assert_same_norm(&a.norm, &b.norm);
    assert_eq!(a.splits, b.splits);
    assert_eq!(a.function_evaluations, b.function_evaluations);
    assert_eq!(a.cache_hits, b.cache_hits);
    assert_eq!(a.measurement_evaluations, b.measurement_evaluations);
    assert_eq!(a.audit_evaluations, b.audit_evaluations);
    assert_eq!(a.verification_failures, b.verification_failures);
    assert_eq!(a.engine_retries, b.engine_retries);
    assert_eq!(first.zero_patches, second.zero_patches);
    assert_eq!(a.zero_patches.len(), b.zero_patches.len());
    for (x, y) in a.zero_patches.iter().zip(&b.zero_patches) {
        assert_eq!(x.acceptance, y.acceptance);
        assert_eq!(x.audit, y.audit);
    }
    assert_eq!(a.accepted.len(), b.accepted.len());
    for (x, y) in a.accepted.iter().zip(&b.accepted) {
        assert_eq!(x.termination, y.termination);
        assert_eq!(x.max_bond_dim, y.max_bond_dim);
        assert_eq!(x.retries_used, y.retries_used);
        assert_eq!(
            x.engine_error_estimate.to_bits(),
            y.engine_error_estimate.to_bits()
        );
        assert_eq!(
            x.max_sample_magnitude.to_bits(),
            y.max_sample_magnitude.to_bits()
        );
        assert_eq!(x.acceptance, y.acceptance);
        assert_eq!(x.audit, y.audit);
        assert_eq!(x.status, y.status);
    }
    // Accepted projectors by position, and the stored data.
    assert_eq!(first.patches, second.patches);
}
