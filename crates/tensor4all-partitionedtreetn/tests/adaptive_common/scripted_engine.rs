//! Scripted test engines for the L2 error contract, and an independent
//! implementation of the driver's measurement streams.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Mutex;

use tensor4all_core::{
    outer_product, ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, IndexLike,
    TensorElement,
};
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationOutcome, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};
use tensor4all_treetn::TreeTN;

use super::{columns_of, factorize_dense, full_domain};

/// A rank-one network over the problem's active sites with dimension-one
/// links: the product of `factor(position, coordinate)` over the sites, in
/// the problem's site order.
pub(crate) fn rank_one_network<T, V>(
    problem: &InterpolationProblem<V>,
    factor: impl Fn(usize, usize) -> T,
) -> TreeTN<IdxTensor, V>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let topology = problem.topology();
    let graph = topology.graph();
    let mut links: HashMap<V, Vec<DynIndex>> = HashMap::new();
    for edge in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge).unwrap();
        let link = DynIndex::new_dyn(1);
        for node in [a, b] {
            let name = topology.node_name(node).unwrap().clone();
            links.entry(name).or_default().push(link.clone());
        }
    }
    let mut position = 0;
    let mut names = Vec::new();
    let mut tensors = Vec::new();
    for (node, sites) in problem.node_sites() {
        let mut tensor = IdxTensor::from_dense(
            links.remove(node).unwrap_or_default(),
            vec![T::from_f64(1.0)],
        )
        .unwrap();
        for site in sites {
            let values = (0..site.dim()).map(|x| factor(position, x)).collect();
            let factor = IdxTensor::from_dense(vec![site.clone()], values).unwrap();
            tensor = outer_product(&tensor, &factor).unwrap();
            position += 1;
        }
        names.push(node.clone());
        tensors.push(tensor);
    }
    TreeTN::from_tensors(tensors, names).unwrap()
}

/// The network a scripted engine call returns.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Network {
    /// The exact dense factorization of the patch.
    Exact,
    /// The constant function with this value (rank one).
    Constant(f64),
    /// A rank-one network with a NaN entry.
    NonFinite,
}

/// One step of a scripted engine: what the call returns.
#[derive(Clone, Debug)]
pub(crate) struct Step {
    pub(crate) network: Network,
    pub(crate) termination: InterpolationTermination,
    /// Returned pivots in active coordinates.
    pub(crate) pivots: Vec<Vec<usize>>,
    /// The returned engine error estimate.
    pub(crate) error_estimate: f64,
}

impl Step {
    pub(crate) fn converged(network: Network) -> Self {
        Self {
            network,
            termination: InterpolationTermination::Converged,
            pivots: Vec::new(),
            error_estimate: 0.0,
        }
    }

    /// A `BondCapReached` step returning the zero function.
    pub(crate) fn capped() -> Self {
        Self::capped_with(Network::Constant(0.0))
    }

    /// A `BondCapReached` step returning `network`.
    pub(crate) fn capped_with(network: Network) -> Self {
        Self {
            network,
            termination: InterpolationTermination::BondCapReached,
            pivots: Vec::new(),
            error_estimate: 0.0,
        }
    }

    /// The same step with another termination.
    pub(crate) fn with_termination(mut self, termination: InterpolationTermination) -> Self {
        self.termination = termination;
        self
    }

    pub(crate) fn with_pivots(mut self, pivots: Vec<Vec<usize>>) -> Self {
        self.pivots = pivots;
        self
    }

    pub(crate) fn with_error_estimate(mut self, error_estimate: f64) -> Self {
        self.error_estimate = error_estimate;
        self
    }
}

/// One call seen by a test engine.
#[derive(Clone, Debug)]
pub(crate) struct Call {
    pub(crate) site_order: Vec<DynIndex>,
    pub(crate) initial_pivots: Vec<Vec<usize>>,
    pub(crate) tolerance: f64,
    pub(crate) seed: u64,
}

fn record_call<V>(problem: &InterpolationProblem<V>) -> Call
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    Call {
        site_order: problem.site_order().to_vec(),
        initial_pivots: columns_of(problem.initial_pivots()),
        tolerance: problem.absolute_tolerance(),
        seed: problem.seed(),
    }
}

fn pivot_array(pivots: &[Vec<usize>], n_active: usize) -> Option<ColMajorArray<usize>> {
    if pivots.is_empty() {
        return None;
    }
    Some(ColMajorArray::new(pivots.concat(), vec![n_active, pivots.len()]).unwrap())
}

/// Returns the step of its call index (the last step repeats) and records
/// every call. It samples its initial pivots like a real engine.
pub(crate) struct ScriptedEngine {
    steps: Vec<Step>,
    calls: Mutex<Vec<Call>>,
}

impl ScriptedEngine {
    pub(crate) fn new(steps: Vec<Step>) -> Self {
        Self {
            steps,
            calls: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl<T> TreeInterpolator<T> for ScriptedEngine
where
    T: CommonScalar + TensorElement,
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
        let evaluator = |source: anyhow::Error| InterpolationError::Evaluator { source };
        let index = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(record_call(problem));
            calls.len() - 1
        };
        let step = self.steps[index.min(self.steps.len() - 1)].clone();
        let initial = evaluate(problem.initial_pivots().as_ref()).map_err(evaluator)?;
        let n_active = problem.site_order().len();
        let network = match step.network {
            Network::Exact => {
                let dims: Vec<usize> = problem.site_order().iter().map(IndexLike::dim).collect();
                let domain: Vec<usize> = full_domain(&dims).concat();
                let shape = [n_active, domain.len() / n_active];
                let values =
                    evaluate(ColMajorArrayRef::new(&domain, &shape).unwrap()).map_err(evaluator)?;
                factorize_dense(problem, values, false)?
            }
            Network::Constant(value) => rank_one_network(problem, |position, _| {
                T::from_f64(if position == 0 { value } else { 1.0 })
            }),
            Network::NonFinite => rank_one_network(problem, |position, x| {
                T::from_f64(if position == 0 && x == 0 {
                    f64::NAN
                } else {
                    1.0
                })
            }),
        };
        Ok(InterpolationOutcome {
            network,
            termination: step.termination,
            error_estimate: step.error_estimate,
            max_sample_magnitude: initial.iter().map(|v| v.abs_val()).fold(0.0, f64::max),
            pivots: pivot_array(&step.pivots, n_active),
        })
    }
}

/// A dense engine that misses a narrow feature: unless the feature point
/// (restricted to the active sites) is among its initial pivots, it replaces
/// the value there by the rank-one product formula through the opposite
/// corner, which reproduces any product function. With the point among its
/// pivots it factorizes the exact data.
pub(crate) struct SpikeBlindEngine {
    /// The feature point as full-domain coordinates by site.
    spike: HashMap<DynIndex, usize>,
    calls: Mutex<Vec<Call>>,
}

impl SpikeBlindEngine {
    pub(crate) fn new(sites: &[DynIndex], spike: &[usize]) -> Self {
        Self {
            spike: sites.iter().cloned().zip(spike.iter().copied()).collect(),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl TreeInterpolator<f64> for SpikeBlindEngine {
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>>,
    {
        let evaluator = |source: anyhow::Error| InterpolationError::Evaluator { source };
        let call = record_call(problem);
        self.calls.lock().unwrap().push(call.clone());
        let sites = problem.site_order();
        let dims: Vec<usize> = sites.iter().map(IndexLike::dim).collect();
        let n_active = dims.len();
        let spike: Vec<usize> = sites.iter().map(|site| self.spike[site]).collect();
        let seen = call.initial_pivots.contains(&spike);
        evaluate(problem.initial_pivots().as_ref()).map_err(evaluator)?;

        let points = full_domain(&dims);
        let flat: Vec<usize> = points.concat();
        let shape = [n_active, points.len()];
        let mut values =
            evaluate(ColMajorArrayRef::new(&flat, &shape).unwrap()).map_err(evaluator)?;
        if !seen {
            // g(x*) = g(p) prod_i g(p with x*_i) / g(p), p the opposite corner.
            let corner: Vec<usize> = spike
                .iter()
                .zip(&dims)
                .map(|(&x, &d)| (x + 1) % d)
                .collect();
            let value_at =
                |point: &[usize]| values[points.iter().position(|p| p == point).unwrap()];
            let base = value_at(&corner);
            let mut filled = base;
            for i in 0..n_active {
                let mut point = corner.clone();
                point[i] = spike[i];
                filled *= value_at(&point) / base;
            }
            let index = points.iter().position(|p| *p == spike).unwrap();
            values[index] = filled;
        }
        let max_sample_magnitude = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let network = factorize_dense(problem, values, false)?;
        let rank = network.link_dims().into_iter().max().unwrap_or(1);
        let termination = if problem.max_bond_dim().is_some_and(|cap| rank >= cap.get()) {
            InterpolationTermination::BondCapReached
        } else {
            InterpolationTermination::Converged
        };
        Ok(InterpolationOutcome {
            network,
            termination,
            error_estimate: 0.0,
            max_sample_magnitude,
            pivots: pivot_array(&call.initial_pivots, n_active),
        })
    }
}

/// An independent implementation of the driver's documented streams:
/// SplitMix64, Lemire's draw, the path state, and the stream selectors.
pub(crate) mod streams {
    const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
    const PATH_DOMAIN: u64 = 0x7472_6565_7061_7463;
    const VERIFY_STREAM: u64 = 0x7665_7269_6679_7374;
    const AUDIT_STREAM: u64 = 0x6175_6469_7473_7472;
    const ZERO_SCREEN_STREAM: u64 = 0x7a65_726f_7363_726e;
    const SCALE_STREAM: u64 = 0x7363_616c_6573_7472;

    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(GAMMA);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn mix(value: u64) -> u64 {
        let mut state = value;
        next(&mut state)
    }

    fn below(state: &mut u64, bound: u64) -> u64 {
        let mut product = u128::from(next(state)) * u128::from(bound);
        if (product as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (product as u64) < threshold {
                product = u128::from(next(state)) * u128::from(bound);
            }
        }
        (product >> 64) as u64
    }

    /// The path state of a patch.
    pub(crate) fn path_state(seed: u64, path: &[(usize, usize)]) -> u64 {
        path.iter().fold(mix(seed ^ PATH_DOMAIN), |state, &(p, v)| {
            mix(mix(state ^ p as u64) ^ v as u64)
        })
    }

    pub(crate) fn verify(state: u64, attempt: usize) -> u64 {
        mix(mix(state ^ VERIFY_STREAM) ^ attempt as u64)
    }

    pub(crate) fn audit(state: u64) -> u64 {
        mix(state ^ AUDIT_STREAM)
    }

    pub(crate) fn zero_screen(state: u64) -> u64 {
        mix(state ^ ZERO_SCREEN_STREAM)
    }

    pub(crate) fn scale(state: u64) -> u64 {
        mix(state ^ SCALE_STREAM)
    }

    /// `count` points with coordinates drawn in active-site order.
    pub(crate) fn draw(dims: &[usize], count: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                dims.iter()
                    .map(|&d| below(&mut state, d as u64) as usize)
                    .collect()
            })
            .collect()
    }
}
