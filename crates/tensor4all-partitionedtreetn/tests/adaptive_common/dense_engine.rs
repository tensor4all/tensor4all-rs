//! The dense test engine shared by the adaptive interpolation tests.
//!
//! It evaluates the whole active domain, factorizes it exactly, and reports
//! `BondCapReached` when the exact rank reaches the cap, so splitting is
//! tested independently of any real engine. Above the cap its network
//! exceeds the cap, which violates the M1 contract (an outcome network's
//! bonds never exceed the cap); the patch-size tests use that on purpose as
//! an engine fault, and every other test splits such a patch unused.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Mutex;

use tensor4all_core::{
    contract_pair, outer_product, ColMajorArray, ColMajorArrayRef, CommonScalar, DynIndex,
    FactorizeOptions, IdxTensor, IndexLike, TensorElement,
};
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationOutcome, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};
use tensor4all_treetn::{factorize_tensor_to_treetn_with, TreeTN, TreeTopology};

use super::full_domain;

/// Misbehavior injected into the dense test engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    None,
    /// Return a network whose first active site has another identity.
    WrongLayout,
    /// Return pivots with a wrong number of rows.
    BadPivots,
    /// Report `AllSamplesZero` although the driver screened the patch.
    AllZero,
    /// Stop with `IterationLimit` instead of converging.
    IterationLimit,
    /// Report `Converged` whatever the rank, even at the cap.
    ConvergedAtCap,
    /// Sample only the initial pivots and stop at the cap with an empty
    /// network: unused when the driver splits, an engine fault when it
    /// retains the patch.
    CapAfterPivots,
    /// Send a batch with the wrong number of rows.
    BadBatch,
    /// Send a batch with an out-of-range coordinate.
    OutOfRange,
}

/// One problem seen by the dense test engine.
#[derive(Clone, Debug)]
pub(crate) struct Seen {
    pub(crate) site_order: Vec<DynIndex>,
    pub(crate) initial_pivots: Vec<Vec<usize>>,
    pub(crate) returned_pivots: Vec<Vec<usize>>,
    pub(crate) seed: u64,
    pub(crate) tolerance: f64,
}

/// Evaluates the whole active domain, factorizes it exactly (SVD with the
/// default relative threshold), and reports `BondCapReached` when the exact
/// rank reaches the cap; above the cap its network is not truncated (see the
/// module documentation). Nodes without active sites are supported through a
/// temporary dimension-one site that is contracted away afterwards.
pub(crate) struct DenseEngine {
    fault: Fault,
    seen: Mutex<Vec<Seen>>,
}

impl DenseEngine {
    pub(crate) fn new() -> Self {
        Self::with_fault(Fault::None)
    }

    pub(crate) fn with_fault(fault: Fault) -> Self {
        Self {
            fault,
            seen: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

pub(crate) fn columns_of(array: &ColMajorArray<usize>) -> Vec<Vec<usize>> {
    (0..array.ncols().unwrap_or(0))
        .map(|column| array.column(column).unwrap().to_vec())
        .collect()
}

impl<T> TreeInterpolator<T> for DenseEngine
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
        let site_order = problem.site_order().to_vec();
        let dims: Vec<usize> = site_order.iter().map(IndexLike::dim).collect();
        let n_active = dims.len();

        match self.fault {
            Fault::AllZero => return Err(InterpolationError::AllSamplesZero),
            Fault::BadBatch => {
                let data = vec![0; n_active + 1];
                let shape = [n_active + 1, 1];
                evaluate(ColMajorArrayRef::new(&data, &shape).unwrap()).map_err(evaluator)?;
            }
            Fault::OutOfRange => {
                let data = dims.clone();
                let shape = [n_active, 1];
                evaluate(ColMajorArrayRef::new(&data, &shape).unwrap()).map_err(evaluator)?;
            }
            _ => {}
        }

        let initial = evaluate(problem.initial_pivots().as_ref()).map_err(evaluator)?;
        if initial.iter().all(|value| value.abs_val() == 0.0) {
            return Err(InterpolationError::AllSamplesZero);
        }
        if self.fault == Fault::CapAfterPivots {
            return Ok(InterpolationOutcome {
                network: TreeTN::new(),
                termination: InterpolationTermination::BondCapReached,
                error_estimate: 1.0,
                max_sample_magnitude: 1.0,
                pivots: None,
            });
        }
        let domain: Vec<usize> = full_domain(&dims).concat();
        let shape = [n_active, domain.len() / n_active];
        let values =
            evaluate(ColMajorArrayRef::new(&domain, &shape).unwrap()).map_err(evaluator)?;
        let (argmax, max_sample_magnitude) =
            values.iter().map(|value| value.abs_val()).enumerate().fold(
                (0, 0.0_f64),
                |best, (i, m)| if m > best.1 { (i, m) } else { best },
            );

        let network = factorize_dense(problem, values, self.fault == Fault::WrongLayout)?;
        let rank = network.link_dims().into_iter().max().unwrap_or(1);
        let termination = if self.fault == Fault::IterationLimit {
            InterpolationTermination::IterationLimit
        } else if self.fault == Fault::ConvergedAtCap {
            InterpolationTermination::Converged
        } else if problem.max_bond_dim().is_some_and(|cap| rank >= cap.get()) {
            InterpolationTermination::BondCapReached
        } else {
            InterpolationTermination::Converged
        };

        let mut returned = columns_of(problem.initial_pivots());
        returned.push(full_domain(&dims)[argmax].clone());
        let rows = if self.fault == Fault::BadPivots {
            n_active + 1
        } else {
            n_active
        };
        let flat: Vec<usize> = returned
            .iter()
            .flat_map(|p| {
                p.iter()
                    .copied()
                    .chain(std::iter::repeat_n(0, rows - n_active))
            })
            .collect();
        let pivots = ColMajorArray::new(flat, vec![rows, returned.len()]).unwrap();
        self.seen.lock().unwrap().push(Seen {
            site_order,
            initial_pivots: columns_of(problem.initial_pivots()),
            returned_pivots: returned,
            seed: problem.seed(),
            tolerance: problem.absolute_tolerance(),
        });
        Ok(InterpolationOutcome {
            network,
            termination,
            error_estimate: 0.0,
            max_sample_magnitude,
            pivots: Some(pivots),
        })
    }
}

/// The exact network of dense active-domain `values` (column-major in the
/// problem's site order), factorized by SVD with the default threshold.
/// Nodes without active sites get a temporary dimension-one site that is
/// contracted away. With `wrong_layout`, the first active site is replaced
/// by a similar index of another identity.
pub(crate) fn factorize_dense<T, V>(
    problem: &InterpolationProblem<V>,
    values: Vec<T>,
    wrong_layout: bool,
) -> Result<TreeTN<IdxTensor, V>, InterpolationError>
where
    T: CommonScalar + TensorElement,
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let engine = |source: anyhow::Error| InterpolationError::Engine { source };
    let site_order = problem.site_order().to_vec();
    let mut indices = site_order.clone();
    if wrong_layout {
        indices[0] = indices[0].sim();
    }
    let mut dense = IdxTensor::from_dense(indices.clone(), values).map_err(|e| engine(e.into()))?;
    let one = vec![T::from_f64(1.0)];
    let mut nodes = HashMap::new();
    let mut dummies = HashMap::new();
    let mut position = 0;
    for (node, sites) in problem.node_sites() {
        let node_indices = indices[position..position + sites.len()].to_vec();
        position += sites.len();
        if node_indices.is_empty() {
            let dummy = DynIndex::new_dyn(1);
            let ones = IdxTensor::from_dense(vec![dummy.clone()], one.clone()).unwrap();
            dense = outer_product(&dense, &ones).map_err(|e| engine(e.into()))?;
            nodes.insert(node.clone(), vec![dummy]);
            dummies.insert(node.clone(), ones);
        } else {
            nodes.insert(node.clone(), node_indices);
        }
    }
    let topology = problem.topology();
    let graph = topology.graph();
    let edges = graph
        .edge_indices()
        .map(|edge| {
            let (a, b) = graph.edge_endpoints(edge).unwrap();
            (
                topology.node_name(a).unwrap().clone(),
                topology.node_name(b).unwrap().clone(),
            )
        })
        .collect();
    let root = problem.node_sites().keys().next().unwrap();
    let factorized = factorize_tensor_to_treetn_with(
        &dense,
        &TreeTopology::new(nodes, edges),
        FactorizeOptions::svd(),
        root,
    )
    .map_err(|e| engine(e.into()))?;
    let names: Vec<V> = problem.node_sites().keys().cloned().collect();
    let tensors = names
        .iter()
        .map(|name| {
            let tensor = factorized
                .tensor(factorized.node_index(name).unwrap())
                .unwrap();
            match dummies.get(name) {
                Some(ones) => contract_pair(tensor, ones).unwrap(),
                None => tensor.clone(),
            }
        })
        .collect();
    let network = TreeTN::from_tensors(tensors, names).map_err(|e| engine(e.into()))?;

    Ok(network)
}
