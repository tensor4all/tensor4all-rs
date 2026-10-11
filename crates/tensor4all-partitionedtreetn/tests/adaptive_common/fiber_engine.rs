//! A test engine for product functions, without dense evaluation.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{outer_product, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::interpolation::{
    InterpolationError, InterpolationOutcome, InterpolationProblem, InterpolationTermination,
    TreeInterpolator,
};
use tensor4all_treetn::TreeTN;

/// A test engine for product functions: it samples the fibers through the
/// first initial pivot `p` and returns the exact rank-one network
/// `f(p) * prod_i f(p with site i = x_i) / f(p)`, without dense evaluation.
pub(crate) struct FiberEngine;

impl TreeInterpolator<f64> for FiberEngine {
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
        let pivot = problem.initial_pivots().column(0).unwrap().to_vec();
        let n = pivot.len();
        let dims: Vec<usize> = problem.site_order().iter().map(IndexLike::dim).collect();
        let mut fibers = Vec::new();
        for (i, &dim) in dims.iter().enumerate() {
            for x in 0..dim {
                let mut point = pivot.clone();
                point[i] = x;
                fibers.extend(point);
            }
        }
        let shape = [n, fibers.len() / n];
        let values =
            evaluate(ColMajorArrayRef::new(&fibers, &shape).unwrap()).map_err(evaluator)?;
        let center = values[pivot[0]];
        let mut factors = Vec::new();
        let mut offset = 0;
        for &dim in &dims {
            factors.push(
                values[offset..offset + dim]
                    .iter()
                    .map(|v| v / center)
                    .collect::<Vec<_>>(),
            );
            offset += dim;
        }
        factors[0].iter_mut().for_each(|v| *v *= center);

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
            let mut tensor =
                IdxTensor::from_dense(links.remove(node).unwrap_or_default(), vec![1.0]).unwrap();
            for site in sites {
                let factor =
                    IdxTensor::from_dense(vec![site.clone()], factors[position].clone()).unwrap();
                tensor = outer_product(&tensor, &factor).unwrap();
                position += 1;
            }
            names.push(node.clone());
            tensors.push(tensor);
        }
        Ok(InterpolationOutcome {
            network: TreeTN::from_tensors(tensors, names).unwrap(),
            termination: InterpolationTermination::Converged,
            error_estimate: 0.0,
            max_sample_magnitude: values.iter().fold(0.0, |m, v| m.max(v.abs())),
            pivots: None,
        })
    }
}
