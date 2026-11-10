//! Tests of the element-wise interpolation contract.

use std::collections::BTreeMap;

use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{explicit::Session, CpuBackend};

use super::*;
use crate::interpolation::{InterpolationTermination, Parallelism};
use crate::NodeNameNetwork;

/// A black-box engine for one node: it evaluates every point of the node's
/// sites (first site fastest) and stores them as one tensor.
struct Exhaustive;

impl TreeInterpolator<f64> for Exhaustive {
    fn interpolate<V, F>(
        &self,
        problem: &InterpolationProblem<V>,
        ctx: &mut EvalContext<'_>,
        mut evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
        F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<f64>>,
    {
        let sites = problem.site_order().to_vec();
        let total: usize = sites.iter().map(|site| site.dim).product();
        let mut points = Vec::with_capacity(total * sites.len());
        for mut position in 0..total {
            for site in &sites {
                points.push(position % site.dim);
                position /= site.dim;
            }
        }
        let shape = [sites.len(), total];
        let batch = ColMajorArrayRef::new(&points, &shape).unwrap();
        let values =
            evaluate(batch, ctx).map_err(|source| InterpolationError::Evaluator { source })?;
        let node = problem.node_sites().keys().next().unwrap().clone();
        let tensor = IdxTensor::from_dense(sites, values).unwrap();
        Ok(InterpolationOutcome {
            network: TreeTN::from_tensors(vec![tensor], vec![node]).unwrap(),
            termination: InterpolationTermination::Converged,
            error_estimate: 0.0,
            max_sample_magnitude: 0.0,
            pivots: None,
        })
    }
}

fn problem(sites: &[DynIndex]) -> InterpolationProblem<usize> {
    let mut topology = NodeNameNetwork::new();
    topology.add_node(0usize).unwrap();
    InterpolationProblem::new(
        topology,
        BTreeMap::from([(0usize, sites.to_vec())]),
        ColMajorArray::new(vec![0; sites.len()], vec![sites.len(), 1]).unwrap(),
        1e-12,
        None,
        0,
    )
    .unwrap()
}

fn network(sites: &[DynIndex], values: Vec<f64>) -> TreeTN<IdxTensor, usize> {
    TreeTN::from_tensors(
        vec![IdxTensor::from_dense(sites.to_vec(), values).unwrap()],
        vec![0usize],
    )
    .unwrap()
}

fn run<Op: ElementwiseOperator<f64>>(
    sites: &[DynIndex],
    inputs: &[TreeTN<IdxTensor, usize>],
    operator: &Op,
) -> Result<InterpolationOutcome<usize>, InterpolationError> {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
    PointwiseElementwise(Exhaustive).interpolate_elementwise(
        &problem(sites),
        inputs,
        operator,
        &mut ctx,
    )
}

#[test]
fn pointwise_engine_applies_the_operator_to_every_input_point() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(3)];
    let a: Vec<f64> = (0..6).map(|i| 1.0 + i as f64).collect();
    let b: Vec<f64> = (0..6).map(|i| 0.5 - i as f64).collect();
    let c: Vec<f64> = (0..6).map(|i| (i as f64).sin()).collect();
    let inputs = [
        network(&sites, a.clone()),
        network(&sites, b.clone()),
        network(&sites, c.clone()),
    ];

    let product = run(&sites, &inputs, &Product).unwrap();
    let expected: Vec<f64> = (0..6).map(|i| a[i] * b[i] * c[i]).collect();
    assert_eq!(
        product.network.to_dense().unwrap().to_vec::<f64>().unwrap(),
        expected
    );

    // A closure operator: f(a, b, c) = a / (1 + b^2) - c.
    let rational = |inputs: ColMajorArrayRef<'_, f64>, output: &mut [f64]| -> anyhow::Result<()> {
        for (column, value) in inputs.data().chunks(3).zip(output.iter_mut()) {
            *value = column[0] / (1.0 + column[1] * column[1]) - column[2];
        }
        Ok(())
    };
    let outcome = run(&sites, &inputs, &rational).unwrap();
    let expected: Vec<f64> = (0..6).map(|i| a[i] / (1.0 + b[i] * b[i]) - c[i]).collect();
    assert_eq!(
        outcome.network.to_dense().unwrap().to_vec::<f64>().unwrap(),
        expected
    );
}

#[test]
fn pointwise_engine_reports_missing_inputs_and_operator_failures() {
    let sites = [DynIndex::new_dyn(2)];
    let error = run(&sites, &[], &Product).unwrap_err();
    assert!(
        matches!(error, InterpolationError::InvalidProblem { .. }),
        "{error}"
    );

    let failing = |_: ColMajorArrayRef<'_, f64>, _: &mut [f64]| -> anyhow::Result<()> {
        anyhow::bail!("operator refused")
    };
    let error = run(&sites, &[network(&sites, vec![1.0, 2.0])], &failing).unwrap_err();
    assert!(
        matches!(error, InterpolationError::Evaluator { .. }),
        "{error}"
    );
    assert!(
        format!("{error:#}").contains("operator refused")
            || error.to_string().contains("evaluator")
    );
}

#[test]
fn product_rejects_malformed_batches() {
    let values = [1.0_f64, 2.0];
    let mut output = [0.0; 2];
    let empty: [f64; 0] = [];
    assert!(Product
        .apply(ColMajorArrayRef::new(&empty, &[0, 2]).unwrap(), &mut output)
        .is_err());
    assert!(Product
        .apply(
            ColMajorArrayRef::new(&values, &[2, 1]).unwrap(),
            &mut output
        )
        .is_err());
    assert!(Product
        .apply(ColMajorArrayRef::new(&values, &[2]).unwrap(), &mut output)
        .is_err());
    let mut one = [0.0];
    Product
        .apply(ColMajorArrayRef::new(&values, &[2, 1]).unwrap(), &mut one)
        .unwrap();
    assert_eq!(one, [2.0]);
}
