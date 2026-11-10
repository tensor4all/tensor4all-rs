//! Tests of patched element-wise operations on a star with a degree-3 hub.
//!
//! The inputs are pseudo-random bond-dimension-2 networks masked onto
//! irregular layouts with gaps. The oracle evaluates the inputs with
//! `PartitionedTreeTN::evaluate_batched` at every point and applies `f`.

use std::collections::BTreeMap;

use tensor4all_core::{AnyScalar, ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
use tensor4all_treeaci::TreeAciInterpolator;
use tensor4all_treersi::TreeRsiInterpolator;
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{
    ElementwiseInterpolator, EvalContext, InterpolationTermination, Parallelism,
    PointwiseElementwise, Product,
};
use tensor4all_treetn::TreeTN;

use super::*;
use crate::adaptive_interpolation::{
    GlobalL2Error, MeasurementMethod, NormReport, VerificationOptions,
};
use crate::{ErrorNorm, ErrorTolerance, L2Reference};

/// Hub 0 with leaves 1, 2, 3; two binary sites per node, in node order.
struct Star {
    sites: Vec<DynIndex>,
    per_node: usize,
}

impl Star {
    fn new() -> Self {
        Self {
            sites: (0..8).map(|_| DynIndex::new_dyn(2)).collect(),
            per_node: 2,
        }
    }

    /// A pseudo-random network with bond dimension 2 and entries in
    /// `offset + [-1, 1)`.
    fn network(&self, seed: u64, offset: f64) -> TreeTN<IdxTensor, usize> {
        let bonds: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
        let fill = |legs: &[DynIndex], salt: u64| -> IdxTensor {
            let len: usize = legs.iter().map(|leg| leg.dim).product();
            let values = (0..len)
                .map(|i| {
                    let state = (seed * 1_000 + salt * 100 + i as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left(29);
                    offset + (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
                })
                .collect();
            IdxTensor::from_dense(legs.to_vec(), values).unwrap()
        };
        let node_sites =
            |node: usize| self.sites[node * self.per_node..(node + 1) * self.per_node].to_vec();
        let mut hub = node_sites(0);
        hub.extend(bonds.iter().cloned());
        let mut tensors = vec![fill(&hub, 0)];
        for leaf in 1..4 {
            let mut legs = vec![bonds[leaf - 1].clone()];
            legs.extend(node_sites(leaf));
            tensors.push(fill(&legs, leaf as u64));
        }
        TreeTN::from_tensors(tensors, vec![0usize, 1, 2, 3]).unwrap()
    }

    fn projector(&self, pairs: &[(usize, usize)]) -> Projector {
        Projector::from_pairs(
            pairs
                .iter()
                .map(|&(site, value)| (self.sites[site].clone(), value)),
        )
        .unwrap()
    }

    /// `network` masked onto the patches of `layout`.
    fn partition(
        &self,
        network: &TreeTN<IdxTensor, usize>,
        layout: &[&[(usize, usize)]],
    ) -> PartitionedTreeTN<usize> {
        PartitionedTreeTN::from_subdomains(
            layout
                .iter()
                .map(|pairs| SubDomainTreeTN::new(network.clone(), self.projector(pairs)).unwrap())
                .collect(),
        )
        .unwrap()
    }

    /// The values of `partition` at every point, site `k` read from bit `k`.
    fn values(&self, partition: &PartitionedTreeTN<usize>) -> Vec<f64> {
        let n = self.sites.len();
        let grid: Vec<usize> = (0..1usize << n)
            .flat_map(|i| (0..n).map(move |k| (i >> k) & 1))
            .collect();
        partition
            .evaluate_batched(
                &self.sites,
                ColMajorArrayRef::new(&grid, &[n, 1 << n]).unwrap(),
            )
            .unwrap()
            .iter()
            .map(AnyScalar::real)
            .collect()
    }
}

/// With `a`, `b`, `c` the first sites of nodes 0, 1, 2: x is split on `a`,
/// then `{a = 1}` on `b`. y has a patch `{c = 0}` that crosses x's layout, a
/// patch `{c = 1, a = 1}`, and a gap `{c = 1, a = 0}`.
fn inputs(star: &Star) -> (PartitionedTreeTN<usize>, PartitionedTreeTN<usize>) {
    let (a, b, c) = (0, star.per_node, 2 * star.per_node);
    let x = star.partition(
        &star.network(1, 0.0),
        &[&[(a, 0)], &[(a, 1), (b, 0)], &[(a, 1), (b, 1)]],
    );
    let y = star.partition(&star.network(2, 1.5), &[&[(c, 0)], &[(c, 1), (a, 1)]]);
    (x, y)
}

/// The sites the inputs fix first, then the others, so that the structural
/// splits stop at depth three.
fn split_first(star: &Star) -> Vec<DynIndex> {
    let fixed = [0, star.per_node, 2 * star.per_node];
    fixed
        .iter()
        .map(|&site| star.sites[site].clone())
        .chain(
            (0..star.sites.len())
                .filter(|site| !fixed.contains(site))
                .map(|site| star.sites[site].clone()),
        )
        .collect()
}

fn options(reference: f64, rtol: f64) -> PatchedInterpolationOptions {
    PatchedInterpolationOptions::new(8)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(reference)))
        .with_tolerance(ErrorTolerance { rtol, atol: 0.0 })
        .with_verification(VerificationOptions::new().with_max_exhaustive_points(1 << 12))
}

fn run<Op: ElementwiseOperator<f64>>(
    inputs: &[&PartitionedTreeTN<usize>],
    operator: &Op,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<usize>, PatchedInterpolationError> {
    run_with(
        &PointwiseElementwise(TreeTciInterpolator::default()),
        inputs,
        operator,
        options,
    )
}

/// [`run`] with another element-wise engine.
fn run_with<E, Op>(
    engine: &E,
    inputs: &[&PartitionedTreeTN<usize>],
    operator: &Op,
    options: &PatchedInterpolationOptions,
) -> Result<PatchedInterpolationResult<usize>, PatchedInterpolationError>
where
    E: ElementwiseInterpolator<f64, Op> + Sync,
    Op: ElementwiseOperator<f64>,
{
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
    patched_elementwise(
        engine,
        &TreeTciInterpolator::default(),
        inputs,
        operator,
        &mut ctx,
        options,
    )
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn pointwise(
    star: &Star,
    inputs: &[&PartitionedTreeTN<usize>],
    f: impl Fn(&[f64]) -> f64,
) -> Vec<f64> {
    let values: Vec<Vec<f64>> = inputs.iter().map(|input| star.values(input)).collect();
    (0..values[0].len())
        .map(|point| f(&values.iter().map(|v| v[point]).collect::<Vec<_>>()))
        .collect()
}

/// The dense error is within the run's `delta`, and the result is certified
/// (every measurement here is exhaustive).
fn check(star: &Star, result: &PatchedInterpolationResult<usize>, expected: &[f64], rtol: f64) {
    let actual = star.values(&result.partition);
    let error: Vec<f64> = actual.iter().zip(expected).map(|(a, e)| a - e).collect();
    let delta = rtol * l2(expected);
    assert!(
        l2(&error) <= delta * (1.0 + 1e-9),
        "L2 error {:e} above delta {delta:e}",
        l2(&error)
    );
    let NormReport::L2 { error, .. } = &result.report.norm else {
        panic!("an L2 run");
    };
    assert!(
        matches!(error.global, GlobalL2Error::Certified { .. }),
        "{:?}",
        error.global
    );
}

/// Every patch with more than one active site lies inside one patch of each
/// input, or outside all of its patches; an exact patch may cross inputs.
fn assert_resolvable(
    result: &PatchedInterpolationResult<usize>,
    inputs: &[&PartitionedTreeTN<usize>],
) {
    let regions = result
        .report
        .accepted
        .iter()
        .map(|record| &record.projector)
        .chain(
            result
                .report
                .zero_patches
                .iter()
                .map(|record| &record.projector),
        );
    for region in regions.filter(|region| region.len() < 7) {
        for input in inputs {
            let overlapping: Vec<&Projector> = input
                .projectors()
                .filter(|patch| patch.is_compatible_with(region))
                .collect();
            assert!(
                overlapping.is_empty()
                    || (overlapping.len() == 1 && region.is_subset_of(overlapping[0])),
                "{region:?} is not resolvable"
            );
        }
    }
}

#[test]
fn a_product_matches_the_pointwise_product_and_is_zero_in_the_gap() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    let expected = pointwise(&star, &[&x, &y], |v| v[0] * v[1]);
    let rtol = 1e-8;
    let result = run(&[&x, &y], &Product, &options(l2(&expected), rtol)).unwrap();
    check(&star, &result, &expected, rtol);
    assert_resolvable(&result, &[&x, &y]);
    // y has no patch on {s0 = 0, s4 = 1}: that region is a measured zero patch.
    let gap = star.projector(&[(0, 0), (4, 1)]);
    assert!(result
        .report
        .zero_patches
        .iter()
        .any(|record| record.projector.is_subset_of(&gap)));
    assert!(result.report.splits > 0);
    // The result carries the run's contract, so M6b's merge applies to it.
    let merged = crate::adaptive_interpolation::merge_siblings(
        result,
        &0,
        &crate::adaptive_interpolation::SiblingMergeOptions::new(),
    )
    .unwrap();
    let actual = star.values(&merged.partition);
    let error: Vec<f64> = actual.iter().zip(&expected).map(|(a, e)| a - e).collect();
    assert!(l2(&error) <= rtol * l2(&expected) * (1.0 + 1e-9));
}

/// B0: the same product through `patched_interpolate` with a black-box
/// evaluator that evaluates the inputs pointwise. It needs no element-wise
/// engine and meets the same contract; it is the oracle of the kernels.
#[test]
fn the_black_box_route_meets_the_same_contract() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    let expected = pointwise(&star, &[&x, &y], |v| v[0] * v[1]);
    let rtol = 1e-8;
    let options = options(l2(&expected), rtol);
    let mut topology = tensor4all_treetn::NodeNameNetwork::new();
    for node in 0..4usize {
        topology.add_node(node).unwrap();
    }
    for leaf in 1..4usize {
        topology.add_edge(&0, &leaf).unwrap();
    }
    let node_sites: BTreeMap<usize, Vec<DynIndex>> = (0..4)
        .map(|node| (node, star.sites[2 * node..2 * node + 2].to_vec()))
        .collect();
    let sites = star.sites.clone();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
    let result = crate::adaptive_interpolation::patched_interpolate(
        &TreeTciInterpolator::default(),
        topology,
        node_sites,
        ColMajorArray::new(vec![], vec![8, 0]).unwrap(),
        &mut ctx,
        |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| {
            let a = x.evaluate_batched(&sites, batch)?;
            let b = y.evaluate_batched(&sites, batch)?;
            Ok(a.iter().zip(&b).map(|(a, b)| a.real() * b.real()).collect())
        },
        &options,
    )
    .unwrap();
    check(&star, &result, &expected, rtol);
}

#[test]
fn a_nonlinear_map_covers_the_regions_where_an_input_is_absent() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    // f(x, y) = 1 / (4 - x y) is 1 / 4 where y has no patch, so the result
    // must cover the gap.
    let rational = |inputs: ColMajorArrayRef<'_, f64>, output: &mut [f64]| -> anyhow::Result<()> {
        for (column, value) in inputs.data().chunks(2).zip(output.iter_mut()) {
            *value = 1.0 / (4.0 - column[0] * column[1]);
        }
        Ok(())
    };
    let expected = pointwise(&star, &[&x, &y], |v| 1.0 / (4.0 - v[0] * v[1]));
    let rtol = 1e-8;
    let result = run(&[&x, &y], &rational, &options(l2(&expected), rtol)).unwrap();
    check(&star, &result, &expected, rtol);
    assert_resolvable(&result, &[&x, &y]);
    assert!(result.report.zero_patches.is_empty());
}

/// The same product with tree ACI as the per-patch kernel, against the
/// dense oracle: the kernel receives the restricted input networks.
#[test]
fn tree_aci_products_meet_the_contract() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    let expected = pointwise(&star, &[&x, &y], |v| v[0] * v[1]);
    let rtol = 1e-8;
    let result = run_with(
        &TreeAciInterpolator::default(),
        &[&x, &y],
        &Product,
        &options(l2(&expected), rtol),
    )
    .unwrap();
    check(&star, &result, &expected, rtol);
    assert_resolvable(&result, &[&x, &y]);
    let gap = star.projector(&[(0, 0), (4, 1)]);
    assert!(result
        .report
        .zero_patches
        .iter()
        .any(|record| record.projector.is_subset_of(&gap)));
    // Every engine run was tree ACI's, on a patch below the cap.
    assert!(!result.report.accepted.is_empty());
    assert!(result
        .report
        .accepted
        .iter()
        .all(|record| record.termination == InterpolationTermination::Converged));
}

/// A nonlinear map through tree ACI, including where an input is absent.
#[test]
fn tree_aci_nonlinear_maps_meet_the_contract() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    let rational = |inputs: ColMajorArrayRef<'_, f64>, output: &mut [f64]| -> anyhow::Result<()> {
        for (column, value) in inputs.data().chunks(2).zip(output.iter_mut()) {
            *value = 1.0 / (4.0 - column[0] * column[1]);
        }
        Ok(())
    };
    let expected = pointwise(&star, &[&x, &y], |v| 1.0 / (4.0 - v[0] * v[1]));
    let rtol = 1e-8;
    let result = run_with(
        &TreeAciInterpolator::default(),
        &[&x, &y],
        &rational,
        &options(l2(&expected), rtol),
    )
    .unwrap();
    check(&star, &result, &expected, rtol);
    assert_resolvable(&result, &[&x, &y]);
    assert!(result.report.zero_patches.is_empty());
}

/// The same product with tree RSI as the per-patch kernel. Tree RSI checks
/// no error, so every patch it computes is accepted on the driver's L2
/// measurement.
#[test]
fn tree_rsi_products_meet_the_contract() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    let expected = pointwise(&star, &[&x, &y], |v| v[0] * v[1]);
    let rtol = 1e-8;
    let result = run_with(
        &TreeRsiInterpolator::default(),
        &[&x, &y],
        &Product,
        &options(l2(&expected), rtol),
    )
    .unwrap();
    check(&star, &result, &expected, rtol);
    assert_resolvable(&result, &[&x, &y]);
    let gap = star.projector(&[(0, 0), (4, 1)]);
    assert!(result
        .report
        .zero_patches
        .iter()
        .any(|record| record.projector.is_subset_of(&gap)));
    assert!(!result.report.accepted.is_empty());
    for record in &result.report.accepted {
        assert_eq!(
            record.termination,
            InterpolationTermination::ErrorNotMeasured
        );
        assert!(record.acceptance.is_some());
    }
}

#[test]
fn a_sampled_run_measures_through_the_residual_engine() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    let expected = pointwise(&star, &[&x, &y], |v| v[0] * v[1]);
    let rtol = 1e-8;
    // Patches of depth three have 32 points, more than the 16 samples, so
    // they are sampled.
    let options = options(l2(&expected), rtol)
        .with_patch_order(split_first(&star))
        .with_verification(
            VerificationOptions::new()
                .with_samples(16)
                .with_max_exhaustive_points(0),
        );
    let result = run(&[&x, &y], &Product, &options).unwrap();
    // Sampled patches are decided by the residual-TT estimator, which runs
    // the residual engine (its points may all be cached already).
    assert!(result.report.accepted.iter().any(|record| record
        .acceptance
        .as_ref()
        .is_some_and(|measurement| measurement.method == MeasurementMethod::ResidualTt)));
    let actual = star.values(&result.partition);
    let error: Vec<f64> = actual.iter().zip(&expected).map(|(a, e)| a - e).collect();
    assert!(l2(&error) <= rtol * l2(&expected) * (1.0 + 1e-9));
}

#[test]
fn an_empty_input_is_the_zero_function() {
    let star = Star::new();
    let (x, _) = inputs(&star);
    let empty = PartitionedTreeTN::<usize>::new();
    // f(x, z) = x + 1 + z with z empty, so the result is x + 1 everywhere.
    let shifted = |inputs: ColMajorArrayRef<'_, f64>, output: &mut [f64]| -> anyhow::Result<()> {
        for (column, value) in inputs.data().chunks(2).zip(output.iter_mut()) {
            *value = column[0] + 1.0 + column[1];
        }
        Ok(())
    };
    let expected = pointwise(&star, &[&x], |v| v[0] + 1.0);
    let rtol = 1e-8;
    let result = run(&[&x, &empty], &shifted, &options(l2(&expected), rtol)).unwrap();
    check(&star, &result, &expected, rtol);
}

#[test]
fn invalid_inputs_fail_before_any_evaluation() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    let options = options(1.0, 1e-8);
    let invalid = |result: Result<PatchedInterpolationResult<usize>, PatchedInterpolationError>| {
        matches!(result, Err(PatchedInterpolationError::InvalidInput { .. }))
    };
    assert!(invalid(run(&[], &Product, &options)));
    let empty = PartitionedTreeTN::<usize>::new();
    assert!(invalid(run(&[&empty, &empty], &Product, &options)));
    // patch_order must list every site an input projector fixes (x fixes 2).
    let partial = options
        .clone()
        .with_patch_order(vec![star.sites[0].clone(), star.sites[4].clone()]);
    assert!(invalid(run(&[&x, &y], &Product, &partial)));
    // Inputs on another site space are rejected.
    let other = Star::new();
    let foreign = other.partition(&other.network(3, 0.0), &[&[(0, 0)]]);
    let result = run(&[&x, &foreign], &Product, &options);
    assert!(
        matches!(result, Err(PatchedInterpolationError::Partition { .. })),
        "{:?}",
        result.err()
    );
}

/// The canonical form of a result's layout: accepted and zero projectors.
fn layout_of(result: &PatchedInterpolationResult<usize>) -> (Vec<String>, Vec<String>) {
    let key = |projector: &Projector| {
        let mut entries: Vec<String> = projector
            .iter()
            .map(|(index, value)| format!("{index:?}={value}"))
            .collect();
        entries.sort();
        entries.join(",")
    };
    let mut accepted: Vec<String> = result
        .report
        .accepted
        .iter()
        .map(|record| key(&record.projector))
        .collect();
    let mut zeros: Vec<String> = result
        .report
        .zero_patches
        .iter()
        .map(|record| key(&record.projector))
        .collect();
    accepted.sort();
    zeros.sort();
    (accepted, zeros)
}

#[test]
fn the_lanes_entry_matches_the_serial_entry() {
    let star = Star::new();
    let (x, y) = inputs(&star);
    let rational = |inputs: ColMajorArrayRef<'_, f64>, output: &mut [f64]| -> anyhow::Result<()> {
        for (column, value) in inputs.data().chunks(2).zip(output.iter_mut()) {
            *value = 1.0 / (4.0 - column[0] * column[1]);
        }
        Ok(())
    };
    let expected = pointwise(&star, &[&x, &y], |v| 1.0 / (4.0 - v[0] * v[1]));
    // Splitting on the inputs' sites first keeps the three runs small while
    // every patch still runs the engine on five active sites.
    let options = options(l2(&expected), 1e-8).with_patch_order(split_first(&star));
    let serial = run(&[&x, &y], &rational, &options).unwrap();
    assert!(serial
        .report
        .accepted
        .iter()
        .any(|record| record.projector.len() <= 3 && record.max_bond_dim > 1));
    let serial_values = star.values(&serial.partition);
    let threads = 3;
    let engines = (0..threads)
        .map(|_| tenferro_cpu::CpuEngine::serial())
        .collect();
    let mut lanes = hataori_tenferro::TenferroLanes::new(engines).unwrap();
    let fanned = patched_elementwise_on_lanes(
        &mut lanes,
        &PointwiseElementwise(TreeTciInterpolator::default()),
        &TreeTciInterpolator::default(),
        &[&x, &y],
        &rational,
        &options,
    )
    .unwrap();
    assert_eq!(layout_of(&fanned), layout_of(&serial), "{threads} lanes");
    assert_eq!(fanned.report.splits, serial.report.splits);
    assert_eq!(
        fanned.report.function_evaluations,
        serial.report.function_evaluations
    );
    let values = star.values(&fanned.partition);
    let difference: Vec<f64> = values
        .iter()
        .zip(&serial_values)
        .map(|(a, b)| a - b)
        .collect();
    assert!(l2(&difference) <= 1e-12 * l2(&expected), "{threads} lanes");
}
