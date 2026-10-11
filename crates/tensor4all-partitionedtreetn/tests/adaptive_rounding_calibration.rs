//! Calibration measurement of `MEASUREMENT_ROUNDING_FACTOR` (not a contract
//! test): the evaluation error of the cached evaluator (as the driver uses
//! it) and of `contract_to_tensor`, against values computed in double-double
//! arithmetic, for networks on the test trees, including cancelling ones.
//! The ratio reported is `||evaluated - exact|| / (eps * ||exact||)`.
//!
//! Run with `cargo test --release -p tensor4all-partitionedtreetn --test
//! adaptive_rounding_calibration -- --ignored --nocapture`.

mod adaptive_common;

use std::collections::HashMap;

use adaptive_common::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationOptions;
use tensor4all_partitionedtreetn::{ErrorNorm, L2Reference};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

/// A double-double number `hi + lo`.
#[derive(Clone, Copy, Debug, Default)]
struct Dd {
    hi: f64,
    lo: f64,
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

impl Dd {
    fn from(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    fn add(self, other: Self) -> Self {
        let (s, e) = two_sum(self.hi, other.hi);
        let (hi, lo) = two_sum(s, e + self.lo + other.lo);
        Self { hi, lo }
    }

    fn mul(self, other: Self) -> Self {
        let p = self.hi * other.hi;
        let e = self.hi.mul_add(other.hi, -p);
        let (hi, lo) = two_sum(p, e + self.hi * other.lo + self.lo * other.hi);
        Self { hi, lo }
    }

    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

/// Values of a real network at full-domain points in double-double, by
/// message passing towards the smallest node name.
fn exact_values(
    network: &TreeTN<IdxTensor, String>,
    sites: &[DynIndex],
    points: &[Vec<usize>],
) -> Vec<f64> {
    let mut names = network.node_names();
    names.sort();
    let tensors: HashMap<String, (Vec<DynIndex>, Vec<f64>)> = names
        .iter()
        .map(|name| {
            let tensor = network.tensor(network.node_index(name).unwrap()).unwrap();
            (
                name.clone(),
                (tensor.indices().to_vec(), tensor.to_vec::<f64>().unwrap()),
            )
        })
        .collect();
    // Node pairs linked by a shared non-site index.
    let mut owners: HashMap<DynIndex, Vec<String>> = HashMap::new();
    for (name, (legs, _)) in &tensors {
        for leg in legs {
            if !sites.contains(leg) {
                owners.entry(leg.clone()).or_default().push(name.clone());
            }
        }
    }
    let root = names[0].clone();

    fn message(
        node: &str,
        parent: Option<&DynIndex>,
        point: &[usize],
        sites: &[DynIndex],
        tensors: &HashMap<String, (Vec<DynIndex>, Vec<f64>)>,
        owners: &HashMap<DynIndex, Vec<String>>,
    ) -> Vec<Dd> {
        let (legs, data) = &tensors[node];
        // Child messages for every bond except the parent's.
        let mut child: HashMap<usize, Vec<Dd>> = HashMap::new();
        for (axis, leg) in legs.iter().enumerate() {
            if sites.contains(leg) || Some(leg) == parent {
                continue;
            }
            let other = owners[leg].iter().find(|n| n.as_str() != node).unwrap();
            child.insert(
                axis,
                message(other, Some(leg), point, sites, tensors, owners),
            );
        }
        let parent_axis = parent.map(|p| legs.iter().position(|l| l == p).unwrap());
        let parent_dim = parent.map_or(1, IndexLike::dim);
        let mut out = vec![Dd::default(); parent_dim];
        let dims: Vec<usize> = legs.iter().map(IndexLike::dim).collect();
        'entries: for (flat, &value) in data.iter().enumerate() {
            let mut rest = flat;
            let mut term = Dd::from(value);
            let mut out_index = 0;
            for (axis, &dim) in dims.iter().enumerate() {
                let coordinate = rest % dim;
                rest /= dim;
                let leg = &legs[axis];
                if let Some(position) = sites.iter().position(|s| s == leg) {
                    if point[position] != coordinate {
                        continue 'entries;
                    }
                } else if Some(axis) == parent_axis {
                    out_index = coordinate;
                } else {
                    term = term.mul(child[&axis][coordinate]);
                }
            }
            out[out_index] = out[out_index].add(term);
        }
        out
    }

    points
        .iter()
        .map(|point| message(&root, None, point, sites, &tensors, &owners)[0].value())
        .collect()
}

/// The driver's evaluation: a fresh evaluator centered at the smallest node
/// name, the default hint, and chunks of 256 points.
fn cached_values(
    network: &TreeTN<IdxTensor, String>,
    sites: &[DynIndex],
    points: &[Vec<usize>],
) -> Vec<f64> {
    let center = network.node_names().into_iter().min();
    let options = CachedEvaluatorOptions {
        center,
        ..CachedEvaluatorOptions::default()
    };
    let mut evaluator = TreeTNCachedEvaluator::new(network, sites, options).unwrap();
    let flat: Vec<usize> = points.concat();
    let mut values = Vec::new();
    for chunk in flat.chunks(256 * sites.len()) {
        let shape = [sites.len(), chunk.len() / sites.len()];
        values.extend(
            evaluator
                .evaluate_batched_typed::<f64>(
                    ColMajorArrayRef::new(chunk, &shape).unwrap(),
                    EvaluationHint::default(),
                )
                .unwrap(),
        );
    }
    values
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The two ratios of one network: cached evaluator and dense contraction.
fn ratios(network: &TreeTN<IdxTensor, String>, problem: &Problem) -> (f64, f64) {
    let points = full_domain(&problem.dims());
    let exact = exact_values(network, &problem.sites, &points);
    let cached = cached_values(network, &problem.sites, &points);
    let norm = l2(&exact);
    let cached_error = l2(&cached
        .iter()
        .zip(&exact)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>());
    let reference = IdxTensor::from_dense(problem.sites.clone(), exact).unwrap();
    let dense_error = network
        .contract_to_tensor()
        .unwrap()
        .sub(&reference)
        .unwrap()
        .norm()
        .unwrap();
    (
        cached_error / (f64::EPSILON * norm),
        dense_error / (f64::EPSILON * norm),
    )
}

/// A random network on the problem's tree with uniform entries in [-1, 1].
fn random_network(
    problem: &Problem,
    bond_dim: usize,
    rng: &mut ChaCha8Rng,
) -> TreeTN<IdxTensor, String> {
    let graph = problem.topology.graph();
    let mut links: HashMap<String, Vec<DynIndex>> = HashMap::new();
    for edge in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge).unwrap();
        let link = DynIndex::new_dyn(bond_dim);
        for node in [a, b] {
            let name = problem.topology.node_name(node).unwrap().clone();
            links.entry(name).or_default().push(link.clone());
        }
    }
    let mut names = Vec::new();
    let mut tensors = Vec::new();
    for (node, sites) in &problem.node_sites {
        let mut legs = sites.clone();
        legs.extend(links.remove(node).unwrap_or_default());
        let size: usize = legs.iter().map(IndexLike::dim).product();
        let data = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
        names.push(node.clone());
        tensors.push(IdxTensor::from_dense(legs, data).unwrap());
    }
    TreeTN::from_tensors(tensors, names).unwrap()
}

/// `a - b` for `b` a copy of `a` with every entry perturbed by a relative
/// `noise`: the contraction cancels to about `noise` of its terms.
fn cancelling_network(
    a: &TreeTN<IdxTensor, String>,
    noise: f64,
    rng: &mut ChaCha8Rng,
) -> TreeTN<IdxTensor, String> {
    let mut names = a.node_names();
    names.sort();
    let tensors = names
        .iter()
        .map(|name| {
            let tensor = a.tensor(a.node_index(name).unwrap()).unwrap();
            let data = tensor
                .to_vec::<f64>()
                .unwrap()
                .into_iter()
                .map(|v| v * (1.0 + noise * rng.random_range(-1.0..1.0)))
                .collect();
            IdxTensor::from_dense(tensor.indices().to_vec(), data).unwrap()
        })
        .collect();
    let b = TreeTN::from_tensors(tensors, names).unwrap();
    a.axpby(1.0, &b, -1.0).unwrap()
}

#[test]
#[ignore = "calibration measurement for MEASUREMENT_ROUNDING_FACTOR; run in release with --ignored --nocapture"]
fn measurement_rounding_calibration() {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut worst: f64 = 0.0;
    let mut record = |label: &str, (cached, dense): (f64, f64)| {
        eprintln!("{label}: cached {cached:.2}, contract_to_tensor {dense:.2}");
        worst = worst.max(cached).max(dense);
    };
    for (tree_name, problem) in [
        ("extended quantics tree", extended_quantics_tree()),
        ("raw-kernel tree", raw_kernel_tree()),
        ("branched", branched()),
    ] {
        for bond_dim in [2, 4, 8] {
            for k in 0..3 {
                let a = random_network(&problem, bond_dim, &mut rng);
                record(
                    &format!("{tree_name} random D={bond_dim} #{k}"),
                    ratios(&a, &problem),
                );
                for noise in [0.1, 0.01] {
                    let c = cancelling_network(&a, noise, &mut rng);
                    record(
                        &format!("{tree_name} cancelling noise={noise} D={bond_dim} #{k}"),
                        ratios(&c, &problem),
                    );
                }
            }
        }
    }
    // Driver patches: the certified TreeTCI run of test 1.
    let problem = extended_quantics_tree();
    let (_, norm) = dense_reference(&problem, &extended_peak);
    let options = PatchedInterpolationOptions::new(4)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
        .with_tolerance(tol(1e-6))
        .with_patch_order(extended_order(&problem))
        .with_seed(5);
    let result = run(
        &TreeTciInterpolator::default(),
        &problem,
        &extended_peak,
        &[],
        &options,
    )
    .unwrap();
    for (index, patch) in result.partition.values().enumerate() {
        record(
            &format!("TreeTCI patch {index}"),
            ratios(patch.data(), &problem),
        );
    }
    eprintln!("largest ratio {worst:.2}");
}
