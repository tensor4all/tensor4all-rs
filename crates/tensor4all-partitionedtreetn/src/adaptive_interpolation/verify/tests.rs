//! Tests of the driver-side measurement.

use std::collections::{BTreeMap, HashMap};

use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::InterpolationProblem;
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

use super::{
    approximation_rms, global_error, network_values, statistics, Contribution, Measured, PointPlan,
    ScaledSquares, MEASUREMENT_CHUNK,
};
use crate::adaptive_interpolation::{patched_interpolate, PatchedInterpolationOptions};
use crate::adaptive_interpolation::{
    GlobalL2Error, L2Measurement, MeasurementMethod, ToleranceNotMetBasis, GLOBAL_ROUNDING_MARGIN,
    MEASUREMENT_ROUNDING_FACTOR,
};
use crate::{ErrorNorm, ErrorTolerance};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

/// A named tree with its sites in the derived site order.
struct Tree {
    topology: NodeNameNetwork<String>,
    node_sites: BTreeMap<String, Vec<DynIndex>>,
    sites: Vec<DynIndex>,
}

fn tree(nodes: &[(&str, &[usize])], edges: &[(&str, &str)]) -> Tree {
    let mut topology = NodeNameNetwork::new();
    for (node, _) in nodes {
        topology.add_node(node.to_string()).unwrap();
    }
    for (left, right) in edges {
        topology
            .add_edge(&left.to_string(), &right.to_string())
            .unwrap();
    }
    let node_sites: BTreeMap<String, Vec<DynIndex>> = nodes
        .iter()
        .map(|(node, dims)| {
            let sites = dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
            (node.to_string(), sites)
        })
        .collect();
    let sites = InterpolationProblem::derive_site_order(&node_sites);
    Tree {
        topology,
        node_sites,
        sites,
    }
}

fn max_degree(tree: &Tree) -> usize {
    let graph = tree.topology.graph();
    graph
        .node_indices()
        .map(|node| graph.neighbors(node).count())
        .max()
        .unwrap_or(0)
}

/// One binary site per node around a junction `c` of degree three: the
/// cached evaluator's raw kernels apply. Site order a0, a1, b0, b1, c, d0, d1.
fn raw_kernel_tree() -> Tree {
    tree(
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

/// The M2 `quantics_tree` (site-free junction `r` of degree three) extended
/// by a leaf `w` with two sites of dimensions 2 and 3: the generic path.
/// Site order w0, w1, x0, x1, x2, y0, y1, y2, z.
fn generic_path_tree() -> Tree {
    tree(
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

fn quantics(bits: &[usize]) -> f64 {
    bits.iter()
        .enumerate()
        .map(|(k, &bit)| bit as f64 * 0.5_f64.powi(k as i32 + 1))
        .sum()
}

fn peak(x: f64, y: f64) -> f64 {
    (-((x - 0.3) / 0.12).powi(2) - ((y - 0.6) / 0.12).powi(2)).exp()
}

fn raw_kernel_function(p: &[usize]) -> f64 {
    let (x, y, u) = (quantics(&p[0..2]), quantics(&p[2..4]), quantics(&p[5..7]));
    (1.0 + 0.5 * p[4] as f64) * peak(x, y) + 0.2 * (x * y + u)
}

fn generic_path_function(p: &[usize]) -> f64 {
    let (x, y) = (quantics(&p[2..5]), quantics(&p[5..8]));
    (1.0 + 0.25 * p[0] as f64 + 0.125 * p[1] as f64) * (1.0 + 0.5 * p[8] as f64) * peak(x, y)
        + 0.2 * x * y
}

/// Raw data of one stored node: name, legs, column-major values.
type RawNode = (String, Vec<DynIndex>, Vec<f64>);

/// A stored patch as raw node data and the full-domain points of the patch.
struct RawPatch {
    nodes: Vec<RawNode>,
    points: Vec<usize>,
}

/// Every point of the domain, column-major (first site fastest).
fn domain(dims: &[usize]) -> Vec<Vec<usize>> {
    let n: usize = dims.iter().product();
    (0..n)
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

/// Run the M2 driver with TreeTCI and return its stored patches as raw data,
/// with the points inside each patch.
fn stored_patches(tree: &Tree, f: fn(&[usize]) -> f64) -> Vec<RawPatch> {
    let n_sites = tree.sites.len();
    let dims: Vec<usize> = tree.sites.iter().map(IndexLike::dim).collect();
    let all = domain(&dims);
    let scale = all.iter().map(|p| f(p).abs()).fold(0.0, f64::max);
    let options = PatchedInterpolationOptions::new(3)
        .with_error_norm(ErrorNorm::sampled_max_with_reference(scale))
        .with_tolerance(ErrorTolerance {
            rtol: 1e-8,
            atol: 0.0,
        })
        .with_seed(7);
    let result = patched_interpolate(
        &TreeTciInterpolator::default(),
        tree.topology.clone(),
        tree.node_sites.clone(),
        ColMajorArray::new(vec![], vec![n_sites, 0]).unwrap(),
        |batch: ColMajorArrayRef<'_, usize>| -> anyhow::Result<Vec<f64>> {
            Ok(batch.data().chunks(n_sites).map(f).collect())
        },
        &options,
    )
    .unwrap();
    assert!(result.report.splits >= 1, "the root must split");
    result
        .report
        .accepted
        .iter()
        .map(|record| {
            let data = result.partition.get(&record.projector).unwrap().data();
            let mut names = data.node_names();
            names.sort();
            let nodes = names
                .iter()
                .map(|name| {
                    let tensor = data.tensor(data.node_index(name).unwrap()).unwrap();
                    (
                        name.clone(),
                        tensor.indices().to_vec(),
                        tensor.to_vec::<f64>().unwrap(),
                    )
                })
                .collect();
            let points = all
                .iter()
                .filter(|point| {
                    record.projector.iter().all(|(site, &value)| {
                        point[tree.sites.iter().position(|s| s == site).unwrap()] == value
                    })
                })
                .flatten()
                .copied()
                .collect();
            RawPatch { nodes, points }
        })
        .collect()
}

/// Rebuild a patch from raw data with a fresh ID for every bond index and a
/// fresh `TreeTN::from_tensors`.
fn rebuild(patch: &RawPatch, sites: &[DynIndex]) -> TreeTN<IdxTensor, String> {
    let mut fresh: HashMap<DynIndex, DynIndex> = HashMap::new();
    let mut names = Vec::new();
    let mut tensors = Vec::new();
    for (name, legs, values) in &patch.nodes {
        let legs: Vec<DynIndex> = legs
            .iter()
            .map(|leg| {
                if sites.contains(leg) {
                    leg.clone()
                } else {
                    fresh
                        .entry(leg.clone())
                        .or_insert_with(|| DynIndex::new_dyn(leg.dim()))
                        .clone()
                }
            })
            .collect();
        names.push(name.clone());
        tensors.push(IdxTensor::from_dense(legs, values.clone()).unwrap());
    }
    TreeTN::from_tensors(tensors, names).unwrap()
}

/// Measure every stored patch in `repetitions` fresh threads, each with a
/// rebuilt patch and a fresh evaluator. Returns the values per repetition.
fn values_across_threads(
    patches: &std::sync::Arc<Vec<RawPatch>>,
    sites: &[DynIndex],
    repetitions: usize,
) -> Vec<Vec<f64>> {
    (0..repetitions)
        .map(|_| {
            let patches = std::sync::Arc::clone(patches);
            let sites = sites.to_vec();
            std::thread::spawn(move || {
                patches
                    .iter()
                    .flat_map(|patch| {
                        let network = rebuild(patch, &sites);
                        network_values::<f64, String>(&network, &sites, &patch.points).unwrap()
                    })
                    .collect()
            })
            .join()
            .unwrap()
        })
        .collect()
}

/// Largest difference of any repetition from the first, relative to the
/// largest magnitude of the first.
fn max_relative_difference(runs: &[Vec<f64>]) -> f64 {
    let scale = runs[0].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    runs[1..]
        .iter()
        .flat_map(|run| run.iter().zip(&runs[0]).map(|(a, b)| (a - b).abs()))
        .fold(0.0_f64, f64::max)
        / scale
}

/// A digest of the values of one repetition, printed so that separate test
/// processes can be compared by hand (open question 9).
fn digest(values: &[f64]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ value.to_bits()).wrapping_mul(0x0100_0000_01b3)
        })
}

/// Number of values that differ from the first repetition, per repetition.
fn differing_values(runs: &[Vec<f64>]) -> Vec<usize> {
    runs[1..]
        .iter()
        .map(|run| {
            run.iter()
                .zip(&runs[0])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count()
        })
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Whether every node of every stored patch satisfies the raw-kernel
/// condition of the cached evaluator: exactly one site leg, and its legs equal
/// to its neighbor count plus one. A neighbor is a node sharing a bond leg.
fn takes_raw_kernels(patches: &[RawPatch], sites: &[DynIndex]) -> bool {
    patches.iter().all(|patch| {
        patch.nodes.iter().all(|(name, legs, _)| {
            let site_legs = legs.iter().filter(|leg| sites.contains(leg)).count();
            let neighbors = patch
                .nodes
                .iter()
                .filter(|(other, other_legs, _)| {
                    other != name
                        && other_legs
                            .iter()
                            .any(|leg| !sites.contains(leg) && legs.contains(leg))
                })
                .count();
            site_legs == 1 && legs.len() == neighbors + 1
        })
    })
}

const REPETITIONS: usize = 6;

#[test]
fn measurement_is_bitwise_reproducible_across_threads_on_raw_kernel_trees() {
    let tree = raw_kernel_tree();
    assert_eq!(max_degree(&tree), 3);
    let patches = stored_patches(&tree, raw_kernel_function);
    // Stored patches carry every site, fixed ones re-embedded by one-hot
    // factors, so every node keeps exactly one site leg.
    assert!(patches.iter().any(|patch| patch.points.len() < 128));
    assert!(takes_raw_kernels(&patches, &tree.sites));
    let patches = std::sync::Arc::new(patches);
    let runs = values_across_threads(&patches, &tree.sites, REPETITIONS);
    let difference = max_relative_difference(&runs);
    eprintln!(
        "{} values per repetition; differing per repetition {:?}; max relative difference \
         {difference:e}; digests {:x?}",
        runs[0].len(),
        differing_values(&runs),
        runs.iter().map(|run| digest(run)).collect::<Vec<_>>()
    );
    for run in &runs[1..] {
        assert_eq!(
            bits(run),
            bits(&runs[0]),
            "max relative difference {difference:e}"
        );
    }
}

#[test]
#[ignore = "open question 8 of docs/design/tree-patching-error-contract.md: the generic IdxTensor \
            path of TreeTNCachedEvaluator contracts N-ary operand lists through omeco's greedy \
            planner, which breaks cost ties in HashMap order, so values differ across threads"]
fn measurement_is_bitwise_reproducible_across_threads_on_generic_path_trees() {
    let tree = generic_path_tree();
    assert_eq!(max_degree(&tree), 3);
    let patches = stored_patches(&tree, generic_path_function);
    assert!(!takes_raw_kernels(&patches, &tree.sites));
    let patches = std::sync::Arc::new(patches);
    let runs = values_across_threads(&patches, &tree.sites, REPETITIONS);
    let difference = max_relative_difference(&runs);
    eprintln!(
        "{} values per repetition; differing per repetition {:?}; max relative difference \
         {difference:e}; digests {:x?}",
        runs[0].len(),
        differing_values(&runs),
        runs.iter().map(|run| digest(run)).collect::<Vec<_>>()
    );
    for run in &runs[1..] {
        assert_eq!(
            bits(run),
            bits(&runs[0]),
            "max relative difference {difference:e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Statistics, worst points, and the global error
// ---------------------------------------------------------------------------

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-14 * a.abs().max(b.abs())
}

#[test]
fn statistics_use_scaled_sums_and_report_the_relative_standard_error() {
    let exhaustive = statistics(&[3.0, 4.0], MeasurementMethod::Exhaustive, 2.0);
    assert!(close(exhaustive.rms, (12.5_f64).sqrt()));
    assert_eq!(exhaustive.max_residual, 4.0);
    assert_eq!(exhaustive.mean_square_rel_std_error, 0.0);
    assert_eq!(exhaustive.points, 2);
    assert!(close(exhaustive.error_norm().unwrap(), 5.0));

    // Sampled: squares 1, 4, 9, 16 have mean 7.5 and sample variance 43.
    let sampled = statistics(&[1.0, 2.0, 3.0, 4.0], MeasurementMethod::Sampled, 100.0);
    assert!(close(sampled.rms, 7.5_f64.sqrt()));
    let expected = (43.0 / 4.0_f64).sqrt() / 7.5;
    assert!(close(sampled.mean_square_rel_std_error, expected));

    // Residuals near the largest float do not overflow the sum of squares.
    let large = statistics(&[1e300, 1e300], MeasurementMethod::Exhaustive, 2.0);
    assert!(close(large.rms, 1e300));
    // A zero residual carries no information.
    let zero = statistics(&[0.0, 0.0], MeasurementMethod::Sampled, 4.0);
    assert_eq!((zero.rms, zero.mean_square_rel_std_error), (0.0, 0.0));
    assert_eq!(statistics(&[], MeasurementMethod::Sampled, 4.0).rms, 0.0);
    // An overflowing L2 norm is reported as None.
    let huge = statistics(&[1e300], MeasurementMethod::Exhaustive, 1e300);
    assert_eq!(huge.error_norm(), None);
}

#[test]
fn worst_points_are_distinct_and_sorted_with_ties_in_measurement_order() {
    let measured = Measured {
        measurement: statistics(&[], MeasurementMethod::Sampled, 1.0),
        points: vec![vec![0], vec![1], vec![2], vec![1], vec![3], vec![4]],
        residuals: vec![0.5, 2.0, 3.0, 2.0, 2.0, 0.1],
    };
    assert_eq!(measured.worst_points(1.0, 10), [vec![2], vec![1], vec![3]]);
    assert_eq!(measured.worst_points(1.0, 2), [vec![2], vec![1]]);
    assert_eq!(measured.worst_points(0.0, 10).len(), 5);
}

#[test]
fn point_plans_are_exhaustive_up_to_the_inclusive_threshold() {
    assert_eq!(
        PointPlan::for_patch(Some(16), 16, 0, 7),
        PointPlan::Exhaustive { count: 16 }
    );
    assert_eq!(
        PointPlan::for_patch(Some(17), 16, 0, 7),
        PointPlan::Sampled { count: 16, seed: 7 }
    );
    assert_eq!(
        PointPlan::for_patch(Some(1024), 64, 1024, 7),
        PointPlan::Exhaustive { count: 1024 }
    );
    // A point count that overflows usize is never exhaustive, even with an
    // unbounded limit.
    for (samples, max_exhaustive) in [(64, usize::MAX), (usize::MAX, 0)] {
        assert!(matches!(
            PointPlan::for_patch(None, samples, max_exhaustive, 7),
            PointPlan::Sampled { .. }
        ));
    }
}

#[test]
fn scaled_squares_match_the_plain_sum() {
    let mut sum = ScaledSquares::default();
    for a in [3.0, 0.0, 4.0, 12.0] {
        sum.add(a);
    }
    assert!(close(sum.norm(), 13.0));
    let mut large = ScaledSquares::default();
    large.add(1e300);
    large.add(1e300);
    assert!(close(large.norm(), 2.0_f64.sqrt() * 1e300));
}

fn measurement(method: MeasurementMethod, patch_points: f64, rms: f64, rel: f64) -> L2Measurement {
    let mut measurement = statistics(&[], method, patch_points);
    measurement.rms = rms;
    measurement.mean_square_rel_std_error = rel;
    measurement
}

#[test]
fn global_error_combines_disjoint_patches_by_volume() {
    let a = measurement(MeasurementMethod::Exhaustive, 3.0, 0.1, 0.0);
    let b = measurement(MeasurementMethod::Exact, 1.0, 0.0, 0.0);
    let contributions = [
        Contribution {
            patch_points: 3.0,
            acceptance: &a,
            audit: None,
            within_tolerance: true,
        },
        Contribution {
            patch_points: 1.0,
            acceptance: &b,
            audit: None,
            within_tolerance: true,
        },
    ];
    let (global, certified) = global_error(&contributions, 4.0, 0.2, Some(1.0));
    assert_eq!(certified, 1.0);
    let GlobalL2Error::Certified {
        rms_error,
        rounding_allowance_rms,
        rounding_limited,
        relative_error_bound,
    } = global
    else {
        panic!("expected a certified error");
    };
    assert!(close(rms_error, (0.75_f64 * 0.01).sqrt()));
    let rounding = MEASUREMENT_ROUNDING_FACTOR * f64::EPSILON;
    assert_eq!(rounding_allowance_rms, Some(rounding));
    assert_eq!(rounding_limited, Some(false));
    let upper = rms_error * (1.0 + GLOBAL_ROUNDING_MARGIN) + rounding;
    let bound = upper / ((1.0 - GLOBAL_ROUNDING_MARGIN) - upper);
    assert!(close(relative_error_bound.unwrap(), bound));

    // No approximation norm: no rounding term, flag, or relative bound.
    let (global, _) = global_error(&contributions, 4.0, 0.2, None);
    assert!(matches!(
        global,
        GlobalL2Error::Certified {
            rounding_allowance_rms: None,
            rounding_limited: None,
            relative_error_bound: None,
            ..
        }
    ));
    // A denominator that is not positive gives no relative statement.
    let (global, _) = global_error(&contributions, 4.0, 0.2, Some(0.05));
    assert!(matches!(
        global,
        GlobalL2Error::Certified {
            relative_error_bound: None,
            ..
        }
    ));
    // A tau below the rounding term is rounding-limited.
    let (global, _) = global_error(&contributions, 4.0, 1e-20, Some(1.0));
    assert!(matches!(
        global,
        GlobalL2Error::Certified {
            rounding_limited: Some(true),
            ..
        }
    ));
}

#[test]
fn sampled_contributions_are_audited_or_acceptance_only() {
    let exact = measurement(MeasurementMethod::Exhaustive, 2.0, 0.1, 0.0);
    let sampled = measurement(MeasurementMethod::Sampled, 2.0, 0.05, 0.3);
    let audit = measurement(MeasurementMethod::Sampled, 2.0, 0.2, 0.5);
    let audited = [
        Contribution {
            patch_points: 2.0,
            acceptance: &exact,
            audit: None,
            within_tolerance: true,
        },
        Contribution {
            patch_points: 2.0,
            acceptance: &sampled,
            audit: Some(&audit),
            within_tolerance: true,
        },
    ];
    let (global, certified) = global_error(&audited, 4.0, 0.1, Some(1.0));
    assert_eq!(certified, 0.5);
    let GlobalL2Error::Audited {
        rms_error_estimate,
        mean_square_rel_std_error,
        relative_bound_estimate,
    } = global
    else {
        panic!("expected an audited error");
    };
    // Mean square 0.5 * 0.01 + 0.5 * 0.04 = 0.025; the audited term 0.02
    // has standard error 0.01.
    assert!(close(rms_error_estimate, 0.025_f64.sqrt()));
    assert!(close(mean_square_rel_std_error, 0.01 / 0.025));
    let estimate = 0.025_f64.sqrt();
    assert!(close(
        relative_bound_estimate.unwrap(),
        estimate / (1.0 - estimate)
    ));
    let (global, _) = global_error(&audited, 4.0, 0.1, Some(0.1));
    assert!(matches!(
        global,
        GlobalL2Error::Audited {
            relative_bound_estimate: None,
            ..
        }
    ));

    let unaudited = [
        Contribution {
            patch_points: 2.0,
            acceptance: &exact,
            audit: None,
            within_tolerance: true,
        },
        Contribution {
            patch_points: 2.0,
            acceptance: &sampled,
            audit: None,
            within_tolerance: true,
        },
    ];
    let (global, _) = global_error(&unaudited, 4.0, 0.1, Some(1.0));
    let GlobalL2Error::AcceptanceOnly {
        acceptance_statistic_rms,
    } = global
    else {
        panic!("expected an acceptance-only error");
    };
    assert!(close(
        acceptance_statistic_rms,
        (0.5_f64 * 0.01 + 0.5 * 0.0025).sqrt()
    ));
}

#[test]
fn unmet_contributions_keep_the_m3_classification_as_basis() {
    // Two halves of a 4-point domain; the second one misses its allowance.
    let met = measurement(MeasurementMethod::Exhaustive, 2.0, 0.1, 0.0);
    let unmet = measurement(MeasurementMethod::Exhaustive, 2.0, 0.3, 0.0);
    let contribution = |acceptance, audit, within_tolerance| Contribution {
        patch_points: 2.0,
        acceptance,
        audit,
        within_tolerance,
    };
    let exhaustive = [
        contribution(&met, None, true),
        contribution(&unmet, None, false),
    ];
    let (global, certified) = global_error(&exhaustive, 4.0, 0.1, Some(1.0));
    // The unmet half is not certified although it was measured exhaustively.
    assert_eq!(certified, 0.5);
    let GlobalL2Error::ToleranceNotMet {
        measured_rms,
        unmet_fraction,
        basis:
            ToleranceNotMetBasis::ExactOrExhaustive {
                rounding_allowance_rms,
                rounding_limited,
                relative_error_bound,
            },
    } = global
    else {
        panic!("expected ToleranceNotMet with an exact basis, got {global:?}");
    };
    let rms = (0.5_f64 * 0.01 + 0.5 * 0.09).sqrt();
    assert!(close(measured_rms, rms));
    assert_eq!(unmet_fraction, 0.5);
    let rounding = MEASUREMENT_ROUNDING_FACTOR * f64::EPSILON;
    assert_eq!(rounding_allowance_rms, Some(rounding));
    assert_eq!(rounding_limited, Some(false));
    // The relative bound uses the measured value, never tau.
    let upper = rms * (1.0 + GLOBAL_ROUNDING_MARGIN) + rounding;
    let bound = upper / ((1.0 - GLOBAL_ROUNDING_MARGIN) - upper);
    assert!(close(relative_error_bound.unwrap(), bound));

    let sampled = measurement(MeasurementMethod::Sampled, 2.0, 0.3, 0.2);
    let audit = measurement(MeasurementMethod::Sampled, 2.0, 0.4, 0.5);
    let audited = [
        contribution(&met, None, true),
        contribution(&sampled, Some(&audit), false),
    ];
    let (global, certified) = global_error(&audited, 4.0, 0.1, Some(1.0));
    assert_eq!(certified, 0.5);
    let GlobalL2Error::ToleranceNotMet {
        measured_rms,
        unmet_fraction,
        basis:
            ToleranceNotMetBasis::Audited {
                mean_square_rel_std_error,
                relative_bound_estimate,
            },
    } = global
    else {
        panic!("expected ToleranceNotMet with an audited basis, got {global:?}");
    };
    // Mean square 0.5 * 0.01 + 0.5 * 0.16 = 0.085; the audited term 0.08
    // has standard error 0.04.
    let estimate = 0.085_f64.sqrt();
    assert!(close(measured_rms, estimate));
    assert_eq!(unmet_fraction, 0.5);
    assert!(close(mean_square_rel_std_error, 0.04 / 0.085));
    assert!(close(
        relative_bound_estimate.unwrap(),
        estimate / (1.0 - estimate)
    ));

    let unaudited = [
        contribution(&met, None, true),
        contribution(&sampled, None, false),
    ];
    let (global, certified) = global_error(&unaudited, 4.0, 0.1, Some(1.0));
    assert_eq!(certified, 0.5);
    let GlobalL2Error::ToleranceNotMet {
        measured_rms,
        unmet_fraction,
        basis: ToleranceNotMetBasis::AcceptanceOnly,
    } = global
    else {
        panic!("expected ToleranceNotMet with an acceptance-only basis, got {global:?}");
    };
    assert!(close(measured_rms, (0.5_f64 * 0.01 + 0.5 * 0.09).sqrt()));
    assert_eq!(unmet_fraction, 0.5);
    assert_eq!(global.rms_value(), measured_rms);
}

#[test]
fn approximation_rms_combines_log_norms_and_reports_overflow() {
    // Patches of norms 3 and 4 on a domain of 8 points: ||f~|| = 5.
    let rms = approximation_rms(
        [(Ok(3.0_f64.ln()), 2.0), (Ok(4.0_f64.ln()), 6.0)].into_iter(),
        8.0,
    )
    .unwrap();
    assert!(close(rms, 5.0 / 8.0_f64.sqrt()));
    // A zero patch contributes nothing; an overflowing one gives None.
    let zero = approximation_rms([(Ok(f64::NEG_INFINITY), 2.0)].into_iter(), 8.0);
    assert_eq!(zero, Some(0.0));
    assert_eq!(
        approximation_rms([(Ok(f64::INFINITY), 2.0)].into_iter(), 8.0),
        None
    );
}

// ---------------------------------------------------------------------------
// Chunked evaluation
// ---------------------------------------------------------------------------

/// A star of three arms of three binary sites around a binary center: 1024
/// points, more than `MEASUREMENT_CHUNK`.
fn star_tree() -> Tree {
    let names = ["c", "a0", "a1", "a2", "b0", "b1", "b2", "d0", "d1", "d2"];
    let nodes: Vec<(&str, &[usize])> = names.iter().map(|name| (*name, &[2usize][..])).collect();
    tree(
        &nodes,
        &[
            ("c", "a0"),
            ("a0", "a1"),
            ("a1", "a2"),
            ("c", "b0"),
            ("b0", "b1"),
            ("b1", "b2"),
            ("c", "d0"),
            ("d0", "d1"),
            ("d1", "d2"),
        ],
    )
}

#[test]
fn chunked_measurement_matches_a_single_batch_on_exact_data() {
    let tree = star_tree();
    assert_eq!(max_degree(&tree), 3);
    // Small integers make every contraction exact, whatever the batching.
    let mut links: HashMap<(String, String), DynIndex> = HashMap::new();
    let graph = tree.topology.graph();
    for edge in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge).unwrap();
        let (a, b) = (
            tree.topology.node_name(a).unwrap().clone(),
            tree.topology.node_name(b).unwrap().clone(),
        );
        links.insert((a, b), DynIndex::new_dyn(2));
    }
    let mut names = Vec::new();
    let mut tensors = Vec::new();
    for (k, (node, sites)) in tree.node_sites.iter().enumerate() {
        let mut legs = sites.clone();
        legs.extend(
            links
                .iter()
                .filter(|((a, b), _)| a == node || b == node)
                .map(|(_, link)| link.clone()),
        );
        let size: usize = legs.iter().map(IndexLike::dim).product();
        let data: Vec<f64> = (0..size)
            .map(|i| ((i * 7 + k * 3) % 5) as f64 - 2.0)
            .collect();
        names.push(node.clone());
        tensors.push(IdxTensor::from_dense(legs, data).unwrap());
    }
    let network = TreeTN::from_tensors(tensors, names).unwrap();
    let dims: Vec<usize> = tree.sites.iter().map(IndexLike::dim).collect();
    let points: Vec<usize> = domain(&dims).into_iter().flatten().collect();
    let n_points = points.len() / tree.sites.len();
    assert!(n_points > 2 * MEASUREMENT_CHUNK);

    let chunked = network_values::<f64, String>(&network, &tree.sites, &points).unwrap();
    let center = network.node_names().into_iter().min();
    let options = CachedEvaluatorOptions {
        center,
        ..CachedEvaluatorOptions::default()
    };
    let mut evaluator = TreeTNCachedEvaluator::new(&network, &tree.sites, options).unwrap();
    let shape = [tree.sites.len(), n_points];
    let single: Vec<f64> = evaluator
        .evaluate_batched_typed(
            ColMajorArrayRef::new(&points, &shape).unwrap(),
            EvaluationHint::default(),
        )
        .unwrap();
    assert_eq!(bits(&chunked), bits(&single));
    // The data is exact: every value is an integer.
    assert!(chunked.iter().all(|value| value.fract() == 0.0));
    assert!(chunked.iter().any(|&value| value != 0.0));
    // The measurements of the two agree exactly.
    let a = statistics(
        &chunked.iter().map(|v| v.abs()).collect::<Vec<_>>(),
        MeasurementMethod::Exhaustive,
        n_points as f64,
    );
    let b = statistics(
        &single.iter().map(|v| v.abs()).collect::<Vec<_>>(),
        MeasurementMethod::Exhaustive,
        n_points as f64,
    );
    assert_eq!(a, b);
}
