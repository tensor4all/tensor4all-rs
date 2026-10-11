// M4 eager-versus-compact patch representation comparison at realistic bond
// dimensions. The body is included by
// `crates/tensor4all-partitionedtreetn/examples/benchmark_tree_patch_representation.rs`.
//
// Usage (all modes print one JSON object per line):
//
//   calibrate <chain|tree> <bits> <eta> <cap>
//       One monolithic TreeTCI run (no patching) of the workload with bond
//       cap `cap`; reports the realized rank and the run time.
//   patch <chain|tree> <bits> <eta> <cap>
//       One `patched_interpolate` run with per-patch cap `cap`; reports the
//       patch count and the per-patch realized bond dimensions. No timing of
//       representation operations.
//   check <chain|tree> <bits> <eta> <cap>
//       `measure` without any timing: patches, payload, compact conversion,
//       and every correctness check (before and after the operations).
//   measure <chain|tree> <bits> <eta> <cap> [label]
//       The pre-registered comparison: patches from `patched_interpolate`,
//       compact conversion, independent correctness checks, then per
//       operation a warm-up, an eager/eager noise study, and alternating
//       eager/compact pairs for `TreeTN::norm`, rank-capped
//       `TreeTN::truncate`, and the direct sum `TreeTN::add`.
//
// The workload is the spectral function of a three-dimensional tight-binding
// band, `A(x, y, z) = eta / ((e(x, y, z) - MU)^2 + eta^2)` with
// `e = -2 (cos 2 pi x + cos 2 pi y + cos 2 pi z)`, on `bits` quantics bits per
// variable. Its maximum is `1 / eta` on the surface `e = MU`.
use std::collections::BTreeMap;
use std::error::Error;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_interpolate, PatchedInterpolationOptions, PatchedInterpolationResult,
};
use tensor4all_partitionedtreetn::{ErrorNorm, ErrorTolerance, Projector};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{InterpolationProblem, TreeInterpolator};
use tensor4all_treetn::{
    CachedEvaluatorOptions, EvaluationHint, NodeNameNetwork, TreeTN, TreeTNCachedEvaluator,
    TruncationOptions,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Name = String;
type Network = TreeTN<IdxTensor, Name>;

/// Relative tolerance of every interpolation (sampled max norm).
const RTOL: f64 = 1.0e-4;
/// Chemical potential of the band.
const MU: f64 = 0.5;
/// Seed of the interpolation runs and of the correctness sample points.
const SEED: u64 = 7;
/// Eager/compact pairs per operation.
const PAIRS: usize = 10;
/// Eager/eager pairs per operation in the noise study.
const NOISE_PAIRS: usize = 6;
/// Largest median eager/eager relative gap for a valid stratum.
const NOISE_LIMIT: f64 = 0.05;
/// Minimum duration of one timing sample; passes per sample are chosen from
/// the eager warm-up pass so that a sample lasts at least this long.
const MIN_SAMPLE_NS: u128 = 200_000_000;
/// Upper limit on passes per sample.
const MAX_PASSES: u128 = 50;
/// Sample points per patch inside its projector for value comparisons.
const INSIDE_POINTS: usize = 256;
/// Sample points per patch outside its projector for the eager zero check.
const OUTSIDE_POINTS: usize = 64;
/// Largest admissible relative residual (max absolute difference over the
/// sampled maximum magnitude) between eager and compact values.
const RESIDUAL_LIMIT: f64 = 1.0e-10;
/// The same limit for truncated results. Truncation of the two
/// representations keeps the same singular values, but rounding can rotate
/// nearly degenerate singular vectors at the cut, so the limit is looser.
const TRUNCATE_RESIDUAL_LIMIT: f64 = 1.0e-8;
const VARIABLES: [&str; 3] = ["x", "y", "z"];

#[derive(Clone, Copy)]
struct Spectral {
    eta: f64,
}

impl Spectral {
    fn value(&self, coordinates: [f64; 3]) -> f64 {
        let tau = std::f64::consts::TAU;
        let energy = -2.0 * coordinates.iter().map(|c| (tau * c).cos()).sum::<f64>();
        self.eta / ((energy - MU).powi(2) + self.eta * self.eta)
    }

    fn max_reference(&self) -> f64 {
        1.0 / self.eta
    }
}

struct Workload {
    topology_name: String,
    bits: usize,
    topology: NodeNameNetwork<Name>,
    node_sites: BTreeMap<Name, Vec<DynIndex>>,
    /// Sites in the derived interpolation order.
    sites: Vec<DynIndex>,
    /// `(variable, weight)` of every site in `sites`.
    weights: Vec<(usize, f64)>,
    /// Most significant bits first, variables interleaved.
    patch_order: Vec<DynIndex>,
    center: Name,
    function: Spectral,
}

impl Workload {
    fn evaluate_point(&self, point: &[usize]) -> f64 {
        let mut coordinates = [0.0; 3];
        for (&value, &(variable, weight)) in point.iter().zip(&self.weights) {
            coordinates[variable] += value as f64 * weight;
        }
        self.function.value(coordinates)
    }

    fn evaluator<'a>(
        &'a self,
        counter: &'a AtomicUsize,
    ) -> impl Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<f64>> + Send + Sync + 'a {
        let n_sites = self.sites.len();
        move |batch| {
            counter.fetch_add(batch.shape()[1], Ordering::Relaxed);
            Ok(batch
                .data()
                .chunks(n_sites)
                .map(|point| self.evaluate_point(point))
                .collect())
        }
    }

    /// Points on the surface `e = MU` for a few fixed `(y, z)`, quantized to
    /// the grid: the maximum of `f` up to quantization.
    fn initial_pivots(&self) -> Result<ColMajorArray<usize>> {
        let tau = std::f64::consts::TAU;
        let mut data = Vec::new();
        let mut count = 0;
        for (y, z) in [
            (0.1, 0.2),
            (0.3, 0.05),
            (0.45, 0.4),
            (0.15, 0.35),
            (0.05, 0.45),
        ] {
            let cosine: f64 = -0.5 * MU - (tau * y).cos() - (tau * z).cos();
            if cosine.abs() > 1.0 {
                continue;
            }
            let x = cosine.acos() / tau;
            let scale = (1u64 << self.bits) as f64;
            let quantized = [x, y, z].map(|c| ((c * scale).round() as u64) % (1u64 << self.bits));
            for site in &self.sites {
                let (variable, bit) = self.variable_bit(site)?;
                data.push(((quantized[variable] >> (self.bits - 1 - bit)) & 1) as usize);
            }
            count += 1;
        }
        if count == 0 {
            data = vec![0; self.sites.len()];
            count = 1;
        }
        Ok(ColMajorArray::new(data, vec![self.sites.len(), count])?)
    }

    fn variable_bit(&self, site: &DynIndex) -> Result<(usize, usize)> {
        for (name, sites) in &self.node_sites {
            if sites.first() == Some(site) {
                let variable = VARIABLES
                    .iter()
                    .position(|v| name.starts_with(v))
                    .ok_or("site node has no variable prefix")?;
                let bit = name[1..].parse::<usize>()?;
                return Ok((variable, bit));
            }
        }
        Err("site is not on any node".into())
    }
}

fn node_name(variable: usize, bit: usize) -> Name {
    format!("{}{bit:02}", VARIABLES[variable])
}

fn build_workload(topology_name: &str, bits: usize, eta: f64) -> Result<Workload> {
    if bits == 0 || bits > 30 {
        return Err("bits must be in 1..=30".into());
    }
    let mut node_sites: BTreeMap<Name, Vec<DynIndex>> = BTreeMap::new();
    for bit in 0..bits {
        for variable in 0..3 {
            node_sites.insert(node_name(variable, bit), vec![DynIndex::new_dyn(2)]);
        }
    }
    let mut edges = Vec::new();
    let center = match topology_name {
        "chain" => {
            let order: Vec<Name> = (0..bits)
                .flat_map(|bit| (0..3).map(move |variable| node_name(variable, bit)))
                .collect();
            for pair in order.windows(2) {
                edges.push((pair[0].clone(), pair[1].clone()));
            }
            order[order.len() / 2].clone()
        }
        "tree" => {
            node_sites.insert("r".to_string(), Vec::new());
            for variable in 0..3 {
                edges.push(("r".to_string(), node_name(variable, 0)));
                for bit in 1..bits {
                    edges.push((node_name(variable, bit - 1), node_name(variable, bit)));
                }
            }
            "r".to_string()
        }
        other => return Err(format!("unknown topology {other:?}; use chain or tree").into()),
    };
    let mut topology = NodeNameNetwork::new();
    for node in node_sites.keys() {
        topology.add_node(node.clone())?;
    }
    for (left, right) in &edges {
        topology.add_edge(left, right)?;
    }
    let sites = InterpolationProblem::derive_site_order(&node_sites);
    let patch_order = (0..bits)
        .flat_map(|bit| (0..3).map(move |variable| (variable, bit)))
        .map(|(variable, bit)| node_sites[&node_name(variable, bit)][0].clone())
        .collect();
    let mut workload = Workload {
        topology_name: topology_name.to_string(),
        bits,
        topology,
        node_sites,
        sites,
        weights: Vec::new(),
        patch_order,
        center,
        function: Spectral { eta },
    };
    let weights = workload
        .sites
        .iter()
        .map(|site| {
            let (variable, bit) = workload.variable_bit(site)?;
            Ok((variable, 0.5f64.powi(bit as i32 + 1)))
        })
        .collect::<Result<Vec<_>>>()?;
    workload.weights = weights;
    Ok(workload)
}

struct Arguments {
    mode: String,
    topology: String,
    bits: usize,
    eta: f64,
    cap: usize,
    label: String,
}

fn parse_arguments() -> Result<Arguments> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() < 5 {
        return Err(
            "usage: <calibrate|patch|measure> <chain|tree> <bits> <eta> <cap> [label]".into(),
        );
    }
    Ok(Arguments {
        mode: arguments[0].clone(),
        topology: arguments[1].clone(),
        bits: arguments[2].parse()?,
        eta: arguments[3].parse()?,
        cap: arguments[4].parse()?,
        label: arguments
            .get(5)
            .cloned()
            .unwrap_or_else(|| "unlabelled".to_string()),
    })
}

fn main() -> Result<()> {
    let arguments = parse_arguments()?;
    let workload = build_workload(&arguments.topology, arguments.bits, arguments.eta)?;
    println!(
        "{}",
        json!({
            "kind": "build",
            "benchmark": "tree_patch_representation",
            "build_commit": option_env!("T4A_BENCH_GIT_COMMIT").unwrap_or("unrecorded"),
            "mode": arguments.mode,
            "label": arguments.label,
            "topology": workload.topology_name,
            "bits_per_variable": workload.bits,
            "sites": workload.sites.len(),
            "eta": arguments.eta,
            "mu": MU,
            "cap": arguments.cap,
            "rtol": RTOL,
            "error_norm": "sampled max with reference 1/eta",
            "seed": SEED,
            "thread_policy": "pin one CPU; Rayon, OMP and BLAS thread counts are one",
        })
    );
    match arguments.mode.as_str() {
        "calibrate" => calibrate(&workload, arguments.cap),
        "patch" => {
            let (result, seconds, evaluations) = interpolate_patches(&workload, arguments.cap)?;
            print_patches(&workload, &result, seconds, evaluations);
            Ok(())
        }
        "check" => measure(&workload, arguments.cap, false),
        "measure" => measure(&workload, arguments.cap, true),
        other => Err(format!("unknown mode {other:?}").into()),
    }
}

fn calibrate(workload: &Workload, cap: usize) -> Result<()> {
    let max_reference = workload.function.max_reference();
    let problem = InterpolationProblem::new(
        workload.topology.clone(),
        workload.node_sites.clone(),
        workload.initial_pivots()?,
        RTOL * max_reference,
        NonZeroUsize::new(cap),
        SEED,
    )?;
    let counter = AtomicUsize::new(0);
    let start = Instant::now();
    let outcome =
        TreeTciInterpolator::default().interpolate(&problem, workload.evaluator(&counter))?;
    let seconds = start.elapsed().as_secs_f64();
    let link_dims = outcome.network.link_dims();
    println!(
        "{}",
        json!({
            "kind": "calibration",
            "topology": workload.topology_name,
            "bits_per_variable": workload.bits,
            "eta": workload.function.eta,
            "cap": cap,
            "rank": link_dims.iter().copied().max().unwrap_or(0),
            "link_dims": link_dims,
            "termination": format!("{:?}", outcome.termination),
            "relative_error_estimate": outcome.error_estimate / max_reference,
            "max_sample_over_reference": outcome.max_sample_magnitude / max_reference,
            "function_evaluations": counter.load(Ordering::Relaxed),
            "seconds": seconds,
        })
    );
    Ok(())
}

fn interpolate_patches(
    workload: &Workload,
    cap: usize,
) -> Result<(PatchedInterpolationResult<Name>, f64, usize)> {
    let options = PatchedInterpolationOptions::new(cap)
        .with_tolerance(ErrorTolerance {
            rtol: RTOL,
            atol: 0.0,
        })
        .with_error_norm(ErrorNorm::sampled_max_with_reference(
            workload.function.max_reference(),
        ))
        .with_patch_order(workload.patch_order.clone())
        .with_seed(SEED);
    let counter = AtomicUsize::new(0);
    let start = Instant::now();
    let result = patched_interpolate(
        &TreeTciInterpolator::default(),
        workload.topology.clone(),
        workload.node_sites.clone(),
        workload.initial_pivots()?,
        workload.evaluator(&counter),
        &options,
    )?;
    let seconds = start.elapsed().as_secs_f64();
    Ok((result, seconds, counter.load(Ordering::Relaxed)))
}

fn print_patches(
    workload: &Workload,
    result: &PatchedInterpolationResult<Name>,
    seconds: f64,
    evaluator_points: usize,
) {
    let mut bond_dims: Vec<usize> = result
        .report
        .accepted
        .iter()
        .map(|record| record.max_bond_dim)
        .collect();
    bond_dims.sort_unstable();
    let mut fixed_counts: Vec<usize> = result
        .report
        .accepted
        .iter()
        .map(|record| record.projector.iter().count())
        .collect();
    fixed_counts.sort_unstable();
    println!(
        "{}",
        json!({
            "kind": "patches",
            "topology": workload.topology_name,
            "bits_per_variable": workload.bits,
            "eta": workload.function.eta,
            "accepted_patches": result.report.accepted.len(),
            "zero_patches": result.report.zero_patches.len(),
            "splits": result.report.splits,
            "function_evaluations": result.report.function_evaluations,
            "evaluator_points": evaluator_points,
            "cache_hits": result.report.cache_hits,
            "patch_max_bond_dims_sorted": bond_dims,
            "patch_fixed_site_counts_sorted": fixed_counts,
            "max_patch_bond_dim": bond_dims.last().copied().unwrap_or(0),
            "seconds": seconds,
        })
    );
}

/// One accepted patch in both representations.
struct PatchPair {
    projector: Projector,
    eager: Network,
    compact: Network,
    /// Sites of `compact`, in workload site order.
    active_sites: Vec<DynIndex>,
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Norm,
    Truncate,
    Add,
}

impl Operation {
    fn name(self) -> &'static str {
        match self {
            Operation::Norm => "norm",
            Operation::Truncate => "truncate",
            Operation::Add => "add",
        }
    }
}

fn measure(workload: &Workload, cap: usize, timed: bool) -> Result<()> {
    let (result, seconds, evaluator_points) = interpolate_patches(workload, cap)?;
    print_patches(workload, &result, seconds, evaluator_points);

    // Conversion: only the per-patch `select_indices` and rebuild are timed.
    let mut patches = Vec::new();
    let mut conversion_ns = 0u128;
    let (mut eager_bytes, mut compact_bytes) = (0u128, 0u128);
    let mut entries: Vec<_> = result.partition.iter().collect();
    entries.sort_by_key(|(projector, _)| format!("{projector:?}"));
    for (projector, subdomain) in entries {
        let eager = subdomain.data().clone();
        let start = Instant::now();
        let compact = remove_projected_sites(&eager, projector, &workload.node_sites)?;
        conversion_ns += start.elapsed().as_nanos();
        if !eager.same_topology(&compact) {
            return Err("projected-site removal changed the named tree topology".into());
        }
        eager_bytes += payload_bytes(&eager)?;
        compact_bytes += payload_bytes(&compact)?;
        let active_sites = workload
            .sites
            .iter()
            .filter(|site| !projector.is_projected_at(site))
            .cloned()
            .collect();
        patches.push(PatchPair {
            projector: projector.clone(),
            eager,
            compact,
            active_sites,
        });
    }
    let reduction = 100.0 * (eager_bytes as f64 - compact_bytes as f64) / eager_bytes as f64;
    println!(
        "{}",
        json!({
            "kind": "payload",
            "patches": patches.len(),
            "eager_payload_bytes": eager_bytes,
            "compact_payload_bytes": compact_bytes,
            "payload_reduction_percent": reduction,
            "compact_conversion_ns": conversion_ns,
        })
    );

    // Correctness before timing: independent point evaluation of both
    // networks, the eager zero check outside the projector, and norms.
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut worst_inside = 0.0f64;
    let mut worst_outside = 0.0f64;
    let mut worst_norm = 0.0f64;
    for patch in &patches {
        let points = inside_points(workload, &patch.projector, &mut rng);
        let (residual, _) = compare_values(
            &patch.eager,
            &patch.compact,
            workload,
            &patch.active_sites,
            &points,
        )?;
        worst_inside = worst_inside.max(residual);
        worst_outside = worst_outside.max(outside_max_abs(workload, patch, &mut rng)?);
        let eager_norm = patch.eager.clone().norm()?;
        let compact_norm = patch.compact.clone().norm()?;
        worst_norm =
            worst_norm.max((eager_norm - compact_norm).abs() / eager_norm.max(f64::MIN_POSITIVE));
    }
    let passes =
        worst_inside <= RESIDUAL_LIMIT && worst_outside == 0.0 && worst_norm <= RESIDUAL_LIMIT;
    println!(
        "{}",
        json!({
            "kind": "correctness",
            "stage": "before_timing",
            "inside_points_per_patch": INSIDE_POINTS,
            "outside_points_per_patch": OUTSIDE_POINTS,
            "max_relative_value_residual": worst_inside,
            "max_abs_eager_outside_projector": worst_outside,
            "max_relative_norm_difference": worst_norm,
            "limit": RESIDUAL_LIMIT,
            "pass": passes,
        })
    );
    if !passes {
        return Err("eager and compact patches disagree before timing".into());
    }

    let truncation_cap = (cap / 2).max(1);
    for operation in [Operation::Norm, Operation::Truncate, Operation::Add] {
        if timed {
            time_operation(workload, &patches, operation, truncation_cap)?;
        }
        check_operation(workload, &patches, operation, truncation_cap, &mut rng)?;
    }
    println!("{}", json!({"kind": "done"}));
    Ok(())
}

fn time_operation(
    workload: &Workload,
    patches: &[PatchPair],
    operation: Operation,
    truncation_cap: usize,
) -> Result<()> {
    let eager: Vec<&Network> = patches.iter().map(|patch| &patch.eager).collect();
    let compact: Vec<&Network> = patches.iter().map(|patch| &patch.compact).collect();
    let center = &workload.center;

    // Warm-up: one pass per representation, not used for the decision. A
    // failure is recorded and the operation is skipped.
    let warm_eager = match pass_ns(&eager, operation, center, truncation_cap) {
        Ok(ns) => ns,
        Err(error) => return record_failure(operation, "eager", &*error),
    };
    let warm_compact = match pass_ns(&compact, operation, center, truncation_cap) {
        Ok(ns) => ns,
        Err(error) => return record_failure(operation, "compact", &*error),
    };
    let passes = MIN_SAMPLE_NS
        .div_ceil(warm_eager.max(1))
        .clamp(1, MAX_PASSES);
    println!(
        "{}",
        json!({
            "kind": "warmup",
            "operation": operation.name(),
            "eager_pass_ns": warm_eager,
            "compact_pass_ns": warm_compact,
            "passes_per_sample": passes,
        })
    );

    let sample = |trees: &[&Network]| -> Result<u128> {
        let mut total = 0u128;
        for _ in 0..passes {
            total += pass_ns(trees, operation, center, truncation_cap)?;
        }
        Ok(total / passes)
    };

    // Noise study after the warm-up.
    let mut gaps = Vec::with_capacity(NOISE_PAIRS);
    for pair in 0..NOISE_PAIRS {
        let first = sample(&eager)?;
        let second = sample(&eager)?;
        let gap = first.abs_diff(second) as f64 / ((first + second) as f64 * 0.5).max(1.0);
        gaps.push(gap);
        println!(
            "{}",
            json!({
                "kind": "noise_pair",
                "operation": operation.name(),
                "pair": pair,
                "first_ns": first,
                "second_ns": second,
                "relative_gap": gap,
            })
        );
    }
    let max_gap = gaps.iter().copied().fold(0.0, f64::max);
    let noise_median = median_f64(&mut gaps);
    println!(
        "{}",
        json!({
            "kind": "noise",
            "operation": operation.name(),
            "median_pair_relative_gap": noise_median,
            "max_pair_relative_gap": max_gap,
            "limit": NOISE_LIMIT,
            "valid": noise_median <= NOISE_LIMIT,
        })
    );

    // Alternating eager/compact pairs.
    let mut eager_samples = Vec::with_capacity(PAIRS);
    let mut compact_samples = Vec::with_capacity(PAIRS);
    let mut ratios = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let eager_first = pair % 2 == 0;
        let (eager_ns, compact_ns) = if eager_first {
            let eager_ns = sample(&eager)?;
            (eager_ns, sample(&compact)?)
        } else {
            let compact_ns = sample(&compact)?;
            (sample(&eager)?, compact_ns)
        };
        let ratio = compact_ns as f64 / eager_ns.max(1) as f64;
        eager_samples.push(eager_ns);
        compact_samples.push(compact_ns);
        ratios.push(ratio);
        println!(
            "{}",
            json!({
                "kind": "pair",
                "operation": operation.name(),
                "pair": pair,
                "eager_first": eager_first,
                "eager_ns": eager_ns,
                "compact_ns": compact_ns,
                "compact_over_eager": ratio,
            })
        );
    }
    let min_ratio = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ratio = ratios.iter().copied().fold(0.0, f64::max);
    println!(
        "{}",
        json!({
            "kind": "summary",
            "operation": operation.name(),
            "eager_median_ns": median(&mut eager_samples),
            "compact_median_ns": median(&mut compact_samples),
            "median_paired_compact_over_eager": median_f64(&mut ratios),
            "min_paired_ratio": min_ratio,
            "max_paired_ratio": max_ratio,
        })
    );

    Ok(())
}

/// Correctness of the operation results: independent point evaluation of the
/// eager and compact results, and the eager zero check outside the projector.
/// An operation failure is recorded instead of aborting the run.
fn check_operation(
    workload: &Workload,
    patches: &[PatchPair],
    operation: Operation,
    truncation_cap: usize,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let center = &workload.center;
    let limit = match operation {
        Operation::Norm => return Ok(()),
        Operation::Truncate => TRUNCATE_RESIDUAL_LIMIT,
        Operation::Add => RESIDUAL_LIMIT,
    };
    let apply = |tree: &Network| -> Result<Network> {
        match operation {
            Operation::Truncate => truncate(tree, center, truncation_cap),
            _ => Ok(tree.add(tree)?),
        }
    };
    let mut worst = 0.0f64;
    // Largest eager magnitude outside the projector over the sampled maximum
    // inside it: rounding can leave tiny values outside after truncation.
    let mut worst_outside = 0.0f64;
    for patch in patches {
        let eager_result = match apply(&patch.eager) {
            Ok(result) => result,
            Err(error) => return record_failure(operation, "eager", &*error),
        };
        let compact_result = match apply(&patch.compact) {
            Ok(result) => result,
            Err(error) => return record_failure(operation, "compact", &*error),
        };
        let points = inside_points(workload, &patch.projector, rng);
        let (residual, scale) = compare_values(
            &eager_result,
            &compact_result,
            workload,
            &patch.active_sites,
            &points,
        )?;
        worst = worst.max(residual);
        let result_pair = PatchPair {
            projector: patch.projector.clone(),
            eager: eager_result,
            compact: compact_result,
            active_sites: patch.active_sites.clone(),
        };
        worst_outside = worst_outside
            .max(outside_max_abs(workload, &result_pair, rng)? / scale.max(f64::MIN_POSITIVE));
    }
    println!(
        "{}",
        json!({
            "kind": "correctness",
            "stage": "operation_result",
            "operation": operation.name(),
            "max_relative_value_residual": worst,
            "max_relative_eager_outside_projector": worst_outside,
            "limit": limit,
            "pass": worst <= limit && worst_outside <= limit,
        })
    );
    Ok(())
}

fn record_failure(operation: Operation, representation: &str, error: &dyn Error) -> Result<()> {
    println!(
        "{}",
        json!({
            "kind": "operation_failure",
            "operation": operation.name(),
            "representation": representation,
            "error": error.to_string(),
        })
    );
    Ok(())
}

fn truncate(tree: &Network, center: &Name, cap: usize) -> Result<Network> {
    Ok(tree.clone().truncate(
        [center.clone()],
        TruncationOptions::default().with_max_bond_dim(cap),
    )?)
}

/// Time one operation on every patch and return the summed nanoseconds. The
/// clone that `norm` and `truncate` consume is made outside the timed region,
/// and results are dropped after it.
fn pass_ns(trees: &[&Network], operation: Operation, center: &Name, cap: usize) -> Result<u128> {
    let mut total = 0u128;
    for tree in trees {
        match operation {
            Operation::Norm => {
                let mut copy = (*tree).clone();
                let start = Instant::now();
                let norm = copy.norm()?;
                total += start.elapsed().as_nanos();
                black_box(norm);
            }
            Operation::Truncate => {
                let copy = (*tree).clone();
                let options = TruncationOptions::default().with_max_bond_dim(cap);
                let start = Instant::now();
                let truncated = copy.truncate([center.clone()], options)?;
                total += start.elapsed().as_nanos();
                black_box(&truncated);
            }
            Operation::Add => {
                let start = Instant::now();
                let sum = tree.add(tree)?;
                total += start.elapsed().as_nanos();
                black_box(&sum);
            }
        }
    }
    Ok(total)
}

fn remove_projected_sites(
    tree: &Network,
    projector: &Projector,
    node_sites: &BTreeMap<Name, Vec<DynIndex>>,
) -> Result<Network> {
    let mut node_names = tree.node_names();
    node_names.sort();
    let mut tensors = Vec::with_capacity(node_names.len());
    for node_name in &node_names {
        let node = tree
            .node_index(node_name)
            .ok_or_else(|| format!("missing node {node_name:?}"))?;
        let mut tensor = tree
            .tensor(node)
            .ok_or_else(|| format!("missing tensor {node_name:?}"))?
            .clone();
        for site in node_sites.get(node_name).into_iter().flatten() {
            if let Some(position) = projector.get(site) {
                tensor = tensor.select_indices(std::slice::from_ref(site), &[position])?;
            }
        }
        tensors.push(tensor);
    }
    Ok(TreeTN::from_tensors(tensors, node_names)?)
}

fn payload_bytes(tree: &Network) -> Result<u128> {
    let mut bytes = 0u128;
    for node_name in tree.node_names() {
        let node = tree
            .node_index(&node_name)
            .ok_or_else(|| format!("missing node {node_name:?}"))?;
        let tensor = tree
            .tensor(node)
            .ok_or_else(|| format!("missing tensor {node_name:?}"))?;
        if !tensor.is_f64() {
            return Err(format!("{node_name} is not f64").into());
        }
        let elements = tensor.indices().iter().try_fold(1u128, |size, index| {
            size.checked_mul(index.dim() as u128)
                .ok_or("tensor payload size overflow")
        })?;
        bytes = bytes
            .checked_add(elements.checked_mul(8).ok_or("tensor byte size overflow")?)
            .ok_or("TreeTN payload byte sum overflow")?;
    }
    Ok(bytes)
}

/// Uniform points of the patch in workload site order (column-major).
fn inside_points(workload: &Workload, projector: &Projector, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut points = Vec::with_capacity(INSIDE_POINTS * workload.sites.len());
    for _ in 0..INSIDE_POINTS {
        for site in &workload.sites {
            let value = match projector.get(site) {
                Some(value) => value,
                None => rng.random_range(0..site.dim()),
            };
            points.push(value);
        }
    }
    points
}

/// Evaluate a network at column-major points over `sites`.
fn values(tree: &Network, sites: &[DynIndex], points: &[usize]) -> Result<Vec<f64>> {
    let mut evaluator =
        TreeTNCachedEvaluator::new(tree, sites, CachedEvaluatorOptions::<Name>::default())?;
    let shape = [sites.len(), points.len() / sites.len().max(1)];
    Ok(evaluator.evaluate_batched_typed::<f64>(
        ColMajorArrayRef::new(points, &shape)?,
        EvaluationHint::default(),
    )?)
}

/// Max relative difference between the eager network at full points and the
/// compact network at the same points restricted to its active sites, and the
/// largest eager magnitude at those points (the scale of the residual). The two
/// networks are evaluated independently; neither is derived from the other
/// inside this check.
fn compare_values(
    eager: &Network,
    compact: &Network,
    workload: &Workload,
    active_sites: &[DynIndex],
    points: &[usize],
) -> Result<(f64, f64)> {
    let n_sites = workload.sites.len();
    let active_positions: Vec<usize> = workload
        .sites
        .iter()
        .enumerate()
        .filter(|(_, site)| active_sites.contains(site))
        .map(|(position, _)| position)
        .collect();
    let active_points: Vec<usize> = points
        .chunks(n_sites)
        .flat_map(|point| {
            active_positions
                .iter()
                .map(move |&position| point[position])
        })
        .collect();
    let eager_values = values(eager, &workload.sites, points)?;
    let compact_values = if active_sites.is_empty() {
        // Every site is fixed: the compact network is a scalar network.
        let scalar = compact.clone().contract_to_tensor()?.to_vec::<f64>()?;
        vec![scalar.first().copied().ok_or("empty scalar network")?; eager_values.len()]
    } else {
        values(compact, active_sites, &active_points)?
    };
    if eager_values.len() != compact_values.len() {
        return Err("eager and compact value counts differ".into());
    }
    let mut scale = 0.0f64;
    let mut difference = 0.0f64;
    for (&e, &c) in eager_values.iter().zip(&compact_values) {
        if !e.is_finite() || !c.is_finite() {
            return Err("non-finite network value".into());
        }
        scale = scale.max(e.abs());
        difference = difference.max((e - c).abs());
    }
    Ok((difference / scale.max(f64::MIN_POSITIVE), scale))
}

/// Max |eager| at uniform points outside the projector: one projected site
/// takes another value, the remaining sites are uniform.
fn outside_max_abs(workload: &Workload, patch: &PatchPair, rng: &mut ChaCha8Rng) -> Result<f64> {
    let projected: Vec<(usize, usize)> = workload
        .sites
        .iter()
        .enumerate()
        .filter_map(|(position, site)| patch.projector.get(site).map(|value| (position, value)))
        .collect();
    if projected.is_empty() {
        return Ok(0.0);
    }
    let mut points = Vec::with_capacity(OUTSIDE_POINTS * workload.sites.len());
    for _ in 0..OUTSIDE_POINTS {
        let start = points.len();
        for site in &workload.sites {
            points.push(rng.random_range(0..site.dim()));
        }
        for &(position, value) in &projected {
            points[start + position] = value;
        }
        let (position, value) = projected[rng.random_range(0..projected.len())];
        let dim = workload.sites[position].dim();
        points[start + position] = (value + 1 + rng.random_range(0..dim - 1)) % dim;
    }
    Ok(values(&patch.eager, &workload.sites, &points)?
        .into_iter()
        .fold(0.0, |worst, value| worst.max(value.abs())))
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2
    } else {
        values[middle]
    }
}

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}
