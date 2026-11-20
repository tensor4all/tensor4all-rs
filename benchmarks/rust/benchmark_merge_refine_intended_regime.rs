// Merge-refine schedule against the whole-state application in the intended
// regime for issue #777 item F4: few input patches, many output bits, and a
// localized chirp input whose patch transforms are high rank.
//
// One cell per process, so `/usr/bin/time -v` reports that cell's peak RSS.
// Protocol notes are in `benchmarks/README.md`.
//
// Example:
//   benchmark_merge_refine_intended_regime --bits 12 --input-depth 2 \
//     --output-depth 2 --goal 8 --rtol 1e-6 --input-tol 1e-8 --repeats 3
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use serde_json::{json, Value};
use tensor4all_core::{
    factorize, Canonical, DynIndex, FactorizeOptions, IdxTensor, SvdTruncationPolicy,
};
use tensor4all_partitionedtreetn::{
    reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN, TreeTN,
};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{apply_linear_operator_to_indices, ApplyOptions, LinearOperator};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type MappingPairs = Vec<(DynIndex, DynIndex)>;

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let flag = |name: &str| -> Option<&String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let number = |name: &str, default: f64| -> Result<f64> {
        match flag(name) {
            None => Ok(default),
            Some(value) => Ok(value.parse()?),
        }
    };
    let bits = number("--bits", 12.0)? as usize;
    let input_depth = number("--input-depth", 2.0)? as usize;
    let output_depth = number("--output-depth", 2.0)? as usize;
    let goal = number("--goal", 8.0)? as usize;
    let rtol = number("--rtol", 1e-6)?;
    let input_tol = number("--input-tol", 1e-8)?;
    let chirp_turns = number("--chirp-turns", 16.0)?;
    let window = number("--window", 6.0)?;
    let repeats = number("--repeats", 3.0)? as usize;
    let per_level = !args.iter().any(|a| a == "--no-per-level");
    let deviation = !args.iter().any(|a| a == "--no-deviation");
    let with_schedule = !args.iter().any(|a| a == "--baseline-only");
    let with_baseline = !args.iter().any(|a| a == "--schedule-only");
    // One method per process keeps a peak-RSS reading attributable to it.
    let only = flag("--only").map(String::as_str);

    println!(
        "{}",
        json!({
            "kind": "build",
            "benchmark": "merge_refine_intended_regime",
            "issue": 777,
            "item": "F4",
            "build_commit": option_env!("T4A_BENCH_GIT_COMMIT").unwrap_or("unrecorded"),
            "bits": bits,
            "input_depth": input_depth,
            "output_depth": output_depth,
            "target_bond_dim": goal,
            "rtol": rtol,
            "input_truncation_threshold": input_tol,
            "chirp_turns": chirp_turns,
            "window_divisor": window,
            "repeats": repeats,
            "deviation": deviation,
            "only": only.unwrap_or("all"),
            "input": "w(x) = exp(-((x - N/2)/(N/window))^2) * exp(2*pi*i*chirp_turns*(x/N)^2)",
        })
    );

    let case = Case::new(bits, input_depth, chirp_turns, window, input_tol)?;
    println!("{}", case.input_row());
    let reference = if deviation {
        Some(case.reference()?)
    } else {
        None
    };
    if with_baseline {
        println!(
            "{}",
            case.sum_then_transform(reference.as_deref(), repeats)?
        );
    }
    if with_schedule {
        for (label, options) in [
            (
                "merge_refine_exact",
                MergeRefineOptions {
                    target_bond_dim: None,
                    ..Default::default()
                },
            ),
            (
                "merge_refine_adaptive",
                MergeRefineOptions {
                    target_bond_dim: Some(goal),
                    ..Default::default()
                },
            ),
        ] {
            if only.is_some_and(|name| !label.ends_with(name)) {
                continue;
            }
            println!(
                "{}",
                case.schedule(
                    label,
                    &options,
                    &Run {
                        rtol,
                        output_depth,
                        repeats,
                        per_level,
                    },
                    reference.as_deref(),
                )?
            );
        }
    }
    Ok(())
}

/// One protocol point: tolerance, output depth, repeats and per-level reporting.
struct Run {
    rtol: f64,
    output_depth: usize,
    repeats: usize,
    per_level: bool,
}

/// One input on `bits` coordinates, split into `2^input_depth` dyadic patches,
/// plus the QFT operator and the dense reference trajectory.
struct Case {
    bits: usize,
    input_depth: usize,
    sites: Vec<DynIndex>,
    tree: TreeTN<IdxTensor, usize>,
    input_rank: usize,
    input_error: f64,
    input_norm_error: f64,
    preimage: ReconstructionTarget,
    operator: LinearOperator<IdxTensor, usize>,
}

impl Case {
    fn new(
        bits: usize,
        input_depth: usize,
        chirp_turns: f64,
        window: f64,
        input_tol: f64,
    ) -> Result<Self> {
        if input_depth > bits {
            return Err("--input-depth must not exceed --bits".into());
        }
        let sites: Vec<DynIndex> = (0..bits).map(|_| DynIndex::new_dyn(2)).collect();
        let values = chirp(bits, chirp_turns, window);
        // `IdxTensor` storage is column-major with the first index varying
        // fastest, so the first site carries the least significant bit of the
        // storage index while the operator reads the first site as the most
        // significant coordinate bit.
        let storage: Vec<Complex64> = (0..values.len())
            .map(|x| values[bit_reverse(x, bits as u32)])
            .collect();
        let (tree, input_rank) = truncated_mps(&sites, &storage, input_tol)?;
        let dense = dense_msb_first(&tree, &sites)?;
        let input_error = max_error(&dense, &values);
        let input_norm_error = (norm(&dense) - norm(&values)).abs() / norm(&values);
        let preimage = dyadic_patches(&tree, &sites, input_depth)?;
        let operator = quantics_fourier_operator(bits, FourierOptions::default())?;
        Ok(Self {
            bits,
            input_depth,
            sites,
            tree,
            input_rank,
            input_error,
            input_norm_error,
            preimage,
            operator,
        })
    }

    fn input_row(&self) -> Value {
        json!({
            "kind": "input",
            "bits": self.bits,
            "input_depth": self.input_depth,
            "input_leaf_count": 1usize << self.input_depth,
            "input_rank": self.input_rank,
            "input_max_deviation_from_analytic": self.input_error,
            "input_relative_norm_deviation": self.input_norm_error,
        })
    }

    /// The dense normalized DFT of the represented input, in the operator's
    /// documented convention (selected bits most significant first, frequency
    /// bit `t` on selected position `t`).
    fn reference(&self) -> Result<Vec<Complex64>> {
        let values = dense_msb_first(&self.tree, &self.sites)?;
        Ok(storage_dft(&values))
    }

    /// Apply the complete transform to the whole input at once.
    fn sum_then_transform(&self, reference: Option<&[Complex64]>, repeats: usize) -> Result<Value> {
        let (inputs, outputs) = self.mappings();
        let (elapsed, applied) = median_time(repeats, || {
            Ok(apply_linear_operator_to_indices(
                &self.operator,
                &self.tree,
                &inputs,
                &outputs,
                ApplyOptions::naive(),
            )?)
        })?;
        let (dense, l2) = match reference {
            Some(reference) => {
                let dense = dense_msb_first(&applied, &self.sites)?;
                (
                    json!(max_error(&dense, reference)),
                    json!(l2_error(&dense, reference)),
                )
            }
            None => (Value::Null, Value::Null),
        };
        Ok(json!({
            "kind": "measurement",
            "method": "sum_then_transform",
            "bits": self.bits,
            "input_depth": self.input_depth,
            "output_depth": self.bits,
            "elapsed_ms": elapsed.as_secs_f64() * 1e3,
            "max_bond_dim": max_bond_dim(&applied),
            "max_transient_bond_dim": Value::Null,
            "region_count": 1,
            "term_count": 1,
            "applied_operator_count": 1,
            "additions": 0,
            "compressions": 0,
            "error_bound": 0.0,
            "deviation_from_reference": dense,
            "l2_deviation_from_reference": l2,
            "reference_l2": norm(reference.unwrap_or(&[])),
        }))
    }

    /// One merge-refine run at `output_depth`, plus one run per shallower depth
    /// when `per_level` is set.
    fn schedule(
        &self,
        label: &str,
        options: &MergeRefineOptions<usize>,
        run: &Run,
        reference: Option<&[Complex64]>,
    ) -> Result<Value> {
        let Run {
            rtol,
            output_depth,
            repeats,
            per_level,
        } = *run;
        let run = |depth: usize| -> Result<(Duration, MergeRefineResult<usize>)> {
            let mut options = options.clone();
            options.output_depth = Some(depth);
            median_time(repeats, || {
                Ok(schedule_merge_refine(
                    &self.preimage,
                    &0,
                    &self.operator,
                    &self.sites,
                    &SubsetOperatorOptions { unitary: true },
                    ReconstructionTolerance { rtol, atol: 0.0 },
                    &options,
                )?)
            })
        };
        // The depths below the requested one describe the same trajectory, but
        // each is a separate run whose own level count splits the allowance, so
        // report each run's cumulative time rather than differencing silently.
        let mut rows = Vec::new();
        let mut previous_ms = 0.0;
        let depths: Vec<usize> = if per_level {
            (0..=output_depth).collect()
        } else {
            vec![output_depth]
        };
        for depth in depths {
            let (elapsed, result) = run(depth)?;
            let report = result.report();
            let (deviation, l2) = match reference {
                Some(reference) => {
                    let dense = dense_of(&result, &self.sites)?;
                    (
                        json!(max_error(&dense, reference)),
                        json!(l2_error(&dense, reference)),
                    )
                }
                None => (Value::Null, Value::Null),
            };
            let elapsed_ms = elapsed.as_secs_f64() * 1e3;
            rows.push(json!({
                "kind": "level",
                "method": label,
                "bits": self.bits,
                "input_depth": self.input_depth,
                "output_depth": depth,
                "level_count": report.level_count,
                "cumulative_ms": elapsed_ms,
                "delta_ms": elapsed_ms - previous_ms,
                "max_bond_dim": report.max_bond_dim,
                "max_transient_bond_dim": report.max_transient_bond_dim,
                "region_count": report.region_count,
                "term_count": report.term_count,
                "peak_work_items": report.peak_work_items,
                "work_items_per_level": report.work_items_per_level,
                "applied_operator_count": report.applied_operator_count,
                "additions": report.additions,
                "projections": report.projections,
                "compression_attempts": report.compression_attempts,
                "compressions": report.compressions,
                "dropped_terms": report.dropped_terms,
                "refined_regions": report.refined_regions,
                "stopped_regions": report.stopped_regions,
                "logical_parameters": report.logical_parameters,
                "error_bound": report.error_bound,
                "deviation_from_reference": deviation,
                "l2_deviation_from_reference": l2,
                "reference_l2": norm(reference.unwrap_or(&[])),
            }));
            previous_ms = elapsed_ms;
        }
        let last = rows.last().cloned().ok_or("no schedule depth")?;
        for row in &rows[..rows.len() - 1] {
            println!("{row}");
        }
        let mut measurement = last;
        measurement["kind"] = json!("measurement");
        Ok(measurement)
    }

    fn mappings(&self) -> (MappingPairs, MappingPairs) {
        let mut nodes = self.operator.mpo().node_names();
        nodes.sort();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for (node, index) in nodes.iter().zip(&self.sites) {
            inputs.push((
                self.operator
                    .get_input_mapping(node)
                    .expect("input mapping")
                    .true_index
                    .clone(),
                index.clone(),
            ));
            outputs.push((
                self.operator
                    .get_output_mapping(node)
                    .expect("output mapping")
                    .true_index
                    .clone(),
                index.clone(),
            ));
        }
        (inputs, outputs)
    }
}

/// The normalized chirp used as input, in the operator's coordinate order.
fn chirp(bits: usize, turns: f64, window: f64) -> Vec<Complex64> {
    let n = 1usize << bits;
    let sigma = n as f64 / window;
    let center = n as f64 / 2.0;
    (0..n)
        .map(|x| {
            let x = x as f64;
            let envelope = (-((x - center) / sigma).powi(2)).exp();
            let phase = 2.0 * std::f64::consts::PI * turns * (x / n as f64).powi(2);
            Complex64::from_polar(envelope, phase)
        })
        .collect()
}

/// Truncated left-to-right TT-SVD of a dense vector, most significant site first.
fn truncated_mps(
    sites: &[DynIndex],
    values: &[Complex64],
    tolerance: f64,
) -> Result<(TreeTN<IdxTensor, usize>, usize)> {
    let options = FactorizeOptions::svd()
        .with_canonical(Canonical::Left)
        .with_svd_policy(SvdTruncationPolicy::new(tolerance));
    let mut rest = IdxTensor::from_dense(sites.to_vec(), values.to_vec())?;
    let mut tensors = Vec::with_capacity(sites.len());
    let mut rank = 0usize;
    for site in &sites[..sites.len() - 1] {
        let left = match tensors.last() {
            None => vec![site.clone()],
            Some(_) => vec![rest.indices()[0].clone(), site.clone()],
        };
        let factorized = factorize(&rest, &left, &options)?;
        rank = rank.max(factorized.rank);
        tensors.push(factorized.left);
        rest = factorized.right;
    }
    tensors.push(rest);
    let tree = TreeTN::from_tensors(tensors, (0..sites.len()).collect())?;
    Ok((tree, rank))
}

/// The `2^input_depth` dyadic patches fixing a prefix of the selected bits.
fn dyadic_patches(
    tree: &TreeTN<IdxTensor, usize>,
    sites: &[DynIndex],
    input_depth: usize,
) -> Result<ReconstructionTarget> {
    let full = SubDomainTreeTN::from_treetn(tree.clone())?;
    let mut leaves = Vec::with_capacity(1usize << input_depth);
    for leaf in 0..1usize << input_depth {
        let pairs = sites[..input_depth]
            .iter()
            .enumerate()
            .map(|(position, site)| (site.clone(), (leaf >> (input_depth - 1 - position)) & 1));
        let projector = Projector::from_pairs(pairs)?;
        leaves.push(full.project(&projector)?.ok_or("zero input patch")?);
    }
    Ok(ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(leaves)?,
    )?)
}

/// Sum every retained term of a schedule result into the reference order.
fn dense_of(result: &MergeRefineResult<usize>, sites: &[DynIndex]) -> Result<Vec<Complex64>> {
    let mut indices: Option<Vec<DynIndex>> = None;
    let mut total: Option<Vec<Complex64>> = None;
    for (_, terms) in result.regions() {
        for term in terms {
            let dense = term.data().to_dense()?;
            let term_indices = dense.indices().to_vec();
            let values = dense.to_vec::<Complex64>()?;
            match (&indices, &mut total) {
                (None, _) => {
                    indices = Some(term_indices);
                    total = Some(values);
                }
                (Some(target), Some(slot)) => {
                    let aligned = reorder(&values, &term_indices, target);
                    for (sum, value) in slot.iter_mut().zip(aligned) {
                        *sum += value;
                    }
                }
                _ => return Err("inconsistent accumulator".into()),
            }
        }
    }
    let values = total.ok_or("no retained terms")?;
    let mut target: Vec<DynIndex> = sites.to_vec();
    target.reverse();
    Ok(reorder(
        &values,
        indices.as_ref().ok_or("no index order")?,
        &target,
    ))
}

fn dense_msb_first(tree: &TreeTN<IdxTensor, usize>, sites: &[DynIndex]) -> Result<Vec<Complex64>> {
    let dense = tree.to_dense()?;
    let mut target: Vec<DynIndex> = sites.to_vec();
    target.reverse();
    Ok(reorder(
        &dense.to_vec::<Complex64>()?,
        dense.indices(),
        &target,
    ))
}

/// Reorder a column-major dense vector from one index order to another.
fn reorder(values: &[Complex64], from: &[DynIndex], to: &[DynIndex]) -> Vec<Complex64> {
    let n = to.len();
    let mut out = vec![Complex64::new(0.0, 0.0); 1usize << n];
    let positions: Vec<usize> = to
        .iter()
        .map(|index| {
            from.iter()
                .position(|candidate| candidate == index)
                .expect("index present in the source order")
        })
        .collect();
    for (storage, slot) in out.iter_mut().enumerate() {
        let mut source = 0usize;
        for (position, from_position) in positions.iter().enumerate() {
            source |= ((storage >> position) & 1) << from_position;
        }
        *slot = values[source];
    }
    out
}

/// Normalized DFT in the operator's storage convention: a dense index `x`
/// carries the selected bits most significant first, and frequency bit `t`
/// lands on selected position `t`, so the output is the bit-reversed transform.
fn storage_dft(values: &[Complex64]) -> Vec<Complex64> {
    let n = values.len();
    let scale = 1.0 / (n as f64).sqrt();
    let spectrum = fft(values);
    let bits = n.trailing_zeros();
    spectrum.into_iter().enumerate().fold(
        vec![Complex64::new(0.0, 0.0); n],
        |mut out, (k, value)| {
            out[bit_reverse(k, bits)] = value * scale;
            out
        },
    )
}

/// In-place-free iterative radix-2 FFT with the negative exponent.
fn fft(values: &[Complex64]) -> Vec<Complex64> {
    let n = values.len();
    assert!(n.is_power_of_two());
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = values[bit_reverse(k, n.trailing_zeros())];
    }
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for j in 0..len / 2 {
                let twiddle = Complex64::from_polar(1.0, angle * j as f64);
                let even = out[start + j];
                let odd = out[start + j + len / 2] * twiddle;
                out[start + j] = even + odd;
                out[start + j + len / 2] = even - odd;
            }
        }
        len *= 2;
    }
    out
}

fn bit_reverse(value: usize, bits: u32) -> usize {
    (0..bits).fold(0usize, |out, bit| {
        out | (((value >> bit) & 1) << (bits - 1 - bit))
    })
}

fn norm(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// Euclidean norm of the deviation, the quantity `error_bound` bounds.
fn l2_error(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn max_error(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max)
}

fn max_bond_dim(tree: &TreeTN<IdxTensor, usize>) -> usize {
    let mut max = 0usize;
    for node in tree.node_names() {
        if let Some(index) = tree.node_index(&node) {
            for (edge, _) in tree.edges_for_node(index) {
                if let Some(bond) = tree.bond_index(edge) {
                    max = max.max(bond.dim);
                }
            }
        }
    }
    max
}

/// Median wall time of `repeats` runs, plus the last successful result.
fn median_time<T>(repeats: usize, mut run: impl FnMut() -> Result<T>) -> Result<(Duration, T)> {
    let mut samples = Vec::with_capacity(repeats);
    let mut last: Option<T> = None;
    for _ in 0..repeats {
        let start = Instant::now();
        let value = run()?;
        samples.push(start.elapsed());
        last = Some(value);
    }
    black_box(&samples);
    samples.sort();
    let median = samples[samples.len() / 2];
    last.map(|value| (median, value))
        .ok_or_else(|| "no measurement repeats".into())
}
