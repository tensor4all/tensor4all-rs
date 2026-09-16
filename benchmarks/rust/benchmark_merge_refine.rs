// Merge-refine schedule and greedy-reconstruction comparison for issue #752.
//
// The body is included by
// `crates/tensor4all-partitionedtreetn/examples/benchmark_merge_refine.rs`.
// Protocol notes are in `benchmarks/README.md`.
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use serde_json::{json, Value};
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN, TreeTN,
};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{
    apply_linear_operator_to_indices, ApplyOptions, LinearOperator,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
/// Operator true-index to state-index bindings for one side of the operator.
type MappingPairs = Vec<(DynIndex, DynIndex)>;

/// Measurements per configuration; the reported elapsed time is the median.
const REPEATS: usize = 3;
/// Relative tolerance for every approximate method, held fixed across methods.
const RTOL: f64 = 1e-6;
/// Soft rank goal for every rank-limited method, held fixed across methods.
const GOAL: usize = 8;
/// Selected-bit counts to exercise.
const BITS: [usize; 2] = [3, 4];
/// Input families.
const FAMILIES: [&str; 3] = ["smooth", "spike", "comb"];

fn main() -> Result<()> {
    println!(
        "{}",
        json!({
            "kind": "build",
            "benchmark": "merge_refine_schedule",
            "issue": 752,
            "build_commit": option_env!("T4A_BENCH_GIT_COMMIT").unwrap_or("unrecorded"),
            "repeats": REPEATS,
            "rtol": RTOL,
            "target_bond_dim": GOAL,
            "protocol": "serial schedule; pin affinity and the Rayon/OMP/BLAS thread \
                         variables to one for comparable timings",
        })
    );
    for bits in BITS {
        for family in FAMILIES {
            let case = Case::new(family, bits)?;
            let reference = case.exact_reference()?;
            println!("{}", case.sum_then_transform(&reference)?);
            println!("{}", case.exact(&reference)?);
            println!("{}", case.adaptive(&reference)?);
            println!("{}", case.adaptive_with_truncating_application(&reference)?);
            println!("{}", case.greedy(&reference)?);
        }
    }
    Ok(())
}

/// One input family on `bits` selected sites, plus its dyadic input partition.
struct Case {
    family: &'static str,
    sites: Vec<DynIndex>,
    tree: TreeTN<IdxTensor, usize>,
    preimage: ReconstructionTarget,
    operator: LinearOperator<IdxTensor, usize>,
}

impl Case {
    fn new(family: &'static str, bits: usize) -> Result<Self> {
        let sites: Vec<DynIndex> = (0..bits).map(|_| DynIndex::new_dyn(2)).collect();
        let size = 1usize << bits;
        let values: Vec<Complex64> = (0..size)
            .map(|m| match family {
                "smooth" => Complex64::new(1.0 + (0.3 * m as f64).sin(), (0.2 * m as f64).cos()),
                "spike" => one_hot(m == size / 2),
                "comb" => one_hot(m % 4 == 0),
                other => unreachable!("unknown family {other}"),
            })
            .collect();
        let tree = mps(&sites, &values)?;
        let preimage = dyadic_input_leaves(&tree, &sites)?;
        let operator = quantics_fourier_operator(bits, FourierOptions::default())?;
        Ok(Self {
            family,
            sites,
            tree,
            preimage,
            operator,
        })
    }

    fn bits(&self) -> usize {
        self.sites.len()
    }

    /// Apply the complete transform to the whole input at once: the baseline the
    /// schedule avoids, since it materializes the global output rank.
    fn sum_then_transform(
        &self,
        reference: &(Vec<DynIndex>, Vec<Complex64>),
    ) -> Result<Value> {
        let (inputs, outputs) = self.mappings();
        let (elapsed, applied) = median_time(REPEATS, || {
            let applied = apply_linear_operator_to_indices(
                &self.operator,
                &self.tree,
                &inputs,
                &outputs,
                ApplyOptions::naive(),
            )?;
            Ok(applied)
        })?;
        let dense = dense_values(&applied)?;
        Ok(json!({
            "kind": "measurement",
            "family": self.family,
            "bits": self.bits(),
            "method": "sum_then_transform",
            "elapsed_ms": elapsed.as_secs_f64() * 1e3,
            "max_bond_dim": max_bond_dim(&applied),
            "deviation_from_exact": max_error(&dense.1, &reference.1),
        }))
    }

    /// The exact merge-refine trajectory, which the tests establish as the DFT.
    fn exact(&self, reference: &(Vec<DynIndex>, Vec<Complex64>)) -> Result<Value> {
        let options = MergeRefineOptions {
            target_bond_dim: None,
            ..Default::default()
        };
        let (elapsed, result) = median_time(REPEATS, || {
            let result = schedule_merge_refine(
                &self.preimage,
                &0,
                &self.operator,
                &self.sites,
                &SubsetOperatorOptions { unitary: true },
                ReconstructionTolerance {
                    rtol: 0.0,
                    atol: 0.0,
                },
                &options,
            )?;
            Ok(result)
        })?;
        let deviation = max_error(&dense_of(&result)?.1, &reference.1);
        Ok(self.schedule_row("merge_refine_exact", elapsed, result.report(), deviation))
    }

    fn adaptive(&self, reference: &(Vec<DynIndex>, Vec<Complex64>)) -> Result<Value> {
        self.approximate_row(
            "merge_refine_adaptive",
            &MergeRefineOptions {
                target_bond_dim: Some(GOAL),
                ..Default::default()
            },
            reference,
        )
    }

    fn adaptive_with_truncating_application(
        &self,
        reference: &(Vec<DynIndex>, Vec<Complex64>),
    ) -> Result<Value> {
        self.approximate_row(
            "merge_refine_adaptive_truncated_apply",
            &MergeRefineOptions {
                target_bond_dim: Some(GOAL),
                apply_options: Some(ApplyOptions::zipup().with_max_bond_dim(GOAL)),
                ..Default::default()
            },
            reference,
        )
    }

    fn approximate_row(
        &self,
        label: &str,
        options: &MergeRefineOptions,
        reference: &(Vec<DynIndex>, Vec<Complex64>),
    ) -> Result<Value> {
        let (elapsed, result) = median_time(REPEATS, || {
            let result = schedule_merge_refine(
                &self.preimage,
                &0,
                &self.operator,
                &self.sites,
                &SubsetOperatorOptions { unitary: true },
                ReconstructionTolerance { rtol: RTOL, atol: 0.0 },
                options,
            )?;
            Ok(result)
        })?;
        let deviation = max_error(&dense_of(&result)?.1, &reference.1);
        Ok(self.schedule_row(label, elapsed, result.report(), deviation))
    }

    /// The greedy engine on the same subset-operator target.
    fn greedy(&self, reference: &(Vec<DynIndex>, Vec<Complex64>)) -> Result<Value> {
        let (elapsed, result) = median_time(REPEATS, || {
            let target = ReconstructionTarget::from_subset_operator(
                &self.preimage,
                &0,
                &self.operator,
                &self.sites,
                &SubsetOperatorOptions { unitary: true },
            )?;
            let result = reconstruct(
                &target,
                &0,
                ReconstructionTolerance { rtol: RTOL, atol: 0.0 },
                &ReconstructionOptions {
                    target_bond_dim: Some(GOAL),
                    ..Default::default()
                },
            )?;
            Ok(result)
        })?;
        let report = result.report().clone();
        let deviation = max_error(&dense_of_reconstruction(&result)?.1, &reference.1);
        Ok(json!({
            "kind": "measurement",
            "family": self.family,
            "bits": self.bits(),
            "method": "greedy_reconstruct",
            "elapsed_ms": elapsed.as_secs_f64() * 1e3,
            "reference_scale": report.reference_scale,
            "absolute_tolerance": report.absolute_tolerance,
            "error_bound": report.error_bound,
            "region_count": report.region_count,
            "term_count": report.term_count,
            "max_bond_dim": report.max_bond_dim,
            "split_count": report.split_count,
            "merge_count": report.merge_count,
            "deviation_from_exact": deviation,
        }))
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

    /// Dense exact trajectory, used as the deviation reference for every method.
    fn exact_reference(&self) -> Result<(Vec<DynIndex>, Vec<Complex64>)> {
        let result = schedule_merge_refine(
            &self.preimage,
            &0,
            &self.operator,
            &self.sites,
            &SubsetOperatorOptions { unitary: true },
            ReconstructionTolerance {
                rtol: 0.0,
                atol: 0.0,
            },
            &MergeRefineOptions {
                target_bond_dim: None,
                ..Default::default()
            },
        )?;
        dense_of(&result)
    }

    fn schedule_row(
        &self,
        method: &str,
        elapsed: Duration,
        report: &MergeRefineReport,
        deviation: f64,
    ) -> Value {
        json!({
            "kind": "measurement",
            "family": self.family,
            "bits": self.bits(),
            "method": method,
            "elapsed_ms": elapsed.as_secs_f64() * 1e3,
            "reference_scale": report.reference_scale,
            "absolute_tolerance": report.absolute_tolerance,
            "error_bound": report.error_bound,
            "level_count": report.level_count,
            "applied_operator_count": report.applied_operator_count,
            "additions": report.additions,
            "projections": report.projections,
            "compression_attempts": report.compression_attempts,
            "compressions": report.compressions,
            "work_items_per_level": report.work_items_per_level,
            "peak_work_items": report.peak_work_items,
            "refined_regions": report.refined_regions,
            "stopped_regions": report.stopped_regions,
            "region_count": report.region_count,
            "term_count": report.term_count,
            "max_bond_dim": report.max_bond_dim,
            "max_transient_bond_dim": report.max_transient_bond_dim,
            "logical_parameters": report.logical_parameters,
            "deviation_from_exact": deviation,
        })
    }
}

fn one_hot(flag: bool) -> Complex64 {
    if flag {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(0.0, 0.0)
    }
}

/// Most-significant-site-first chain MPS over `sites`.
fn mps(sites: &[DynIndex], values: &[Complex64]) -> Result<TreeTN<IdxTensor, usize>> {
    let r = sites.len();
    let bonds: Vec<DynIndex> = (1..r).map(|i| DynIndex::new_dyn(1usize << i)).collect();
    let mut tensors = Vec::with_capacity(r);
    for (i, site) in sites.iter().enumerate() {
        let left_dim = 1usize << i;
        let has_right = i + 1 < r;
        let right_dim = if has_right { 1usize << (i + 1) } else { 1 };
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(site.clone());
        if has_right {
            indices.push(bonds[i].clone());
        }
        let mut data = vec![Complex64::new(0.0, 0.0); left_dim * 2 * right_dim];
        for left in 0..left_dim {
            for bit in 0..2 {
                if has_right {
                    let right = left * 2 + bit;
                    let offset = if i > 0 {
                        left + left_dim * (bit + 2 * right)
                    } else {
                        bit + 2 * right
                    };
                    data[offset] = Complex64::new(1.0, 0.0);
                } else {
                    let offset = if i > 0 { left + left_dim * bit } else { bit };
                    data[offset] = values[left * 2 + bit];
                }
            }
        }
        tensors.push(IdxTensor::from_dense(indices, data)?);
    }
    Ok(TreeTN::from_tensors(tensors, (0..r).collect())?)
}

/// The `2^bits` dyadic input leaves as an immutable target.
fn dyadic_input_leaves(
    tree: &TreeTN<IdxTensor, usize>,
    sites: &[DynIndex],
) -> Result<ReconstructionTarget> {
    let full = SubDomainTreeTN::from_treetn(tree.clone())?;
    let mut leaves = Vec::with_capacity(1usize << sites.len());
    for leaf in 0..1usize << sites.len() {
        let pairs = sites.iter().enumerate().map(|(position, site)| {
            (site.clone(), (leaf >> (sites.len() - 1 - position)) & 1)
        });
        let projector = Projector::from_pairs(pairs)?;
        leaves.push(full.project(&projector)?.ok_or("zero input leaf")?);
    }
    Ok(ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(leaves)?,
    )?)
}

/// Sum every retained term of a schedule result, in one canonical index order.
fn dense_of(result: &MergeRefineResult<usize>) -> Result<(Vec<DynIndex>, Vec<Complex64>)> {
    let mut accumulator = DenseAccumulator::default();
    for (_, terms) in result.regions() {
        for term in terms {
            accumulator.add(term.data())?;
        }
    }
    accumulator.finish()
}

/// Sum every retained term of a greedy reconstruction result.
fn dense_of_reconstruction(
    result: &ReconstructedTreeTN<usize>,
) -> Result<(Vec<DynIndex>, Vec<Complex64>)> {
    let mut accumulator = DenseAccumulator::default();
    for (_, terms) in result.regions() {
        for term in terms {
            accumulator.add(term.data())?;
        }
    }
    accumulator.finish()
}

fn dense_values(tree: &TreeTN<IdxTensor, usize>) -> Result<(Vec<DynIndex>, Vec<Complex64>)> {
    let dense = tree.to_dense()?;
    Ok((dense.indices().to_vec(), dense.to_vec::<Complex64>()?))
}

/// Accumulate dense networks into one canonical index order.
#[derive(Default)]
struct DenseAccumulator {
    indices: Option<Vec<DynIndex>>,
    total: Option<Vec<Complex64>>,
}

impl DenseAccumulator {
    fn add(&mut self, tree: &TreeTN<IdxTensor, usize>) -> Result<()> {
        let dense = tree.to_dense()?;
        let indices = dense.indices().to_vec();
        let values = dense.to_vec::<Complex64>()?;
        if self.total.is_none() {
            self.indices = Some(indices);
            self.total = Some(values);
            return Ok(());
        }
        let target = self.indices.as_ref().ok_or("missing index order")?;
        let aligned = reorder(&values, &indices, target);
        let slot = self.total.as_mut().ok_or("missing accumulator")?;
        for (sum, value) in slot.iter_mut().zip(aligned) {
            *sum += value;
        }
        Ok(())
    }

    fn finish(self) -> Result<(Vec<DynIndex>, Vec<Complex64>)> {
        self.indices
            .zip(self.total)
            .ok_or_else(|| "no retained terms to accumulate".into())
    }
}

/// Reorder a column-major dense vector from one index order to another.
fn reorder(values: &[Complex64], from: &[DynIndex], to: &[DynIndex]) -> Vec<Complex64> {
    let n = to.len();
    let mut out = vec![Complex64::new(0.0, 0.0); 1usize << n];
    for (storage, slot) in out.iter_mut().enumerate() {
        let mut source = 0usize;
        for (position, index) in to.iter().enumerate() {
            let q = from
                .iter()
                .position(|candidate| candidate == index)
                .expect("index present in the source order");
            source |= ((storage >> position) & 1) << q;
        }
        *slot = values[source];
    }
    out
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
