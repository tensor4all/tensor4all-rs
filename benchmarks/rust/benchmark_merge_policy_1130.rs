//! Issue #1130 acceptance input and independent dense oracle.
//!
//! `benchmark_merge_policy_1130 <original|interleaved> <goal8|tolerance> [measure]`
//! runs one fresh-process case. Omit `measure` for a functional check with no
//! timing/RSS output. Timing runs require the registered plan and agreed limits
//! in `docs/experiments/issue-1130-implementation/benchmark-plan.md`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use serde_json::json;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    reconstruction::{
        schedule_merge_refine, CoordinateGroup, MergePolicy, MergeRefineOptions,
        ReconstructionTarget, ReconstructionTolerance, SubsetOperatorOptions,
    },
    PartitionedTreeTN, Projector, SubDomainTreeTN, TreeTN,
};
use tensor4all_quanticstransform::{
    quantics_fourier_operator, quantics_fourier_operator_multiaxis, AxisBit, FourierOptions,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const SPECTATORS: usize = 3;
const MAX_INPUT_BOND: usize = 4;

/// splitmix64, so the input is reproducible without an RNG dependency.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Random complex chain on `sites` (node `i` holds `sites[i]`), bond <= 4.
fn random_chain(sites: &[DynIndex], rng: &mut Rng) -> Result<TreeTN<IdxTensor, usize>> {
    let n = sites.len();
    let bonds: Vec<DynIndex> = (0..n - 1)
        .map(|i| {
            let dim = MAX_INPUT_BOND.min(1 << (i + 1)).min(1 << (n - 1 - i));
            DynIndex::new_dyn(dim)
        })
        .collect();
    let mut tensors = Vec::with_capacity(n);
    for (i, site) in sites.iter().enumerate() {
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(site.clone());
        if i + 1 < n {
            indices.push(bonds[i].clone());
        }
        let size: usize = indices.iter().map(|index| index.dim).product();
        let data: Vec<Complex64> = (0..size)
            .map(|_| Complex64::new(rng.uniform(), rng.uniform()))
            .collect();
        tensors.push(IdxTensor::from_dense(indices, data)?);
    }
    Ok(TreeTN::from_tensors(tensors, (0..n).collect())?)
}

/// Dense values of `tree`, ordered most-significant site first:
/// `x = sum_j x_j 2^(n-1-j)` where site `j` carries `x_j`.
fn dense_msb_first(tree: &TreeTN<IdxTensor, usize>, sites: &[DynIndex]) -> Result<Vec<Complex64>> {
    let dense = tree.to_dense()?;
    let indices = dense.indices().to_vec();
    let storage_values = dense.to_vec::<Complex64>()?;
    let n = sites.len();
    let mut values = vec![Complex64::new(0.0, 0.0); 1 << n];
    for (x, value) in values.iter_mut().enumerate() {
        let mut storage = 0usize;
        for (position, index) in indices.iter().enumerate() {
            let j = sites
                .iter()
                .position(|s| s == index)
                .ok_or("unknown index")?;
            storage |= ((x >> (n - 1 - j)) & 1) << position;
        }
        *value = storage_values[storage];
    }
    Ok(values)
}

/// Product DFT from MSB-first input to column-major output, reversing
/// frequency significance separately on each selected axis.
fn dft_oracle(
    indices: &[DynIndex],
    sites: &[DynIndex],
    axes: &[Vec<usize>],
    values: &[Complex64],
) -> Result<Vec<Complex64>> {
    let n = sites.len();
    let mut expected = vec![Complex64::new(0.0, 0.0); 1 << n];
    let positions = indices
        .iter()
        .map(|index| {
            sites
                .iter()
                .position(|s| s == index)
                .ok_or("unknown output index")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (x, value) in values.iter().enumerate() {
        for frequency in 0..64usize {
            let mut phase = 0.0;
            for axis in axes {
                let input = axis
                    .iter()
                    .fold(0usize, |v, j| 2 * v + ((x >> (n - 1 - j)) & 1));
                let output = axis
                    .iter()
                    .rev()
                    .fold(0usize, |v, j| 2 * v + ((frequency >> j) & 1));
                phase += (input * output) as f64 / (1usize << axis.len()) as f64;
            }
            let mut storage = 0usize;
            for (position, &j) in positions.iter().enumerate() {
                let bit = if j < 6 {
                    (frequency >> j) & 1
                } else {
                    (x >> (n - 1 - j)) & 1
                };
                storage |= bit << position;
            }
            expected[storage] +=
                value * Complex64::from_polar(1.0 / 8.0, -2.0 * std::f64::consts::PI * phase);
        }
    }
    Ok(expected)
}

fn peak_rss_kib() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let value = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or("missing VmHWM")?;
    Ok(value.parse()?)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 || args.get(2).is_some_and(|s| s != "measure") {
        return Err(
            "usage: benchmark_merge_policy_1130 <original|interleaved> <goal8|tolerance> [measure]"
                .into(),
        );
    }
    let axes: Vec<Vec<usize>> = match args[0].as_str() {
        "original" => vec![(0..6).collect()],
        "interleaved" => vec![vec![0, 2, 4], vec![1, 3, 5]],
        _ => return Err("unknown case".into()),
    };
    let policy = match args[1].as_str() {
        "goal8" => MergePolicy::RankGoal(8_usize.try_into()?),
        "tolerance" => MergePolicy::Tolerance,
        _ => return Err("unknown policy".into()),
    };
    let measure = args.len() == 3;
    let sites: Vec<_> = (0..6 + SPECTATORS).map(|_| DynIndex::new_dyn(2)).collect();
    let mut rng = Rng(0x5eed_0006);
    let tree = random_chain(&sites, &mut rng)?;
    let full = SubDomainTreeTN::from_treetn(tree.clone())?;
    let leaves = (0..64usize)
        .map(|leaf| {
            let projector = Projector::from_pairs(
                sites[..6]
                    .iter()
                    .enumerate()
                    .map(|(j, site)| (site.clone(), (leaf >> (5 - j)) & 1)),
            )?;
            full.project(&projector)?
                .ok_or_else(|| "zero input leaf".into())
        })
        .collect::<Result<Vec<_>>>()?;
    let target =
        ReconstructionTarget::from_partition(&PartitionedTreeTN::from_subdomains(leaves)?)?;
    let operator = if axes.len() == 1 {
        quantics_fourier_operator(6, FourierOptions::default())?
    } else {
        let layout: Vec<_> = (0..6)
            .map(|j| AxisBit {
                axis: j % 2,
                bit: j / 2,
            })
            .collect();
        quantics_fourier_operator_multiaxis(&layout, FourierOptions::default())?
    };
    let groups = axes
        .iter()
        .map(|axis| CoordinateGroup::new(axis.iter().map(|&j| sites[j].clone()).collect()))
        .collect();
    let options = MergeRefineOptions {
        merge_policy: policy,
        coordinate_groups: Some(groups),
        ..Default::default()
    };
    if measure {
        let prime = Instant::now();
        let mut value = 1_u64;
        while prime.elapsed() < Duration::from_secs(1) {
            value = black_box(value.wrapping_mul(6364136223846793005).wrapping_add(1));
        }
        black_box(value);
    }
    let start = measure.then(Instant::now);
    let result = schedule_merge_refine(
        &target,
        &0,
        &operator,
        &sites[..6],
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance {
            rtol: 1e-8,
            atol: 0.0,
        },
        &options,
    )?;
    let seconds = start.map(|t| t.elapsed().as_secs_f64());
    let peak = if measure { Some(peak_rss_kib()?) } else { None };
    let report = result.report().clone();
    assert!(report.error_bound <= report.absolute_tolerance);
    assert_eq!(report.application_error_bound, 0.0);
    assert_eq!(report.applied_operator_count, 64);
    if policy == MergePolicy::Tolerance {
        assert_eq!(report.term_count, report.region_count);
        assert!(result.regions().all(|(_, terms)| terms.len() == 1));
    }
    let projectors: Vec<_> = result.regions().map(|(p, _)| p.clone()).collect();
    assert!(Projector::are_disjoint(&projectors));
    let mut total: Option<IdxTensor> = None;
    for patch in result.regions().flat_map(|(_, terms)| terms) {
        let dense = patch.data().to_dense()?;
        total = Some(match total {
            Some(previous) => previous.add(&dense)?,
            None => dense,
        });
    }
    let dense = total.ok_or("empty output")?;
    let input = dense_msb_first(&tree, &sites)?;
    let expected = IdxTensor::from_dense(
        dense.indices().to_vec(),
        dft_oracle(dense.indices(), &sites, &axes, &input)?,
    )?;
    let difference = dense.sub(&expected)?;
    let l2_error = difference.norm()?;
    let max_abs_error = difference.maxabs()?;
    assert!(
        l2_error <= report.error_bound + 1e-9,
        "error={l2_error:e}, report={report:?}"
    );
    assert!(max_abs_error <= report.error_bound + 1e-9);
    if policy == MergePolicy::Tolerance {
        result.into_partition()?;
    }
    println!(
        "{}",
        json!({
            "case": args[0], "policy": args[1], "seed": "0x5eed0006",
            "input_bond": MAX_INPUT_BOND, "spectators": SPECTATORS,
            "regions": report.region_count, "terms": report.term_count,
            "logical_parameters": report.logical_parameters, "max_bond": report.max_bond_dim,
            "transient_bond": report.max_transient_bond_dim,
            "compressions": report.compressions, "compression_attempts": report.compression_attempts,
            "error_bound": report.error_bound, "allowance": report.absolute_tolerance,
            "l2_error": l2_error, "max_abs_error": max_abs_error,
            "scheduler_seconds": seconds, "peak_rss_kib": peak,
        })
    );
    Ok(())
}
