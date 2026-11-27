//! M6a benchmark, workload D: the 2D momentum Fourier transform
//! `G0(kx, ky, n) -> G(rx, ry, n)` of the gw-rs Matsubara G0, the first step
//! of gw-rs's `to_real_space`, on its `tt_nblock` chain or comb CTTN.
//!
//! Arms (`--arm`):
//! - `patched-contract`: patched G0 contracted with the Fourier operator by
//!   `contract_adaptive` (the operator is one patch on the whole tree);
//! - `unpatched-src`: unpatched G0 with `apply_linear_operator` and adaptive
//!   SRC, as gw-rs applies its transforms;
//! - `patched-merge-refine`: patched G0 with `patched_fourier_transform`
//!   (the lazy merge-refine front end of #1275); `--certified-share` selects
//!   the certified operator application of #1286, which needs a chain;
//! - `patched-src-each`: diagnostic, `apply_linear_operator` with adaptive SRC
//!   on every patch of the patched G0 separately, then a truncated pairwise
//!   sum when the images overlap.
//!
//! The Fourier operator is `quantics_fourier_operator_multiaxis` on the
//! contiguous interleaved k block of `tt_nblock`, and one
//! `quantics_fourier_operator` per axis on that axis's arm of the CTTN. Its
//! output bits are bit-reversed: the output site of node `3 b + a` carries bit
//! `b` (least significant first) of `r_a`.
//!
//! Accuracy is measured against the exact transform: for `--slices`
//! Matsubara indices `n`, `G(r, n)` is the normalized 2D FFT of `G0(k, n)` on
//! the full `2^R x 2^R` grid, sampled at `--samples` random points in total.
//! The slices and points are drawn from `--ref-seed` (default 42), not from
//! `--seed`, so runs at different seeds are scored on the same points.
//! `--check-dense true` compares every arm with the dense DFT on the whole
//! grid instead (small R only), which validates the conventions.
//!
//! Output: one JSON record per line (`config`, `g0`, `operator`, `result`).
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;
use tensor4all_partitionedtreetn::reconstruction::MergePolicy;

use anyhow::{bail, Context, Result};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike, SvdTruncationPolicy};
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    patched_elementwise, PatchedInterpolationOptions, VerificationOptions,
};
use tensor4all_partitionedtreetn::{
    contract_adaptive, ErrorNorm, ErrorTolerance, L2Reference, PartitionedTreeTN, PatchingOptions,
    Projector, SubDomainTreeTN,
};
use tensor4all_quanticstransform::{
    patched_fourier_transform, quantics_fourier_operator, quantics_fourier_operator_multiaxis,
    AxisBit, FourierOptions, PatchedFourierOptions,
};
use tensor4all_tensorbackend::explicit::Session;
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_treeaci::{
    tree_elementwise_batched_with_rng_on_session, TreeAciInterpolator, TreeAciOptions,
    TreeElementwiseBatch,
};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::contraction::{ContractionMethod, ContractionOptions, SrcOptions};
use tensor4all_treetn::interpolation::{EvalContext, Parallelism};
use tensor4all_treetn::{
    apply_linear_operator, compose_exclusive_linear_operators, ApplyOptions, IndexMapping,
    LinearOperator, PairwiseSumOrder, PerAddCutoff, TreeTN,
};

// The M6a benchmark uses only part of the shared grid.
#[allow(dead_code)]
#[path = "g0_grid.rs"]
mod g0_grid;
use g0_grid::{frequency_sums, Grid, MU};

type Tree = TreeTN<IdxTensor, usize>;
type Partition = PartitionedTreeTN<usize>;

// ---------------------------------------------------------------------------
// Arguments

#[derive(Debug, Clone)]
struct Args {
    r: usize,
    temperature: f64,
    topology: String,
    /// Seed of the G0 interpolation (pivot search).
    seed: u64,
    /// Seed of the accuracy reference (frequency slices and sample points),
    /// kept apart from `seed` so every seed is scored on the same points.
    ref_seed: u64,
    /// Patch cap of the patched G0 input.
    g0_cap: usize,
    /// Relative L2 tolerance of the G0 input (patched and unpatched).
    g0_rtol: f64,
    /// Sites the patched G0 may split on: `all` (every site, as M6c) or `k`
    /// (the momentum bits only, so no patch constrains a frequency bit; the
    /// merge-refine arm needs this, as `schedule_merge_refine` requires equal
    /// spectator constraints on every input patch).
    g0_patch_order: String,
    arm: String,
    /// Bond cap of the transform: patch cap (contract), SRC rank cap
    /// (unpatched) or merge-refine target bond (merge-refine).
    cap: usize,
    /// Relative tolerance of the transform.
    rtol: f64,
    /// Output split order of the patched contraction: `none` (empty; the first
    /// pair is then contracted to completion before any probe, as
    /// `PatchingOptions::patch_order` documents), `sites` (output indices in
    /// node order) or `n-first` (frequency bits from the most significant,
    /// then the momentum bits in node order).
    out_patch_order: String,
    /// Rank policy of the patched contraction's zip-up: `exact` (no
    /// truncation), `svd` (SVD policy with `rtol`) or `src` (randomized
    /// compression with `rtol`, rank cap `4 * cap`).
    contract: String,
    /// Share of the allowance for a certified application in the
    /// `patched-merge-refine` arm (#1286); `None` applies exactly.
    certified_share: Option<f64>,
    slices: usize,
    samples: usize,
    check_dense: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        r: 6,
        temperature: 0.1,
        topology: "tt_nblock".into(),
        seed: 42,
        ref_seed: 42,
        g0_cap: 64,
        g0_rtol: 1e-6,
        g0_patch_order: "all".into(),
        arm: "patched-contract".into(),
        cap: 64,
        rtol: 1e-4,
        out_patch_order: "sites".into(),
        contract: "svd".into(),
        certified_share: None,
        slices: 16,
        samples: 20000,
        check_dense: false,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    for pair in raw.chunks(2) {
        let [key, value] = pair else {
            bail!("arguments come in --key value pairs");
        };
        match key.as_str() {
            "--r" => args.r = value.parse()?,
            "--temperature" => args.temperature = value.parse()?,
            "--topology" => args.topology = value.clone(),
            "--seed" => args.seed = value.parse()?,
            "--ref-seed" => args.ref_seed = value.parse()?,
            "--g0-cap" => args.g0_cap = value.parse()?,
            "--g0-rtol" => args.g0_rtol = value.parse()?,
            "--g0-patch-order" => args.g0_patch_order = value.clone(),
            "--arm" => args.arm = value.clone(),
            "--cap" => args.cap = value.parse()?,
            "--rtol" => args.rtol = value.parse()?,
            "--out-patch-order" => args.out_patch_order = value.clone(),
            "--contract" => args.contract = value.clone(),
            "--certified-share" => args.certified_share = Some(value.parse()?),
            "--slices" => args.slices = value.parse()?,
            "--samples" => args.samples = value.parse()?,
            "--check-dense" => args.check_dense = value.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(args)
}

fn emit(value: Value) {
    println!("{value}");
}

// ---------------------------------------------------------------------------
// G0 input

/// `1 / (MU + 2 cx + 2 cy + i w)` from the three exact inputs.
fn dyson_value(cx: Complex64, cy: Complex64, w: Complex64) -> Complex64 {
    1.0 / (MU + 2.0 * cx + 2.0 * cy + Complex64::i() * w)
}

fn dyson(inputs: ColMajorArrayRef<'_, Complex64>, output: &mut [Complex64]) -> Result<()> {
    let data = inputs.data();
    for (p, value) in output.iter_mut().enumerate() {
        *value = dyson_value(data[3 * p], data[3 * p + 1], data[3 * p + 2]);
    }
    Ok(())
}

fn dyson_batch(
    batch: TreeElementwiseBatch<'_, Complex64>,
    output: &mut [Complex64],
) -> tensor4all_treeaci::Result<()> {
    for (p, value) in output.iter_mut().enumerate() {
        *value = dyson_value(batch.get(0, p)?, batch.get(1, p)?, batch.get(2, p)?);
    }
    Ok(())
}

fn aci_options(args: &Args) -> TreeAciOptions<usize> {
    TreeAciOptions {
        max_sweeps: 50,
        global_tolerance_margin: 10.0,
        max_working_bytes: 8usize << 30,
        max_frame_bytes: 8usize << 30,
        max_sample_arena_bytes: 8usize << 30,
        message_cache_max_bytes: 8usize << 30,
        rng_seed: args.seed,
        ..Default::default()
    }
}

/// `||G0||_2` over the whole grid, from the per-frequency sums.
fn g0_norm(grid: &Grid) -> f64 {
    (2.0 * frequency_sums(grid, 1).iter().sum::<f64>()).sqrt()
}

fn single_patch(tree: Tree) -> Result<Partition> {
    Ok(PartitionedTreeTN::from_subdomain(
        SubDomainTreeTN::from_treetn(tree)?,
    )?)
}

/// Patched G0 by `patched_elementwise` with the TreeACI kernel (the M6c
/// defaults: L2 acceptance with the exact norm, kernel tolerance factor 3).
fn patched_g0(grid: &Grid, args: &Args, norm: f64) -> Result<Partition> {
    let mut backend = CpuBackend::with_threads(1)?;
    let inputs = grid
        .inputs()?
        .into_iter()
        .map(single_patch)
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Partition> = inputs.iter().collect();
    let options = PatchedInterpolationOptions::new(args.g0_cap)
        .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
        .with_verification(VerificationOptions::new().with_audit(true))
        .with_tolerance(ErrorTolerance {
            rtol: args.g0_rtol,
            atol: 0.0,
        })
        .with_patch_order(match args.g0_patch_order.as_str() {
            "all" => grid.sites.clone(),
            "k" => grid
                .sites
                .iter()
                .enumerate()
                .filter(|(node, _)| node % 3 != 2)
                .map(|(_, site)| site.clone())
                .collect(),
            "n" => grid
                .sites
                .iter()
                .enumerate()
                .filter(|(node, _)| node % 3 == 2)
                .map(|(_, site)| site.clone())
                .collect(),
            // Matsubara bits first, then momentum bits.
            "nk" => {
                let n = grid
                    .sites
                    .iter()
                    .enumerate()
                    .filter(|(node, _)| node % 3 == 2);
                let k = grid
                    .sites
                    .iter()
                    .enumerate()
                    .filter(|(node, _)| node % 3 != 2);
                n.chain(k).map(|(_, site)| site.clone()).collect()
            }
            other => bail!("--g0-patch-order must be all, k, n or nk, not {other}"),
        })
        .with_variables(grid.variables())
        .with_seed(args.seed);
    let kernel = TreeAciInterpolator::new(aci_options(args))?.with_tolerance_factor(3.0)?;
    let residual = TreeTciInterpolator::default();
    let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
    Ok(patched_elementwise::<Complex64, _, _, _, _>(
        &kernel, &residual, &refs, &dyson, &mut ctx, &options,
    )?
    .partition)
}

/// Unpatched G0 by TreeACI with the relative tolerance `g0_rtol`.
fn unpatched_g0(grid: &Grid, args: &Args) -> Result<Tree> {
    let mut backend = CpuBackend::with_threads(1)?;
    let options = TreeAciOptions {
        tolerance: args.g0_rtol,
        scale_tolerance: true,
        max_bond_dim: Some(4000),
        ..aci_options(args)
    };
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut session = Session::new(backend.engine_mut());
    let result = tree_elementwise_batched_with_rng_on_session(
        dyson_batch,
        &grid.inputs()?,
        &options,
        &mut rng,
        &mut session,
    )?;
    Ok(result.tree)
}

// ---------------------------------------------------------------------------
// Fourier operator

/// The momentum Fourier operator on the whole tree, as a `LinearOperator`
/// whose inputs are G0's site indices and whose outputs are `outputs`, with
/// identities on the frequency nodes.
fn momentum_operator(
    grid: &Grid,
    topology: &str,
    outputs: &[DynIndex],
    target: &Tree,
) -> Result<LinearOperator<IdxTensor, usize>> {
    let r = grid.r;
    let bind = |op: LinearOperator<IdxTensor, usize>, nodes: &[usize]| -> Result<_> {
        let mut inputs = HashMap::new();
        let mut outs = HashMap::new();
        for &node in nodes {
            let input = op
                .get_input_mapping(&node)
                .context("operator input mapping")?
                .internal_index
                .clone();
            let output = op
                .get_output_mapping(&node)
                .context("operator output mapping")?
                .internal_index
                .clone();
            inputs.insert(
                node,
                IndexMapping {
                    true_index: grid.sites[node].clone(),
                    internal_index: input,
                },
            );
            outs.insert(
                node,
                IndexMapping {
                    true_index: outputs[node].clone(),
                    internal_index: output,
                },
            );
        }
        Ok(LinearOperator::new(op.mpo().clone(), inputs, outs))
    };
    let operators = match topology {
        "tt_nblock" => {
            let layout: Vec<AxisBit> = (0..r)
                .flat_map(|bit| (0..2).map(move |axis| AxisBit { axis, bit }))
                .collect();
            let renames: Vec<(usize, usize)> = layout
                .iter()
                .enumerate()
                .map(|(j, b)| (j, 3 * b.bit + b.axis))
                .collect();
            let nodes: Vec<usize> = renames.iter().map(|&(_, node)| node).collect();
            let op = quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward())?;
            vec![bind(op.rename_nodes(&renames)?, &nodes)?]
        }
        "cttn" => (0..2)
            .map(|axis| {
                let renames: Vec<(usize, usize)> =
                    (0..r).map(|bit| (bit, 3 * bit + axis)).collect();
                let nodes: Vec<usize> = renames.iter().map(|&(_, node)| node).collect();
                let op = quantics_fourier_operator(r, FourierOptions::forward())?;
                bind(op.rename_nodes(&renames)?, &nodes)
            })
            .collect::<Result<Vec<_>>>()?,
        other => bail!("topology must be tt_nblock or cttn, not {other}"),
    };
    let gaps: HashMap<usize, Vec<(DynIndex, DynIndex)>> = (0..r)
        .map(|bit| {
            let node = 3 * bit + 2;
            (
                node,
                vec![(grid.sites[node].clone(), outputs[node].clone())],
            )
        })
        .collect();
    let refs: Vec<&LinearOperator<IdxTensor, usize>> = operators.iter().collect();
    Ok(compose_exclusive_linear_operators(
        target.site_index_network(),
        &refs,
        &gaps,
    )?)
}

// ---------------------------------------------------------------------------
// Exact reference

/// In-place iterative radix-2 FFT, `sum_k x_k exp(-2 pi i j k / N)`.
fn fft(data: &mut [Complex64]) {
    let n = data.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let w = Complex64::from_polar(1.0, -2.0 * PI / len as f64);
        for start in (0..n).step_by(len) {
            let mut factor = Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let a = data[start + k];
                let b = data[start + k + len / 2] * factor;
                data[start + k] = a + b;
                data[start + k + len / 2] = a - b;
                factor *= w;
            }
        }
        len <<= 1;
    }
}

/// `G(rx, ry, n) = (1/N) sum_k exp(-2 pi i (kx rx + ky ry) / N) G0(kx, ky, n)`,
/// as a row-major `[rx][ry]` array.
fn exact_slice(grid: &Grid, n: usize) -> Vec<Complex64> {
    let size = grid.size();
    let mut data: Vec<Complex64> = (0..size * size)
        .map(|i| grid.g0(i / size, i % size, n))
        .collect();
    for row in data.chunks_mut(size) {
        fft(row);
    }
    let mut column = vec![Complex64::new(0.0, 0.0); size];
    for c in 0..size {
        for (r, value) in column.iter_mut().enumerate() {
            *value = data[r * size + c];
        }
        fft(&mut column);
        for (r, value) in column.iter().enumerate() {
            data[r * size + c] = value / size as f64;
        }
    }
    data
}

/// The site values of output point `(rx, ry, n)` in node order: momentum
/// output bits are bit-reversed, frequency bits are most significant first.
fn output_point(grid: &Grid, rx: usize, ry: usize, n: usize) -> Vec<usize> {
    let r = grid.r;
    let mut point = vec![0; 3 * r];
    for bit in 0..r {
        point[3 * bit] = (rx >> bit) & 1;
        point[3 * bit + 1] = (ry >> bit) & 1;
        point[3 * bit + 2] = (n >> (r - 1 - bit)) & 1;
    }
    point
}

struct Reference {
    points: Vec<usize>,
    exact: Vec<Complex64>,
    slices: Vec<usize>,
}

fn reference(grid: &Grid, args: &Args) -> Reference {
    let size = grid.size();
    let mut rng = ChaCha8Rng::seed_from_u64(args.ref_seed ^ 0x6d61_6667);
    // The two frequencies nearest the Fermi level, then random ones.
    let mut slices = vec![size / 2 - 1, size / 2];
    while slices.len() < args.slices.max(2) {
        let n = rng.random_range(0..size);
        if !slices.contains(&n) {
            slices.push(n);
        }
    }
    let per_slice = args.samples.div_ceil(slices.len());
    let mut points = Vec::new();
    let mut exact = Vec::new();
    for &n in &slices {
        let slice = exact_slice(grid, n);
        for _ in 0..per_slice {
            let (rx, ry) = (rng.random_range(0..size), rng.random_range(0..size));
            points.extend(output_point(grid, rx, ry, n));
            exact.push(slice[rx * size + ry]);
        }
    }
    Reference {
        points,
        exact,
        slices,
    }
}

fn accuracy(values: &[Complex64], exact: &[Complex64]) -> Value {
    let mut diff2 = 0.0;
    let mut norm2 = 0.0;
    let mut max_diff: f64 = 0.0;
    let mut max_exact: f64 = 0.0;
    for (a, e) in values.iter().zip(exact) {
        diff2 += (a - e).norm_sqr();
        norm2 += e.norm_sqr();
        max_diff = max_diff.max((a - e).norm());
        max_exact = max_exact.max(e.norm());
    }
    json!({
        "samples": exact.len(),
        "rel_l2": (diff2 / norm2).sqrt(),
        "rel_max": max_diff / max_exact,
    })
}

fn evaluate(
    partition: &Partition,
    indices: &[DynIndex],
    points: &[usize],
) -> Result<Vec<Complex64>> {
    let shape = [indices.len(), points.len() / indices.len()];
    Ok(partition
        .evaluate_batched(indices, ColMajorArrayRef::new(points, &shape)?)?
        .into_iter()
        .map(|value| Complex64::new(value.real(), value.imag()))
        .collect())
}

fn partition_stats(partition: &Partition) -> Value {
    let params: usize = partition
        .values()
        .map(|patch| {
            let data = patch.data();
            data.node_names()
                .iter()
                .filter_map(|node| data.node_index(node))
                .filter_map(|index| data.tensor(index))
                .map(|tensor| {
                    tensor
                        .indices()
                        .iter()
                        .map(IndexLike::dim)
                        .product::<usize>()
                })
                .sum::<usize>()
        })
        .sum();
    json!({
        "patches": partition.len(),
        "params": params,
        "max_bond": partition.values().map(SubDomainTreeTN::max_bond_dim).max().unwrap_or(0),
    })
}

// ---------------------------------------------------------------------------
// Arms

fn run_arm(
    args: &Args,
    grid: &Grid,
    norm: f64,
    outputs: &[DynIndex],
) -> Result<(Partition, Vec<DynIndex>, Value)> {
    let center = 0usize;
    match args.arm.as_str() {
        "patched-contract" => {
            let start = Instant::now();
            let g0 = patched_g0(grid, args, norm)?;
            let g0_seconds = start.elapsed().as_secs_f64();
            eprintln!("g0: {} in {g0_seconds:.1}s", partition_stats(&g0));
            let template = g0.values().next().context("empty G0")?.data().clone();
            let operator = momentum_operator(grid, &args.topology, outputs, &template)?;
            // The operator on G0's whole tree, with true indices and one
            // dtype: a `PartitionedTreeTN` operand (#1298). Every declared
            // output is a fresh index, so the legs are the declared outputs.
            let (network, _) = operator
                .to_state_network(template.site_index_network())?
                .into_parts();
            emit(json!({"kind": "operator", "link_dims": network.link_dims()}));
            let r = grid.r;
            let patch_order = match args.out_patch_order.as_str() {
                "none" => Vec::new(),
                "sites" => outputs.to_vec(),
                "n-first" => (0..r)
                    .map(|bit| outputs[3 * bit + 2].clone())
                    .chain(
                        (0..3 * r)
                            .filter(|node| node % 3 != 2)
                            .map(|node| outputs[node].clone()),
                    )
                    .collect(),
                other => bail!("--out-patch-order must be none, sites or n-first, not {other}"),
            };
            let patching = PatchingOptions {
                max_bond_dim: Some(args.cap),
                cutoff: args.rtol * args.rtol,
                patch_order,
                ..PatchingOptions::default()
            };
            let contract_options = match args.contract.as_str() {
                "exact" => ContractionOptions::default(),
                "svd" => ContractionOptions::default()
                    .with_svd_policy(SvdTruncationPolicy::new(args.rtol)),
                "src" => ContractionOptions::new(ContractionMethod::Src)
                    .with_src_options(
                        SrcOptions::adaptive(args.rtol, 4 * args.cap).with_final_svd(false),
                    )
                    .with_max_bond_dim(4 * args.cap)
                    .with_svd_policy(SvdTruncationPolicy::new(args.rtol)),
                other => bail!("--contract must be exact, svd or src, not {other}"),
            };
            let operand = single_patch(network)?;
            let start = Instant::now();
            let out = contract_adaptive(&operand, &g0, &center, &contract_options, &patching)?;
            let seconds = start.elapsed().as_secs_f64();
            Ok((
                out,
                outputs.to_vec(),
                json!({"g0_seconds": g0_seconds, "g0": partition_stats(&g0), "seconds": seconds}),
            ))
        }
        "unpatched-src" => {
            let start = Instant::now();
            let g0 = unpatched_g0(grid, args)?;
            let g0_seconds = start.elapsed().as_secs_f64();
            let operator = momentum_operator(grid, &args.topology, outputs, &g0)?;
            let options = ApplyOptions::src()
                .with_src_options(SrcOptions::adaptive(args.rtol, args.cap).with_final_svd(false))
                .with_max_bond_dim(args.cap)
                .with_svd_policy(SvdTruncationPolicy::new(args.rtol));
            let start = Instant::now();
            let out = apply_linear_operator(&operator, &g0, options)?;
            let seconds = start.elapsed().as_secs_f64();
            let g0_stats = json!({"patches": 1, "max_bond": g0.link_dims().into_iter().max()});
            Ok((
                single_patch(out)?,
                outputs.to_vec(),
                json!({"g0_seconds": g0_seconds, "g0": g0_stats, "seconds": seconds}),
            ))
        }
        "patched-merge-refine" => {
            let start = Instant::now();
            let g0 = patched_g0(grid, args, norm)?;
            let g0_seconds = start.elapsed().as_secs_f64();
            eprintln!("g0: {} in {g0_seconds:.1}s", partition_stats(&g0));
            let axes: Vec<Vec<DynIndex>> = grid.variables().into_iter().take(2).collect();
            let mut options = PatchedFourierOptions::default();
            options.tolerance.rtol = args.rtol;
            options.merge_refine.merge_policy = MergePolicy::RankGoal(args.cap.try_into()?);
            if let Some(share) = args.certified_share {
                options.merge_refine.application =
                    tensor4all_partitionedtreetn::reconstruction::OperatorApplication::certified(
                        share,
                    );
            }
            let start = Instant::now();
            let result = patched_fourier_transform(
                &g0,
                &center,
                &axes,
                FourierOptions::forward(),
                &options,
            )?;
            let seconds = start.elapsed().as_secs_f64();
            let report = json!({
                "error_bound": result.report().error_bound,
                "absolute_tolerance": result.report().absolute_tolerance,
            });
            // The transform keeps the selected site indices.
            Ok((
                result.into_partition()?,
                grid.sites.clone(),
                json!({"g0_seconds": g0_seconds, "g0": partition_stats(&g0), "seconds": seconds, "merge_refine": report}),
            ))
        }
        "patched-src-each" => {
            // Diagnostic arm: SRC applied to every patch separately, as the
            // unpatched arm applies it to the whole G0, with no group sum. When
            // the patches fix only Matsubara bits the images are disjoint and
            // form the output partition; otherwise only the costs are reported.
            let start = Instant::now();
            let g0 = patched_g0(grid, args, norm)?;
            let g0_seconds = start.elapsed().as_secs_f64();
            let g0_stats = partition_stats(&g0);
            eprintln!("g0: {g0_stats} in {g0_seconds:.1}s");
            let template = g0.values().next().context("empty G0")?.data().clone();
            let operator = momentum_operator(grid, &args.topology, outputs, &template)?;
            let options = ApplyOptions::src()
                .with_src_options(SrcOptions::adaptive(args.rtol, args.cap).with_final_svd(false))
                .with_max_bond_dim(args.cap)
                .with_svd_policy(SvdTruncationPolicy::new(args.rtol));
            let n_sites: HashMap<DynIndex, DynIndex> = (0..grid.r)
                .map(|bit| {
                    (
                        grid.sites[3 * bit + 2].clone(),
                        outputs[3 * bit + 2].clone(),
                    )
                })
                .collect();
            let mut images = Vec::with_capacity(g0.len());
            let mut raw_images: Vec<Tree> = Vec::with_capacity(g0.len());
            let mut patch_seconds = Vec::with_capacity(g0.len());
            let mut image_bonds = Vec::with_capacity(g0.len());
            let mut input_bonds = Vec::with_capacity(g0.len());
            let mut disjoint = true;
            let start = Instant::now();
            for patch in g0.values() {
                let begin = Instant::now();
                let image = apply_linear_operator(&operator, patch.data(), options.clone())?;
                patch_seconds.push(begin.elapsed().as_secs_f64());
                input_bonds.push(patch.max_bond_dim());
                image_bonds.push(image.link_dims().into_iter().max().unwrap_or(1));
                let mut pairs = Vec::new();
                for (index, &value) in patch.projector().iter() {
                    match n_sites.get(index) {
                        Some(output) => pairs.push((output.clone(), value)),
                        None => disjoint = false,
                    }
                }
                if disjoint {
                    images.push(SubDomainTreeTN::new(
                        image.clone(),
                        Projector::from_pairs(pairs)?,
                    )?);
                }
                raw_images.push(image);
            }
            let seconds = start.elapsed().as_secs_f64();
            let total = |values: &[usize]| values.iter().sum::<usize>();
            let max_of = |values: &[f64]| values.iter().cloned().fold(0.0, f64::max);
            let mut timing = json!({
                "g0_seconds": g0_seconds, "g0": g0_stats, "seconds": seconds,
                "per_patch": {
                    "patches": patch_seconds.len(),
                    "max_seconds": max_of(&patch_seconds),
                    "sum_input_max_bond": total(&input_bonds),
                    "max_image_bond": image_bonds.iter().max(),
                    "sum_image_max_bond": total(&image_bonds),
                    "disjoint_images": disjoint,
                },
            });
            eprintln!("per-patch SRC: {timing}");
            if !disjoint {
                // The images overlap: sum them by a balanced tree of adds,
                // truncating after every add with a relative squared cutoff
                // (rtol / levels)^2, levels = ceil(log2 N).
                let levels = (raw_images.len() as f64).log2().ceil().max(1.0);
                let cutoff = (args.rtol / levels).powi(2);
                let refs: Vec<&Tree> = raw_images.iter().collect();
                let begin = Instant::now();
                let sum = TreeTN::pairwise_sum(
                    &refs,
                    PairwiseSumOrder::Balanced,
                    Some(PerAddCutoff {
                        center: &center,
                        cutoff,
                        max_bond_dim: None,
                    }),
                )?;
                let sum_seconds = begin.elapsed().as_secs_f64();
                eprintln!(
                    "pairwise sum: {sum_seconds:.1}s, bonds max {:?}",
                    sum.link_dims().into_iter().max()
                );
                drop(raw_images);
                if let Some(object) = timing.as_object_mut() {
                    object.insert("sum_seconds".into(), json!(sum_seconds));
                    object.insert("sum_cutoff".into(), json!(cutoff));
                    object.insert(
                        "total_seconds".into(),
                        json!(g0_seconds + seconds + sum_seconds),
                    );
                }
                return Ok((single_patch(sum)?, outputs.to_vec(), timing));
            }
            drop(raw_images);
            let out = PartitionedTreeTN::from_subdomains(images)?;
            let indices = outputs.to_vec();
            Ok((out, indices, timing))
        }
        other => bail!("unknown --arm {other}"),
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let grid = Grid::new(args.r, args.temperature, &args.topology)?;
    emit(json!({
        "kind": "config", "r": args.r, "temperature": args.temperature,
        "topology": args.topology, "seed": args.seed, "ref_seed": args.ref_seed, "g0_cap": args.g0_cap,
        "g0_rtol": args.g0_rtol, "g0_patch_order": args.g0_patch_order, "arm": args.arm, "cap": args.cap, "rtol": args.rtol,
        "out_patch_order": args.out_patch_order, "contract": args.contract, "certified_share": args.certified_share,
        "slices": args.slices, "samples": args.samples, "check_dense": args.check_dense,
    }));
    let norm = g0_norm(&grid);
    let outputs: Vec<DynIndex> = (0..3 * args.r).map(|_| DynIndex::new_dyn(2)).collect();
    let (out, indices, timing) = run_arm(&args, &grid, norm, &outputs)?;

    let size = grid.size();
    let (accuracy_value, slices) = if args.check_dense {
        if 3 * args.r > 18 {
            bail!("--check-dense true needs R <= 6");
        }
        let mut points = Vec::new();
        let mut exact = Vec::new();
        for n in 0..size {
            let slice = exact_slice(&grid, n);
            for rx in 0..size {
                for ry in 0..size {
                    points.extend(output_point(&grid, rx, ry, n));
                    exact.push(slice[rx * size + ry]);
                }
            }
        }
        let values = evaluate(&out, &indices, &points)?;
        (accuracy(&values, &exact), (0..size).collect::<Vec<_>>())
    } else {
        let start = Instant::now();
        let reference = reference(&grid, &args);
        eprintln!(
            "reference: {} slices in {:.1}s",
            reference.slices.len(),
            start.elapsed().as_secs_f64()
        );
        let values = evaluate(&out, &indices, &reference.points)?;
        (accuracy(&values, &reference.exact), reference.slices)
    };
    let mut result = json!({
        "kind": "result",
        "arm": args.arm,
        "topology": args.topology,
        "accuracy": accuracy_value,
        "output": partition_stats(&out),
        "slices": slices.len(),
    });
    if let (Some(object), Some(timing)) = (result.as_object_mut(), timing.as_object()) {
        for (key, value) in timing {
            object.insert(key.clone(), value.clone());
        }
    }
    emit(result);
    Ok(())
}
