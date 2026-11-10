//! Reproducer for tensor4all-rs#1186 part 2: what the return visit of the
//! two-site Euler tour in `TreeTN::truncate` does, and what it costs.
//!
//! Public API only, so the same file builds against a checkout with and without
//! the change it measures. It lives next to the operator it applies: the state
//! is the one the issue is about, a comb TreeTN after a per-branch zip-up apply
//! of the transposed quantics Fourier transform.
//!
//! ```text
//! ex=repro_1186_truncate_two_visit
//! # per-edge classification of one truncation (both visits of every edge)
//! cargo run --release -p tensor4all-quanticstransform --example $ex -- diag B1 1e-9
//! # end-to-end: build, truncate, sample 8192 amplitudes, write them
//! cargo run --release -p tensor4all-quanticstransform --example $ex -- \
//!     run B1 1e-9 base /tmp/s-base-B1.bin
//! ```
//!
//! States (`kind`):
//! - `A` - three branches of 20 binary sites, bond `min(2^(i+1), 128)`, per-leg
//!   weight `exp(-0.1 * bond index)`, canonicalized at the junction (state A of
//!   the issue thread);
//! - `B1` - `A` with the transposed 20-site quantics Fourier operator applied to
//!   branch 0 by `ApplyOptions::zipup()` at rtol 1e-8 (state B1);
//! - `B2` - `B1` truncated once at rtol 1e-9 (state B2);
//! - `IFT:chi:decay[:seed]` - the reporter's state: all three branches applied.
//!
//! `diag` prints, per two-site step, the edge, which visit of the edge it is
//! (1 = outbound, 2 = return), the bond before and after, how many singular
//! values the cutoff dropped and the SVD time; the `RETURN` and `SKIP` lines
//! classify the return visits. `run` prints the truncation wall time, the bond
//! dimensions before and after, and the fingerprint.
//!
//! Compare two binaries by running `run` with the same arguments and taking the
//! relative L2 distance between the amplitude files: the tree inner product
//! cannot resolve differences below about 1e-8, because
//! `||a||^2 + ||b||^2 - 2<a,b>` cancels there.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, Result};
use num_complex::Complex64;
use tensor4all_core::{
    Canonical, ColMajorArrayRef, DynIndex, FactorizeOptions, IdxTensor, IndexLike,
    SvdTruncationPolicy, TensorContractionLike, TensorIndex,
};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CachedEvaluatorOptions,
    CanonicalizationOptions, EvaluationHint, LinearOperator, LocalUpdateStep, LocalUpdateSweepPlan,
    LocalUpdater, TreeTN, TreeTNCachedEvaluator, TruncationOptions,
};

type Tn = TreeTN<IdxTensor, usize>;

const R: usize = 20;
const D: usize = 3;
const N_SAMPLES: usize = 8192;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fnv(hash: &mut u64, word: u64) {
    for byte in word.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01B3);
    }
}

fn fingerprint(tn: &Tn, n_nodes: usize) -> Result<u64> {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for v in 0..n_nodes {
        let node = tn.node_index(&v).ok_or_else(|| anyhow!("node {v}"))?;
        let t = tn.tensor(node).ok_or_else(|| anyhow!("tensor {v}"))?;
        for &dim in &t.dims() {
            fnv(&mut hash, dim as u64);
        }
        for z in t.to_vec::<Complex64>()? {
            fnv(&mut hash, z.re.to_bits());
            fnv(&mut hash, z.im.to_bits());
        }
    }
    Ok(hash)
}

fn bond_stats(tn: &Tn) -> (usize, usize) {
    let dims = tn.link_dims();
    (dims.iter().copied().max().unwrap_or(0), dims.iter().sum())
}

/// The random complex comb of `simonb282/t4a-perf-repro` (`ift_speed/src/main.rs`).
fn synthetic_comb(r: usize, chis: &[usize], decay: f64, seed: u64) -> Result<Tn> {
    let d = chis.len();
    let mut state = seed;
    let mut random = move || {
        let x = splitmix(&mut state);
        (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let bond_dim = |i: usize, j: usize| (1usize << (i + 1).min(40)).min(chis[j]);
    let bonds: Vec<Vec<DynIndex>> = (0..d)
        .map(|j| (0..r).map(|i| DynIndex::new_dyn(bond_dim(i, j))).collect())
        .collect();
    let mut tensors = Vec::with_capacity(r * d + 1);
    for i in 0..r {
        for branch in &bonds {
            let mut indices = vec![DynIndex::new_dyn(2)];
            if i > 0 {
                indices.push(branch[i - 1].clone());
            }
            indices.push(branch[i].clone());
            let len: usize = indices.iter().map(|x| x.dim()).product();
            let below = len / branch[i].dim();
            let data: Vec<Complex64> = (0..len)
                .map(|k| {
                    let z = Complex64::new(random(), random());
                    if decay > 0.0 {
                        z * (-decay * (k / below) as f64).exp()
                    } else {
                        z
                    }
                })
                .collect();
            tensors.push(IdxTensor::from_dense(indices, data)?);
        }
    }
    let mut indices = vec![DynIndex::new_dyn(1)];
    indices.extend(bonds.iter().map(|branch| branch[r - 1].clone()));
    let len: usize = indices.iter().map(|x| x.dim()).product();
    let data: Vec<Complex64> = (0..len)
        .map(|_| Complex64::new(random(), random()))
        .collect();
    tensors.push(IdxTensor::from_dense(indices, data)?);
    Ok(TreeTN::from_tensors(tensors, (0..=r * d).collect())?)
}

fn fourier_op(
    r: usize,
    d: usize,
    branch: usize,
    tn: &Tn,
) -> Result<LinearOperator<IdxTensor, usize>> {
    let op = quantics_fourier_operator(
        r,
        FourierOptions {
            sign: 1.0,
            tolerance: 1e-15,
            max_bond_dim: None,
            ..FourierOptions::forward()
        },
    )?
    .transpose();
    let mapping: Vec<(usize, usize)> = (0..r).map(|b| (b, (r - 1 - b) * d + branch)).collect();
    let mut on_branch = op.rename_nodes(&mapping)?;
    on_branch.align_to_state(tn)?;
    Ok(on_branch)
}

/// Per-branch zip-up apply on `branch` (rtol 1e-8), the way the reproducer does it.
fn apply_branch(tn: &Tn, branch: usize, rtol: f64) -> Result<Tn> {
    let junction = R * D;
    let mut tn = tn
        .clone()
        .canonicalize([branch], CanonicalizationOptions::forced())?;
    let on_branch = fourier_op(R, D, branch, &tn)?;
    let options = ApplyOptions {
        center: Some(junction),
        ..ApplyOptions::zipup().with_svd_policy(SvdTruncationPolicy::new(rtol))
    };
    tn = apply_linear_operator(&on_branch, &tn, options)?;
    Ok(tn)
}

fn build(kind: &str) -> Result<Tn> {
    let parts: Vec<&str> = kind.split(':').collect();
    let tn = match parts[0] {
        // state A of the thread: canonical random comb, chi 128, decay 0.1
        "A" => synthetic_comb(R, &[128, 128, 128], 0.1, 12345)?
            .canonicalize([R * D], CanonicalizationOptions::default())?,
        // state B1: A + one-branch zip-up apply (rtol 1e-8). `B2` is B1
        // truncated once, which the subcommands apply after the build.
        "B1" | "B2" => {
            let a = synthetic_comb(R, &[128, 128, 128], 0.1, 12345)?
                .canonicalize([R * D], CanonicalizationOptions::default())?;
            apply_branch(&a, 0, 1e-8)?
        }
        // one branch applied only, configurable chi/decay (lighter variants)
        "Aone" => {
            let chi: usize = parts[1].parse()?;
            let decay: f64 = parts[2].parse()?;
            let a = synthetic_comb(R, &[chi; D], decay, 12345)?
                .canonicalize([R * D], CanonicalizationOptions::default())?;
            apply_branch(&a, 0, 1e-8)?
        }
        // the reporter's state: comb, then all three branches zip-up applied (rtol 1e-8)
        "IFT" => {
            let chi: usize = parts[1].parse()?;
            let decay: f64 = parts[2].parse()?;
            let seed: u64 = parts.get(3).map_or(Ok(12345), |s| s.parse())?;
            let mut tn = synthetic_comb(R, &[chi; D], decay, seed)?;
            tn = tn.canonicalize([R * D], CanonicalizationOptions::default())?;
            for branch in 0..D {
                tn = apply_branch(&tn, branch, 1e-8)?;
            }
            tn
        }
        other => return Err(anyhow!("unknown build kind {other}")),
    };
    let (max, sum) = bond_stats(&tn);
    if std::env::var_os("ECHO_BUILD").is_some() {
        println!("BUILD {kind} bonds max {max} sum {sum}");
    }
    Ok(tn)
}

/// The one compression `B2` applies to `B1` in the thread.
fn truncate_once(tn: Tn) -> Result<Tn> {
    Ok(tn.truncate(
        [R * D],
        TruncationOptions::default().with_svd_policy(SvdTruncationPolicy::new(1e-9)),
    )?)
}

/// Deterministic sample points and the site indices, in node order.
fn sample_columns(tn: &Tn) -> Result<(Vec<DynIndex>, Vec<usize>)> {
    let junction = R * D;
    let mut sites = Vec::new();
    for v in 0..=junction {
        let space = tn.site_space(&v).ok_or_else(|| anyhow!("site space {v}"))?;
        if space.len() != 1 {
            return Err(anyhow!("node {v} has {} site indices", space.len()));
        }
        sites.push(space.iter().next().cloned().unwrap());
    }
    let mut state = 2024u64;
    let mut columns = Vec::with_capacity(sites.len() * N_SAMPLES);
    for _ in 0..N_SAMPLES {
        for (k, site) in sites.iter().enumerate() {
            let value = if k == junction {
                0
            } else {
                (splitmix(&mut state) & 1) as usize
            };
            columns.push(value.min(site.dim() - 1));
        }
    }
    Ok((sites, columns))
}

fn amplitudes(tn: &Tn) -> Result<Vec<Complex64>> {
    let (sites, columns) = sample_columns(tn)?;
    let mut ev = TreeTNCachedEvaluator::new(
        tn,
        &sites,
        CachedEvaluatorOptions {
            center: Some(R * D),
            ..CachedEvaluatorOptions::default()
        },
    )?;
    let shape = [sites.len(), N_SAMPLES];
    let batch = ColMajorArrayRef::new(&columns, &shape)?;
    Ok(ev.evaluate_batched_typed::<Complex64>(batch, EvaluationHint::default())?)
}

fn write_samples(values: &[Complex64], path: &str) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 16);
    for z in values {
        bytes.extend_from_slice(&z.re.to_le_bytes());
        bytes.extend_from_slice(&z.im.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn run(kind: &str, cutoff: f64, label: &str, samples_out: &str) -> Result<()> {
    let tn = build(kind)?;
    let tn = if kind.starts_with("B2") {
        truncate_once(tn)?
    } else {
        tn
    };
    let (max0, sum0) = bond_stats(&tn);
    let start = Instant::now();
    let tn = tn.truncate(
        [R * D],
        TruncationOptions::default().with_svd_policy(SvdTruncationPolicy::new(cutoff)),
    )?;
    let wall = start.elapsed().as_secs_f64();
    let (max1, sum1) = bond_stats(&tn);
    let hash = fingerprint(&tn, R * D + 1)?;
    let values = amplitudes(&tn)?;
    write_samples(&values, samples_out)?;
    println!(
        "RESULT {label} cutoff {cutoff:e} wall {wall:.3}s bonds {max0}/{sum0} -> {max1}/{sum1} \
         units {} fingerprint {hash:016x} samples {samples_out}",
        sum0 as i64 - sum1 as i64
    );
    Ok(())
}

/// Every bond dimension, ordered by edge, before and after one truncation.
fn dims(kind: &str, cutoff: f64) -> Result<()> {
    let tn = build(kind)?;
    let tn = if kind.starts_with("B2") {
        truncate_once(tn)?
    } else {
        tn
    };
    let before: Vec<usize> = edge_dims(&tn);
    let tn = tn.truncate(
        [R * D],
        TruncationOptions::default().with_svd_policy(SvdTruncationPolicy::new(cutoff)),
    )?;
    let after: Vec<usize> = edge_dims(&tn);
    println!("DIMS before {before:?}");
    println!("DIMS after {after:?}");
    Ok(())
}

/// Bond dimensions in edge order, so two binaries can be compared edge by edge.
fn edge_dims(tn: &Tn) -> Vec<usize> {
    let mut dims = Vec::new();
    for a in 0..=R * D {
        for b in (a + 1)..=R * D {
            if let Some(bond) = tn.edge_between(&a, &b).and_then(|edge| tn.bond_index(edge)) {
                dims.push(bond.dim());
            }
        }
    }
    dims
}

fn samples(kind: &str, out: &str) -> Result<()> {
    let tn = build(kind)?;
    let tn = if kind.starts_with("B2") {
        truncate_once(tn)?
    } else {
        tn
    };
    let values = amplitudes(&tn)?;
    write_samples(&values, out)?;
    let (max, sum) = bond_stats(&tn);
    println!("RESULT samples bonds {max}/{sum} -> {out}");
    Ok(())
}

// ---------------------------------------------------------------------------
// diag: replay the truncation sweep with a recording updater
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct StepRecord {
    edge: (usize, usize),
    visit: usize,
    bond_before: usize,
    left_dim: usize,
    right_dim: usize,
    rank: usize,
    kept_rank: usize,
    /// Singular values the threshold dropped at this visit.
    dropped: usize,
    sigma_max: f64,
    sigma_min: f64,
    kept_min: f64,
    discarded_rel: f64,
    seconds: f64,
}

#[derive(Debug, Clone)]
struct DiagUpdater {
    cutoff: f64,
    visits: HashMap<(usize, usize), usize>,
    /// Outbound visit kept every singular value, per edge.
    outbound_kept_all: HashMap<(usize, usize), bool>,
    /// Rank the outbound visit removed on the edge (`bond_before - rank`).
    outbound_removed: HashMap<(usize, usize), usize>,
    records: Vec<StepRecord>,
}

impl DiagUpdater {
    fn new(cutoff: f64) -> Self {
        Self {
            cutoff,
            visits: HashMap::new(),
            outbound_kept_all: HashMap::new(),
            outbound_removed: HashMap::new(),
            records: Vec::new(),
        }
    }
}

fn factorize_pair(
    tensor_ab: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<tensor4all_core::FactorizeResult<IdxTensor>> {
    tensor4all_core::factorize(tensor_ab, left_inds, options).map_err(|e| anyhow!("factorize: {e}"))
}

impl LocalUpdater<IdxTensor, usize> for DiagUpdater {
    fn update(
        &mut self,
        mut subtree: Tn,
        step: &LocalUpdateStep<usize>,
        _full: &Tn,
    ) -> std::result::Result<Tn, tensor4all_treetn::TreeTNOperationError> {
        let node_a = step.nodes[0];
        let node_b = step.nodes[1];
        let idx_a = subtree
            .node_index(&node_a)
            .ok_or_else(|| anyhow!("node a"))?;
        let idx_b = subtree
            .node_index(&node_b)
            .ok_or_else(|| anyhow!("node b"))?;
        let edge = subtree
            .edge_between(&node_a, &node_b)
            .ok_or_else(|| anyhow!("edge"))?;
        let bond = subtree
            .bond_index(edge)
            .ok_or_else(|| anyhow!("bond"))?
            .clone();
        let tensor_a = subtree.tensor(idx_a).ok_or_else(|| anyhow!("ta"))?.clone();
        let tensor_b = subtree.tensor(idx_b).ok_or_else(|| anyhow!("tb"))?.clone();
        let bond_before = bond.dim();
        let tensor_ab = tensor_a
            .contract_pair(&tensor_b)
            .map_err(|e| anyhow!("{e}"))?;
        let left_inds: Vec<_> = tensor_a
            .external_indices()
            .iter()
            .filter(|i| **i != bond)
            .cloned()
            .collect();
        let left_dim: usize = left_inds.iter().map(|i| i.dim()).product();
        let total: usize = tensor_ab.dims().iter().product();
        let right_dim = total / left_dim;

        // Full spectrum (no truncation) for the diagnostic.
        let full = factorize_pair(
            &tensor_ab,
            &left_inds,
            &FactorizeOptions::svd()
                .with_canonical(Canonical::Left)
                .with_svd_policy(SvdTruncationPolicy::new(0.0)),
        )
        .map_err(|e| anyhow!("full svd: {e}"))?;
        let spectrum = full.singular_values.clone().unwrap_or_default();
        let sigma_max = spectrum.first().copied().unwrap_or(f64::NAN);
        let sigma_min = spectrum.last().copied().unwrap_or(f64::NAN);
        let threshold = self.cutoff * sigma_max;
        let kept_rank = spectrum.iter().filter(|s| **s > threshold).count();
        let kept_min = if kept_rank == 0 {
            f64::NAN
        } else {
            spectrum[kept_rank - 1]
        };
        // Sum the discarded squares directly: `1 - kept/all` cancels the tail.
        let discarded_sq: f64 = spectrum.iter().skip(kept_rank).map(|s| s * s).sum();
        let all_sq: f64 = spectrum.iter().map(|s| s * s).sum();
        let full_rank = left_dim.min(right_dim);
        let dropped = full_rank - kept_rank;
        let discarded_rel = if all_sq > 0.0 {
            (discarded_sq / all_sq).sqrt()
        } else {
            0.0
        };

        // The actual truncating factorization, exactly as TruncateUpdater does it.
        let start = Instant::now();
        let options = FactorizeOptions::svd()
            .with_canonical(Canonical::Left)
            .with_svd_policy(SvdTruncationPolicy::new(self.cutoff));
        let result = factorize_pair(&tensor_ab, &left_inds, &options)
            .map_err(|e| anyhow!("truncated svd: {e}"))?;
        let seconds = start.elapsed().as_secs_f64();
        let rank = result.rank;
        let new_bond = result.bond_index.clone();
        subtree.replace_edge_bond(edge, new_bond.clone())?;
        subtree.replace_tensor(idx_a, result.left)?;
        subtree.replace_tensor(idx_b, result.right)?;
        subtree.set_ortho_towards(&new_bond, Some(step.new_center));
        subtree.set_canonical_region([step.new_center])?;

        let key = if node_a < node_b {
            (node_a, node_b)
        } else {
            (node_b, node_a)
        };
        let visit = self.visits.entry(key).or_insert(0);
        *visit += 1;
        let visit = *visit;
        if visit == 1 {
            self.outbound_kept_all.insert(key, dropped == 0);
            self.outbound_removed.insert(key, bond_before - rank);
        }
        self.records.push(StepRecord {
            edge: key,
            visit,
            bond_before,
            left_dim,
            right_dim,
            rank,
            kept_rank,
            dropped,
            sigma_max,
            sigma_min,
            kept_min,
            discarded_rel,
            seconds,
        });
        Ok(subtree)
    }
}

fn diag(kind: &str, cutoff: f64) -> Result<()> {
    let tn = build(kind)?;
    let tn = if kind.starts_with("B2") {
        truncate_once(tn)?
    } else {
        tn
    };
    let (max0, sum0) = bond_stats(&tn);
    let junction = R * D;
    let start = Instant::now();
    let mut tn = tn.canonicalize(
        [junction],
        CanonicalizationOptions::default().with_form(tensor4all_treetn::CanonicalForm::Unitary),
    )?;
    let canon = start.elapsed().as_secs_f64();
    let plan =
        LocalUpdateSweepPlan::from_treetn(&tn, &junction, 2).ok_or_else(|| anyhow!("plan"))?;
    let mut updater = DiagUpdater::new(cutoff);
    let start = Instant::now();
    apply_local_update_sweep(&mut tn, &plan, &mut updater)?;
    let sweep = start.elapsed().as_secs_f64();
    let (max1, sum1) = bond_stats(&tn);

    println!(
        "DIAG cutoff {cutoff:e} canon {canon:.3}s sweep {sweep:.3}s bonds {max0}/{sum0} -> {max1}/{sum1}"
    );
    println!("step edge visit bond_before leftxright rank kept dropped sigma_min/sigma_max kept_min/sigma_max discarded_rel seconds");
    for (i, r) in updater.records.iter().enumerate() {
        println!(
            "{i} {}-{} {} {} {}x{} {} {} {} {:.3e} {:.3e} {:.3e} {:.4}",
            r.edge.0,
            r.edge.1,
            r.visit,
            r.bond_before,
            r.left_dim,
            r.right_dim,
            r.rank,
            r.kept_rank,
            r.dropped,
            r.sigma_min / r.sigma_max,
            r.kept_min / r.sigma_max,
            r.discarded_rel,
            r.seconds
        );
    }
    // Summary, classified per edge visit.
    for visit in [1usize, 2] {
        let rows: Vec<&StepRecord> = updater
            .records
            .iter()
            .filter(|r| r.visit == visit)
            .collect();
        let units: i64 = rows
            .iter()
            .map(|r| r.bond_before as i64 - r.rank as i64)
            .sum();
        let discarding = rows.iter().filter(|r| r.discarded_rel > 0.0).count();
        let max_disc = rows.iter().map(|r| r.discarded_rel).fold(0.0f64, f64::max);
        let secs: f64 = rows.iter().map(|r| r.seconds).sum();
        println!(
            "SUMMARY visit {visit}: steps {} units_removed {units} steps_discarding {discarding} \
             max_discarded_rel {max_disc:.3e} svd_seconds {secs:.3}s",
            rows.len()
        );
    }
    // Which return visits discard, and what their outbound visit did.
    let returns: Vec<&StepRecord> = updater.records.iter().filter(|r| r.visit == 2).collect();
    let return_seconds: f64 = returns.iter().map(|r| r.seconds).sum();
    let all_seconds: f64 = updater.records.iter().map(|r| r.seconds).sum();
    let mut skip_count = 0usize;
    let mut skip_seconds = 0.0f64;
    let mut skip_removes_rank = 0usize;
    for r in &returns {
        let removed_out = updater
            .outbound_removed
            .get(&r.edge)
            .copied()
            .unwrap_or(usize::MAX);
        // The candidate rule: the outbound visit on this edge removed no rank.
        let skip = removed_out == 0;
        if skip {
            skip_count += 1;
            skip_seconds += r.seconds;
            if r.rank < r.bond_before {
                skip_removes_rank += 1;
            }
        }
        println!(
            "RETURN edge {}-{} bond {}->{} removed_out {removed_out} removed_return {} \
             sigma_min/max {:.3e} skipped_if_rule {skip} seconds {:.4}",
            r.edge.0,
            r.edge.1,
            r.bond_before,
            r.rank,
            r.bond_before - r.rank,
            r.sigma_min / r.sigma_max,
            r.seconds
        );
    }
    println!(
        "SKIP rule = the outbound visit removed no rank on that edge: {skip_count}/{} return steps, \
         {:.1}% of the return SVD time, {:.1}% of the sweep SVD time; \
         {skip_removes_rank} of those return visits would still remove rank. \
         return SVD {return_seconds:.3}s of {all_seconds:.3}s",
        returns.len(),
        100.0 * skip_seconds / return_seconds.max(1e-12),
        100.0 * skip_seconds / all_seconds.max(1e-12),
    );
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize| -> Result<&str> {
        a.get(i)
            .map(String::as_str)
            .ok_or_else(|| anyhow!("missing arg {i}"))
    };
    match arg(1)? {
        "build" => build(arg(2)?).map(|_| ()),
        "run" => run(arg(2)?, arg(3)?.parse()?, arg(4)?, arg(5)?),
        "samples" => samples(arg(2)?, arg(3)?),
        "dims" => dims(arg(2)?, arg(3)?.parse()?),
        "diag" => diag(arg(2)?, arg(3)?.parse()?),
        other => Err(anyhow!("unknown subcommand {other}")),
    }
}
