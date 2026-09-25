//! Opt-in stagnation diagnostics for chain Hadamard fixtures (issue triage).
//!
//! Run with `T4A_STAG_CASE=<case.json> cargo test -p tensor4all-treeaci
//! --release --lib stagnation_diagnostics -- --ignored --nocapture`.
//! The case format is a list of column-major chain cores per input
//! (`inputs[n][site] = {shape: [left, phys, right], values}`).

use std::collections::HashMap;

use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use crate::{
    hadamard_many,
    schedule::stagnation_trace,
    test_support::{random_tree, SplitMix},
    TreeAciOptions,
};

type Chain = TreeTN<IdxTensor, usize>;
type TestResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn load_chain_case(path: &str) -> TestResult<(Vec<Chain>, Vec<DynIndex>)> {
    let case: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let dims = case["dims"]
        .as_array()
        .ok_or("dims")?
        .iter()
        .map(|d| d.as_u64().map(|d| d as usize).ok_or("dim"))
        .collect::<Result<Vec<_>, _>>()?;
    let n = dims.len();
    let sites = dims
        .iter()
        .map(|&d| DynIndex::new_dyn(d))
        .collect::<Vec<_>>();
    let mut inputs = Vec::new();
    for cores in case["inputs"].as_array().ok_or("inputs")? {
        let cores = cores.as_array().ok_or("cores")?;
        let bonds = (1..n)
            .map(|site| {
                cores[site]["shape"][0]
                    .as_u64()
                    .map(|b| DynIndex::new_dyn(b as usize))
                    .ok_or("bond")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut tensors = Vec::with_capacity(n);
        for (site, core) in cores.iter().enumerate() {
            let values = core["values"]
                .as_array()
                .ok_or("values")?
                .iter()
                .map(|v| v.as_f64().ok_or("value"))
                .collect::<Result<Vec<_>, _>>()?;
            let mut indices = Vec::with_capacity(3);
            if site > 0 {
                indices.push(bonds[site - 1].clone());
            }
            indices.push(sites[site].clone());
            if site + 1 < n {
                indices.push(bonds[site].clone());
            }
            tensors.push(IdxTensor::from_dense(indices, values)?);
        }
        inputs.push(TreeTN::from_tensors(tensors, (0..n).collect())?);
    }
    Ok((inputs, sites))
}

fn dense(tree: &Chain, sites: &[DynIndex]) -> TestResult<Vec<f64>> {
    Ok(tree.to_dense()?.permute_indices(sites)?.to_vec::<f64>()?)
}

struct ErrorSummary {
    max_abs: f64,
    rel_frobenius: f64,
}

fn summarize(approx: &[f64], target: &[f64]) -> ErrorSummary {
    let mut max_abs = 0.0f64;
    let (mut diff2, mut norm2) = (0.0f64, 0.0f64);
    for (a, t) in approx.iter().zip(target) {
        let d = a - t;
        max_abs = max_abs.max(d.abs());
        diff2 += d * d;
        norm2 += t * t;
    }
    ErrorSummary {
        max_abs,
        rel_frobenius: (diff2 / norm2).sqrt(),
    }
}

fn flat_index(
    point: &[usize],
    coordinate_indices: &[Vec<DynIndex>],
    site_strides: &HashMap<DynIndex, usize>,
) -> usize {
    point
        .iter()
        .zip(coordinate_indices)
        .map(|(&value, indices)| {
            assert_eq!(indices.len(), 1, "diagnostics expect one site per node");
            value * site_strides[&indices[0]]
        })
        .sum()
}

#[test]
#[ignore = "opt-in diagnostic; needs T4A_STAG_CASE or T4A_STAG_GEN"]
fn stagnation_trace_chain_hadamard() -> TestResult<()> {
    let (mut inputs, sites, path) = if let Ok(path) = std::env::var("T4A_STAG_CASE") {
        let (inputs, sites) = load_chain_case(&path)?;
        (inputs, sites, path)
    } else if let Ok(spec) = std::env::var("T4A_STAG_GEN") {
        // `sites,dim,bond,inputs,seed`
        let v = spec
            .split(',')
            .map(|x| x.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?;
        let mut rng = SplitMix(v[4] as u64);
        let sites = (0..v[0])
            .map(|_| DynIndex::new_dyn(v[1]))
            .collect::<Vec<_>>();
        let edges = (1..v[0]).map(|node| (node - 1, node)).collect::<Vec<_>>();
        let inputs = (0..v[3])
            .map(|_| random_tree::<f64>(&edges, &sites, v[2], &mut rng))
            .collect::<Vec<_>>();
        (inputs, sites, format!("gen:{spec}"))
    } else {
        eprintln!("T4A_STAG_CASE / T4A_STAG_GEN unset; skipping");
        return Ok(());
    };
    let entries: usize = sites.iter().map(|s| s.dim()).product();
    // Unit-RMS rescaling (historical lane), then an optional homogeneous
    // factor on the first input only.
    if env_or("T4A_STAG_RMS", 1usize) == 1 {
        for input in &mut inputs {
            let norm = input.norm()?;
            let factor = (entries as f64).sqrt() / norm;
            *input = input.scale(AnyScalar::new_real(factor))?;
        }
    }
    let extra: f64 = env_or("T4A_STAG_SCALE", 1.0);
    if extra != 1.0 {
        inputs[0] = inputs[0].scale(AnyScalar::new_real(extra))?;
    }
    let dense_inputs = inputs
        .iter()
        .map(|input| dense(input, &sites))
        .collect::<TestResult<Vec<_>>>()?;
    let target = (0..entries)
        .map(|i| dense_inputs.iter().map(|x| x[i]).product::<f64>())
        .collect::<Vec<_>>();
    let target_max = target.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let target_rms = (target.iter().map(|v| v * v).sum::<f64>() / entries as f64).sqrt();
    let mut site_strides = HashMap::new();
    let mut stride = 1;
    for site in &sites {
        site_strides.insert(site.clone(), stride);
        stride *= site.dim();
    }

    let defaults = TreeAciOptions::<usize>::default();
    let cap: usize = env_or("T4A_STAG_CAP", 0);
    let options = TreeAciOptions::<usize> {
        max_bond_dim: (cap > 0).then_some(cap),
        tolerance: env_or("T4A_STAG_TOL", 1e-14),
        rng_seed: env_or("T4A_STAG_SEED", 1),
        max_sweeps: env_or("T4A_STAG_SWEEPS", 20),
        enable_global_guard: env_or("T4A_STAG_GUARD", 1usize) == 1,
        ..defaults
    };
    println!(
        "case={path} sites={} entries={entries} target_max={target_max:.6e} target_rms={target_rms:.6e} cap={:?} tol={:e} seed={} sweeps={} guard={}",
        sites.len(),
        options.max_bond_dim,
        options.tolerance,
        options.rng_seed,
        options.max_sweeps,
        options.enable_global_guard
    );

    stagnation_trace::enable();
    let started = std::time::Instant::now();
    let result = hadamard_many::<f64, _>(&inputs, &options)?;
    let elapsed = started.elapsed();
    let passes = stagnation_trace::take();

    let snapshots = passes
        .iter()
        .map(|pass| {
            let output = pass
                .output
                .as_ref()
                .and_then(|o| o.downcast_ref::<Chain>())
                .ok_or("snapshot type")?;
            dense(output, &sites)
        })
        .collect::<TestResult<Vec<_>>>()?;
    println!("pass maxrank ranks | localmetric worst_edge stable limited | maxabs_err relfrob_err | guard: scale thr walks>thr found injected ranks_after_max | pivot residual before -> after next pass");
    for (k, pass) in passes.iter().enumerate() {
        let summary = summarize(&snapshots[k], &target);
        let worst_edge = pass
            .edge_errors
            .iter()
            .zip(&pass.edge_scales)
            .map(|(error, scale)| if *scale > 0.0 { error / scale } else { *error })
            .enumerate()
            .fold((0, 0.0f64), |best, (edge, metric)| {
                if metric > best.1 {
                    (edge, metric)
                } else {
                    best
                }
            })
            .0;
        let mut line = format!(
            "{:>4} {:>4} {:?} | {:.3e} {} {} {} | {:.3e} {:.3e}",
            pass.pass,
            pass.edge_ranks.iter().max().copied().unwrap_or(0),
            pass.edge_ranks,
            pass.max_error_metric,
            worst_edge,
            pass.stable_rank_passes,
            pass.rank_limited,
            summary.max_abs,
            summary.rel_frobenius,
        );
        if let Some(search) = &pass.search {
            let above = search
                .walks
                .iter()
                .filter(|(e, _)| *e > search.threshold)
                .count();
            let injection = pass.injection.as_ref();
            line += &format!(
                " | {:.3e} {:.3e} {}/{} {} {} {}",
                search.max_start_output,
                search.threshold,
                above,
                search.walks.len(),
                injection.map_or(0, |i| i.found),
                injection.map_or(0, |i| i.injected),
                injection
                    .map(|i| i.ranks_after.iter().max().copied().unwrap_or(0))
                    .unwrap_or(0),
            );
            let residuals = search
                .pivots
                .iter()
                .map(|point| {
                    let flat = flat_index(point, &search.coordinate_indices, &site_strides);
                    let before = (snapshots[k][flat] - target[flat]).abs();
                    let after = snapshots
                        .get(k + 1)
                        .map(|next| (next[flat] - target[flat]).abs());
                    match after {
                        Some(after) => format!("{before:.2e}->{after:.2e}"),
                        None => format!("{before:.2e}->end"),
                    }
                })
                .collect::<Vec<_>>();
            line += &format!(" | {}", residuals.join(" "));
        }
        println!("{line}");
    }
    let final_dense = dense(&result.tree, &sites)?;
    let summary = summarize(&final_dense, &target);
    println!(
        "final termination={:?} passes={} link_max={} maxabs_err={:.6e} relfrob_err={:.6e} maxabs/target_max={:.3e} evaluated={} elapsed={:.3?}",
        result.termination,
        result.max_ranks.len(),
        result.diagnostics.edge_ranks.iter().map(|e| e.2).max().unwrap_or(0),
        summary.max_abs,
        summary.rel_frobenius,
        summary.max_abs / target_max,
        result.diagnostics.evaluated_points,
        elapsed,
    );
    Ok(())
}
