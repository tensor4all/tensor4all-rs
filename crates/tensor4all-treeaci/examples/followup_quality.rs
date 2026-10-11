//! Reproduction for two TreeACI quality observations (investigation data only).
//!
//! `star <cap> <seed>`: degree-4 star with four 4-node arms (17 nodes, d = 2),
//! two random inputs with bond dimension 8 whose j-th bond component carries
//! weight 0.5^j, `hadamard_many` with `max_bond_dim = cap`, default options.
//! The dense product (2^17 entries) gives the independent error.
//!
//! `chain <case.json> <cap> <tolerance> <out_prefix>`: chain inputs from a JSON
//! file `{"dims": [...], "inputs": [[{"shape": [l, d, r], "values": [...]}]]}`
//! (column-major values), `scale_tolerance = true`. Output cores are written to
//! `<out_prefix>_s<sweeps>.json` for an independent error computed outside.
//!
//! Both modes rerun with three passes and the final pass limit, printing
//! public histories and final per-edge ranks. Each rerun checks that its
//! rank/error history matches the full run's prefix. The three-pass run is
//! skipped when the full run stops sooner; duplicate limits run only once.
//! This diagnostic materializes dense values only for the bounded star case.
//! The chain mode writes cores without materializing its full physical space.
//!
//! Run: cargo run --release -p tensor4all-treeaci --example followup_quality -- star 8 1

use std::env;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde_json::{json, Value};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treeaci::{hadamard_many, TreeAciOptions, TreeAciResult};
use tensor4all_treetn::TreeTN;

type Tree = TreeTN<IdxTensor, usize>;

fn star_edges(delta: usize, arm: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for a in 0..delta {
        let first = 1 + a * arm;
        edges.push((0, first));
        for j in 1..arm {
            edges.push((first + j - 1, first + j));
        }
    }
    edges
}

/// Random tree: entries N(0, 1) * chi^(-deg/4); the j-th component of every
/// bond is weighted by 0.5^j at the edge's first endpoint.
fn random_tree(
    n: usize,
    edges: &[(usize, usize)],
    sites: &[DynIndex],
    chi: usize,
    seed: u64,
) -> Tree {
    let mut rng = StdRng::seed_from_u64(seed);
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(chi))
        .collect::<Vec<_>>();
    let degree = |v: usize| edges.iter().filter(|&&(a, b)| a == v || b == v).count();
    let mut tensors = Vec::with_capacity(n);
    for (node, site) in sites.iter().enumerate().take(n) {
        let mut indices = vec![site.clone()];
        let mut decay_axes = Vec::new();
        for (e, &(a, b)) in edges.iter().enumerate() {
            if a == node || b == node {
                if a == node {
                    decay_axes.push(indices.len());
                }
                indices.push(bonds[e].clone());
            }
        }
        let dims = indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
        let sigma = (chi as f64).powf(-(degree(node) as f64) / 4.0);
        let mut values = (0..dims.iter().product::<usize>())
            .map(|_| sigma * rng.sample::<f64, _>(StandardNormal))
            .collect::<Vec<_>>();
        for &axis in &decay_axes {
            let stride = dims[..axis].iter().product::<usize>();
            for (flat, value) in values.iter_mut().enumerate() {
                *value *= 0.5_f64.powi(((flat / stride) % dims[axis]) as i32);
            }
        }
        tensors.push(IdxTensor::from_dense(indices, values).expect("core"));
    }
    TreeTN::from_tensors(tensors, (0..n).collect()).expect("tree")
}

fn dense(tree: &Tree, sites: &[DynIndex]) -> Vec<f64> {
    tree.to_dense()
        .expect("dense")
        .permute_indices(sites)
        .expect("order")
        .to_vec::<f64>()
        .expect("values")
}

fn summary(result: &TreeAciResult<usize>) -> Value {
    json!({
        "termination": format!("{:?}", result.termination),
        "sweeps": result.max_ranks.len(),
        "max_ranks": result.max_ranks,
        "max_errors": result.max_errors,
        "global_pivots_found": result.global_pivots_found,
        "edge_ranks": result.diagnostics.edge_ranks,
        "link_dims": result.tree.link_dims(),
    })
}

fn sweep_series(
    inputs: &[Tree],
    base: &TreeAciOptions<usize>,
    mut each: impl FnMut(usize, &TreeAciResult<usize>, Value),
) {
    let full = hadamard_many::<f64, _>(inputs, base).expect("treeaci");
    let total = full.max_ranks.len();
    for sweeps in std::collections::BTreeSet::from([3, total])
        .into_iter()
        .filter(|&s| s <= total)
    {
        let options = TreeAciOptions {
            max_sweeps: sweeps,
            min_sweeps: base.min_sweeps.min(sweeps),
            ..base.clone()
        };
        let result = hadamard_many::<f64, _>(inputs, &options).expect("treeaci");
        let prefix = result.max_ranks[..] == full.max_ranks[..result.max_ranks.len()]
            && result.max_errors[..] == full.max_errors[..result.max_errors.len()];
        let mut line = summary(&result);
        line["max_sweeps_option"] = json!(sweeps);
        line["history_is_prefix_of_full_run"] = json!(prefix);
        each(sweeps, &result, line);
    }
    let mut line = summary(&full);
    line["full_run"] = json!(true);
    println!("{line}");
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    match args.get(1).map(String::as_str) {
        Some("star") => {
            let cap: usize = args[2].parse().expect("cap");
            let seed: u64 = args[3].parse().expect("seed");
            let edges = star_edges(4, 4);
            let n = 17;
            let sites = (0..n).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
            let inputs = (0..2u64)
                .map(|i| random_tree(n, &edges, &sites, 8, seed * 1000 + i + 1))
                .collect::<Vec<_>>();
            let (a, b) = (dense(&inputs[0], &sites), dense(&inputs[1], &sites));
            let exact = a.iter().zip(&b).map(|(x, y)| x * y).collect::<Vec<_>>();
            let norm = exact.iter().map(|x| x * x).sum::<f64>().sqrt();
            let options = TreeAciOptions::<usize> { max_bond_dim: Some(cap), rng_seed: seed, ..TreeAciOptions::default() };
            sweep_series(&inputs, &options, |_, result, mut line| {
                let approx = dense(&result.tree, &sites);
                let err = approx.iter().zip(&exact).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt() / norm;
                line["relative_error_dense"] = json!(err);
                println!("{line}");
            });
        }
        Some("chain") => {
            let case: Value = serde_json::from_str(&std::fs::read_to_string(&args[2]).expect("case")).expect("json");
            let cap: usize = args[3].parse().expect("cap");
            let tolerance: f64 = args[4].parse().expect("tolerance");
            let prefix = &args[5];
            let dims = case["dims"].as_array().expect("dims").iter().map(|d| d.as_u64().expect("dim") as usize).collect::<Vec<_>>();
            let n = dims.len();
            let sites = dims.iter().map(|&d| DynIndex::new_dyn(d)).collect::<Vec<_>>();
            let inputs = case["inputs"]
                .as_array()
                .expect("inputs")
                .iter()
                .map(|cores| {
                    let cores = cores.as_array().expect("cores");
                    let bonds = (1..n)
                        .map(|s| DynIndex::new_dyn(cores[s]["shape"][0].as_u64().expect("bond") as usize))
                        .collect::<Vec<_>>();
                    let tensors = cores
                        .iter()
                        .enumerate()
                        .map(|(s, core)| {
                            let values = core["values"].as_array().expect("values").iter().map(|v| v.as_f64().expect("f64")).collect();
                            let mut indices = Vec::with_capacity(3);
                            if s > 0 {
                                indices.push(bonds[s - 1].clone());
                            }
                            indices.push(sites[s].clone());
                            if s + 1 < n {
                                indices.push(bonds[s].clone());
                            }
                            IdxTensor::from_dense(indices, values).expect("core")
                        })
                        .collect::<Vec<_>>();
                    TreeTN::from_tensors(tensors, (0..n).collect()).expect("chain")
                })
                .collect::<Vec<_>>();
            let options = TreeAciOptions::<usize> {
                max_bond_dim: Some(cap),
                rng_seed: 1,
                tolerance,
                scale_tolerance: true,
                ..TreeAciOptions::default()
            };
            sweep_series(&inputs, &options, |sweeps, result, mut line| {
                // Chain output cores in site order, column-major (left bond, site, right bond).
                let tree = &result.tree;
                let link = |a: usize, b: usize| {
                    tree.bond_index(tree.edge_between(&a, &b).expect("edge")).expect("bond").clone()
                };
                let mut cores = Vec::new();
                for s in 0..n {
                    let tensor = tree.tensor(tree.node_index(&s).expect("node")).expect("tensor");
                    let mut order = Vec::new();
                    if s > 0 {
                        order.push(link(s - 1, s));
                    }
                    order.push(sites[s].clone());
                    if s + 1 < n {
                        order.push(link(s, s + 1));
                    }
                    let left = if s > 0 { order[0].dim() } else { 1 };
                    let right = if s + 1 < n { order[order.len() - 1].dim() } else { 1 };
                    let values = tensor.permute_indices(&order).expect("order").to_vec::<f64>().expect("values");
                    cores.push(json!({"shape": [left, dims[s], right], "values": values}));
                }
                let path = format!("{prefix}_s{sweeps}.json");
                std::fs::write(&path, json!({"cores": cores}).to_string()).expect("write");
                line["output"] = json!(path);
                println!("{line}");
            });
        }
        _ => panic!("usage: quality_repro star <cap> <seed> | chain <case.json> <cap> <tolerance> <out_prefix>"),
    }
}
