//! Input preparation only: current native TreeTCI, excluded from product timing.
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};
use tensor4all_core::IndexLike;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciGraph, TreeTciOptions,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let config: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let n = config["n"].as_u64().unwrap() as usize;
    let cap = config["cap"].as_u64().unwrap() as usize;
    let name = config["function"].as_str().unwrap();
    let mu = config["mu"].as_f64().unwrap_or(0.5);
    let sigma = config["sigma"].as_f64().unwrap_or(0.15);
    let eval = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
        (0..batch.n_points())
            .map(|p| {
                let mut x = 0.0;
                for site in 0..n {
                    x += batch
                        .get(site, p)
                        .ok_or_else(|| anyhow::anyhow!("index out of bounds"))?
                        as f64
                        * 2.0f64.powi(-(site as i32) - 1);
                }
                Ok(match name {
                    "gaussian" => (-(x - mu).powi(2) / (2.0 * sigma * sigma)).exp(),
                    "osc1" => {
                        (1024.0 * x).cos() * (-x * x).exp() + 4.0 * x.exp() - 3.0 * x * x + 10.0 * x
                    }
                    "osc2" => (1024.0 * x).sin() * ((x * x).exp() + 5.0 * x + 2.0) - 4.0 * x,
                    "relu_input" => {
                        (10.0 * x).cos() * (32.0 * x).sin() * ((x * x).exp() + 5.0 * x + 2.0)
                            - 4.0 * x
                    }
                    _ => return Err(anyhow::anyhow!("unknown function")),
                })
            })
            .collect()
    };
    let initial_points: Vec<f64> = match config["pivots"].as_array() {
        Some(points) => points
            .iter()
            .map(|x| x.as_f64().ok_or("invalid pivot"))
            .collect::<Result<_, _>>()?,
        None => vec![mu],
    };
    let pivots = initial_points
        .iter()
        .map(|x| {
            let k = (x * (1u64 << n) as f64).floor() as u64;
            (0..n).map(|i| ((k >> (n - i - 1)) & 1) as usize).collect()
        })
        .collect();
    let (tree, ranks, errors) = crossinterpolate2::<f64, _, _>(
        eval,
        vec![2; n],
        TreeTciGraph::linear_chain(n)?,
        pivots,
        TreeTciOptions {
            max_bond_dim: Some(cap),
            tolerance: 1e-14,
            max_iter: 30,
            seed: Some(20261001),
            ..Default::default()
        },
        Some(n - 1),
        &DefaultProposer,
    )?;
    let output = Path::new(&args[2]);
    let mut data = BufWriter::new(fs::File::create(output)?);
    let mut cores = Vec::new();
    let mut offset = 0;
    for v in 0..n {
        let neighbors: Vec<_> = (0..n).filter(|&w| v.abs_diff(w) == 1).collect();
        let mut inds = Vec::new();
        for w in &neighbors {
            inds.push(
                tree.bond_index(tree.edge_between(&v, w).ok_or("edge")?)
                    .ok_or("bond")?
                    .clone(),
            );
        }
        inds.extend(tree.site_space(&v).ok_or("physical")?.iter().cloned());
        let shape: Vec<_> = inds.iter().map(|x| x.dim()).collect();
        let values = tree
            .tensor(tree.node_index(&v).ok_or("node")?)
            .ok_or("core")?
            .permute_indices(&inds)?
            .to_vec::<f64>()?;
        cores.push(json!({"neighbors":neighbors,"shape":shape,"offset":offset,"len":values.len()}));
        offset += values.len();
        for x in values {
            data.write_all(&x.to_le_bytes())?;
            data.write_all(&0.0f64.to_le_bytes())?;
        }
    }
    data.flush()?;
    fs::write(
        output.with_extension("cores.json"),
        serde_json::to_vec(&cores)?,
    )?;
    println!(
        "{}",
        json!({"ranks":ranks,"local_errors":errors,"config":config})
    );
    Ok(())
}
