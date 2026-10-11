//! Agreement worker: runs tree RSI or TreeACI once on a fixture and writes the
//! output cores. The Python driver owns fixtures, references and comparisons.
//!
//! Usage: `agreement-worker <fixture-dir> <rsi|aci> <request.json> <output-prefix>`
//!
//! Fixture: `fixture.json` lists `n`, `physical_dims`, and per operand per node
//! `neighbors`, `shape` (neighbor bonds, then the site), `offset`, `len` into
//! `inputs.bin` (little-endian f64, column-major cores).
//! Request: `cap`, `seed`, `root`, and algorithm options; RSI may name a
//! `probes` file of little-endian f64 column-major `d x k` blocks listed in
//! `probe_layout` as `[input, node, d, k, offset]`.
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    io::{BufWriter, Write},
    time::Instant,
};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_tensorbackend::Matrix;
use tensor4all_treetn::TreeTN;

type Tree = TreeTN<IdxTensor, usize>;
type Error = Box<dyn std::error::Error>;

fn read_f64(path: &std::path::Path) -> Result<Vec<f64>, Error> {
    Ok(fs::read(path)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect())
}

fn usize_of(v: &Value, what: &str) -> Result<usize, Error> {
    Ok(v.as_u64()
        .ok_or_else(|| format!("{what} is not an integer"))? as usize)
}

fn edge_ranks(tree: &Tree) -> Result<Vec<(usize, usize, usize)>, Error> {
    let mut out = Vec::new();
    for (a, b) in tree.site_index_network().edges() {
        let edge = tree.edge_between(&a, &b).ok_or("missing edge")?;
        out.push((
            a.min(b),
            a.max(b),
            tree.bond_index(edge).ok_or("missing bond")?.dim(),
        ));
    }
    out.sort();
    Ok(out)
}

/// Builds one TreeTN per operand of a fixture-format manifest on `sites`.
fn load_trees(manifest: &Value, data: &[f64], sites: &[DynIndex]) -> Result<Vec<Tree>, Error> {
    let n = usize_of(&manifest["n"], "n")?;
    let mut trees = Vec::new();
    for operand in manifest["operands"].as_array().ok_or("operands")? {
        let cores = operand.as_array().ok_or("cores")?;
        let mut bonds = HashMap::new();
        for (v, core) in cores.iter().enumerate() {
            for (axis, w) in core["neighbors"]
                .as_array()
                .ok_or("neighbors")?
                .iter()
                .enumerate()
            {
                let w = usize_of(w, "neighbor")?;
                let dim = usize_of(&core["shape"][axis], "bond dim")?;
                bonds
                    .entry((v.min(w), v.max(w)))
                    .or_insert_with(|| DynIndex::new_dyn(dim));
            }
        }
        let mut tensors = Vec::new();
        for (v, core) in cores.iter().enumerate() {
            let mut indices = Vec::new();
            for w in core["neighbors"].as_array().ok_or("neighbors")? {
                let w = usize_of(w, "neighbor")?;
                indices.push(bonds[&(v.min(w), v.max(w))].clone());
            }
            indices.push(sites[v].clone());
            let offset = usize_of(&core["offset"], "offset")?;
            let len = usize_of(&core["len"], "len")?;
            tensors.push(IdxTensor::from_dense(
                indices,
                data[offset..offset + len].to_vec(),
            )?);
        }
        trees.push(TreeTN::from_tensors(tensors, (0..n).collect())?);
    }
    Ok(trees)
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    run(&args[1..])
}

/// `args`: fixture directory, algorithm, request file, output prefix.
fn run(args: &[String]) -> Result<(), Error> {
    if args.len() != 4 {
        return Err("usage: agreement-worker <fixture> <rsi|aci> <request.json> <output>".into());
    }
    let fixture = std::path::Path::new(&args[0]);
    let algorithm = args[1].as_str();
    let request: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let output = std::path::Path::new(&args[3]);
    let manifest: Value = serde_json::from_slice(&fs::read(fixture.join("fixture.json"))?)?;

    let n = usize_of(&manifest["n"], "n")?;
    let sites: Vec<DynIndex> = manifest["physical_dims"]
        .as_array()
        .ok_or("physical_dims")?
        .iter()
        .map(|d| usize_of(d, "physical dim").map(DynIndex::new_dyn))
        .collect::<Result<_, _>>()?;
    let data = read_f64(&fixture.join("inputs.bin"))?;
    let operands = manifest["operands"].as_array().ok_or("operands")?;
    let inputs = load_trees(&manifest, &data, &sites)?;

    let cap = usize_of(&request["cap"], "cap")?;
    let seed = request["seed"].as_u64().unwrap_or(0);
    let root = usize_of(&request["root"], "root")?;
    let start = Instant::now();
    let (tree, diagnostics) = match algorithm {
        "rsi" => {
            let probes = match request.get("probes").and_then(Value::as_str) {
                Some(file) => {
                    let values = read_f64(std::path::Path::new(file))?;
                    let mut maps = vec![HashMap::new(); inputs.len()];
                    for entry in request["probe_layout"].as_array().ok_or("probe_layout")? {
                        let [a, v, d, k, off] =
                            [0, 1, 2, 3, 4].map(|i| usize_of(&entry[i], "probe layout entry"));
                        let (a, v, d, k, off) = (a?, v?, d?, k?, off?);
                        maps[a].insert(
                            v,
                            Matrix::from_col_major_vec(d, k, values[off..off + d * k].to_vec()),
                        );
                    }
                    Some(tensor4all_treersi::TreeRsiProbes::new(maps))
                }
                None => None,
            };
            let options = tensor4all_treersi::TreeRsiOptions {
                max_bond_dim: Some(cap),
                sketch_dim: request["sketch_dim"].as_u64().map(|k| k as usize),
                oversampling: request["oversampling"].as_u64().unwrap_or(5) as usize,
                rel_tol: request["rel_tol"].as_f64().unwrap_or(1e-14),
                root: Some(root),
                seed,
                probes,
                ..Default::default()
            };
            let result = tensor4all_treersi::hadamard_many::<f64, _>(&inputs, &options)?;
            let d = &result.diagnostics;
            let diag = json!({
                "sketch_dim": d.sketch_dim,
                "edges": d.edges.iter().map(|e| json!({
                    "child": e.child, "parent": e.parent, "rank": e.rank,
                    "rows": e.rows, "columns": e.columns,
                    "exact_columns": e.exact_columns,
                    "rank_limit_reached": e.rank_limit_reached,
                    "relative_pivot": e.relative_pivot,
                })).collect::<Vec<_>>(),
            });
            (result.tree, diag)
        }
        "aci" => {
            let options = tensor4all_treeaci::TreeAciOptions {
                max_bond_dim: Some(cap),
                tolerance: request["tolerance"].as_f64().unwrap_or(1e-12),
                max_sweeps: request["max_sweeps"].as_u64().unwrap_or(20) as usize,
                min_sweeps: request["min_sweeps"].as_u64().unwrap_or(2) as usize,
                rng_seed: seed,
                root: Some(root),
                max_working_bytes: request["max_working_bytes"]
                    .as_u64()
                    .map(|b| b as usize)
                    .unwrap_or(
                        tensor4all_treeaci::TreeAciOptions::<usize>::default().max_working_bytes,
                    ),
                ..Default::default()
            };
            let result = tensor4all_treeaci::hadamard_many::<f64, _>(&inputs, &options)?;
            let diag = json!({
                "termination": format!("{:?}", result.termination),
                "sweep_max_ranks": result.max_ranks,
                "sweep_max_errors": result.max_errors,
                "global_pivots": result.global_pivots_found,
            });
            (result.tree, diag)
        }
        other => return Err(format!("unknown algorithm {other}").into()),
    };
    let seconds = start.elapsed().as_secs_f64();

    // Export cores in the fixture's neighbor order, then the site, column-major.
    let mut metadata = Vec::new();
    let mut flat: Vec<f64> = Vec::new();
    for v in 0..n {
        let neighbors = operands[0][v]["neighbors"].as_array().ok_or("neighbors")?;
        let mut indices = Vec::new();
        for w in neighbors {
            let w = usize_of(w, "neighbor")?;
            let edge = tree.edge_between(&v, &w).ok_or("output edge")?;
            indices.push(tree.bond_index(edge).ok_or("output bond")?.clone());
        }
        indices.push(sites[v].clone());
        let shape: Vec<usize> = indices.iter().map(|i| i.dim()).collect();
        let values = tree
            .tensor(tree.node_index(&v).ok_or("output node")?)
            .ok_or("output tensor")?
            .permute_indices(&indices)?
            .to_vec::<f64>()?;
        metadata.push(json!({
            "neighbors": neighbors, "shape": shape, "offset": flat.len(), "len": values.len(),
        }));
        flat.extend(values);
    }
    let mut file = BufWriter::new(fs::File::create(output.with_extension("bin"))?);
    for value in flat {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    let record = json!({
        "algorithm": algorithm,
        "seconds": seconds,
        "edge_ranks": edge_ranks(&tree)?,
        "diagnostics": diagnostics,
        "cores": metadata,
    });
    fs::write(output.with_extension("json"), serde_json::to_vec(&record)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{load_trees, read_f64, run, Error, Tree};
    use rand::{Rng, SeedableRng};
    use serde_json::{json, Value};
    use std::{fs, path::PathBuf};
    use tensor4all_core::{ColMajorArrayRef, DynIndex};
    use tensor4all_treetn::{CachedEvaluatorOptions, TreeTNCachedEvaluator};

    const N: usize = 4;

    /// Two random signed bond-2 chains in fixture format; returns their manifest.
    fn write_fixture(dir: &PathBuf) -> Value {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut data = Vec::new();
        let mut operands = Vec::new();
        for _ in 0..2 {
            let mut cores = Vec::new();
            for v in 0..N {
                let neighbors: Vec<usize> = [v.checked_sub(1), (v + 1 < N).then_some(v + 1)]
                    .into_iter()
                    .flatten()
                    .collect();
                let mut shape = vec![2; neighbors.len()];
                shape.push(2);
                let len: usize = shape.iter().product();
                cores.push(json!({"neighbors": neighbors, "shape": shape, "offset": data.len(), "len": len}));
                data.extend((0..len).map(|_| rng.random_range(-1.0..1.0)));
            }
            operands.push(cores);
        }
        let manifest = json!({"n": N, "physical_dims": vec![2; N], "operands": operands});
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("fixture.json"), manifest.to_string()).unwrap();
        let bytes: Vec<u8> = data.iter().flat_map(|x: &f64| x.to_le_bytes()).collect();
        fs::write(dir.join("inputs.bin"), bytes).unwrap();
        manifest
    }

    fn all_values(tree: &Tree, sites: &[DynIndex]) -> Vec<f64> {
        let points: Vec<usize> = (0..1 << N)
            .flat_map(|i| (0..N).map(move |b| (i >> b) & 1))
            .collect();
        let mut evaluator =
            TreeTNCachedEvaluator::new(tree, sites, CachedEvaluatorOptions::default()).unwrap();
        evaluator
            .evaluate_batched(ColMajorArrayRef::new(&points, &[N, 1 << N]).unwrap())
            .unwrap()
            .into_iter()
            .map(|x| x.real())
            .collect()
    }

    /// Runs the worker and returns the maximum deviation from the exact product.
    fn product_error(
        dir: &PathBuf,
        algorithm: &str,
        request: Value,
        tag: &str,
    ) -> Result<f64, Error> {
        let manifest: Value = serde_json::from_slice(&fs::read(dir.join("fixture.json"))?)?;
        let sites: Vec<DynIndex> = (0..N).map(|_| DynIndex::new_dyn(2)).collect();
        let inputs = load_trees(&manifest, &read_f64(&dir.join("inputs.bin"))?, &sites)?;
        let request_path = dir.join(format!("{tag}.request.json"));
        fs::write(&request_path, request.to_string())?;
        let prefix = dir.join(tag);
        let args = [dir, &PathBuf::from(algorithm), &request_path, &prefix]
            .map(|p| p.to_string_lossy().into_owned());
        run(&args)?;
        let record: Value = serde_json::from_slice(&fs::read(prefix.with_extension("json"))?)?;
        assert_eq!(record["algorithm"], algorithm);
        let output = json!({"n": N, "operands": [record["cores"]]});
        let tree =
            load_trees(&output, &read_f64(&prefix.with_extension("bin"))?, &sites)?.remove(0);
        let (a, b, c) = (
            all_values(&inputs[0], &sites),
            all_values(&inputs[1], &sites),
            all_values(&tree, &sites),
        );
        Ok(a.iter()
            .zip(&b)
            .zip(&c)
            .map(|((x, y), z)| (x * y - z).abs())
            .fold(0.0, f64::max))
    }

    #[test]
    fn rsi_aci_and_probe_replay_reproduce_the_exact_product() {
        let dir = std::env::temp_dir().join(format!("agreement-worker-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        write_fixture(&dir);
        // Cap 4 holds the exact product of two bond-2 chains.
        let base = json!({"cap": 4, "seed": 1, "root": N - 1});
        assert!(product_error(&dir, "rsi", base.clone(), "rsi").unwrap() < 1e-12);
        let mut aci = base.clone();
        aci["max_working_bytes"] = json!(1u64 << 28);
        assert!(product_error(&dir, "aci", aci, "aci").unwrap() < 1e-12);

        // Explicit probes on the sketched nodes 2..N-1, as the driver passes the author's draws.
        let probes: Vec<f64> = (0..2 * 2 * (N - 2) * 2)
            .map(|i| ((i * 7 % 5) as f64) - 2.0)
            .collect();
        let probe_path = dir.join("probes.bin");
        fs::write(
            &probe_path,
            probes
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let layout: Vec<Value> = (0..2)
            .flat_map(|a| (2..N).map(move |v| (a, v)))
            .enumerate()
            .map(|(i, (a, v))| json!([a, v, 2, 2, 4 * i]))
            .collect();
        let mut replay = base.clone();
        replay["sketch_dim"] = json!(2);
        replay["rel_tol"] = json!(0.0);
        replay["probes"] = json!(probe_path.to_string_lossy());
        replay["probe_layout"] = json!(layout);
        assert!(product_error(&dir, "rsi", replay, "replay").unwrap() < 1e-10);

        assert!(product_error(&dir, "bogus", base, "bogus")
            .unwrap_err()
            .to_string()
            .contains("unknown algorithm"));
        assert!(run(&["only-one".to_string()]).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
