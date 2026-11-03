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
            cores.push(
                json!({"neighbors": neighbors, "shape": shape, "offset": data.len(), "len": len}),
            );
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
fn product_error(dir: &PathBuf, algorithm: &str, request: Value, tag: &str) -> Result<f64, Error> {
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
    let tree = load_trees(&output, &read_f64(&prefix.with_extension("bin"))?, &sites)?.remove(0);
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
