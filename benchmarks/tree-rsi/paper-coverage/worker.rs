//! Isolated native TreeTN worker; the Python driver owns independent validation.
use num_complex::Complex64;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    io::{BufWriter, Write},
    time::Instant,
};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
#[cfg(feature = "treeaci")]
use tensor4all_treeaci::TreeAciScalar as BenchScalar;
#[cfg(feature = "rsi")]
use tensor4all_treersi::TreeRsiScalar as BenchScalar;
use tensor4all_treetn::TreeTN;
type Tree = TreeTN<IdxTensor, usize>;
fn emit(v: Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn ranks(tree: &Tree) -> Result<Vec<(usize, usize, usize)>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for (a, b) in tree.site_index_network().edges() {
        out.push((
            a.min(b),
            a.max(b),
            tree.bond_index(tree.edge_between(&a, &b).ok_or("missing edge")?)
                .ok_or("missing bond")?
                .dim(),
        ));
    }
    out.sort();
    Ok(out)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let fixture = std::path::Path::new(&args[1]);
    let manifest: Value = serde_json::from_slice(&fs::read(fixture.join("fixture.json"))?)?;
    let dtype = manifest["dtype"].as_str().unwrap_or("f64");
    if dtype == "c64" {
        run::<Complex64>(
            &args,
            manifest,
            |re, im| Complex64::new(re, im),
            |x| (x.re, x.im),
        )
    } else {
        run::<f64>(&args, manifest, |re, _| re, |x| (x, 0.0))
    }
}
fn run<T: BenchScalar>(
    args: &[String],
    manifest: Value,
    decode: fn(f64, f64) -> T,
    encode: fn(T) -> (f64, f64),
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = std::path::Path::new(&args[1]);
    let cap: usize = args[2].parse()?;
    let seed: u64 = args[3].parse()?;
    let output = std::path::Path::new(&args[4]);
    let n = manifest["n"].as_u64().ok_or("n")? as usize;
    let root = manifest["root"].as_u64().ok_or("root")? as usize;
    let sites: Vec<_> = manifest["physical_dims"]
        .as_array()
        .ok_or("physical dims")?
        .iter()
        .map(|d| DynIndex::new_dyn(d.as_u64().unwrap() as usize))
        .collect();
    let options: Value = if args.len() > 5 {
        serde_json::from_slice(&fs::read(&args[5])?)?
    } else {
        json!({})
    };
    let local_tol = options["local_tolerance"].as_f64().unwrap_or(1e-12);
    let bytes = fs::read(fixture.join("inputs.bin"))?;
    let data: Vec<T> = bytes
        .chunks_exact(16)
        .map(|s| {
            decode(
                f64::from_le_bytes(s[..8].try_into().unwrap()),
                f64::from_le_bytes(s[8..].try_into().unwrap()),
            )
        })
        .collect();
    let mut inputs = Vec::new();
    for operand in manifest["operands"].as_array().ok_or("operands")? {
        let mut bonds = HashMap::new();
        for (v, core) in operand.as_array().ok_or("cores")?.iter().enumerate() {
            for (axis, w) in core["neighbors"]
                .as_array()
                .ok_or("neighbors")?
                .iter()
                .enumerate()
            {
                let w = w.as_u64().ok_or("neighbor")? as usize;
                let dim = core["shape"][axis].as_u64().ok_or("bond dimension")? as usize;
                if v < w {
                    bonds.insert((v, w), DynIndex::new_dyn(dim));
                }
            }
        }
        let mut tensors = Vec::new();
        for (v, core) in operand.as_array().ok_or("cores")?.iter().enumerate() {
            let mut inds = core["neighbors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| {
                    let w = w.as_u64().unwrap() as usize;
                    bonds[&(v.min(w), v.max(w))].clone()
                })
                .collect::<Vec<_>>();
            inds.push(sites[v].clone());
            let offset = core["offset"].as_u64().unwrap() as usize;
            let len = core["len"].as_u64().unwrap() as usize;
            tensors.push(IdxTensor::from_dense(
                inds,
                data[offset..offset + len].to_vec(),
            )?);
        }
        inputs.push(TreeTN::from_tensors(tensors, (0..n).collect())?);
    }
    emit(
        json!({"phase":"inputs_ready", "input_ranks": inputs.iter().map(ranks).collect::<Result<Vec<_>,_>>()?}),
    );
    // Initialize the selected backend with one bounded native contraction.
    use tensor4all_core::ColMajorArrayRef;
    use tensor4all_treetn::{CachedEvaluatorOptions, TreeTNCachedEvaluator};
    let points = vec![0usize; n];
    for input in &inputs {
        let mut eval =
            TreeTNCachedEvaluator::new(input, &sites, CachedEvaluatorOptions::default())?;
        std::hint::black_box(eval.evaluate_batched(ColMajorArrayRef::new(&points, &[n, 1])?)?);
    }
    emit(json!({"phase":"algorithm_start"}));
    #[cfg(feature = "rsi")]
    let (tree, elapsed, diagnostics) = {
        let options = tensor4all_treersi::TreeRsiOptions {
            max_bond_dim: Some(cap),
            seed,
            root: Some(root),
            rel_tol: local_tol,
            sketch_dim: options["sketch_dim"].as_u64().map(|v| v as usize),
            oversampling: options["oversampling"].as_u64().unwrap_or(5) as usize,
            ..Default::default()
        };
        let start = Instant::now();
        let result = std::hint::black_box(tensor4all_treersi::hadamard_many::<T, _>(
            std::hint::black_box(&inputs),
            &options,
        ));
        let elapsed = start.elapsed().as_secs_f64();
        match result {
            Ok(r) => {
                let d = json!({"sketch_dim":r.diagnostics.sketch_dim,"edges":r.diagnostics.edges.iter().map(|e| json!({"child":e.child,"parent":e.parent,"rank":e.rank,"rows":e.rows,"columns":e.columns,"exact_columns":e.exact_columns,"relative_pivot":e.relative_pivot})).collect::<Vec<_>>()});
                (r.tree, elapsed, d)
            }
            Err(e) => {
                emit(json!({"phase":"algorithm_error","seconds":elapsed,"error":e.to_string()}));
                return Ok(());
            }
        }
    };
    #[cfg(feature = "treeaci")]
    let (tree, elapsed, diagnostics) = {
        let options = tensor4all_treeaci::TreeAciOptions {
            max_bond_dim: Some(cap),
            rng_seed: seed,
            root: Some(root),
            tolerance: local_tol,
            max_sweeps: options["max_sweeps"].as_u64().unwrap_or(20) as usize,
            min_sweeps: options["min_sweeps"].as_u64().unwrap_or(2) as usize,
            scale_tolerance: true,
            ..Default::default()
        };
        let start = Instant::now();
        let result = std::hint::black_box(tensor4all_treeaci::hadamard_many::<T, _>(
            std::hint::black_box(&inputs),
            &options,
        ));
        let elapsed = start.elapsed().as_secs_f64();
        match result {
            Ok(r) => {
                let d = json!({"termination":format!("{:?}",r.termination), "sweep_ranks":r.max_ranks,"local_errors":r.max_errors,"global_pivots":r.global_pivots_found,"edge_ranks":r.diagnostics.edge_ranks,"candidate_sizes":r.diagnostics.candidate_set_sizes,"evaluated_points":r.diagnostics.evaluated_points,"frame_bytes":r.diagnostics.frame_retained_bytes,"sample_arena_bytes":r.diagnostics.sample_arena_retained_bytes});
                (r.tree, elapsed, d)
            }
            Err(e) => {
                emit(json!({"phase":"algorithm_error","seconds":elapsed,"error":e.to_string()}));
                return Ok(());
            }
        }
    };
    emit(
        json!({"phase":"algorithm_done","seconds":elapsed,"output_ranks":ranks(&tree)?,"diagnostics":diagnostics}),
    );
    let mut metadata = Vec::new();
    let mut flat = Vec::new();
    for v in 0..n {
        let neighbors = manifest["operands"][0][v]["neighbors"]
            .as_array()
            .ok_or("neighbors")?;
        let mut inds = Vec::new();
        for w in neighbors {
            let w = w.as_u64().ok_or("neighbor")? as usize;
            inds.push(
                tree.bond_index(tree.edge_between(&v, &w).ok_or("edge")?)
                    .ok_or("bond")?
                    .clone(),
            );
        }
        inds.push(sites[v].clone());
        let shape: Vec<_> = inds.iter().map(|i| i.dim()).collect();
        let values = tree
            .tensor(tree.node_index(&v).ok_or("node")?)
            .ok_or("tensor")?
            .permute_indices(&inds)?
            .to_vec::<T>()?;
        metadata.push(
            json!({"neighbors":neighbors,"shape":shape,"offset":flat.len(),"len":values.len()}),
        );
        flat.extend(values);
    }
    let mut file = BufWriter::new(fs::File::create(output)?);
    for value in flat {
        let (re, im) = encode(value);
        file.write_all(&re.to_le_bytes())?;
        file.write_all(&im.to_le_bytes())?;
    }
    file.flush()?;
    fs::write(
        output.with_extension("cores.json"),
        serde_json::to_vec(&metadata)?,
    )?;
    emit(json!({"phase":"validation_ready","cores":metadata}));
    Ok(())
}
