//! Isolated native TreeTN worker; the Python driver owns independent validation.
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs,
    io::{BufWriter, Write},
    time::Instant,
};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
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
    let cap: usize = args[2].parse()?;
    let seed: u64 = args[3].parse()?;
    let output = std::path::Path::new(&args[4]);
    let manifest: Value = serde_json::from_slice(&fs::read(fixture.join("fixture.json"))?)?;
    let n = manifest["n"].as_u64().ok_or("missing n")? as usize;
    let root = manifest["root"].as_u64().ok_or("missing root")? as usize;
    let sites: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(2)).collect();
    let bytes = fs::read(fixture.join("inputs.bin"))?;
    let data: Vec<_> = bytes
        .chunks_exact(8)
        .map(|s| f64::from_le_bytes(s.try_into().unwrap()))
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
    // Check both input contractions against the independent fixture oracle.
    // This also warms backend initialization, outside the product timer.
    for (i, tree) in inputs.iter().enumerate() {
        let values = tree.to_dense()?.permute_indices(&sites)?.to_vec::<f64>()?;
        write_values(&output.with_extension(format!("input{i}.bin")), &values)?;
    }
    emit(json!({"phase":"algorithm_start"}));
    #[cfg(feature = "rsi")]
    let (tree, elapsed, diagnostics) = {
        let options = tensor4all_treersi::TreeRsiOptions {
            max_bond_dim: Some(cap),
            seed,
            root: Some(root),
            rel_tol: 1e-12,
            ..Default::default()
        };
        let start = Instant::now();
        let result = std::hint::black_box(tensor4all_treersi::hadamard_many::<f64, _>(
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
            tolerance: 1e-12,
            scale_tolerance: true,
            ..Default::default()
        };
        let start = Instant::now();
        let result = std::hint::black_box(tensor4all_treeaci::hadamard_many::<f64, _>(
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
    let actual = tree.to_dense()?.permute_indices(&sites)?.to_vec::<f64>()?;
    write_values(output, &actual)?;
    emit(json!({"phase":"validation_ready","count":actual.len()}));
    Ok(())
}
fn write_values(path: &std::path::Path, values: &[f64]) -> std::io::Result<()> {
    let mut file = BufWriter::new(fs::File::create(path)?);
    for v in values {
        file.write_all(&v.to_le_bytes())?;
    }
    file.flush()
}
