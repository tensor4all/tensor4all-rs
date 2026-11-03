//! Baseline attribution and matched-rank contraction probe for issue #1108.
use std::hint::black_box;
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor, TensorContractionLike};
use tensor4all_treetn::contraction::{contract, ContractionOptions};
use tensor4all_treetn::TreeTN;

fn pair(
    chi: usize,
    d: usize,
) -> anyhow::Result<(TreeTN<IdxTensor, usize>, TreeTN<IdxTensor, usize>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(1108);
    let length = 6;
    let shared: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(d)).collect();
    let mut trees = Vec::new();
    for _ in 0..2 {
        let bonds: Vec<_> = (1..length).map(|_| DynIndex::new_dyn(chi)).collect();
        let mut tensors = Vec::new();
        for site in 0..length {
            let mut indices = vec![shared[site].clone(), DynIndex::new_dyn(d)];
            if site > 0 {
                indices.push(bonds[site - 1].clone());
            }
            if site + 1 < length {
                indices.push(bonds[site].clone());
            }
            let tensor = IdxTensor::random::<f64, _>(&mut rng, indices)?;
            tensors.push(tensor.scale((1.0 / tensor.norm()?).into())?);
        }
        trees.push(TreeTN::from_tensors(tensors, (0..length).collect())?);
    }
    let b = trees.pop().ok_or_else(|| anyhow::anyhow!("missing B"))?;
    let a = trees.pop().ok_or_else(|| anyhow::anyhow!("missing A"))?;
    Ok((a, b))
}

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|arg| arg == "--decompositions") {
        return decompositions();
    }
    println!("method,chi,d,repeat,seconds,norm");
    for (chi, d) in [(8, 2), (16, 2), (32, 2), (16, 4)] {
        let (a, b) = pair(chi, d)?;
        for (method, options) in [
            ("zipup", ContractionOptions::zipup()),
            ("fit", ContractionOptions::fit().with_nfullsweeps(2)),
        ] {
            let options = options.with_max_bond_dim(chi);
            // One untimed correctness/initialization call per cell.
            let mut reference = contract(&a, &b, &0, options.clone())?;
            let reference_norm = reference.norm()?;
            for repeat in 0..3 {
                eprintln!("CASE {method} chi={chi} d={d} repeat={repeat}");
                let start = Instant::now();
                let mut result =
                    black_box(contract(black_box(&a), black_box(&b), &0, options.clone())?);
                let seconds = start.elapsed().as_secs_f64();
                let norm = result.norm()?;
                assert!((norm - reference_norm).abs() <= 1e-12 * reference_norm);
                println!("{method},{chi},{d},{repeat},{seconds},{norm}");
            }
        }
    }
    Ok(())
}

fn decompositions() -> anyhow::Result<()> {
    use std::{fs, io::Write, process::Command};
    use tensor4all_core::{
        factorize, qr, randomized_svd, FactorizeOptions, RsvdOptions, SvdTruncationPolicy,
    };
    let reference = std::env::var("T4A_LANCZOS_REFERENCE").map_err(|_| {
        anyhow::anyhow!(
            "set T4A_LANCZOS_REFERENCE to the binary built by scripts/build-rsvd-reference.py"
        )
    })?;
    println!("chi,d,rows,cols,rank,method,repeat,seconds,relative_error,gemm_qr_fraction");
    for (chi, d) in [(16, 2), (32, 2), (64, 2), (16, 4)] {
        let n = chi * d * d;
        let i = DynIndex::new_dyn(n);
        let j = DynIndex::new_dyn(n);
        let x = DynIndex::new_dyn(n);
        let y = DynIndex::new_dyn(n);
        let mut rng = ChaCha8Rng::seed_from_u64(1108);
        let (u, _) = qr::<f64>(
            &IdxTensor::random::<f64, _>(&mut rng, vec![i.clone(), x])?,
            std::slice::from_ref(&i),
        )?;
        let (v, _) = qr::<f64>(
            &IdxTensor::random::<f64, _>(&mut rng, vec![j.clone(), y])?,
            std::slice::from_ref(&j),
        )?;
        let singular: Vec<_> = (0..n).map(|p| 0.9_f64.powi(p as i32)).collect();
        let s = IdxTensor::from_diag(
            vec![u.indices()[1].clone(), v.indices()[1].clone()],
            singular.clone(),
        )?;
        let a = u.contract_pair(&s)?.contract_pair(&v)?;
        let norm = a.norm()?;
        let optimal = singular[chi..].iter().map(|s| s * s).sum::<f64>().sqrt() / norm;
        let path =
            std::env::temp_dir().join(format!("t4a-1108-{}-{n}-{chi}.bin", std::process::id()));
        {
            let mut file = fs::File::create(&path)?;
            for value in [n, n, chi] {
                file.write_all(&(value as u64).to_le_bytes())?;
            }
            for value in a.to_vec::<f64>()? {
                file.write_all(&value.to_le_bytes())?;
            }
        }
        let prime = Instant::now();
        while prime.elapsed().as_secs_f64() < 1.0 {
            std::hint::spin_loop();
        }
        for method in ["svd", "rsvd"] {
            for repeat in 0..4 {
                let mut rng = ChaCha8Rng::seed_from_u64(1108);
                let started = Instant::now();
                let (reconstructed, seconds, fraction) = if method == "svd" {
                    let result = factorize(
                        &a,
                        std::slice::from_ref(&i),
                        &FactorizeOptions::svd()
                            .with_max_bond_dim(chi)
                            .with_svd_policy(SvdTruncationPolicy::new(0.)),
                    )?;
                    let seconds = started.elapsed().as_secs_f64();
                    (result.left.contract_pair(&result.right)?, seconds, 0.)
                } else {
                    let result = randomized_svd(
                        &a,
                        std::slice::from_ref(&i),
                        &RsvdOptions::new(chi),
                        &mut rng,
                    )?;
                    let seconds = started.elapsed().as_secs_f64();
                    let fraction = (result.timings.gemm + result.timings.qr).as_secs_f64()
                        / result.timings.total.as_secs_f64();
                    (
                        result
                            .u
                            .contract_pair(&result.s)?
                            .contract_pair(&result.vh)?,
                        seconds,
                        fraction,
                    )
                };
                let residual = a.sub(&reconstructed)?.norm()? / norm;
                anyhow::ensure!(
                    residual <= 1.05 * optimal,
                    "{method} error {residual}, optimal {optimal}"
                );
                if repeat > 0 {
                    println!(
                        "{chi},{d},{n},{n},{chi},{method},{},{seconds},{residual},{fraction}",
                        repeat - 1
                    );
                }
            }
        }
        let output = Command::new(&reference).arg(&path).output()?;
        anyhow::ensure!(
            output.status.success(),
            "Lanczos reference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        for line in std::str::from_utf8(&output.stdout)?.lines() {
            let fields: Vec<_> = line.split(',').collect();
            anyhow::ensure!(
                fields.len() == 5 && fields[0] == "lanczos",
                "malformed Lanczos output"
            );
            let residual: f64 = fields[3].parse()?;
            anyhow::ensure!(
                residual <= 1.001 * optimal,
                "Lanczos error {residual}, optimal {optimal}"
            );
            println!(
                "{chi},{d},{n},{n},{chi},lanczos,{},{},{},0",
                fields[1], fields[2], fields[3]
            );
        }
        fs::remove_file(path)?;
    }
    Ok(())
}
