//! Compare density-matrix summation with one N-ary sum and truncation.
//!
//! Families and splitmix64 generator follow the public reproducer in
//! tensor4all/tensor4all-rs#1107 (maxsroda); no tensor-network kernel is copied.
//! Run one method per process for meaningful peak RSS, for example:
//! `VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 /usr/bin/time -l
//! target/release/examples/density_sum_benchmark smooth density 12 64`.
//! Arguments: family (smooth/random/mixed), method (density/nary), R, N.

use std::{error::Error, hint::black_box, time::Instant};

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, SvdTruncationPolicy};
use tensor4all_treetn::{SumEngine, TreeTN, TruncationOptions};

type Net = TreeTN<IdxTensor, usize>;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const BOND: usize = 10;
const CUTOFF: f64 = 1e-8;

fn signed(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
}

fn term(sites: &[DynIndex], family: &str, i: usize) -> Result<Net> {
    let seed = match family {
        "smooth" => 1000,
        "random" => 5000,
        "mixed" => 9000,
        _ => return Err("family must be smooth, random or mixed".into()),
    };
    let mut state = seed + i as u64;
    let rates: Vec<_> = match family {
        "smooth" => (0..BOND)
            .map(|r| {
                10f64.powf(3.0 * r as f64 / (BOND - 1) as f64) * (1.0 + 0.1 * signed(&mut state))
            })
            .collect(),
        "mixed" => (0..BOND)
            .map(|_| 10f64.powf(1.5 * (signed(&mut state) + 1.0)))
            .collect(),
        _ => Vec::new(),
    };
    let bonds: Vec<_> = (1..sites.len()).map(|_| DynIndex::new_dyn(BOND)).collect();
    let mut tensors = Vec::with_capacity(sites.len());
    for j in 0..sites.len() {
        let left = if j == 0 { 1 } else { BOND };
        let right = if j + 1 == sites.len() { 1 } else { BOND };
        let mut indices = Vec::new();
        if j > 0 {
            indices.push(bonds[j - 1].clone());
        }
        indices.push(sites[j].clone());
        if j + 1 < sites.len() {
            indices.push(bonds[j].clone());
        }
        let mut data = Vec::with_capacity(left * 2 * right);
        // Column-major (left, site, right), matching #1107's random stream.
        for r in 0..right {
            for s in 0..2 {
                for l in 0..left {
                    let value = if family == "random" {
                        Complex64::new(signed(&mut state), signed(&mut state))
                    } else {
                        let rate = if j == 0 { rates[r] } else { rates[l] };
                        let value = if j == 0 || j + 1 == sites.len() || l == r {
                            (-rate * 0.5f64.powi(j as i32 + 1) * s as f64).exp()
                        } else {
                            0.0
                        };
                        Complex64::new(value, 0.0)
                    };
                    data.push(value);
                }
            }
        }
        tensors.push(IdxTensor::from_dense(indices, data)?);
    }
    Ok(Net::from_tensors(tensors, (0..sites.len()).collect())?)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let family = args.get(1).map(String::as_str).unwrap_or("smooth");
    let method = args.get(2).map(String::as_str).unwrap_or("density");
    let r = args.get(3).map_or(Ok(12), |x| x.parse::<usize>())?;
    let n = args.get(4).map_or(Ok(64), |x| x.parse::<usize>())?;
    if !(2..=16).contains(&r) || n == 0 {
        return Err("dense-reference benchmark requires 2 <= R <= 16 and N > 0".into());
    }
    let sites: Vec<_> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let terms: Vec<_> = (0..n)
        .map(|i| term(&sites, family, i))
        .collect::<Result<_>>()?;
    let refs: Vec<_> = terms.iter().collect();
    let mut exact = terms[0].to_dense()?;
    for net in &terms[1..] {
        exact = exact.add(&net.to_dense()?)?;
    }
    let norm = exact.norm_squared()?;
    let policy = SvdTruncationPolicy::new(CUTOFF)
        .with_relative()
        .with_squared_values()
        .with_discarded_tail_sum();
    let start = Instant::now();
    let (network, discarded) = match method {
        "density" => {
            let result = Net::density_matrix_sum(black_box(&refs), &0, CUTOFF, None)?;
            (result.network, Some(result.discarded_weight))
        }
        "nary" => (
            Net::exact_sum(black_box(&refs), SumEngine::Nary)?
                .network
                .truncate([0], TruncationOptions::new().with_svd_policy(policy))?,
            None,
        ),
        _ => return Err("method must be density or nary".into()),
    };
    let seconds = start.elapsed().as_secs_f64();
    let network = black_box(network);
    let residual = network.to_dense()?.sub(&exact)?;
    let error_squared = residual.norm_squared()?;
    let error = (error_squared / norm).sqrt();
    let max_error = residual.maxabs()?;
    // This executable is also the benchmark's value-dependent self-check.
    assert!(error <= ((r - 1) as f64 * CUTOFF).sqrt() + 1e-7);
    if let Some(weight) = discarded {
        assert!(error_squared <= weight + 1e-12 * norm);
    }
    println!("family={family} method={method} R={r} N={n} D={BOND} cutoff={CUTOFF:e} seconds={seconds:.9} max_bond={} relative_error={error:.9e} max_error={max_error:.9e} discarded_weight={discarded:?}", network.link_dims().into_iter().max().unwrap_or(0));
    Ok(())
}
