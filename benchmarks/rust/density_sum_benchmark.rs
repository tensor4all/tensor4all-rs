//! Compare density-matrix summation with one N-ary sum and truncation.
//!
//! Families and splitmix64 generator follow the public reproducer in
//! tensor4all/tensor4all-rs#1107 (maxsroda); no tensor-network kernel is copied.
//! Run one method per process for meaningful peak RSS, for example:
//! `VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 /usr/bin/time -l
//! target/release/examples/density_sum_benchmark smooth density 12 64`.
//! Arguments: family (smooth/random/mixed), method (density/nary), R, N.

use std::{error::Error, hint::black_box, time::Instant};

use tensor4all_core::{DynIndex, IdxTensor, SvdTruncationPolicy};
use tensor4all_treetn::{SumEngine, TreeTN, TruncationOptions};

mod sum_benchmark_inputs;
use sum_benchmark_inputs::{term, BOND};

type Net = TreeTN<IdxTensor, usize>;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const CUTOFF: f64 = 1e-8;

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
        .map(|i| term(&sites, family, i, 0))
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
