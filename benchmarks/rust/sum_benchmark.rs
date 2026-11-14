//! One fresh-process summation measurement for the bounded #1127 experiment.

use std::{hint::black_box, time::Instant};

use num_complex::Complex64 as C;
use serde_json::{json, Value};
use tensor4all_core::{IdxTensor, SvdTruncationPolicy};
use tensor4all_treeaci::{
    tree_elementwise_batched, TreeAciDiagnostics, TreeAciOptions, TreeAciTermination,
};
use tensor4all_treetn::{PairwiseSumOrder, PerAddCutoff, SumEngine, TreeTN, TruncationOptions};

#[path = "sum_benchmark/inputs.rs"]
mod inputs;
mod sum_benchmark_inputs;
#[cfg(test)]
#[path = "sum_benchmark/tests.rs"]
mod tests;

type Net = TreeTN<IdxTensor, usize>;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
struct Config {
    original: Value,
    family: String,
    size: usize,
    terms: usize,
    input_seed: u64,
    aci_seed: u64,
    method: String,
    tolerance: f64,
    cutoff: f64,
    prime_seconds: f64,
}

struct Outcome {
    network: Net,
    discarded_weight: Option<f64>,
    aci: Option<AciInfo>,
}

struct AciInfo {
    termination: TreeAciTermination,
    max_ranks: Vec<usize>,
    max_errors: Vec<f64>,
    global_pivots_found: Vec<usize>,
    diagnostics: TreeAciDiagnostics<usize>,
}

fn parse_config(value: Value) -> Result<Config> {
    let keys = [
        "family",
        "size",
        "terms",
        "input_seed",
        "aci_seed",
        "method",
        "tolerance",
        "cutoff",
        "prime_seconds",
    ];
    let object = value
        .as_object()
        .ok_or("configuration must be a JSON object")?;
    if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {
        return Err("configuration must contain exactly the nine documented fields".into());
    }
    let integer = |key: &str| -> Result<u64> {
        value[key]
            .as_u64()
            .ok_or_else(|| format!("{key} must be an unsigned integer").into())
    };
    let number = |key: &str| -> Result<f64> {
        value[key]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| format!("{key} must be a finite number").into())
    };
    let string = |key: &str| -> Result<String> {
        value[key]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("{key} must be a string").into())
    };
    let family = string("family")?;
    let method = string("method")?;
    let size = usize::try_from(integer("size")?)?;
    let terms = usize::try_from(integer("terms")?)?;
    let input_seed = integer("input_seed")?;
    let aci_seed = integer("aci_seed")?;
    let tolerance = number("tolerance")?;
    let cutoff = number("cutoff")?;
    let prime_seconds = number("prime_seconds")?;
    let seed_base = match family.as_str() {
        "fourier" => {
            if !(1..=6).contains(&size) || terms != 1usize << size {
                return Err("Fourier requires 1 <= size <= 6 and terms = 2^size".into());
            }
            0x5EED_0000 + size as u64
        }
        "smooth" | "random" | "mixed" => {
            if !(2..=12).contains(&size) || !(1..=64).contains(&terms) {
                return Err(
                    "#1107 dense reference requires 2 <= size <= 12 and 1 <= terms <= 64".into(),
                );
            }
            let base = match family.as_str() {
                "smooth" => 1000,
                "random" => 5000,
                _ => 9000,
            };
            base + terms as u64 - 1
        }
        _ => return Err("family must be fourier, smooth, random or mixed".into()),
    };
    input_seed
        .checked_add(seed_base)
        .ok_or("input seed offset overflows u64")?;
    if ![
        "aci",
        "sequential",
        "balanced",
        "density",
        "exact",
        "exact_truncate",
    ]
    .contains(&method.as_str())
    {
        return Err("unknown summation method".into());
    }
    if tolerance <= 0.0 || tolerance > 1.0 || !(0.0..=1.0).contains(&cutoff) {
        return Err("require 0 < tolerance <= 1 and 0 <= cutoff <= 1".into());
    }
    if !(0.0..=60.0).contains(&prime_seconds) {
        return Err("prime_seconds must be between 0 and 60".into());
    }
    Ok(Config {
        original: value,
        family,
        size,
        terms,
        input_seed,
        aci_seed,
        method,
        tolerance,
        cutoff,
        prime_seconds,
    })
}

fn run_method(config: &Config, terms: &[Net]) -> Result<Outcome> {
    if config.method == "aci" {
        let result = tree_elementwise_batched::<C, _, _>(
            |batch, output| {
                for (value, inputs) in output
                    .iter_mut()
                    .zip(batch.as_col_major_slice().chunks_exact(batch.n_inputs()))
                {
                    *value = inputs.iter().copied().sum();
                }
                Ok(())
            },
            black_box(terms),
            &TreeAciOptions {
                tolerance: config.tolerance,
                rng_seed: config.aci_seed,
                ..Default::default()
            },
        )?;
        return Ok(Outcome {
            network: result.tree,
            discarded_weight: None,
            aci: Some(AciInfo {
                termination: result.termination,
                max_ranks: result.max_ranks,
                max_errors: result.max_errors,
                global_pivots_found: result.global_pivots_found,
                diagnostics: result.diagnostics,
            }),
        });
    }
    let refs: Vec<_> = terms.iter().collect();
    let mut discarded_weight = None;
    let network = match config.method.as_str() {
        "sequential" | "balanced" => {
            let order = if config.method == "sequential" {
                PairwiseSumOrder::Sequential
            } else {
                PairwiseSumOrder::Balanced
            };
            Net::pairwise_sum(
                black_box(&refs),
                order,
                Some(PerAddCutoff {
                    center: &0,
                    cutoff: config.cutoff,
                    max_bond_dim: None,
                }),
            )?
        }
        "density" => {
            let result = Net::density_matrix_sum(black_box(&refs), &0, config.cutoff, None)?;
            discarded_weight = Some(result.discarded_weight);
            result.network
        }
        "exact" | "exact_truncate" => {
            let exact = Net::exact_sum(black_box(&refs), SumEngine::Nary)?.network;
            if config.method == "exact_truncate" {
                let policy = SvdTruncationPolicy::new(config.cutoff)
                    .with_relative()
                    .with_squared_values()
                    .with_discarded_tail_sum();
                exact.truncate([0], TruncationOptions::new().with_svd_policy(policy))?
            } else {
                exact
            }
        }
        _ => return Err("unknown summation method".into()),
    };
    Ok(Outcome {
        network,
        discarded_weight,
        aci: None,
    })
}

fn checksum(values: &[C]) -> Result<Value> {
    let mut sum = C::new(0.0, 0.0);
    let mut weighted = sum;
    let mut sum_sq = 0.0;
    let mut maxabs: f64 = 0.0;
    for (offset, value) in values.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err("nonfinite dense tensor value".into());
        }
        sum += value;
        // Fixed column-major weights make index-order drift visible across arms.
        weighted += value * ((offset + 1) as f64 / values.len() as f64);
        sum_sq += value.norm_sqr();
        maxabs = maxabs.max(value.norm());
    }
    if [sum.re, sum.im, weighted.re, weighted.im, sum_sq, maxabs]
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err("nonfinite tensor checksum".into());
    }
    Ok(
        json!({ "sum_re": sum.re, "sum_im": sum.im, "sum_sq": sum_sq, "maxabs": maxabs, "weighted_re": weighted.re, "weighted_im": weighted.im }),
    )
}

fn report(config: &Config, case: &inputs::Case, outcome: Outcome, seconds: f64) -> Result<Value> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err("operation time must be finite and nonnegative".into());
    }
    let sites = case.oracle.indices().len();
    let expected_sites = config.size + if config.family == "fourier" { 3 } else { 0 };
    if sites != expected_sites || case.terms.len() != config.terms {
        return Err("input fixture differs from its configuration".into());
    }
    let dense = black_box(outcome.network.to_dense()?).permute_indices(case.oracle.indices())?;
    let residual = dense.sub(&case.oracle)?;
    let values = dense.to_vec::<C>()?;
    let reference = checksum(&case.oracle.to_vec::<C>()?)?;
    let checksum = checksum(&values)?;
    if case.mask.len() != values.len() || values.is_empty() {
        return Err("cancellation mask differs from dense tensor shape".into());
    }
    let norm = case.oracle.norm()?;
    let maxabs = case.oracle.maxabs()?;
    if !norm.is_finite() || norm <= 0.0 || !maxabs.is_finite() || maxabs <= 0.0 {
        return Err("oracle must have a finite, positive norm and maximum magnitude".into());
    }
    let relative_error = residual.norm()? / norm;
    let maxabs_error = residual.maxabs()?;
    let maxabs_relative_error = maxabs_error / maxabs;
    let residual_values = residual
        .permute_indices(case.oracle.indices())?
        .to_vec::<C>()?;
    let cancellation_error = residual_values
        .iter()
        .zip(&case.mask)
        .filter(|(_, selected)| **selected)
        .map(|(value, _)| value.norm())
        .fold(0.0, f64::max);
    let cancellation_maxabs_relative_error = cancellation_error / maxabs;
    let cancellation_fraction =
        case.mask.iter().filter(|selected| **selected).count() as f64 / values.len() as f64;
    if [
        relative_error,
        maxabs_error,
        maxabs_relative_error,
        cancellation_maxabs_relative_error,
    ]
    .iter()
    .any(|x| !x.is_finite())
        || outcome
            .discarded_weight
            .is_some_and(|weight| !weight.is_finite() || weight < 0.0)
    {
        return Err("nonfinite or invalid numerical diagnostics".into());
    }
    let thread_env: serde_json::Map<String, Value> = [
        "RAYON_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TENFERRO_NUM_THREADS",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), json!(std::env::var(key).ok())))
    .collect();
    let aci = if let Some(info) = outcome.aci {
        if info.max_errors.iter().any(|error| !error.is_finite()) {
            return Err("ACI returned nonfinite convergence diagnostics".into());
        }
        Some(json!({
            "termination": format!("{:?}", info.termination),
            "sweeps": info.max_ranks.len(), "max_ranks": info.max_ranks,
            "max_errors": info.max_errors, "global_pivots_found": info.global_pivots_found,
            "evaluated_points": info.diagnostics.evaluated_points,
            "sample_arena_records": info.diagnostics.sample_arena_records,
            "sample_arena_retained_bytes": info.diagnostics.sample_arena_retained_bytes,
            "frame_records": info.diagnostics.frame_records,
            "frame_retained_bytes": info.diagnostics.frame_retained_bytes,
            "edge_ranks": info.diagnostics.edge_ranks,
            "saturated_edges": info.diagnostics.saturated_edges,
            "candidate_set_sizes": info.diagnostics.candidate_set_sizes,
        }))
    } else {
        None
    };
    Ok(json!({
        "schema_version": 1, "config": config.original,
        "build_commit": option_env!("T4A_BENCH_GIT_COMMIT").unwrap_or("unrecorded"),
        "release": !cfg!(debug_assertions), "thread_env": thread_env, "scalar": "c64",
        "sites": sites, "terms": case.terms.len(), "seconds": seconds,
        "max_bond_dim": outcome.network.link_dims().into_iter().max().unwrap_or(0),
        "relative_error": relative_error, "maxabs_relative_error": maxabs_relative_error, "maxabs_error": maxabs_error,
        "cancellation_fraction": cancellation_fraction, "cancellation_maxabs_relative_error": cancellation_maxabs_relative_error,
        "discarded_weight": outcome.discarded_weight, "aci": aci, "reference": reference, "checksum": checksum,
    }))
}

fn prime_cpu(seconds: f64) {
    let start = Instant::now();
    let mut state = 0x5EED_1127u64;
    while start.elapsed().as_secs_f64() < seconds {
        for _ in 0..4096 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = black_box(state);
        }
    }
    black_box(state);
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let argument = args
        .next()
        .ok_or("expected one JSON configuration argument")?;
    if args.next().is_some() {
        return Err("expected exactly one JSON configuration argument".into());
    }
    let config = parse_config(serde_json::from_str(&argument)?)?;
    prime_cpu(config.prime_seconds);
    let case = inputs::build_case(&config)?;
    let start = Instant::now();
    let outcome = run_method(&config, black_box(&case.terms))?;
    let seconds = start.elapsed().as_secs_f64();
    let result = report(&config, &case, black_box(outcome), seconds)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
