//! Replays G0 convergence and construction cost for #784/#854.
//!
//! Arguments: `R topology temperature`, where topology is `nblock`, `cttn`,
//! or `swap`. Uses the #885 settings: relative tolerance 1e-4, seed 42,
//! cap 2000, 50 passes and default Guard. Only construction is timed.
//! Checks 1,093 deterministic and held-out points against the analytic G0;
//! this is sampling evidence, not a full-grid error certificate.
//!
//! Example: `cargo run --release -p tensor4all-treeaci --example convergence_g0 -- 6 nblock 0.01`
#[path = "../tests/support/g0.rs"]
mod g0;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treeaci::tree_elementwise_batched;
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

fn main() -> g0::TestResult<()> {
    let args: Vec<_> = std::env::args().collect();
    let r: usize = args.get(1).map_or(Ok(6), |s| s.parse())?;
    let mode = args.get(2).map_or("nblock", String::as_str);
    let temperature = args.get(3).map_or(Ok(0.01), |s| s.parse())?;
    let (sites, inputs) = g0::fixture_with_temperature(r, mode, temperature)?;
    let mut options = g0::options();
    options.tolerance = 1e-4;
    options.scale_tolerance = true;
    let start = std::time::Instant::now();
    let result = tree_elementwise_batched(g0::operator, &inputs, &options)?;
    let seconds = start.elapsed().as_secs_f64();

    let size = 1usize << r;
    let mut points = Vec::new();
    for x in [0, size / 2 - 1, size / 2, size - 1] {
        for y in [0, size / 2 - 1, size / 2, size - 1] {
            for n in [0, size / 2 - 1, size / 2, size - 1] {
                points.push((x, y, n));
            }
        }
    }
    points.extend(
        [
            (292, 70, 256),
            (292, 442, 256),
            (242, 70, 256),
            (230, 442, 256),
            (243, 454, 256),
        ]
        .map(|(x, y, n)| (x % size, y % size, n % size)),
    );
    let mut rng = ChaCha8Rng::seed_from_u64(854);
    points.extend((0..1024).map(|_| {
        (
            rng.random_range(0..size),
            rng.random_range(0..size),
            rng.random_range(0..size),
        )
    }));
    let mut coordinates = Vec::with_capacity(points.len() * 3 * r);
    for &(x, y, n) in &points {
        for bit in 0..r {
            for value in [x, y, n] {
                coordinates.push((value >> (r - 1 - bit)) & 1);
            }
        }
    }
    let mut evaluator =
        TreeTNCachedEvaluator::new(&result.tree, &sites, CachedEvaluatorOptions::default())?;
    let actual = evaluator.evaluate_batched_typed::<Complex64>(
        ColMajorArrayRef::new(&coordinates, &[3 * r, points.len()])?,
        EvaluationHint::default(),
    )?;
    let mut max_error = 0.0_f64;
    let mut scale = 0.0_f64;
    let mut residual_sq = 0.0;
    let mut reference_sq = 0.0;
    for (value, &(x, y, n)) in actual.into_iter().zip(&points) {
        let expected = g0::exact_with_temperature(r, x, y, n, temperature);
        let error = (value - expected).norm();
        assert!(error.is_finite());
        max_error = max_error.max(error);
        scale = scale.max(expected.norm());
        residual_sq += error * error;
        reference_sq += expected.norm_sqr();
    }
    let normalized_max_error = max_error / scale;
    let relative_sampled_l2 = (residual_sq / reference_sq).sqrt();
    println!(
        "{}",
        json!({"r":r,"topology":mode,"temperature":temperature,
        "seconds":seconds,"termination":format!("{:?}",result.termination),
        "ranks":result.max_ranks,"errors":result.max_errors,"guard":result.global_pivots_found,
        "evaluated_points":result.diagnostics.evaluated_points,
        "sample_count":points.len(),"normalized_max_error":normalized_max_error,"relative_sampled_l2":relative_sampled_l2})
    );
    assert!(
        normalized_max_error <= 5e-4,
        "normalized sampled max residual={normalized_max_error:e}"
    );
    assert!(
        relative_sampled_l2 <= 0.007,
        "relative sampled L2 residual={relative_sampled_l2:e}"
    );
    Ok(())
}
