//! Reproducer for the #784 seed-2026 structural nesting audit.
//!
//! Arguments: R, topology, temperature, seed, relative-tolerance mode.
//! Uses the three-input explicit-session M6a route and a nonbinding rank cap.

#[path = "../tests/support/g0.rs"]
mod g0;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use tensor4all_core::ColMajorArrayRef;

use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

fn main() -> g0::TestResult<()> {
    let args: Vec<_> = std::env::args().collect();
    let r: usize = args.get(1).map_or(Ok(6), |s| s.parse())?;
    let mode = args.get(2).map_or("nblock", String::as_str);
    let temperature = args.get(3).map_or(Ok(0.01), |s| s.parse())?;
    let (sites, mut inputs) = g0::fixture_with_temperature(r, mode, temperature)?;
    inputs.truncate(3);
    let seed: u64 = args.get(4).map_or(Ok(2026), |s| s.parse())?;
    let relative: bool = args.get(5).map_or(Ok(true), |s| s.parse())?;
    let mut options = g0::options();
    options.rng_seed = seed;
    options.max_bond_dim = Some(4000);
    options.tolerance = if relative { 1e-4 } else { 0.003183 };
    options.scale_tolerance = relative;
    let start = std::time::Instant::now();
    let context = tensor4all_tensorbackend::default_cpu_execution_context();
    let mut algorithm_rng = ChaCha8Rng::seed_from_u64(seed);
    let result = context.with_concrete_session(|session| {
        tensor4all_treeaci::tree_elementwise_batched_with_rng_on_session::<Complex64, _, _, _>(
            |batch, output| {
                for (p, value) in output.iter_mut().enumerate() {
                    *value = 1.0
                        / (Complex64::new(0.5, 0.0)
                            + 2.0 * batch.get(0, p)?
                            + 2.0 * batch.get(1, p)?
                            + Complex64::i() * batch.get(2, p)?);
                }
                Ok(())
            },
            &inputs,
            &options,
            &mut algorithm_rng,
            session,
        )
    })??;
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
    // Independent global reference peak: the smallest Matsubara magnitude
    // occurs at n=size/2 or size/2-1. Enumerate the bounded 2D momentum grid
    // (at most 1024^2 pairs, O(1) storage), not the 3D tensor. Keep every
    // original sample and add this point so random misses cannot underestimate
    // the normalization scale of this high-dynamic-range reference (#854).
    let mut peak = (0, 0, size / 2);
    let mut peak_scale = 0.0_f64;
    for x in 0..size {
        for y in 0..size {
            let magnitude = g0::exact_with_temperature(r, x, y, size / 2, temperature).norm();
            if magnitude > peak_scale {
                peak_scale = magnitude;
                peak = (x, y, size / 2);
            }
        }
    }
    points.push(peak);
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
    assert_eq!(
        scale, peak_scale,
        "reference peak must normalize all samples"
    );
    let normalized_max_error = max_error / scale;
    let relative_sampled_l2 = (residual_sq / reference_sq).sqrt();
    println!(
        "{}",
        json!({"r":r,"topology":mode,"temperature":temperature,
        "seconds":seconds,"termination":format!("{:?}",result.termination),
        "ranks":result.max_ranks,"errors":result.max_errors,"guard":result.global_pivots_found,
        "evaluated_points":result.diagnostics.evaluated_points,
        "sample_count":points.len(),"reference_peak":peak,"reference_scale":scale,"normalized_max_error":normalized_max_error,"relative_sampled_l2":relative_sampled_l2})
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
