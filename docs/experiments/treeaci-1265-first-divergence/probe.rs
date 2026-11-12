//! Reproducer for the #784 seed-2026 structural nesting audit.
//!
//! Arguments: R, topology, temperature, seed, relative-tolerance mode.
//! Uses the three-input explicit-session M6a route and a nonbinding rank cap.

#[path = "g0.rs"]
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
    if let Ok(path) = std::env::var("T4A_DIAG_MAKE_VALIDATION") {
        make_validation(r, temperature, seed, &path)?;
        return Ok(());
    }
    if let Ok(max) = std::env::var("T4A_DIAG_MAX_SWEEPS") {
        options.max_sweeps = max.parse()?;
    }
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
    let importance = if std::env::var("T4A_ROOT_IMPORTANCE").as_deref() == Ok("on") {
        let grid = Grid { r, temperature };
        let samples = draw(&grid, 1, 5000, seed ^ 0x5eed);
        let approx = evaluator.evaluate_batched_typed::<Complex64>(
            ColMajorArrayRef::new(&samples.points, &[3 * r, samples.coords.len()])?,
            EvaluationHint::default(),
        )?;
        let exact: Vec<_> = samples
            .coords
            .iter()
            .map(|&[x, y, n]| grid.g0(x, y, n))
            .collect();
        let (rel, se) = relative_l2(&samples, &approx, &exact);
        Some(
            json!({"rel_l2":rel,"rel_l2_se":se,"samples":samples.coords.len(),"seed":seed^0x5eed,"norm":samples.norm}),
        )
    } else {
        None
    };
    println!(
        "{}",
        json!({"importance":importance,"r":r,"topology":mode,"temperature":temperature,
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

// Importance sampler retained from the M6a benchmark; see the investigation provenance.
use std::f64::consts::PI;
const MU: f64 = 0.5;
struct Grid {
    r: usize,
    temperature: f64,
}
impl Grid {
    fn size(&self) -> usize {
        1 << self.r
    }
    fn omega(&self, n: usize) -> f64 {
        2.0 * PI * self.temperature * (n as f64 - self.size() as f64 / 2.0 + 0.5)
    }
    fn g0(&self, x: usize, y: usize, n: usize) -> Complex64 {
        g0::exact_with_temperature(self.r, x, y, n, self.temperature)
    }
    fn point(&self, c: [usize; 3]) -> Vec<usize> {
        (0..self.r)
            .flat_map(|bit| c.map(|v| (v >> (self.r - 1 - bit)) & 1))
            .collect()
    }
}
fn frequency_sums(grid: &Grid, power: i32) -> Vec<f64> {
    let size = grid.size();
    let half = size / 2;
    let cosines: Vec<(f64, f64)> = (0..=half)
        .map(|k| {
            let multiplicity = if k == 0 || k == half { 1.0 } else { 2.0 };
            ((2.0 * PI * k as f64 / size as f64).cos(), multiplicity)
        })
        .collect();
    let omegas2: Vec<f64> = (0..half)
        .map(|h| {
            let w = 2.0 * PI * grid.temperature * (h as f64 + 0.5);
            w * w
        })
        .collect();
    let mut sums = vec![0.0; half];
    for (i, &(ci, mi)) in cosines.iter().enumerate() {
        for &(cj, mj) in &cosines[i..] {
            let weight = if ci == cj { mi * mj } else { 2.0 * mi * mj };
            let a = MU + 2.0 * ci + 2.0 * cj;
            let a2 = a * a;
            for (sum, &w2) in sums.iter_mut().zip(&omegas2) {
                let inverse = 1.0 / (a2 + w2);
                *sum += weight * inverse.powi(power);
            }
        }
    }
    sums
}

struct Samples {
    /// Column-major `[3 R, n]` site values.
    points: Vec<usize>,
    /// Grid coordinates `(kx, ky, n)` of each point.
    coords: Vec<[usize; 3]>,
    /// The density of the mixture at each point.
    density: Vec<f64>,
    /// The exact L2 norm `S` of the analytic function on the grid.
    norm: f64,
}

/// Draw `count` points from `q = (|f|^2 / S^2 + 1 / V) / 2` with
/// `|f|^2 = |G0|^(2 power)`.
fn draw(grid: &Grid, power: i32, count: usize, seed: u64) -> Samples {
    let size = grid.size();
    let half = size / 2;
    let sums = frequency_sums(grid, power);
    let total: f64 = 2.0 * sums.iter().sum::<f64>();
    // CDF over all frequencies n in 0..size.
    let mass = |n: usize| {
        let h = if n >= half { n - half } else { half - 1 - n };
        sums[h]
    };
    let mut cdf = Vec::with_capacity(size);
    let mut acc = 0.0;
    for n in 0..size {
        acc += mass(n);
        cdf.push(acc / total);
    }
    let volume = (size as f64).powi(3);
    let modulus = |kx: usize, ky: usize, n: usize| grid.g0(kx, ky, n).norm_sqr().powi(power);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut coords = Vec::with_capacity(count);
    for _ in 0..count {
        if rng.random::<bool>() {
            coords.push([
                rng.random_range(0..size),
                rng.random_range(0..size),
                rng.random_range(0..size),
            ]);
            continue;
        }
        let u: f64 = rng.random();
        let n = cdf.partition_point(|&c| c < u).min(size - 1);
        let w2 = grid.omega(n).powi(2);
        // Rejection from uniform k: |G0|^2 <= 1 / omega^2.
        loop {
            let (kx, ky) = (rng.random_range(0..size), rng.random_range(0..size));
            let accept = (modulus(kx, ky, n) * w2.powi(power)).min(1.0);
            if rng.random::<f64>() < accept {
                coords.push([kx, ky, n]);
                break;
            }
        }
    }
    let density = coords
        .iter()
        .map(|&[kx, ky, n]| 0.5 * modulus(kx, ky, n) / total + 0.5 / volume)
        .collect();
    let points = coords.iter().flat_map(|&c| grid.point(c)).collect();
    Samples {
        points,
        coords,
        density,
        norm: total.sqrt(),
    }
}

fn relative_l2(samples: &Samples, approx: &[Complex64], exact: &[Complex64]) -> (f64, f64) {
    let terms: Vec<f64> = approx
        .iter()
        .zip(exact)
        .zip(&samples.density)
        .map(|((a, e), q)| (a - e).norm_sqr() / q)
        .collect();
    let n = terms.len() as f64;
    let mean = terms.iter().sum::<f64>() / n;
    let var = terms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let rel = mean.sqrt() / samples.norm;
    // delta method: se(sqrt(E)) = se(E) / (2 sqrt(E)).
    let se = if mean > 0.0 {
        (var / n).sqrt() / (2.0 * mean.sqrt()) / samples.norm
    } else {
        0.0
    };
    (rel, se)
}

fn make_validation(r: usize, temperature: f64, seed: u64, path: &str) -> g0::TestResult<()> {
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

    let grid = Grid { r, temperature };
    let samples = draw(&grid, 1, 5000, seed ^ 0x5eed);
    let mut all_points: Vec<Vec<usize>> = samples
        .points
        .chunks_exact(3 * r)
        .map(|p| p.to_vec())
        .collect();
    let mut expected: Vec<_> = samples
        .coords
        .iter()
        .map(|&[x, y, n]| {
            let v = grid.g0(x, y, n);
            [v.re, v.im]
        })
        .collect();
    all_points.extend(coordinates.chunks_exact(3 * r).map(|p| p.to_vec()));
    expected.extend(points.iter().map(|&(x, y, n)| {
        let v = grid.g0(x, y, n);
        [v.re, v.im]
    }));
    let data = json!({"points":all_points,"expected":expected,"density":samples.density,"importance_count":samples.coords.len(),"norm":samples.norm,"peak_scale":peak_scale});
    std::fs::write(path, serde_json::to_vec(&data)?)?;
    Ok(())
}
