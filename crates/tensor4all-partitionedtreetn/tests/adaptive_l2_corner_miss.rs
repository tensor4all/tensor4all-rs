//! Reproduction of a known limitation of the L2 error contract of
//! `patched_interpolate`: uniform-sample acceptance and the audit can miss a
//! localized feature that enters a patch only through a corner or an edge.
//! See "Known limitation: corner-localized misses" in
//! `docs/design/tree-patching-error-contract.md`. The test fails while the
//! limitation exists. Run it with
//!
//! ```text
//! cargo test --release -p tensor4all-partitionedtreetn --test adaptive_l2_corner_miss \
//!     -- --ignored --nocapture
//! ```

mod adaptive_common;

use adaptive_common::*;
use tensor4all_core::DynIndex;
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    GlobalL2Error, MeasurementMethod, NormReport, PatchedInterpolationOptions,
};
use tensor4all_partitionedtreetn::{ErrorNorm, L2Reference};
use tensor4all_treetci::TreeTciInterpolator;

/// Bits per variable: 2^18 points. The review that found the limitation used
/// 7 bits; 6 is the smallest size that shows both kinds of miss.
const BITS: usize = 6;
/// Width of the ridge, about 1.3 grid spacings at 6 bits.
const WIDTH: f64 = 0.02;
const CAP: usize = 16;
const RTOL: f64 = 1e-4;
/// Seeds that miss at this configuration. Seeds 3 and 4 each accept a patch
/// in which the engine never sampled the feature (rank 1); seeds 2 and 4
/// accept patches in which the engine saw it and its own estimate was below
/// `tau`.
const SEEDS: [u64; 3] = [2, 3, 4];

/// Three quantics variables `x`, `y`, `z` of `bits` bits each on the three
/// branches of a site-free root `r` (degree three), one binary site per node.
fn ridge_tree(bits: usize) -> Problem {
    let mut nodes: Vec<(String, Vec<usize>)> = vec![("r".into(), vec![])];
    let mut edges: Vec<(String, String)> = Vec::new();
    for v in ["x", "y", "z"] {
        for k in 0..bits {
            nodes.push((format!("{v}{k}"), vec![2]));
            let parent = if k == 0 {
                "r".to_string()
            } else {
                format!("{v}{}", k - 1)
            };
            edges.push((parent, format!("{v}{k}")));
        }
    }
    let nodes: Vec<(&str, &[usize])> = nodes
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();
    let edges: Vec<(&str, &str)> = edges
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    Problem::new(&nodes, &edges)
}

/// Most significant bits first, interleaved `x, y, z`.
fn msb_order(problem: &Problem, bits: usize) -> Vec<DynIndex> {
    let mut order = Vec::new();
    for k in 0..bits {
        for v in ["x", "y", "z"] {
            order.push(problem.site(&format!("{v}{k}"), 0));
        }
    }
    order
}

/// Run the ridge at every seed of [`SEEDS`] and return the seeds whose true
/// error exceeds ten times the allowance, with what the run reported.
fn corner_failures(cache_candidates: bool) -> Vec<(u64, f64, String)> {
    let problem = ridge_tree(BITS);
    let positions: Vec<[usize; 3]> = (0..BITS)
        .map(|k| ["x", "y", "z"].map(|v| problem.position(&problem.site(&format!("{v}{k}"), 0))))
        .collect();
    // A Gaussian ridge along x = y = z: f = exp(-d^2 / (2 w^2)), with d the
    // distance from the diagonal of the unit cube.
    let f = move |p: &[usize]| -> f64 {
        let mut c = [0.0_f64; 3];
        for (k, axes) in positions.iter().enumerate() {
            for (a, &position) in axes.iter().enumerate() {
                c[a] += p[position] as f64 * 0.5_f64.powi(k as i32 + 1);
            }
        }
        let s = c[0] + c[1] + c[2];
        let d2 = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - s * s / 3.0).max(0.0);
        (-d2 / (2.0 * WIDTH * WIDTH)).exp()
    };
    let (reference, norm) = dense_reference(&problem, &f);
    let points = full_domain(&problem.dims());
    let mut failures = Vec::new();
    for seed in SEEDS {
        // Default verification: 64 samples, audit on.
        let options = PatchedInterpolationOptions::new(CAP)
            .with_error_norm(ErrorNorm::l2(L2Reference::Given(norm)))
            .with_tolerance(tol(RTOL))
            .with_patch_order(msb_order(&problem, BITS))
            .with_cache_candidates(cache_candidates)
            .with_seed(seed);
        let start = std::time::Instant::now();
        let result = run(&TreeTciInterpolator::default(), &problem, &f, &[], &options).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let report = &result.report;
        let NormReport::L2 { tau, ref error, .. } = report.norm else {
            panic!("an L2 run reports an L2 error");
        };
        let delta = report.norm.delta().unwrap();
        let reported = match &error.global {
            GlobalL2Error::Audited {
                mean_square_rel_std_error,
                ..
            } => format!(
                "audited E/||f|| {:.2e}, rel. SE of E^2 {mean_square_rel_std_error:.2}",
                error.error_norm().unwrap() / norm
            ),
            other => format!("{other:?}"),
        };
        // Materialize once and subtract; the residual is then laid out in the
        // column-major order of `full_domain`.
        let dense = result
            .partition
            .to_treetn()
            .unwrap()
            .contract_to_tensor()
            .unwrap();
        let residual: Vec<f64> = reference
            .sub(&dense)
            .unwrap()
            .permute_indices(&problem.sites)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        let true_error = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        eprintln!(
            "seed {seed}: {elapsed:.1}s, {} patches, \
             true E/||f|| {:.2e} (E/delta {:.1e}), {reported}",
            report.accepted.len(),
            true_error / norm,
            true_error / delta,
        );
        // The sampled patches whose true RMS residual exceeds tau.
        for record in &report.accepted {
            let acceptance = record.acceptance.as_ref().unwrap();
            if acceptance.method != MeasurementMethod::Sampled {
                continue;
            }
            let fixed: Vec<(usize, usize)> = record
                .projector
                .iter()
                .map(|(site, &value)| (problem.position(site), value))
                .collect();
            let (mut squares, mut count) = (0.0, 0usize);
            for (point, r) in points.iter().zip(&residual) {
                if fixed
                    .iter()
                    .all(|&(position, value)| point[position] == value)
                {
                    squares += r * r;
                    count += 1;
                }
            }
            let rms = (squares / count as f64).sqrt();
            if rms > tau {
                let audit = record.audit.as_ref().map_or(f64::NAN, |a| a.rms);
                eprintln!(
                    "   missed patch: {} fixed sites, rank {}, engine error/tau {:.1e}, \
                     max sample/tau {:.1e}, true rms/tau {:.1e}, acceptance/tau {:.1e}, \
                     audit/tau {:.1e}",
                    record.projector.len(),
                    record.max_bond_dim,
                    record.engine_error_estimate / tau,
                    record.max_sample_magnitude / tau,
                    rms / tau,
                    acceptance.rms / tau,
                    audit / tau,
                );
            }
        }
        // A sampled run is an estimate, not a bound, so allow a factor of 10
        // over the allowance; the misses at these seeds exceed it about
        // 700-900 times.
        if true_error > 10.0 * delta {
            failures.push((seed, true_error / norm, reported));
        }
    }
    failures
}

#[test]
#[ignore = "known limitation: corner-localized features missed by sampled acceptance; fix deferred to post-M9 global review"]
fn corner_localized_ridge_is_not_missed_by_sampled_acceptance() {
    let failures = corner_failures(false);
    assert!(
        failures.is_empty(),
        "corner-localized misses (seed, true E/||f||, reported): {failures:?}"
    );
}

#[test]
#[ignore = "measurement: cache candidates remove the never-sampled misses but not the sampled-acceptance misses; fails while those exist (release build)"]
fn cache_candidates_against_corner_localized_misses() {
    let failures = corner_failures(true);
    assert!(
        failures.is_empty(),
        "corner-localized misses with cache candidates (seed, true E/||f||, reported): \
         {failures:?}"
    );
}
