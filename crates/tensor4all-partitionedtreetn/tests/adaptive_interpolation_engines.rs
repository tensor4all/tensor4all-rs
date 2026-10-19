//! `patched_interpolate` end to end with the TreeTCI engine, and with a
//! fiber test engine on a domain wider than 128 bits.

use crate::adaptive_common;

use adaptive_common::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_partitionedtreetn::adaptive_interpolation::{
    PatchedInterpolationOptions, PatchedInterpolationResult,
};
use tensor4all_partitionedtreetn::ErrorNorm;
use tensor4all_tensorbackend::explicit::{HeldSession, Session};
use tensor4all_tensorbackend::{default_cpu_execution_context, CpuExecutionContext};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{EvalContext, Parallelism};

/// Run the driver on a held session over `context`, the canonical execution
/// capability of one run.
fn run_on(
    context: &CpuExecutionContext,
    problem: &Problem,
    f: &dyn Fn(&[usize]) -> f64,
    pivots: &[Vec<usize>],
    options: &PatchedInterpolationOptions,
) -> Result<
    PatchedInterpolationResult<Name>,
    tensor4all_partitionedtreetn::adaptive_interpolation::PatchedInterpolationError,
> {
    let backend: tenferro_cpu::CpuBackend = context.with_backend(|backend| backend.clone());
    let mut held = HeldSession::open(&backend).expect("held session");
    let n_sites = problem.sites.len();
    let result = held.with_session(|view| {
        let mut ctx = EvalContext::new(
            Session::from_backend_session(view.backend()),
            Parallelism::Serial,
        );
        tensor4all_partitionedtreetn::adaptive_interpolation::patched_interpolate(
            &TreeTciInterpolator::default(),
            problem.topology.clone(),
            problem.node_sites.clone(),
            problem.pivots(pivots),
            &mut ctx,
            |batch: ColMajorArrayRef<'_, usize>, _ctx| -> anyhow::Result<Vec<f64>> {
                Ok(batch.data().chunks(n_sites).map(f).collect())
            },
            options,
        )
    });
    held.close().expect("affinity restores");
    result
}

// ---------------------------------------------------------------------------
// TreeTCI end to end
// ---------------------------------------------------------------------------

#[test]
fn treetci_patches_a_localized_function_on_a_quantics_chain() {
    // 128 points; two narrow peaks give the whole domain a rank above the cap.
    let problem = chain("q", 7, 2);
    let f = |p: &[usize]| {
        let x = quantics(p);
        gaussian(x, 0.3, 0.02) + 0.5 * gaussian(x, 0.71, 0.05)
    };
    let rtol = 1e-8;
    let options = sampled_max(4)
        .with_tolerance(tol(rtol))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(max_abs(&problem, &f)));
    let result = run(
        &TreeTciInterpolator::default(),
        &problem,
        &f,
        &[vec![0, 1, 0, 0, 1, 1, 0]],
        &options,
    )
    .unwrap();
    assert!(result.report.splits >= 1, "the root must not converge");
    assert!(result
        .report
        .accepted
        .iter()
        .any(|record| record.max_bond_dim >= 2));
    assert_accurate(&result, &problem, &f, rtol);
}

fn quantics_tree_options(problem: &Problem, recycle: bool) -> PatchedInterpolationOptions {
    let order = ["x0", "y0", "x1", "y1", "x2", "y2", "z"]
        .iter()
        .map(|node| problem.site(node, 0))
        .collect();
    sampled_max(4)
        .with_tolerance(tol(1e-8))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(max_abs(
            problem, &tree_peak,
        )))
        .with_patch_order(order)
        .with_recycle_pivots(recycle)
        .with_seed(5)
}

#[test]
fn treetci_patches_a_function_on_a_branched_tree_deterministically() {
    let problem = quantics_tree();
    let index = problem.topology.node_index(&"r".to_string()).unwrap();
    assert_eq!(problem.topology.graph().neighbors(index).count(), 3);
    let engine = TreeTciInterpolator::default();
    let pivots = [vec![1, 0, 0, 1, 0, 0, 1]];
    let mut runs = Vec::new();
    for recycle in [false, true, true] {
        let options = quantics_tree_options(&problem, recycle);
        let result = run(&engine, &problem, &tree_peak, &pivots, &options).unwrap();
        assert!(result.report.splits >= 1, "the root must not converge");
        // Some patch is accepted from the engine on the degree-three tree,
        // not only from the exact small-patch path.
        assert!(result
            .report
            .accepted
            .iter()
            .any(|record| record.max_bond_dim >= 2));
        assert_accurate(&result, &problem, &tree_peak, options.tolerance.rtol);
        runs.push(result);
    }
    assert_same_run(&problem, &runs[1], &runs[2]);
}

#[test]
fn the_cache_supports_domains_wider_than_128_bits() {
    // Three variables of 43 bits each on a chain of 129 binary sites.
    let problem = chain("s", 129, 2);
    // exp(x + y + z) is a product over the bits.
    let f = |p: &[usize]| p.chunks(43).map(quantics).sum::<f64>().exp();
    let options = sampled_max(2).with_error_norm(ErrorNorm::sampled_max_with_reference(20.0));
    let result = run(
        &FiberEngine,
        &problem,
        &f,
        &[vec![0; 129], vec![1; 129]],
        &options,
    )
    .unwrap();
    assert_eq!(result.report.splits, 0);
    assert_eq!(result.report.accepted.len(), 1);
    // Five candidates (two user pivots, three random points), then the 129
    // new fiber points; the fibers repeat the cached pivot 129 times.
    assert_eq!(result.report.function_evaluations, 5 + 129);
    assert_eq!(result.report.cache_hits, 129);

    // Sampled comparison: materializing 2^129 values is impossible.
    let network = result.partition.to_treetn().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(129);
    for _ in 0..16 {
        let point: Vec<usize> = (0..129).map(|_| rng.random_range(0..2)).collect();
        let value = network
            .evaluate_point(&problem.sites, &point)
            .unwrap()
            .real();
        let expected = f(&point);
        assert!(
            (value - expected).abs() <= 1e-12 * expected,
            "{value} vs {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// Caller-owned context engine (issue #859 F7)
// ---------------------------------------------------------------------------

/// `patched_interpolate` gives the same result on a caller-owned CPU context
/// and on the process-global backend: same seed, same patches, dense values
/// equal up to tight tolerance, and the same accuracy contract. Both runs take
/// the canonical execution capability; only the backend behind the held
/// session differs.
#[test]
fn caller_owned_context_engine_matches_plain_engine_end_to_end() {
    let problem = chain("q", 7, 2);
    let f = |p: &[usize]| {
        let x = quantics(p);
        gaussian(x, 0.3, 0.02) + 0.5 * gaussian(x, 0.71, 0.05)
    };
    let rtol = 1e-8;
    let options = sampled_max(4)
        .with_tolerance(tol(rtol))
        .with_error_norm(ErrorNorm::sampled_max_with_reference(max_abs(&problem, &f)));
    let pivots = [vec![0, 1, 0, 0, 1, 1, 0]];

    let plain_context = default_cpu_execution_context();
    let plain = run_on(&plain_context, &problem, &f, &pivots, &options).unwrap();

    let caller_owned =
        CpuExecutionContext::from_backend(tenferro_cpu::CpuBackend::with_threads(1).unwrap());
    let contextual = run_on(&caller_owned, &problem, &f, &pivots, &options).unwrap();

    // Same patch structure on the same seed.
    assert_eq!(contextual.report.splits, plain.report.splits);

    // Dense parity over the whole domain, up to a tight relative tolerance.
    let dense = |result: &PatchedInterpolationResult<Name>| {
        let network = result.partition.to_treetn().unwrap();
        full_domain(&problem.dims())
            .iter()
            .map(|p| network.evaluate_point(&problem.sites, p).unwrap().real())
            .collect::<Vec<f64>>()
    };
    let a = dense(&plain);
    let b = dense(&contextual);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(
            (x - y).abs() <= 1e-12 * scale,
            "caller-owned engine diverged from the plain engine: |{x} - {y}|"
        );
    }

    // Both routes still meet the accuracy contract.
    assert_accurate(&plain, &problem, &f, rtol);
    assert_accurate(&contextual, &problem, &f, rtol);
}
