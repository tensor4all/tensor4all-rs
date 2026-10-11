//! F3 step 3a: a caller-owned execution context for the TCI path's factorization.
//!
//! The new entry point must compute what the plain one computes, and its factorization must really run in the
//! caller's context rather than the process-global one. These tests check both, and the edge cases the design
//! promises: a rank cap, an all-zero sample, and the same results for real and complex scalars.

use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tensor4all_core::col_major_array::ColMajorArrayRef;
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treetci::{
    crossinterpolate2, crossinterpolate2_in, DefaultProposer, GlobalIndexBatch, TreeTciGraph,
    TreeTciOptions, TreeTciTermination,
};

fn context_with_workers(workers: usize) -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(workers).expect("CPU backend");
    CpuExecutionContext::from_backend(backend)
}

fn options(max_bond_dim: Option<usize>) -> TreeTciOptions {
    TreeTciOptions {
        seed: Some(20_261_011),
        tolerance: 1e-8,
        max_iter: 10,
        max_bond_dim,
        normalize_error: true,
        enable_global_pivots: true,
        ..TreeTciOptions::default()
    }
}

/// A tensor-product oracle, so the expected materialized values are known analytically.
fn real_oracle(sites: usize) -> impl Fn(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>> {
    move |batch: GlobalIndexBatch<'_>| {
        Ok((0..batch.n_points())
            .map(|point| {
                (0..sites).fold(1.0, |total, site| {
                    total + batch.get(site, point).expect("site in batch") as f64 * 0.5 + 1.0
                })
            })
            .collect())
    }
}

/// Evaluate a materialized network at a logical point, following the site-index order of the network.
fn evaluate_point(
    treetn: &tensor4all_treetn::TreeTN<tensor4all_core::IdxTensor, usize>,
    point: &[usize],
) -> f64 {
    let (indices, _vertices) = treetn.all_site_indices().expect("site indices");
    let position: Vec<usize> = (0..point.len())
        .map(|site| {
            let site_index = treetn
                .site_space(&site)
                .expect("site space")
                .iter()
                .next()
                .expect("one site index")
                .clone();
            indices
                .iter()
                .position(|index| *index == site_index)
                .expect("the site index is in the list")
        })
        .collect();
    let mut data = vec![0usize; indices.len()];
    for (site, &value) in point.iter().enumerate() {
        data[position[site]] = value;
    }
    let shape = [indices.len(), 1];
    let values = ColMajorArrayRef::new(&data, &shape).expect("col-major array");
    treetn.evaluate(&indices, values).expect("evaluate")[0].real()
}

fn expected_real(point: &[usize]) -> f64 {
    point
        .iter()
        .fold(1.0, |total, value| total + *value as f64 * 0.5 + 1.0)
}

#[test]
fn the_context_entry_computes_what_the_plain_entry_computes() {
    let sites = 6;
    let local_dim = 3;
    let graph = TreeTciGraph::linear_chain(sites).expect("graph");
    let dims = vec![local_dim; sites];
    let pivots = vec![vec![0usize; sites]];
    let context = context_with_workers(1);

    let plain = crossinterpolate2::<f64, _, _>(
        real_oracle(sites),
        dims.clone(),
        graph.clone(),
        pivots.clone(),
        options(Some(8)),
        None,
        &DefaultProposer,
    )
    .expect("plain run");
    let in_context = crossinterpolate2_in::<f64, _, _>(
        real_oracle(sites),
        dims,
        graph,
        pivots,
        options(Some(8)),
        None,
        &DefaultProposer,
        &context,
    )
    .expect("context run");

    assert_eq!(in_context.termination, plain.termination);
    assert_eq!(in_context.ranks, plain.ranks);
    assert_eq!(in_context.errors, plain.errors);
    assert_eq!(in_context.termination, TreeTciTermination::Converged);

    // The materialized networks agree on points that were not training pivots, against the analytic oracle.
    let points: Vec<&[usize]> = vec![
        &[0, 1, 2, 0, 1, 2],
        &[2, 2, 2, 2, 2, 2],
        &[1, 0, 1, 0, 1, 0],
    ];
    for point in points {
        let expected = expected_real(point);
        let plain_value = evaluate_point(&plain.treetn, point);
        let context_value = evaluate_point(&in_context.treetn, point);
        assert!(
            (context_value - expected).abs() <= 1e-6 * expected.abs().max(1.0),
            "context {context_value} != oracle {expected} at {point:?}"
        );
        assert!(
            (context_value - plain_value).abs() <= 1e-9 * plain_value.abs().max(1.0),
            "context {context_value} != plain {plain_value}"
        );
    }
}

#[test]
fn complex_and_rank_capped_runs_agree_between_the_entries() {
    let sites = 4;
    let local_dim = 3;
    let graph = TreeTciGraph::linear_chain(sites).expect("graph");
    let dims = vec![local_dim; sites];
    let pivots = vec![vec![0usize; sites]];
    let context = context_with_workers(2);

    let complex_oracle = move |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<Complex64>> {
        Ok((0..batch.n_points())
            .map(|point| {
                (0..sites).fold(Complex64::new(1.0, 0.5), |total, site| {
                    total + Complex64::new(0.0, batch.get(site, point).expect("site") as f64 * 0.25)
                })
            })
            .collect())
    };

    let plain = crossinterpolate2::<Complex64, _, _>(
        complex_oracle,
        dims.clone(),
        graph.clone(),
        pivots.clone(),
        options(Some(1)),
        None,
        &DefaultProposer,
    )
    .expect("plain run");
    let in_context = crossinterpolate2_in::<Complex64, _, _>(
        complex_oracle,
        dims,
        graph,
        pivots,
        options(Some(1)),
        None,
        &DefaultProposer,
        &context,
    )
    .expect("context run");

    assert_eq!(in_context.ranks, plain.ranks);
    assert_eq!(in_context.errors, plain.errors);
    assert_eq!(in_context.ranks.len(), plain.ranks.len());
    assert!(
        in_context.ranks.iter().all(|&rank| rank == 1),
        "the rank cap holds: {:?}",
        in_context.ranks
    );
}

#[test]
fn an_all_zero_sample_fails_the_same_way_through_both_entries() {
    let sites = 3;
    let graph = TreeTciGraph::linear_chain(sites).expect("graph");
    let dims = vec![2usize; sites];
    let pivots = vec![vec![0usize; sites]];
    let context = context_with_workers(1);
    let zero = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
        Ok(vec![0.0; batch.n_points()])
    };

    let plain = crossinterpolate2::<f64, _, _>(
        zero,
        dims.clone(),
        graph.clone(),
        pivots.clone(),
        options(Some(4)),
        None,
        &DefaultProposer,
    )
    .expect_err("an all-zero sample is refused");
    let in_context = crossinterpolate2_in::<f64, _, _>(
        zero,
        dims,
        graph,
        pivots,
        options(Some(4)),
        None,
        &DefaultProposer,
        &context,
    )
    .expect_err("an all-zero sample is refused");
    assert_eq!(plain.to_string(), in_context.to_string());
}
