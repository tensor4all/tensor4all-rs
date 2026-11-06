//! Fit environment-cache benchmark for #735.
//!
//! `contract_fit` walks every directed edge of the tree once per sweep and
//! repeatedly reads the environment cache (`FitEnvironment`/`SumFitEnvironment`).
//! This fixture is a branched star whose degree-three centre has several
//! neighbours, so the per-sweep environment traffic exercises both the
//! tuple-key clone-for-lookup and the sorted-neighbour order that #735
//! restructures. A and B share one contracted site per node and carry their
//! own private sites, so the fitted result keeps all private sites external.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{contract_fit, FitOptions, TreeTN};

/// Deterministic, non-degenerate sample values.
fn sample_values(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|k| ((k as f64 + 1.0) * seed).sin() + 0.05 * k as f64)
        .collect()
}

fn dense(indices: Vec<DynIndex>, seed: f64) -> IdxTensor {
    let len = indices.iter().map(|index| index.dim()).product();
    IdxTensor::from_dense(indices, sample_values(len, seed)).unwrap()
}

/// A star with a degree-`n_leaves` centre `0` and leaves `1..=n_leaves`.
///
/// Both A and B carry the shared contracted site plus a private site at every
/// node, with separate bond dimensions `chi`.
fn star_pair(
    n_leaves: usize,
    chi: usize,
    phys: usize,
) -> (TreeTN<IdxTensor, usize>, TreeTN<IdxTensor, usize>) {
    let n_nodes = n_leaves + 1;
    let shared: Vec<DynIndex> = (0..n_nodes).map(|_| DynIndex::new_dyn(phys)).collect();
    let a_site: Vec<DynIndex> = (0..n_nodes).map(|_| DynIndex::new_dyn(phys)).collect();
    let b_site: Vec<DynIndex> = (0..n_nodes).map(|_| DynIndex::new_dyn(phys)).collect();
    let la: Vec<DynIndex> = (0..n_leaves).map(|_| DynIndex::new_dyn(chi)).collect();
    let lb: Vec<DynIndex> = (0..n_leaves).map(|_| DynIndex::new_dyn(chi)).collect();

    let mut ta = Vec::with_capacity(n_nodes);
    let mut tb = Vec::with_capacity(n_nodes);

    let mut center_a: Vec<DynIndex> = la.to_vec();
    center_a.push(shared[0].clone());
    center_a.push(a_site[0].clone());
    let mut center_b: Vec<DynIndex> = lb.to_vec();
    center_b.push(shared[0].clone());
    center_b.push(b_site[0].clone());
    ta.push(dense(center_a, 0.7));
    tb.push(dense(center_b, 0.9));

    for leaf in 0..n_leaves {
        ta.push(dense(
            vec![
                la[leaf].clone(),
                shared[leaf + 1].clone(),
                a_site[leaf + 1].clone(),
            ],
            0.3 + leaf as f64,
        ));
        tb.push(dense(
            vec![
                lb[leaf].clone(),
                shared[leaf + 1].clone(),
                b_site[leaf + 1].clone(),
            ],
            1.1 + leaf as f64,
        ));
    }

    let names: Vec<usize> = (0..n_nodes).collect();
    (
        TreeTN::from_tensors(ta, names.clone()).unwrap(),
        TreeTN::from_tensors(tb, names).unwrap(),
    )
}

/// Paired A/B fixture for #735: the same branched topology, bond dimensions,
/// sweep count, and centre are used for both the pre- and post-restructure
/// measurement. `nfullsweeps = 2` forces the environment cache through miss
/// and invalidation cycles instead of stopping at the exact zip-up result.
fn bench_fit_environment(c: &mut Criterion) {
    let mut group = c.benchmark_group("treetn_fit_environment");
    group.sample_size(10);
    for (n_leaves, chi) in [(3usize, 4usize), (3, 8), (5, 8)] {
        let (a, b) = star_pair(n_leaves, chi, 2);
        let expected = a.contract_naive(&b).unwrap();
        let fitted = contract_fit(
            &a,
            &b,
            &0usize,
            FitOptions::new(2).with_convergence_tol(0.0),
        )
        .unwrap();
        assert!(
            fitted.to_dense().unwrap().distance(&expected).unwrap() < 1e-8,
            "benchmark fixture must be numerically exact"
        );
        group.bench_with_input(
            BenchmarkId::new("contract_fit", format!("l{n_leaves}_chi{chi}")),
            &(),
            |bencher, _| {
                bencher.iter(|| {
                    contract_fit(
                        black_box(&a),
                        black_box(&b),
                        &0usize,
                        FitOptions::new(2).with_convergence_tol(0.0),
                    )
                    .unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fit_environment);
criterion_main!(benches);
