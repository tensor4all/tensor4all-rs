use super::{
    cached_batched_readout, find_global_pivots, search_with_readout, ScalarParts, SearchParams,
};
use crate::{
    materialize::to_treetn, optimize_with_proposer, DefaultProposer, GlobalIndexBatch, TreeTCI2,
    TreeTciEdge, TreeTciGraph, TreeTciOptions,
};
use anyhow::Result;
use num_complex::{Complex32, Complex64};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cell::RefCell;
use tensor4all_core::{
    AnyScalar, ColMajorArrayRef, DynIndex, IdxTensor, MatrixLuciScalar as Scalar, TensorElement,
};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::TreeTN;

/// Batch evaluator for a two-peak function on a 10-site chain.
///
/// f(idx) = 3 * exp(-beta * d2(idx, A)) + 2 * exp(-beta * d2(idx, B)) with
/// d2(x, y) = |x - y|^2. Each single Gaussian is exactly rank 1, so the
/// full function needs rank 2 — but only if both peaks are sampled. A sweep
/// seeded only at A walks toward B one coordinate at a time, and the walk
/// stalls once intermediate |f| drops below the absolute tolerance, so the
/// near-degenerate second peak is silently lost (the gw-rs defect).
const N: usize = 10;
const PEAK_A: [usize; N] = [0; N];
const PEAK_B: [usize; N] = [3; N];
const BETA: f64 = 0.5;

fn peak_value(index: &[usize]) -> f64 {
    let d2 = |peak: &[usize; N]| -> f64 {
        (0..N)
            .map(|i| {
                let d = index[i] as f64 - peak[i] as f64;
                d * d
            })
            .sum()
    };
    3.0 * (-BETA * d2(&PEAK_A)).exp() + 2.0 * (-BETA * d2(&PEAK_B)).exp()
}

fn evaluate(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
    let mut values = Vec::with_capacity(batch.n_points());
    for point in 0..batch.n_points() {
        let mut index = [0usize; N];
        for (site, slot) in index.iter_mut().enumerate() {
            *slot = batch.get(site, point).unwrap();
        }
        values.push(peak_value(&index));
    }
    Ok(values)
}

fn chain_graph() -> TreeTciGraph {
    let edges: Vec<TreeTciEdge> = (0..N - 1).map(|i| TreeTciEdge::new(i, i + 1)).collect();
    TreeTciGraph::new(N, &edges).unwrap()
}

fn options(enable_global_pivots: bool) -> TreeTciOptions {
    TreeTciOptions {
        tolerance: 1e-8,
        max_iter: 30,
        max_bond_dim: None,
        normalize_error: true,
        enable_global_pivots,
        nsearch: 10,
        max_nglobal_pivot: 5,
        tol_margin_global_search: 1.0,
        seed: Some(42),
    }
}

fn seeded_state() -> TreeTCI2<f64> {
    let mut tci = TreeTCI2::<f64>::new(vec![4; N], chain_graph()).unwrap();
    // Seed the sweep in the first basin only.
    tci.add_global_pivots(&[PEAK_A.to_vec()]).unwrap();
    let flat: Vec<usize> = PEAK_A.to_vec();
    let init_batch = GlobalIndexBatch::new(&flat, N, 1).unwrap();
    let init_values = evaluate(init_batch).unwrap();
    tci.max_sample_value = init_values.iter().copied().fold(0.0f64, f64::max);
    tci
}

/// Evaluate the materialized tree at a full-site index.
fn tree_value(tree: &TreeTN<IdxTensor, usize>, index: &[usize]) -> f64 {
    let mut site_indices = Vec::with_capacity(index.len());
    for site in 0..index.len() {
        let node = tree.node_index(&site).unwrap();
        let tensor = tree.tensor(node).unwrap();
        site_indices.push(tensor.indices()[0].clone());
    }
    tree.evaluate_point(&site_indices, index).unwrap().real()
}

fn run(enable_global_pivots: bool) -> (Vec<usize>, Vec<f64>, f64, f64) {
    let mut tci = seeded_state();
    let opts = options(enable_global_pivots);
    let (ranks, errors) =
        optimize_with_proposer(&mut tci, evaluate, &opts, &DefaultProposer).unwrap();
    let tree = to_treetn(&tci, evaluate, None).unwrap();
    (
        ranks,
        errors,
        tree_value(&tree, &PEAK_A),
        tree_value(&tree, &PEAK_B),
    )
}

#[test]
fn global_pivot_search_finds_pivots_with_large_error() {
    // Direct finder check: with a rank-1 state seeded only at peak A, the
    // search must escape the A basin and return points near peak B.
    let tci = seeded_state();
    let pivots =
        find_global_pivots(&tci, evaluate, 10, 5, 1.0, 1e-8 * tci.max_sample_value, 42).unwrap();

    let closer_to_b = |pivot: &Vec<usize>| {
        let d2 = |peak: &[usize; N]| {
            (0..N)
                .map(|i| {
                    let d = pivot[i] as i64 - peak[i] as i64;
                    d * d
                })
                .sum::<i64>()
        };
        d2(&PEAK_B) < d2(&PEAK_A)
    };
    assert!(
        !pivots.is_empty() && pivots.iter().all(closer_to_b),
        "global pivot search must escape the seeded basin, got {pivots:?}"
    );
}

#[test]
fn global_pivots_capture_both_near_degenerate_basins() {
    // Regression for the gw-rs defect (lingrui96/gw-rs#9): with initial
    // pivots confined to one basin, the sweep self-reports convergence while
    // a whole near-degenerate peak silently vanishes. Enabling the global
    // pivot search must recover both peaks.
    let (ranks, errors, value_a, value_b) = run(true);

    assert!(
        errors.last().copied().unwrap_or(1.0) < 1e-6,
        "expected convergence with global pivots, errors={errors:?}"
    );
    assert!(
        (value_a - peak_value(&PEAK_A)).abs() < 1e-6,
        "first basin lost: tree value at A is {value_a}, expected {}",
        peak_value(&PEAK_A)
    );
    assert!(
        (value_b - peak_value(&PEAK_B)).abs() < 1e-6,
        "second basin lost: tree value at B is {value_b}, expected {}",
        peak_value(&PEAK_B)
    );
    assert!(*ranks.last().unwrap() >= 2, "both peaks need rank >= 2");
}

#[test]
fn without_global_pivots_second_basin_is_lost() {
    // Documents why the option exists: the same instance without the global
    // pivot search converges (error estimate near peak A) but the materialized
    // tree evaluates to ~zero at peak B, where f(B) = 2.
    let (_, errors, value_a, value_b) = run(false);

    assert!(errors.last().copied().unwrap_or(1.0) < 1e-6);
    assert!((value_a - peak_value(&PEAK_A)).abs() < 1e-6);
    assert!(
        (value_b - peak_value(&PEAK_B)).abs() > 1.0,
        "expected the second basin to be missed without global pivots, tree value at B is {value_b}"
    );
}

#[test]
fn find_global_pivots_rejects_invalid_parameters() {
    let tci = seeded_state();
    // Non-finite or negative tolerances are rejected.
    assert!(find_global_pivots(&tci, evaluate, 5, 5, 1.0, f64::NAN, 1).is_err());
    assert!(find_global_pivots(&tci, evaluate, 5, 5, -1.0, 1e-8, 1).is_err());
    assert!(find_global_pivots(&tci, evaluate, 5, 5, 1.0, -1.0, 1).is_err());
    // A disabled search (no starting points or no pivot budget) is a no-op.
    assert!(find_global_pivots(&tci, evaluate, 0, 5, 1.0, 1e-8, 1)
        .unwrap()
        .is_empty());
    assert!(find_global_pivots(&tci, evaluate, 5, 0, 1.0, 1e-8, 1)
        .unwrap()
        .is_empty());
}

#[test]
fn find_global_pivots_rejects_bad_batch_length() {
    let tci = seeded_state();
    // Evaluator returns one value regardless of the requested batch size.
    let bad = |_batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![0.0]) };
    assert!(find_global_pivots(&tci, bad, 5, 5, 1.0, 1e-8, 1).is_err());
}

#[test]
fn find_global_pivots_respects_threshold_and_limit() {
    let tci = seeded_state();
    // A huge absolute tolerance rejects every candidate.
    let none = find_global_pivots(&tci, evaluate, 10, 5, 1.0, 1e10, 1).unwrap();
    assert!(none.is_empty());
    // A strict pivot budget truncates the found candidates.
    let few =
        find_global_pivots(&tci, evaluate, 10, 2, 1.0, 1e-8 * tci.max_sample_value, 1).unwrap();
    assert!(!few.is_empty());
    assert!(few.len() <= 2);
}

#[test]
fn find_global_pivots_supports_complex_scalars() {
    let graph = chain_graph();
    let mut tci = TreeTCI2::<Complex64>::new(vec![4; N], graph).unwrap();
    tci.add_global_pivots(&[PEAK_A.to_vec()]).unwrap();

    // Nonzero complex function: mixture + i * mixture / 2.
    let complex_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<Complex64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let mut index = [0usize; N];
            for (site, slot) in index.iter_mut().enumerate() {
                *slot = batch.get(site, point).unwrap();
            }
            let v = peak_value(&index);
            values.push(Complex64::new(v, 0.5 * v));
        }
        Ok(values)
    };

    let pivots = find_global_pivots(&tci, complex_eval, 10, 5, 1.0, 1e-8, 1).unwrap();
    assert!(!pivots.is_empty());
}

// ---------------------------------------------------------------------------
// Batched readout regression (issue #792).
// ---------------------------------------------------------------------------

/// Quantics value of binary digits (most significant first) in `[0, 1)`.
fn quantics(bits: &[usize]) -> f64 {
    bits.iter()
        .enumerate()
        .map(|(k, &b)| b as f64 * 0.5f64.powi(k as i32 + 1))
        .sum()
}

/// A TreeTCI problem: local dimensions, topology and a pointwise target.
struct Fixture {
    local_dims: Vec<usize>,
    edges: Vec<TreeTciEdge>,
    target: fn(&[usize]) -> f64,
}

impl Fixture {
    fn graph(&self) -> TreeTciGraph {
        TreeTciGraph::new(self.local_dims.len(), &self.edges).unwrap()
    }

    fn evaluate(&self) -> impl Fn(GlobalIndexBatch<'_>) -> Result<Vec<f64>> + '_ {
        // Column-major batch: each column of `n_sites` entries is one point.
        move |batch: GlobalIndexBatch<'_>| {
            Ok(batch
                .data()
                .chunks(batch.n_sites())
                .map(self.target)
                .collect())
        }
    }

    fn seeded_state(&self) -> TreeTCI2<f64> {
        let origin = vec![0usize; self.local_dims.len()];
        let mut tci = TreeTCI2::<f64>::new(self.local_dims.clone(), self.graph()).unwrap();
        tci.add_global_pivots(std::slice::from_ref(&origin))
            .unwrap();
        tci.max_sample_value = (self.target)(&origin).abs();
        tci
    }

    /// State seeded at the all-zero point and swept once without a global
    /// search, so the approximation is partially converged (rank > 1 with a
    /// remaining error the search must find).
    fn one_sweep_state(&self) -> TreeTCI2<f64> {
        let mut tci = self.seeded_state();
        let options = TreeTciOptions {
            max_iter: 1,
            enable_global_pivots: false,
            ..TreeTciOptions::default()
        };
        optimize_with_proposer(&mut tci, self.evaluate(), &options, &DefaultProposer).unwrap();
        tci
    }

    /// Full default optimization (global search on, fixed seed).
    fn full_run(&self) -> (Vec<usize>, Vec<f64>) {
        let mut tci = self.seeded_state();
        let options = TreeTciOptions {
            seed: Some(1),
            ..TreeTciOptions::default()
        };
        optimize_with_proposer(&mut tci, self.evaluate(), &options, &DefaultProposer).unwrap()
    }
}

fn quantics_chain_target(point: &[usize]) -> f64 {
    let x = quantics(point);
    (-3.0 * x).exp() * (40.0 * x).cos() + 1.0 / (1.0 + 25.0 * (x - 0.4).powi(2))
}

/// 16-site binary quantics chain.
fn chain_fixture() -> Fixture {
    Fixture {
        local_dims: vec![2; 16],
        edges: (0..15).map(|i| TreeTciEdge::new(i, i + 1)).collect(),
        target: quantics_chain_target,
    }
}

const ARM: usize = 4;

fn branched_target(point: &[usize]) -> f64 {
    let x = quantics(&point[1..1 + ARM]);
    let y = quantics(&point[1 + ARM..1 + 2 * ARM]);
    let z = quantics(&point[1 + 2 * ARM..1 + 3 * ARM]);
    let flag = point[0] as f64;
    // Couples all three arms through the centre, so the rank is not reached
    // in one sweep and the search has a residual error to find.
    1.0 / (1.0 + 20.0 * (x - y).powi(2) + 20.0 * (y - z).powi(2) + 5.0 * flag * x)
}

/// Centre site 0 (degree 3) with three binary arms of `ARM` sites each.
fn branched_fixture() -> Fixture {
    let mut edges = Vec::new();
    for arm in 0..3 {
        let base = 1 + arm * ARM;
        edges.push(TreeTciEdge::new(0, base));
        for k in 0..ARM - 1 {
            edges.push(TreeTciEdge::new(base + k, base + k + 1));
        }
    }
    let fixture = Fixture {
        local_dims: vec![2; 1 + 3 * ARM],
        edges,
        target: branched_target,
    };
    // A path is a chain: the regression must cover a genuine branch point.
    assert_eq!(fixture.graph().neighbors(0).unwrap().len(), 3);
    fixture
}

fn site_indices(treetn: &TreeTN<IdxTensor, usize>) -> Vec<DynIndex> {
    (0..treetn.node_count())
        .map(|site| {
            let node = treetn.node_index(&site).unwrap();
            treetn.tensor(node).unwrap().indices()[0].clone()
        })
        .collect()
}

/// The readout the search used before issue #792: `TreeTN::evaluate`, which
/// contracts the whole network once per candidate. Kept as the reference.
fn pointwise_readout(
    treetn: &TreeTN<IdxTensor, usize>,
    site_indices: &[DynIndex],
    candidates: ColMajorArrayRef<'_, usize>,
) -> Result<Vec<AnyScalar>> {
    Ok(treetn.evaluate(site_indices, candidates)?)
}

fn search_params(state: &TreeTCI2<f64>, seed: u64) -> SearchParams {
    SearchParams {
        nsearch: 20,
        max_nglobal_pivot: 5,
        tol_margin: 1.0,
        abs_tol: 1e-8 * state.max_sample_value,
        seed,
    }
}

/// Compares the cached batched readout with `TreeTN::evaluate` on random
/// points of one materialized tree.
fn assert_readout_matches_pointwise(treetn: &TreeTN<IdxTensor, usize>, local_dims: &[usize]) {
    let n_sites = local_dims.len();
    let n_points = 300;
    let mut rng = ChaCha8Rng::seed_from_u64(792);
    // Column-major `[n_sites, n_points]`: one random point per column.
    let flat: Vec<usize> = (0..n_points)
        .flat_map(|_| {
            local_dims
                .iter()
                .map(|&d| rng.random_range(0..d))
                .collect::<Vec<_>>()
        })
        .collect();
    let shape = [n_sites, n_points];
    let points = ColMajorArrayRef::new(&flat, &shape).unwrap();
    let indices = site_indices(treetn);

    let cached = cached_batched_readout(treetn, &indices, points).unwrap();
    let reference = treetn.evaluate(&indices, points).unwrap();
    assert_eq!(cached.len(), n_points);
    let scale = reference.iter().map(AnyScalar::abs).fold(0.0f64, f64::max);
    let max_diff = cached
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a.clone() - b.clone()).abs())
        .fold(0.0f64, f64::max);
    // The two readouts contract in different orders; they agree to rounding
    // (measured below 5e-15 relative on the issue #792 workloads).
    assert!(
        max_diff <= 1e-13 * scale,
        "cached readout differs from TreeTN::evaluate by {max_diff:e} (scale {scale:e})"
    );

    // Layout: column `p` is point `p`, independent of the batched contract.
    for (point, value) in flat.chunks(n_sites).zip(&cached).take(3) {
        let single = treetn.evaluate_point(&indices, point).unwrap();
        assert!((value.clone() - single).abs() <= 1e-13 * scale);
    }

    // Fresh evaluators on this tree reproduce the same batch values bit for
    // bit within this process.
    let again = cached_batched_readout(treetn, &indices, points).unwrap();
    for (a, b) in cached.iter().zip(&again) {
        assert_eq!(a.real().to_bits(), b.real().to_bits());
        assert_eq!(a.imag().to_bits(), b.imag().to_bits());
    }
}

#[test]
fn cached_readout_matches_pointwise_evaluate_on_chain_and_branched_tree() {
    for fixture in [chain_fixture(), branched_fixture()] {
        let state = fixture.one_sweep_state();
        // Materialize once; both readouts evaluate the same tree.
        let treetn = to_treetn(&state, fixture.evaluate(), None).unwrap();
        assert_readout_matches_pointwise(&treetn, &fixture.local_dims);
    }
}

#[test]
fn cached_readout_matches_pointwise_evaluate_for_complex_scalars() {
    let fixture = branched_fixture();
    let n_sites = fixture.local_dims.len();
    let phase = Complex64::new(1.0, 0.5);
    let complex_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<Complex64>> {
        Ok(batch
            .data()
            .chunks(batch.n_sites())
            .map(|point| phase * branched_target(point))
            .collect())
    };
    let origin = vec![0usize; n_sites];
    let mut tci = TreeTCI2::<Complex64>::new(fixture.local_dims.clone(), fixture.graph()).unwrap();
    tci.add_global_pivots(std::slice::from_ref(&origin))
        .unwrap();
    tci.max_sample_value = (phase * branched_target(&origin)).norm();
    let options = TreeTciOptions {
        max_iter: 1,
        enable_global_pivots: false,
        ..TreeTciOptions::default()
    };
    optimize_with_proposer(&mut tci, complex_eval, &options, &DefaultProposer).unwrap();
    let treetn = to_treetn(&tci, complex_eval, None).unwrap();
    assert_readout_matches_pointwise(&treetn, &fixture.local_dims);
}

/// Exercises TreeTCI's materialize-and-search route for the two scalar kinds
/// that use the cached evaluator's generic contraction path. The raw-message
/// kernels are intentionally limited to f64/c64, so these trees must take the
/// `IdxTensor` fallback even though `to_treetn` creates one physical site per
/// node.
fn assert_32bit_global_search_uses_generic_readout<T, F>(evaluate: F)
where
    T: FullPivLuScalar + Scalar + TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>> + Copy,
{
    let fixture = chain_fixture();
    let n_sites = fixture.local_dims.len();
    let origin = vec![0usize; n_sites];
    let mut state = TreeTCI2::<T>::new(fixture.local_dims.clone(), fixture.graph()).unwrap();
    state
        .add_global_pivots(std::slice::from_ref(&origin))
        .unwrap();
    state.max_sample_value = 1.0;

    let treetn = to_treetn(&state, evaluate, None).unwrap();
    let indices = site_indices(&treetn);
    let n_points = 8;
    let local_dims = &fixture.local_dims;
    let flat: Vec<usize> = (0..n_points)
        .flat_map(|point| (0..n_sites).map(move |site| (point * 7 + site * 3) % local_dims[site]))
        .collect();
    let shape = [n_sites, n_points];
    let candidates = ColMajorArrayRef::new(&flat, &shape).unwrap();
    let cached = cached_batched_readout(&treetn, &indices, candidates).unwrap();
    let reference = treetn.evaluate(&indices, candidates).unwrap();
    let scale = reference.iter().map(AnyScalar::abs).fold(0.0f64, f64::max);
    let max_diff = cached
        .iter()
        .zip(&reference)
        .map(|(cached, pointwise)| (cached.clone() - pointwise.clone()).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_diff <= 1e-5 * scale.max(1.0),
        "32-bit cached readout differs from TreeTN::evaluate by {max_diff:e} (scale {scale:e})"
    );

    // Exercise the supported public search entry point too: it builds the
    // same cached evaluator internally while comparing target and TT values.
    let pivots = find_global_pivots(&state, evaluate, 8, 4, 1.0, 1e-6, 792).unwrap();
    assert!(
        !pivots.is_empty(),
        "the generic 32-bit readout found no pivots"
    );
}

#[test]
fn global_search_supports_f32_and_complex32_cached_fallback() {
    let real_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f32>> {
        Ok(batch
            .data()
            .chunks(batch.n_sites())
            .map(|point| quantics_chain_target(point) as f32)
            .collect())
    };
    assert_32bit_global_search_uses_generic_readout(real_eval);

    let phase = Complex32::new(1.0, 0.5);
    let complex_eval = move |batch: GlobalIndexBatch<'_>| -> Result<Vec<Complex32>> {
        Ok(batch
            .data()
            .chunks(batch.n_sites())
            .map(|point| phase * quantics_chain_target(point) as f32)
            .collect())
    };
    assert_32bit_global_search_uses_generic_readout(complex_eval);
}

#[test]
fn cached_readout_rejects_mismatched_indices_and_shapes() {
    let fixture = chain_fixture();
    let state = fixture.one_sweep_state();
    let treetn = to_treetn(&state, fixture.evaluate(), None).unwrap();
    let indices = site_indices(&treetn);
    let flat = vec![0usize; indices.len()];

    // Indices that are not the tree's site indices.
    let foreign: Vec<DynIndex> = indices.iter().map(|_| DynIndex::new_dyn(2)).collect();
    let shape = [indices.len(), 1];
    let points = ColMajorArrayRef::new(&flat, &shape).unwrap();
    assert!(cached_batched_readout(&treetn, &foreign, points).is_err());

    // A batch whose row count is not the number of site indices.
    let short_shape = [1, indices.len()];
    let short = ColMajorArrayRef::new(&flat, &short_shape).unwrap();
    assert!(cached_batched_readout(&treetn, &indices, short).is_err());
}

/// Pivots returned on the #793 base with `ChaCha8Rng` and pointwise readout
/// for the one-sweep states of the two fixtures.
fn recorded_pivots(fixture: &str, seed: u64) -> Vec<Vec<usize>> {
    match (fixture, seed) {
        ("chain", 3) => vec![
            vec![0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            vec![0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
            vec![1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
            vec![0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        ],
        ("chain", 11) => vec![
            vec![0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            vec![1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            vec![0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
            vec![0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        ],
        ("branched", 3) => vec![
            vec![0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            vec![0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            vec![0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
            vec![0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
            vec![0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        ],
        ("branched", 11) => vec![
            vec![0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            vec![0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            vec![0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
            vec![1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        ],
        _ => unreachable!("no recording for {fixture} seed {seed}"),
    }
}

#[test]
fn global_search_pivots_match_pointwise_reference_on_chain_and_branched_tree() {
    for (name, fixture) in [("chain", chain_fixture()), ("branched", branched_fixture())] {
        let state = fixture.one_sweep_state();
        for seed in [3u64, 11] {
            let params = search_params(&state, seed);
            let cached = find_global_pivots(
                &state,
                fixture.evaluate(),
                params.nsearch,
                params.max_nglobal_pivot,
                params.tol_margin,
                params.abs_tol,
                seed,
            )
            .unwrap();
            let pointwise =
                search_with_readout(&state, fixture.evaluate(), params, pointwise_readout).unwrap();
            // Same pivots, in the same order, as the pointwise readout and
            // the fixed-seed recording for this base.
            assert_eq!(cached, pointwise, "{name} seed {seed}");
            assert_eq!(cached, recorded_pivots(name, seed), "{name} seed {seed}");
        }
    }
}

#[test]
fn full_runs_reproduce_recorded_ranks_and_errors() {
    // Recorded on the #793 base (9ad67f2c) with `ChaCha8Rng` and the
    // pointwise readout: default options, seed 1, global search enabled.
    let (ranks, errors) = chain_fixture().full_run();
    assert_eq!(ranks, vec![4, 8, 8, 8]);
    let recorded = [
        4.210785455368414e-9,
        8.68318251630876e-9,
        5.771110563560244e-9,
        6.057293820959278e-9,
    ];
    assert_eq!(errors.len(), recorded.len());
    for (error, expected) in errors.iter().zip(recorded) {
        assert!(
            (error - expected).abs() <= 1e-9 * expected,
            "chain errors {errors:?}, recorded {recorded:?}"
        );
    }

    let (ranks, errors) = branched_fixture().full_run();
    assert_eq!(ranks, vec![4, 16, 16, 16]);
    // The branched fixture is represented exactly at rank 16.
    assert_eq!(errors.len(), 4);
    assert!(errors.iter().all(|&error| error <= 1e-14), "{errors:?}");
}

#[test]
fn exact_error_ties_keep_the_first_generated_candidate() {
    // f = 2 everywhere and a readout of exactly 1 give every candidate the
    // same error, 1. The first candidate of each start must win, starts must
    // stay in generation order, and the budget truncates that order.
    let fixture = chain_fixture();
    let n_sites = fixture.local_dims.len();
    let per_start: usize = fixture.local_dims.iter().sum();
    let constant =
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![2.0; batch.n_points()]) };
    let mut state = TreeTCI2::<f64>::new(fixture.local_dims.clone(), fixture.graph()).unwrap();
    state.add_global_pivots(&[vec![0; n_sites]]).unwrap();
    state.max_sample_value = 2.0;

    let seen = RefCell::new(Vec::new());
    let tied_readout = |_: &TreeTN<IdxTensor, usize>,
                        _: &[DynIndex],
                        candidates: ColMajorArrayRef<'_, usize>|
     -> Result<Vec<AnyScalar>> {
        seen.replace(candidates.data().to_vec());
        Ok(vec![AnyScalar::new_real(1.0); candidates.shape()[1]])
    };
    let params = SearchParams {
        nsearch: 6,
        max_nglobal_pivot: 3,
        tol_margin: 1.0,
        abs_tol: 0.5,
        seed: 5,
    };
    let pivots = search_with_readout(&state, constant, params, tied_readout).unwrap();

    let candidates = seen.into_inner();
    let mut expected: Vec<Vec<usize>> = Vec::new();
    for start in 0..params.nsearch {
        let first = start * per_start * n_sites;
        let point = candidates[first..first + n_sites].to_vec();
        // The first candidate of a start varies site 0 to its first value.
        assert_eq!(point[0], 0);
        if !expected.contains(&point) {
            expected.push(point);
        }
    }
    expected.truncate(params.max_nglobal_pivot);
    assert_eq!(pivots, expected);
}

#[test]
fn search_rejects_failing_or_short_readouts() {
    let fixture = chain_fixture();
    let state = fixture.one_sweep_state();
    let params = search_params(&state, 3);

    let failing = |_: &TreeTN<IdxTensor, usize>,
                   _: &[DynIndex],
                   _: ColMajorArrayRef<'_, usize>|
     -> Result<Vec<AnyScalar>> { Err(anyhow::anyhow!("readout failed")) };
    let error = search_with_readout(&state, fixture.evaluate(), params, failing).unwrap_err();
    assert!(error.to_string().contains("readout failed"), "{error}");

    let short = |_: &TreeTN<IdxTensor, usize>,
                 _: &[DynIndex],
                 _: ColMajorArrayRef<'_, usize>|
     -> Result<Vec<AnyScalar>> { Ok(vec![AnyScalar::new_real(0.0)]) };
    let error = search_with_readout(&state, fixture.evaluate(), params, short).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("approximation readout returned 1 values"),
        "{error}"
    );
}
