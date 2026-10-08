use super::{
    cached_walk_readout, find_global_pivots, search_with_readout, ScalarParts, SearchParams,
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
    AnyScalar, ColMajorArrayRef, CommonScalar, DynIndex, IdxTensor, MatrixLuciScalar as Scalar,
    TensorElement,
};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};

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
        evaluation_cache_bytes: None,
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
    let crate::TreeTciOptimizationResult { ranks, errors, .. } =
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

fn assert_coordinate_walk<T>(phase: T)
where
    T: FullPivLuScalar + Scalar + TensorElement + ScalarParts,
{
    for (dims, edges, a, b) in [
        (vec![4, 4], vec![TreeTciEdge::new(0, 1)], 0, 1),
        (
            vec![1, 4, 4, 2],
            vec![
                TreeTciEdge::new(0, 1),
                TreeTciEdge::new(0, 2),
                TreeTciEdge::new(0, 3),
            ],
            1,
            2,
        ),
    ] {
        let n_sites = dims.len();
        let graph = TreeTciGraph::new(n_sites, &edges).unwrap();
        let mut state = TreeTCI2::<T>::new(dims.clone(), graph).unwrap();
        state.add_global_pivots(&[vec![0; n_sites]]).unwrap();
        // Pin a start at (1, 1). Starting at (0, 0) on an i*j residual
        // gives flat zero fibers, which a greedy walk cannot escape either.
        let seed = (0..1000)
            .find(|&seed| {
                let mut rng = seeded_rng(seed);
                let start: Vec<usize> = dims.iter().map(|&d| rng.random_range(0..d)).collect();
                start[a] == 1 && start[b] == 1
            })
            .unwrap();
        for repeat_sweep in [false, true] {
            // All pivot cross-sections are 1, so the rank-one approximation
            // is phase everywhere. The residual is the table below or i*j.
            // The table walk visits (2,1), (2,3), then (3,3) next sweep.
            let table = [[1.0, 4.0, 2.0], [3.0, 5.0, 6.0], [2.0, 7.0, 9.0]];
            let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
                Ok(batch
                    .data()
                    .chunks(n_sites)
                    .map(|point| {
                        let i = point[a];
                        let j = point[b];
                        let residual = if i == 0 || j == 0 {
                            0.0
                        } else if repeat_sweep {
                            table[i - 1][j - 1]
                        } else {
                            (i * j) as f64
                        };
                        phase * T::from_f64(1.0 + residual)
                    })
                    .collect())
            };
            // Original-start axis scans reach at most 3 (i*j) or 4 (table),
            // below these thresholds for every scalar variant below.
            let threshold = if repeat_sweep { 8.0 } else { 4.0 };
            let pivots = find_global_pivots(&state, evaluate, 1, 1, 1.0, threshold, seed).unwrap();
            let mut expected = vec![0; n_sites];
            expected[a] = 3;
            expected[b] = 3;
            assert_eq!(pivots, vec![expected], "repeat_sweep={repeat_sweep}");
        }
    }
}

#[test]
fn global_search_retains_coordinate_moves_and_repeats_sweeps() {
    assert_coordinate_walk(1.0_f64);
    assert_coordinate_walk(Complex64::new(1.0, 0.5));
    assert_coordinate_walk(1.0_f32);
    assert_coordinate_walk(Complex32::new(1.0, 0.5));
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

    fn evaluate(&self) -> impl Fn(GlobalIndexBatch<'_>) -> Result<Vec<f64>> + Copy + '_ {
        self.evaluate_as(|value| value)
    }

    /// Batch evaluator returning `map(target(point))`, e.g. a 32-bit or
    /// complex version of the target.
    fn evaluate_as<T, M>(
        &self,
        map: M,
    ) -> impl Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>> + Copy + '_
    where
        M: Fn(f64) -> T + Copy + 'static,
    {
        // Column-major batch: each column of `n_sites` entries is one point.
        move |batch: GlobalIndexBatch<'_>| {
            Ok(batch
                .data()
                .chunks(batch.n_sites())
                .map(|point| map((self.target)(point)))
                .collect())
        }
    }

    /// State with the single pivot at the all-zero point.
    fn seeded_state_with<T, F>(&self, evaluate: F) -> TreeTCI2<T>
    where
        T: FullPivLuScalar + Scalar + TensorElement + ScalarParts,
        F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    {
        let n_sites = self.local_dims.len();
        let origin = vec![0usize; n_sites];
        let mut tci = TreeTCI2::<T>::new(self.local_dims.clone(), self.graph()).unwrap();
        tci.add_global_pivots(std::slice::from_ref(&origin))
            .unwrap();
        let value = evaluate(GlobalIndexBatch::new(&origin, n_sites, 1).unwrap()).unwrap()[0];
        tci.max_sample_value = value.real_part().hypot(value.imag_part());
        tci
    }

    fn seeded_state(&self) -> TreeTCI2<f64> {
        self.seeded_state_with(self.evaluate())
    }

    /// State seeded at the all-zero point and swept once without a global
    /// search, so the approximation is partially converged (rank > 1 with a
    /// remaining error the search must find).
    fn one_sweep_state_with<T, F>(&self, evaluate: F) -> TreeTCI2<T>
    where
        T: FullPivLuScalar + Scalar + CommonScalar + TensorElement + ScalarParts,
        F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    {
        let mut tci = self.seeded_state_with(&evaluate);
        let options = TreeTciOptions {
            max_iter: 1,
            enable_global_pivots: false,
            ..TreeTciOptions::default()
        };
        optimize_with_proposer(&mut tci, &evaluate, &options, &DefaultProposer).unwrap();
        tci
    }

    fn one_sweep_state(&self) -> TreeTCI2<f64> {
        self.one_sweep_state_with(self.evaluate())
    }

    /// Full default optimization (global search on, fixed seed). Returns the
    /// rank and error histories and the final state.
    fn full_run(&self) -> (Vec<usize>, Vec<f64>, TreeTCI2<f64>) {
        let mut tci = self.seeded_state();
        let options = TreeTciOptions {
            seed: Some(1),
            ..TreeTciOptions::default()
        };
        let crate::TreeTciOptimizationResult { ranks, errors, .. } =
            optimize_with_proposer(&mut tci, self.evaluate(), &options, &DefaultProposer).unwrap();
        (ranks, errors, tci)
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

fn cached_batched_readout(
    treetn: &TreeTN<IdxTensor, usize>,
    indices: &[DynIndex],
    candidates: ColMajorArrayRef<'_, usize>,
) -> Result<Vec<AnyScalar>> {
    let mut cache = TreeTNCachedEvaluator::new(treetn, indices, CachedEvaluatorOptions::default())?;
    cached_walk_readout(&mut cache, treetn, indices, candidates, None)
}

/// The readout the search used before issue #792: `TreeTN::evaluate`, which
/// contracts the whole network once per candidate. Kept as the reference.
fn pointwise_readout(
    _cache: &mut TreeTNCachedEvaluator<'_, usize>,
    treetn: &TreeTN<IdxTensor, usize>,
    site_indices: &[DynIndex],
    candidates: ColMajorArrayRef<'_, usize>,
    _scan_site: Option<usize>,
) -> Result<Vec<AnyScalar>> {
    Ok(treetn.evaluate(site_indices, candidates)?)
}

/// Parameters of the fixed-seed searches compared with the pointwise
/// reference. `nsearch` is kept small because the pointwise reference
/// contracts the whole network once per candidate, which dominates the debug
/// test time.
fn search_params<T>(state: &TreeTCI2<T>) -> SearchParams {
    SearchParams {
        nsearch: 4,
        max_nglobal_pivot: 5,
        tol_margin: 1.0,
        abs_tol: 1e-8 * state.max_sample_value,
    }
}

/// The named stream a fixed-seed search uses.
fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Compares the cached batched readout with `TreeTN::evaluate` on
/// `n_points` random points of one materialized tree. The two readouts must
/// agree within `rel_tol` times the largest `|tt|` of the batch.
fn assert_readout_matches_pointwise(
    treetn: &TreeTN<IdxTensor, usize>,
    local_dims: &[usize],
    n_points: usize,
    rel_tol: f64,
) {
    let n_sites = local_dims.len();
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
    assert!(
        max_diff <= rel_tol * scale,
        "cached readout differs from TreeTN::evaluate by {max_diff:e} (scale {scale:e})"
    );

    // Layout: column `p` is point `p`, independent of the batched contract.
    for (point, value) in flat.chunks(n_sites).zip(&cached).take(3) {
        let single = treetn.evaluate_point(&indices, point).unwrap();
        assert!((value.clone() - single).abs() <= rel_tol * scale);
    }

    // Fresh evaluators on this tree reproduce the same batch values bit for
    // bit on the same thread (the generic path used for f32/Complex32 is not
    // reproducible across threads, see issue #795).
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
        // The two readouts contract in different orders; they agree to
        // rounding (measured below 5e-15 relative on the issue #792
        // workloads).
        assert_readout_matches_pointwise(&treetn, &fixture.local_dims, 300, 1e-13);
    }
}

#[test]
fn cached_readout_matches_pointwise_evaluate_for_complex_scalars() {
    let fixture = branched_fixture();
    let complex_eval = fixture.evaluate_as(|value| Complex64::new(1.0, 0.5) * value);
    let state = fixture.one_sweep_state_with(complex_eval);
    let treetn = to_treetn(&state, complex_eval, None).unwrap();
    assert_readout_matches_pointwise(&treetn, &fixture.local_dims, 300, 1e-13);
}

/// Readout tolerance for the 32-bit trees, in units of `f32::EPSILON` times
/// the largest `|tt|` of the batch. A first-order rounding bound for one
/// readout value is (number of contractions) x (bond dimension) x eps for
/// sums without cancellation: the branched fixture has 13 nodes and its
/// one-sweep state has bond dimension at most 4, so about 52 eps. The value
/// 64 covers that bound with a small margin. Measured on the 32 random points
/// of the test: 0.57 eps for f32 and 3.1 eps for Complex32.
const READOUT_TOL_32BIT_EPS: f64 = 64.0;

/// Fixed-seed global search on a swept 32-bit state of the branched
/// fixture (centre of degree 3). The cached readout must select exactly the
/// pivots the pointwise `TreeTN::evaluate` reference selects.
///
/// f32 and Complex32 trees are expected to take the cached evaluator's
/// generic `IdxTensor` contraction path, because
/// `TreeTNCachedEvaluator::can_use_raw_messages`
/// (`tensor4all-treetn/src/treetn/cached_evaluator.rs`) accepts only the
/// `F64 | C64` scalar kinds. This test checks results only and cannot observe
/// which internal path ran.
fn assert_32bit_search_matches_pointwise_reference<T, F>(evaluate: F)
where
    T: FullPivLuScalar + Scalar + CommonScalar + TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>> + Copy,
{
    let fixture = branched_fixture();
    let state = fixture.one_sweep_state_with(evaluate);
    let treetn = to_treetn(&state, evaluate, None).unwrap();
    assert_readout_matches_pointwise(
        &treetn,
        &fixture.local_dims,
        32,
        READOUT_TOL_32BIT_EPS * f64::from(f32::EPSILON),
    );

    // Exact equality is asserted rather than a near-tie-tolerant comparison
    // because no decision of this search is close at f32 precision. A
    // one-off diagnostic on this fixture and seed (both scalar types)
    // measured, relative to the largest `|tt|`: readout differences up to
    // 1.2e-7 between the two readouts, a smallest winner/runner-up gap
    // within a start of 1.6e-2, a smallest gap between accepted pivots of
    // 6.7e-3, and a smallest distance of 3.5e-2 between a start's best
    // error and the threshold (itself 1.1e-3). If a legitimate change brings
    // a decision within 32-bit rounding, compare instead that every pivot
    // exceeds `abs_tol`, that the pivots are distinct, and that they match
    // the reference except for the documented near-tie.
    let params = SearchParams {
        abs_tol: 1e-3 * state.max_sample_value,
        ..search_params(&state)
    };
    let cached = find_global_pivots(
        &state,
        evaluate,
        params.nsearch,
        params.max_nglobal_pivot,
        params.tol_margin,
        params.abs_tol,
        3,
    )
    .unwrap();
    let mut rng = seeded_rng(3);
    let pointwise =
        search_with_readout(&state, evaluate, params, &mut rng, pointwise_readout).unwrap();
    assert!(
        !cached.is_empty(),
        "the swept 32-bit state must leave errors above the threshold"
    );
    assert_eq!(cached, pointwise);
}

#[test]
fn global_search_32bit_pivots_match_pointwise_reference_on_branched_tree() {
    let fixture = branched_fixture();
    assert_32bit_search_matches_pointwise_reference(fixture.evaluate_as(|value| value as f32));
    assert_32bit_search_matches_pointwise_reference(
        fixture.evaluate_as(|value| Complex32::new(1.0, 0.5) * value as f32),
    );
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

/// Pivots returned by the fixed-seed searches of
/// `global_search_pivots_match_pointwise_reference_on_chain_and_branched_tree`
/// on the one-sweep states of the two fixtures. Re-recorded from the
/// pointwise readout on the b1bb828d base with #812's retained-coordinate
/// walk, `ChaCha8Rng`, and `search_params` (`nsearch = 4`).
///
/// These lists lock the search trajectory on purpose: any change to the
/// starting points, the local search, the tie-breaking or the sweep that
/// produces the state changes them. Re-record them deliberately, from the
/// pointwise reference, only when such a change is intended (for example a
/// new RNG or search rule), and say so in the commit.
fn recorded_pivots(fixture: &str, seed: u64) -> Vec<Vec<usize>> {
    match (fixture, seed) {
        ("chain", 3) => vec![
            vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            vec![1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        ("branched", 3) => vec![
            vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            vec![1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            vec![1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        ("branched", 11) => vec![
            vec![0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            vec![1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        _ => unreachable!("no recording for {fixture} seed {seed}"),
    }
}

#[test]
fn global_search_pivots_match_pointwise_reference_on_chain_and_branched_tree() {
    // The degree-3 branched tree is the main target and gets two seeds; the
    // chain gets one. The pointwise reference dominates the cost of this
    // test.
    let cases = [
        ("chain", chain_fixture(), &[3u64][..]),
        ("branched", branched_fixture(), &[3u64, 11][..]),
    ];
    for (name, fixture, seeds) in cases {
        let state = fixture.one_sweep_state();
        for &seed in seeds {
            let params = search_params(&state);
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
            let mut rng = seeded_rng(seed);
            let pointwise = search_with_readout(
                &state,
                fixture.evaluate(),
                params,
                &mut rng,
                pointwise_readout,
            )
            .unwrap();
            // Same pivots, in the same order, as the pointwise readout and
            // the fixed-seed recording.
            assert_eq!(cached, pointwise, "{name} seed {seed}");
            assert_eq!(cached, recorded_pivots(name, seed), "{name} seed {seed}");
        }
    }
}

#[test]
fn full_runs_reproduce_recorded_ranks_and_errors() {
    // Re-recorded on b1bb828d with #812's retained-coordinate search:
    // default options, seed 1, global search enabled. No tolerance changed. Like
    // `recorded_pivots`, these values lock the whole optimization trajectory
    // on purpose and must be re-recorded deliberately when the search, the
    // sweep or the RNG legitimately changes.
    let (ranks, errors, _) = chain_fixture().full_run();
    assert_eq!(ranks, vec![4, 8, 8, 8]);
    let recorded = [
        4.210785455368414e-9,
        7.104994852102056e-9,
        6.760951106568071e-9,
        6.850685368779623e-9,
    ];
    assert_eq!(errors.len(), recorded.len());
    for (error, expected) in errors.iter().zip(recorded) {
        assert!(
            (error - expected).abs() <= 1e-9 * expected,
            "chain errors {errors:?}, recorded {recorded:?}"
        );
    }

    let fixture = branched_fixture();
    let (ranks, errors, state) = fixture.full_run();
    assert_eq!(ranks, vec![4, 16, 16, 16]);
    // At rank 16 the branched fixture is represented exactly, so the error
    // estimates are rounding-level and alone say little about the run.
    assert_eq!(errors.len(), 4);
    assert!(errors.iter().all(|&error| error <= 1e-14), "{errors:?}");
    // The stronger check: materialize the result once over all 2^13 points
    // and compare it with the dense target.
    let treetn = to_treetn(&state, fixture.evaluate(), None).unwrap();
    let dense = treetn.to_dense().unwrap();
    let indices = site_indices(&treetn);
    let n_points: usize = fixture.local_dims.iter().product();
    // Column-major: site 0 varies fastest.
    let target: Vec<f64> = (0..n_points)
        .map(|linear| {
            let mut rest = linear;
            let point: Vec<usize> = fixture
                .local_dims
                .iter()
                .map(|&dim| {
                    let value = rest % dim;
                    rest /= dim;
                    value
                })
                .collect();
            (fixture.target)(&point)
        })
        .collect();
    let scale = target
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    let target = IdxTensor::from_dense(indices, target).unwrap();
    let residual = dense.sub(&target).unwrap().maxabs().unwrap();
    assert!(
        residual <= 1e-12 * scale,
        "branched full run differs from the target by {residual:e} (scale {scale:e})"
    );
}

#[test]
fn equal_errors_keep_generation_order_and_duplicates_collapse() {
    // Each injected start has error level 1, 2 or 3. Site 0 has tied
    // candidates, so its lowest value wins; all other coordinates have a
    // strict local maximum at the held value. Distinct starts are retained,
    // letting us exercise stable sorting and deduplication across starts.
    let n_sites = 4;
    let graph = TreeTciGraph::linear_chain(n_sites).unwrap();
    let mut state = TreeTCI2::<f64>::new(vec![2; n_sites], graph).unwrap();
    state.add_global_pivots(&[vec![0; n_sites]]).unwrap();
    state.max_sample_value = 2.0;
    let nsearch = 48;
    let level = |start: usize| 1 + (start * 7 + start / 5) % 3;
    let constant =
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![2.0; batch.n_points()]) };
    let seen = RefCell::new(Vec::new());
    let search = |max_nglobal_pivot: usize| {
        seen.borrow_mut().clear();
        let mut start = 0;
        let mut active_level = 0.0;
        let leveled_readout = |_: &mut TreeTNCachedEvaluator<'_, usize>,
                               _: &TreeTN<IdxTensor, usize>,
                               _: &[DynIndex],
                               candidates: ColMajorArrayRef<'_, usize>,
                               site: Option<usize>|
         -> Result<Vec<AnyScalar>> {
            if site.is_none() {
                active_level = level(start) as f64;
                start += 1;
                let mut point = candidates.data().to_vec();
                point[0] = 0;
                seen.borrow_mut().push(point);
            }
            let error = if site.is_none() || site == Some(0) {
                active_level
            } else {
                0.0
            };
            Ok(vec![
                AnyScalar::new_real(2.0 - error);
                candidates.shape()[1]
            ])
        };
        let params = SearchParams {
            nsearch,
            max_nglobal_pivot,
            tol_margin: 1.0,
            abs_tol: 0.5,
        };
        let mut rng = seeded_rng(5);
        search_with_readout(&state, constant, params, &mut rng, leveled_readout).unwrap()
    };
    let pivots = search(nsearch);
    let candidates = seen.borrow().clone();
    assert_eq!(candidates.len(), nsearch);
    assert!(candidates.iter().all(|point| point[0] == 0));
    let mut expected = Vec::new();
    let mut repeats = 0;
    for error_level in (1..=3).rev() {
        let mut level_points = Vec::new();
        for start in (0..nsearch).filter(|&start| level(start) == error_level) {
            let point = candidates[start].clone();
            if level_points.contains(&point) {
                repeats += 1;
            } else {
                level_points.push(point.clone());
            }
            if !expected.contains(&point) {
                expected.push(point);
            }
        }
    }
    assert!(repeats > 0);
    let mut generation_order = Vec::new();
    for point in candidates {
        if !generation_order.contains(&point) {
            generation_order.push(point);
        }
    }
    assert_ne!(generation_order, expected);
    assert_eq!(pivots, expected);
    assert_eq!(search(3), expected[..3].to_vec());
}

#[test]
fn search_rejects_failing_or_short_readouts() {
    let fixture = chain_fixture();
    let state = fixture.one_sweep_state();
    let params = search_params(&state);

    let failing = |_: &mut TreeTNCachedEvaluator<'_, usize>,
                   _: &TreeTN<IdxTensor, usize>,
                   _: &[DynIndex],
                   _: ColMajorArrayRef<'_, usize>,
                   _: Option<usize>|
     -> Result<Vec<AnyScalar>> { Err(anyhow::anyhow!("readout failed")) };
    let mut rng = seeded_rng(3);
    let error =
        search_with_readout(&state, fixture.evaluate(), params, &mut rng, failing).unwrap_err();
    assert!(error.to_string().contains("readout failed"), "{error}");

    let short = |_: &mut TreeTNCachedEvaluator<'_, usize>,
                 _: &TreeTN<IdxTensor, usize>,
                 _: &[DynIndex],
                 _: ColMajorArrayRef<'_, usize>,
                 _: Option<usize>|
     -> Result<Vec<AnyScalar>> { Ok(Vec::new()) };
    let mut rng = seeded_rng(3);
    let error =
        search_with_readout(&state, fixture.evaluate(), params, &mut rng, short).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("approximation readout returned 0 values"),
        "{error}"
    );
}

#[test]
fn coordinate_scans_propagate_evaluator_and_readout_failures() {
    use std::cell::Cell;
    let fixture = Fixture {
        local_dims: vec![3, 3],
        edges: vec![TreeTciEdge::new(0, 1)],
        target: |_| 1.0,
    };
    let state = fixture.seeded_state();
    let params = search_params(&state);
    // Materialization and the initial-point evaluation succeed; fail the
    // first coordinate scan to cover errors inside the walk, after setup.
    for wrong_length in [false, true] {
        let started = Cell::new(false);
        let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            if started.get() {
                if wrong_length {
                    return Ok(vec![1.0; batch.n_points() + 1]);
                }
                return Err(anyhow::anyhow!("coordinate oracle failed"));
            }
            Ok(vec![1.0; batch.n_points()])
        };
        let readout = |cache: &mut TreeTNCachedEvaluator<'_, usize>,
                       tree: &TreeTN<IdxTensor, usize>,
                       indices: &[DynIndex],
                       points: ColMajorArrayRef<'_, usize>,
                       site| {
            started.set(true);
            cached_walk_readout(cache, tree, indices, points, site)
        };
        let error =
            search_with_readout(&state, evaluate, params, &mut seeded_rng(1), readout).unwrap_err();
        let message = if wrong_length {
            "batch evaluator returned 3 values for 2"
        } else {
            "coordinate oracle failed"
        };
        assert!(error.to_string().contains(message), "{error}");
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let readout = |_: &mut TreeTNCachedEvaluator<'_, usize>,
                       _: &TreeTN<IdxTensor, usize>,
                       _: &[DynIndex],
                       points: ColMajorArrayRef<'_, usize>,
                       site: Option<usize>| {
            let value = if site.is_some() { value } else { 1.0 };
            Ok(vec![AnyScalar::new_real(value); points.shape()[1]])
        };
        let error = search_with_readout(
            &state,
            fixture.evaluate(),
            params,
            &mut seeded_rng(1),
            readout,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("non-finite global pivot residual"),
            "{error}"
        );
    }
    let short = |_: &mut TreeTNCachedEvaluator<'_, usize>,
                 _: &TreeTN<IdxTensor, usize>,
                 _: &[DynIndex],
                 points: ColMajorArrayRef<'_, usize>,
                 site: Option<usize>| {
        Ok(if site.is_some() {
            Vec::new()
        } else {
            vec![AnyScalar::new_real(1.0); points.shape()[1]]
        })
    };
    let error = search_with_readout(
        &state,
        fixture.evaluate(),
        params,
        &mut seeded_rng(1),
        short,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("approximation readout returned 0 values for 2"),
        "{error}"
    );
}

#[test]
fn coordinate_walk_honors_early_stop_and_sweep_bound() {
    let fixture = Fixture {
        local_dims: vec![2, 2],
        edges: vec![TreeTciEdge::new(0, 1)],
        target: |_| 1.0,
    };
    let state = fixture.seeded_state();
    // A diagnostic readout whose error strictly increases on every call
    // never hits a local maximum, so only the two safety bounds stop it.
    for (threshold, expected_calls) in [(2.0, 21), (1e9, 201)] {
        let mut calls = 0;
        let readout = |_: &mut TreeTNCachedEvaluator<'_, usize>,
                       _: &TreeTN<IdxTensor, usize>,
                       _: &[DynIndex],
                       points: ColMajorArrayRef<'_, usize>,
                       _: Option<usize>| {
            calls += 1;
            Ok(vec![
                AnyScalar::new_real(1.0 - calls as f64);
                points.shape()[1]
            ])
        };
        let params = SearchParams {
            nsearch: 1,
            max_nglobal_pivot: 1,
            tol_margin: 1.0,
            abs_tol: threshold,
        };
        let pivots = search_with_readout(
            &state,
            fixture.evaluate(),
            params,
            &mut seeded_rng(1),
            readout,
        )
        .unwrap();
        assert_eq!(calls, expected_calls);
        assert_eq!(pivots.len(), usize::from(threshold == 2.0));
    }
}
