//! Cross-component source sharing of `quanticscrossinterpolate_multicomponent`
//! (#1463): one memo per call, identical fits, rejection and limit behavior.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;

use anyhow::{anyhow, Result};
use num_complex::Complex64;
use quanticsgrids::DiscretizedGrid;
use rand::RngCore as _;
use rand::SeedableRng as _;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::MultiIndexCache;
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar, Tensor3Ops};

use super::super::*;
use crate::quantics_tci::shared_sampling::shared_point_payload_bytes;
use crate::{quanticscrossinterpolate_batch_with_rng, QtciOptions, QuanticsTCIError};

const N_COMPONENTS: usize = 3;

/// Scalar types the shared-sampling tests run on.
trait Lift: Copy {
    fn lift(re: f64, im: f64) -> Self;
}

impl Lift for f64 {
    fn lift(re: f64, _im: f64) -> Self {
        re
    }
}

impl Lift for Complex64 {
    fn lift(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

trait TestScalar:
    TTScalar
    + Default
    + Clone
    + 'static
    + tensor4all_core::TensorElement
    + tensor4all_core::MatrixLuciScalar
    + tensor4all_tensorbackend::FullPivLuScalar
    + tensor4all_treetci::globalpivot::ScalarParts
    + Lift
    + PartialEq
    + std::fmt::Debug
{
}

impl<T> TestScalar for T where
    T: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_tensorbackend::FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts
        + Lift
        + PartialEq
        + std::fmt::Debug
{
}

/// Three components of different structure: a smooth rank-one product, a
/// smooth low-rank sum, and a localized bump. All are nonzero at the origin
/// (the default initial pivot).
fn components<V: Lift>(x: f64, y: f64) -> [V; N_COMPONENTS] {
    let bump = (-((x - 0.3).powi(2) + (y - 0.7).powi(2)) / 0.02).exp();
    [
        V::lift((1.0 + x) * (2.0 + y), 0.5 * x),
        V::lift(1.0 + x * y + 0.5 * x * x, y),
        V::lift(1.0 + bump, x - y),
    ]
}

fn grid_2d() -> DiscretizedGrid {
    DiscretizedGrid::builder(&[5, 5])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[1.0, 1.0])
        .build()
        .unwrap()
}

fn options(n_random_init_pivot: usize) -> QtciOptions {
    QtciOptions {
        tolerance: 1e-8,
        max_bond_dim: Some(8),
        n_random_init_pivot,
        ..QtciOptions::default()
    }
}

fn stream() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(7)
}

#[derive(Clone, Copy)]
enum Fault {
    Short,
    Long,
    Error,
}

/// A batched all-components callback that records every point it receives
/// (with the component run that asked for it) and can fail on one call.
struct Recorder {
    current: Cell<usize>,
    points: RefCell<Vec<(usize, Vec<u64>)>>,
    batch_sizes: RefCell<Vec<usize>>,
    /// Values the callback evaluated (returned, before any injected fault).
    values: Cell<usize>,
    fault: Option<(usize, Fault)>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            current: Cell::new(0),
            points: RefCell::new(Vec::new()),
            batch_sizes: RefCell::new(Vec::new()),
            values: Cell::new(0),
            fault: None,
        }
    }

    fn failing(call: usize, fault: Fault) -> Self {
        Self {
            fault: Some((call, fault)),
            ..Self::new()
        }
    }

    fn callback<V: Lift>(&self) -> impl Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>> + '_ {
        move |batch| {
            let call = self.batch_sizes.borrow().len();
            self.batch_sizes.borrow_mut().push(batch.n_points());
            let mut values = Vec::new();
            for point in 0..batch.n_points() {
                let coordinates = batch.point(point).unwrap();
                self.points.borrow_mut().push((
                    self.current.get(),
                    coordinates.iter().map(|c| c.to_bits()).collect(),
                ));
                values.extend(components::<V>(coordinates[0], coordinates[1]));
            }
            self.values.set(self.values.get() + values.len());
            match self.fault {
                Some((at, Fault::Short)) if at == call => {
                    values.pop();
                    Ok(values)
                }
                Some((at, Fault::Long)) if at == call => {
                    values.push(values[0]);
                    Ok(values)
                }
                Some((at, Fault::Error)) if at == call => Err(anyhow!("boom")),
                _ => Ok(values),
            }
        }
    }

    fn n_points(&self) -> usize {
        self.points.borrow().len()
    }

    fn n_calls(&self) -> usize {
        self.batch_sizes.borrow().len()
    }

    /// Points received per component run.
    fn sets(&self) -> Vec<BTreeSet<Vec<u64>>> {
        let mut sets = vec![BTreeSet::new(); N_COMPONENTS];
        for (component, point) in self.points.borrow().iter() {
            sets[*component].insert(point.clone());
        }
        sets
    }
}

/// One component's fit: tensor train, ranks, errors.
type Fit<V> = (SimpleTensorTrain<V>, Vec<usize>, Vec<f64>);

/// A test-local copy of the per-component loop as it was before sharing: each
/// component run has its own memo and the callback is asked for all
/// components of every point the run evaluates.
fn reference_fits<V, F>(
    grid: &DiscretizedGrid,
    f: &F,
    options: &QtciOptions,
    rng: &mut ChaCha8Rng,
    recorder: &Recorder,
) -> QtciResult<Vec<Fit<V>>>
where
    V: TestScalar,
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
{
    let mut fits = Vec::new();
    for component in 0..N_COMPONENTS {
        recorder.current.set(component);
        let adapter = |batch: QuanticsBatch<'_, f64>| -> Result<Vec<V>> {
            let n_points = batch.n_points();
            if n_points == 0 {
                return Ok(Vec::new());
            }
            let returned = f(batch)?;
            anyhow::ensure!(
                returned.len() == n_points * N_COMPONENTS,
                "callback returned {} values for {n_points} points",
                returned.len()
            );
            Ok((0..n_points)
                .map(|point| returned[point * N_COMPONENTS + component])
                .collect())
        };
        let (qtci, ranks, errors) =
            quanticscrossinterpolate_batch_with_rng(grid, adapter, None, options.clone(), rng)?;
        fits.push((qtci.tensor_train(), ranks, errors));
    }
    Ok(fits)
}

/// The shared path, driven component by component.
fn shared_fits<V, F>(
    sampling: &SharedSampling<V>,
    grid: &DiscretizedGrid,
    f: &F,
    options: &QtciOptions,
    rng: &mut ChaCha8Rng,
    recorder: &Recorder,
) -> QtciResult<Vec<Fit<V>>>
where
    V: TestScalar,
    F: Fn(QuanticsBatch<'_, f64>) -> Result<Vec<V>>,
{
    let mut fits = Vec::new();
    for component in 0..N_COMPONENTS {
        recorder.current.set(component);
        fits.push(sampling.run_component(grid, f, component, None, options, rng)?);
    }
    Ok(fits)
}

fn assert_same_tt<V: TestScalar>(left: &SimpleTensorTrain<V>, right: &SimpleTensorTrain<V>) {
    assert_eq!(left.len(), right.len());
    for site in 0..left.len() {
        let (a, b) = (left.site_tensor(site), right.site_tensor(site));
        assert_eq!(
            (a.left_dim(), a.site_dim(), a.right_dim()),
            (b.left_dim(), b.site_dim(), b.right_dim())
        );
        for l in 0..a.left_dim() {
            for s in 0..a.site_dim() {
                for r in 0..a.right_dim() {
                    assert_eq!(
                        a.get3(l, s, r),
                        b.get3(l, s, r),
                        "core {site} ({l},{s},{r})"
                    );
                }
            }
        }
    }
}

fn assert_same_fits<V: TestScalar>(left: &[Fit<V>], right: &[Fit<V>]) {
    assert_eq!(left.len(), right.len());
    for ((tt_a, ranks_a, errors_a), (tt_b, ranks_b, errors_b)) in left.iter().zip(right) {
        assert_same_tt(tt_a, tt_b);
        assert_eq!(ranks_a, ranks_b);
        assert_eq!(errors_a, errors_b);
    }
}

fn stream_tail(rng: &mut ChaCha8Rng) -> Vec<u64> {
    (0..16).map(|_| rng.next_u64()).collect()
}

/// What a comparison of the shared path against the reference loop measured.
struct Comparison {
    /// Points the reference loop's callback received, summed over components.
    reference_points: usize,
    /// Values the reference loop evaluated: `N_COMPONENTS * reference_points`.
    reference_values: usize,
    /// Values the shared run evaluated.
    shared_values: usize,
    /// The stream after the shared run.
    tail: Vec<u64>,
    /// Distinct points over all components' sampled sets.
    union: usize,
    sets: Vec<BTreeSet<Vec<u64>>>,
    /// Points the shared run's callback received.
    shared_points: usize,
    shared_distinct: usize,
    num_evals: usize,
}

/// Run the reference loop and the shared path on the same target and stream;
/// assert the fits, ranks, errors and final stream position are identical.
fn compare_with_reference<V: TestScalar>(n_random: usize, limit: usize) -> Comparison {
    let grid = grid_2d();
    let options = options(n_random);

    let reference_recorder = Recorder::new();
    let mut reference_rng = stream();
    let reference = reference_fits::<V, _>(
        &grid,
        &reference_recorder.callback::<V>(),
        &options,
        &mut reference_rng,
        &reference_recorder,
    )
    .unwrap();

    let shared_recorder = Recorder::new();
    let mut shared_rng = stream();
    let sampling = SharedSampling::<V>::new(&grid.local_dimensions(), N_COMPONENTS, limit).unwrap();
    let shared = shared_fits(
        &sampling,
        &grid,
        &shared_recorder.callback::<V>(),
        &options,
        &mut shared_rng,
        &shared_recorder,
    )
    .unwrap();

    assert_same_fits(&reference, &shared);
    let tail = stream_tail(&mut shared_rng);
    assert_eq!(stream_tail(&mut reference_rng), tail);

    let sets = reference_recorder.sets();
    let union: BTreeSet<_> = sets.iter().flatten().cloned().collect();
    let shared_distinct: BTreeSet<_> = shared_recorder
        .points
        .borrow()
        .iter()
        .map(|(_, point)| point.clone())
        .collect();
    Comparison {
        reference_points: reference_recorder.n_points(),
        reference_values: reference_recorder.values.get(),
        shared_values: shared_recorder.values.get(),
        tail,
        union: union.len(),
        sets,
        shared_points: shared_recorder.n_points(),
        shared_distinct: shared_distinct.len(),
        num_evals: sampling.stats().num_evals,
    }
}

fn assert_partial_overlap(sets: &[BTreeSet<Vec<u64>>]) {
    let union: BTreeSet<_> = sets.iter().flatten().cloned().collect();
    let intersection: BTreeSet<_> = union
        .iter()
        .filter(|point| sets.iter().all(|set| set.contains(*point)))
        .cloned()
        .collect();
    assert!(!intersection.is_empty(), "components share no point");
    assert!(
        intersection.len() < union.len(),
        "components sample identical sets; the test would not exercise partial overlap"
    );
    // Every component also samples points no other component samples.
    for (component, set) in sets.iter().enumerate() {
        let others: BTreeSet<_> = sets
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != component)
            .flat_map(|(_, set)| set.iter().cloned())
            .collect();
        assert!(
            set.iter().any(|point| !others.contains(point)),
            "component {component} samples nothing of its own"
        );
    }
}

/// Run a generic test body on `f64` and `Complex64`.
macro_rules! both_scalars {
    ($real:ident, $complex:ident, $body:ident) => {
        #[test]
        fn $real() {
            $body::<f64>();
        }
        #[test]
        fn $complex() {
            $body::<Complex64>();
        }
    };
}

fn shared_sampling_saves_source_calls_and_keeps_fits<V: TestScalar>() {
    // Same initial pivot for every component, then component-dependent random
    // pivots: the sampled sets overlap only partly.
    for n_random in [0, 2] {
        let comparison = compare_with_reference::<V>(n_random, MULTICOMPONENT_SHARED_MEMO_BYTES);
        // Nothing is evicted, so the callback sees each point of the union once.
        assert_eq!(comparison.shared_points, comparison.union);
        assert_eq!(comparison.shared_distinct, comparison.union);
        assert_eq!(comparison.num_evals, comparison.union);
        // Evaluated values: n_components * sum |S_i| before, n_components * |union| now.
        assert_eq!(
            comparison.reference_values,
            N_COMPONENTS * comparison.reference_points
        );
        assert_eq!(comparison.shared_values, N_COMPONENTS * comparison.union);
        assert!(comparison.shared_values < comparison.reference_values);
        assert!(
            comparison.union < comparison.reference_points,
            "sharing saved no source call: union {} vs per-component sum {}",
            comparison.union,
            comparison.reference_points
        );
        if n_random > 0 {
            assert_partial_overlap(&comparison.sets);
        }
    }
}
both_scalars!(
    shared_sampling_saves_source_calls_and_keeps_fits_f64,
    shared_sampling_saves_source_calls_and_keeps_fits_c64,
    shared_sampling_saves_source_calls_and_keeps_fits
);

fn tiny_limits_evict_and_keep_fits<V: TestScalar>() {
    // 3000 bytes holds a few dozen points (evictions); 0 retains nothing
    // (every insert is dropped). The values served never change.
    // Evictions must not move a draw: the stream position after the call equals
    // the one without a limit.
    let unlimited = compare_with_reference::<V>(2, MULTICOMPONENT_SHARED_MEMO_BYTES);
    for limit in [3000usize, 0] {
        let comparison = compare_with_reference::<V>(2, limit);
        assert_eq!(
            comparison.tail, unlimited.tail,
            "limit {limit} moved the stream"
        );
        assert!(comparison.shared_values > unlimited.shared_values);
        assert_partial_overlap(&comparison.sets);
        assert!(
            comparison.shared_points > comparison.union,
            "limit {limit} evicted nothing: {} callback points for {} distinct",
            comparison.shared_points,
            comparison.union
        );
        assert_eq!(comparison.num_evals, comparison.shared_points);
    }
}
both_scalars!(
    tiny_limits_evict_and_keep_fits_f64,
    tiny_limits_evict_and_keep_fits_c64,
    tiny_limits_evict_and_keep_fits
);

fn public_entry_matches_the_reference_loop<V: TestScalar>() {
    let grid = grid_2d();
    let options = options(2);
    let reference_recorder = Recorder::new();
    let mut reference_rng = stream();
    let reference = reference_fits::<V, _>(
        &grid,
        &reference_recorder.callback::<V>(),
        &options,
        &mut reference_rng,
        &reference_recorder,
    )
    .unwrap();
    let tts: Vec<_> = reference.iter().map(|(tt, _, _)| tt.clone()).collect();
    let expected_tt = combine_component_tts(&tts).unwrap();
    let expected_ranks: Vec<usize> = (0..reference[0].1.len())
        .map(|i| {
            reference
                .iter()
                .map(|(_, ranks, _)| ranks[i])
                .max()
                .unwrap()
        })
        .collect();

    let recorder = Recorder::new();
    let mut rng = stream();
    let (result, ranks, _errors) = quanticscrossinterpolate_multicomponent_with_rng::<V, _, _>(
        &grid,
        recorder.callback::<V>(),
        &[N_COMPONENTS],
        None,
        options,
        &mut rng,
    )
    .unwrap();
    assert_same_tt(&expected_tt, result.tensor_train());
    assert_eq!(expected_ranks, ranks);
    assert_eq!(stream_tail(&mut reference_rng), stream_tail(&mut rng));
    assert!(recorder.n_points() < reference_recorder.n_points());
}
both_scalars!(
    public_entry_matches_the_reference_loop_f64,
    public_entry_matches_the_reference_loop_c64,
    public_entry_matches_the_reference_loop
);

fn one_component_matches_the_plain_batch_fit<V: TestScalar>() {
    let grid = grid_2d();
    let options = options(2);
    let f = |batch: QuanticsBatch<'_, f64>| -> Result<Vec<V>> {
        Ok((0..batch.n_points())
            .map(|p| components::<V>(batch.get(0, p).unwrap(), batch.get(1, p).unwrap())[0])
            .collect())
    };
    let mut plain_rng = stream();
    let (plain, plain_ranks, plain_errors) =
        quanticscrossinterpolate_batch_with_rng(&grid, f, None, options.clone(), &mut plain_rng)
            .unwrap();
    let mut shared_rng = stream();
    let sampling = SharedSampling::<V>::new(
        &grid.local_dimensions(),
        1,
        MULTICOMPONENT_SHARED_MEMO_BYTES,
    )
    .unwrap();
    let (tt, ranks, errors) = sampling
        .run_component(&grid, &f, 0, None, &options, &mut shared_rng)
        .unwrap();
    assert_same_tt(&plain.tensor_train(), &tt);
    assert_eq!((plain_ranks, plain_errors), (ranks, errors));
    assert_eq!(stream_tail(&mut plain_rng), stream_tail(&mut shared_rng));
}
both_scalars!(
    one_component_matches_the_plain_batch_fit_f64,
    one_component_matches_the_plain_batch_fit_c64,
    one_component_matches_the_plain_batch_fit
);

/// Run component 0 cleanly, then component 1 with `fault` injected at call
/// `fault_call` of that run (or of component 0 when `fault_component == 0`).
/// Returns the error and the number of memo entries before the failing run and
/// after it, plus the points the failing run's callback saw before the fault.
fn run_with_fault<V: TestScalar>(
    fault_component: usize,
    fault_call: usize,
    fault: Fault,
) -> (QuanticsTCIError, usize, usize, usize) {
    let grid = grid_2d();
    let options = options(2);
    let sampling = SharedSampling::<V>::new(
        &grid.local_dimensions(),
        N_COMPONENTS,
        MULTICOMPONENT_SHARED_MEMO_BYTES,
    )
    .unwrap();
    let mut rng = stream();
    if fault_component == 1 {
        let clean = Recorder::new();
        sampling
            .run_component(&grid, &clean.callback::<V>(), 0, None, &options, &mut rng)
            .unwrap();
    }
    let before = sampling.cache().borrow().len();
    let failing = Recorder::failing(fault_call, fault);
    let error = sampling
        .run_component(
            &grid,
            &failing.callback::<V>(),
            fault_component,
            None,
            &options,
            &mut rng,
        )
        .err()
        .unwrap_or_else(|| panic!("the injected fault must abort the run"));
    let after = sampling.cache().borrow().len();
    // Points of the batches before the failing one: those were inserted.
    let earlier: usize = failing.batch_sizes.borrow()[..fault_call].iter().sum();
    assert_eq!(
        failing.n_calls(),
        fault_call + 1,
        "run continued past the fault"
    );
    assert_eq!(after, before + earlier, "the failed batch inserted values");
    (error, before, after, earlier)
}

fn faults_abort_and_insert_nothing<V: TestScalar>() {
    for fault_component in [0usize, 1] {
        for fault_call in [0usize, 1] {
            for fault in [Fault::Short, Fault::Long, Fault::Error] {
                let (error, ..) = run_with_fault::<V>(fault_component, fault_call, fault);
                let QuanticsTCIError::Operation { source } = &error else {
                    panic!("expected an Operation error, got {error:?}");
                };
                let text = format!("{source:#}");
                assert!(
                    text.contains(&format!("component {fault_component} of 3")),
                    "{text}"
                );
                match fault {
                    Fault::Error => {
                        assert!(text.contains("boom"), "{text}");
                        // The callback's own error stays in the source chain.
                        assert!(source.chain().any(|cause| cause.to_string() == "boom"));
                    }
                    _ => assert!(
                        text.contains("expected exactly 3 components per point"),
                        "{text}"
                    ),
                }
            }
        }
    }
}
both_scalars!(
    faults_abort_and_insert_nothing_f64,
    faults_abort_and_insert_nothing_c64,
    faults_abort_and_insert_nothing
);

fn all_zero_component_aborts_like_before<V: TestScalar>() {
    // Component 1 vanishes at the origin, the only initial pivot.
    let grid = grid_2d();
    let options = options(0);
    let f = |batch: QuanticsBatch<'_, f64>| -> Result<Vec<V>> {
        let mut values = Vec::new();
        for p in 0..batch.n_points() {
            let (x, y) = (batch.get(0, p).unwrap(), batch.get(1, p).unwrap());
            values.extend([V::lift(1.0 + x, 0.0), V::lift(x * y, 0.0)]);
        }
        Ok(values)
    };
    let shared = quanticscrossinterpolate_multicomponent_with_rng::<V, _, _>(
        &grid,
        &f,
        &[2],
        None,
        options.clone(),
        &mut stream(),
    )
    .err()
    .unwrap();
    // The per-component loop it replaces aborts the same way.
    let reference = quanticscrossinterpolate_batch_with_rng::<V, _, _>(
        &grid,
        |batch: QuanticsBatch<'_, f64>| -> Result<Vec<V>> {
            Ok(f(batch)?.chunks(2).map(|pair| pair[1]).collect())
        },
        None,
        options,
        &mut stream(),
    )
    .err()
    .unwrap();
    assert!(matches!(
        shared,
        QuanticsTCIError::InvalidConfiguration { .. }
    ));
    // Same variant and message as the failing component's own run, now naming it.
    let QuanticsTCIError::InvalidConfiguration { message } = &shared else {
        unreachable!()
    };
    let QuanticsTCIError::InvalidConfiguration { message: own } = &reference else {
        panic!("{reference:?}");
    };
    assert_eq!(message, &format!("component 1 of 2: {own}"));
}
both_scalars!(
    all_zero_component_aborts_like_before_f64,
    all_zero_component_aborts_like_before_c64,
    all_zero_component_aborts_like_before
);

fn payload_is_counted_in_retained_bytes<V: TestScalar>() {
    let grid = grid_2d();
    let local_dims = grid.local_dimensions();
    let recorder = Recorder::new();
    let (_result, _ranks, _errors, sampling) = multicomponent_with_sampling::<V, _, _>(
        &grid,
        recorder.callback::<V>(),
        &[N_COMPONENTS],
        None,
        options(2),
        &mut stream(),
        MULTICOMPONENT_SHARED_MEMO_BYTES,
    )
    .unwrap();
    let cache = sampling.cache().borrow();
    let payload = shared_point_payload_bytes::<V>(N_COMPONENTS);
    // All components' values plus the strong and weak counters of the `Arc`.
    assert_eq!(
        payload,
        N_COMPONENTS * std::mem::size_of::<V>() + 2 * std::mem::size_of::<usize>()
    );
    assert_eq!(cache.value_payload_bytes(), payload);
    assert_eq!(cache.len(), sampling.stats().num_evals);

    // The same number of entries without payload occupies the table bytes.
    let mut bare = MultiIndexCache::<std::sync::Arc<[V]>>::new(&local_dims).unwrap();
    for entry in 0..cache.len() {
        let mut rest = entry;
        let key: Vec<usize> = local_dims
            .iter()
            .map(|&dim| {
                let digit = rest % dim;
                rest /= dim;
                digit
            })
            .collect();
        bare.insert(&key, std::sync::Arc::from(Vec::new())).unwrap();
    }
    let table_bytes = bare.retained_bytes() - bare.len() * bare.value_payload_bytes();
    assert_eq!(cache.retained_bytes(), table_bytes + cache.len() * payload);
}
both_scalars!(
    payload_is_counted_in_retained_bytes_f64,
    payload_is_counted_in_retained_bytes_c64,
    payload_is_counted_in_retained_bytes
);

#[test]
fn limited_memo_stays_within_its_limit() {
    let grid = grid_2d();
    let recorder = Recorder::new();
    let (_result, _ranks, _errors, sampling) = multicomponent_with_sampling::<f64, _, _>(
        &grid,
        recorder.callback::<f64>(),
        &[N_COMPONENTS],
        None,
        options(2),
        &mut stream(),
        3000,
    )
    .unwrap();
    let stats = sampling.stats();
    let cache = sampling.cache().borrow();
    assert!(cache.retained_bytes() <= 3000);
    assert!(stats.evicted_entries > 0);
    assert_eq!(
        stats.num_evals,
        cache.len() + stats.evicted_entries + stats.dropped_inserts
    );
    assert_eq!(stats.num_evals, recorder.n_points());
}

#[test]
fn too_wide_grid_is_rejected_before_any_callback_call() {
    // 40 variables of 30 bits: a 1200-bit quantics key, wider than the cache's
    // widest built-in key.
    let grid = DiscretizedGrid::builder(&[30; 40])
        .with_lower_bound(&[0.0; 40])
        .with_upper_bound(&[1.0; 40])
        .build()
        .unwrap();
    let calls = Cell::new(0usize);
    let error = quanticscrossinterpolate_multicomponent_with_rng::<f64, _, _>(
        &grid,
        |batch: QuanticsBatch<'_, f64>| {
            calls.set(calls.get() + 1);
            Ok(vec![1.0; 2 * batch.n_points()])
        },
        &[2],
        None,
        QtciOptions::default(),
        &mut stream(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(&error, QuanticsTCIError::InvalidConfiguration { message } if message.contains("cannot be memoized")),
        "{error:?}"
    );
    assert_eq!(calls.get(), 0);
}

/// A batched callback for `n_components` entries of a matrix-valued target that
/// records the points it receives.
fn matrix_callback<'a>(
    recorder: &'a Recorder,
    n_components: usize,
    entry: impl Fn(f64, f64, usize) -> f64 + 'a,
) -> impl Fn(QuanticsBatch<'_, f64>) -> Result<Vec<f64>> + 'a {
    move |batch| {
        let mut values = Vec::new();
        for p in 0..batch.n_points() {
            let (x, y) = (batch.get(0, p).unwrap(), batch.get(1, p).unwrap());
            recorder
                .points
                .borrow_mut()
                .push((recorder.current.get(), vec![x.to_bits(), y.to_bits()]));
            values.extend((0..n_components).map(|c| entry(x, y, c)));
        }
        recorder.values.set(recorder.values.get() + values.len());
        Ok(values)
    }
}

/// Source-call and evaluated-value counts of matrix-valued targets, recorded in
/// `docs/worklogs/2026-11-27-multicomponent-shared-sampling.md`.
///
/// Three kinds of assertion, each with its own message:
///
/// 1. **One-sided bound** (what sharing is for): the shared memo calls the
///    source no more than, and here strictly fewer than, the per-component runs.
/// 2. **Structural identities, exact on any platform**: the shared run asks for
///    exactly the union of the points the per-component runs sample, once each,
///    and every callback point evaluates `n_components` values.
/// 3. **Regression pins** of the exact counts, valid only because the target is
///    platform-stable. The entries are sums of two exponentials (quantics rank
///    at most two), so the search converges far below the tolerance and pivot
///    choices do not hinge on rounding noise. A smooth target that is not
///    low-rank within the tolerance (sines of growing frequency under a bond
///    cap) picks pivots from noise and its counts differ between aarch64 and
///    x86_64, for the memo-free per-component path too. The second pass of each
///    case multiplies every value by `1 + 1e-13 * noise`, far above one-ulp libm
///    differences, and must give the same counts.
#[test]
fn matrix_valued_source_call_counts() {
    let grid = grid_2d();
    let options = options(0);
    for (rows, cols, expected_per_component, expected_shared, noise) in [
        (2usize, 3usize, 1254usize, 321usize, 0.0),
        (2, 3, 1254, 321, 1e-13),
        (8, 8, 13136, 428, 0.0),
        (8, 8, 13136, 428, 1e-13),
    ] {
        let n_components = rows * cols;
        let entry = |x: f64, y: f64, component: usize| {
            let (i, j) = ((component / cols) as f64, (component % cols) as f64);
            let smooth = (-0.3 * (1.0 + i) * x + 0.2 * (1.0 + j) * y).exp()
                + 0.5 * (0.4 * j * x - 0.25 * (1.0 + i) * y).exp();
            smooth * (1.0 + noise * (x * 1e3 + y * 7e3 + i).sin())
        };
        // Per-component runs, as before sharing.
        let per_component = Recorder::new();
        let f = matrix_callback(&per_component, n_components, entry);
        let mut rng = stream();
        for component in 0..n_components {
            per_component.current.set(component);
            let adapter = |batch: QuanticsBatch<'_, f64>| -> Result<Vec<f64>> {
                Ok(f(batch)?
                    .chunks(n_components)
                    .map(|point| point[component])
                    .collect())
            };
            quanticscrossinterpolate_batch_with_rng(
                &grid,
                adapter,
                None,
                options.clone(),
                &mut rng,
            )
            .unwrap();
        }

        // One shared memo.
        let shared = Recorder::new();
        let (_, _, _, sampling) = multicomponent_with_sampling::<f64, _, _>(
            &grid,
            matrix_callback(&shared, n_components, entry),
            &[rows, cols],
            None,
            options.clone(),
            &mut stream(),
            MULTICOMPONENT_SHARED_MEMO_BYTES,
        )
        .unwrap();
        let union: BTreeSet<_> = per_component
            .points
            .borrow()
            .iter()
            .map(|(_, point)| point.clone())
            .collect();
        let label = format!("{rows}x{cols}, relative noise {noise}");
        let (per_component_points, shared_points) = (per_component.n_points(), shared.n_points());

        // 1. One-sided: sharing never costs more source calls, and here saves some.
        assert!(
            shared_points <= per_component_points,
            "shared memo calls the source no more than per-component runs ({label}): \
             shared {shared_points} vs per-component {per_component_points}"
        );
        assert!(
            shared_points < per_component_points,
            "shared memo calls the source strictly fewer times than per-component runs ({label})"
        );

        // 2. Structural identities.
        assert_eq!(
            shared_points,
            union.len(),
            "structural: the shared run asks for exactly the union of the per-component sampled sets, once each ({label})"
        );
        assert_eq!(
            sampling.stats().num_evals,
            union.len(),
            "structural: memo num_evals equals the union of the sampled sets ({label})"
        );
        assert_eq!(
            per_component.values.get(),
            n_components * per_component_points,
            "structural: per-component runs evaluate n_components values per point ({label})"
        );
        assert_eq!(
            shared.values.get(),
            n_components * shared_points,
            "structural: the shared run evaluates n_components values per point ({label})"
        );

        // 3. Regression pins on the platform-stable rank-2 target.
        assert_eq!(
            per_component_points, expected_per_component,
            "regression pin (rank-2 target, platform-stable): per-component points ({label})"
        );
        assert_eq!(
            shared_points, expected_shared,
            "regression pin (rank-2 target, platform-stable): shared points ({label})"
        );
        assert_eq!(
            per_component.values.get(),
            n_components * expected_per_component,
            "regression pin (rank-2 target, platform-stable): per-component values ({label})"
        );
        assert_eq!(
            shared.values.get(),
            n_components * expected_shared,
            "regression pin (rank-2 target, platform-stable): shared values ({label})"
        );
    }
}
