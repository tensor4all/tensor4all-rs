use super::*;
use std::cell::{Cell, RefCell};

const SHAPE: MatrixShape = MatrixShape { rows: 2, cols: 3 };

/// Accounted payload of one cached 2 x 3 `f64` point: the six entries and the
/// two reference-count words of the shared allocation.
const POINT_PAYLOAD: usize = 6 * std::mem::size_of::<f64>() + 2 * std::mem::size_of::<usize>();

/// Entry `(i, j)` of the matrix at `x = (a, b)`; distinct for every slot.
fn value(x: &[usize], i: usize, j: usize) -> f64 {
    1000.0 * x[0] as f64 + 100.0 * x[1] as f64 + 10.0 * i as f64 + j as f64
}

fn layout_source<'a>(
    calls: &'a Cell<usize>,
    points_seen: &'a Cell<usize>,
) -> impl Fn(&[MultiIndex], &mut [f64]) + 'a {
    move |points, out| {
        calls.set(calls.get() + 1);
        points_seen.set(points_seen.get() + points.len());
        assert_eq!(out.len(), points.len() * 6);
        assert!(out.iter().all(|&v| v == 0.0), "out must start at default");
        for (p, x) in points.iter().enumerate() {
            for j in 0..3 {
                for i in 0..2 {
                    out[p * 6 + i + 2 * j] = value(x, i, j);
                }
            }
        }
    }
}

#[test]
fn entries_use_point_major_column_major_layout() {
    let calls = Cell::new(0);
    let seen = Cell::new(0);
    let mut state = SourceState::new(layout_source(&calls, &seen), SHAPE, &[3, 4]).unwrap();
    let requests: Vec<(&[usize], usize, usize)> = vec![
        (&[2, 3], 1, 2),
        (&[0, 1], 0, 1),
        (&[2, 3], 0, 0),
        (&[1, 0], 1, 0),
    ];
    let values = state.entries(&requests);
    for (&(x, i, j), v) in requests.iter().zip(&values) {
        assert_eq!(*v, value(x, i, j));
    }
    // Three distinct points in one batch, one callback call.
    assert_eq!((calls.get(), seen.get()), (1, 3));
}

#[test]
fn stats_count_source_points_requests_and_hits() {
    let calls = Cell::new(0);
    let seen = Cell::new(0);
    let mut state = SourceState::new(layout_source(&calls, &seen), SHAPE, &[3, 4]).unwrap();
    state.entries(&[(&[1, 1], 0, 0), (&[1, 1], 1, 2), (&[2, 2], 0, 0)]);
    // Second batch: one cached point, one new point.
    state.entries(&[(&[1, 1], 1, 1), (&[0, 3], 0, 2)]);
    let stats = state.finished_stats();
    assert_eq!(stats.source_points, 3);
    assert_eq!(stats.entry_requests, 5);
    assert_eq!(stats.cache_hits, 2);
    assert_eq!(stats.payload_bytes, 3 * POINT_PAYLOAD);
    assert!(stats.table_bytes > 0);
    assert_eq!(
        stats.retained_bytes,
        stats.table_bytes + stats.payload_bytes
    );
    assert_eq!(seen.get(), 3);
}

#[test]
fn single_entry_requests_hit_the_cache() {
    let calls = Cell::new(0);
    let seen = Cell::new(0);
    let mut state = SourceState::new(layout_source(&calls, &seen), SHAPE, &[3, 4]).unwrap();
    assert_eq!(
        state.entries(&[(&[2, 1], 1, 2)]),
        vec![value(&[2, 1], 1, 2)]
    );
    assert_eq!(
        state.entries(&[(&[2, 1], 0, 1)]),
        vec![value(&[2, 1], 0, 1)]
    );
    assert_eq!(calls.get(), 1);
}

fn noop(_: &[MultiIndex], _: &mut [f64]) {}

fn run(
    shape: MatrixShape,
    dims: Vec<usize>,
    pivots: Vec<MultiIndex>,
) -> Result<MatrixTCI2Result<f64>> {
    crossinterpolate2_matrix::<f64, _>(noop, shape, dims, pivots, TCI2Options::default())
}

#[test]
fn validation_rejects_malformed_problems_before_sampling() {
    let ok = MatrixShape { rows: 2, cols: 3 };
    assert!(matches!(
        run(MatrixShape { rows: 0, cols: 3 }, vec![2], vec![]),
        Err(TCIError::InvalidConfiguration { .. })
    ));
    assert!(matches!(
        run(MatrixShape { rows: 2, cols: 0 }, vec![2], vec![]),
        Err(TCIError::InvalidConfiguration { .. })
    ));
    assert!(matches!(
        run(ok, vec![], vec![]),
        Err(TCIError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        run(ok, vec![2, 0], vec![]),
        Err(TCIError::InvalidConfiguration { .. })
    ));
    assert!(matches!(
        run(
            MatrixShape {
                rows: usize::MAX,
                cols: 2
            },
            vec![2],
            vec![]
        ),
        Err(TCIError::DimensionMismatch { .. })
    ));
    // Pivot of the wrong length (physical-only index) and out of range.
    assert!(matches!(
        run(ok, vec![2], vec![vec![0, 0]]),
        Err(TCIError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        run(ok, vec![2], vec![vec![0, 2, 0]]),
        Err(TCIError::IndexOutOfBounds { .. })
    ));
    assert!(matches!(
        run(ok, vec![2], vec![vec![2, 0, 0]]),
        Err(TCIError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn all_zero_initial_pivot_is_rejected() {
    assert!(matches!(
        run(SHAPE, vec![2], vec![]),
        Err(TCIError::InvalidPivot { .. })
    ));
}

#[test]
fn validation_runs_before_any_callback() {
    let calls = Cell::new(0);
    let result = crossinterpolate2_matrix::<f64, _>(
        |_: &[MultiIndex], _: &mut [f64]| calls.set(calls.get() + 1),
        SHAPE,
        vec![2, 0],
        vec![],
        TCI2Options::default(),
    );
    assert!(result.is_err());
    assert_eq!(calls.get(), 0);
}

#[test]
fn evaluate_checks_its_argument() {
    let fit = crossinterpolate2_matrix::<f64, _>(
        |points: &[MultiIndex], out: &mut [f64]| {
            for (p, x) in points.iter().enumerate() {
                out[p * 6..(p + 1) * 6].fill(1.0 + x[0] as f64);
            }
        },
        SHAPE,
        vec![3],
        vec![],
        TCI2Options {
            seed: Some(0),
            ..TCI2Options::default()
        },
    )
    .unwrap();
    assert!(matches!(
        fit.evaluate(&[0, 0]),
        Err(TCIError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        fit.evaluate(&[3]),
        Err(TCIError::IndexOutOfBounds { .. })
    ));
    let m = fit.evaluate(&[2]).unwrap();
    assert_eq!(m.as_col_major_slice().len(), 6);
    assert!(m
        .as_col_major_slice()
        .iter()
        .all(|v| (v - 3.0).abs() < 1e-9));
}

#[test]
fn col_major_gemm_multiplies_rectangular_operands() {
    // [[1, 2, 3], [4, 5, 6]] * [[1, 0], [0, 1], [1, 1]] (column-major storage)
    let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    assert_eq!(col_major_gemm(&a, 2, 3, &b, 2), vec![4.0, 10.0, 5.0, 11.0]);
}

/// Records every callback call as the list of points it received.
fn recording_source<'a>(
    log: &'a RefCell<Vec<Vec<Vec<usize>>>>,
) -> impl Fn(&[MultiIndex], &mut [f64]) + 'a {
    move |points, out| {
        log.borrow_mut().push(points.to_vec());
        for (p, x) in points.iter().enumerate() {
            for j in 0..3 {
                for i in 0..2 {
                    out[p * 6 + i + 2 * j] = value(x, i, j);
                }
            }
        }
    }
}

type OwnedRequests = Vec<(Vec<usize>, usize, usize)>;

/// One request batch with repeats, five distinct misses and (in the second
/// batch) cached points.
fn split_batches() -> (OwnedRequests, OwnedRequests) {
    let first = vec![
        (vec![2, 3], 1, 2),
        (vec![0, 1], 0, 1),
        (vec![2, 3], 0, 0), // repeat of the first point
        (vec![1, 0], 1, 0),
        (vec![0, 1], 1, 2), // repeat of the second point
        (vec![2, 0], 0, 2),
        (vec![1, 3], 1, 1),
    ];
    let second = vec![
        (vec![1, 0], 0, 0), // cached by the first batch
        (vec![0, 0], 1, 1), // new
        (vec![2, 3], 1, 0), // cached
    ];
    (first, second)
}

fn as_requests(batch: &[(Vec<usize>, usize, usize)]) -> Vec<(&[usize], usize, usize)> {
    batch
        .iter()
        .map(|(x, i, j)| (x.as_slice(), *i, *j))
        .collect()
}

#[test]
fn miss_split_is_deterministic_and_matches_the_unsplit_run() {
    let (first, second) = split_batches();

    // Unsplit run: the default chunk size is far above five points.
    let unsplit_log = RefCell::new(Vec::new());
    let mut unsplit = SourceState::new(recording_source(&unsplit_log), SHAPE, &[3, 4]).unwrap();
    let unsplit_values: Vec<Vec<f64>> = [&first, &second]
        .iter()
        .map(|batch| unsplit.entries(&as_requests(batch)))
        .collect();
    assert_eq!(
        unsplit_log
            .borrow()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        vec![5, 1]
    );

    // Split run: chunks of at most two points.
    let log = RefCell::new(Vec::new());
    let mut split = SourceState::new(recording_source(&log), SHAPE, &[3, 4]).unwrap();
    split.max_points_per_call = 2;
    let split_values: Vec<Vec<f64>> = [&first, &second]
        .iter()
        .map(|batch| split.entries(&as_requests(batch)))
        .collect();

    // Call sizes [2, 2, 1] for the five distinct misses, then [1] for the
    // single new point of the second batch.
    let calls = log.borrow();
    assert_eq!(
        calls.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![2, 2, 1, 1]
    );
    // First-occurrence order across the consecutive chunks.
    let seen_first: Vec<Vec<usize>> = calls[..3].iter().flatten().cloned().collect();
    assert_eq!(
        seen_first,
        vec![vec![2, 3], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 3]]
    );
    assert_eq!(calls[3], vec![vec![0, 0]]);

    // Values are identical to the unsplit run and to the target.
    assert_eq!(split_values, unsplit_values);
    for (batch, values) in [&first, &second].iter().zip(&split_values) {
        for ((x, i, j), v) in batch.iter().zip(values) {
            assert_eq!(*v, value(x, *i, *j));
        }
    }

    // Stats are independent of the chunk size.
    let stats = split.finished_stats();
    assert_eq!(stats, unsplit.finished_stats());
    assert_eq!(stats.source_points, 6);
    assert_eq!(stats.entry_requests, 10);
    assert_eq!(stats.cache_hits, 4);
    assert_eq!(stats.payload_bytes, 6 * POINT_PAYLOAD);
}

#[test]
fn scrambled_requests_keep_every_component_in_its_own_slot() {
    // Every (point, i, j) of the 3 x 4 grid, requested in a scrambled order
    // that interleaves the entries of different points and repeats some.
    let mut requests: Vec<(Vec<usize>, usize, usize)> = Vec::new();
    for a in 0..3 {
        for b in 0..4 {
            for j in 0..3 {
                for i in 0..2 {
                    requests.push((vec![a, b], i, j));
                }
            }
        }
    }
    let n = requests.len(); // 72
    assert_eq!(n, 72);
    // 72 and 29 are coprime, so this is a permutation of the 72 requests.
    let scrambled: Vec<_> = (0..n).map(|k| requests[(k * 29 + 7) % n].clone()).collect();
    let mut with_repeats = scrambled.clone();
    with_repeats.extend(scrambled[..10].iter().cloned());

    for chunk in [1, 2, 1000] {
        let calls = Cell::new(0);
        let seen = Cell::new(0);
        let mut state = SourceState::new(layout_source(&calls, &seen), SHAPE, &[3, 4]).unwrap();
        state.max_points_per_call = chunk;
        // Pre-cache a few points so cached and fresh blocks mix in one batch.
        state.entries(&[(&[1, 2], 0, 0), (&[2, 0], 0, 0)]);
        let values = state.entries(&as_requests(&with_repeats));
        let mut hit = std::collections::HashSet::new();
        for ((x, i, j), v) in with_repeats.iter().zip(&values) {
            assert_eq!(*v, value(x, *i, *j), "slot ({x:?}, {i}, {j})");
            hit.insert((x.clone(), *i, *j));
        }
        // No component lost: all 72 distinct slots were answered.
        assert_eq!(hit.len(), n);
        assert_eq!(state.finished_stats().source_points, 12);
    }
}

#[test]
fn point_payload_counts_entries_and_the_arc_header() {
    assert_eq!(point_payload_bytes::<f64>(6), POINT_PAYLOAD);
    assert_eq!(
        point_payload_bytes::<num_complex::Complex64>(1),
        16 + 2 * std::mem::size_of::<usize>()
    );
    assert_eq!(point_payload_bytes::<f64>(usize::MAX), usize::MAX);
}

#[test]
fn wide_matrices_count_toward_the_limit_and_drive_eviction() {
    let calls = Cell::new(0);
    let seen = Cell::new(0);
    let mut state = SourceState::new(layout_source(&calls, &seen), SHAPE, &[3, 4]).unwrap();
    state.entries(&[(&[0, 0], 0, 0), (&[1, 0], 0, 0)]);
    let two_points = state.finished_stats();
    assert_eq!(two_points.payload_bytes, 2 * POINT_PAYLOAD);
    assert_eq!(
        two_points.retained_bytes,
        two_points.table_bytes + two_points.payload_bytes
    );
    // The table alone has room for a third entry (capacity 3); the limit below
    // pays for two payloads only, so the payload is what forces the eviction.
    state
        .cache
        .set_retained_byte_limit(two_points.retained_bytes);
    state.entries(&[(&[2, 0], 0, 0)]);
    assert_eq!(state.cache.evicted_entries(), 2);
    assert_eq!(state.cache.dropped_inserts(), 0);
    let stats = state.finished_stats();
    assert_eq!(stats.payload_bytes, POINT_PAYLOAD);
    assert!(stats.retained_bytes <= two_points.retained_bytes);
    assert_eq!(stats.source_points, 3);

    // An evicted point is evaluated again; a retained one is not.
    let value_again = state.entries(&[(&[0, 0], 1, 2), (&[2, 0], 1, 2)]);
    assert_eq!(
        value_again,
        vec![value(&[0, 0], 1, 2), value(&[2, 0], 1, 2)]
    );
    assert_eq!(state.finished_stats().source_points, 4);
}

#[test]
fn a_matrix_larger_than_the_limit_is_never_retained_but_still_served() {
    let calls = Cell::new(0);
    let seen = Cell::new(0);
    let mut state = SourceState::new(layout_source(&calls, &seen), SHAPE, &[3, 4]).unwrap();
    state.cache.set_retained_byte_limit(POINT_PAYLOAD);
    let values = state.entries(&[(&[1, 1], 1, 2)]);
    assert_eq!(values, vec![value(&[1, 1], 1, 2)]);
    assert_eq!(state.cache.dropped_inserts(), 1);
    let stats = state.finished_stats();
    assert_eq!((stats.retained_bytes, stats.payload_bytes), (0, 0));
    // The point is evaluated again on the next request.
    state.entries(&[(&[1, 1], 0, 0)]);
    assert_eq!(state.finished_stats().source_points, 2);
}
