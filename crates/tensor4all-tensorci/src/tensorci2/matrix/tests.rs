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
fn point_payload_is_the_arc_allocation_size() {
    // f64 and Complex64 are at most word-aligned (on 64-bit), so the header
    // is exactly the two counts and nothing is padded.
    let header = 2 * std::mem::size_of::<usize>();
    assert_eq!(point_payload_bytes::<f64>(6), POINT_PAYLOAD);
    assert_eq!(point_payload_bytes::<f64>(6), header + 6 * 8);
    assert_eq!(
        point_payload_bytes::<num_complex::Complex64>(1),
        header + 16
    );
    assert_eq!(
        point_payload_bytes::<num_complex::Complex64>(5),
        header + 5 * 16
    );
    assert_eq!(point_payload_bytes::<f64>(usize::MAX), usize::MAX);
}

#[test]
fn point_payload_pads_the_header_for_over_aligned_values() {
    #[repr(align(32))]
    #[derive(Clone, Copy)]
    struct Wide(#[allow(dead_code)] [u8; 32]);
    // Counts (16 bytes) are padded to the 32-byte alignment of the slice.
    assert_eq!(point_payload_bytes::<Wide>(3), 32 + 3 * 32);
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

/// Fit of `F(x)[i, j] = value(x, i, j)`-like data on `dims`, exactly low rank.
fn fit_for_folding(dims: Vec<usize>) -> MatrixTCI2Result<f64> {
    crossinterpolate2_matrix::<f64, _>(
        |points: &[MultiIndex], out: &mut [f64]| {
            for (p, x) in points.iter().enumerate() {
                let s: f64 = x.iter().map(|&c| 1.0 + c as f64).sum();
                for j in 0..3 {
                    for i in 0..2 {
                        out[p * 6 + i + 2 * j] = (1 + i + 2 * j) as f64 * s;
                    }
                }
            }
        },
        SHAPE,
        dims,
        vec![],
        TCI2Options {
            seed: Some(3),
            tolerance: 1e-12,
            ..TCI2Options::default()
        },
    )
    .unwrap()
}

#[test]
fn folded_end_cores_carry_the_matrix_legs() {
    for dims in [vec![3], vec![3, 4], vec![3, 2, 4]] {
        let fit = fit_for_folding(dims.clone());
        let folded = fit.to_folded().unwrap();
        let core_dims = folded.core_dims();
        assert_eq!(folded.len(), dims.len());
        assert_eq!(folded.site_dims(), dims);
        assert!(!folded.is_empty());
        assert_eq!(core_dims[0][0], 2, "rows leads");
        assert_eq!(core_dims[dims.len() - 1][2], 3, "cols trails");
        for pair in core_dims.windows(2) {
            assert_eq!(pair[0][2], pair[1][0], "bonds match");
        }
    }
}

#[test]
fn folded_evaluate_checks_its_argument() {
    let folded = fit_for_folding(vec![3, 4]).to_folded().unwrap();
    assert!(matches!(
        folded.evaluate(&[0]),
        Err(TCIError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        folded.evaluate(&[0, 4]),
        Err(TCIError::IndexOutOfBounds { .. })
    ));
    let m = folded.evaluate(&[2, 1]).unwrap();
    // F[i, j] = (1 + i + 2 j) * (3 + 2) at x = (2, 1).
    for j in 0..3 {
        for i in 0..2 {
            assert!((m[[i, j]] - ((1 + i + 2 * j) * 5) as f64).abs() < 1e-9);
        }
    }
}

#[test]
fn folded_single_site_core_is_the_dense_slab_stack() {
    let fit = fit_for_folding(vec![3]);
    let folded = fit.to_folded().unwrap();
    assert_eq!(folded.core_dims(), vec![[2, 3, 3]]);
    assert_eq!(
        folded.retained_bytes(),
        2 * 3 * 3 * std::mem::size_of::<f64>()
    );
    for x in 0..3 {
        let a = folded.evaluate(&[x]).unwrap();
        let b = fit.evaluate(&[x]).unwrap();
        for (u, v) in a.as_col_major_slice().iter().zip(b.as_col_major_slice()) {
            assert!((u - v).abs() < 1e-9);
        }
    }
}

/// The pre-borrow readout: every slab is copied out with `slice_site` and the
/// end cores are read element by element with `get3`.
fn evaluate_with_slice_site(fit: &MatrixTCI2Result<f64>, x: &[usize]) -> Vec<f64> {
    let tci = &fit.result.tci;
    let MatrixShape { rows, cols } = fit.shape;
    let first = tci.site_tensor(0);
    let mut width = first.right_dim();
    let mut current = vec![0.0; rows * width];
    for a in 0..width {
        for i in 0..rows {
            current[i + rows * a] = *first.get3(0, i, a);
        }
    }
    for (k, &xk) in x.iter().enumerate() {
        let core = tci.site_tensor(k + 1);
        let slab = core.slice_site(xk);
        current = col_major_gemm(&current, rows, width, &slab, core.right_dim());
        width = core.right_dim();
    }
    let last = tci.site_tensor(x.len() + 1);
    let mut tail = vec![0.0; width * cols];
    for j in 0..cols {
        for a in 0..width {
            tail[a + width * j] = *last.get3(a, j, 0);
        }
    }
    col_major_gemm(&current, rows, width, &tail, cols)
}

#[test]
fn evaluate_is_bit_identical_to_slice_site_extraction() {
    for dims in [vec![3], vec![3, 4], vec![3, 2, 4]] {
        let fit = fit_for_folding(dims.clone());
        let mut x = vec![0; dims.len()];
        loop {
            let got = fit.evaluate(&x).unwrap();
            let want = evaluate_with_slice_site(&fit, &x);
            let got_bits: Vec<u64> = got
                .as_col_major_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let want_bits: Vec<u64> = want.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got_bits, want_bits, "dims {dims:?}, x = {x:?}");
            // Odometer over the physical coordinates.
            let Some(k) = (0..dims.len()).rev().find(|&k| x[k] + 1 < dims[k]) else {
                break;
            };
            x[k] += 1;
            x[k + 1..].fill(0);
        }
    }
}

#[test]
fn strided_gemm_reads_a_slab_in_place() {
    // A 2 x 3 x 2 core, column-major: element (l, s, r) = 1 + l + 2 s + 6 r.
    let core: Vec<f64> = (0..12).map(|i| 1.0 + i as f64).collect();
    let a = [1.0, 2.0, 3.0, 4.0]; // 2 x 2
    for s in 0..3 {
        let slab: Vec<f64> = (0..2)
            .flat_map(|r| (0..2).map(move |l| 1.0 + l as f64 + 2.0 * s as f64 + 6.0 * r as f64))
            .collect();
        assert_eq!(
            col_major_gemm_ld(&a, 2, 2, &core, 2 * s, 2 * 3, 2),
            col_major_gemm(&a, 2, 2, &slab, 2)
        );
    }
}
