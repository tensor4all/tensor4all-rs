use super::*;
use std::cell::Cell;

const SHAPE: MatrixShape = MatrixShape { rows: 2, cols: 3 };

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
    assert_eq!(stats.payload_bytes, 3 * 6 * std::mem::size_of::<f64>());
    assert!(stats.table_bytes > 0);
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
