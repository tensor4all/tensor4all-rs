use super::super::*;

fn assert_rrlu_invariant(action: impl FnOnce() + std::panic::UnwindSafe) {
    let payload = std::panic::catch_unwind(action).expect_err("invalid internal bounds must fail");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(message.starts_with("rrLU "), "unexpected panic: {message}");
}

#[test]
fn col_major_access_rejects_invalid_indices() {
    for (nrows, row, col) in [
        (2, 2, 0), // A bad row must not alias the next column.
        (0, 0, 0),
        (2, 0, 2),
        (2, 0, usize::MAX),
        (2, 1, usize::MAX / 2),
        (usize::MAX, 1, 1),
    ] {
        assert_rrlu_invariant(|| {
            col_major_get(&[1, 2, 3, 4], nrows, row, col);
        });
        assert_rrlu_invariant(|| {
            col_major_set(&mut [1, 2, 3, 4], nrows, row, col, 9);
        });
    }
}

#[test]
fn submatrix_argmax_rejects_invalid_ranges_and_shapes() {
    for (nrows, ncols, row_start, row_end, col_start, col_end) in [
        (2, 2, 0, 0, 0, 2),
        (2, 2, 1, 0, 0, 2),
        (2, 2, 0, 3, 0, 2),
        (2, 2, 0, 2, 0, 0),
        (2, 2, 0, 2, 1, 0),
        (2, 2, 0, 2, 0, 3),
        (2, 3, 0, 2, 0, 2),
        (usize::MAX, 2, 0, 1, 0, 1),
    ] {
        assert_rrlu_invariant(|| {
            submatrix_argmax_col_major(
                &[1.0, 2.0, 3.0, 4.0],
                nrows,
                ncols,
                row_start,
                row_end,
                col_start,
                col_end,
            );
        });
    }
}

#[test]
fn col_major_swaps_reject_invalid_bounds_before_noops() {
    for (nrows, ncols, a, b) in [(2, 2, 2, 0), (2, 2, 0, 2), (2, 2, 2, 2), (2, 3, 0, 0)] {
        assert_rrlu_invariant(|| {
            swap_rows_col_major(&mut [1, 2, 3, 4], nrows, ncols, a, b);
        });
        assert_rrlu_invariant(|| {
            swap_cols_col_major(&mut [1, 2, 3, 4], nrows, ncols, a, b);
        });
    }
}

#[test]
fn col_major_access_and_swaps_preserve_rectangular_layout() {
    let mut data = [1, 2, 3, 4, 5, 6];
    assert_eq!(col_major_get(&data, 2, 1, 2), 6);
    col_major_set(&mut data, 2, 1, 2, 9);
    assert_eq!(data, [1, 2, 3, 4, 5, 9]);
    swap_rows_col_major(&mut data, 2, 3, 0, 1);
    assert_eq!(data, [2, 1, 4, 3, 9, 5]);
    swap_cols_col_major(&mut data, 2, 3, 0, 2);
    assert_eq!(data, [9, 5, 4, 3, 2, 1]);
    swap_cols_col_major(&mut data, 2, 3, 2, 0);
    assert_eq!(data, [2, 1, 4, 3, 9, 5]);
    swap_rows_col_major(&mut data, 2, 3, 1, 1);
    swap_cols_col_major(&mut data, 2, 3, 2, 2);
    assert_eq!(data, [2, 1, 4, 3, 9, 5]);
    swap_rows_col_major::<i32>(&mut [], 2, 0, 0, 1);
    swap_cols_col_major::<i32>(&mut [], 0, 3, 0, 2);
}

#[test]
fn submatrix_argmax_uses_column_major_tie_order() {
    let data = [100.0, 2.0, -7.0, 100.0, 7.0, 3.0];
    assert_eq!(
        submatrix_argmax_col_major(&data, 3, 2, 1, 3, 0, 2),
        (2, 0, -7.0)
    );
    assert_eq!(
        submatrix_argmax_col_major(&data, 3, 2, 2, 3, 1, 2),
        (2, 1, 3.0)
    );
}
