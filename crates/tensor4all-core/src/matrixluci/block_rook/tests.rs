use crate::matrixluci::block_rook::{remaining_indices, LazyBlockRookKernel};
use crate::matrixluci::dense::DenseLuKernel;
use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::source::{CandidateMatrixSource, DenseMatrixSource, LazyMatrixSource};
use crate::matrixluci::types::PivotKernelOptions;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type LazyDenseSource = LazyMatrixSource<f64, Box<dyn Fn(&[usize], &[usize], &mut [f64])>>;

fn unique_pivot_test_matrix() -> Vec<f64> {
    vec![
        9.0, 0.2, 0.3, 0.4, //
        0.1, 8.0, 0.2, 0.3, //
        0.2, 0.1, 7.0, 0.2, //
        0.3, 0.2, 0.1, 6.0, //
    ]
}

fn dense_to_lazy(data: Vec<f64>, nrows: usize, ncols: usize) -> LazyDenseSource {
    LazyMatrixSource::new(
        nrows,
        ncols,
        Box::new(move |rows, cols, out: &mut [f64]| {
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = data[row + nrows * col];
                }
            }
        }),
    )
}

#[test]
fn lazy_block_rook_kernel_matches_dense_kernel_on_unique_pivot_matrix() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);

    let dense_out = DenseLuKernel
        .factorize(&dense, &PivotKernelOptions::no_truncation())
        .unwrap();
    let lazy_out = LazyBlockRookKernel
        .factorize(&lazy, &PivotKernelOptions::no_truncation())
        .unwrap();

    assert_eq!(lazy_out.row_indices, dense_out.row_indices);
    assert_eq!(lazy_out.col_indices, dense_out.col_indices);
    assert_eq!(lazy_out.rank, dense_out.rank);
    assert_eq!(lazy_out.pivot_errors, dense_out.pivot_errors);
}

#[test]
fn lazy_block_rook_kernel_matches_dense_kernel_for_abs_tol_stop() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);
    let options = PivotKernelOptions {
        abs_tol: 6.5,
        ..PivotKernelOptions::default()
    };

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    assert_eq!(lazy_out.row_indices, dense_out.row_indices);
    assert_eq!(lazy_out.col_indices, dense_out.col_indices);
    assert_eq!(lazy_out.rank, dense_out.rank);
    assert_eq!(lazy_out.pivot_errors, dense_out.pivot_errors);
}

#[test]
fn lazy_block_rook_kernel_avoids_full_matrix_request() {
    let data = unique_pivot_test_matrix();
    let max_requested = Arc::new(AtomicUsize::new(0));
    let lazy = LazyMatrixSource::new(4, 4, {
        let max_requested = max_requested.clone();
        move |rows, cols, out: &mut [f64]| {
            max_requested.fetch_max(rows.len() * cols.len(), Ordering::SeqCst);
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = data[row + 4 * col];
                }
            }
        }
    });

    let out = LazyBlockRookKernel
        .factorize(&lazy, &PivotKernelOptions::no_truncation())
        .unwrap();

    assert_eq!(out.rank, 4);
    assert!(max_requested.load(Ordering::SeqCst) < 16);
}

#[test]
fn lazy_block_rook_reconstructs_across_zero_fibers() {
    fn check<T: crate::MatrixLuciScalar>(value: T) {
        for (nrows, ncols) in [(2, 2), (3, 5), (5, 3), (16, 16)] {
            for diagonal_gap in [false, true] {
                for left_orthogonal in [false, true] {
                    let mut data = vec![T::zero(); nrows * ncols];
                    data[nrows * ncols - 1] = value;
                    if diagonal_gap {
                        data[0] = value;
                    }
                    let factors = crate::matrix_luci_factors_from_blocks(
                        nrows,
                        ncols,
                        |rows, cols, out| {
                            for (j, &col) in cols.iter().enumerate() {
                                for (i, &row) in rows.iter().enumerate() {
                                    out[i + rows.len() * j] = data[row + nrows * col];
                                }
                            }
                        },
                        crate::RrLUOptions {
                            left_orthogonal,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    assert_eq!(factors.rank, if diagonal_gap { 2 } else { 1 });
                    let reconstructed =
                        tensor4all_tensorbackend::mat_mul(&factors.left, &factors.right).unwrap();
                    let error = reconstructed
                        .as_col_major_slice()
                        .iter()
                        .zip(&data)
                        .map(|(&actual, &expected)| (actual - expected).abs_val())
                        .fold(0.0_f64, f64::max);
                    assert!(error < 1e-6, "reconstruction error: {error}");
                }
            }
        }
    }
    check(2.0_f32);
    check(2.0_f64);
    check(num_complex::Complex32::new(2.0, 1.0));
    check(num_complex::Complex64::new(2.0, 1.0));
}

#[test]
fn lazy_block_rook_certifies_zero_without_materializing_matrix() {
    for (nrows, ncols) in [(1, 1), (1, 8), (8, 1), (8, 16), (32, 64)] {
        let evaluated = AtomicUsize::new(0);
        let largest_block = AtomicUsize::new(0);
        let source = LazyMatrixSource::new(nrows, ncols, |rows, cols, out: &mut [f64]| {
            evaluated.fetch_add(out.len(), Ordering::Relaxed);
            largest_block.fetch_max(rows.len() * cols.len(), Ordering::Relaxed);
            out.fill(0.0);
        });
        let selection = LazyBlockRookKernel
            .factorize(&source, &PivotKernelOptions::no_truncation())
            .unwrap();
        assert_eq!(selection.rank, 0);
        assert_eq!(selection.pivot_errors, vec![0.0]);
        // Certifying zero requires inspecting all entries, but only a column
        // is materialized at a time, including rectangular and singleton cases.
        assert_eq!(evaluated.load(Ordering::Relaxed), nrows * ncols);
        assert_eq!(largest_block.load(Ordering::Relaxed), nrows);
    }
}

fn rank_cap_options(max_bond_dim: usize) -> PivotKernelOptions<'static> {
    PivotKernelOptions {
        max_bond_dim,
        rel_tol: 0.0,
        abs_tol: 0.0,
        left_orthogonal: true,
        seeds: &[],
    }
}

/// Structured matrix whose diagonal magnitudes decrease in column order, so
/// complete pivoting (dense) and rook search (lazy) select the same pivots.
fn decreasing_pivot_matrix(nrows: usize, ncols: usize, seed: u64) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let full_rank = nrows.min(ncols);
    let mut data: Vec<f64> = (0..nrows * ncols)
        .map(|_| 0.01 * (rng.random::<f64>() - 0.5))
        .collect();
    for i in 0..full_rank {
        data[i + nrows * i] = (full_rank - i) as f64;
    }
    data
}

#[test]
fn lazy_rank_cap_error_matches_dense_kernel_across_shapes_and_caps() {
    for (nrows, ncols) in [(4usize, 4usize), (6, 6), (8, 5), (5, 8), (12, 12), (20, 10)] {
        let full_rank = nrows.min(ncols);
        for cap in 1..full_rank {
            for seed in 1..=5u64 {
                let data = decreasing_pivot_matrix(nrows, ncols, seed);
                let dense_source = DenseMatrixSource::from_column_major(&data, nrows, ncols);
                let lazy_source = dense_to_lazy(data.clone(), nrows, ncols);
                let options = rank_cap_options(cap);

                let dense_out = DenseLuKernel.factorize(&dense_source, &options).unwrap();
                let lazy_out = LazyBlockRookKernel
                    .factorize(&lazy_source, &options)
                    .unwrap();

                // Comparing errors is only meaningful on the same pivot set.
                assert_eq!(lazy_out.rank, cap, "{nrows}x{ncols} cap {cap}");
                assert_eq!(dense_out.row_indices, lazy_out.row_indices);
                assert_eq!(dense_out.col_indices, lazy_out.col_indices);
                assert_eq!(lazy_out.pivot_errors.len(), cap + 1);

                let dense_error = *dense_out.pivot_errors.last().unwrap();
                let lazy_error = *lazy_out.pivot_errors.last().unwrap();
                // On this diagonally dominant family the rook pivot a cap
                // rejects is also the remaining maximum, so the two kernels
                // agree. The dense kernel reads it off its LU diagonal and the
                // lazy kernel subtracts A[:, J] A[I, J]^{-1} A[I, :], so
                // agreement holds to rounding noise only (measured worst case:
                // 1.3 ULP). That agreement is a property of these matrices; see
                // `lazy_zero_rank_cap_reports_the_local_rook_pivot_not_the_matrix_maximum`.
                let relative = (dense_error - lazy_error).abs() / dense_error.abs();
                assert!(
                    relative <= 4.0 * f64::EPSILON,
                    "{nrows}x{ncols} cap {cap} seed {seed}: dense {dense_error:e} lazy {lazy_error:e}"
                );
            }
        }
    }
}

#[test]
fn lazy_rank_cap_error_is_the_rejected_rook_pivot() {
    // A[0, 0] = 4 is the largest entry, so both kernels pivot there first and
    // the remaining 2x2 Schur complement is exactly [[3, 0], [0, 1]]. The extra
    // walk at the cap starts in its first column and finds 3.
    fn check<T>(four: f64, two: f64, three: f64, one: f64)
    where
        T: crate::MatrixLuciScalar,
        DenseLuKernel: PivotKernel<T>,
        LazyBlockRookKernel: PivotKernel<T>,
    {
        let convert = |value: f64| T::from_f64(value);
        let data = vec![
            convert(four),
            convert(two),
            convert(0.0),
            convert(two),
            convert(four),
            convert(0.0),
            convert(0.0),
            convert(0.0),
            convert(one),
        ];
        let dense = DenseMatrixSource::from_column_major(&data, 3, 3);
        let lazy = LazyMatrixSource::new(3, 3, {
            let data = data.clone();
            move |rows, cols, out: &mut [T]| {
                for (j, &col) in cols.iter().enumerate() {
                    for (i, &row) in rows.iter().enumerate() {
                        out[i + rows.len() * j] = data[row + 3 * col];
                    }
                }
            }
        });
        let options = rank_cap_options(1);
        let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
        let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

        assert_eq!(dense_out.pivot_errors, vec![four, three]);
        assert_eq!(lazy_out.pivot_errors, vec![four, three]);
        // The #1089 value was the last accepted pivot (4), not the rejected one.
        assert_ne!(lazy_out.pivot_errors[0], lazy_out.pivot_errors[1]);
    }
    check::<f64>(4.0, 2.0, 3.0, 1.0);
    check::<f32>(4.0, 2.0, 3.0, 1.0);
    check::<num_complex::Complex64>(4.0, 2.0, 3.0, 1.0);
    check::<num_complex::Complex32>(4.0, 2.0, 3.0, 1.0);
}

#[test]
fn lazy_rank_cap_error_ignores_the_last_accepted_pivot() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);
    let options = rank_cap_options(2);

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    assert_eq!(lazy_out.pivot_errors, dense_out.pivot_errors);
    let accepted = lazy_out.pivot_errors[lazy_out.rank - 1];
    assert_eq!(
        lazy_out.pivot_errors.last().copied(),
        Some(6.990983606557377)
    );
    assert_eq!(accepted, 7.997777777777777);
}

#[test]
fn lazy_zero_rank_cap_reports_a_rook_pivot() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);
    let options = rank_cap_options(0);

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    // Nothing was accepted, so the trailing entry is the pivot of one walk over
    // the whole matrix; here it is also the maximum. The #1089 value was NaN.
    assert_eq!(lazy_out.rank, 0);
    assert_eq!(dense_out.pivot_errors, vec![9.0]);
    assert_eq!(lazy_out.pivot_errors, vec![9.0]);
}

#[test]
fn full_rank_reports_zero_last_error_in_both_kernels() {
    for (nrows, ncols) in [(4usize, 4usize), (8, 5), (5, 8)] {
        let data = decreasing_pivot_matrix(nrows, ncols, 7);
        let dense = DenseMatrixSource::from_column_major(&data, nrows, ncols);
        let lazy = dense_to_lazy(data.clone(), nrows, ncols);
        let full_rank = nrows.min(ncols);

        for options in [
            PivotKernelOptions::no_truncation(),
            rank_cap_options(full_rank + 3),
        ] {
            let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
            let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();
            assert_eq!(dense_out.rank, full_rank);
            assert_eq!(lazy_out.rank, full_rank);
            assert_eq!(dense_out.pivot_errors.last().copied(), Some(0.0));
            assert_eq!(lazy_out.pivot_errors.last().copied(), Some(0.0));
        }
    }
}

#[test]
fn lazy_bond_error_reports_the_rejected_rook_pivot_at_a_rank_cap() {
    let data = unique_pivot_test_matrix();
    let factors = crate::matrix_luci_factors_from_blocks(
        4,
        4,
        |rows, cols, out| {
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = data[row + 4 * col];
                }
            }
        },
        crate::RrLUOptions {
            max_bond_dim: 2,
            rel_tol: 0.0,
            abs_tol: 0.0,
            left_orthogonal: true,
        },
    )
    .unwrap();

    assert_eq!(factors.rank, 2);
    // Downstream bond error: the rook pivot the cap rejected. Before #1090
    // this reported the last accepted pivot, 7.997777777777777.
    assert_eq!(
        factors.pivot_errors,
        vec![9.0, 7.997777777777777, 6.990983606557377]
    );
}

#[test]
fn lazy_zero_rank_cap_reports_the_local_rook_pivot_not_the_matrix_maximum() {
    // The rook search converges on the local maximum 3 at (1, 0) while complete
    // pivoting reaches the global maximum 10 at (2, 2). Rook reports its own
    // pivot, which is not a bound on the residual; Full reports the maximum.
    let data = vec![
        1.0_f64, 3.0, 0.0, // column 0
        2.0, 0.5, 0.0, // column 1
        0.0, 0.0, 10.0, // column 2
    ];
    let dense = DenseMatrixSource::from_column_major(&data, 3, 3);
    let lazy = dense_to_lazy(data.clone(), 3, 3);
    let options = rank_cap_options(0);

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    assert_eq!(dense_out.pivot_errors, vec![10.0]);
    assert_eq!(lazy_out.pivot_errors, vec![3.0]);
}

#[test]
fn lazy_rook_walk_rejects_non_finite_residuals_it_visits() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // A zero cap still walks once, so the walk sees the only entry.
        let lazy = dense_to_lazy(vec![invalid], 1, 1);
        let error = LazyBlockRookKernel
            .factorize(&lazy, &rank_cap_options(0))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::matrixluci::MatrixLuciError::NaNEncountered { .. }
        ));

        // After the pivot at (0, 0) the walk at the cap starts in column 1 and
        // meets the entry at (1, 1).
        let data = vec![4.0, 0.0, 0.0, 0.0, invalid, 0.0, 0.0, 0.0, 1.0];
        let lazy = dense_to_lazy(data, 3, 3);
        let error = LazyBlockRookKernel
            .factorize(&lazy, &rank_cap_options(1))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::matrixluci::MatrixLuciError::NaNEncountered { .. }
        ));
    }
}

/// The heuristic contract also covers non-finite values: an entry outside the
/// rows and columns the walks visit is never evaluated, so `Rook` cannot see
/// it. Only `Full` evaluates every entry.
#[test]
fn rook_does_not_see_a_non_finite_entry_its_walks_never_visit() {
    for invalid in [f64::NAN, f64::INFINITY] {
        // After the pivot 4 at (0, 0), the walk at the cap starts in column 1,
        // stops on the 1 at (1, 1) and never reads (2, 2).
        let data = vec![4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, invalid];
        let out = LazyBlockRookKernel
            .factorize(&dense_to_lazy(data, 3, 3), &rank_cap_options(1))
            .unwrap();
        assert_eq!(out.pivot_errors, vec![4.0, 1.0]);
    }
}

/// Source wrapper that counts the entries a kernel asks for.
///
/// `requested` counts every entry copied out, reloads included; `unique`
/// counts first touches, which is what a memoizing evaluator pays for.
struct CountingSource<S> {
    inner: S,
    requested: std::cell::Cell<usize>,
    touched: std::cell::RefCell<std::collections::HashSet<(usize, usize)>>,
}

impl<S> CountingSource<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            requested: std::cell::Cell::new(0),
            touched: std::cell::RefCell::new(std::collections::HashSet::new()),
        }
    }

    fn requested(&self) -> usize {
        self.requested.get()
    }

    fn unique(&self) -> usize {
        self.touched.borrow().len()
    }
}

impl<T: crate::MatrixLuciScalar, S: CandidateMatrixSource<T>> CandidateMatrixSource<T>
    for CountingSource<S>
{
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn get_block(&self, rows: &[usize], cols: &[usize], out: &mut [T]) {
        self.requested
            .set(self.requested.get() + rows.len() * cols.len());
        let mut touched = self.touched.borrow_mut();
        for &col in cols {
            for &row in rows {
                touched.insert((row, col));
            }
        }
        self.inner.get_block(rows, cols, out);
    }
}

/// `A = U Vᵀ` with `U`, `V` of shape `n × rank`, column-major.
fn exact_rank_matrix(n: usize, rank: usize, seed: u64) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let u: Vec<f64> = (0..n * rank).map(|_| rng.random::<f64>() - 0.5).collect();
    let v: Vec<f64> = (0..n * rank).map(|_| rng.random::<f64>() - 0.5).collect();
    let mut data = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            data[row + n * col] = (0..rank).map(|k| u[row + n * k] * v[col + n * k]).sum();
        }
    }
    data
}

/// The rook walk as it was before #1247: every column and row block reloads
/// `A[I, J]` and `A[rows, J]`, and every block reloads `A[I, cols]`. It is the
/// oracle for the hoisted walk, which must perform the same arithmetic on the
/// same values.
fn reloading_rook_pivot<S: CandidateMatrixSource<f64>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
) -> (usize, usize, f64) {
    use super::{argmax_abs, eliminate, load_block};
    let block = |rows: &[usize], cols: &[usize]| {
        let mut residual = load_block(source, rows, cols);
        if selected_rows.is_empty() {
            return residual;
        }
        let pivot = load_block(source, selected_rows, selected_cols);
        let a_rj = load_block(source, rows, selected_cols);
        let a_ic = load_block(source, selected_rows, cols);
        eliminate(&mut residual, &pivot, &a_rj, &a_ic).unwrap();
        residual
    };

    let mut columns = remaining_cols.iter().copied();
    let (mut current_row, mut current_col);
    loop {
        let Some(col) = columns.next() else {
            return (remaining_rows[0], remaining_cols[0], 0.0);
        };
        current_col = col;
        let (row_pos, _, best_abs) = argmax_abs(&block(remaining_rows, &[col]));
        if best_abs != 0.0 {
            current_row = remaining_rows[row_pos];
            break;
        }
    }
    for _ in 0..remaining_rows.len() + remaining_cols.len() + 1 {
        let (_, col_pos, best_abs) = argmax_abs(&block(&[current_row], remaining_cols));
        let next_col = remaining_cols[col_pos];
        if next_col == current_col {
            return (current_row, current_col, best_abs);
        }
        current_col = next_col;
        let (row_pos, _, _) = argmax_abs(&block(remaining_rows, &[current_col]));
        current_row = remaining_rows[row_pos];
    }
    let (_, col_pos, best_abs) = argmax_abs(&block(&[current_row], remaining_cols));
    (current_row, remaining_cols[col_pos], best_abs)
}

#[test]
fn lazy_rook_walk_loads_the_pivot_operands_once_and_selects_the_same_pivots() {
    use super::rook_pivot;
    // The shape measured on #1247: a binding cap below the exact rank.
    let (n, exact_rank, cap) = (400usize, 30usize, 15usize);
    let data = exact_rank_matrix(n, exact_rank, 1247);
    let source = CountingSource::new(dense_to_lazy(data.clone(), n, n));

    let out = LazyBlockRookKernel
        .factorize(&source, &rank_cap_options(cap))
        .unwrap();
    assert_eq!(out.rank, cap);
    // A rank-cap exit measures the rejected pivot with one more walk, never a
    // scan of the remaining block: 16 walks touch 15,980 of the 160,000
    // entries.
    assert!(
        source.unique() < n * n / 8,
        "unique {} of {} entries: the rank-cap exit scanned the block",
        source.unique(),
        n * n
    );

    // Same pivots, same pivot magnitudes, bit for bit, as the reloading walk.
    // Step `cap` is the extra walk whose pivot is the trailing entry.
    let hoisted = CountingSource::new(dense_to_lazy(data.clone(), n, n));
    let reloading = CountingSource::new(dense_to_lazy(data, n, n));
    for step in 0..=cap {
        let selected_rows = &out.row_indices[..step];
        let selected_cols = &out.col_indices[..step];
        let remaining_rows = remaining_indices(n, selected_rows);
        let remaining_cols = remaining_indices(n, selected_cols);
        let (row, col, abs) = rook_pivot(
            &hoisted,
            &remaining_rows,
            &remaining_cols,
            selected_rows,
            selected_cols,
            None,
        )
        .unwrap();
        let (oracle_row, oracle_col, oracle_abs) = reloading_rook_pivot(
            &reloading,
            &remaining_rows,
            &remaining_cols,
            selected_rows,
            selected_cols,
        );
        assert_eq!((row, col), (oracle_row, oracle_col), "step {step}");
        assert_eq!(abs.to_bits(), oracle_abs.to_bits(), "step {step}");
        if step < cap {
            assert_eq!((row, col), (out.row_indices[step], out.col_indices[step]));
        }
        assert_eq!(
            abs.to_bits(),
            out.pivot_errors[step].to_bits(),
            "step {step}"
        );
    }
    // Reloading the walk invariants per column or row block requests 235,200
    // entries over the 16 walks; loading them once per walk requests 122,242
    // (1.92x fewer). The bound leaves ~7% headroom.
    let ratio = reloading.requested() as f64 / hoisted.requested() as f64;
    assert!(
        ratio > 1.8,
        "reloading {} / hoisted {} = {ratio:.3}",
        reloading.requested(),
        hoisted.requested()
    );
}

/// Smooth ridge from tensor4all-rs#1256: a residual with several separated
/// bumps, on which a rook walk ends at a local maximum far below the rest.
fn smooth_ridge_matrix(n: usize) -> Vec<f64> {
    let entry = |i: usize, j: usize| {
        let x = i as f64 / n as f64;
        let y = (j as f64 + 0.5) / n as f64;
        (-(x - y) * (x - y) / 0.002).exp() + 0.3 * (3.0 * x * y).cos()
    };
    let mut data = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            data[i + n * j] = entry(i, j);
        }
    }
    data
}

/// `max |A - A[:, J] A[I, J]^{-1} A[I, :]|` over every entry of `A`, built
/// densely and independently of the kernel's residual helpers.
fn interpolation_max_error(data: &[f64], n: usize, rows: &[usize], cols: &[usize]) -> f64 {
    use tensor4all_tensorbackend::{mat_mul, solve_matrix, Matrix};
    let block = |rs: &[usize], cs: &[usize]| {
        let mut out = Vec::with_capacity(rs.len() * cs.len());
        for &c in cs {
            for &r in rs {
                out.push(data[r + n * c]);
            }
        }
        Matrix::from_col_major_vec(rs.len(), cs.len(), out)
    };
    let all: Vec<usize> = (0..n).collect();
    let solved = solve_matrix(&block(rows, cols), &block(rows, &all)).unwrap();
    let approx = mat_mul(&block(&all, cols), &solved).unwrap();
    let mut max_err = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let err = (data[i + n * j] - approx[[i, j]]).abs();
            assert!(
                err.is_finite(),
                "non-finite interpolation error at ({i}, {j})"
            );
            max_err = max_err.max(err);
        }
    }
    max_err
}

fn ridge_options() -> PivotKernelOptions<'static> {
    PivotKernelOptions {
        rel_tol: 1e-8,
        abs_tol: 0.0,
        ..PivotKernelOptions::default()
    }
}

#[test]
fn dense_tolerance_stop_reports_the_true_interpolation_error() {
    // The certified strategy on the tensor4all-rs#1256 ridge: Full's trailing
    // entry is the maximum of the remaining residual, so it is the true
    // interpolation error and below the tolerance.
    let n = 200;
    let options = ridge_options();
    let data = smooth_ridge_matrix(n);
    let dense = DenseLuKernel
        .factorize(&DenseMatrixSource::from_column_major(&data, n, n), &options)
        .unwrap();
    let reported = *dense.pivot_errors.last().unwrap();
    let first_pivot = dense.pivot_errors[0];
    let true_err = interpolation_max_error(&data, n, &dense.row_indices, &dense.col_indices);
    // The selected rows and columns carry only rounding error, so the reported
    // remaining maximum and the true maximum agree closely.
    assert!(
        (reported - true_err).abs() <= 1e-3 * true_err + 1e-13 * first_pivot,
        "reported error {reported:.3e} is not the true error {true_err:.3e}"
    );
    assert!(
        true_err < options.rel_tol * first_pivot,
        "stopped with true error {true_err:.3e} >= tolerance {:.3e}",
        options.rel_tol * first_pivot
    );
}

#[test]
fn rook_tolerance_stop_on_the_1256_ridge_underestimates_the_true_error() {
    // The heuristic contract on realistic data: the walk stops on a rook pivot
    // 2.5e-9 of the first pivot while the interpolant it returns is off by
    // 1.3e-6, above the 1e-8 tolerance. `pivot_errors` is not a bound.
    let n = 200;
    let options = ridge_options();
    let data = smooth_ridge_matrix(n);
    let rook = LazyBlockRookKernel
        .factorize(&dense_to_lazy(data.clone(), n, n), &options)
        .unwrap();
    let reported = *rook.pivot_errors.last().unwrap();
    let first_pivot = rook.pivot_errors[0];
    let true_err = interpolation_max_error(&data, n, &rook.row_indices, &rook.col_indices);
    assert!(
        reported < options.rel_tol * first_pivot,
        "rook did not stop on tolerance"
    );
    assert!(
        true_err > 100.0 * options.rel_tol * first_pivot,
        "true error {true_err:.3e} is no longer far above the tolerance; \
         the limitation this test pins has changed"
    );
}

/// The smallest case of the heuristic contract: after the first pivot the next
/// walk starts in column 1, whose only residual entry is a local pivot below
/// tolerance, so `Rook` stops there and reports it, although the remaining
/// block still holds the entry 5. `Full` takes 5 as its second pivot.
#[test]
fn rook_stops_on_a_small_local_pivot_and_misses_a_larger_remaining_entry() {
    // diag(10, 0.01, 5), column-major.
    let data = vec![
        10.0, 0.0, 0.0, //
        0.0, 0.01, 0.0, //
        0.0, 0.0, 5.0, //
    ];
    let options = PivotKernelOptions {
        rel_tol: 1e-2,
        abs_tol: 0.0,
        ..PivotKernelOptions::default()
    };
    let rook = LazyBlockRookKernel
        .factorize(&dense_to_lazy(data.clone(), 3, 3), &options)
        .unwrap();
    let full = DenseLuKernel
        .factorize(&DenseMatrixSource::from_column_major(&data, 3, 3), &options)
        .unwrap();

    // Rook: one pivot, then a stop on 0.01 < 1e-2 * 10. The reported error is
    // the local pivot, 500x below the true remaining error.
    assert_eq!(rook.rank, 1);
    assert_eq!(rook.pivot_errors, vec![10.0, 0.01]);
    assert_eq!(
        interpolation_max_error(&data, 3, &rook.row_indices, &rook.col_indices),
        5.0
    );

    // Full does not stop there: it accepts 5 and stops on the true maximum.
    assert_eq!(full.rank, 2);
    assert_eq!(full.col_indices, vec![0, 2]);
    assert_eq!(full.pivot_errors, vec![10.0, 5.0, 0.01]);
    assert_eq!(
        interpolation_max_error(&data, 3, &full.row_indices, &full.col_indices),
        0.01
    );
}

#[test]
fn lazy_seeded_walk_requests_less_than_the_unseeded_walk() {
    use super::stats;
    // Seeding is what `arrlu` does with its `I0`/`J0`: the previous pivots are
    // where the walks start. On the #1256 ridge the seeded revisit asks for
    // 742,898 entries where the unseeded run asks for 945,422.
    let n = 200;
    let data = smooth_ridge_matrix(n);
    let options = ridge_options();

    stats::reset();
    let unseeded_source = CountingSource::new(dense_to_lazy(data.clone(), n, n));
    let unseeded = LazyBlockRookKernel
        .factorize(&unseeded_source, &options)
        .unwrap();
    assert_eq!(stats::snapshot().seeded_starts, 0);

    let seeds: Vec<(usize, usize)> = unseeded
        .row_indices
        .iter()
        .copied()
        .zip(unseeded.col_indices.iter().copied())
        .collect();
    let seeded_options = PivotKernelOptions {
        seeds: &seeds,
        ..options
    };
    stats::reset();
    let seeded_source = CountingSource::new(dense_to_lazy(data.clone(), n, n));
    let seeded = LazyBlockRookKernel
        .factorize(&seeded_source, &seeded_options)
        .unwrap();

    // The seeds are used for the walks, not merely tolerated.
    let seeded_starts = stats::snapshot().seeded_starts;
    assert!(
        seeded_starts > seeds.len() / 2,
        "only {seeded_starts} of {} walks started on a seed",
        seeds.len()
    );
    assert!(
        seeded_source.requested() < unseeded_source.requested(),
        "seeded walk requested {} entries, unseeded {}",
        seeded_source.requested(),
        unseeded_source.requested()
    );

    // The seeded run stops by the same heuristic rule: its trailing entry is
    // the rejected rook pivot, below the tolerance (not a bound; see
    // `rook_tolerance_stop_on_the_1256_ridge_underestimates_the_true_error`).
    assert!(*seeded.pivot_errors.last().unwrap() < options.rel_tol * seeded.pivot_errors[0]);
}

/// The pivot sweep of `lazy_rank_cap_error_matches_dense_kernel_across_shapes_and_caps`,
/// factorized a second time with the first run's pivots as seeds.
///
/// On this sweep the walk converges on the same pivots wherever it starts, so
/// seeding has to change neither the selection nor the errors. That is a
/// property of these matrices, not a promise of seeding in general: elsewhere a
/// seed moves a walk to a different local rook (see
/// `lazy_seeded_walk_requests_less_than_the_unseeded_walk`). Seeds never add an
/// error guarantee.
#[test]
fn lazy_seeded_sweep_matches_the_unseeded_and_dense_kernels() {
    for (nrows, ncols) in [(4usize, 4usize), (6, 6), (8, 5), (5, 8), (12, 12), (20, 10)] {
        let full_rank = nrows.min(ncols);
        for cap in 1..full_rank {
            for seed in 1..=5u64 {
                let data = decreasing_pivot_matrix(nrows, ncols, seed);
                let dense_source = DenseMatrixSource::from_column_major(&data, nrows, ncols);
                let lazy_source = dense_to_lazy(data.clone(), nrows, ncols);
                let options = rank_cap_options(cap);

                let dense_out = DenseLuKernel.factorize(&dense_source, &options).unwrap();
                let unseeded = LazyBlockRookKernel
                    .factorize(&lazy_source, &options)
                    .unwrap();
                let seeds: Vec<(usize, usize)> = unseeded
                    .row_indices
                    .iter()
                    .copied()
                    .zip(unseeded.col_indices.iter().copied())
                    .collect();
                let seeded = LazyBlockRookKernel
                    .factorize(
                        &dense_to_lazy(data, nrows, ncols),
                        &PivotKernelOptions {
                            seeds: &seeds,
                            ..options
                        },
                    )
                    .unwrap();

                let case = format!("{nrows}x{ncols} cap {cap} seed {seed}");
                assert_eq!(seeded.row_indices, unseeded.row_indices, "{case}");
                assert_eq!(seeded.col_indices, unseeded.col_indices, "{case}");
                assert_eq!(seeded.row_indices, dense_out.row_indices, "{case}");
                assert_eq!(seeded.pivot_errors, unseeded.pivot_errors, "{case}");
                let dense_error = *dense_out.pivot_errors.last().unwrap();
                let seeded_error = *seeded.pivot_errors.last().unwrap();
                let relative = (dense_error - seeded_error).abs() / dense_error.abs();
                assert!(
                    relative <= 4.0 * f64::EPSILON,
                    "{case}: dense {dense_error:e} seeded {seeded_error:e}"
                );
            }
        }
    }
}

#[test]
fn lazy_walk_ignores_seeds_that_do_not_lead_anywhere() {
    // Seeds are a starting point, not a constraint. Nonsense seeds (pivots of
    // something else, in reverse) change where the walks begin, but on an
    // exact rank-6 matrix the factorization still finds rank 6 and
    // reconstructs the matrix.
    let (n, exact_rank) = (40usize, 6usize);
    let data = exact_rank_matrix(n, exact_rank, 11);
    let options = PivotKernelOptions {
        rel_tol: 1e-10,
        abs_tol: 0.0,
        ..PivotKernelOptions::default()
    };
    let nonsense: Vec<(usize, usize)> = (0..n).rev().map(|index| (index, index)).collect();
    let out = LazyBlockRookKernel
        .factorize(
            &dense_to_lazy(data.clone(), n, n),
            &PivotKernelOptions {
                seeds: &nonsense,
                ..options
            },
        )
        .unwrap();

    assert_eq!(out.rank, exact_rank);
    assert_eq!(out.row_indices.len(), exact_rank);
    assert_eq!(out.col_indices.len(), exact_rank);
    let scale = out.pivot_errors[0];
    let error = interpolation_max_error(&data, n, &out.row_indices, &out.col_indices);
    assert!(error < 1e-12 * scale, "reconstruction error {error:e}");
}

#[test]
fn the_seeded_entry_point_forwards_the_seeds_to_the_kernel() {
    // The kernel tests above call `LazyBlockRookKernel` directly. Callers reach
    // it through `matrix_luci_factors_from_blocks` and its seeded sibling, so
    // cover the forwarding as well: dropping the seeds on the way would leave
    // every test above green.
    use std::cell::Cell;
    let n = 120;
    let data = smooth_ridge_matrix(n);
    let options = crate::RrLUOptions {
        rel_tol: 1e-8,
        abs_tol: 0.0,
        ..crate::RrLUOptions::default()
    };
    let run = |seeds: &[(usize, usize)]| -> (usize, crate::MatrixLuciFactors<f64>) {
        let requested = Cell::new(0usize);
        let factors = crate::matrix_luci_factors_from_blocks_seeded(
            n,
            n,
            |rows, cols, out: &mut [f64]| {
                requested.set(requested.get() + rows.len() * cols.len());
                for (j, &col) in cols.iter().enumerate() {
                    for (i, &row) in rows.iter().enumerate() {
                        out[i + rows.len() * j] = data[row + n * col];
                    }
                }
            },
            seeds,
            options.clone(),
        )
        .unwrap();
        (requested.get(), factors)
    };

    let (unseeded_requested, unseeded) = run(&[]);
    let seeds: Vec<(usize, usize)> = unseeded
        .row_indices
        .iter()
        .copied()
        .zip(unseeded.col_indices.iter().copied())
        .collect();
    let (seeded_requested, seeded) = run(&seeds);

    assert_eq!(unseeded.rank, seeded.rank);
    assert!(
        seeded_requested < unseeded_requested,
        "seeded entry point requested {seeded_requested} entries, unseeded {unseeded_requested}"
    );
    assert!(
        *seeded.pivot_errors.last().unwrap() < options.rel_tol * seeded.pivot_errors[0],
        "the seeded factorization stopped above the tolerance"
    );
}
