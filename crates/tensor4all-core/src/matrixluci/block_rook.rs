//! Lazy pivot-kernel implementations.

use crate::matrixlu::pivot_is_undividable;
use crate::matrixluci::error::MatrixLuciError;
use crate::matrixluci::factors::{load_block, subtract_inplace};
use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::scalar::MatrixLuciScalar;
use crate::matrixluci::source::CandidateMatrixSource;
use crate::matrixluci::types::{PivotKernelOptions, PivotSelectionCore};
use crate::matrixluci::Result;
use num_complex::{Complex32, Complex64};
use tensor4all_tensorbackend::{mat_mul, solve_matrix, Matrix};

/// Lazy pivot kernel based on residual row/column rook search.
///
/// Selects pivots by computing residual blocks on demand, avoiding full
/// matrix materialization. Suitable for large matrices accessed via
/// [`LazyMatrixSource`](super::source::LazyMatrixSource).
/// An exactly zero starting residual column is skipped. Certifying an all-zero
/// residual may inspect every remaining entry, one column at a time.
///
/// Each pivot is found by an O(d) rook walk: alternate residual column and row
/// argmax until a row and a column agree. The walk moves only to a strictly
/// larger magnitude, so it ends at the first entry that no row or column step
/// improves. A rook pivot is therefore the largest
/// entry of its own residual row and column only, and its magnitude can be far
/// below the maximum of the remaining residual.
///
/// The stop is heuristic. The factorization ends when the current rook pivot
/// passes the stopping test (tolerance or undividable pivot) or the rank cap is
/// reached, without scanning the remaining block. Before a tolerance stop, a
/// few more walks start from columns spread over the remaining block, and the
/// first pivot they find above the threshold is taken instead; this makes a
/// false stop on one local maximum less likely, but does not exclude it. The trailing `pivot_errors`
/// entry is that last rook pivot, the one the factorization rejected (at a rank
/// cap, one extra walk measures it), or `0.0` at full rank or when the walk
/// found the remaining residual exactly zero. It is **not** an upper bound on
/// the remaining residual, and a non-finite entry the walks never visit is not
/// detected. Use the dense kernel (`PivotSearchStrategy::Full` in TCI) when a
/// certified bound is required.
///
/// A walk starts on the pivot column the previous factorization of the same
/// candidate matrix chose ([`PivotKernelOptions::seeds`]), as `arrlu` starts
/// from its `I0`/`J0`, instead of rediscovering it from the first remaining
/// column. Seeds change where walks start, and so the pivot history; they do
/// not add any error guarantee.
#[derive(Default)]
pub struct LazyBlockRookKernel;

/// Extra walks from spread-out starting columns before a tolerance stop.
///
/// On the tensor4all-rs#1256 ridge and a 4-site ridge, two walks remove the
/// false stops that one walk makes; four and eight walks give the same result.
const STOP_CHECK_WALKS: usize = 2;

/// Subtract `a_rj * pivot^{-1} * a_ic` from `residual` in place.
fn eliminate<T: MatrixLuciScalar>(
    residual: &mut Matrix<T>,
    pivot: &Matrix<T>,
    a_rj: &Matrix<T>,
    a_ic: &Matrix<T>,
) -> Result<()> {
    let solved = solve_matrix(pivot, a_ic).map_err(|err| MatrixLuciError::InvalidArgument {
        message: format!("residual pivot solve failed: {err}"),
    })?;
    let approx = mat_mul(a_rj, &solved).map_err(|err| MatrixLuciError::InvalidArgument {
        message: format!("residual multiplication failed: {err}"),
    })?;
    subtract_inplace(residual, &approx);
    Ok(())
}

/// Copy column `col` of `matrix` into an `nrows x 1` matrix.
fn column_of<T: MatrixLuciScalar>(matrix: &Matrix<T>, col: usize) -> Matrix<T> {
    let data = (0..matrix.nrows()).map(|row| matrix[[row, col]]).collect();
    Matrix::from_col_major_vec(matrix.nrows(), 1, data)
}

/// Copy row `row` of `matrix` into a `1 x ncols` matrix.
fn row_of<T: MatrixLuciScalar>(matrix: &Matrix<T>, row: usize) -> Matrix<T> {
    let data = (0..matrix.ncols()).map(|col| matrix[[row, col]]).collect();
    Matrix::from_col_major_vec(1, matrix.ncols(), data)
}

/// The blocks of `A` that stay fixed during one rook walk.
///
/// Within a walk only the current row and column move; the selected and
/// remaining index sets do not. The pivot block `A[I, J]`, the pivot columns
/// `A[remaining_rows, J]` and the pivot rows `A[I, remaining_cols]` are
/// therefore read once per walk, and each column or row block takes its
/// operands from them. The operands passed to the solve and the product have
/// the same shapes and values as a fresh load, so the residuals are bitwise
/// those of a walk that reloads every block.
struct WalkOperands<T: MatrixLuciScalar> {
    /// `A[I, J]`.
    pivot: Matrix<T>,
    /// `A[remaining_rows, J]`.
    pivot_cols: Matrix<T>,
    /// `A[I, remaining_cols]`.
    pivot_rows: Matrix<T>,
}

impl<T: MatrixLuciScalar> WalkOperands<T> {
    /// Load the walk invariants; `None` before the first pivot, when the
    /// residual is the matrix itself.
    fn load<S: CandidateMatrixSource<T>>(
        source: &S,
        remaining_rows: &[usize],
        remaining_cols: &[usize],
        selected_rows: &[usize],
        selected_cols: &[usize],
    ) -> Option<Self> {
        if selected_rows.is_empty() {
            return None;
        }
        Some(Self {
            pivot: load_block(source, selected_rows, selected_cols),
            pivot_cols: load_block(source, remaining_rows, selected_cols),
            pivot_rows: load_block(source, selected_rows, remaining_cols),
        })
    }
}

/// Residual column `A[remaining_rows, remaining_cols[col_pos]]`.
fn residual_column<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    col_pos: usize,
    operands: Option<&WalkOperands<T>>,
) -> Result<Matrix<T>> {
    let mut residual = load_block(source, remaining_rows, &[remaining_cols[col_pos]]);
    if let Some(ops) = operands {
        let a_ic = column_of(&ops.pivot_rows, col_pos);
        eliminate(&mut residual, &ops.pivot, &ops.pivot_cols, &a_ic)?;
    }
    Ok(residual)
}

/// Residual row `A[remaining_rows[row_pos], remaining_cols]`.
fn residual_row<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    row_pos: usize,
    operands: Option<&WalkOperands<T>>,
) -> Result<Matrix<T>> {
    let mut residual = load_block(source, &[remaining_rows[row_pos]], remaining_cols);
    if let Some(ops) = operands {
        let a_rj = row_of(&ops.pivot_cols, row_pos);
        eliminate(&mut residual, &ops.pivot, &a_rj, &ops.pivot_rows)?;
    }
    Ok(residual)
}

fn argmax_abs<T: MatrixLuciScalar>(matrix: &Matrix<T>) -> (usize, usize, f64) {
    let mut best_row = 0usize;
    let mut best_col = 0usize;
    let mut best_abs = -1.0f64;
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[[row, col]].abs_val();
            if !value.is_finite() {
                return (row, col, value);
            }
            if value > best_abs {
                best_row = row;
                best_col = col;
                best_abs = value;
            }
        }
    }
    (best_row, best_col, best_abs.max(0.0))
}

fn remaining_indices(total: usize, selected: &[usize]) -> Vec<usize> {
    let mut used = vec![false; total];
    for &idx in selected {
        used[idx] = true;
    }
    (0..total).filter(|&idx| !used[idx]).collect()
}

/// Position of `index` in `indices`, which is ascending.
fn position_of(indices: &[usize], index: usize) -> Option<usize> {
    indices.binary_search(&index).ok()
}

/// The first seed whose row and column are both still unselected.
///
/// Seeds arrive in the order of the previous factorization's pivots, so taking
/// them in order presents the k-th walk with the k-th pivot of that run.
fn next_seed(
    seeds: &mut impl Iterator<Item = (usize, usize)>,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
) -> Option<(usize, usize)> {
    seeds.find(|&(row, col)| {
        position_of(remaining_rows, row).is_some() && position_of(remaining_cols, col).is_some()
    })
}

fn rook_pivot<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
    seed: Option<(usize, usize)>,
) -> Result<(usize, usize, f64)> {
    let operands = WalkOperands::load(
        source,
        remaining_rows,
        remaining_cols,
        selected_rows,
        selected_cols,
    );
    let operands = operands.as_ref();
    let mut current_col_pos;
    let mut current_row_pos;
    // Magnitude of the entry the walk stands on.
    let mut current_abs;
    // A seeded walk starts on the column the previous factorization pivoted on,
    // which is where `arrlu` starts from its `J0`. The scan below stays the
    // fallback: it is what a first factorization of a matrix has, and what a
    // seeded column with no residual left falls back to.
    let mut seeded = seed.and_then(|(_, col)| position_of(remaining_cols, col));
    let mut columns = 0..remaining_cols.len();
    loop {
        let col_pos = match seeded.take() {
            Some(col_pos) => col_pos,
            None => match columns.next() {
                Some(col_pos) => col_pos,
                // INVARIANT: a zero fiber does not certify a zero residual.
                // Scan lazily until a nonzero starting column is found; never
                // materialize the full remaining matrix. The ordinary
                // nonzero-column path performs no extra block evaluations.
                None => return Ok((remaining_rows[0], remaining_cols[0], 0.0)),
            },
        };
        current_col_pos = col_pos;
        let col_residual =
            residual_column(source, remaining_rows, remaining_cols, col_pos, operands)?;
        let (best_row_pos, _, best_abs) = argmax_abs(&col_residual);
        // Skip only an exactly zero fiber. Non-finite magnitudes must reach
        // the kernel's existing numerical-error check, not look like zeros.
        if best_abs != 0.0 {
            current_row_pos = best_row_pos;
            current_abs = best_abs;
            break;
        }
    }
    if !current_abs.is_finite() {
        return Ok((
            remaining_rows[current_row_pos],
            remaining_cols[current_col_pos],
            current_abs,
        ));
    }

    // The walk moves only to a strictly larger magnitude. At roundoff level the
    // same residual entry computed from its row and from its column can differ
    // in the last bits, so "the row maximum is in another column" does not mean
    // that column holds a larger entry; following it lets the walk cycle among
    // roundoff-sized entries until `max_steps`.
    let max_steps = remaining_rows.len() + remaining_cols.len() + 1;
    for _ in 0..max_steps {
        #[cfg(test)]
        stats::update(|c| c.walk_steps += 1);
        let row_residual = residual_row(
            source,
            remaining_rows,
            remaining_cols,
            current_row_pos,
            operands,
        )?;
        let (_, best_col_pos, best_abs) = argmax_abs(&row_residual);
        if !best_abs.is_finite() {
            return Ok((
                remaining_rows[current_row_pos],
                remaining_cols[best_col_pos],
                best_abs,
            ));
        }

        if best_col_pos == current_col_pos || best_abs <= current_abs {
            return Ok((
                remaining_rows[current_row_pos],
                remaining_cols[current_col_pos],
                current_abs.max(best_abs),
            ));
        }
        current_col_pos = best_col_pos;
        current_abs = best_abs;
        let col_residual = residual_column(
            source,
            remaining_rows,
            remaining_cols,
            current_col_pos,
            operands,
        )?;
        let (best_row_pos, _, col_best_abs) = argmax_abs(&col_residual);
        if !col_best_abs.is_finite() {
            return Ok((
                remaining_rows[best_row_pos],
                remaining_cols[current_col_pos],
                col_best_abs,
            ));
        }
        if col_best_abs > current_abs {
            current_row_pos = best_row_pos;
            current_abs = col_best_abs;
        }
    }

    let row_residual = residual_row(
        source,
        remaining_rows,
        remaining_cols,
        current_row_pos,
        operands,
    )?;
    let (_, best_col_pos, best_abs) = argmax_abs(&row_residual);
    Ok((
        remaining_rows[current_row_pos],
        remaining_cols[best_col_pos],
        best_abs,
    ))
}

/// Counters the tests read to observe the walks.
///
/// Thread-local, because the test harness factorizes on several threads at
/// once and each of these describes one factorization.
#[cfg(test)]
pub(super) mod stats {
    use std::cell::Cell;

    /// How many walks started on a seed, how many row steps all walks took,
    /// and how many stop-check walks ran.
    #[derive(Default, Clone, Copy)]
    pub(super) struct Counters {
        pub(super) seeded_starts: usize,
        pub(super) walk_steps: usize,
        pub(super) stop_check_walks: usize,
    }

    thread_local! {
        static COUNTERS: Cell<Counters> = const {
            Cell::new(Counters {
                seeded_starts: 0,
                walk_steps: 0,
                stop_check_walks: 0,
            })
        };
    }

    pub(super) fn update(change: impl FnOnce(&mut Counters)) {
        COUNTERS.with(|counters| {
            let mut value = counters.get();
            change(&mut value);
            counters.set(value);
        });
    }

    pub(super) fn reset() {
        COUNTERS.with(|counters| counters.set(Counters::default()));
    }

    pub(super) fn snapshot() -> Counters {
        COUNTERS.with(Cell::get)
    }
}

fn factorize_lazy<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    options: &PivotKernelOptions<'_>,
) -> Result<PivotSelectionCore> {
    let nrows = source.nrows();
    let ncols = source.ncols();
    let full_rank = nrows.min(ncols);
    if full_rank == 0 {
        return Ok(PivotSelectionCore {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            pivot_errors: vec![0.0],
            rank: 0,
        });
    }

    let max_bond_dim = options.max_bond_dim.min(full_rank);
    let mut selected_rows = Vec::with_capacity(max_bond_dim);
    let mut selected_cols = Vec::with_capacity(max_bond_dim);
    let mut accepted = Vec::with_capacity(max_bond_dim + 1);
    let mut max_error = 0.0f64;
    let mut seeds = options.seeds.iter().copied();

    // Every exit reports the last rook pivot, never a scan of the remaining
    // block: the trailing entry is the pivot this loop rejects (tolerance,
    // undividable pivot or rank cap), 0 when the walk found the remaining
    // residual exactly zero, and 0 at full rank, where nothing remains.
    let last_error = loop {
        let remaining_rows = remaining_indices(nrows, &selected_rows);
        let remaining_cols = remaining_indices(ncols, &selected_cols);
        if remaining_rows.is_empty() || remaining_cols.is_empty() {
            break 0.0;
        }

        let seed = next_seed(&mut seeds, &remaining_rows, &remaining_cols);
        #[cfg(test)]
        if seed.is_some() {
            stats::update(|c| c.seeded_starts += 1);
        }
        let (pivot_row, pivot_col, pivot_abs) = rook_pivot(
            source,
            &remaining_rows,
            &remaining_cols,
            &selected_rows,
            &selected_cols,
            seed,
        )?;
        // Checked before the stopping test: NaN compares false against every
        // threshold and must not be read as a converged residual.
        if !pivot_abs.is_finite() {
            return Err(MatrixLuciError::NaNEncountered {
                matrix: "rook residual",
            });
        }
        if pivot_abs == 0.0 {
            // `rook_pivot` returns an exact zero only after it has found
            // every remaining column to be zero.
            break 0.0;
        }

        // Heuristic stop: a rook pivot is the largest entry of its own residual
        // row and column only, so a small one does not bound the rest of the
        // block. Callers that need that bound use the dense kernel
        // (`PivotSearchStrategy::Full`). At the rank cap the walk above is the
        // one extra O(d) walk that measures the rejected pivot.
        let n_selected = selected_rows.len();
        let passes_stop = |abs: f64| abs < options.rel_tol * max_error || abs < options.abs_tol;
        let mut below_tolerance = n_selected > 0 && passes_stop(pivot_abs);
        let (mut pivot_row, mut pivot_col, mut pivot_abs) = (pivot_row, pivot_col, pivot_abs);
        // Before a tolerance stop, walk again from a few columns spread over
        // the remaining block. A walk that finds a pivot above the threshold
        // replaces the stopping pivot; the factorization stops only if none
        // does. This is still a heuristic, not a bound.
        if below_tolerance && n_selected < max_bond_dim {
            for k in 1..=STOP_CHECK_WALKS {
                let start = remaining_cols[k * remaining_cols.len() / (STOP_CHECK_WALKS + 1)];
                #[cfg(test)]
                stats::update(|c| c.stop_check_walks += 1);
                let (row, col, abs) = rook_pivot(
                    source,
                    &remaining_rows,
                    &remaining_cols,
                    &selected_rows,
                    &selected_cols,
                    Some((remaining_rows[0], start)),
                )?;
                if !abs.is_finite() {
                    return Err(MatrixLuciError::NaNEncountered {
                        matrix: "rook residual",
                    });
                }
                if !passes_stop(abs) {
                    (pivot_row, pivot_col, pivot_abs) = (row, col, abs);
                    below_tolerance = false;
                    break;
                }
            }
        }
        if n_selected == max_bond_dim || below_tolerance || pivot_is_undividable::<T>(pivot_abs) {
            break pivot_abs;
        }

        max_error = max_error.max(pivot_abs);
        selected_rows.push(pivot_row);
        selected_cols.push(pivot_col);
        accepted.push(pivot_abs);
    };
    let rank = selected_rows.len();
    accepted.push(last_error);

    Ok(PivotSelectionCore {
        row_indices: selected_rows,
        col_indices: selected_cols,
        pivot_errors: accepted,
        rank,
    })
}

macro_rules! impl_lazy_block_rook_kernel {
    ($t:ty) => {
        impl PivotKernel<$t> for LazyBlockRookKernel {
            fn factorize<S: CandidateMatrixSource<$t>>(
                &self,
                source: &S,
                options: &PivotKernelOptions<'_>,
            ) -> Result<PivotSelectionCore> {
                factorize_lazy(source, options)
            }
        }
    };
}

impl_lazy_block_rook_kernel!(f32);
impl_lazy_block_rook_kernel!(f64);
impl_lazy_block_rook_kernel!(Complex32);
impl_lazy_block_rook_kernel!(Complex64);

#[cfg(test)]
mod tests;
