//! Complete-pivoting LU of a square dense matrix that stops at a rank cap.
//!
//! The factorization is faer's complete-pivoting LU, run in place on the
//! matrix's own column-major buffer with `Par::Seq`. A rank cap is handed to the
//! elimination itself, so a rank-`r` answer costs `O(n^2 r)` instead of the
//! `O(n^3)` of a full factorization truncated afterwards.
//!
//! Only the cap is handed to faer. Which pivots a caller accepts - absolute and
//! relative tolerances, degenerate-pivot rules - is the caller's decision, made
//! on the diagonal this module returns: faer's own tolerance rule rejects a
//! pivot *at* the threshold and measures it against the first pivot, which is
//! not the rule every caller wants.
//!
//! Two properties of the elimination callers can rely on:
//!
//! - The pivot search maximises `abs1` (`|re| + |im|` for complex scalars) over
//!   the trailing block, so the next pivot is whatever faer selects, not a rule
//!   a caller re-derives from the diagonal (`diag(10, 3, 2.5 + 2.5i)` takes `3`
//!   second).
//! - A zero or subnormal Schur complement stops the elimination before the cap
//!   does. [`CappedFullPivLu::rank`] then reports that earlier stop, and the
//!   trailing diagonal is left un-eliminated.

use anyhow::anyhow;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::full_pivoting::factor::{self as faer_flu, FullPivLuParams};
use faer::{Auto, Par};
use num_complex::{Complex32, Complex64};

use crate::backend::BackendLinalgError;
use crate::matrix::Matrix;

mod sealed {
    use super::{CappedFullPivLu, Matrix};

    /// The in-place factorization each supported scalar provides.
    pub trait Factor: Copy + Sized {
        fn factor_in_place(a: Matrix<Self>, max_rank: Option<usize>) -> CappedFullPivLu<Self>;
    }
}

/// Scalar types accepted by [`capped_full_piv_lu`]: `f32`, `f64`, `Complex32`
/// and `Complex64`.
///
/// The trait is sealed; the factorization behind it is the backend's.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{capped_full_piv_lu, CappedFullPivLuScalar, Matrix};
///
/// fn leading_pivot<T: CappedFullPivLuScalar>(a: Matrix<T>) -> Option<T> {
///     capped_full_piv_lu(a, Some(1)).ok()?.diagonal().first().copied()
/// }
/// let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 5.0]);
/// assert_eq!(leading_pivot(a), Some(5.0));
/// ```
pub trait CappedFullPivLuScalar: sealed::Factor {}

/// The pivots a [`capped_full_piv_lu`] call took, and the diagonal it left.
///
/// Holds no factor matrices: the row and column permutations, the in-place
/// diagonal, and the number of pivots the elimination actually took.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{capped_full_piv_lu, Matrix};
///
/// let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 5.0]);
/// let lu = capped_full_piv_lu(a, None)?;
/// assert_eq!(lu.rank(), 2);
/// assert_eq!(lu.row_perm(), &[1, 0]);
/// assert_eq!(lu.col_perm(), &[1, 0]);
/// assert_eq!(lu.diagonal(), &[5.0, 1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct CappedFullPivLu<T> {
    row_perm: Vec<usize>,
    col_perm: Vec<usize>,
    diagonal: Vec<T>,
    rank: usize,
}

impl<T> CappedFullPivLu<T> {
    /// Row permutation: `row_perm()[k]` is the input row moved to position `k`.
    ///
    /// Has `n` entries; the first [`Self::rank`] are the rows of the pivots taken.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{capped_full_piv_lu, Matrix};
    ///
    /// let a = Matrix::from_col_major_vec(2, 2, vec![0.0_f64, 3.0, 1.0, 0.0]);
    /// let lu = capped_full_piv_lu(a, Some(1))?;
    /// assert_eq!(lu.row_perm()[0], 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn row_perm(&self) -> &[usize] {
        &self.row_perm
    }

    /// Column permutation: `col_perm()[k]` is the input column moved to position `k`.
    ///
    /// Has `n` entries; the first [`Self::rank`] are the columns of the pivots taken.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{capped_full_piv_lu, Matrix};
    ///
    /// let a = Matrix::from_col_major_vec(2, 2, vec![0.0_f64, 3.0, 1.0, 0.0]);
    /// let lu = capped_full_piv_lu(a, Some(1))?;
    /// assert_eq!(lu.col_perm()[0], 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn col_perm(&self) -> &[usize] {
        &self.col_perm
    }

    /// The diagonal of the in-place factorization, `n` entries.
    ///
    /// The first [`Self::rank`] entries are the pivots taken, in order. The rest
    /// are the trailing block's diagonal as the elimination left it: the Schur
    /// complement's diagonal after a cap, the untouched input diagonal (under
    /// the permutations) after a zero or subnormal stop.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{capped_full_piv_lu, Matrix};
    ///
    /// let a = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 0.0, 0.0, -7.0]);
    /// let lu = capped_full_piv_lu(a, None)?;
    /// assert_eq!(lu.diagonal(), &[-7.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn diagonal(&self) -> &[T] {
        &self.diagonal
    }

    /// Number of pivots the elimination took.
    ///
    /// This is the raw count, not one clamped to the cap: `n` when uncapped,
    /// the cap when the cap stops the elimination, and fewer when a zero or
    /// subnormal Schur complement stops it first.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{capped_full_piv_lu, Matrix};
    ///
    /// let a = Matrix::from_col_major_vec(3, 3, vec![
    ///     4.0_f64, 0.0, 0.0,
    ///     0.0, 2.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// ]);
    /// assert_eq!(capped_full_piv_lu(a.clone(), None)?.rank(), 3);
    /// assert_eq!(capped_full_piv_lu(a, Some(2))?.rank(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Complete-pivoting LU of a square matrix, stopped after at most `max_rank` pivots.
///
/// The elimination runs in place on `a`'s column-major buffer, sequentially.
/// `None` factors to full rank; `Some(r)` stops once `r` pivots are taken, so
/// the work is `O(n^2 r)`. Pivot acceptance (tolerances, degenerate pivots) is
/// left to the caller, who reads it from [`CappedFullPivLu::diagonal`].
///
/// Non-finite entries are not checked here: validate the input first if a
/// non-finite value must be refused.
///
/// # Errors
///
/// Returns [`BackendLinalgError`] when `a` is not square.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{capped_full_piv_lu, Matrix};
///
/// // diag(1, 4, 2): complete pivoting takes 4, then 2, then 1.
/// let a = Matrix::from_col_major_vec(3, 3, vec![
///     1.0_f64, 0.0, 0.0,
///     0.0, 4.0, 0.0,
///     0.0, 0.0, 2.0,
/// ]);
/// let lu = capped_full_piv_lu(a, Some(2))?;
/// assert_eq!(lu.rank(), 2);
/// assert_eq!(&lu.row_perm()[..2], &[1, 2]);
/// assert_eq!(&lu.col_perm()[..2], &[1, 2]);
/// assert_eq!(&lu.diagonal()[..2], &[4.0, 2.0]);
/// // The trailing entry is the remaining Schur complement.
/// assert_eq!(lu.diagonal()[2], 1.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn capped_full_piv_lu<T: CappedFullPivLuScalar>(
    a: Matrix<T>,
    max_rank: Option<usize>,
) -> Result<CappedFullPivLu<T>, BackendLinalgError> {
    if a.nrows() != a.ncols() {
        return Err(anyhow!(
            "capped_full_piv_lu needs a square matrix, got {}x{}",
            a.nrows(),
            a.ncols()
        )
        .into());
    }
    Ok(T::factor_in_place(a, max_rank))
}

fn factor_with_faer<T>(mut a: Matrix<T>, max_rank: Option<usize>) -> CappedFullPivLu<T>
where
    T: faer::traits::ComplexField + Copy,
{
    let n = a.nrows();
    if n == 0 {
        return CappedFullPivLu {
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            diagonal: Vec::new(),
            rank: 0,
        };
    }
    let params = FullPivLuParams {
        max_rank,
        ..<FullPivLuParams as Auto<T>>::auto()
    };
    let (mut row_perm, mut row_perm_inv) = (vec![0usize; n], vec![0usize; n]);
    let (mut col_perm, mut col_perm_inv) = (vec![0usize; n], vec![0usize; n]);
    let mut scratch = MemBuffer::new(faer_flu::lu_in_place_scratch::<usize, T>(
        n,
        n,
        Par::Seq.shape(),
        params.into(),
    ));
    let buffer = a.as_col_major_mut_slice();
    let (info, _, _) = faer_flu::lu_in_place(
        faer::MatMut::from_column_major_slice_mut(buffer, n, n),
        &mut row_perm,
        &mut row_perm_inv,
        &mut col_perm,
        &mut col_perm_inv,
        Par::Seq,
        MemStack::new(&mut scratch),
        params.into(),
    );
    let diagonal = (0..n).map(|i| buffer[i + n * i]).collect();
    CappedFullPivLu {
        row_perm,
        col_perm,
        diagonal,
        rank: info.rank.min(n),
    }
}

macro_rules! impl_capped_full_piv_lu_scalar {
    ($($t:ty),*) => {$(
        impl sealed::Factor for $t {
            fn factor_in_place(a: Matrix<Self>, max_rank: Option<usize>) -> CappedFullPivLu<Self> {
                factor_with_faer(a, max_rank)
            }
        }
        impl CappedFullPivLuScalar for $t {}
    )*};
}

impl_capped_full_piv_lu_scalar!(f32, f64, Complex32, Complex64);

#[cfg(test)]
mod tests;
