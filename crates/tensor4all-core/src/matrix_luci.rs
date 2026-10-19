//! Matrix LU-based Cross Interpolation (MatrixLUCI) implementation.
//!
//! [`MatrixLUCI`] provides a higher-level [`Matrix`] API over the lower-level
//! `matrixluci` substrate. It decomposes a matrix into left and right factors
//! via LU cross interpolation and implements [`AbstractMatrixCI`].

use crate::error::{MatrixCIError, Result};
use crate::matrixlu::{
    rrlu, rrlu_left_mut, rrlu_mut, validate_luci_options, validate_owned_luci_input, RrLU,
    RrLUOptions,
};
use crate::matrixluci::block_rook::LazyBlockRookKernel;
use crate::matrixluci::factors::CrossFactors;
use crate::matrixluci::source::LazyMatrixSource;
use crate::matrixluci::types::{PivotKernelOptions, PivotSelectionCore};
use crate::matrixluci::PivotKernel;
use crate::scalar::Scalar;
use crate::traits::AbstractMatrixCI;
use tensor4all_tensorbackend::{
    explicit::Session, mat_mul_owned, submatrix, triangular_solve_matrix_owned,
    CpuExecutionContext, Matrix,
};

/// Where MatrixLUCI factor assembly runs its matrix products and solves.
///
/// `Global` keeps the compatibility routes of the legacy entry points;
/// `Context` runs every operation in one explicit concrete session of the
/// caller's context and never consults the process-global default.
pub(crate) enum Linalg<'a, 'session> {
    Global,
    Context(&'a CpuExecutionContext),
    /// A caller-supplied session, entered by the caller rather than opened here.
    ///
    /// This is the form a worker lane uses: the session was entered by the phase that owns
    /// the worker, so re-entering a context here would either be refused or pick another
    /// backend.
    Session(&'a mut Session<'session>),
}

impl Linalg<'_, '_> {
    fn mat_mul<T>(&mut self, a: Matrix<T>, b: Matrix<T>) -> std::result::Result<Matrix<T>, String>
    where
        T: Scalar + crate::MatrixLuciScalar,
    {
        match self {
            Self::Global => mat_mul_owned(a, b).map_err(|err| err.to_string()),
            Self::Context(context) => context
                .with_concrete_session(|session| session.mat_mul(&a, &b))
                .map_err(|err| err.to_string())?
                .map_err(|err| err.to_string()),
            Self::Session(session) => session.mat_mul(&a, &b).map_err(|err| err.to_string()),
        }
    }

    fn triangular_solve<T>(
        &mut self,
        a: Matrix<T>,
        b: Matrix<T>,
        left_side: bool,
        lower: bool,
    ) -> std::result::Result<Matrix<T>, String>
    where
        T: Scalar + crate::MatrixLuciScalar,
    {
        match self {
            Self::Global => triangular_solve_matrix_owned(a, b, left_side, lower, false, false)
                .map_err(|err| err.to_string()),
            Self::Context(context) => context
                .with_concrete_session(|session| {
                    T::matrix_luci_triangular_solve_in(session, &a, &b, left_side, lower)
                })
                .map_err(|err| err.to_string())?
                .map_err(|err| err.to_string()),
            Self::Session(session) => {
                T::matrix_luci_triangular_solve_in(session, &a, &b, left_side, lower)
                    .map_err(|err| err.to_string())
            }
        }
    }
}

/// Matrix LU-based Cross Interpolation.
///
/// This is a higher-level [`Matrix`] wrapper around the lower-level `matrixluci`
/// substrate.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{AbstractMatrixCI, MatrixLUCI};
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
///     vec![7.0, 8.0, 9.0],
/// ]);
///
/// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
/// // The approximation must have at most rank min(nrows, ncols)
/// assert!(ci.rank() <= 3);
/// // Reconstructed matrix should match original at pivot positions
/// let row_indices = ci.row_indices().to_vec();
/// let col_indices = ci.col_indices().to_vec();
/// for (&i, &j) in row_indices.iter().zip(col_indices.iter()) {
///     let approx = ci.evaluate(i, j);
///     let exact = m[[i, j]];
///     assert!((approx - exact).abs() < 1e-10);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MatrixLUCI<T: Scalar + crate::MatrixLuciScalar> {
    nrows: usize,
    ncols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    left: Matrix<T>,
    right: Matrix<T>,
    pivot_errors: Vec<f64>,
}

/// High-level factors produced by MatrixLUCI.
///
/// This is the public result of the LUCI factorization facade. It exposes
/// the selected pivot metadata and the left and right factors
/// needed by higher-level tensor-network code without exposing the low-level
/// pivot kernel substrate.
///
/// Related types: [`MatrixLUCI`] is the owning CI wrapper, while
/// [`MatrixACA`](crate::MatrixACA) and [`RrLU`](crate::RrLU) are related
/// matrix factorization entry points.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrix_luci_factors_from_matrix;
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0],
///     vec![3.0, 4.0],
/// ]);
/// let factors = matrix_luci_factors_from_matrix(&m, None).unwrap();
/// assert!(factors.rank >= 1);
/// assert_eq!(factors.row_indices.len(), factors.rank);
/// assert_eq!(factors.left.nrows(), m.nrows());
/// assert_eq!(factors.right.ncols(), m.ncols());
/// ```
#[derive(Debug, Clone)]
pub struct MatrixLuciFactors<T> {
    /// Selected row indices.
    pub row_indices: Vec<usize>,
    /// Selected column indices.
    pub col_indices: Vec<usize>,
    /// Pivot error history, or one measured maximum residual for fixed pivots.
    pub pivot_errors: Vec<f64>,
    /// Selected rank.
    pub rank: usize,
    /// Left factor.
    pub left: Matrix<T>,
    /// Right factor.
    pub right: Matrix<T>,
}

/// Row interpolation `A ~= interpolation * A[rows, :]` selected by MatrixLUCI.
/// Unlike [`MatrixLuciFactors`], this does not build an unused right factor.
///
/// # Examples
/// ```
/// use tensor4all_core::matrix_luci_row_interpolation_owned_in;
/// use tensor4all_tensorbackend::{default_cpu_execution_context, Matrix};
/// let a = Matrix::from_col_major_vec(2, 1, vec![2.0_f64, 4.0]);
/// let id = matrix_luci_row_interpolation_owned_in(a, None, &default_cpu_execution_context())?;
/// assert_eq!(id.rows, vec![1]);
/// assert_eq!(id.interpolation.as_col_major_slice(), &[0.5, 1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct MatrixLuciRowInterpolation<T> {
    /// Original matrix row indices, in pivot order.
    pub rows: Vec<usize>,
    /// Row weights, with an identity submatrix at the selected rows.
    pub interpolation: Matrix<T>,
    /// Accepted LU pivot magnitudes followed by the maximum absolute entry of
    /// the remaining Schur complement. This is a local matrix diagnostic.
    pub pivot_magnitudes: Vec<f64>,
}

/// Selects a row interpolant without constructing the complementary right factor.
///
/// Consumes column-major `a`, uses `options` for pivot rank/tolerances, and
/// performs the factor solve in the supplied CPU `context`. `left_orthogonal`
/// is always set to true because this operation returns row interpolation.
/// A zero matrix yields zero columns; the caller chooses its zero-rank convention.
///
/// # Errors
/// Returns [`MatrixCIError::InvalidArgument`] for negative or nonfinite
/// tolerances, infinite input magnitudes, or backend solve failures; returns
/// [`MatrixCIError::NaNEncountered`] for NaN input or nonfinite factors or
/// residual magnitudes.
///
/// # Examples
/// ```
/// use tensor4all_core::{matrix_luci_row_interpolation_owned_in, RrLUOptions};
/// use tensor4all_tensorbackend::{default_cpu_execution_context, Matrix};
/// let a = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 6.0]);
/// let id = matrix_luci_row_interpolation_owned_in(a, Some(RrLUOptions::default()),
///     &default_cpu_execution_context())?;
/// assert_eq!(id.rows, vec![1]);
/// assert_eq!(id.interpolation.as_col_major_slice(), &[0.5, 1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn matrix_luci_row_interpolation_owned_in<T>(
    mut a: Matrix<T>,
    options: Option<RrLUOptions>,
    context: &CpuExecutionContext,
) -> Result<MatrixLuciRowInterpolation<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let mut options = options.unwrap_or_default();
    options.left_orthogonal = true;
    validate_owned_luci_input(&a, &options)?;
    let lu = rrlu_left_mut(&mut a, Some(options))?;
    let rank = lu.n_pivot;
    Ok(MatrixLuciRowInterpolation {
        rows: lu.row_permutation[..rank].to_vec(),
        interpolation: cols_times_pivot_solve(
            &lu.l,
            &lu.row_permutation,
            rank,
            &mut Linalg::Context(context),
        )?,
        pivot_magnitudes: lu.pivot_magnitudes,
    })
}

// Validation for the new owned/context facades lives with the rrLU entry
// points (`matrixlu::validate_owned_luci_input`) so the selection-only path
// checks inputs and tolerances identically.

pub(crate) fn map_backend_error(err: crate::matrixluci::MatrixLuciError) -> MatrixCIError {
    match err {
        crate::matrixluci::MatrixLuciError::NaNEncountered { matrix } => {
            MatrixCIError::NaNEncountered {
                matrix: matrix.to_string(),
            }
        }
        crate::matrixluci::MatrixLuciError::InvalidArgument { message } => {
            MatrixCIError::InvalidArgument { message }
        }
    }
}

fn factors_to_public<T>(
    selection: PivotSelectionCore,
    factors: CrossFactors<T>,
    left_orthogonal: bool,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let left = if left_orthogonal {
        factors.cols_solve_pivot().map_err(map_backend_error)?
    } else {
        factors.pivot_cols.clone()
    };
    let right = if left_orthogonal {
        factors.pivot_rows.clone()
    } else {
        factors.solve_pivot_rows().map_err(map_backend_error)?
    };

    Ok(MatrixLuciFactors {
        row_indices: selection.row_indices,
        col_indices: selection.col_indices,
        pivot_errors: selection.pivot_errors,
        rank: selection.rank,
        left,
        right,
    })
}

fn backend_linalg_error(message: impl Into<String>) -> MatrixCIError {
    MatrixCIError::InvalidArgument {
        message: message.into(),
    }
}

fn index_range(start: usize, end: usize) -> Vec<usize> {
    (start..end).collect()
}

fn identity_rect<T: Scalar>(nrows: usize, ncols: usize) -> Matrix<T> {
    let mut result = Matrix::zeros(nrows, ncols);
    for i in 0..nrows.min(ncols) {
        result[[i, i]] = T::one();
    }
    result
}

fn apply_row_permutation<T: Scalar>(matrix: &Matrix<T>, permutation: &[usize]) -> Matrix<T> {
    let mut result = Matrix::zeros(matrix.nrows(), matrix.ncols());
    for col in 0..matrix.ncols() {
        for (new_row, &old_row) in permutation.iter().enumerate().take(matrix.nrows()) {
            result[[old_row, col]] = matrix[[new_row, col]];
        }
    }
    result
}

fn apply_col_permutation<T: Scalar>(matrix: &Matrix<T>, permutation: &[usize]) -> Matrix<T> {
    let mut result = Matrix::zeros(matrix.nrows(), matrix.ncols());
    for (new_col, &old_col) in permutation.iter().enumerate().take(matrix.ncols()) {
        for row in 0..matrix.nrows() {
            result[[row, old_col]] = matrix[[row, new_col]];
        }
    }
    result
}

pub(crate) fn rrlu_colmatrix<T>(lu: &RrLU<T>, linalg: &mut Linalg<'_, '_>) -> Result<Matrix<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let rank = lu.npivots();
    if rank == 0 {
        return Ok(Matrix::zeros(lu.nrows(), 0));
    }
    let rows = index_range(0, rank);
    let cols = index_range(0, rank);
    let right_pivot_cols = submatrix(lu.right_unpermuted(), &rows, &cols);
    linalg
        .mat_mul(lu.left(true), right_pivot_cols)
        .map_err(|err| backend_linalg_error(format!("MatrixLUCI colmatrix multiply failed: {err}")))
}

pub(crate) fn rrlu_rowmatrix<T>(lu: &RrLU<T>, linalg: &mut Linalg<'_, '_>) -> Result<Matrix<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let rank = lu.npivots();
    if rank == 0 {
        return Ok(Matrix::zeros(0, lu.ncols()));
    }
    let rows = index_range(0, rank);
    let cols = index_range(0, rank);
    let left_pivot_rows = submatrix(lu.left_unpermuted(), &rows, &cols);
    linalg
        .mat_mul(left_pivot_rows, lu.right(true))
        .map_err(|err| backend_linalg_error(format!("MatrixLUCI rowmatrix multiply failed: {err}")))
}

pub(crate) fn rrlu_cols_times_pivot_solve<T>(
    lu: &RrLU<T>,
    linalg: &mut Linalg<'_, '_>,
) -> Result<Matrix<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    cols_times_pivot_solve(
        lu.left_unpermuted(),
        lu.row_permutation(),
        lu.npivots(),
        linalg,
    )
}

fn cols_times_pivot_solve<T>(
    left: &Matrix<T>,
    row_permutation: &[usize],
    rank: usize,
    linalg: &mut Linalg<'_, '_>,
) -> Result<Matrix<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let mut result = identity_rect(left.nrows(), rank);
    if rank > 0 && rank < left.nrows() {
        let pivot_rows = index_range(0, rank);
        let pivot_cols = index_range(0, rank);
        let rest_rows = index_range(rank, left.nrows());
        let pivot = submatrix(left, &pivot_rows, &pivot_cols);
        let rest = submatrix(left, &rest_rows, &pivot_cols);
        let solved = linalg
            .triangular_solve(pivot, rest, false, true)
            .map_err(|err| {
                backend_linalg_error(format!("MatrixLUCI lower triangular solve failed: {err}"))
            })?;
        for row in 0..solved.nrows() {
            for col in 0..solved.ncols() {
                result[[rank + row, col]] = solved[[row, col]];
            }
        }
    }
    Ok(apply_row_permutation(&result, row_permutation))
}

pub(crate) fn rrlu_pivot_solve_times_rows<T>(
    lu: &RrLU<T>,
    linalg: &mut Linalg<'_, '_>,
) -> Result<Matrix<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let rank = lu.npivots();
    let mut result = identity_rect(rank, lu.ncols());
    if rank > 0 && rank < lu.ncols() {
        let pivot_rows = index_range(0, rank);
        let pivot_cols = index_range(0, rank);
        let rest_cols = index_range(rank, lu.ncols());
        let pivot = submatrix(lu.right_unpermuted(), &pivot_rows, &pivot_cols);
        let rest = submatrix(lu.right_unpermuted(), &pivot_rows, &rest_cols);
        let solved = linalg
            .triangular_solve(pivot, rest, true, false)
            .map_err(|err| {
                backend_linalg_error(format!("MatrixLUCI upper triangular solve failed: {err}"))
            })?;
        for row in 0..solved.nrows() {
            for col in 0..solved.ncols() {
                result[[row, rank + col]] = solved[[row, col]];
            }
        }
    }
    Ok(apply_col_permutation(&result, lu.col_permutation()))
}

fn factors_from_rrlu<T>(lu: &RrLU<T>, linalg: &mut Linalg<'_, '_>) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let left = if lu.is_left_orthogonal() {
        rrlu_cols_times_pivot_solve(lu, linalg)?
    } else {
        rrlu_colmatrix(lu, linalg)?
    };
    let right = if lu.is_left_orthogonal() {
        rrlu_rowmatrix(lu, linalg)?
    } else {
        rrlu_pivot_solve_times_rows(lu, linalg)?
    };

    Ok(MatrixLuciFactors {
        row_indices: lu.row_indices(),
        col_indices: lu.col_indices(),
        pivot_errors: lu.pivot_errors(),
        rank: lu.npivots(),
        left,
        right,
    })
}

pub(crate) fn dense_matrix_luci_factors_from_matrix<T>(
    a: &Matrix<T>,
    options: RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    validate_owned_luci_input(a, &options)?;
    let lu = rrlu(a, Some(options))?;
    factors_from_rrlu(&lu, &mut Linalg::Global)
}

pub(crate) fn dense_matrix_luci_factors_from_matrix_owned<T>(
    mut a: Matrix<T>,
    options: RrLUOptions,
    mut linalg: Linalg<'_, '_>,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let lu = rrlu_mut(&mut a, Some(options))?;
    factors_from_rrlu(&lu, &mut linalg)
}

pub(crate) fn lazy_matrix_luci_factors_from_blocks<T, F>(
    nrows: usize,
    ncols: usize,
    fill_block: F,
    options: RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
    F: Fn(&[usize], &[usize], &mut [T]),
    LazyBlockRookKernel: PivotKernel<T>,
{
    validate_luci_options(&options)?;
    let source = LazyMatrixSource::new(nrows, ncols, fill_block);
    let kernel_options = PivotKernelOptions {
        max_bond_dim: options.max_bond_dim,
        rel_tol: options.rel_tol,
        abs_tol: options.abs_tol,
        left_orthogonal: options.left_orthogonal,
    };

    let selection = LazyBlockRookKernel
        .factorize(&source, &kernel_options)
        .map_err(map_backend_error)?;
    let factors = CrossFactors::from_source(&source, &selection).map_err(map_backend_error)?;
    factors_to_public(selection, factors, options.left_orthogonal)
}

/// Factorize a dense matrix with MatrixLUCI.
///
/// Returns the selected pivot metadata together with left and
/// right factors. This is the public facade used by higher-level crates.
///
/// # Arguments
///
/// * `a` - Dense matrix to factorize.
/// * `options` - Optional rank and tolerance controls. `None` uses the
///
///   default LUCI settings.
///
/// # Returns
///
/// A [`MatrixLuciFactors`] value containing the selected pivot indices,
/// error history, rank, and factors.
///
/// # Errors
///
/// Returns [`MatrixCIError::InvalidArgument`] for negative or non-finite
/// tolerances, infinite input magnitudes, or backend failures; returns
/// [`MatrixCIError::NaNEncountered`] for NaN input or non-finite factors.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrix_luci_factors_from_matrix;
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 0.0],
///     vec![0.0, 2.0],
/// ]);
/// let factors = matrix_luci_factors_from_matrix(&m, None).unwrap();
/// assert_eq!(factors.left.nrows(), 2);
/// assert_eq!(factors.right.ncols(), 2);
/// assert_eq!(factors.row_indices.len(), factors.rank);
/// ```
pub fn matrix_luci_factors_from_matrix<T>(
    a: &Matrix<T>,
    options: Option<RrLUOptions>,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    <T as crate::MatrixLuciScalar>::matrix_luci_factors_from_matrix(a, options.unwrap_or_default())
}

/// Factorize a dense matrix with MatrixLUCI while consuming the input matrix.
///
/// This is the owned-buffer counterpart of [`matrix_luci_factors_from_matrix`].
/// It reuses the local matrix storage for the rrLU pivot selection step.
///
/// # Errors
///
/// Returns a [`MatrixCIError`] if the factorization fails, for example if the
/// pivot block is singular or the backend rejects a factor solve.
pub fn matrix_luci_factors_from_matrix_owned<T>(
    a: Matrix<T>,
    options: Option<RrLUOptions>,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let options = options.unwrap_or_default();
    validate_owned_luci_input(&a, &options)?;
    dense_matrix_luci_factors_from_matrix_owned(a, options, Linalg::Global)
}

/// Factorize a dense matrix with MatrixLUCI in a caller-owned CPU context.
///
/// Context-scoped counterpart of [`matrix_luci_factors_from_matrix_owned`].
/// The rank-revealing pivot search is identical; the left and right factors
/// are assembled with `context`'s configured backend (matrix products and
/// triangular solves). No process-global context is consulted.
///
/// # Arguments
///
/// * `a` - Dense column-major matrix to factorize; its storage is reused.
/// * `options` - Optional rank and tolerance controls. `None` uses the
///   default LUCI settings.
/// * `context` - Caller-owned CPU execution context running the factor
///   assembly.
///
/// # Returns
///
/// A [`MatrixLuciFactors`] value with the selected pivots, pivot error
/// history, rank, and factors satisfying `a ~= left * right`.
///
/// # Errors
///
/// Returns [`MatrixCIError::InvalidArgument`] for negative or nonfinite
/// tolerances, infinite input magnitudes, or backend solve failures; returns
/// [`MatrixCIError::NaNEncountered`] for NaN input or factors.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{matrix_luci_factors_from_matrix_owned_in, RrLUOptions};
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul, CpuExecutionContext};
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::new());
/// // Rank-1 matrix [1 2; 2 4].
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![2.0, 4.0]]);
/// let options = RrLUOptions { left_orthogonal: true, ..RrLUOptions::default() };
/// let factors = matrix_luci_factors_from_matrix_owned_in(a.clone(), Some(options), &context)?;
/// assert_eq!(factors.rank, 1);
/// let rebuilt = mat_mul(&factors.left, &factors.right)?;
/// for (x, y) in rebuilt.as_col_major_slice().iter().zip(a.as_col_major_slice()) {
///     assert!((x - y).abs() < 1e-12);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn matrix_luci_factors_from_matrix_owned_in<T>(
    a: Matrix<T>,
    options: Option<RrLUOptions>,
    context: &CpuExecutionContext,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let options = options.unwrap_or_default();
    validate_owned_luci_input(&a, &options)?;
    dense_matrix_luci_factors_from_matrix_owned(a, options, Linalg::Context(context))
}

/// Factorize a dense matrix with MatrixLUCI in a session the caller already entered.
///
/// Session-taking counterpart of [`matrix_luci_factors_from_matrix_owned_in`], for callers that own their
/// execution rather than a context: a worker lane entered by the phase that owns it, for example, whose session
/// must be used as it is. The rank-revealing pivot search is identical; the factor assembly's matrix products
/// and triangular solves run on `session`, and neither a process-global context nor a fresh session entry is
/// consulted.
///
/// # Arguments
///
/// * `a` - Dense column-major matrix to factorize; its storage is reused.
/// * `options` - Optional rank and tolerance controls. `None` uses the default LUCI settings.
/// * `session` - An already-entered explicit session that runs the factor assembly.
///
/// # Returns
///
/// A [`MatrixLuciFactors`] value with the selected pivots, pivot error history, rank, and factors satisfying
/// `a ~= left * right`.
///
/// # Errors
///
/// Returns [`MatrixCIError::InvalidArgument`] for negative or nonfinite tolerances, infinite input magnitudes,
/// or backend solve failures; returns [`MatrixCIError::NaNEncountered`] for NaN input or factors.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{matrix_luci_factors_from_matrix_owned_on_session, RrLUOptions};
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul, CpuExecutionContext};
/// use tenferro_cpu::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::new());
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![2.0, 4.0]]);
/// let factors = context.with_concrete_session(|session| {
///     matrix_luci_factors_from_matrix_owned_on_session(
///         a.clone(),
///         Some(RrLUOptions { left_orthogonal: true, ..RrLUOptions::default() }),
///         session,
///     )
/// })??;
/// assert_eq!(factors.rank, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn matrix_luci_factors_from_matrix_owned_on_session<T>(
    a: Matrix<T>,
    options: Option<RrLUOptions>,
    session: &mut Session<'_>,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    let options = options.unwrap_or_default();
    validate_owned_luci_input(&a, &options)?;
    dense_matrix_luci_factors_from_matrix_owned(a, options, Linalg::Session(session))
}

/// Conservative logical working-byte estimate for owned dense LUCI factors.
///
/// Includes the consumed matrix, rrLU L/U, factor construction, permutation
/// and triangular-solve operands/results, and pivot/index metadata. The
/// factor rank is bounded by `min(nrows, ncols, max_rank)`. The estimate sums
/// potentially overlapping factor buffers conservatively; it excludes
/// allocator overhead and provider-private workspace. Callers must add their
/// own buffers retained across the factorization.
///
/// Returns `None` when checked shape/byte arithmetic overflows `usize`.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrix_luci_factors_working_bytes;
/// let bytes = matrix_luci_factors_working_bytes::<f64>(32, 32, 32).unwrap();
/// // Even just the matrix and full-rank L/U require three matrix payloads.
/// assert!(bytes >= 3 * 32 * 32 * std::mem::size_of::<f64>());
/// assert!(matrix_luci_factors_working_bytes::<f64>(usize::MAX, 2, 1).is_none());
/// ```
pub fn matrix_luci_factors_working_bytes<T: Scalar>(
    nrows: usize,
    ncols: usize,
    max_rank: usize,
) -> Option<usize> {
    let rank = max_rank.min(nrows).min(ncols);
    let axes = nrows.checked_add(ncols)?;
    let matrix = nrows.checked_mul(ncols)?;
    // L/U, both output factors, permutation copies, and triangular solve
    // operands/results. Charge both orientations rather than assuming which
    // side has the larger solve, including Matrix/backend operand copies.
    let factors = axes.checked_mul(rank)?.checked_mul(6)?;
    let pivots = rank.checked_mul(rank)?.checked_mul(2)?;
    let scalar_bytes = matrix
        .checked_add(factors)?
        .checked_add(pivots)?
        .checked_mul(std::mem::size_of::<T>())?;
    let index_bytes = axes
        .checked_add(rank)?
        .checked_mul(4)?
        .checked_mul(std::mem::size_of::<usize>())?;
    let error_bytes = rank
        .checked_add(1)?
        .checked_mul(std::mem::size_of::<f64>())?;
    scalar_bytes
        .checked_add(index_bytes)?
        .checked_add(error_bytes)
}

/// Factorize a lazily supplied matrix with MatrixLUCI block-rook search.
///
/// The caller provides a block-fill closure that receives row and column
/// index lists and writes the corresponding matrix block in column-major
/// order.
/// Exactly zero starting residual columns are skipped. An all-zero residual
/// can require inspecting all remaining entries, without materializing the
/// full candidate matrix.
///
/// # Arguments
///
/// * `nrows` - Number of matrix rows.
/// * `ncols` - Number of matrix columns.
/// * `fill_block` - Closure that fills `out` with `A[rows, cols]` in
///
///   column-major order.
/// * `options` - Rank and tolerance controls.
///
/// # Returns
///
/// A [`MatrixLuciFactors`] value containing the pivot metadata and
/// factors.
///
/// # Errors
///
/// Returns [`MatrixCIError::InvalidArgument`] for negative or non-finite
/// tolerances; returns a [`MatrixCIError`] if the lazy factorization fails or
/// if the block callback is inconsistent.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrix_luci_factors_from_blocks;
/// use tensor4all_core::RrLUOptions;
///
/// let factors = matrix_luci_factors_from_blocks(
///     2,
///     2,
///     |rows, cols, out| {
///         for (j, &col) in cols.iter().enumerate() {
///             for (i, &row) in rows.iter().enumerate() {
///                 out[i + rows.len() * j] = if row == col { 1.0 } else { 0.0 };
///             }
///         }
///     },
///     RrLUOptions::default(),
/// ).unwrap();
/// assert_eq!(factors.rank, 2);
/// ```
pub fn matrix_luci_factors_from_blocks<T, F>(
    nrows: usize,
    ncols: usize,
    fill_block: F,
    options: RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + crate::MatrixLuciScalar,
    F: Fn(&[usize], &[usize], &mut [T]),
{
    <T as crate::MatrixLuciScalar>::matrix_luci_factors_from_blocks(
        nrows, ncols, fill_block, options,
    )
}

impl<T> MatrixLUCI<T>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    /// Create a MatrixLUCI from a dense matrix.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixCIError`] when `a` is not a valid matrix for LU cross
    /// interpolation or when the factor construction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![
    ///     vec![2.0_f64, 0.0],
    ///     vec![0.0, 3.0],
    /// ]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// assert!(ci.rank() >= 1);
    /// ```
    pub fn from_matrix(a: &Matrix<T>, options: Option<RrLUOptions>) -> Result<Self> {
        let factors = matrix_luci_factors_from_matrix(a, options)?;

        Ok(Self {
            nrows: a.nrows(),
            ncols: a.ncols(),
            row_indices: factors.row_indices,
            col_indices: factors.col_indices,
            left: factors.left,
            right: factors.right,
            pivot_errors: factors.pivot_errors,
        })
    }

    /// Left CI factor (shape: `nrows x rank`).
    ///
    /// The approximation is `left * right`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
    ///
    /// let m = from_vec2d(vec![
    ///     vec![1.0_f64, 2.0],
    ///     vec![3.0, 4.0],
    /// ]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let reconstructed = mat_mul(&ci.left(), &ci.right()).unwrap();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((reconstructed[[i, j]] - m[[i, j]]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn left(&self) -> Matrix<T> {
        self.left.clone()
    }

    /// Right CI factor (shape: `rank x ncols`).
    ///
    /// The approximation is `left * right`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let r = ci.right();
    /// assert_eq!(r.nrows(), ci.rank());
    /// assert_eq!(r.ncols(), ci.ncols());
    /// // left * right reconstructs the matrix
    /// let recon = mat_mul(&ci.left(), &r).unwrap();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((recon[[i, j]] - m[[i, j]]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn right(&self) -> Matrix<T> {
        self.right.clone()
    }

    /// Pivot error history (one entry per pivot, plus a final residual estimate).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MatrixLUCI;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let errs = ci.pivot_errors();
    /// assert!(!errs.is_empty());
    /// // All errors are non-negative
    /// for &e in &errs {
    ///     assert!(e >= 0.0);
    /// }
    /// ```
    pub fn pivot_errors(&self) -> Vec<f64> {
        self.pivot_errors.clone()
    }

    /// Last pivot error (the residual estimate after all pivots).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::MatrixLUCI;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let err = ci.last_pivot_error();
    /// assert!(err >= 0.0);
    /// ```
    pub fn last_pivot_error(&self) -> f64 {
        self.pivot_errors.last().copied().unwrap_or(0.0)
    }
}

impl<T> AbstractMatrixCI<T> for MatrixLUCI<T>
where
    T: Scalar + crate::MatrixLuciScalar,
{
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn rank(&self) -> usize {
        self.row_indices.len()
    }

    fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    fn evaluate(&self, i: usize, j: usize) -> T {
        let mut sum = T::zero();
        for k in 0..self.rank() {
            sum = sum + self.left[[i, k]] * self.right[[k, j]];
        }
        sum
    }

    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<T> {
        let r = self.rank();
        let mut result = Matrix::zeros(rows.len(), cols.len());
        for (j_out, &col) in cols.iter().enumerate() {
            for (i_out, &row) in rows.iter().enumerate() {
                let mut sum = T::zero();
                for k in 0..r {
                    sum = sum + self.left[[row, k]] * self.right[[k, col]];
                }
                result[[i_out, j_out]] = sum;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests;
