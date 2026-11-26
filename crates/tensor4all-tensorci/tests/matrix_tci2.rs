//! Matrix-valued chain TCI2: parity with the scalar endpoint adapter,
//! reconstruction on every physical point, axis-orientation and
//! unrelated-component guards, for several matrix shapes.

use num_complex::Complex64;
use std::cell::RefCell;
use std::collections::HashMap;
use tensor4all_core::{MatrixLuciScalar, MultiIndex, Scalar};
use tensor4all_simplett::TTScalar;
use tensor4all_tensorbackend::Matrix;
use tensor4all_tensorci::{
    crossinterpolate2, crossinterpolate2_matrix, MatrixShape, PivotSearchStrategy, TCI2Options,
    TCI2Termination,
};

trait TestScalar: Scalar + TTScalar + Default + MatrixLuciScalar {
    fn make(re: f64, im: f64) -> Self;
}

impl TestScalar for f64 {
    fn make(re: f64, _im: f64) -> Self {
        re
    }
}

impl TestScalar for Complex64 {
    fn make(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

const PHYSICAL_DIMS: [usize; 3] = [3, 4, 2];

/// Non-symmetric rank-two target: two separable terms with opposite slopes in
/// `x` and different row and column factors.
fn target<T: TestScalar>(shape: MatrixShape, x: &[usize]) -> Matrix<T> {
    let (left, right) = x
        .iter()
        .zip(PHYSICAL_DIMS)
        .fold((1.0, 1.0), |(l, r), (&c, d)| {
            (
                l * (1.0 + c as f64 / 4.0),
                r * (1.0 + (d - 1 - c) as f64 / 8.0),
            )
        });
    let mut data = Vec::with_capacity(shape.rows * shape.cols);
    for j in 0..shape.cols {
        for i in 0..shape.rows {
            let first = (1.0 + i as f64) * (1.0 + 2.0 * j as f64) * left;
            let second = (3.0 - i as f64) * (1.0 + (j * j) as f64) * right;
            data.push(T::make(first + second, 0.5 * first - 0.25 * second));
        }
    }
    Matrix::from_col_major_vec(shape.rows, shape.cols, data)
}

fn options(pivot_search: PivotSearchStrategy) -> TCI2Options {
    TCI2Options {
        pivot_search,
        tolerance: 1e-10,
        seed: Some(11),
        ..TCI2Options::default()
    }
}

fn internal_dims(shape: MatrixShape, physical: &[usize]) -> Vec<usize> {
    let mut dims = vec![shape.rows];
    dims.extend(physical);
    dims.push(shape.cols);
    dims
}

/// Fills `out` with `block(x)` for every requested point (library layout).
fn fill_blocks<T: TestScalar>(
    points: &[Vec<usize>],
    out: &mut [T],
    block: impl Fn(&[usize]) -> Matrix<T>,
) {
    for (p, x) in points.iter().enumerate() {
        let b = block(x);
        let len = b.as_col_major_slice().len();
        out[p * len..(p + 1) * len].copy_from_slice(b.as_col_major_slice());
    }
}

/// Every physical point of `dims` in lexicographic order.
fn all_points(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut points = vec![vec![]];
    for &d in dims {
        points = points
            .into_iter()
            .flat_map(|prefix| {
                (0..d).map(move |c| {
                    let mut x = prefix.clone();
                    x.push(c);
                    x
                })
            })
            .collect();
    }
    points
}

/// Scalar endpoint adapter owned by the test: a plain hash memo of whole
/// matrices keyed by `x`, with both `f` and `batched_f` supplied.
struct EndpointAdapter<T> {
    shape: MatrixShape,
    memo: RefCell<HashMap<Vec<usize>, Matrix<T>>>,
}

impl<T: TestScalar> EndpointAdapter<T> {
    fn entry(&self, index: &MultiIndex) -> T {
        let n = index.len();
        let x = index[1..n - 1].to_vec();
        let mut memo = self.memo.borrow_mut();
        let matrix = memo
            .entry(x)
            .or_insert_with_key(|x| target::<T>(self.shape, x));
        matrix[[index[0], index[n - 1]]]
    }
}

/// Initial pivot `[i, x, j]` that is nonzero for every tested shape.
fn initial_pivot(shape: MatrixShape) -> MultiIndex {
    vec![1.min(shape.rows - 1), 1, 2, 1, 1.min(shape.cols - 1)]
}

fn parity<T: TestScalar>(shape: MatrixShape, pivot_search: PivotSearchStrategy) {
    let initial = vec![initial_pivot(shape)];
    let block_len = shape.rows * shape.cols;

    let matrix_fit = crossinterpolate2_matrix::<T, _>(
        |points: &[Vec<usize>], out: &mut [T]| {
            assert_eq!(out.len(), points.len() * block_len);
            fill_blocks(points, out, |x| target::<T>(shape, x));
        },
        shape,
        PHYSICAL_DIMS.to_vec(),
        initial.clone(),
        options(pivot_search),
    )
    .unwrap();

    let adapter = EndpointAdapter::<T> {
        shape,
        memo: RefCell::new(HashMap::new()),
    };
    let scalar = crossinterpolate2::<T, _, _>(
        |index| adapter.entry(index),
        Some(|indices: &[MultiIndex]| indices.iter().map(|i| adapter.entry(i)).collect()),
        internal_dims(shape, &PHYSICAL_DIMS),
        initial,
        options(pivot_search),
    )
    .unwrap();

    let result = &matrix_fit.result;
    assert_eq!(result.termination, TCI2Termination::Converged);
    assert_eq!(result.termination, scalar.termination);
    // The target is rank two, so the comparison is not a rank-one triviality.
    assert!(result.ranks.iter().any(|&rank| rank >= 2));
    assert_eq!(result.ranks, scalar.ranks);
    assert_eq!(result.errors, scalar.errors);
    assert_eq!(result.tci.max_sample_value(), scalar.tci.max_sample_value());
    for site in 0..internal_dims(shape, &PHYSICAL_DIMS).len() {
        assert_eq!(
            result.tci.i_set(site),
            scalar.tci.i_set(site),
            "I set {site}"
        );
        assert_eq!(
            result.tci.j_set(site),
            scalar.tci.j_set(site),
            "J set {site}"
        );
    }

    // Distinct points were sampled at most once each while retained.
    let stats = matrix_fit.sampling_stats();
    let distinct: usize = PHYSICAL_DIMS.iter().product();
    assert!(stats.source_points <= distinct);
    if block_len > 1 {
        assert!(stats.entry_requests > stats.source_points);
    }
    assert_eq!(stats.entry_requests - stats.cache_hits, stats.source_points);
}

/// Largest entrywise deviation of the fit from `block` over `points`,
/// materializing each point once as a dense matrix.
fn max_deviation<T: TestScalar>(
    fit: &tensor4all_tensorci::MatrixTCI2Result<T>,
    points: &[Vec<usize>],
    block: impl Fn(&[usize]) -> Matrix<T>,
) -> f64 {
    let shape = fit.shape();
    let mut max_error = 0.0_f64;
    for x in points {
        let approx = fit.evaluate(x).unwrap();
        let exact = block(x);
        assert_eq!((approx.nrows(), approx.ncols()), (shape.rows, shape.cols));
        for (a, e) in approx
            .as_col_major_slice()
            .iter()
            .zip(exact.as_col_major_slice())
        {
            max_error = max_error.max(Scalar::abs_sq(*a - *e).sqrt());
        }
    }
    max_error
}

/// Largest entry magnitude of `block` over `points`.
fn scale<T: TestScalar>(points: &[Vec<usize>], block: impl Fn(&[usize]) -> Matrix<T>) -> f64 {
    points
        .iter()
        .flat_map(|x| block(x).as_col_major_slice().to_vec())
        .map(|v| Scalar::abs_sq(v).sqrt())
        .fold(0.0, f64::max)
}

/// The target is exactly rank two and the solver tolerance is `1e-10`, so the
/// interpolant agrees with it to rounding plus the tolerance; `1e-10` relative
/// to the target scale is tight but leaves headroom for accumulated rounding.
const RECONSTRUCTION_RTOL: f64 = 1e-10;

fn reconstruction<T: TestScalar>(shape: MatrixShape, pivot_search: PivotSearchStrategy) {
    let fit = crossinterpolate2_matrix::<T, _>(
        |points: &[Vec<usize>], out: &mut [T]| {
            fill_blocks(points, out, |x| target::<T>(shape, x));
        },
        shape,
        PHYSICAL_DIMS.to_vec(),
        vec![initial_pivot(shape)],
        options(pivot_search),
    )
    .unwrap();
    assert_eq!(fit.shape(), shape);

    let points = all_points(&PHYSICAL_DIMS);
    let block = |x: &[usize]| target::<T>(shape, x);
    let error = max_deviation(&fit, &points, block);
    let bound = RECONSTRUCTION_RTOL * scale(&points, block);
    assert!(
        error < bound,
        "max reconstruction error = {error} (bound {bound})"
    );
}

/// Generates the parity and reconstruction tests of one matrix shape; each
/// pivot strategy and scalar type is a separate test.
macro_rules! shape_tests {
    ($module:ident, $rows:expr, $cols:expr) => {
        mod $module {
            use super::*;

            const SHAPE: MatrixShape = MatrixShape {
                rows: $rows,
                cols: $cols,
            };

            #[test]
            fn parity_full_f64() {
                parity::<f64>(SHAPE, PivotSearchStrategy::Full);
            }
            #[test]
            fn parity_rook_f64() {
                parity::<f64>(SHAPE, PivotSearchStrategy::Rook);
            }
            #[test]
            fn parity_full_complex64() {
                parity::<Complex64>(SHAPE, PivotSearchStrategy::Full);
            }
            #[test]
            fn parity_rook_complex64() {
                parity::<Complex64>(SHAPE, PivotSearchStrategy::Rook);
            }
            #[test]
            fn reconstruction_full_f64() {
                reconstruction::<f64>(SHAPE, PivotSearchStrategy::Full);
            }
            #[test]
            fn reconstruction_rook_f64() {
                reconstruction::<f64>(SHAPE, PivotSearchStrategy::Rook);
            }
            #[test]
            fn reconstruction_full_complex64() {
                reconstruction::<Complex64>(SHAPE, PivotSearchStrategy::Full);
            }
            #[test]
            fn reconstruction_rook_complex64() {
                reconstruction::<Complex64>(SHAPE, PivotSearchStrategy::Rook);
            }
        }
    };
}

shape_tests!(shape_2x3, 2, 3);
shape_tests!(shape_3x2, 3, 2);
shape_tests!(shape_1x3, 1, 3);
shape_tests!(shape_3x1, 3, 1);
shape_tests!(shape_1x1, 1, 1);

/// Entry `(i, j)` at `x = (a, b)` of a target that is not symmetric: the
/// leading `2 x 2` corner satisfies `F[0, 1] != F[1, 0]` at every point.
/// Two separable terms (rank two): `(1 + a) (1 + i)` and `(2 + b) 10 (1 + j)`.
fn asymmetric<T: TestScalar>(shape: MatrixShape, x: &[usize]) -> Matrix<T> {
    let mut data = Vec::new();
    for j in 0..shape.cols {
        for i in 0..shape.rows {
            let re = (1.0 + x[0] as f64) * (1.0 + i as f64)
                + (2.0 + x[1] as f64) * 10.0 * (1.0 + j as f64);
            data.push(T::make(re, -0.5 * re + i as f64));
        }
    }
    Matrix::from_col_major_vec(shape.rows, shape.cols, data)
}

/// A swapped row/column layout (in the callback contract or in `evaluate`)
/// would read `F[j, i]` where `F[i, j]` belongs; with an asymmetric target at
/// `2 x 3` and `3 x 2` that fails either the indexed or the shape check.
fn swapped_axes_guard<T: TestScalar>(shape: MatrixShape, pivot_search: PivotSearchStrategy) {
    let dims = [3, 4];
    let fit = crossinterpolate2_matrix::<T, _>(
        |points: &[Vec<usize>], out: &mut [T]| {
            fill_blocks(points, out, |x| asymmetric::<T>(shape, x));
        },
        shape,
        dims.to_vec(),
        vec![],
        options(pivot_search),
    )
    .unwrap();
    let points = all_points(&dims);
    let block = |x: &[usize]| asymmetric::<T>(shape, x);
    let bound = RECONSTRUCTION_RTOL * scale(&points, block);
    for x in &points {
        let exact = asymmetric::<T>(shape, x);
        let approx = fit.evaluate(x).unwrap();
        assert_eq!((approx.nrows(), approx.ncols()), (shape.rows, shape.cols));
        for i in 0..shape.rows {
            for j in 0..shape.cols {
                let diff = Scalar::abs_sq(approx[[i, j]] - exact[[i, j]]).sqrt();
                assert!(diff < bound, "entry ({i}, {j}) at {x:?}: {diff}");
            }
        }
        // The transposed reading of the leading 2 x 2 corner is a different
        // number, so a swapped layout cannot pass the loop above.
        let swapped = Scalar::abs_sq(exact[[0, 1]] - exact[[1, 0]]).sqrt();
        assert!(swapped > 1.0, "target must be asymmetric, got {swapped}");
        assert!(Scalar::abs_sq(approx[[0, 1]] - exact[[1, 0]]).sqrt() > 0.5 * swapped);
    }
}

macro_rules! swapped_axes_tests {
    ($($name:ident: $ty:ty, $rows:expr, $cols:expr, $strategy:expr;)*) => {
        $(
            #[test]
            fn $name() {
                swapped_axes_guard::<$ty>(
                    MatrixShape { rows: $rows, cols: $cols },
                    $strategy,
                );
            }
        )*
    };
}

swapped_axes_tests! {
    swapped_axes_guard_2x3_f64: f64, 2, 3, PivotSearchStrategy::Full;
    swapped_axes_guard_3x2_f64: f64, 3, 2, PivotSearchStrategy::Full;
    swapped_axes_guard_2x3_complex64: Complex64, 2, 3, PivotSearchStrategy::Full;
    swapped_axes_guard_3x2_complex64: Complex64, 3, 2, PivotSearchStrategy::Full;
    swapped_axes_guard_rook_2x3_f64: f64, 2, 3, PivotSearchStrategy::Rook;
    swapped_axes_guard_rook_3x2_f64: f64, 3, 2, PivotSearchStrategy::Rook;
    swapped_axes_guard_rook_2x3_complex64: Complex64, 2, 3, PivotSearchStrategy::Rook;
    swapped_axes_guard_rook_3x2_complex64: Complex64, 3, 2, PivotSearchStrategy::Rook;
}

/// Components are unrelated functions of `x` (different frequencies and
/// phases per `(i, j)`), so the matrix has no shared low-rank structure across
/// components, and the grid is small enough to check every `(i, x, j)`.
fn unrelated<T: TestScalar>(shape: MatrixShape, x: &[usize]) -> Matrix<T> {
    let mut data = Vec::new();
    for j in 0..shape.cols {
        for i in 0..shape.rows {
            let k = (1 + i + 3 * j) as f64;
            let re = (k * (1.0 + x[0] as f64) + 0.3 * k * k).sin()
                + (0.7 * k * (1.0 + x[1] as f64)).cos() * (1.0 + (i * j) as f64);
            let im = (1.3 * k * (2.0 + x[0] as f64 * x[1] as f64)).cos();
            data.push(T::make(re, im));
        }
    }
    Matrix::from_col_major_vec(shape.rows, shape.cols, data)
}

/// Exhaustive check on a grid where the chain `[rows, d1, d2, cols]` can
/// represent any tensor exactly (bonds saturate), so every `(i, x, j)` entry
/// must match to the tolerance, through `evaluate` and through the scalar
/// tensor train alike.
fn unrelated_components<T: TestScalar>(pivot_search: PivotSearchStrategy) {
    let shape = MatrixShape { rows: 2, cols: 3 };
    let dims = [3, 2];
    let fit = crossinterpolate2_matrix::<T, _>(
        |points: &[Vec<usize>], out: &mut [T]| {
            fill_blocks(points, out, |x| unrelated::<T>(shape, x));
        },
        shape,
        dims.to_vec(),
        vec![],
        TCI2Options {
            tolerance: 1e-12,
            ..options(pivot_search)
        },
    )
    .unwrap();
    let points = all_points(&dims);
    let block = |x: &[usize]| unrelated::<T>(shape, x);
    let bound = RECONSTRUCTION_RTOL * scale(&points, block);
    assert!(max_deviation(&fit, &points, block) < bound);

    // One dense materialization of the (L+2)-site train (column-major over
    // `[rows, d1, d2, cols]`), compared with the dense target assembled once in
    // the same layout.
    let (tt_data, tt_dims) = fit.to_tensor_train().unwrap().full_tensor().unwrap();
    assert_eq!(tt_dims, internal_dims(shape, &dims));
    let mut target_data = vec![T::default(); tt_data.len()];
    for x in &points {
        let exact = unrelated::<T>(shape, x);
        for i in 0..shape.rows {
            for j in 0..shape.cols {
                let offset = x[0] + dims[0] * x[1];
                target_data[i + shape.rows * (offset + dims[0] * dims[1] * j)] = exact[[i, j]];
            }
        }
    }
    let tt_error = tt_data
        .iter()
        .zip(&target_data)
        .map(|(a, e)| Scalar::abs_sq(*a - *e).sqrt())
        .fold(0.0, f64::max);
    assert!(tt_error < bound, "tensor-train max error = {tt_error}");
}

#[test]
fn unrelated_components_full_f64() {
    unrelated_components::<f64>(PivotSearchStrategy::Full);
}

#[test]
fn unrelated_components_full_complex64() {
    unrelated_components::<Complex64>(PivotSearchStrategy::Full);
}

#[test]
fn unrelated_components_rook_f64() {
    unrelated_components::<f64>(PivotSearchStrategy::Rook);
}

#[test]
fn unrelated_components_rook_complex64() {
    unrelated_components::<Complex64>(PivotSearchStrategy::Rook);
}
