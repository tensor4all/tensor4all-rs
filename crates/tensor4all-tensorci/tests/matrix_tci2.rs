//! Matrix-valued chain TCI2: parity with the scalar endpoint adapter and
//! reconstruction of a non-symmetric target.

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

const SHAPE: MatrixShape = MatrixShape { rows: 2, cols: 3 };
const PHYSICAL_DIMS: [usize; 3] = [3, 4, 2];

/// Non-symmetric `2 x 3` target: two separable terms with opposite slopes in `x`
/// and different row and column factors, so the joint rank is two.
fn target<T: TestScalar>(x: &[usize]) -> Matrix<T> {
    let (left, right) = x
        .iter()
        .zip(PHYSICAL_DIMS)
        .fold((1.0, 1.0), |(l, r), (&c, d)| {
            (
                l * (1.0 + c as f64 / 4.0),
                r * (1.0 + (d - 1 - c) as f64 / 8.0),
            )
        });
    let mut data = Vec::with_capacity(SHAPE.rows * SHAPE.cols);
    for j in 0..SHAPE.cols {
        for i in 0..SHAPE.rows {
            let first = (1.0 + i as f64) * (1.0 + 2.0 * j as f64) * left;
            let second = (3.0 - i as f64) * (1.0 + (j * j) as f64) * right;
            data.push(T::make(first + second, 0.5 * first - 0.25 * second));
        }
    }
    Matrix::from_col_major_vec(SHAPE.rows, SHAPE.cols, data)
}

fn options(pivot_search: PivotSearchStrategy) -> TCI2Options {
    TCI2Options {
        pivot_search,
        tolerance: 1e-10,
        seed: Some(11),
        ..TCI2Options::default()
    }
}

fn internal_dims() -> Vec<usize> {
    let mut dims = vec![SHAPE.rows];
    dims.extend(PHYSICAL_DIMS);
    dims.push(SHAPE.cols);
    dims
}

/// Scalar endpoint adapter owned by the test: a plain hash memo of whole
/// matrices keyed by `x`, with both `f` and `batched_f` supplied.
struct EndpointAdapter<T> {
    memo: RefCell<HashMap<Vec<usize>, Matrix<T>>>,
}

impl<T: TestScalar> EndpointAdapter<T> {
    fn entry(&self, index: &MultiIndex) -> T {
        let n = index.len();
        let x = index[1..n - 1].to_vec();
        let mut memo = self.memo.borrow_mut();
        let matrix = memo.entry(x).or_insert_with_key(|x| target::<T>(x));
        matrix[[index[0], index[n - 1]]]
    }
}

fn parity<T: TestScalar>(pivot_search: PivotSearchStrategy) {
    let initial = vec![vec![1, 1, 2, 1, 1]];

    let matrix_fit = crossinterpolate2_matrix::<T, _>(
        |points: &[Vec<usize>], out: &mut [T]| {
            for (p, x) in points.iter().enumerate() {
                let block = target::<T>(x);
                out[p * 6..(p + 1) * 6].copy_from_slice(block.as_col_major_slice());
            }
        },
        SHAPE,
        PHYSICAL_DIMS.to_vec(),
        initial.clone(),
        options(pivot_search),
    )
    .unwrap();

    let adapter = EndpointAdapter::<T> {
        memo: RefCell::new(HashMap::new()),
    };
    let scalar = crossinterpolate2::<T, _, _>(
        |index| adapter.entry(index),
        Some(|indices: &[MultiIndex]| indices.iter().map(|i| adapter.entry(i)).collect()),
        internal_dims(),
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
    for site in 0..internal_dims().len() {
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
    assert!(stats.entry_requests > stats.source_points);
    assert_eq!(stats.entry_requests - stats.cache_hits, stats.source_points);
}

#[test]
fn parity_full_f64() {
    parity::<f64>(PivotSearchStrategy::Full);
}

#[test]
fn parity_rook_f64() {
    parity::<f64>(PivotSearchStrategy::Rook);
}

#[test]
fn parity_full_complex64() {
    parity::<Complex64>(PivotSearchStrategy::Full);
}

#[test]
fn parity_rook_complex64() {
    parity::<Complex64>(PivotSearchStrategy::Rook);
}

fn reconstruction<T: TestScalar>() {
    let fit = crossinterpolate2_matrix::<T, _>(
        |points: &[Vec<usize>], out: &mut [T]| {
            for (p, x) in points.iter().enumerate() {
                out[p * 6..(p + 1) * 6].copy_from_slice(target::<T>(x).as_col_major_slice());
            }
        },
        SHAPE,
        PHYSICAL_DIMS.to_vec(),
        vec![],
        options(PivotSearchStrategy::Full),
    )
    .unwrap();
    assert_eq!(fit.shape(), SHAPE);

    // Materialize every point densely and subtract once per point.
    let mut max_error = 0.0_f64;
    for x0 in 0..PHYSICAL_DIMS[0] {
        for x1 in 0..PHYSICAL_DIMS[1] {
            for x2 in 0..PHYSICAL_DIMS[2] {
                let x = [x0, x1, x2];
                let approx = fit.evaluate(&x).unwrap();
                let exact = target::<T>(&x);
                assert_eq!((approx.nrows(), approx.ncols()), (2, 3));
                for (a, e) in approx
                    .as_col_major_slice()
                    .iter()
                    .zip(exact.as_col_major_slice())
                {
                    max_error = max_error.max(Scalar::abs_sq(*a - *e).sqrt());
                }
            }
        }
    }
    assert!(max_error < 1e-9, "max reconstruction error = {max_error}");
}

#[test]
fn reconstruction_f64() {
    reconstruction::<f64>();
}

#[test]
fn reconstruction_complex64() {
    reconstruction::<Complex64>();
}
